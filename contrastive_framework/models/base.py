"""
base.py – Klasa bazowa dla wszystkich modeli kontrastowych.
Wszystkie konkretne algorytmy (MoCo, BYOL, SCARF, SubTab, TS-TCC)
dziedziczą po BaseContrastiveModel i implementują pretrain_step().
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pomocnicze bloki budulcowe
# ---------------------------------------------------------------------------

def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    dropout_p: float = 0.0,
) -> nn.Sequential:
    """Buduje MLP: Linear → BN → ReLU → [Dropout] × (num_layers-1) → Linear."""
    layers: list[nn.Module] = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
    return nn.Sequential(*layers)


def build_encoder(encoder_type: str, input_dim: int, hidden_dim: int, output_dim: int,
                  bn_splits: int = 1) -> nn.Module:
    """
    Fabryka koderów.

    Parameters
    ----------
    encoder_type : str
        'mlp'        – dla danych tabelarycznych (input_dim = liczba cech)
        'cnn'        – 1-D CNN dla szeregów czasowych (input_dim = liczba kanałów)
        'cnn_image'  – 2-D CNN dla obrazów (input_dim = liczba kanałów, np. 3 dla RGB).
                       Lekka ResNet-like architektura dla obrazów 28×28–64×64.
                       Zachowuje strukturę przestrzenną, kompatybilna z augmentacjami
                       torchvision (RandomResizedCrop, ColorJitter, itd).
        'resnet_like' – DEPRECATED dla obrazów; głęboki MLP na spłaszczonych pikselach.
                        Zachowane dla wsteczności i testów na płaskich wektorach.
    bn_splits : int
        Tylko dla 'cnn_image'. Liczba podgrup SplitBatchNorm (1 = zwykły BN).
        Używane przez MoCo w gałęzi encoder_k.
    """
    if encoder_type == "mlp":
        return build_mlp(input_dim, hidden_dim, output_dim, num_layers=3)

    if encoder_type == "cnn":
        # Wejście: (batch, input_dim, seq_len)  →  wyjście: (batch, output_dim)
        # `input_dim` to liczba kanałów wejściowych (sensorów), np. 9 dla UCI HAR.
        return nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
        )

    if encoder_type == "cnn_image":
        # Wejście: (batch, input_dim, H, W) → wyjście: (batch, output_dim).
        # `input_dim` to liczba kanałów (3 dla RGB, 1 dla skali szarości).
        return _ImageCNN(in_channels=input_dim, output_dim=output_dim, bn_splits=bn_splits)

    if encoder_type == "resnet_like":
        # Głęboki MLP z połączeniami residualnymi – działa na spłaszczonych pikselach
        return _ResidualMLP(input_dim, hidden_dim, output_dim)

    raise ValueError(f"Nieznany encoder_type: '{encoder_type}'. "
                     "Wybierz 'mlp', 'cnn', 'cnn_image' lub 'resnet_like'.")


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class _ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden_dim) for _ in range(4)])
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class SplitBatchNorm2d(nn.Module):
    """
    SplitBatchNorm dla MoCo na jednej karcie GPU.

    Zastępuje Shuffle BN z oryginalnego MoCo (He et al. 2020, Alg. 3), który
    wymaga wielu GPU. Na jednej karcie dzielimy batch na `num_splits` podgrup
    i liczymy statystyki BN niezależnie dla każdej z nich — zapobiega to temu,
    żeby BN „podpowiadał" pary pozytywne poprzez wspólną średnią/wariancję.

    Gdy batch_size < num_splits lub nie jest podzielny, ostatni niecały split
    jest pomijany (resztę pikseli liczymy ze statystyk pełnego batcha):
    faktycznie dla takich batchów używamy zwykłego BN.

    Źródło inspiracji: MoCo CIFAR notebook
    (https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb)
    Licencja oryginału: CC BY-NC 4.0.
    Zmiany: reimplementacja w PyTorch bez zależności od fb-internal code.
    """

    def __init__(self, num_features: int, num_splits: int = 8, **kwargs) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_splits = num_splits
        # Wspólne parametry γ/β — uczone globalnie
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))
        # running stats — aktualizowane jak w zwykłym BN
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))
        self.momentum = kwargs.get("momentum", 0.1)
        self.eps      = kwargs.get("eps", 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        split_size = N // self.num_splits

        if not self.training or split_size < 1:
            # Eval mode lub za mały batch → zwykły BN z running stats
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias,
                training=self.training, momentum=self.momentum, eps=self.eps,
            )

        # Liczymy statystyki per-split, ale aktualizujemy running stats globalnie
        usable = split_size * self.num_splits
        x_splits = x[:usable].view(self.num_splits, split_size, C, H, W)
        # batch_norm na każdym splicie osobno (training=True, running stats None)
        out_splits = F.batch_norm(
            x_splits.reshape(self.num_splits * split_size, C, H, W),
            # Przekazujemy None jako running stats żeby nie były aktualizowane tu
            None, None,
            None, None,
            training=True, momentum=0.0, eps=self.eps,
        )
        # Aktualizuj running stats ze statystyk całego usable batcha
        with torch.no_grad():
            mean = x[:usable].mean(dim=(0, 2, 3))
            var  = x[:usable].var(dim=(0, 2, 3), unbiased=False)
            self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var  * self.momentum)

        # Zastosuj γ/β ręcznie
        out_splits = out_splits * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        if usable == N:
            return out_splits

        # Ostatni niecały split — zwykły BN z running stats (brak aktualizacji)
        tail = F.batch_norm(
            x[usable:], self.running_mean, self.running_var,
            self.weight, self.bias,
            training=False, momentum=self.momentum, eps=self.eps,
        )
        return torch.cat([out_splits, tail], dim=0)


def _make_bn2d(num_features: int, num_splits: int = 1) -> nn.Module:
    """Fabryka BN: zwykły BatchNorm2d gdy num_splits<=1, SplitBatchNorm2d gdy >1."""
    if num_splits > 1:
        return SplitBatchNorm2d(num_features, num_splits=num_splits)
    return nn.BatchNorm2d(num_features)


class _ImageConvBlock(nn.Module):
    """Conv + BN + ReLU + Conv + BN + skip + ReLU (rezydualny blok dla 2D)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 bn_splits: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = _make_bn2d(out_ch, bn_splits)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = _make_bn2d(out_ch, bn_splits)
        # Skip connection – downsample jeśli zmieniają się wymiary
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                _make_bn2d(out_ch, bn_splits),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class _ImageCNN(nn.Module):
    """
    Lekki CNN dla małych obrazów (28×28 – 64×64).

    Architektura inspirowana CIFAR-ResNet (He et al. 2015 sekcja 4.2):
    - Pierwsza warstwa: Conv 3×3 stride 1 (bez maxpool – nie tracimy rozdzielczości)
    - 3 stage'e bloków rezydualnych z stride 2 między stage'ami
    - AdaptiveAvgPool2d na końcu → klasyfikator liniowy

    Dla 28×28 obrazu wejściowego:
      28×28 → 28×28 (stem) → 14×14 (stage 2) → 7×7 (stage 3) → 1×1 (avgpool)

    Parameters
    ----------
    bn_splits : int
        Liczba podgrup BN (SplitBatchNorm). 1 = zwykły BatchNorm2d.
        Wartość > 1 używana przez MoCo w gałęzi kluczy (encoder_k).
    """

    def __init__(self, in_channels: int, output_dim: int,
                 base_channels: int = 64, bn_splits: int = 1) -> None:
        super().__init__()
        c = base_channels

        # Stem – brak agresywnego stride'u na początku (mały obraz)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1, bias=False),
            _make_bn2d(c, bn_splits),
            nn.ReLU(inplace=True),
        )

        # 3 stage'e, każdy z 2 blokami rezydualnymi.
        # Pierwszy blok w stage 2/3 ma stride=2 (downsampling 2×).
        self.stage1 = nn.Sequential(
            _ImageConvBlock(c,     c,     stride=1, bn_splits=bn_splits),
            _ImageConvBlock(c,     c,     stride=1, bn_splits=bn_splits),
        )
        self.stage2 = nn.Sequential(
            _ImageConvBlock(c,     c * 2, stride=2, bn_splits=bn_splits),
            _ImageConvBlock(c * 2, c * 2, stride=1, bn_splits=bn_splits),
        )
        self.stage3 = nn.Sequential(
            _ImageConvBlock(c * 2, c * 4, stride=2, bn_splits=bn_splits),
            _ImageConvBlock(c * 4, c * 4, stride=1, bn_splits=bn_splits),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c * 4, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Klasa bazowa
# ---------------------------------------------------------------------------

class BaseContrastiveModel(ABC, nn.Module):
    """
    Abstrakcyjna klasa bazowa dla metod uczenia kontrastowego
    i pokrewnych metod samo-nadzorowanych.

    Parametry
    ---------
    encoder_type : str
        Typ kodera – 'mlp', 'cnn' lub 'resnet_like'.
    input_dim : int
        Wymiarowość surowego wejścia (piksele, cechy, długość szeregu).
    hidden_dim : int
        Szerokość ukrytych warstw kodera.
    proj_dim : int
        Wymiar wektora projekcji (przestrzeń kontrastowa).
    num_classes : int
        Liczba klas do liniowej ewaluacji (linear readout).
    """

    def __init__(
        self,
        encoder_type: str,
        input_dim: int,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.encoder: nn.Module = build_encoder(encoder_type, input_dim,
                                                hidden_dim, hidden_dim)
        self.projector: nn.Module = build_mlp(hidden_dim, hidden_dim, proj_dim)
        self.classifier: nn.Linear = nn.Linear(hidden_dim, num_classes)
        self._encoder_frozen: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przepuszcza dane przez koder → zwraca reprezentację (embedding).
        Używane zarówno podczas pre-treningu, jak i ewaluacji.
        """
        return self.encoder(x)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Normalizuje i przepuszcza embedding przez głowicę projekcji."""
        z = self.projector(h)
        return F.normalize(z, dim=-1)

    @abstractmethod
    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Jeden krok pre-treningu samo-nadzorowanego.

        Każda klasa potomna musi zaimplementować tę metodę.
        Powinna zwrócić słownik zawierający przynajmniej klucz 'loss'.

        Parameters
        ----------
        batch : Any
            Surowa paczka danych – format zależy od algorytmu.
        """

    def linear_readout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Klasyfikacja na zamrożonych cechach (Linear Evaluation Protocol).
        Koder NIE jest aktualizowany – gradient zatrzymuje się na wejściu
        do klasyfikatora.
        """
        with torch.set_grad_enabled(not self._encoder_frozen):
            h = self.encoder(x)
        return self.classifier(h)

    def freeze_encoder(self) -> None:
        """Zamraża wagi kodera (etap liniowej ewaluacji).

        Wywołuje encoder.eval() żeby BatchNorm używał zapisanych running statistics
        zamiast aktualizować je w trakcie fine-tuningu klasyfikatora.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self._encoder_frozen = True

    def train(self, mode: bool = True) -> "BaseContrastiveModel":
        """Nadpisuje nn.Module.train() – koder pozostaje w eval() gdy zamrożony."""
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def unfreeze_encoder(self) -> None:
        """Odmraża wagi kodera (np. do fine-tuningu end-to-end)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._encoder_frozen = False

    def get_embedding_dim(self) -> int:
        """Zwraca wymiarowość embeddings kodera (wejście klasyfikatora)."""
        return self.classifier.in_features

    def extra_repr(self) -> str:
        return (f"encoder_frozen={self._encoder_frozen}, "
                f"embedding_dim={self.get_embedding_dim()}")

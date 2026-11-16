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
    wymaga wielu GPU. Na jednej karcie przenosimy próbki do wymiaru kanałów:
    input (N, C, H, W) → (N//S, C*S, H, W), gdzie S = num_splits.
    Zwykły BN na tak reshape'owanym tensorze liczy statystyki per-split
    (każde C kanałów jednej próbki to osobny „kanał" w reshaped widoku).
    Po normalizacji reshape z powrotem do (N, C, H, W).
    Running stats uśredniane po splitach.

    Implementacja 1:1 wg oficjalnego notebooka MoCo CIFAR:
    https://colab.research.google.com/github/facebookresearch/moco/blob/
    colab-notebook/colab/moco_cifar10_demo.ipynb  (CC BY-NC 4.0)

    Gdy batch_size < num_splits lub nie jest podzielny przez num_splits —
    fallback do zwykłego BN.
    """

    def __init__(self, num_features: int, num_splits: int = 8, **kwargs) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_splits = num_splits
        self.momentum = kwargs.get("momentum", 0.1)
        self.eps      = kwargs.get("eps", 1e-5)
        # Wspólne γ/β — uczone globalnie, jak w zwykłym BN
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))
        # running stats na rozmiarze num_features*num_splits —
        # każdy split ma własne running_mean/var; uśredniane do num_features przy ewaluacji
        # Port z notebooka MoCo CIFAR: running_mean/var w rozmiarze C*num_splits
        self.register_buffer("running_mean",
                             torch.zeros(num_features * num_splits))
        self.register_buffer("running_var",
                             torch.ones(num_features * num_splits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        S = self.num_splits

        if not self.training or N < S or N % S != 0:
            # Fallback: zwykły BN z uśrednionymi running stats po splitach
            mean_avg = self.running_mean.view(S, C).mean(dim=0)
            var_avg  = self.running_var.view(S, C).mean(dim=0)
            return F.batch_norm(
                x, mean_avg, var_avg,
                self.weight, self.bias,
                training=self.training, momentum=self.momentum, eps=self.eps,
            )

        # Podziel batch na S bloków po N//S kolejnych próbek każdy.
        # view(S, N//S, C, H, W): blok s zawiera próbki s*(N//S) .. (s+1)*(N//S)-1.
        x_splits = x.view(S, N // S, C, H, W)
        outputs = []
        for s in range(S):
            # running stats splitu s: running_mean[s*C:(s+1)*C]
            rm_s = self.running_mean[s * C: (s + 1) * C]
            rv_s = self.running_var[s * C: (s + 1) * C]
            out_s = F.batch_norm(
                x_splits[s].contiguous(),
                rm_s, rv_s,
                self.weight, self.bias,
                training=True, momentum=self.momentum, eps=self.eps,
            )
            outputs.append(out_s)

        # cat po dim=0: odtwarza kolejność próbek (N, C, H, W)
        return torch.cat(outputs, dim=0)


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

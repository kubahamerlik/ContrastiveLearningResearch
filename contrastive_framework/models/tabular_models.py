"""
tabular_models.py – Implementacje SCARF i SubTab dla danych tabelarycznych.

SCARF  : Bahri et al., 2021 – "SCARF: Self-Supervised Contrastive Learning
         using Random Feature Corruption"
SubTab : Ucar et al., 2021  – "SubTab: Subsetting Features of Tabular Data
         for Self-Supervised Representation Learning"
"""

from typing import Any, Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContrastiveModel, build_mlp


# ===========================================================================
# Wspólna strata InfoNCE (NT-Xent) dla danych tabelarycznych
# ===========================================================================

def info_nce_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """
    Symetryczna strata NT-Xent dla pary (z1, z2).

    Każda próbka z batcha ma dokładnie jedną parę pozytywną (drugą augmentację
    tej samej próbki) i (2B - 2) próbek negatywnych.

    Parameters
    ----------
    z1, z2 : (B, D) – znormalizowane wektory embeddinów
    temperature : float – parametr τ
    """
    B = z1.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z1.device)

    # Połącz oba widoki: (2B, D)
    z = torch.cat([z1, z2], dim=0)

    # Macierz podobieństwa kosinusowego: (2B, 2B)
    sim = torch.mm(z, z.T) / temperature

    # Maska wykluczająca przekątną (próbka nie jest parą pozytywną sama z sobą)
    mask_diag = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask_diag, float("-inf"))

    # Etykiety: próbka i ma parę pozytywną pod indeksem (i + B) % 2B
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])         # (2B,)

    return F.cross_entropy(sim, labels)


# ===========================================================================
# 1. SCARF – Self-Supervised Contrastive Learning using Random Feature Corruption
# ===========================================================================

class SCARFModel(BaseContrastiveModel):
    """
    SCARF – kontrastowe uczenie na danych tabelarycznych przez losowe
    uszkadzanie kolumn.

    Augmentacja
    -----------
    Dla każdego wiersza x losowo wybieramy `corruption_rate` kolumn i
    zastępujemy ich wartości próbkami z empirycznego rozkładu brzegowego
    tej kolumny (tzn. innymi wartościami tej samej kolumny z batcha).

    Parametry
    ---------
    num_features   : int   – liczba kolumn (cech) wejściowych
    corruption_rate: float – odsetek kolumn do uszkodzenia (domyślnie 0.6)
    temperature    : float – τ w InfoNCE
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        num_classes: int = 2,
        corruption_rate: float = 0.6,
        temperature: float = 0.5,
    ) -> None:
        super().__init__(
            encoder_type="mlp",
            input_dim=num_features,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_classes=num_classes,
        )
        self.num_features = num_features
        self.corruption_rate = corruption_rate
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Augmentacja
    # ------------------------------------------------------------------

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bernoulli(p=corruption_rate) per komórka — zgodnie z Bahri et al. 2021.
        Dawca jest zawsze różny od wiersza źródłowego (brak self-corruption).
        """
        B, F = x.shape

        if B < 2:
            return x.clone()

        col_mask = torch.rand(B, F, device=x.device) < self.corruption_rate

        # Losuj dawcę != i: losuj z [0,B-1], przesuń >= i o +1
        donors = torch.randint(0, B - 1, (B, F), device=x.device)
        row_idx = torch.arange(B, device=x.device).unsqueeze(1)
        donors[donors >= row_idx] += 1

        col_idx = torch.arange(F, device=x.device).unsqueeze(0).expand(B, F)
        donor_values = x[donors, col_idx]
        x_corrupted = torch.where(col_mask, donor_values, x)

        return x_corrupted

    # ------------------------------------------------------------------

    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Anchor = oryginał, positive = wersja uszkodzona (Bahri et al. 2021)."""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        x_anchor = x
        x_positive = self._corrupt(x)

        # Embedding → projekcja → normalizacja
        z1 = F.normalize(self.projector(self.encoder(x_anchor)), dim=-1)
        z2 = F.normalize(self.projector(self.encoder(x_positive)), dim=-1)

        loss = info_nce_loss(z1, z2, self.temperature)
        return {"loss": loss}


# ===========================================================================
# 2. SubTab – Subsetting Features of Tabular Data
# ===========================================================================

class SubTabModel(BaseContrastiveModel):
    """
    SubTab – uczenie reprezentacji przez dzielenie kolumn na zachodzące
    na siebie podzbiory i maksymalizowanie podobieństwa między wektorami
    z tego samego wiersza.

    Augmentacja
    -----------
    Dzielimy F kolumn na `n_subsets` podzbiorów o rozmiarze `subset_size`.
    Podzbiory mogą na siebie zachodzić (`overlap` kolumn wspólnych z
    sąsiednim podzbiorem). Do każdego podzbioru dodajemy szum Gaussa.

    Parametry
    ---------
    num_features : int   – liczba kolumn wejściowych
    n_subsets    : int   – liczba podzbiorów do wygenerowania
    overlap      : int   – liczba nakładających się kolumn między podzbiorami
    noise_std    : float – odchylenie standardowe szumu Gaussa
    temperature  : float – τ w InfoNCE
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        num_classes: int = 2,
        n_subsets: int = 3,
        overlap: int = 2,
        noise_std: float = 0.1,
        temperature: float = 0.5,
        subset_seed: Optional[int] = None,
        recon_weight: float = 1.0,
    ) -> None:
        super().__init__(
            encoder_type="mlp",
            input_dim=num_features,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_classes=num_classes,
        )

        self.num_features = num_features
        self.n_subsets = n_subsets
        self.noise_std = noise_std
        self.temperature = temperature
        self.recon_weight = recon_weight

        # Iteracja do punktu stałego gwarantuje overlap < step,
        # by podzbiory nie były identyczne (bezwartościowy sygnał InfoNCE).
        _ov = overlap
        for _ in range(2):
            _step = max(1, (num_features - _ov) // n_subsets)
            _ov_clamped = min(_ov, max(0, _step - 1))
            if _ov_clamped == _ov:
                break
            _ov = _ov_clamped
        self.overlap = _ov_clamped
        self._step = _step  # zapamiętany krok — _build_subsets używa go bezpośrednio

        subset_size = math.ceil(num_features / n_subsets) + self.overlap
        self.subset_size = min(subset_size, num_features)

        self._subset_seed: Optional[int] = subset_seed
        self._subset_indices: List[torch.Tensor] = self._build_subsets()

        # Osobne enkodery per-podzbiór (Ucar et al. 2021) — każdy widzi tylko
        # swoje kolumny, nie pełne F z zerami (model nie uczy się wzorca zer).
        self.subset_encoders = nn.ModuleList([
            build_mlp(len(idx), hidden_dim, hidden_dim, num_layers=3)
            for idx in self._subset_indices
        ])
        self.encoder = nn.Identity()  # placeholder — właściwe enkodery są w subset_encoders

        # Dekoder rekonstruuje pełny wektor cech z embeddings (Ucar et al. 2021)
        self.decoder: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
        )

    # ------------------------------------------------------------------

    def _build_subsets(self) -> List[torch.Tensor]:
        """Tworzy listę tensorów indeksów kolumn dla każdego podzbioru."""
        if self._subset_seed is not None:
            g = torch.Generator()
            g.manual_seed(self._subset_seed)
            col_order = torch.randperm(self.num_features, generator=g).tolist()
        else:
            col_order = list(range(self.num_features))

        indices = []
        step = self._step
        for i in range(self.n_subsets):
            start = i * step
            end = min(start + self.subset_size, self.num_features)
            idx = torch.tensor(col_order[start:end], dtype=torch.long)
            indices.append(idx)
        return indices

    def _make_subset_view(
        self, x: torch.Tensor, col_idx: torch.Tensor
    ) -> torch.Tensor:
        """Ekstrahuje kolumny col_idx i dodaje szum Gaussa. Zwraca (B, S)."""
        sub = x[:, col_idx]
        noise = torch.randn_like(sub) * self.noise_std
        return sub + noise

    # ------------------------------------------------------------------

    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Contrastive (InfoNCE) + reconstruction loss (Ucar et al. 2021)."""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        device = x.device

        h_list: List[torch.Tensor] = []
        z_list: List[torch.Tensor] = []
        for i, col_idx in enumerate(self._subset_indices):
            col_idx = col_idx.to(device)
            view = self._make_subset_view(x, col_idx)
            h = self.subset_encoders[i](view)
            z = F.normalize(self.projector(h), dim=-1)
            h_list.append(h)
            z_list.append(z)

        contrastive_loss = torch.tensor(0.0, device=device)
        n_pairs = 0
        for i in range(len(z_list)):
            for j in range(i + 1, len(z_list)):
                contrastive_loss = contrastive_loss + info_nce_loss(
                    z_list[i], z_list[j], self.temperature
                )
                n_pairs += 1
        contrastive_loss = contrastive_loss / max(n_pairs, 1)

        recon_loss = torch.tensor(0.0, device=device)
        for h in h_list:
            x_hat = self.decoder(h)
            recon_loss = recon_loss + F.mse_loss(x_hat, x)
        recon_loss = recon_loss / len(h_list)

        loss = contrastive_loss + self.recon_weight * recon_loss
        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss.detach(),
            "recon_loss": recon_loss.detach(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Uśredniony embedding ze wszystkich podzbiorów (B, hidden_dim)."""
        device = x.device
        embeddings: List[torch.Tensor] = []
        for i, col_idx in enumerate(self._subset_indices):
            col_idx = col_idx.to(device)
            sub = x[:, col_idx]  # bez szumu — deterministyczna reprezentacja
            embeddings.append(self.subset_encoders[i](sub))
        return torch.stack(embeddings, dim=0).mean(dim=0)

    def linear_readout(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self._encoder_frozen):
            h = self.forward(x)
        return self.classifier(h)

    def freeze_encoder(self) -> None:
        """Zamraża wszystkie per-podzbiorowe enkodery."""
        for enc in self.subset_encoders:
            for p in enc.parameters():
                p.requires_grad = False
            enc.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self._encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        for enc in self.subset_encoders:
            for p in enc.parameters():
                p.requires_grad = True
        for p in self.encoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = False

    def train(self, mode: bool = True) -> "SubTabModel":
        super().train(mode)
        if self._encoder_frozen:
            for enc in self.subset_encoders:
                enc.eval()
            self.encoder.eval()
        return self

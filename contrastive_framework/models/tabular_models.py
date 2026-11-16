"""
tabular_models.py – Implementacje SCARF i SubTab dla danych tabelarycznych.

SCARF  : Bahri et al., 2021 – "SCARF: Self-Supervised Contrastive Learning
         using Random Feature Corruption"
SubTab : Ucar et al., 2021  – "SubTab: Subsetting Features of Tabular Data
         for Self-Supervised Representation Learning"
         Źródło: https://github.com/AstraZeneca/SubTab (Apache-2.0)
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContrastiveModel


# ===========================================================================
# Strata InfoNCE dla SCARF
# Port z: Bahri et al. 2022, arXiv:2106.15147, Algorithm 1 linia 8
# ===========================================================================

def info_nce_loss(
    z_anchor: torch.Tensor, z_positive: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    InfoNCE z kotwicy do pozytywa (Bahri et al. 2022, Algorithm 1 linia 8).

    Port z arXiv:2106.15147 Algorithm 1:
        L_cont = -log( exp(s_{i,i}/τ) / Σ_k exp(s_{i,k}/τ) )
    gdzie s_{i,k} = z_anchor[i] · z_positive[k] (cosine similarity po normalizacji).

    Zakres: negatives = wszystkie wiersze pozytywa oprócz diagonali (B-1 negatywów).
    Etykieta = pozycja na diagonali (0, 1, ..., B-1).

    Uwaga: artykuł używa τ=1 jako domyślną (Section 4).
    Poprzednia implementacja (NT-Xent symetryczny) jest opisana w DEVIATIONS.md.
    """
    B = z_anchor.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_anchor.device, requires_grad=True)
    # (B, B) — wiersze = kotwice, kolumny = kandydaci-pozytywy
    sim = torch.mm(z_anchor, z_positive.T) / temperature
    # etykieta i = i (pozytyw na diagonali)
    labels = torch.arange(B, device=z_anchor.device)
    return F.cross_entropy(sim, labels)


# ===========================================================================
# 1. SCARF – Self-Supervised Contrastive Learning using Random Feature Corruption
# Port z: Bahri et al. 2022, arXiv:2106.15147 (brak oficjalnego kodu)
# ===========================================================================

class SCARFModel(BaseContrastiveModel):
    """
    SCARF – kontrastowe uczenie na danych tabelarycznych przez losowe
    uszkadzanie cech (Bahri et al., ICLR 2022, arXiv:2106.15147).

    Korupcja (Algorithm 1):
      • Dla każdego wiersza x wylosuj DOKŁADNIE q = floor(c·M) cech bez powtórzeń
        (uniform sampling bez zamiany). Nie Bernoulli per komórka.
      • Każdą wybraną cechę j zastąp wartością v ~ X̂_j, gdzie X̂_j to równomierny
        rozkład nad {x_j : x ∈ X_train} — empiryczny rozkład brzegowy na CAŁYM
        zbiorze treningowym.
      • Wartości zastępcze są próbkowane z `marginals` — kolumn zbioru treningowego
        przekazanych przez `set_marginals()`. Bez wycieku: marginals = tylko train split.

    Para: anchor = oryginał, positive = uszkodzony (Algorithm 1 linia 7-8).

    Strata: InfoNCE (nie NT-Xent symetryczny) — L_cont = -log(e^{s_{ii}/τ} / Σ_k e^{s_{ik}/τ})
    Port z Algorithm 1 linia 8. Temperatura domyślna τ=1 (Section 4 artykułu).

    Architektura (Section 4): f = 4 warstwy ReLU, g = 2 warstwy, hidden=256.
    Nasze odstępstwo: f = 3 warstwy (build_mlp num_layers=3), g = 2 warstwy —
    opisane w DEVIATIONS.md.

    Parametry
    ---------
    num_features    : M — liczba kolumn wejściowych
    corruption_rate : c ∈ (0,1) — ułamek cech do uszkodzenia (domyślnie 0.6)
    temperature     : τ w InfoNCE (domyślnie 1.0 — wartość z artykułu Section 4)
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        num_classes: int = 2,
        corruption_rate: float = 0.6,
        temperature: float = 1.0,
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

        # Marginalne rozkłady cech: (N_train, num_features) — tylko zbiór treningowy.
        # Ustawiane przez set_marginals() PRZED trenowaniem.
        # Przechowujemy jako bufor (nie parametr), żeby był przenoszony razem z modelem.
        self._marginals: Optional[torch.Tensor] = None

    def set_marginals(self, train_data: torch.Tensor) -> None:
        """
        Zapisuje kolumny zbioru treningowego jako bufor do korupcji.

        Port z Algorithm 1 linia 2:
            X̂_j = uniform distribution over {x_j : x ∈ X} for j = 1,...,M
        MUSI być wywołane z train split (nie val/test) — zapobiega data leakage.

        Parameters
        ----------
        train_data : (N_train, num_features) — dane treningowe PRZED normalizacją
                     lub PO (spójne z danymi wejściowymi do pretrain_step).
        """
        if train_data.ndim != 2 or train_data.shape[1] != self.num_features:
            raise ValueError(
                f"set_marginals oczekuje tensora (N, {self.num_features}), "
                f"dostałem {tuple(train_data.shape)}"
            )
        # Przechowuj jako CPU tensor (może być duży; przenoszony do device w _corrupt)
        self._marginals = train_data.detach().cpu()

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Korupcja q = floor(c·M) cech bez powtórzeń na wiersz.

        Port z Algorithm 1 linia 3-5:
            q = floor(c·M)
            I_i ⊂ {1,...,M}, |I_i| = q  (uniform bez zamiany)
            x̃_j^{(i)} = v ~ X̂_j   dla j ∈ I_i

        Wartości zastępcze losowane z `self._marginals` (zbiór treningowy).
        Jeśli marginals nie są ustawione → fallback: wartości z bieżącego batcha
        (z ostrzeżeniem, dokumentowane jako odstępstwo w DEVIATIONS.md).
        """
        B, M = x.shape
        q = int(self.corruption_rate * M)   # floor przez int()
        if q == 0:
            return x.clone()

        x_tilde = x.clone()
        device = x.device

        if self._marginals is not None:
            # Optymalna ścieżka: próbki z empirycznych rozkładów brzegowych X_train
            # Port z Algorithm 1 linia 2: X̂_j = {x_j : x ∈ X_train}
            marginals = self._marginals.to(device)
            N_train = marginals.shape[0]
            for i in range(B):
                # Wybierz q cech bez powtórzeń (uniform)
                cols = torch.randperm(M, device=device)[:q]
                # Dla każdej wybranej cechy j: próbkuj v ~ X̂_j
                row_indices = torch.randint(0, N_train, (q,), device=device)
                x_tilde[i, cols] = marginals[row_indices, cols]
        else:
            # Fallback (bez set_marginals): losuj wartości z bieżącego batcha.
            # Odstępstwo od artykułu — wartości zastępcze z batcha, nie z X_train.
            # Opisane w DEVIATIONS.md.
            import warnings
            warnings.warn(
                "[SCARFModel] Marginale treningowe nie są ustawione. "
                "Korupcja używa wartości z bieżącego batcha zamiast X_train. "
                "Wywołaj model.set_marginals(train_X) przed trenowaniem.",
                RuntimeWarning,
                stacklevel=2,
            )
            if B < 2:
                return x.clone()
            for i in range(B):
                cols = torch.randperm(M, device=device)[:q]
                # Losuj dawcę różnego od wiersza i
                donor_pool = torch.arange(B, device=device)
                donor_pool = donor_pool[donor_pool != i]
                donor_indices = donor_pool[
                    torch.randint(0, B - 1, (q,), device=device)
                ]
                x_tilde[i, cols] = x[donor_indices, cols]

        return x_tilde

    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Jeden krok pre-treningu SCARF.

        Port z Algorithm 1:
          anchor   = encoder(x_orig)      → z_i
          positive = encoder(corrupt(x))  → z̃_i
          loss = InfoNCE(z_anchor, z_positive)
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        # anchor: oryginalny wiersz
        z_anchor   = F.normalize(self.projector(self.encoder(x)), dim=-1)
        # positive: uszkodzony wiersz (korupcja q = floor(c*M) cech)
        z_positive = F.normalize(self.projector(self.encoder(self._corrupt(x))), dim=-1)
        loss = info_nce_loss(z_anchor, z_positive, self.temperature)
        return {"loss": loss}


# ===========================================================================
# 2. SubTab – pomocnicze moduły
# Port z: third_party_reference/SubTab/utils/model_utils.py
# ===========================================================================

class _HiddenLayers(nn.Module):
    """
    Warstwy ukryte z opcjonalnym BN i Dropout.
    Port z: SubTab/utils/model_utils.py (HiddenLayers)
    dims[0] → dims[1] → ... → dims[-2]  (bez ostatniej warstwy)
    """

    def __init__(self, dims: List[int], is_batchnorm: bool = False,
                 is_dropout: bool = False, dropout_rate: float = 0.2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(1, len(dims) - 1):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if is_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i]))
            layers.append(nn.LeakyReLU(inplace=False))
            if is_dropout:
                layers.append(nn.Dropout(dropout_rate))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Encoder(nn.Module):
    """
    Enkoder – przyjmuje podzbiór (B, subset_dim) → latent (B, latent_dim).
    Port z: SubTab/utils/model_utils.py (Encoder / ShallowEncoder)
    dims[0] = subset_dim (ustawiane przez wywołującego)
    """

    def __init__(self, dims: List[int], is_batchnorm: bool = False,
                 is_dropout: bool = False, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.hidden = _HiddenLayers(dims, is_batchnorm, is_dropout, dropout_rate)
        # Ostatnia warstwa: dims[-2] → dims[-1]
        self.latent = nn.Linear(dims[-2], dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return self.latent(h)


class _Decoder(nn.Module):
    """
    Dekoder – latent (B, latent_dim) → rekonstrukcja (B, recon_dim).
    Port z: SubTab/utils/model_utils.py (Decoder / ShallowDecoder)
    dims_rev = odwrócone dims enkodera; recon_dim = dims_rev[-1]
    """

    def __init__(self, dims_rev: List[int], is_batchnorm: bool = False,
                 is_dropout: bool = False, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.hidden = _HiddenLayers(dims_rev, is_batchnorm, is_dropout, dropout_rate)
        self.logits = nn.Linear(dims_rev[-2], dims_rev[-1])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.hidden(h)
        return self.logits(h)


class _AEWrapper(nn.Module):
    """
    Jeden wspólny Autoenkoder + 2-warstwowy projector dla wszystkich podzbiorów.
    Port z: SubTab/utils/model_utils.py (AEWrapper)

    forward: (B, subset_dim) → z (B, latent_dim), latent (B, latent_dim), xrecon (B, recon_dim)
    """

    def __init__(
        self,
        subset_dim: int,
        recon_dim: int,
        dims: List[int],       # [hidden, ..., latent] (bez input_dim i recon_dim)
        normalize: bool = True,
        p_norm: int = 2,
        is_batchnorm: bool = False,
        is_dropout: bool = False,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.p_norm = p_norm

        # Pełne dims enkodera: [subset_dim] + dims
        # np. dims=[64,32] → enc_dims=[15, 64, 32]
        # Enkoder: HiddenLayers [15→64] + latent [64→32]
        enc_dims = [subset_dim] + list(dims)
        # Dekoder symetryczny: odwracamy kolejność wymiarów ukrytych
        # Dekoder: HiddenLayers [32→64] + logits [64→recon_dim]
        # dec_dims = [latent_dim, hidden_reversed..., recon_dim]
        dec_dims = list(reversed(enc_dims[1:])) + [recon_dim]
        # np. enc_dims=[15,64,32] → reversed([64,32])=[32,64] → dec_dims=[32,64,recon_dim]

        self.encoder = _Encoder(enc_dims, is_batchnorm, is_dropout, dropout_rate)
        self.decoder = _Decoder(dec_dims, is_batchnorm, is_dropout, dropout_rate)

        latent_dim = enc_dims[-1]
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        z = F.leaky_relu(self.proj1(latent))
        z = self.proj2(z)
        if self.normalize:
            z = F.normalize(z, p=self.p_norm, dim=1)
        x_recon = self.decoder(latent)
        return z, latent, x_recon


# ===========================================================================
# JointLoss
# Port z: third_party_reference/SubTab/utils/loss_functions.py (JointLoss)
# ===========================================================================

class _JointLoss(nn.Module):
    """
    Strata łączona SubTab:
      total = recon_loss
            + [contrastive_loss]  jeśli use_contrastive=True
            + [distance_loss]     jeśli use_distance=True

    Contrastive: NT-Xent na projekcjach z (w układzie 2N×2N).
    Distance:    MSE między zi i zj (podzbiorami tej samej próbki).
    Recon:       MSE (reconstruction=True) lub BCE (reconstruction=False).

    Port z: SubTab/utils/loss_functions.py
    """

    def __init__(
        self,
        tau: float = 0.1,
        use_contrastive: bool = True,
        use_distance: bool = True,
        use_reconstruction: bool = True,
        cosine_similarity: bool = False,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.use_contrastive = use_contrastive
        self.use_distance = use_distance
        self.use_reconstruction = use_reconstruction
        self.cosine_similarity = cosine_similarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    @staticmethod
    def _mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE znormalizowany przez rozmiar batcha (jak getMSEloss w oryginale)."""
        return torch.sum((recon - target) ** 2) / recon.size(0)

    def _xneg_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent na macierzy 2N×2N (z = concat([zi, zj], dim=0)).
        Port z: JointLoss.XNegloss()
        """
        B2 = z.size(0)       # 2N
        N = B2 // 2

        if self.cosine_similarity:
            # (2N, 1, D) × (1, 2N, D) → (2N, 2N)
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        else:
            sim = torch.mm(z, z.T)

        # Pozytywne pary: diagonala pierwszego kwadrantu i trzeciego kwadrantu
        l_pos = torch.diag(sim, N)                        # (N,)
        r_pos = torch.diag(sim, -N)                       # (N,)
        positives = torch.cat([l_pos, r_pos]).unsqueeze(1)  # (2N, 1)

        # Maska negatywów: wszystko oprócz diagonali czterech kwadrantów
        eye2N = torch.eye(B2, dtype=torch.bool, device=z.device)
        q1 = torch.zeros(B2, B2, dtype=torch.bool, device=z.device)
        q1[:N, N:] = torch.eye(N, dtype=torch.bool, device=z.device)
        q3 = q1.T
        pos_mask = eye2N | q1 | q3
        neg_mask = ~pos_mask  # (2N, 2N)

        negatives = sim[neg_mask].view(B2, -1)             # (2N, 2N-2)
        logits = torch.cat([positives, negatives], dim=1) / self.tau  # (2N, 2N-1)
        labels = torch.zeros(B2, dtype=torch.long, device=z.device)
        loss = self.criterion(logits, labels) / B2
        return loss

    def forward(
        self,
        z: torch.Tensor,
        x_recon: torch.Tensor,
        x_orig: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z      : (2B, D)  – projekcje (concat zi i zj)
        x_recon: (2B, F)  – rekonstrukcja dekodera
        x_orig : (2B, F)  – oryginalny input (concat oryginału z sobą)

        Zwraca: (total, closs, rloss, zloss)
        """
        # Strata rekonstrukcji
        if self.use_reconstruction:
            rloss = self._mse(x_recon, x_orig)
        else:
            rloss = F.binary_cross_entropy(
                torch.sigmoid(x_recon), x_orig, reduction="sum"
            ) / x_orig.size(0)

        # Inicjalizacja placeholderów (jak w oryginale: closs = rloss jako placeholder)
        closs = rloss
        zloss = rloss
        total = rloss

        if self.use_contrastive:
            closs = self._xneg_loss(z)
            total = total + closs

        if self.use_distance:
            N = z.size(0) // 2
            zi, zj = z[:N], z[N:]
            zloss = self._mse(zi, zj)
            total = total + zloss

        return total, closs, rloss, zloss


# ===========================================================================
# 2. SubTabModel
# ===========================================================================

class SubTabModel(BaseContrastiveModel):
    """
    SubTab – wierna reimplementacja Ucar et al. 2021.

    Architektura zgodna z config/mnist.yaml + artykuł:
      • JEDEN wspólny AEWrapper (enkoder + dekoder + projekcja) dla wszystkich podzbiorów
      • subset_dim = num_features // n_subsets + n_overlap
        gdzie n_overlap = int(overlap_ratio * (num_features // n_subsets))
      • Szum: maska Bernoulliego(masking_ratio) × noise_type
        (swap_noise / gaussian_noise / zero_out)
      • Strata: recon (MSE) + NT-Xent (na z) + distance (MSE między zi i zj)
      • Agregacja do ewaluacji: mean reprezentacji wszystkich podzbiorów (bez szumu)

    Parametry
    ---------
    num_features    : liczba kolumn wejściowych
    n_subsets       : liczba podzbiorów (domyślnie 4, z mnist.yaml)
    overlap         : ułamek [0,1) kolumn nakładających się (domyślnie 0.75)
    masking_ratio   : odsetek kolumn do zaszumienia (domyślnie 0.3)
    noise_type      : "swap_noise" | "gaussian_noise" | "zero_out"
    noise_level     : σ dla gaussian_noise (domyślnie 0.1)
    tau             : temperatura NT-Xent (domyślnie 0.1)
    use_contrastive : czy używać straty kontrastowej (domyślnie True)
    use_distance    : czy używać straty dystansu (domyślnie True)
    reconstruct_subset : rekonstruuj podzbiór zamiast pełnego wektora (domyślnie False)
    dims            : wymiary ukrytych warstw enkodera (bez input_dim i latent_dim)
                      domyślnie [hidden_dim, hidden_dim] (jak w mnist.yaml: [784, 784])
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        proj_dim: int = 128,          # nieużywane (latent_dim bierze z dims)
        num_classes: int = 2,
        # SubTab-specyficzne (z mnist.yaml / artykułu)
        n_subsets: int = 4,
        overlap: float = 0.75,         # ułamek [0, 1)
        masking_ratio: float = 0.3,
        noise_type: str = "swap_noise",
        noise_level: float = 0.1,
        tau: float = 0.1,
        use_contrastive: bool = True,
        use_distance: bool = True,
        use_reconstruction: bool = True,
        reconstruct_subset: bool = False,
        cosine_similarity: bool = False,
        normalize: bool = True,
        is_batchnorm: bool = False,
        is_dropout: bool = False,
        dropout_rate: float = 0.2,
        # dims: [hidden, ..., latent] enkodera
        dims: Optional[List[int]] = None,
    ) -> None:
        # Obliczamy subset_dim przed inicjalizacją BaseContrastiveModel
        n_col_subset = num_features // n_subsets
        n_overlap = int(overlap * n_col_subset)
        subset_dim = n_col_subset + n_overlap

        # latent_dim: ostatni element dims
        _dims = dims if dims is not None else [hidden_dim, hidden_dim]
        latent_dim = _dims[-1]

        super().__init__(
            encoder_type="mlp",
            input_dim=num_features,
            hidden_dim=latent_dim,
            proj_dim=latent_dim,
            num_classes=num_classes,
        )

        self.num_features = num_features
        self.n_subsets = n_subsets
        self.overlap = overlap
        self.masking_ratio = masking_ratio
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.reconstruct_subset = reconstruct_subset

        # n_col_subset i n_overlap do subset_generator
        self._n_col_subset = n_col_subset
        self._n_overlap = n_overlap

        # Flaga: czy potrzebujemy kombinacji podzbiorów
        self._is_combination = use_contrastive or use_distance

        # Wymiar wyjścia dekodera
        recon_dim = subset_dim if reconstruct_subset else num_features
        # Gdy is_combination=True, dekoder dostaje latent z concat(xi, xj) → recon_dim×2
        # Ale recon target to też concat(orig, orig) → recon_dim×2.
        # W rzeczywistości dekoder działa per-próbkę na latent: (2B, latent_dim) → (2B, recon_dim)
        # i porównujemy z Xorig = concat(orig, orig): (2B, recon_dim).
        # Więc recon_dim pozostaje niezmienione.

        # Jeden wspólny AEWrapper dla wszystkich podzbiorów
        self.ae = _AEWrapper(
            subset_dim=subset_dim,
            recon_dim=recon_dim,
            dims=_dims,
            normalize=normalize,
            p_norm=2,
            is_batchnorm=is_batchnorm,
            is_dropout=is_dropout,
            dropout_rate=dropout_rate,
        )

        # Strata
        self.joint_loss = _JointLoss(
            tau=tau,
            use_contrastive=use_contrastive,
            use_distance=use_distance,
            use_reconstruction=use_reconstruction,
            cosine_similarity=cosine_similarity,
        )

        # Klasyfikator na latent_dim (nadpisujemy z BaseContrastiveModel)
        self.classifier = nn.Linear(latent_dim, num_classes)

        # encoder w BaseContrastiveModel → wskazuje na encoder wewnętrzny AE
        # Używane przez freeze_encoder / unfreeze_encoder z klasy bazowej
        self.encoder = self.ae.encoder

    # ------------------------------------------------------------------
    # Generowanie podzbiorów i szumu
    # Port z: SubTab/src/model.py (subset_generator, generate_noisy_xbar, mask_generator)
    # ------------------------------------------------------------------

    def _subset_column_indices(self, mode: str = "test") -> List[List[int]]:
        """
        Wyznacza indeksy kolumn dla każdego podzbioru.
        Port z: SubTab/src/model.py (subset_generator, linie 311–355)

        mode="train" → permutuje kolejność podzbiorów (augmentacja kolejności)
        mode="test"  → stała kolejność (deterministyczna)
        """
        n_col = self.num_features
        n_subsets = self.n_subsets
        n_col_subset = self._n_col_subset
        n_overlap = self._n_overlap
        col_idx = list(range(n_col))

        if mode == "train":
            perm = torch.randperm(n_subsets).tolist()
        else:
            perm = list(range(n_subsets))

        result: List[List[int]] = []
        for i in perm:
            if i == 0:
                start = 0
                stop = n_col_subset + n_overlap
            else:
                start = i * n_col_subset - n_overlap
                stop = (i + 1) * n_col_subset
            result.append(col_idx[start:stop])

        # Dummy copy gdy pojedynczy podzbiór
        if len(result) == 1:
            result.append(result[0])
        return result

    def _generate_noisy_xbar(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generuje zaszumioną wersję x.
        Port z: SubTab/src/model.py (generate_noisy_xbar)
        swap_noise: shuffle kolumnowy, gaussian_noise: x + N(0, σ), zero_out: zera
        """
        B, D = x.shape
        if self.noise_type == "swap_noise":
            x_bar = x.clone()
            for col in range(D):
                idx = torch.randperm(B, device=x.device)
                x_bar[:, col] = x[idx, col]
        elif self.noise_type == "gaussian_noise":
            x_bar = x + torch.randn_like(x) * self.noise_level
        else:
            # zero_out
            x_bar = torch.zeros_like(x)
        return x_bar

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generuje zaszumiony podzbiór: maska Bernoulliego(masking_ratio) × noise_type.
        Port z: SubTab/src/model.py (subset_generator linie 346–353)
        """
        x_bar_noisy = self._generate_noisy_xbar(x)
        # Maska binarna: (B, D) z masking_ratio
        mask = (torch.rand_like(x) < self.masking_ratio).float()
        return x * (1.0 - mask) + x_bar_noisy * mask

    def _build_subset_list(
        self, x: torch.Tensor, mode: str = "train"
    ) -> List[torch.Tensor]:
        """
        Generuje listę podzbiorów (z szumem w trybie train).
        Każdy element: (B, subset_dim) tensor na device x.
        """
        col_idx_list = self._subset_column_indices(mode)
        result: List[torch.Tensor] = []
        for col_idx in col_idx_list:
            idx_t = torch.tensor(col_idx, dtype=torch.long, device=x.device)
            x_sub = x[:, idx_t]
            if mode == "train":
                x_sub = self._add_noise(x_sub)
            result.append(x_sub)
        return result

    def _get_combinations(
        self, subset_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Tworzy listę konkatenacji par podzbiorów.
        Port z: SubTab/src/model.py (get_combinations_of_subsets)
        Wynik: [(xi||xj)] o kształcie (2B, subset_dim)
        """
        combos: List[torch.Tensor] = []
        for xi, xj in itertools.combinations(subset_list, 2):
            combos.append(torch.cat([xi, xj], dim=0))
        return combos

    # ------------------------------------------------------------------
    # Interfejs BaseContrastiveModel
    # ------------------------------------------------------------------

    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Strata SubTab: recon + NT-Xent + distance.
        Port z: SubTab/src/model.py (update_autoencoder + validate)
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        B = x.size(0)

        # Generuj podzbiory z szumem
        subset_list = self._build_subset_list(x, mode="train")

        # Oryginalny x (bez szumu) zduplikowany: (2B, F) – target rekonstrukcji
        # Gdy reconstruct_subset=False, target to pełny wektor x
        x_orig_2 = torch.cat([x, x], dim=0)  # (2B, num_features)

        total_losses: List[torch.Tensor] = []
        c_losses: List[torch.Tensor] = []
        r_losses: List[torch.Tensor] = []
        z_losses: List[torch.Tensor] = []

        if self._is_combination:
            # Kombinacje par: AE widzi concat(xi, xj) jako (2B, subset_dim)
            combos = self._get_combinations(subset_list)
            for x_input in combos:
                z, latent, x_recon = self.ae(x_input)
                # Target rekonstrukcji
                x_target = x_input if self.reconstruct_subset else x_orig_2
                tloss, closs, rloss, zloss = self.joint_loss(z, x_recon, x_target)
                total_losses.append(tloss)
                c_losses.append(closs)
                r_losses.append(rloss)
                z_losses.append(zloss)
        else:
            # Bez kombinacji: każdy podzbiór konkatenowany sam z sobą
            for x_sub in subset_list:
                x_input = torch.cat([x_sub, x_sub], dim=0)
                z, latent, x_recon = self.ae(x_input)
                x_target = x_input if self.reconstruct_subset else x_orig_2
                tloss, closs, rloss, zloss = self.joint_loss(z, x_recon, x_target)
                total_losses.append(tloss)
                c_losses.append(closs)
                r_losses.append(rloss)
                z_losses.append(zloss)

        n = len(total_losses)
        loss = sum(total_losses) / n

        return {
            "loss": loss,
            "contrastive_loss": (sum(c_losses) / n).detach(),
            "recon_loss": (sum(r_losses) / n).detach(),
            "distance_loss": (sum(z_losses) / n).detach(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Agregacja reprezentacji (mean) ze wszystkich podzbiorów – bez szumu.
        Port z: SubTab readme + artykuł (aggregation=mean, Sec. 3.4)
        Zwraca: (B, latent_dim)
        """
        col_idx_list = self._subset_column_indices(mode="test")
        latents: List[torch.Tensor] = []
        for col_idx in col_idx_list:
            idx_t = torch.tensor(col_idx, dtype=torch.long, device=x.device)
            x_sub = x[:, idx_t]
            latent = self.ae.encoder(x_sub)
            latents.append(latent)
        return torch.stack(latents, dim=0).mean(dim=0)   # (B, latent_dim)

    def linear_readout(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self._encoder_frozen):
            h = self.forward(x)
        return self.classifier(h)

    def freeze_encoder(self) -> None:
        """Zamraża wspólny AE encoder (i projector w AE)."""
        for p in self.ae.encoder.parameters():
            p.requires_grad = False
        self.ae.encoder.eval()
        # Zamraż też projektor AE (używany podczas pretrain, nie ewaluacji)
        for p in self.ae.proj1.parameters():
            p.requires_grad = False
        for p in self.ae.proj2.parameters():
            p.requires_grad = False
        self._encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        for p in self.ae.encoder.parameters():
            p.requires_grad = True
        for p in self.ae.proj1.parameters():
            p.requires_grad = True
        for p in self.ae.proj2.parameters():
            p.requires_grad = True
        self._encoder_frozen = False

    def train(self, mode: bool = True) -> "SubTabModel":
        super().train(mode)
        if self._encoder_frozen:
            self.ae.encoder.eval()
        return self


# ===========================================================================
# RandomInit-SubTabArch baseline
# ===========================================================================

class RandomInitSubTabArchModel(BaseContrastiveModel):
    """
    Baseline z losową inicjalizacją, identyczna architektura co SubTabModel.
    Ten sam AEWrapper + ta sama agregacja mean po podzbiorach.
    Bez pre-treningu – losowe wagi.

    Umożliwia uczciwe porównanie SubTab z baseline przy identycznej architekturze.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        num_classes: int = 2,
        n_subsets: int = 4,
        overlap: float = 0.75,
        dims: Optional[List[int]] = None,
        normalize: bool = True,
        is_batchnorm: bool = False,
        is_dropout: bool = False,
        dropout_rate: float = 0.2,
        # Przyjmujemy SubTab-params dla zgodności z filter_model_kwargs
        masking_ratio: float = 0.3,
        noise_type: str = "swap_noise",
        noise_level: float = 0.1,
        tau: float = 0.1,
        use_contrastive: bool = True,
        use_distance: bool = True,
        use_reconstruction: bool = True,
        reconstruct_subset: bool = False,
        cosine_similarity: bool = False,
    ) -> None:
        n_col_subset = num_features // n_subsets
        n_overlap = int(overlap * n_col_subset)
        subset_dim = n_col_subset + n_overlap
        _dims = dims if dims is not None else [hidden_dim, hidden_dim]
        latent_dim = _dims[-1]
        recon_dim = subset_dim if reconstruct_subset else num_features

        super().__init__(
            encoder_type="mlp",
            input_dim=num_features,
            hidden_dim=latent_dim,
            proj_dim=latent_dim,
            num_classes=num_classes,
        )
        self.num_features = num_features
        self.n_subsets = n_subsets
        self._n_col_subset = n_col_subset
        self._n_overlap = n_overlap

        self.ae = _AEWrapper(
            subset_dim=subset_dim,
            recon_dim=recon_dim,
            dims=_dims,
            normalize=normalize,
            is_batchnorm=is_batchnorm,
            is_dropout=is_dropout,
            dropout_rate=dropout_rate,
        )
        self.encoder = self.ae.encoder
        self.classifier = nn.Linear(latent_dim, num_classes)

    def _subset_column_indices(self) -> List[List[int]]:
        n_col = self.num_features
        n_subsets = self.n_subsets
        n_col_subset = self._n_col_subset
        n_overlap = self._n_overlap
        col_idx = list(range(n_col))
        result: List[List[int]] = []
        for i in range(n_subsets):
            if i == 0:
                start, stop = 0, n_col_subset + n_overlap
            else:
                start = i * n_col_subset - n_overlap
                stop = (i + 1) * n_col_subset
            result.append(col_idx[start:stop])
        if len(result) == 1:
            result.append(result[0])
        return result

    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return {"loss": torch.tensor(0.0, device=x.device, requires_grad=True)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean aggregation po podzbiorach – identyczna z SubTabModel.forward()."""
        latents: List[torch.Tensor] = []
        for col_idx in self._subset_column_indices():
            idx_t = torch.tensor(col_idx, dtype=torch.long, device=x.device)
            latents.append(self.ae.encoder(x[:, idx_t]))
        return torch.stack(latents, dim=0).mean(dim=0)

    def linear_readout(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self._encoder_frozen):
            h = self.forward(x)
        return self.classifier(h)

    def freeze_encoder(self) -> None:
        for p in self.ae.encoder.parameters():
            p.requires_grad = False
        self.ae.encoder.eval()
        self._encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        for p in self.ae.encoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = False

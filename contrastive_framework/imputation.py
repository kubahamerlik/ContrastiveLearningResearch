"""
imputation.py – Metody imputacji brakujących wartości w danych tabelarycznych.

Realizuje sugestię promotora: *"podniesienie merytoryczne algorytmów do
augmentacji i **imputacji**"*. Imputacja oparta na denoising autoencoder
(DAE) jest siostrzaną techniką do contrastive learning – obie operują na
zadaniu *zamaskuj fragment danych → odtwórz oryginał*. Różnica:
  - Contrastive: dwa zamaskowane widoki → ucz reprezentacji żeby były podobne
  - Denoising:   jeden zamaskowany widok → ucz dokładnej rekonstrukcji

Trzy klasy z jednolitym API (sklearn-style):

    imputer = MeanImputer()                # albo KNNImputer() / DenoisingAutoencoderImputer()
    imputer.fit(X_train_with_nan)
    X_filled = imputer.transform(X_test_with_nan)

Plus helper `inject_missing(X, rate)` do tworzenia syntetycznych braków
w celach benchmarkingowych.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# BaseImputer – wspólny interfejs
# ---------------------------------------------------------------------------

class BaseImputer(ABC):
    """
    Wspólny interfejs dla wszystkich metod imputacji.

    Konwencje:
      - X może być `np.ndarray` lub `torch.Tensor`
      - NaN reprezentowane jako `np.nan` (float) – `torch.isnan` też je łapie
      - `fit` uczy parametry imputera (np. średnie kolumn)
      - `transform` zwraca kopię X z wypełnionymi NaN, ten sam dtype/shape

    Wzorzec sklearn (`fit` zwraca `self`) ułatwia chainowanie.
    """

    @abstractmethod
    def fit(self, X) -> "BaseImputer": ...

    @abstractmethod
    def transform(self, X): ...

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Helpery wspólne dla wszystkich imputerów
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    @staticmethod
    def _restore_type(X_filled: np.ndarray, X_orig):
        """Konwertuje z powrotem na typ wejścia (np / torch)."""
        if isinstance(X_orig, torch.Tensor):
            return torch.from_numpy(X_filled).to(
                dtype=X_orig.dtype, device=X_orig.device
            )
        return X_filled


# ---------------------------------------------------------------------------
# 1. MeanImputer – baseline (średnia z kolumny)
# ---------------------------------------------------------------------------

class MeanImputer(BaseImputer):
    """
    Najprostsza imputacja: każdą brakującą wartość w kolumnie *j* zastępuje
    średnią pozostałych (nie-NaN) wartości w kolumnie *j*.

    Zalety: O(N×F), brak hiperparametrów, deterministyczne.
    Wady: niszczy korelacje między cechami, traci wariancję.

    Implementuje "naive baseline" do porównania z bardziej wyrafinowanymi
    metodami. W literaturze tabular SSL standardowy punkt odniesienia
    (np. VIME, TabNet ablation studies).
    """

    def __init__(self, strategy: str = "mean") -> None:
        if strategy not in ("mean", "median"):
            raise ValueError(f"strategy musi być 'mean' lub 'median', dostałem '{strategy}'")
        self.strategy = strategy
        self.column_values_: Optional[np.ndarray] = None

    def fit(self, X) -> "MeanImputer":
        X_np = self._to_numpy(X).astype(np.float64)
        if self.strategy == "mean":
            self.column_values_ = np.nanmean(X_np, axis=0)
        else:
            self.column_values_ = np.nanmedian(X_np, axis=0)
        # Jeśli cała kolumna jest NaN → użyj 0
        self.column_values_ = np.nan_to_num(self.column_values_, nan=0.0)
        return self

    def transform(self, X):
        if self.column_values_ is None:
            raise RuntimeError("Wywołaj fit() przed transform().")
        X_np = self._to_numpy(X).astype(np.float64)
        nan_mask = np.isnan(X_np)
        # Broadcast: column_values_ ma shape (F,), X_np ma (N, F)
        col_idx = np.where(nan_mask)[1]
        X_np[nan_mask] = self.column_values_[col_idx]
        return self._restore_type(X_np.astype(np.float32), X)


# ---------------------------------------------------------------------------
# 2. KNNImputerWrapper – wykorzystuje sklearn (K nearest neighbors)
# ---------------------------------------------------------------------------

class KNNImputerWrapper(BaseImputer):
    """
    Imputacja przez K najbliższych sąsiadów (sklearn.impute.KNNImputer).

    Dla każdego wiersza z NaN znajduje K najbliższych wierszy (wg euklidesa
    na cechach które *nie są* NaN), uśrednia ich wartości w brakujących
    kolumnach. Lepsze niż średnia – zachowuje lokalne korelacje.

    Zalety: nieparametryczna, zachowuje strukturę manifoldu danych.
    Wady: O(N²) per próbka przy `transform` – wolne dla dużych zbiorów.

    Parameters
    ----------
    n_neighbors : K w KNN (typowo 3–10)
    """

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self._sk_imputer = None

    def fit(self, X) -> "KNNImputerWrapper":
        try:
            from sklearn.impute import KNNImputer as _SKL_KNN
        except ImportError as e:
            raise ImportError(
                "KNNImputerWrapper wymaga scikit-learn. "
                "Zainstaluj: pip install scikit-learn"
            ) from e

        X_np = self._to_numpy(X).astype(np.float64)
        self._sk_imputer = _SKL_KNN(n_neighbors=self.n_neighbors)
        self._sk_imputer.fit(X_np)
        return self

    def transform(self, X):
        if self._sk_imputer is None:
            raise RuntimeError("Wywołaj fit() przed transform().")
        X_np = self._to_numpy(X).astype(np.float64)
        X_filled = self._sk_imputer.transform(X_np).astype(np.float32)
        return self._restore_type(X_filled, X)


# ---------------------------------------------------------------------------
# 3. DenoisingAutoencoderImputer – metoda samonadzorowana (SSL-flavored)
# ---------------------------------------------------------------------------

class _DAENet(nn.Module):
    """Symetryczny autoencoder MLP do rekonstrukcji zamaskowanych cech."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 bottleneck_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DenoisingAutoencoderImputer(BaseImputer):
    """
    Imputacja przez denoising autoencoder (DAE).

    Mechanizm:
      1. fit(X): trenuje autoencoder na pełnych wierszach (lub po wstępnej
         imputacji średnią). Podczas treningu **losowo maskuje** każdy
         input (zerowanie cech z prawdopodobieństwem `mask_prob`) i uczy
         go rekonstruować oryginał. Sieć uczy się relacji między cechami.
      2. transform(X): dla wierszy z NaN robi pierwszą imputację średnią,
         przepuszcza przez DAE, podmienia w *miejscach NaN* na predykcję.

    To **bezpośredni krewny SSL** – Masked Feature Modeling (analogia BERT
    dla tabeli). Z perspektywy pracy: drugi paradygmat SSL obok
    contrastive (denoising/predictive vs contrastive).

    Parameters
    ----------
    hidden_dim      : szerokość warstwy ukrytej autoencodera
    bottleneck_dim  : wymiar reprezentacji (powinien być < input_dim)
    mask_prob       : prawdopodobieństwo zamaskowania cechy podczas treningu
    epochs          : liczba epok pre-treningu
    batch_size      : batch dla SGD
    lr              : learning rate (Adam)
    device          : 'cuda'/'cpu' (auto-detect)
    verbose         : printuje loss co 10 epok
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        bottleneck_dim: int = 32,
        mask_prob: float = 0.2,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.mask_prob = mask_prob
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = (device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose

        self._net: Optional[_DAENet] = None
        self._fallback_imputer: Optional[MeanImputer] = None

    # ------------------------------------------------------------------

    def fit(self, X) -> "DenoisingAutoencoderImputer":
        X_np = self._to_numpy(X).astype(np.float32)
        N, F = X_np.shape

        # Pierwsza imputacja – średnia (potrzebna żeby dostać "czyste" wejście DAE)
        self._fallback_imputer = MeanImputer().fit(X_np)
        X_clean = self._fallback_imputer.transform(X_np)
        X_clean_t = torch.from_numpy(X_clean).float().to(self.device)

        self._net = _DAENet(
            input_dim=F,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction="mean")

        self._net.train()
        for epoch in range(1, self.epochs + 1):
            # Mini-batch SGD
            perm = torch.randperm(N, device=self.device)
            epoch_losses = []
            for start in range(0, N, self.batch_size):
                idx = perm[start: start + self.batch_size]
                if len(idx) < 2:  # BatchNorm wymaga ≥ 2 próbek
                    continue
                x = X_clean_t[idx]

                # Maska: 1 = zachowaj, 0 = zamaskuj (zeruj)
                keep_mask = (torch.rand_like(x) > self.mask_prob).float()
                x_masked = x * keep_mask

                optimizer.zero_grad()
                x_recon = self._net(x_masked)
                # Strata liczona **tylko na zamaskowanych** cechach
                # (standard DAE – uczymy rekonstrukcji tego co było ukryte)
                mask_loss = (1 - keep_mask)
                loss = ((x_recon - x) ** 2 * mask_loss).sum() / (mask_loss.sum() + 1e-8)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            if self.verbose and (epoch % 10 == 0 or epoch == 1):
                mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
                print(f"  [DAE] Epoka {epoch:>3}/{self.epochs}  loss={mean_loss:.4f}")

        self._net.eval()
        return self

    # ------------------------------------------------------------------

    @torch.no_grad()
    def transform(self, X):
        if self._net is None or self._fallback_imputer is None:
            raise RuntimeError("Wywołaj fit() przed transform().")

        X_np = self._to_numpy(X).astype(np.float32)
        nan_mask = np.isnan(X_np)

        # 1. Pre-fill średnią żeby dać sieci "czyste" wejście
        X_clean = self._fallback_imputer.transform(X_np)
        X_clean_t = torch.from_numpy(X_clean).float().to(self.device)

        # 2. Forward przez DAE
        X_recon = self._net(X_clean_t).cpu().numpy()

        # 3. Podmieniaj wartości tylko w miejscach gdzie był NaN
        X_out = X_clean.copy()
        X_out[nan_mask] = X_recon[nan_mask]

        return self._restore_type(X_out, X)


# ---------------------------------------------------------------------------
# Helper: sztuczne wprowadzenie braków (do benchmarkingu)
# ---------------------------------------------------------------------------

def inject_missing(
    X,
    missing_rate: float,
    seed: int = 0,
    pattern: str = "mcar",
):
    """
    Losowo wstawia NaN do macierzy cech – do benchmarkingu imputacji.

    Parameters
    ----------
    X            : (N, F) np.ndarray lub torch.Tensor
    missing_rate : odsetek komórek do zamaskowania (0.0 – 1.0)
    seed         : seed RNG
    pattern      : tylko 'mcar' (Missing Completely At Random) – każda
                   komórka niezależnie z prawd. `missing_rate`. Inne
                   tryby (MAR, MNAR) są bardziej skomplikowane i nie są
                   tu zaimplementowane.

    Returns
    -------
    X_missing : kopia X z NaN w wybranych miejscach (zachowany typ wejścia)
    mask      : bool array, True = wartość zamaskowana (do ewaluacji
                jakości imputacji – RMSE tylko na zamaskowanych miejscach)
    """
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate musi być w [0, 1], dostałem {missing_rate}")
    if pattern != "mcar":
        raise NotImplementedError(
            f"Pattern '{pattern}' nie zaimplementowany. Dostępne: 'mcar'."
        )

    is_tensor = isinstance(X, torch.Tensor)
    X_np = (X.detach().cpu().numpy() if is_tensor else np.asarray(X)).astype(np.float32)

    rng = np.random.default_rng(seed)
    mask = rng.random(X_np.shape) < missing_rate

    X_missing = X_np.copy()
    X_missing[mask] = np.nan

    if is_tensor:
        # NaN zostaje w torch tensor (float)
        X_missing = torch.from_numpy(X_missing).to(dtype=X.dtype, device=X.device)

    return X_missing, mask

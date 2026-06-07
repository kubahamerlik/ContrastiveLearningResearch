"""
datasets.py – Loadery realnych datasetów pasujących do modalności frameworku.

Każda funkcja zwraca trójkę (X, y, meta):
  X    – torch.Tensor dane wejściowe (kształt zależy od modalności)
  y    – torch.LongTensor etykiety (1-D, wartości 0..num_classes-1)
  meta – dict z opisem (`num_classes`, `class_names`, `shape`, itp.)

Wszystkie loadery wspierają opcjonalny `subsample: int` – stratyfikowany
po klasach wybór `subsample` próbek (idealny do szybkich smoke testów).

Zależności: numpy, pandas, kagglehub, medmnist.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Wspólny helper: stratyfikowany subsample
# ---------------------------------------------------------------------------

def _stratified_subsample(
    y: torch.Tensor,
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Stratyfikowany wybór `n` indeksów: zachowuje proporcję klas.

    Per klasa bierze `round(n * n_c / N)` losowych indeksów (z gwarancją
    minimum 1 próbka na klasę o ile klasa w ogóle istnieje).
    """
    N = y.numel()
    if n >= N:
        return torch.arange(N, dtype=torch.long)

    classes = torch.unique(y).tolist()
    selected: List[int] = []
    for c in classes:
        idx_c = (y == c).nonzero(as_tuple=True)[0]
        n_c = idx_c.numel()
        k = max(1, round(n * n_c / N))
        k = min(k, n_c)
        perm = idx_c[torch.randperm(n_c, generator=generator)][:k]
        selected.extend(perm.tolist())

    t = torch.tensor(selected, dtype=torch.long)
    return t[torch.randperm(len(t), generator=generator)]


def _download_with_progress(url: str, dest: Path) -> None:
    """Pobiera plik z URL z prostym progress bar."""
    print(f"  [Pobieram] {url}")
    print(f"  [Zapisuj] {dest}")

    def _reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        downloaded = blocknum * blocksize
        if totalsize > 0:
            pct = min(100.0, downloaded * 100.0 / totalsize)
            mb = downloaded / 1024 / 1024
            total_mb = totalsize / 1024 / 1024
            print(f"\r    {pct:5.1f}%  ({mb:.1f} / {total_mb:.1f} MB)",
                  end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()  # newline po progress


# ===========================================================================
# 1. UCI HAR – Human Activity Recognition (szeregi czasowe)
# ===========================================================================

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

UCI_HAR_CLASS_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]

# 9 sygnałów inercyjnych z czujników smartfona
UCI_HAR_SIGNALS = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def load_uci_har(
    data_dir: str | Path = "data",
    subsample: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    UCI HAR – Human Activity Recognition Using Smartphones.

    9 czujników (acc/gyro) × 128 timesteps, 6 klas aktywności,
    ~10 299 próbek (train + test scalone, DataModule sam podzieli).

    Returns
    -------
    X    : (N, 9, 128) float32, surowe (standaryzacja w sequence_data_module)
    y    : (N,) long, wartości 0..5
    meta : dict z `in_channels`, `seq_len`, `num_classes`, `class_names`, `shape`
    """
    data_dir = Path(data_dir)
    har_root = data_dir / "uci_har"
    har_root.mkdir(parents=True, exist_ok=True)

    zip_path = har_root / "uci_har.zip"
    extracted_flag = har_root / ".extracted"
    inner_root = har_root / "UCI HAR Dataset"

    if not extracted_flag.exists():
        if not zip_path.exists():
            print("[UCI HAR] Pobieranie datasetu (~62 MB)...")
            _download_with_progress(UCI_HAR_URL, zip_path)

        print("[UCI HAR] Rozpakowywanie...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(har_root)

        # Wewnątrz zipa jest zagnieżdżony jeszcze jeden zip
        nested_zip = har_root / "UCI HAR Dataset.zip"
        if nested_zip.exists():
            with zipfile.ZipFile(nested_zip) as zf:
                zf.extractall(har_root)
            nested_zip.unlink()

        extracted_flag.touch()
        print(f"[UCI HAR] Rozpakowano do {inner_root}")

    if not inner_root.exists():
        raise FileNotFoundError(
            f"Oczekiwano katalogu {inner_root} po rozpakowaniu. "
            f"Usuń {har_root} i uruchom ponownie."
        )

    # Wczytaj sygnały dla obu splitów i skonkatenuj
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for split in ("train", "test"):
        signal_dir = inner_root / split / "Inertial Signals"
        signals: List[np.ndarray] = []
        for sig_name in UCI_HAR_SIGNALS:
            fpath = signal_dir / f"{sig_name}_{split}.txt"
            arr = np.loadtxt(fpath, dtype=np.float32)  # (N_split, 128)
            signals.append(arr)
        # Stack po osi kanałów → (N_split, 9, 128)
        X_split = np.stack(signals, axis=1)
        y_split = np.loadtxt(inner_root / split / f"y_{split}.txt",
                             dtype=np.int64) - 1  # 1..6 → 0..5
        X_parts.append(X_split)
        y_parts.append(y_split)

    X_np = np.concatenate(X_parts, axis=0)   # (N, 9, 128)
    y_np = np.concatenate(y_parts, axis=0)   # (N,)

    # Standaryzacja per-kanał jest celowo pominięta tutaj —
    # wykonywana w sequence_data_module wyłącznie ze statystyk train_idx.

    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).long()

    if subsample is not None and subsample < X.shape[0]:
        g = torch.Generator()
        g.manual_seed(seed)
        idx = _stratified_subsample(y, subsample, g)
        X, y = X[idx], y[idx]

    meta = {
        "in_channels": 9,
        "seq_len": 128,
        "num_classes": 6,
        "class_names": UCI_HAR_CLASS_NAMES,
        "shape": tuple(X.shape),
    }
    return X, y, meta


# ===========================================================================
# 2. Credit Card Fraud – Kaggle mlg-ulb/creditcardfraud (tabelaryczny)
# ===========================================================================

def load_credit_card_fraud(
    data_dir: str | Path = "data",
    subsample: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Credit Card Fraud Detection – 284 807 transakcji × 30 cech, 2 klasy.

    Cechy: Time, V1..V28 (PCA-anonymizowane), Amount.
    Klasy: 0 = normal, 1 = fraud (zaledwie ~0.17% próbek).

    Subsample jest stratyfikowany: wszystkie fraudy + losowe normal
    do osiągnięcia `subsample`.

    Returns
    -------
    X    : (N, 30) float32, surowe (standaryzacja Time/Amount w tabular_data_module)
    y    : (N,) long, wartości {0, 1}
    meta : dict z `num_features`, `num_classes`, `class_names`, `shape`,
           `fraud_rate`, `normalize_cols` (indeksy kolumn do standaryzacji)
    """
    try:
        import kagglehub
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Brakuje zależności dla Credit Card Fraud. "
            "Zainstaluj: pip install kagglehub pandas"
        ) from e

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[Credit Card] Pobieranie z Kaggle (mlg-ulb/creditcardfraud)...")
    csv_dir = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Nie znaleziono pliku CSV w {csv_dir}. "
            f"Sprawdź czy kagglehub poprawnie pobrał dataset."
        )
    csv_path = csv_files[0]
    print(f"[Credit Card] Wczytuję {csv_path}")

    df = pd.read_csv(csv_path)
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X_np = df[feature_cols].to_numpy(dtype=np.float32)
    y_np = df["Class"].to_numpy(dtype=np.int64)

    # Standaryzacja Time i Amount jest celowo pominięta tutaj —
    # wykonywana w tabular_data_module wyłącznie ze statystyk train_idx.

    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).long()

    fraud_rate = float((y == 1).sum().item() / y.numel())

    if subsample is not None and subsample < X.shape[0]:
        # Specjalny tryb: zachowaj WSZYSTKIE fraudy + losowy podzbiór normal
        g = torch.Generator()
        g.manual_seed(seed)
        fraud_idx  = (y == 1).nonzero(as_tuple=True)[0]
        normal_idx = (y == 0).nonzero(as_tuple=True)[0]

        n_fraud_keep  = min(fraud_idx.numel(), subsample)
        n_normal_keep = max(0, subsample - n_fraud_keep)
        n_normal_keep = min(n_normal_keep, normal_idx.numel())

        perm_fraud  = fraud_idx[torch.randperm(fraud_idx.numel(),  generator=g)][:n_fraud_keep]
        perm_normal = normal_idx[torch.randperm(normal_idx.numel(), generator=g)][:n_normal_keep]

        combined = torch.cat([perm_fraud, perm_normal])
        combined = combined[torch.randperm(combined.numel(), generator=g)]
        X, y = X[combined], y[combined]

    meta = {
        "num_features": 30,
        "num_classes": 2,
        "class_names": ["normal", "fraud"],
        "shape": tuple(X.shape),
        "fraud_rate": fraud_rate,
        "normalize_cols": [0, 29],  # Time i Amount — V1..V28 już znorm. przez PCA
    }
    return X, y, meta


# ===========================================================================
# 3. PathMNIST – histologia jelita grubego (obrazowy, medyczny)
# ===========================================================================

PATHMNIST_CLASS_NAMES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]


def load_pathmnist(
    data_dir: str | Path = "data",
    subsample: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    PathMNIST – obrazy histologii jelita grubego (MedMNIST).

    28×28 RGB, 9 klas tkanek, ~107 180 próbek (3 splity scalone).
    Zachowuje strukturę przestrzenną – zwraca shape (N, 3, 28, 28),
    co umożliwia użycie prawdziwego CNN (encoder_type='cnn_image')
    oraz augmentacji przestrzennych (random crop, flip, color jitter).

    Returns
    -------
    X    : (N, 3, 28, 28) float32, znormalizowane do [0, 1]
    y    : (N,) long, wartości 0..8
    meta : dict z `in_channels`, `height`, `width`, `num_classes`,
           `class_names`, `shape`
    """
    try:
        from medmnist import PathMNIST
    except ImportError as e:
        raise ImportError(
            "Brakuje zależności dla PathMNIST. "
            "Zainstaluj: pip install medmnist"
        ) from e

    data_dir = Path(data_dir)
    medmnist_root = data_dir / "medmnist"
    medmnist_root.mkdir(parents=True, exist_ok=True)

    print("[PathMNIST] Wczytuję (auto-pobieranie ~205 MB jeśli brak)...")
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for split in ("train", "val", "test"):
        ds = PathMNIST(split=split, download=True, root=str(medmnist_root))
        # ds.imgs : (N, 28, 28, 3) uint8;  ds.labels : (N, 1) int64
        X_parts.append(ds.imgs)
        y_parts.append(ds.labels.flatten())

    # Zachowaj uint8 do subsamplowania — oszczędność 4× RAM vs float32
    X_np = np.concatenate(X_parts, axis=0)   # uint8, (N, H, W, 3)
    y_np = np.concatenate(y_parts, axis=0).astype(np.int64)

    if subsample is not None and subsample < X_np.shape[0]:
        g = torch.Generator()
        g.manual_seed(seed)
        y_tmp = torch.from_numpy(y_np).long()
        idx = _stratified_subsample(y_tmp, subsample, g)
        X_np = X_np[idx.numpy()]
        y_np = y_np[idx.numpy()]

    # (N, H, W, 3) → (N, 3, H, W) – PyTorch convention; konwersja do float32 po subsamplowaniu
    X_np = np.transpose(X_np.astype(np.float32) / 255.0, (0, 3, 1, 2))

    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).long()

    meta = {
        "in_channels": 3,
        "height": 28,
        "width": 28,
        "num_classes": 9,
        "class_names": PATHMNIST_CLASS_NAMES,
        "shape": tuple(X.shape),
    }
    return X, y, meta

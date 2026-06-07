"""
data_module.py – DataModule z trójdzielnym podziałem train/val/test.

Podziały są zapisywane do JSON i odczytywane przy kolejnych uruchomieniach
— indeksy nie są nigdy generowane losowo od nowa.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


LABEL_FRACTIONS: List[float] = [1.0, 0.5, 0.25, 0.125, 0.0625]


# ---------------------------------------------------------------------------
# AugmentedPairDataset
# ---------------------------------------------------------------------------

class AugmentedPairDataset(Dataset):
    """Dataset produkujący pary augmentowanych widoków.

    Tryb symetryczny: augment_fn (np. SimCLR, MoCo).
    Tryb asymetryczny: augment_fn_v1 + augment_fn_v2 (BYOL: view1 mocny, view2 słaby).
    """

    def __init__(
        self,
        dataset: Dataset,
        augment_fn=None,
        augment_fn_v1=None,
        augment_fn_v2=None,
    ) -> None:
        self.dataset = dataset
        if augment_fn_v1 is not None and augment_fn_v2 is not None:
            # Tryb asymetryczny (BYOL)
            self._fn1 = augment_fn_v1
            self._fn2 = augment_fn_v2
        elif augment_fn is not None:
            self._fn1 = augment_fn
            self._fn2 = augment_fn
        else:
            self._fn1 = None
            self._fn2 = None

    @property
    def augment_fn(self):
        return self._fn1

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[idx]
        x = item[0] if isinstance(item, (list, tuple)) else item
        if self._fn1 is not None:
            return self._fn1(x), self._fn2(x)
        return x, x


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

@dataclass
class DataModuleConfig:
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = torch.cuda.is_available()
    seed: int = 42
    val_fraction: float = 0.15
    label_fractions: List[float] = field(
        default_factory=lambda: list(LABEL_FRACTIONS)
    )


# ---------------------------------------------------------------------------
# SplitManager – zapis i odczyt podziałów z pliku JSON
# ---------------------------------------------------------------------------

class SplitManager:
    """Zapisuje i wczytuje indeksy train/val/test per seed do pliku JSON."""

    def __init__(self, split_file: Path) -> None:
        self.split_file = Path(split_file)
        self._data: Dict[str, Dict[str, List[int]]] = self._load()

    def _load(self) -> Dict[str, Dict[str, List[int]]]:
        if self.split_file.exists():
            with open(self.split_file) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        self.split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.split_file, "w") as f:
            json.dump(self._data, f, indent=2)

    def get_or_create(
        self,
        seed: int,
        n_total: int,
        val_fraction: float,
        test_fraction: float,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Zwraca (train_idx, val_idx, test_idx). Tworzy lub wczytuje z pliku."""
        key = str(seed)

        if key in self._data:
            splits = self._data[key]
            saved_total = (len(splits["train"]) + len(splits["val"])
                           + len(splits["test"]))
            if saved_total != n_total:
                raise ValueError(
                    f"Zapisany podział dla seed={seed} ma {saved_total} próbek, "
                    f"ale dataset ma {n_total}. Usuń {self.split_file} i uruchom ponownie."
                )
            saved_meta = splits.get("_meta", {})
            if saved_meta:
                saved_val_frac  = saved_meta.get("val_fraction")
                saved_test_frac = saved_meta.get("test_fraction")
                if saved_val_frac is not None and abs(saved_val_frac - val_fraction) > 1e-6:
                    raise ValueError(
                        f"Zapisany podział dla seed={seed} używał val_fraction={saved_val_frac}, "
                        f"ale teraz przekazano val_fraction={val_fraction}. "
                        f"Usuń {self.split_file} żeby wygenerować nowy podział."
                    )
                if saved_test_frac is not None and abs(saved_test_frac - test_fraction) > 1e-6:
                    raise ValueError(
                        f"Zapisany podział dla seed={seed} używał test_fraction={saved_test_frac}, "
                        f"ale teraz przekazano test_fraction={test_fraction}. "
                        f"Usuń {self.split_file} żeby wygenerować nowy podział."
                    )
            print(f"  [SplitManager] Wczytano podział seed={seed} z {self.split_file}")
            return splits["train"], splits["val"], splits["test"]

        g = torch.Generator()
        g.manual_seed(seed)

        if labels is not None:
            train_idx, val_idx, test_idx = self._stratified_split(
                labels=labels,
                n_total=n_total,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                generator=g,
            )
            mode = "stratyfikowany"
        else:
            perm = torch.randperm(n_total, generator=g).tolist()
            n_test  = max(1, round(test_fraction * n_total))
            n_val   = max(1, round(val_fraction  * n_total))
            test_idx  = perm[:n_test]
            val_idx   = perm[n_test: n_test + n_val]
            train_idx = perm[n_test + n_val:]
            mode = "losowy"

        self._data[key] = {
            "train": train_idx,
            "val":   val_idx,
            "test":  test_idx,
            "_meta": {"val_fraction": val_fraction, "test_fraction": test_fraction},
        }
        self._save()
        print(f"  [SplitManager] Utworzono {mode} podział seed={seed}: "
              f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
              f"→ {self.split_file}")

        return train_idx, val_idx, test_idx

    @staticmethod
    def _stratified_split(
        labels: torch.Tensor,
        n_total: int,
        val_fraction: float,
        test_fraction: float,
        generator: torch.Generator,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Stratyfikowany podział per-klasa. Gwarantuje obecność każdej klasy (>=3 próbek)."""
        train_idx: List[int] = []
        val_idx:   List[int] = []
        test_idx:  List[int] = []

        unique_classes = torch.unique(labels).tolist()
        for c in unique_classes:
            class_mask = (labels == c).nonzero(as_tuple=True)[0]
            n_c = class_mask.numel()
            perm = class_mask[torch.randperm(n_c, generator=generator)].tolist()

            n_test_c = max(1, round(test_fraction * n_c)) if n_c >= 3 else 1
            n_val_c  = max(1, round(val_fraction  * n_c)) if n_c >= 3 else 1
            if n_test_c + n_val_c >= n_c:
                n_test_c = max(1, n_c // 3)
                n_val_c  = max(1, n_c // 3)

            test_idx.extend(perm[:n_test_c])
            val_idx.extend(perm[n_test_c: n_test_c + n_val_c])
            train_idx.extend(perm[n_test_c + n_val_c:])

        # Permutacja końcowa, żeby kolejność nie była pogrupowana per-klasa
        def shuffle(lst: List[int]) -> List[int]:
            t = torch.tensor(lst, dtype=torch.long)
            return t[torch.randperm(len(t), generator=generator)].tolist()

        return shuffle(train_idx), shuffle(val_idx), shuffle(test_idx)

    def get_label_fractions(
        self, seed: int
    ) -> Optional[Dict[float, List[int]]]:
        """Zwraca zapisane frakcje etykiet lub None jeśli brak."""
        key = str(seed)
        if key not in self._data:
            return None
        raw = self._data[key].get("_label_fractions")
        if raw is None:
            return None
        return {float(k): v for k, v in raw.items()}

    def save_label_fractions(
        self, seed: int, fractions: Dict[float, List[int]]
    ) -> None:
        """Zapisuje frakcje etykiet do JSON. Wymaga istniejącego podziału."""
        key = str(seed)
        if key not in self._data:
            raise RuntimeError(
                f"Nie można zapisać frakcji etykiet dla seed={seed} — "
                f"brak podziału train/val/test w {self.split_file}. "
                "Wywołaj get_or_create() przed save_label_fractions()."
            )
        self._data[key]["_label_fractions"] = {str(k): v for k, v in fractions.items()}
        self._save()


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class DataModule:
    """
    Centralny menedżer danych z trójdzielnym podziałem train/val/test.

    Loadery
    -------
    pretrain_loader()     → cały train, bez etykiet, pary augmentowane
    finetune_loader(frac) → frac * N_train próbek z etykietami
    val_loader()          → cały val  (TYLKO HPO i early stopping)
    test_loader()         → cały test (TYLKO finalna ewaluacja – raz na koniec)
    """

    def __init__(
        self,
        full_dataset: Dataset,
        config: Optional[DataModuleConfig] = None,
        split_file: str | Path = "splits.json",
        test_fraction: float = 0.15,
        augment_fn=None,
    ) -> None:
        self.cfg = config or DataModuleConfig()
        self.augment_fn = augment_fn
        self.hpo_done: bool = False

        N = len(full_dataset)  # type: ignore[arg-type]

        labels_full: Optional[torch.Tensor] = None
        if hasattr(full_dataset, "tensors") and len(full_dataset.tensors) >= 2:  # type: ignore[attr-defined]
            labels_full = full_dataset.tensors[1]  # type: ignore[attr-defined]

        manager = SplitManager(Path(split_file))
        train_idx, val_idx, test_idx = manager.get_or_create(
            seed=self.cfg.seed,
            n_total=N,
            val_fraction=self.cfg.val_fraction,
            test_fraction=test_fraction,
            labels=labels_full,
        )

        self.train_dataset: Dataset = Subset(full_dataset, train_idx)
        self.val_dataset:   Dataset = Subset(full_dataset, val_idx)
        self.test_dataset:  Dataset = Subset(full_dataset, test_idx)

        self.train_labels: Optional[torch.Tensor] = (
            labels_full[torch.tensor(train_idx, dtype=torch.long)]
            if labels_full is not None else None
        )

        # Wczytaj z JSON jeśli istnieją — gwarantuje deterministyczność między restartami.
        saved_fracs = manager.get_label_fractions(self.cfg.seed)
        if saved_fracs is not None:
            self._labeled_indices: Dict[float, List[int]] = saved_fracs
        else:
            self._labeled_indices = self._build_label_fraction_indices(len(train_idx))
            manager.save_label_fractions(self.cfg.seed, self._labeled_indices)


    def _build_label_fraction_indices(self, n_train: int) -> Dict[float, List[int]]:
        """
        Stratyfikowane, zagnieżdżone frakcje etykiet: F(6.25%) ⊂ F(12.5%) ⊂ … ⊂ F(100%).
        Jedna permutacja per klasę — każda frakcja bierze prefiks tej samej permutacji,
        co gwarantuje uczciwe porównanie krzywej "accuracy vs. fraction of labels".
        """
        g = torch.Generator()
        g.manual_seed(self.cfg.seed + 1)

        fracs_sorted = sorted(self.cfg.label_fractions)

        if self.train_labels is None:
            perm = torch.randperm(n_train, generator=g).tolist()
            return {
                frac: perm[: max(1, round(frac * n_train))]
                for frac in fracs_sorted
            }

        class_to_idx: Dict[int, List[int]] = {}
        class_to_perm: Dict[int, List[int]] = {}
        for c in torch.unique(self.train_labels).tolist():
            mask = (self.train_labels == c).nonzero(as_tuple=True)[0]
            idx_list = mask.tolist()
            class_to_idx[c] = idx_list
            perm_c = torch.randperm(len(idx_list), generator=g).tolist()
            class_to_perm[c] = perm_c

        result: Dict[float, List[int]] = {}
        for frac in fracs_sorted:
            selected: List[int] = []
            for c, idx_list in class_to_idx.items():
                n_c = len(idx_list)
                k = max(1, round(frac * n_c))
                k = min(k, n_c)
                selected.extend(idx_list[i] for i in class_to_perm[c][:k])
            t = torch.tensor(selected, dtype=torch.long)
            result[frac] = t[torch.randperm(len(t), generator=g)].tolist()
        return result

    def _make_generator(self, extra: int = 0) -> torch.Generator:
        """Izolowany Generator per-loader — shuffle deterministyczny niezależnie od stanu globalnego."""
        g = torch.Generator()
        g.manual_seed(self.cfg.seed + extra)
        return g

    def pretrain_loader(self) -> DataLoader:
        if getattr(self, "_byol_asymmetric", False):
            aug_dataset = AugmentedPairDataset(
                self.train_dataset,
                augment_fn_v1=self._byol_aug_v1,
                augment_fn_v2=self._byol_aug_v2,
            )
        else:
            aug_dataset = AugmentedPairDataset(self.train_dataset, self.augment_fn)

        effective_bs = min(self.cfg.batch_size, len(self.train_dataset))
        return DataLoader(
            aug_dataset,
            batch_size=effective_bs,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
            generator=self._make_generator(0),
        )

    def finetune_loader(self, fraction: float) -> DataLoader:
        if fraction not in self._labeled_indices:
            raise KeyError(
                f"Frakcja {fraction} niedostępna. "
                f"Dostępne: {sorted(self._labeled_indices)}"
            )
        indices = self._labeled_indices[fraction]
        return DataLoader(
            Subset(self.train_dataset, indices),
            batch_size=min(self.cfg.batch_size, len(indices)),
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            generator=self._make_generator(int(fraction * 1000)),
        )

    def val_loader(self) -> DataLoader:
        """Walidacja – tylko do HPO i early stopping. Nigdy do finalnej ewaluacji."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def test_loader(self) -> DataLoader:
        """
        Zbiór testowy – wyłącznie do finalnej ewaluacji.
        Ustaw dm.hpo_done = True gdy HPO jest zakończone.
        """
        if not self.hpo_done:
            raise RuntimeError(
                "test_loader() wywołany przed ustawieniem hpo_done=True.\n"
                "Upewnij się że HPO jest zakończone, potem: dm.hpo_done = True\n"
                "Przedwczesny dostęp do zbioru testowego to data leakage!"
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def iter_finetune_loaders(self) -> Iterator[Tuple[float, DataLoader]]:
        for frac in sorted(self.cfg.label_fractions):
            yield frac, self.finetune_loader(frac)

    @classmethod
    def from_arrays(
        cls,
        X: torch.Tensor,
        y: torch.Tensor,
        config: Optional[DataModuleConfig] = None,
        split_file: str | Path = "splits.json",
        test_fraction: float = 0.15,
        augment_fn=None,
    ) -> "DataModule":
        """
        Główny punkt wejścia. Przyjmuje surowe (X, y), sam robi podział.

        Parameters
        ----------
        X           : (N, ...) – wszystkie cechy (nie dzielone wcześniej)
        y           : (N,)     – wszystkie etykiety
        split_file  : plik JSON do zapisu/odczytu indeksów podziału
        test_fraction : odsetek N → test (domyślnie 15%)
        """
        return cls(
            TensorDataset(X, y),
            config=config,
            split_file=split_file,
            test_fraction=test_fraction,
            augment_fn=augment_fn,
        )

    def summary(self) -> str:
        n_tr = len(self.train_dataset)  # type: ignore[arg-type]
        n_v  = len(self.val_dataset)    # type: ignore[arg-type]
        n_te = len(self.test_dataset)   # type: ignore[arg-type]
        lines = [
            "=" * 54,
            "DataModule",
            "=" * 54,
            f"  Train  : {n_tr:>5} próbek",
            f"  Val    : {n_v:>5} próbek  ← HPO / early stopping",
            f"  Test   : {n_te:>5} próbek  ← tylko finalna ewaluacja",
            f"  Batch  : {self.cfg.batch_size}",
            f"  Seed   : {self.cfg.seed}",
            f"  hpo_done: {self.hpo_done}",
            "",
            "  Frakcje etykiet (finetune):",
        ]
        for frac in sorted(self.cfg.label_fractions):
            n = len(self._labeled_indices[frac])
            lines.append(f"    {frac * 100:6.2f}%  →  {n} próbek")
        lines.append("=" * 54)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Fabryki per-modalność
# ---------------------------------------------------------------------------

def tabular_data_module(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    config: Optional[DataModuleConfig] = None,
    split_file: str | Path = "splits.json",
    test_fraction: float = 0.15,
    normalize_cols: Optional[List[int]] = None,
) -> DataModule:
    """
    DataModule dla danych tabelarycznych (SCARF, SubTab).

    Modele tabelaryczne implementują własne augmentacje w `pretrain_step`
    (corruption kolumn, subsetting), więc `augment_fn` pozostaje `None` –
    `pretrain_loader` wyemituje pary identycznych wektorów, a model sam
    je zaaugmentuje per algorytm.

    Parameters
    ----------
    normalize_cols : indeksy kolumn do standaryzacji (mean=0, std=1).
        Mean/std liczone wyłącznie z train_idx (brak data leakage).
        Dla Credit Card Fraud: [0, 29] (Time i Amount; V1..V28 znorm. przez PCA).
        None = brak standaryzacji (V1..V28 przekazywane surowo).
    """
    cfg = config if config is not None else DataModuleConfig()

    if normalize_cols is not None:
        train_idx, _, _ = SplitManager(Path(split_file)).get_or_create(
            seed=cfg.seed,
            n_total=len(X),
            val_fraction=cfg.val_fraction,
            test_fraction=test_fraction,
            labels=y,
        )
        train_X = X[torch.tensor(train_idx, dtype=torch.long)]
        X = X.clone()
        for col_idx in normalize_cols:
            col_mean = train_X[:, col_idx].mean()
            col_std  = train_X[:, col_idx].std().clamp(min=1e-7)
            X[:, col_idx] = (X[:, col_idx] - col_mean) / col_std

    return DataModule.from_arrays(
        X, y,
        config=cfg,
        split_file=split_file,
        test_fraction=test_fraction,
        augment_fn=None,
    )


def _build_contrastive_image_augmentation(
    image_size: int,
    jitter_strength: float = 0.5,
    blur_kernel: Optional[int] = None,
    normalize_mean: Optional[tuple] = None,
    normalize_std: Optional[tuple] = None,
):
    """
    Kanoniczny pipeline augmentacji dla kontrastowego uczenia obrazów.

    Pochodzenie zestawu transformacji:
      - Random crop + flip + color jitter + grayscale wprowadzone w
        SimCLR (Chen et al. 2020, arXiv:2002.05709), gdzie ablation
        pokazał że to *kompozycja* augmentacji jest kluczowa.
      - Gaussian blur dodane w MoCo v2 (He et al. 2020,
        arXiv:2003.04297) — "We modify MoCo by replacing the
        augmentation pipeline with that of SimCLR (which includes
        Gaussian blur). These small changes are crucial."
      - BYOL (Grill et al. 2020) używa identycznego zestawu:
        "We use the same set of image augmentations as in SimCLR."

    Czyli ten pipeline to **standard literaturowy** dla MoCo v2 / BYOL /
    SwAV / SimSiam / DINO — nie jest specyficzny dla SimCLR-algorytmu.

    Stosowane transformacje (kompozycja):
      1. RandomResizedCrop  – losowy crop + resize do `image_size`
      2. RandomHorizontalFlip (p=0.5)
      3. ColorJitter (p=0.8)
      4. RandomGrayscale (p=0.2)
      5. GaussianBlur

    Działa na tensor (C, H, W) z wartościami w [0, 1].
    """
    try:
        import torchvision.transforms as T
    except ImportError as e:
        raise ImportError(
            "image_data_module wymaga torchvision (instalowane razem z torch). "
            "Spróbuj: pip install torchvision"
        ) from e

    if blur_kernel is None:
        # Kernel musi być nieparzysty; dla małych obrazów (28x28) wystarcza 3
        blur_kernel = 3 if image_size <= 32 else max(3, (image_size // 10) | 1)

    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(
                brightness=0.8 * jitter_strength,
                contrast=0.8   * jitter_strength,
                saturation=0.8 * jitter_strength,
                hue=0.2        * jitter_strength,
            )
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))], p=0.5),
        *(
            [T.Normalize(mean=normalize_mean, std=normalize_std)]
            if normalize_mean is not None and normalize_std is not None
            else []
        ),
    ])


def _build_byol_augmentation(
    image_size: int,
    view: int,
    jitter_strength: float = 0.5,
    blur_kernel: Optional[int] = None,
):
    """
    Asymetryczne augmentacje dla BYOL (Grill et al. 2020 Appendix B).

    view=1 (mocny): blur p=1.0, solarize p=0.2 (jasne piksele odwrócone)
    view=2 (słaby): blur p=0.1, brak solarize

    Różnica między widokami zmusza sieć online do przewidywania reprezentacji
    z bardziej zniszczonego obrazu — trudniejsze zadanie → lepsze embeddingi.
    """
    try:
        import torchvision.transforms as T
    except ImportError as e:
        raise ImportError("image_data_module wymaga torchvision") from e

    if blur_kernel is None:
        blur_kernel = 3 if image_size <= 32 else max(3, (image_size // 10) | 1)

    blur_p    = 1.0 if view == 1 else 0.1
    solarize_p = 0.2 if view == 1 else 0.0

    transforms = [
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(
                brightness=0.8 * jitter_strength,
                contrast=0.8   * jitter_strength,
                saturation=0.8 * jitter_strength,
                hue=0.2        * jitter_strength,
            )
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))], p=blur_p),
    ]
    if solarize_p > 0.0:
        transforms.append(T.RandomSolarize(threshold=0.5, p=solarize_p))

    return T.Compose(transforms)


def image_data_module(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    config: Optional[DataModuleConfig] = None,
    split_file: str | Path = "splits.json",
    test_fraction: float = 0.15,
    image_size: Optional[int] = None,
    jitter_strength: float = 0.5,
    augment_fn=None,
    byol_asymmetric: bool = False,
) -> DataModule:
    """
    DataModule dla danych obrazowych (MoCo, BYOL).

    Wymaga 4-D tensora `X` o kształcie (N, C, H, W). Każde wywołanie
    augmentacji zwraca inną losową wersję wejścia, więc
    `AugmentedPairDataset` wygeneruje dwa różne widoki tego samego obrazu.

    Normalizacja per-kanał (mean/std z trainu) jest aplikowana do całego X
    przed zbudowaniem DataModule — dzięki temu val_loader i test_loader
    zwracają znormalizowane dane spójne z pretrain_loader i finetune_loader.

    Parameters
    ----------
    image_size       : opcjonalny rozmiar po RandomResizedCrop; domyślnie = H z X.
    jitter_strength  : siła ColorJitter (0–1). 0.5 to wartość z SimCLR.
    augment_fn       : własna augmentacja; jeśli None → pełny SimCLR pipeline.
    byol_asymmetric  : True → asymetryczne pipeline BYOL (view1 blur p=1.0 + solarize,
                       view2 blur p=0.1). Używać dla BYOLModel (Grill et al. 2020 Appendix B).
    """
    if X.ndim != 4:
        raise ValueError(
            f"image_data_module oczekuje 4-D tensora (N, C, H, W); "
            f"dostałem shape {tuple(X.shape)}. "
            f"Jeśli masz spłaszczone obrazy, użyj DataModule.from_arrays() "
            f"z encoder_type='resnet_like' (deprecated)."
        )

    cfg = config if config is not None else DataModuleConfig()

    if augment_fn is None:
        H = image_size if image_size is not None else int(X.shape[-1])
        train_idx, _, _ = SplitManager(Path(split_file)).get_or_create(
            seed=cfg.seed,
            n_total=len(X),
            val_fraction=cfg.val_fraction,
            test_fraction=test_fraction,
            labels=y,
        )
        train_X = X[torch.tensor(train_idx, dtype=torch.long)]
        mean = train_X.mean(dim=(0, 2, 3))          # (C,)
        std  = train_X.std(dim=(0, 2, 3)).clamp(min=1e-7)
        X = (X - mean[None, :, None, None]) / std[None, :, None, None]

        if byol_asymmetric:
            aug_v1 = _build_byol_augmentation(H, view=1, jitter_strength=jitter_strength)
            aug_v2 = _build_byol_augmentation(H, view=2, jitter_strength=jitter_strength)
            dm = DataModule.from_arrays(
                X, y,
                config=cfg,
                split_file=split_file,
                test_fraction=test_fraction,
                augment_fn=None,
            )
            dm._byol_aug_v1 = aug_v1
            dm._byol_aug_v2 = aug_v2
            dm._byol_asymmetric = True
            return dm
        else:
            augment_fn = _build_contrastive_image_augmentation(
                H, jitter_strength,
                normalize_mean=None,
                normalize_std=None,
            )

    dm = DataModule.from_arrays(
        X, y,
        config=cfg,
        split_file=split_file,
        test_fraction=test_fraction,
        augment_fn=augment_fn,
    )
    dm._byol_asymmetric = False
    return dm


def sequence_data_module(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    config: Optional[DataModuleConfig] = None,
    split_file: str | Path = "splits.json",
    test_fraction: float = 0.15,
    normalize: bool = True,
) -> DataModule:
    """
    DataModule dla szeregów czasowych (TS-TCC).

    TS-TCC w `pretrain_step` sam wywołuje `weak_augmentation` i
    `strong_augmentation`, więc tu `augment_fn=None` – wystarczy
    zwykły loader przepuszczający surowe sekwencje (B, C, T).

    Jeśli `normalize=True` (domyślnie), wykonuje standaryzację per-kanał
    wyłącznie ze statystyk zbioru treningowego (bez data leakage).
    X musi mieć kształt (N, C, T).
    """
    cfg = config if config is not None else DataModuleConfig()

    if normalize and X.ndim == 3:
        train_idx, _, _ = SplitManager(Path(split_file)).get_or_create(
            seed=cfg.seed,
            n_total=len(X),
            val_fraction=cfg.val_fraction,
            test_fraction=test_fraction,
            labels=y,
        )
        train_X = X[torch.tensor(train_idx, dtype=torch.long)]
        ch_mean = train_X.mean(dim=(0, 2), keepdim=True)  # (1, C, 1)
        ch_std  = train_X.std(dim=(0, 2), keepdim=True).clamp(min=1e-7)
        X = (X - ch_mean) / ch_std

    return DataModule.from_arrays(
        X, y,
        config=cfg,
        split_file=split_file,
        test_fraction=test_fraction,
        augment_fn=None,
    )

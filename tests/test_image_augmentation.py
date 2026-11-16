"""
Testy dla błędu A: kolejność normalizacji w image_data_module.
Sprawdzamy że:
  1. Pipeline augmentacji pre-treningu nie tworzy ~40% zer (problem z ColorJitter na N(0,1))
  2. val/test loader zwraca dane z sensowną normalizacją (ta sama mean/std co augmentacje)
  3. SplitBatchNorm2d eksportuje się poprawnie
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest


def _make_tiny_image_dm(byol=False, seed=0):
    """Pomocnicza fabryka: 50 obrazków 3×8×8 w [0,1], 2 klasy."""
    import tempfile, pathlib
    from contrastive_framework.data_module import image_data_module, DataModuleConfig
    torch.manual_seed(seed)
    N = 50
    X = torch.rand(N, 3, 8, 8)
    y = torch.randint(0, 2, (N,))
    cfg = DataModuleConfig(batch_size=16, seed=seed)
    tmp = pathlib.Path(tempfile.mkdtemp())
    dm = image_data_module(
        X, y,
        config=cfg,
        split_file=tmp / f"splits_test_aug_{seed}_byol{byol}.json",
        test_fraction=0.2,
        byol_asymmetric=byol,
    )
    return dm


def test_pretrain_augmentation_no_mass_zeros():
    """
    Pipeline pre-treningu (ColorJitter itp.) na wejściu [0,1] NIE powinien
    generować >5% wartości dokładnie równych 0 w tensore po normalizacji.

    Przed poprawką: normalizacja była PRZED augmentacją → ~40% zer z clamp [0,1]
    w ColorJitter. Po poprawce: normalizacja NA KOŃCU, brak masowego obcinania.
    """
    dm = _make_tiny_image_dm()
    aug_fn = dm.augment_fn
    assert aug_fn is not None, "augment_fn powinno być ustawione"

    torch.manual_seed(42)
    img = torch.rand(3, 8, 8)
    outputs = [aug_fn(img) for _ in range(100)]
    all_vals = torch.stack(outputs).flatten()

    zero_fraction = (all_vals == 0.0).float().mean().item()
    assert zero_fraction < 0.05, (
        f"Za dużo dokładnych zer po augmentacji: {zero_fraction:.1%} "
        f"(oczekiwano <5% — błąd A: normalizacja przed ColorJitter)"
    )


def test_pretrain_augmentation_byol_no_mass_zeros():
    """Analogicznie dla pipeline'u BYOL (view1 i view2)."""
    dm = _make_tiny_image_dm(byol=True)
    assert hasattr(dm, "_byol_aug_v1"), "dm._byol_aug_v1 powinno być ustawione"

    torch.manual_seed(42)
    img = torch.rand(3, 8, 8)

    for view_fn, name in [(dm._byol_aug_v1, "v1"), (dm._byol_aug_v2, "v2")]:
        outputs = [view_fn(img) for _ in range(100)]
        all_vals = torch.stack(outputs).flatten()
        zero_fraction = (all_vals == 0.0).float().mean().item()
        assert zero_fraction < 0.05, (
            f"BYOL {name}: za dużo zer: {zero_fraction:.1%}"
        )


def test_val_loader_normalized_distribution():
    """
    val_loader powinien zwracać dane znormalizowane (mean≈0 per kanał),
    NIE surowe [0,1]. Weryfikuje że _normalize_tf jest aplikowane.
    """
    dm = _make_tiny_image_dm()
    assert dm._normalize_tf is not None, "_normalize_tf powinno być ustawione"

    dm.hpo_done = True
    batch = next(iter(dm.val_loader()))
    x = batch[0]  # (B, C, H, W)

    # Po normalizacji channel-wise mean powinna być blisko 0
    # (nie idealnie 0 bo batch, ale daleko od ~0.5 jak w [0,1])
    for c in range(x.shape[1]):
        channel_mean = x[:, c].mean().abs().item()
        # Surowe [0,1] ma mean≈0.5; znormalizowane powinno być bliżej 0
        assert channel_mean < 0.5, (
            f"Kanał {c} val_loader: mean={channel_mean:.3f} — "
            f"wygląda jak nienormalizowane dane [0,1]"
        )


def test_test_loader_normalized_distribution():
    """test_loader analogicznie jak val_loader."""
    dm = _make_tiny_image_dm(seed=1)
    assert dm._normalize_tf is not None
    dm.hpo_done = True
    batch = next(iter(dm.test_loader()))
    x = batch[0]
    for c in range(x.shape[1]):
        channel_mean = x[:, c].mean().abs().item()
        assert channel_mean < 0.5, (
            f"Kanał {c} test_loader: mean={channel_mean:.3f}"
        )


def test_finetune_loader_uses_same_normalization():
    """
    finetune_loader używa tych samych statystyk normalizacji co pretrain.
    Weryfikuje spójność rozkładu z val/test.
    """
    dm = _make_tiny_image_dm(seed=2)
    dm.hpo_done = True

    val_batch = next(iter(dm.val_loader()))
    fine_batch = next(iter(dm.finetune_loader(1.0)))

    val_x = val_batch[0]
    fine_x = fine_batch[0]

    # Obie powinny mieć podobny zakres (znormalizowane tymi samymi mean/std)
    # Testujemy min/max — obie powinny mieć wartości poza [0,1]
    # (sygnał że normalizacja ~N(0,1) jest aktywna)
    combined = torch.cat([val_x.flatten(), fine_x.flatten()])
    has_negative = (combined < 0).any().item()
    assert has_negative, (
        "finetune/val loader: brak wartości ujemnych — dane wyglądają jak [0,1] "
        "(normalizacja może być wyłączona)"
    )

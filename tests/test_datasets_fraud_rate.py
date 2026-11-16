"""
Testy dla punktu p5: fraud_rate w load_credit_card_fraud().

Sprawdza, że meta zawiera ZARÓWNO fraud_rate_original (przed subsamplingiem)
JAK I fraud_rate_effective (po subsamplingu).

Używamy mocka — nie pobieramy prawdziwego datasetu.
"""
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest
import torch

# Dynamiczny import datasets.py
import importlib.util
_DATASETS_PATH = Path(__file__).parent.parent / "contrastive_framework" / "datasets.py"
_spec = importlib.util.spec_from_file_location("cf_datasets", _DATASETS_PATH)
_dmod = importlib.util.module_from_spec(_spec)
sys.modules["cf_datasets"] = _dmod
_spec.loader.exec_module(_dmod)
load_credit_card_fraud = _dmod.load_credit_card_fraud


def _make_fake_csv(tmp_path, n_normal=1000, n_fraud=10):
    """Tworzy CSV imitujący creditcard.csv."""
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(0)
    rows = n_normal + n_fraud
    df = pd.DataFrame({
        "Time": rng.uniform(0, 172800, rows),
        **{f"V{i}": rng.standard_normal(rows) for i in range(1, 29)},
        "Amount": rng.uniform(0, 500, rows),
        "Class": [0] * n_normal + [1] * n_fraud,
    })
    p = tmp_path / "creditcard.csv"
    df.to_csv(p, index=False)
    return p


class TestFraudRateMeta:
    """fraud_rate_original i fraud_rate_effective muszą być w meta."""

    def test_fraud_rate_original_before_subsample(self, tmp_path):
        """fraud_rate_original = stosunek fraudów PRZED subsamplingiem."""
        csv_path = _make_fake_csv(tmp_path, n_normal=1000, n_fraud=10)
        # Mock kagglehub + glob
        with mock.patch("kagglehub.dataset_download", return_value=str(tmp_path)), \
             mock.patch("builtins.print"):
            X, y, meta = load_credit_card_fraud(
                data_dir=str(tmp_path), subsample=200, seed=0
            )
        expected_original = 10 / 1010
        assert "fraud_rate_original" in meta, "Brak fraud_rate_original w meta"
        assert abs(meta["fraud_rate_original"] - expected_original) < 1e-6, (
            f"fraud_rate_original={meta['fraud_rate_original']:.6f} != {expected_original:.6f}"
        )

    def test_fraud_rate_effective_after_subsample(self, tmp_path):
        """fraud_rate_effective = stosunek fraudów PO subsamplingu (wyższy niż original)."""
        csv_path = _make_fake_csv(tmp_path, n_normal=1000, n_fraud=10)
        with mock.patch("kagglehub.dataset_download", return_value=str(tmp_path)), \
             mock.patch("builtins.print"):
            X, y, meta = load_credit_card_fraud(
                data_dir=str(tmp_path), subsample=200, seed=0
            )
        # Po subsamplingu: 10 fraudów + ~190 normal = ~5% fraudów
        effective = float((y == 1).sum().item() / y.numel())
        assert "fraud_rate_effective" in meta, "Brak fraud_rate_effective w meta"
        assert abs(meta["fraud_rate_effective"] - effective) < 1e-6, (
            f"fraud_rate_effective={meta['fraud_rate_effective']:.6f} != obliczone {effective:.6f}"
        )
        # Skuteczna stopa jest wyższa niż oryginalna (subsample zachowuje wszystkie fraudy)
        assert meta["fraud_rate_effective"] > meta["fraud_rate_original"], (
            "Po subsamplingu fraud_rate_effective powinien być wyższy niż original"
        )

    def test_fraud_rate_original_unchanged_when_no_subsample(self, tmp_path):
        """Bez subsamplingu fraud_rate_original == fraud_rate_effective."""
        csv_path = _make_fake_csv(tmp_path, n_normal=100, n_fraud=5)
        with mock.patch("kagglehub.dataset_download", return_value=str(tmp_path)), \
             mock.patch("builtins.print"):
            X, y, meta = load_credit_card_fraud(
                data_dir=str(tmp_path), subsample=None, seed=0
            )
        assert "fraud_rate_original" in meta
        assert "fraud_rate_effective" in meta
        assert abs(meta["fraud_rate_original"] - meta["fraud_rate_effective"]) < 1e-9

    def test_fraud_rate_backward_compat(self, tmp_path):
        """meta['fraud_rate'] nadal istnieje jako alias fraud_rate_original."""
        csv_path = _make_fake_csv(tmp_path, n_normal=100, n_fraud=5)
        with mock.patch("kagglehub.dataset_download", return_value=str(tmp_path)), \
             mock.patch("builtins.print"):
            X, y, meta = load_credit_card_fraud(
                data_dir=str(tmp_path), subsample=None, seed=0
            )
        assert "fraud_rate" in meta, "meta['fraud_rate'] powinno nadal istnieć (backward compat)"

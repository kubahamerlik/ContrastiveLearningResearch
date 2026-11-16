"""
Testy dla punktów p4 i p6:
  p4: recording HPO history, timing, pretrain_loss, dataset_info, environment
  p6: num_workers > 0 działa poprawnie + worker_init_fn dla reprodukowalności
"""
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Dodaj katalog nadrzędny do ścieżki
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Pomocnicze funkcje na poziomie modułu (muszą być pickle-friendly na Windows)
# ---------------------------------------------------------------------------

def _add_noise(x: torch.Tensor) -> torch.Tensor:
    """Prosta augmentacja do testów DataLoadera z num_workers>0 — pickle-safe."""
    return x + 0.01


# ---------------------------------------------------------------------------
# p4: dataset_info.json — liczba próbek, rozkład klas
# ---------------------------------------------------------------------------

class TestDatasetInfo:
    """summarize_results.py powinien zapisać dataset_info.json z meta danych."""

    def test_dataset_info_written(self, tmp_path):
        """dataset_info.json jest zapisywany przez summarize()."""
        from summarize_results import summarize

        fracs = ["0.5", "1.0"]
        good = {f: {"accuracy": 0.8, "macro_f1": 0.75,
                    "balanced_acc": 0.78, "macro_auc": 0.9} for f in fracs}
        bl   = {f: {"accuracy": 0.7, "macro_f1": 0.65,
                    "balanced_acc": 0.68, "macro_auc": 0.8} for f in fracs}
        data = {
            "config": {
                "dataset": "credit_card",
                "seeds": [0, 1],
                "subsample": 500,
                "dataset_info": {
                    "0": {"n_train": 350, "n_val": 75, "n_test": 75,
                          "fraud_rate_effective": 0.0164},
                    "1": {"n_train": 350, "n_val": 75, "n_test": 75,
                          "fraud_rate_effective": 0.0162},
                },
            },
            "results": {
                "0": {"SSL": good, "BL": bl},
                "1": {"SSL": good, "BL": bl},
            },
        }
        p = tmp_path / "results_test.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        out = tmp_path / "out"
        summarize([p], out, pairs=[{"ssl": "SSL", "baseline": "BL"}])
        info_path = out / "dataset_info.json"
        assert info_path.exists(), "dataset_info.json nie istnieje"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        assert "0" in info or "dataset" in info, f"Nieoczekiwana struktura: {info}"

    def test_environment_json_written(self, tmp_path):
        """environment.json jest zapisywany przez summarize()."""
        from summarize_results import summarize

        fracs = ["0.5"]
        good = {"0.5": {"accuracy": 0.8, "macro_f1": 0.75,
                        "balanced_acc": 0.78, "macro_auc": 0.9}}
        data = {
            "config": {
                "dataset": "uci_har",
                "seeds": [0],
                "environment": {
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "python_version": "3.11.9",
                    "torch_version": "2.0.0",
                },
            },
            "results": {
                "0": {"SSL": good, "BL": good},
            },
        }
        p = tmp_path / "results_test.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        out = tmp_path / "out"
        summarize([p], out, pairs=[{"ssl": "SSL", "baseline": "BL"}])
        env_path = out / "environment.json"
        assert env_path.exists(), "environment.json nie istnieje"


# ---------------------------------------------------------------------------
# p4: timing.csv — zapisany przez summarize_results
# ---------------------------------------------------------------------------

class TestTimingCsv:
    """timing.csv generowany przez summarize_results.py."""

    def test_timing_csv_written_when_timing_in_results(self, tmp_path):
        from summarize_results import summarize

        good = {"0.5": {"accuracy": 0.8, "macro_f1": 0.75,
                        "balanced_acc": 0.78, "macro_auc": 0.9}}
        data = {
            "config": {
                "dataset": "uci_har",
                "seeds": [0],
                "timing": {
                    "0": {
                        "TSTCC": {"hpo_seconds": 10.5, "pretrain_seconds": 5.2,
                                  "finetune_seconds": 3.1},
                    }
                },
            },
            "results": {"0": {"TSTCC": good, "RandomInit": good}},
        }
        p = tmp_path / "results_test.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        out = tmp_path / "out"
        summarize([p], out)
        assert (out / "timing.csv").exists(), "timing.csv nie istnieje"


# ---------------------------------------------------------------------------
# p4: pretrain_loss.csv — zapisany przez summarize_results
# ---------------------------------------------------------------------------

class TestPretrainLossCsv:
    """pretrain_loss.csv generowany przez summarize_results.py."""

    def test_pretrain_loss_written_when_data_in_results(self, tmp_path):
        from summarize_results import summarize

        good = {"0.5": {"accuracy": 0.8, "macro_f1": 0.75,
                        "balanced_acc": 0.78, "macro_auc": 0.9}}
        data = {
            "config": {
                "dataset": "uci_har",
                "seeds": [0],
                "pretrain_loss": {
                    "0": {
                        "TSTCC": [0.9, 0.7, 0.5],
                    }
                },
            },
            "results": {"0": {"TSTCC": good}},
        }
        p = tmp_path / "results_test.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        out = tmp_path / "out"
        summarize([p], out)
        assert (out / "pretrain_loss.csv").exists(), "pretrain_loss.csv nie istnieje"

    def test_pretrain_loss_format(self, tmp_path):
        """pretrain_loss.csv ma kolumny: algorithm, seed, epoch, loss."""
        from summarize_results import summarize
        import csv as _csv

        good = {"0.5": {"accuracy": 0.8, "macro_f1": 0.75,
                        "balanced_acc": 0.78, "macro_auc": 0.9}}
        data = {
            "config": {
                "dataset": "uci_har",
                "seeds": [0],
                "pretrain_loss": {
                    "0": {"TSTCC": [0.9, 0.7, 0.5]},
                },
            },
            "results": {"0": {"TSTCC": good}},
        }
        p = tmp_path / "results_test.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        out = tmp_path / "out"
        summarize([p], out)
        rows = list(_csv.DictReader((out / "pretrain_loss.csv").open(encoding="utf-8")))
        assert len(rows) == 3  # 3 epoki
        assert set(rows[0].keys()) >= {"algorithm", "seed", "epoch", "loss"}


# ---------------------------------------------------------------------------
# p6: AugmentedPairDataset + num_workers > 0
# ---------------------------------------------------------------------------

class TestAugmentedPairDatasetWorkers:
    """AugmentedPairDataset działa z num_workers > 0."""

    def test_dataloader_with_one_worker(self):
        """DataLoader z num_workers=1 nie zawiesza się i zwraca dane."""
        from contrastive_framework.data_module import AugmentedPairDataset
        from torch.utils.data import TensorDataset, DataLoader

        X = torch.randn(32, 3, 8, 8)  # minimalny "obraz"
        y = torch.zeros(32, dtype=torch.long)
        base = TensorDataset(X, y)

        # _add_noise jest na poziomie modułu — pickle-safe na Windows (spawn)
        ds = AugmentedPairDataset(base, augment_fn=_add_noise)
        loader = DataLoader(ds, batch_size=8, num_workers=1,
                            persistent_workers=False)
        batch = next(iter(loader))
        assert len(batch) == 2
        v1, v2 = batch
        assert v1.shape == (8, 3, 8, 8)

    def test_worker_init_fn_accepted(self):
        """DataLoader z worker_init_fn nie rzuca błędu."""
        from contrastive_framework.data_module import AugmentedPairDataset
        from torch.utils.data import TensorDataset, DataLoader

        X = torch.randn(16, 10)
        y = torch.zeros(16, dtype=torch.long)
        base = TensorDataset(X, y)
        ds = AugmentedPairDataset(base)

        def _init(worker_id):
            import numpy as np
            np.random.seed(worker_id)

        loader = DataLoader(ds, batch_size=8, num_workers=0,
                            worker_init_fn=_init)
        # Musi działać bez rzucania wyjątku
        batch = next(iter(loader))
        assert batch[0].shape[0] == 8

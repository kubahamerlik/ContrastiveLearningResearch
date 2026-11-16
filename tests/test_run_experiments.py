"""
Testy dla run_experiments.py — punkt 6:
  RuntimeError w run_experiment (np. BYOL NaN) NIE może zatrzymać całej pętli.
  Pętla powinna zapisać informację o błędzie i kontynuować.
"""
import sys
import types
import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Dynamiczny import — run_experiments.py nie jest pakietem
# ---------------------------------------------------------------------------
import importlib.util

_RUN_EXP_PATH = Path(__file__).parent.parent / "run_experiments.py"
_spec = importlib.util.spec_from_file_location("run_experiments", _RUN_EXP_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["run_experiments"] = _mod
_spec.loader.exec_module(_mod)
run_one_seed = _mod.run_one_seed


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    defaults = dict(
        batch_size=16,
        num_workers=0,
        splits_dir="splits",
        dataset="test_ds",
        n_hpo_trials=1,
        hpo_pretrain_epochs=1,
        hpo_finetune_epochs=1,
        hpo_label_fraction=0.5,
        pretrain_epochs=1,
        finetune_epochs=1,
        run_finetune=False,
        visualize=False,
        checkpoints_dir=None,
        monitor_metrics=False,
        seeds=[0],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _stub_model_class(name="AlgoOK"):
    cls = MagicMock()
    cls.__name__ = name
    return cls


class TestByolNanHandling:
    """run_one_seed kontynuuje po RuntimeError w run_experiment lub optimize."""

    def _build_hpo_ok(self, best):
        """Fabryka optimize(), która zawsze zwraca sukces."""
        def _optimize(**kw):
            return {"best_params": best, "best_value": 0.5, "study": MagicMock()}
        return _optimize

    def _build_run_ok(self, frac_results):
        """Fabryka run_experiment(), która zawsze zwraca sukces."""
        def _run(**kw):
            return {"results": frac_results}
        return _run

    def _build_run_fail(self, msg="NaN loss — BYOL diverged"):
        """Fabryka run_experiment(), która zawsze rzuca RuntimeError."""
        def _run(**kw):
            raise RuntimeError(msg)
        return _run

    def _make_factory(self):
        dm = MagicMock()
        dm.summary.return_value = "DataModule mock"
        dm.train_dataset = []
        dm.hpo_done = False
        return lambda *a, **kw: dm

    def test_one_algo_fails_others_complete(self, tmp_path):
        """Gdy run_experiment rzuca RuntimeError dla jednego algorytmu,
        pozostałe algorytmy muszą się wykonać i wyniki muszą trafić do słownika."""
        args = _make_args(splits_dir=str(tmp_path / "splits"))

        algo_ok   = {"name": "AlgoOK",   "model_class": _stub_model_class("AlgoOK"),
                     "fixed_kwargs": {}}
        algo_fail = {"name": "AlgoFAIL", "model_class": _stub_model_class("AlgoFAIL"),
                     "fixed_kwargs": {}}
        algo_ok2  = {"name": "AlgoOK2",  "model_class": _stub_model_class("AlgoOK2"),
                     "fixed_kwargs": {}}

        ok_result  = {0.5: {"accuracy": 0.9, "macro_f1": 0.85,
                            "balanced_acc": 0.88, "macro_auc": 0.9}}
        ok_result2 = {0.5: {"accuracy": 0.8, "macro_f1": 0.75,
                            "balanced_acc": 0.78, "macro_auc": 0.8}}

        best = {"pretrain_lr": 1e-3, "finetune_lr": 1e-3,
                "hidden_dim": 64, "proj_dim": 32}

        def _run_dispatch(**kw):
            cls = kw["model_class"]
            if cls.__name__ == "AlgoFAIL":
                raise RuntimeError("NaN loss — BYOL diverged")
            if cls.__name__ == "AlgoOK2":
                return {"results": ok_result2}
            return {"results": ok_result}

        with patch.object(_mod, "optimize", side_effect=lambda **kw: {
                "best_params": best, "best_value": 0.5, "study": MagicMock()}), \
             patch.object(_mod, "run_experiment", side_effect=_run_dispatch), \
             patch.object(_mod, "filter_model_kwargs", side_effect=lambda cls, kw: kw):

            result = run_one_seed(
                seed=0,
                X=MagicMock(),
                y=MagicMock(),
                algorithms=[algo_ok, algo_fail, algo_ok2],
                factory=self._make_factory(),
                args=args,
                device=MagicMock(),
            )

        assert "AlgoOK" in result,  "AlgoOK powinien być w wynikach"
        assert "AlgoOK2" in result, "AlgoOK2 powinien być w wynikach"
        assert "AlgoFAIL" not in result or result["AlgoFAIL"] == {}, \
            "AlgoFAIL nie powinien mieć normalnych wyników"

    def test_failed_algo_info_saved(self, tmp_path):
        """Po błędzie run_one_seed zwraca informację o awarii (klucz z '_failed'
        lub podobny mechanizm), aby można było zidentyfikować problem."""
        args = _make_args(splits_dir=str(tmp_path / "splits"))

        algo_fail = {"name": "BYOL", "model_class": _stub_model_class("BYOL"),
                     "fixed_kwargs": {}}

        best = {"pretrain_lr": 1e-3, "finetune_lr": 1e-3,
                "hidden_dim": 64, "proj_dim": 32}

        with patch.object(_mod, "optimize", side_effect=lambda **kw: {
                "best_params": best, "best_value": 0.5, "study": MagicMock()}), \
             patch.object(_mod, "run_experiment",
                          side_effect=RuntimeError("NaN loss")), \
             patch.object(_mod, "filter_model_kwargs", side_effect=lambda cls, kw: kw):

            result = run_one_seed(
                seed=0,
                X=MagicMock(),
                y=MagicMock(),
                algorithms=[algo_fail],
                factory=self._make_factory(),
                args=args,
                device=MagicMock(),
            )

        # Funkcja NIE może rzucić wyjątku — to jest główny warunek.
        # Powinna zapisać info o błędzie w jakiś sposób (np. klucz "BYOL_failed").
        failed_keys = [k for k in result if "fail" in k.lower() or "error" in k.lower()]
        assert failed_keys, (
            "run_one_seed powinien zapisać informację o awarii algorytmu "
            f"(np. klucz '*_failed'), ale wynik to: {list(result.keys())}"
        )

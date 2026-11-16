"""
Testy dla run_experiments.py — punkty 6 i p1:
  Punkt 6: RuntimeError w run_experiment (np. BYOL NaN) NIE może zatrzymać całej pętli.
  Punkt p1: Wpis _failures nie może psuć aggregate() ani merge_results.
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
aggregate = _mod.aggregate

_MERGE_PATH = Path(__file__).parent.parent / "scripts" / "merge_results.py"
_mspec = importlib.util.spec_from_file_location("merge_results", _MERGE_PATH)
_mmod = importlib.util.module_from_spec(_mspec)
sys.modules["merge_results"] = _mmod
_mspec.loader.exec_module(_mmod)
_reconstruct_seed_results = _mmod._reconstruct_seed_results
_check_consistency = _mmod._check_consistency


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
        # Powinna zapisać info o błędzie w _failures (lista, nie pseudo-algorytm).
        assert "_failures" in result, (
            "run_one_seed powinien zapisać informację o awarii w result['_failures'], "
            f"ale wynik to: {list(result.keys())}"
        )
        failures = result["_failures"]
        assert len(failures) == 1, f"Oczekiwano 1 awarii, got {failures}"
        assert failures[0]["algo"] == "BYOL"
        assert "error" in failures[0]


# ---------------------------------------------------------------------------
# Punkt p1 — aggregate() i merge_results nie psuć się na _failures
# ---------------------------------------------------------------------------

class TestAggregateWithFailures:
    """aggregate() i _reconstruct_seed_results ignorują klucze zaczynające się od '_'."""

    def _make_seed_results(self, include_failure: bool = True):
        """Tworzy słownik wyników 2 seedów, opcjonalnie z wpisem _failures."""
        good = {
            0.5: {"accuracy": 0.8, "macro_f1": 0.75, "balanced_acc": 0.78, "macro_auc": 0.9},
            1.0: {"accuracy": 0.85, "macro_f1": 0.80, "balanced_acc": 0.83, "macro_auc": 0.92},
        }
        seed0 = {"AlgoA": good, "AlgoB": good}
        seed1 = {"AlgoA": good, "AlgoB": good}
        if include_failure:
            seed0["_failures"] = [{"algo": "BYOL", "error": "NaN loss", "seed": 0}]
        return {0: seed0, 1: seed1}

    def test_aggregate_no_error_with_failures(self):
        """aggregate() nie może rzucić wyjątku gdy seed_results zawiera '_failures'."""
        all_seed_results = self._make_seed_results(include_failure=True)
        result = aggregate(all_seed_results)
        assert "AlgoA" in result
        assert "AlgoB" in result

    def test_aggregate_failures_not_in_result(self):
        """aggregate() nie traktuje '_failures' jako algorytmu."""
        all_seed_results = self._make_seed_results(include_failure=True)
        result = aggregate(all_seed_results)
        for key in result:
            assert not key.startswith("_"), (
                f"aggregate() zwróciło klucz zaczynający się od '_': {key!r}. "
                "Wpisy _failures nie powinny być traktowane jako algorytmy."
            )

    def test_aggregate_without_failures_unchanged(self):
        """Wyniki bez failures są identyczne niezależnie od tego czy failures były."""
        with_fail = aggregate(self._make_seed_results(include_failure=True))
        without_fail = aggregate(self._make_seed_results(include_failure=False))
        assert set(with_fail.keys()) == set(without_fail.keys())
        for algo in with_fail:
            for frac in with_fail[algo]:
                assert abs(
                    with_fail[algo][frac]["acc_mean"] - without_fail[algo][frac]["acc_mean"]
                ) < 1e-9

    def test_reconstruct_seed_results_skips_failures(self):
        """_reconstruct_seed_results() ignoruje '_failures' przy rekonstrukcji."""
        merged = {
            "results": {
                "0": {
                    "AlgoA": {"0.5": {"accuracy": 0.8, "macro_f1": 0.75,
                                      "balanced_acc": 0.78, "macro_auc": 0.9}},
                    "_failures": [{"algo": "BYOL", "error": "NaN", "seed": 0}],
                },
                "1": {
                    "AlgoA": {"0.5": {"accuracy": 0.82, "macro_f1": 0.77,
                                      "balanced_acc": 0.80, "macro_auc": 0.91}},
                },
            }
        }
        out = _reconstruct_seed_results(merged)
        assert 0 in out
        assert "AlgoA" in out[0]
        for key in out[0]:
            assert not key.startswith("_"), (
                f"_reconstruct_seed_results zwróciło klucz '_': {key!r}"
            )

    def test_check_consistency_ignores_failures_in_algo_set(self):
        """_check_consistency() nie liczy '_failures' jako algorytm, więc nie zgłasza
        niezgodności gdy jeden plik ma failures a drugi nie."""
        import argparse as _ap
        data = [
            {
                "config": {"dataset": "test", "subsample": None, "batch_size": 32,
                           "pretrain_epochs": 10, "finetune_epochs": 5,
                           "n_hpo_trials": 5, "hpo_pretrain_epochs": 3,
                           "hpo_finetune_epochs": 2, "hpo_label_fraction": 0.5,
                           "splits_dir": "splits", "num_workers": 0,
                           "correction": "holm", "run_finetune": False},
                "results": {
                    "0": {"AlgoA": {"0.5": {"accuracy": 0.8}},
                          "_failures": [{"algo": "BYOL", "error": "NaN", "seed": 0}]},
                },
            },
            {
                "config": {"dataset": "test", "subsample": None, "batch_size": 32,
                           "pretrain_epochs": 10, "finetune_epochs": 5,
                           "n_hpo_trials": 5, "hpo_pretrain_epochs": 3,
                           "hpo_finetune_epochs": 2, "hpo_label_fraction": 0.5,
                           "splits_dir": "splits", "num_workers": 0,
                           "correction": "holm", "run_finetune": False},
                "results": {
                    "1": {"AlgoA": {"0.5": {"accuracy": 0.82}}},
                },
            },
        ]
        from pathlib import Path as _P
        files = [_P("gpu0.json"), _P("gpu1.json")]
        # Nie powinno rzucić SystemExit
        _check_consistency(files, data)

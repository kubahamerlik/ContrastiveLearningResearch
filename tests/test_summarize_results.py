"""
Testy dla scripts/summarize_results.py — punkt p3.

Sprawdzają:
1. per_seed_metrics.csv – kolumny, format długi
2. summary.csv/json – mean, std, ci95_low, ci95_high, n per algo×frac×metryka
3. paired_differences.csv/json – mean_diff, std_diff, CI, wins, n
4. LaTeX table_<metryka>.tex – podstawowa struktura
5. failures.json – lista porażek
6. Figury (fig_<metryka>.pdf/.png) są tworzone
"""
import json
import sys
from pathlib import Path

import pytest

# Dodaj scripts/ do ścieżki
_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from summarize_results import summarize


# ---------------------------------------------------------------------------
# Dane testowe — miniaturowe results.json z 2 seedami i 2 algorytmami
# ---------------------------------------------------------------------------

def _make_results(tmp_path, with_failure=False):
    """Tworzy minimalny plik results.json i zwraca jego ścieżkę."""
    fracs = ["0.0625", "0.5", "1.0"]
    good = {f: {"accuracy": 0.8 + 0.05 * i, "macro_f1": 0.75,
                "balanced_acc": 0.78, "macro_auc": 0.9}
            for i, f in enumerate(fracs)}
    baseline_good = {f: {"accuracy": 0.7 + 0.05 * i, "macro_f1": 0.65,
                          "balanced_acc": 0.68, "macro_auc": 0.8}
                     for i, f in enumerate(fracs)}

    seed0 = {"SSL": good, "BL": baseline_good}
    seed1_ssl = {f: {"accuracy": 0.82 + 0.05 * i, "macro_f1": 0.77,
                     "balanced_acc": 0.80, "macro_auc": 0.91}
                 for i, f in enumerate(fracs)}
    seed1 = {"SSL": seed1_ssl, "BL": baseline_good}

    if with_failure:
        seed0["_failures"] = [{"algo": "SSL", "error": "NaN", "seed": 0}]

    data = {
        "config": {
            "dataset": "credit_card",
            "seeds": [0, 1],
            "subsample": 500,
        },
        "results": {"0": seed0, "1": seed1},
    }
    p = tmp_path / "results_test.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


# Mapowanie par dla testu (nie używamy DATASET_CONFIG żeby test był niezależny)
_PAIRS = [{"ssl": "SSL", "baseline": "BL"}]


class TestPerSeedMetrics:
    """per_seed_metrics.csv — format długi, 4 metryki."""

    def test_csv_has_required_columns(self, tmp_path):
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        csv = out / "per_seed_metrics.csv"
        assert csv.exists(), "per_seed_metrics.csv nie istnieje"
        import csv as _csv
        rows = list(_csv.DictReader(csv.open(encoding="utf-8")))
        assert rows, "CSV jest pusty"
        expected_cols = {"dataset", "seed", "algorithm", "label_fraction",
                         "accuracy", "macro_f1", "balanced_acc", "macro_auc"}
        assert expected_cols <= set(rows[0].keys()), (
            f"Brak kolumn: {expected_cols - set(rows[0].keys())}"
        )

    def test_csv_row_count(self, tmp_path):
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        import csv as _csv
        rows = list(_csv.DictReader((out / "per_seed_metrics.csv").open(encoding="utf-8")))
        # 2 seedy × 2 algorytmy × 3 frakcje = 12 wierszy
        assert len(rows) == 12, f"Oczekiwano 12 wierszy, got {len(rows)}"


class TestSummaryCsv:
    """summary.csv — mean, std, ci95_low, ci95_high, n."""

    def test_summary_csv_exists(self, tmp_path):
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        assert (out / "summary.csv").exists()
        assert (out / "summary.json").exists()

    def test_summary_has_all_metrics(self, tmp_path):
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        import csv as _csv
        rows = list(_csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
        cols = set(rows[0].keys())
        for m in ("accuracy", "macro_f1", "balanced_acc", "macro_auc"):
            for stat in ("mean", "std", "ci95_low", "ci95_high", "n"):
                col = f"{m}_{stat}"
                assert col in cols, f"Brak kolumny {col!r} w summary.csv"

    def test_summary_mean_correct(self, tmp_path):
        """Sprawdza mean accuracy dla SSL przy frakcji 0.5."""
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        import csv as _csv
        rows = list(_csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
        ssl_05 = [r for r in rows if r["algorithm"] == "SSL" and
                  abs(float(r["label_fraction"]) - 0.5) < 1e-6]
        assert ssl_05, "Brak wiersza SSL + frac=0.5"
        # accuracy dla seeda 0 = 0.8+0.05*1=0.85, seed1=0.82+0.05*1=0.87 → mean=0.86
        expected = (0.85 + 0.87) / 2
        got = float(ssl_05[0]["accuracy_mean"])
        assert abs(got - expected) < 1e-5, f"mean={got}, expected={expected}"


class TestPairedDifferencesCsv:
    """paired_differences.csv/json."""

    def test_paired_csv_exists(self, tmp_path):
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        assert (out / "paired_differences.csv").exists()
        assert (out / "paired_differences.json").exists()

    def test_paired_csv_columns(self, tmp_path):
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        import csv as _csv
        rows = list(_csv.DictReader((out / "paired_differences.csv").open(encoding="utf-8")))
        assert rows
        cols = set(rows[0].keys())
        for col in ("ssl", "baseline", "label_fraction", "metric",
                    "mean_diff", "std_diff", "ci95_low", "ci95_high", "wins", "n"):
            assert col in cols, f"Brak kolumny {col!r}"

    def test_paired_wins_correct(self, tmp_path):
        """SSL zawsze > BL w danych testowych → wins = n = 2."""
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        import csv as _csv
        rows = list(_csv.DictReader((out / "paired_differences.csv").open(encoding="utf-8")))
        acc_rows = [r for r in rows if r["metric"] == "accuracy"]
        for r in acc_rows:
            assert int(r["wins"]) == int(r["n"]), (
                f"wins={r['wins']} != n={r['n']} dla frac={r['label_fraction']}"
            )


class TestLatexTables:
    """table_<metryka>.tex — podstawowa struktura LaTeX."""

    def test_latex_files_exist(self, tmp_path):
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        for m in ("accuracy", "macro_f1", "balanced_acc", "macro_auc"):
            assert (out / f"table_{m}.tex").exists(), f"Brak table_{m}.tex"

    def test_latex_contains_tabular(self, tmp_path):
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        content = (out / "table_accuracy.tex").read_text(encoding="utf-8")
        assert r"\begin{tabular}" in content
        assert r"\hline" in content
        assert "SSL" in content

    def test_latex_diff_tables_exist(self, tmp_path):
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        for m in ("accuracy", "macro_f1", "balanced_acc", "macro_auc"):
            assert (out / f"table_diff_{m}.tex").exists(), f"Brak table_diff_{m}.tex"


class TestFailuresJson:
    """failures.json — lista porażek z _failures."""

    def test_failures_empty_when_no_failures(self, tmp_path):
        p = _make_results(tmp_path, with_failure=False)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        f = json.loads((out / "failures.json").read_text(encoding="utf-8"))
        assert isinstance(f, list)
        assert len(f) == 0

    def test_failures_recorded(self, tmp_path):
        p = _make_results(tmp_path, with_failure=True)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        f = json.loads((out / "failures.json").read_text(encoding="utf-8"))
        assert len(f) == 1
        assert f[0]["algo"] == "SSL"


class TestFigures:
    """fig_<metryka>.pdf/.png — wykresy label-efficiency (pomijane jeśli brak matplotlib)."""

    def test_figures_created(self, tmp_path):
        pytest.importorskip("matplotlib")
        p = _make_results(tmp_path)
        out = tmp_path / "out"
        summarize([p], out, pairs=_PAIRS)
        for m in ("accuracy", "macro_f1", "balanced_acc", "macro_auc"):
            assert (out / f"fig_{m}.png").exists(), f"Brak fig_{m}.png"


class TestMultipleInputFiles:
    """summarize działa na kilku plikach wejściowych (po merge)."""

    def test_two_files_merged(self, tmp_path):
        """Dwa pliki z różnymi seedami → łączna analiza."""
        fracs = ["0.5"]
        good = {"0.5": {"accuracy": 0.8, "macro_f1": 0.75, "balanced_acc": 0.78, "macro_auc": 0.9}}
        bl   = {"0.5": {"accuracy": 0.7, "macro_f1": 0.65, "balanced_acc": 0.68, "macro_auc": 0.8}}
        f1 = tmp_path / "r1.json"
        f2 = tmp_path / "r2.json"
        f1.write_text(json.dumps({
            "config": {"dataset": "uci_har", "seeds": [0]},
            "results": {"0": {"SSL": good, "BL": bl}},
        }), encoding="utf-8")
        f2.write_text(json.dumps({
            "config": {"dataset": "uci_har", "seeds": [1]},
            "results": {"1": {"SSL": good, "BL": bl}},
        }), encoding="utf-8")
        out = tmp_path / "out"
        summarize([f1, f2], out, pairs=[{"ssl": "SSL", "baseline": "BL"}])
        import csv as _csv
        rows = list(_csv.DictReader((out / "per_seed_metrics.csv").open(encoding="utf-8")))
        # 2 seedy × 2 algos × 1 frac = 4 wiersze
        assert len(rows) == 4

"""
summarize_results.py – Eksport wyników eksperymentów do pracy inżynierskiej.

Użycie
------
    python scripts/summarize_results.py RESULTS.json [RESULTS2.json ...] \\
        --out-dir outputs/credit_card/

Wejście
-------
    Jeden lub kilka plików results_*.json (po merge z merge_results.py).
    Format: {"config": {...}, "results": {seed: {algo: {frac: metrics}}}}

Wyjście (katalog --out-dir)
---------------------------
    per_seed_metrics.csv       – format długi: dataset, seed, algorithm,
                                  label_fraction, accuracy, macro_f1,
                                  balanced_acc, macro_auc
    summary.csv / .json        – per algo × frac, dla każdej z 4 metryk:
                                  mean, std, ci95_low, ci95_high, n
    paired_differences.csv/.json – per para (ssl, baseline) × frac × metryka:
                                   mean_diff, std_diff, ci95_low, ci95_high,
                                   wins, n
    table_<metryka>.tex        – tabela LaTeX gotowa do \\input w pracy
    table_diff_<metryka>.tex   – tabela różnic sparowanych LaTeX
    fig_<metryka>.pdf/.png     – wykresy label-efficiency (oś X: frac log2,
                                  oś Y: metryka, mean + 95% CI band)
    failures.json              – lista porażek z _failures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Importy z pakietu macierzystego (jeśli wywoływany ze scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


METRICS = ["accuracy", "macro_f1", "balanced_acc", "macro_auc"]

METRIC_LABELS = {
    "accuracy":     "Accuracy",
    "macro_f1":     "Macro F1",
    "balanced_acc": "Balanced Accuracy",
    "macro_auc":    "Macro AUC-ROC",
}


# ---------------------------------------------------------------------------
# Pomocnicze funkcje statystyczne
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _ci95_half(vals: List[float]) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    from scipy.stats import t as _t
    return float(_t.ppf(0.975, df=n - 1)) * _std(vals) / math.sqrt(n)


# ---------------------------------------------------------------------------
# Wczytywanie i łączenie wielu plików results.json
# ---------------------------------------------------------------------------

def _load_all(paths: List[Path]) -> Tuple[Dict[str, Any], str]:
    """Łączy wiele plików results.json. Zwraca (all_seed_results, dataset_name)."""
    all_results: Dict[int, Dict[str, Any]] = {}
    dataset_name = "unknown"
    for p in paths:
        raw = json.loads(p.read_text(encoding="utf-8"))
        dataset_name = raw.get("config", {}).get("dataset", dataset_name)
        for s, res in raw.get("results", {}).items():
            seed = int(s)
            if seed in all_results:
                continue  # pierwszy plik wygrywa
            # Konwertuj frac-klucze str→float, pomiń _failures
            converted: Dict[str, Any] = {}
            for algo, frac_data in res.items():
                if algo.startswith("_"):
                    converted[algo] = frac_data
                else:
                    converted[algo] = {float(f): m for f, m in frac_data.items()}
            all_results[seed] = converted
    return all_results, dataset_name


# ---------------------------------------------------------------------------
# Budowanie tabel agregowanych
# ---------------------------------------------------------------------------

def _compute_summary(
    all_seed_results: Dict[int, Dict[str, Any]],
) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """
    Zwraca {algo: {frac: {metric_mean, metric_std, metric_ci95_low,
                           metric_ci95_high, metric_n}}}
    """
    raw: Dict[str, Dict[float, Dict[str, List[float]]]] = {}
    for seed_res in all_seed_results.values():
        for algo, frac_data in seed_res.items():
            if algo.startswith("_"):
                continue
            raw.setdefault(algo, {})
            for frac, metrics in frac_data.items():
                raw[algo].setdefault(frac, {m: [] for m in METRICS})
                for m in METRICS:
                    v = metrics.get(m)
                    if v is not None and not math.isnan(v):
                        raw[algo][frac][m].append(v)

    summary: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for algo, frac_map in raw.items():
        summary[algo] = {}
        for frac, metric_lists in frac_map.items():
            row: Dict[str, Any] = {}
            for m in METRICS:
                vals = metric_lists[m]
                n = len(vals)
                mean_v = _mean(vals)
                std_v  = _std(vals)
                ci_h   = _ci95_half(vals)
                row[f"{m}_mean"]     = mean_v
                row[f"{m}_std"]      = std_v
                row[f"{m}_ci95_low"]  = mean_v - ci_h if not math.isnan(ci_h) else float("nan")
                row[f"{m}_ci95_high"] = mean_v + ci_h if not math.isnan(ci_h) else float("nan")
                row[f"{m}_n"]        = n
            summary[algo][frac] = row
    return summary


def _compute_paired(
    all_seed_results: Dict[int, Dict[str, Any]],
    pairs: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Zwraca listę wierszy per para × frac × metryka."""
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        ssl_name = pair["ssl"]
        bl_name  = pair["baseline"]
        # Zbierz różnice per frac × metryka
        raw: Dict[float, Dict[str, List[float]]] = {}
        for seed_res in all_seed_results.values():
            ssl_d = seed_res.get(ssl_name)
            bl_d  = seed_res.get(bl_name)
            if ssl_d is None or bl_d is None:
                continue
            for frac in ssl_d:
                if frac not in bl_d:
                    continue
                raw.setdefault(frac, {m: [] for m in METRICS})
                for m in METRICS:
                    sv = ssl_d[frac].get(m)
                    bv = bl_d[frac].get(m)
                    if sv is not None and bv is not None:
                        raw[frac][m].append(sv - bv)

        for frac in sorted(raw):
            for m in METRICS:
                diffs = raw[frac][m]
                n = len(diffs)
                if n == 0:
                    continue
                mean_d = _mean(diffs)
                std_d  = _std(diffs)
                ci_h   = _ci95_half(diffs)
                wins   = sum(1 for d in diffs if d > 0)
                rows.append({
                    "ssl": ssl_name,
                    "baseline": bl_name,
                    "label_fraction": frac,
                    "metric": m,
                    "mean_diff":  mean_d,
                    "std_diff":   std_d,
                    "ci95_low":   mean_d - ci_h if not math.isnan(ci_h) else float("nan"),
                    "ci95_high":  mean_d + ci_h if not math.isnan(ci_h) else float("nan"),
                    "wins": wins,
                    "n": n,
                })
    return rows


# ---------------------------------------------------------------------------
# Eksport CSV / JSON
# ---------------------------------------------------------------------------

def _save_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_per_seed_csv(
    all_seed_results: Dict[int, Dict[str, Any]],
    dataset_name: str,
    out_path: Path,
) -> None:
    rows = []
    for seed, seed_res in sorted(all_seed_results.items()):
        for algo, frac_data in seed_res.items():
            if algo.startswith("_"):
                continue
            for frac in sorted(frac_data):
                m = frac_data[frac]
                rows.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "algorithm": algo,
                    "label_fraction": frac,
                    "accuracy":     m.get("accuracy", ""),
                    "macro_f1":     m.get("macro_f1", ""),
                    "balanced_acc": m.get("balanced_acc", ""),
                    "macro_auc":    m.get("macro_auc", ""),
                })
    fieldnames = ["dataset", "seed", "algorithm", "label_fraction",
                  "accuracy", "macro_f1", "balanced_acc", "macro_auc"]
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _save_atomic(out_path, buf.getvalue())


def _write_summary(
    summary: Dict[str, Dict[float, Dict[str, Any]]],
    csv_path: Path,
    json_path: Path,
) -> None:
    rows = []
    for algo in sorted(summary):
        for frac in sorted(summary[algo]):
            d = summary[algo][frac]
            row: Dict[str, Any] = {"algorithm": algo, "label_fraction": frac}
            row.update(d)
            rows.append(row)

    fieldnames = ["algorithm", "label_fraction"]
    for m in METRICS:
        for stat in ("mean", "std", "ci95_low", "ci95_high", "n"):
            fieldnames.append(f"{m}_{stat}")

    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _save_atomic(csv_path, buf.getvalue())
    _save_atomic(json_path, json.dumps(rows, indent=2, default=_json_default))


def _write_paired(
    paired_rows: List[Dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    if not paired_rows:
        _save_atomic(csv_path, "ssl,baseline,label_fraction,metric,mean_diff,std_diff,ci95_low,ci95_high,wins,n\n")
        _save_atomic(json_path, "[]")
        return
    fieldnames = ["ssl", "baseline", "label_fraction", "metric",
                  "mean_diff", "std_diff", "ci95_low", "ci95_high", "wins", "n"]
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(paired_rows)
    _save_atomic(csv_path, buf.getvalue())
    _save_atomic(json_path, json.dumps(paired_rows, indent=2, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, float):
        return obj
    raise TypeError(f"Not serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Tabele LaTeX
# ---------------------------------------------------------------------------

def _fmt_cell(mean: float, ci_half: float, decimal_comma: bool) -> str:
    """Formatuje komórkę tabeli LaTeX: mean ± ci."""
    if math.isnan(mean):
        return "—"
    ci = ci_half if not math.isnan(ci_half) else 0.0
    s = f"{mean:.3f}_{{\\pm {ci:.3f}}}"
    if decimal_comma:
        s = s.replace(".", ",")
    return f"${s}$"


def _write_latex_tables(
    summary: Dict[str, Dict[float, Dict[str, Any]]],
    out_dir: Path,
    dataset_name: str,
    decimal_comma: bool = False,
) -> None:
    algos = sorted(summary)
    if not algos:
        return
    fracs = sorted(next(iter(summary.values())).keys())

    for m in METRICS:
        label = METRIC_LABELS[m]
        frac_headers = " & ".join(f"{f*100:g}\\%" for f in fracs)
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{" + label + r" (mean $\pm$ 95\% CI) dla różnych frakcji etykiet, dataset: " + dataset_name + ".}",
            r"\begin{tabular}{l" + "c" * len(fracs) + "}",
            r"\hline",
            f"Algorytm & {frac_headers} \\\\",
            r"\hline",
        ]
        for algo in algos:
            cells = []
            for frac in fracs:
                d = summary[algo].get(frac, {})
                mean_v = d.get(f"{m}_mean", float("nan"))
                ci_lo  = d.get(f"{m}_ci95_low", float("nan"))
                ci_hi  = d.get(f"{m}_ci95_high", float("nan"))
                ci_h   = (ci_hi - ci_lo) / 2 if not math.isnan(ci_lo) and not math.isnan(ci_hi) else float("nan")
                cells.append(_fmt_cell(mean_v, ci_h, decimal_comma))
            lines.append(f"{algo} & " + " & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
        _save_atomic(out_dir / f"table_{m}.tex", "\n".join(lines))


def _write_latex_diff_tables(
    paired_rows: List[Dict[str, Any]],
    out_dir: Path,
    dataset_name: str,
    decimal_comma: bool = False,
) -> None:
    # Grupuj per metryka
    for m in METRICS:
        label = METRIC_LABELS[m]
        m_rows = [r for r in paired_rows if r["metric"] == m]
        if not m_rows:
            # Pusta tabela
            _save_atomic(out_dir / f"table_diff_{m}.tex",
                         f"% Brak danych różnic dla {m}\n")
            continue
        pairs = sorted(set((r["ssl"], r["baseline"]) for r in m_rows))
        fracs = sorted(set(r["label_fraction"] for r in m_rows))
        frac_headers = " & ".join(f"{f*100:g}\\%" for f in fracs)

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{" + label + r" — różnice sparowane SSL − baseline (mean $\pm$ 95\% CI), dataset: " + dataset_name + ".}",
            r"\begin{tabular}{l" + "c" * len(fracs) + "}",
            r"\hline",
            f"Para & {frac_headers} \\\\",
            r"\hline",
        ]
        for ssl_name, bl_name in pairs:
            cells = []
            for frac in fracs:
                match = [r for r in m_rows if r["ssl"] == ssl_name
                         and r["baseline"] == bl_name
                         and abs(r["label_fraction"] - frac) < 1e-9]
                if not match:
                    cells.append("—")
                    continue
                r = match[0]
                mean_d = r["mean_diff"]
                ci_lo  = r["ci95_low"]
                ci_hi  = r["ci95_high"]
                ci_h   = (ci_hi - ci_lo) / 2 if not math.isnan(ci_lo) and not math.isnan(ci_hi) else float("nan")
                wins   = r["wins"]
                n      = r["n"]
                s = f"{mean_d:+.3f}_{{\\pm {ci_h:.3f}}}"
                if decimal_comma:
                    s = s.replace(".", ",")
                cells.append(f"${s}$ ({wins}/{n})")
            lines.append(f"{ssl_name}−{bl_name} & " + " & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
        _save_atomic(out_dir / f"table_diff_{m}.tex", "\n".join(lines))


# ---------------------------------------------------------------------------
# Wykresy label-efficiency
# ---------------------------------------------------------------------------

def _write_figures(
    summary: Dict[str, Dict[float, Dict[str, Any]]],
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib niedostępny — pomijam wykresy.", RuntimeWarning)
        return

    algos = sorted(summary)
    if not algos:
        return
    fracs_all = sorted(next(iter(summary.values())).keys())

    for m in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for algo in algos:
            xs, ys, lows, highs = [], [], [], []
            for frac in fracs_all:
                d = summary[algo].get(frac, {})
                mean_v = d.get(f"{m}_mean", float("nan"))
                ci_lo  = d.get(f"{m}_ci95_low", float("nan"))
                ci_hi  = d.get(f"{m}_ci95_high", float("nan"))
                if math.isnan(mean_v):
                    continue
                xs.append(frac)
                ys.append(mean_v)
                lows.append(ci_lo if not math.isnan(ci_lo) else mean_v)
                highs.append(ci_hi if not math.isnan(ci_hi) else mean_v)
            if not xs:
                continue
            line, = ax.plot(xs, ys, marker="o", label=algo)
            ax.fill_between(xs, lows, highs, alpha=0.15, color=line.get_color())

        ax.set_xscale("log", base=2)
        ax.set_xticks(fracs_all)
        ax.set_xticklabels([f"{f*100:g}%" for f in fracs_all], fontsize=9)
        ax.set_xlabel("Frakcja etykiet", fontsize=10)
        ax.set_ylabel(METRIC_LABELS[m], fontsize=10)
        ax.set_title(METRIC_LABELS[m] + " — label-efficiency curve", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig.savefig(out_dir / f"fig_{m}.png", dpi=150)
        try:
            fig.savefig(out_dir / f"fig_{m}.pdf")
        except Exception:
            pass  # PDF może nie być dostępny bez odpowiedniego backendu
        plt.close(fig)


# ---------------------------------------------------------------------------
# Eksport danych dla rozdziału metodyki (p4)
# ---------------------------------------------------------------------------

def _write_dataset_info(input_paths: List[Path], out_dir: Path) -> None:
    """Zapisuje dataset_info.json z meta danych z config każdego pliku."""
    merged: Dict[str, Any] = {}
    for p in input_paths:
        raw = json.loads(p.read_text(encoding="utf-8"))
        cfg = raw.get("config", {})
        # dataset_info per seed (jeśli jest) lub ogólne meta
        di = cfg.get("dataset_info")
        if di:
            merged.update(di)
        # Dorzuć podstawowe meta z config
        for key in ("dataset", "subsample"):
            if key in cfg and "dataset" not in merged:
                merged[key] = cfg[key]
    if merged:
        _save_atomic(out_dir / "dataset_info.json",
                     json.dumps(merged, indent=2, default=_json_default))


def _write_environment(input_paths: List[Path], out_dir: Path) -> None:
    """Zapisuje environment.json z config pierwszego pliku."""
    for p in input_paths:
        raw = json.loads(p.read_text(encoding="utf-8"))
        env = raw.get("config", {}).get("environment")
        if env:
            _save_atomic(out_dir / "environment.json",
                         json.dumps(env, indent=2, default=_json_default))
            return


def _write_timing_csv(input_paths: List[Path], out_dir: Path) -> None:
    """Zapisuje timing.csv z config.timing (jeśli jest)."""
    rows = []
    for p in input_paths:
        raw = json.loads(p.read_text(encoding="utf-8"))
        timing = raw.get("config", {}).get("timing")
        if not timing:
            continue
        for seed_str, algo_map in timing.items():
            for algo, t in algo_map.items():
                row = {"seed": seed_str, "algorithm": algo}
                if isinstance(t, dict):
                    row.update(t)
                rows.append(row)
    if not rows:
        return
    # Zbierz wszystkie możliwe klucze (poza seed/algorithm)
    extra_keys: list = []
    seen_keys: set = set()
    for r in rows:
        for k in r:
            if k not in ("seed", "algorithm") and k not in seen_keys:
                extra_keys.append(k)
                seen_keys.add(k)
    fieldnames = ["seed", "algorithm"] + extra_keys
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore",
                            restval="")
    writer.writeheader()
    writer.writerows(rows)
    _save_atomic(out_dir / "timing.csv", buf.getvalue())


def _write_pretrain_loss_csv(input_paths: List[Path], out_dir: Path) -> None:
    """Zapisuje pretrain_loss.csv z config.pretrain_loss (jeśli jest)."""
    rows = []
    for p in input_paths:
        raw = json.loads(p.read_text(encoding="utf-8"))
        pl = raw.get("config", {}).get("pretrain_loss")
        if not pl:
            continue
        for seed_str, algo_map in pl.items():
            for algo, losses in algo_map.items():
                if isinstance(losses, (list, tuple)):
                    for epoch, loss in enumerate(losses, start=1):
                        rows.append({
                            "algorithm": algo,
                            "seed": seed_str,
                            "epoch": epoch,
                            "loss": loss,
                        })
    if not rows:
        return
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["algorithm", "seed", "epoch", "loss"])
    writer.writeheader()
    writer.writerows(rows)
    _save_atomic(out_dir / "pretrain_loss.csv", buf.getvalue())


# ---------------------------------------------------------------------------
# Główna funkcja (wywoływana z testów i z CLI)
# ---------------------------------------------------------------------------

def summarize(
    input_paths: List[Path],
    out_dir: Path,
    pairs: Optional[List[Dict[str, str]]] = None,
    decimal_comma: bool = False,
    dataset_name_override: Optional[str] = None,
) -> None:
    """Wczytuje wyniki i zapisuje wszystkie pliki do out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_seed_results, dataset_name = _load_all(input_paths)
    if dataset_name_override:
        dataset_name = dataset_name_override

    if not all_seed_results:
        warnings.warn("Brak danych wyników — out_dir będzie pusty.", RuntimeWarning)
        return

    # 1. per_seed_metrics.csv
    _write_per_seed_csv(all_seed_results, dataset_name,
                        out_dir / "per_seed_metrics.csv")

    # 2. summary.csv / .json
    summary = _compute_summary(all_seed_results)
    _write_summary(summary, out_dir / "summary.csv", out_dir / "summary.json")

    # 3. paired_differences.csv / .json
    if pairs is None:
        # Spróbuj załadować pary z DATASET_CONFIG
        try:
            from run_experiments import DATASET_CONFIG
            if dataset_name in DATASET_CONFIG:
                pairs = [
                    {"ssl": a["name"], "baseline": a["baseline"]}
                    for a in DATASET_CONFIG[dataset_name]["algorithms"]
                    if "baseline" in a
                ]
            else:
                pairs = []
        except ImportError:
            pairs = []
    paired_rows = _compute_paired(all_seed_results, pairs)
    _write_paired(paired_rows, out_dir / "paired_differences.csv",
                  out_dir / "paired_differences.json")

    # 4. Tabele LaTeX
    _write_latex_tables(summary, out_dir, dataset_name, decimal_comma)
    _write_latex_diff_tables(paired_rows, out_dir, dataset_name, decimal_comma)

    # 5. Wykresy
    _write_figures(summary, out_dir)

    # 6. failures.json
    all_failures = [
        f
        for seed_res in all_seed_results.values()
        for f in seed_res.get("_failures", [])
    ]
    _save_atomic(out_dir / "failures.json",
                 json.dumps(all_failures, indent=2, default=_json_default))

    # 7. dataset_info.json (z config)
    _write_dataset_info(input_paths, out_dir)

    # 8. environment.json (z config)
    _write_environment(input_paths, out_dir)

    # 9. timing.csv
    _write_timing_csv(input_paths, out_dir)

    # 10. pretrain_loss.csv
    _write_pretrain_loss_csv(input_paths, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="+", type=Path,
                   help="Pliki wyników (results_*.json) do analizy")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Katalog wyjściowy (domyślnie: outputs/<dataset>/)")
    p.add_argument("--decimal-comma", action="store_true", default=False,
                   help="Użyj przecinka dziesiętnego w tabelach LaTeX (polskie raporty)")
    args = p.parse_args()

    for f in args.files:
        if not f.exists():
            sys.exit(f"ERROR: plik nie istnieje: {f}")

    # Ustal katalog wyjściowy
    if args.out_dir is None:
        # Spróbuj odczytać dataset z pierwszego pliku
        try:
            raw = json.loads(args.files[0].read_text(encoding="utf-8"))
            ds = raw.get("config", {}).get("dataset", "unknown")
        except Exception:
            ds = "unknown"
        out_dir = Path("outputs") / ds
    else:
        out_dir = args.out_dir

    print(f"Zapisuję wyniki do: {out_dir}")
    summarize(args.files, out_dir, decimal_comma=args.decimal_comma)

    print(f"\nWygenerowane pliki:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

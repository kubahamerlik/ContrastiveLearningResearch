"""
merge_results.py – Scala kilka plików wyników (z różnych procesów/kart GPU)
w jeden plik, a następnie odtwarza tabele i statystyki.

Użycie
------
    python scripts/merge_results.py results_gpu0.json results_gpu1.json \\
        --out results_pathmnist.json

Kontrola spójności (exit 1 jeśli wykryto problem)
---------------------------------------------------
- Ta sama konfiguracja CLI (poza --seeds, --device, --results-file, --resume,
  --prepare-only) we wszystkich plikach wejściowych.
- Brak zdublowanych seedów między plikami.
- Te same algorytmy we wszystkich plikach.

Po scaleniu wypisuje tabele Accuracy, Macro F1 (i Balanced Acc / AUC-ROC
jeśli dostępne) oraz blok LaTeX — tak jak run_experiments.py po pełnym
uruchomieniu.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Pozwala importować pakiet z katalogu nadrzędnego
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Klucze konfiguracyjne ignorowane przy porównaniu
# (mogą się różnić między procesami bez naruszenia spójności wyników)
# ---------------------------------------------------------------------------
_IGNORE_CONFIG_KEYS: Set[str] = {
    "seeds",
    "device",
    "results_file",
    "resume",
    "prepare_only",
}


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"ERROR: nie można wczytać {path}: {e}")


def _config_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in _IGNORE_CONFIG_KEYS}


def _check_consistency(files: List[Path], data: List[Dict[str, Any]]) -> None:
    """Weryfikuje spójność konfiguracji i brak zdublowanych seedów."""
    errors: List[str] = []

    # --- Spójna konfiguracja ---
    ref_cfg = _config_fingerprint(data[0].get("config", {}))
    ref_name = files[0].name
    for path, d in zip(files[1:], data[1:]):
        cfg = _config_fingerprint(d.get("config", {}))
        if cfg != ref_cfg:
            diffs = []
            all_keys = set(ref_cfg) | set(cfg)
            for k in sorted(all_keys):
                v_ref = ref_cfg.get(k, "<brak>")
                v_cur = cfg.get(k, "<brak>")
                if v_ref != v_cur:
                    diffs.append(f"  {k}: {ref_name}={v_ref!r}  vs  {path.name}={v_cur!r}")
            errors.append(
                f"Niezgodna konfiguracja między {ref_name} a {path.name}:\n"
                + "\n".join(diffs)
            )

    # --- Brak zdublowanych seedów ---
    all_seeds: List[str] = []
    seed_sources: Dict[str, str] = {}
    for path, d in zip(files, data):
        for s in d.get("results", {}).keys():
            if s in seed_sources:
                errors.append(
                    f"Zdublowany seed {s}: obecny w {seed_sources[s]} i {path.name}"
                )
            else:
                seed_sources[s] = path.name
                all_seeds.append(s)

    # --- Te same algorytmy ---
    def _algos(d: Dict[str, Any]) -> Optional[Set[str]]:
        for seed_data in d.get("results", {}).values():
            return {k for k in seed_data.keys() if not k.startswith("_")}
        return None

    ref_algos = _algos(data[0])
    if ref_algos is not None:
        for path, d in zip(files[1:], data[1:]):
            algos = _algos(d)
            if algos is not None and algos != ref_algos:
                errors.append(
                    f"Różne algorytmy: {files[0].name}={sorted(ref_algos)}  "
                    f"vs  {path.name}={sorted(algos)}"
                )

    if errors:
        print("\nERROR: scalanie anulowane z powodu niezgodności:\n")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


def _merge(data: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    """Scala wszystkie pliki w jeden słownik."""
    merged_results: Dict[str, Any] = {}
    for d in data:
        for s, res in d.get("results", {}).items():
            merged_results[s] = res

    # Użyj konfiguracji z pierwszego pliku, zaktualizuj seeds i results_file
    base_cfg = dict(data[0].get("config", {}))
    all_seeds = sorted(int(s) for s in merged_results)
    base_cfg["seeds"] = all_seeds
    base_cfg["results_file"] = str(args.out)
    base_cfg["device"] = "merged"

    return {"config": base_cfg, "results": merged_results}


def _save_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _reconstruct_seed_results(merged: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Odtwarza słownik {seed: {algo: {frac: metrics}}} dla agregacji."""
    out: Dict[int, Dict[str, Any]] = {}
    for s, res in merged.get("results", {}).items():
        seed = int(s)
        out[seed] = {}
        for algo, frac_data in res.items():
            if algo.startswith("_"):
                continue
            out[seed][algo] = {float(f): m for f, m in frac_data.items()}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path,
                   help="Pliki wyników do scalenia (np. results_gpu0.json results_gpu1.json)")
    p.add_argument("--out", type=Path, default=None,
                   help="Plik wyjściowy (domyślnie: results_<dataset>.json z konfiguracji)")
    p.add_argument("--no-tables", action="store_true", default=False,
                   help="Pomiń wypisywanie tabel (tylko scal i zapisz)")
    p.add_argument("--wilcoxon", action="store_true", default=False,
                   help="Wykonaj test Wilcoxona vs baseline (tak jak --wilcoxon w run_experiments.py)")
    p.add_argument("--correction", choices=["holm", "bh"], default="holm")
    args = p.parse_args()

    files = [Path(f) for f in args.files]
    for f in files:
        if not f.exists():
            sys.exit(f"ERROR: plik nie istnieje: {f}")

    data = [_load(f) for f in files]

    print(f"Sprawdzam spójność {len(files)} pliku/plików...")
    _check_consistency(files, data)
    print("  ✓ Konfiguracja spójna, brak zdublowanych seedów, te same algorytmy.")

    merged = _merge(data, args)
    total_seeds = len(merged["results"])
    print(f"  ✓ Scalono {total_seeds} seedów: {sorted(int(s) for s in merged['results'])}")

    # Ustal plik wyjściowy
    if args.out is None:
        dataset = merged["config"].get("dataset", "unknown")
        out_path = Path(f"results_{dataset}.json")
    else:
        out_path = args.out

    _save_atomic(out_path, merged)
    print(f"  ✓ Zapisano → {out_path}")

    if args.no_tables:
        return

    # Odtwórz tabele i statystyki
    try:
        from run_experiments import (
            aggregate, wilcoxon_vs_baseline, print_results_table,
            paired_differences, print_paired_diff_table, DATASET_CONFIG,
        )
    except ImportError:
        print("\nUWAGA: nie można zaimportować run_experiments — pomijam tabele.")
        return

    all_seed_results = _reconstruct_seed_results(merged)
    if not all_seed_results:
        print("Brak danych do agregacji.")
        return

    aggregated = aggregate(all_seed_results)
    correction = args.correction
    empty_sig: Dict[str, Any] = {}

    if getattr(args, "wilcoxon", False):
        significance = wilcoxon_vs_baseline(aggregated, baseline_name="RandomInit",
                                            correction=correction)
        sig_bac = wilcoxon_vs_baseline(aggregated, baseline_name="RandomInit",
                                       metric_key="bac_values", correction=correction)
        sig_auc = wilcoxon_vs_baseline(aggregated, baseline_name="RandomInit",
                                       metric_key="auc_values", correction=correction)
    else:
        significance = empty_sig
        sig_bac = empty_sig
        sig_auc = empty_sig

    print_results_table(aggregated, significance, metric="acc")
    print()
    print_results_table(aggregated, significance, metric="f1")

    has_bac = any(
        "bac_mean" in fd
        for ad in aggregated.values()
        for fd in ad.values()
    )
    has_auc = any(
        "auc_mean" in fd
        for ad in aggregated.values()
        for fd in ad.values()
    )
    if has_bac:
        print()
        print_results_table(aggregated, sig_bac, metric="bac")
    if has_auc:
        print()
        print_results_table(aggregated, sig_auc, metric="auc")

    # Różnice sparowane z konfiguracji datasetu
    dataset_name = merged.get("config", {}).get("dataset")
    if dataset_name and dataset_name in DATASET_CONFIG:
        pairs = [
            {"ssl": a["name"], "baseline": a["baseline"]}
            for a in DATASET_CONFIG[dataset_name]["algorithms"]
            if "baseline" in a
        ]
        if pairs:
            print_paired_diff_table(all_seed_results, pairs)


if __name__ == "__main__":
    main()

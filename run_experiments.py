"""
run_experiments.py – Kompletna pętla badawcza dla realnych datasetów.

Wybór datasetu przez `--dataset`:
  uci_har      → szeregi czasowe (TS-TCC vs RandomInit)
  credit_card  → tabelaryczny     (SCARF, SubTab vs RandomInit)
  pathmnist    → obrazowy         (MoCo, BYOL  vs RandomInit)

Schemat działania
-----------------
1. Wczytaj dataset (loader z contrastive_framework.datasets)
2. Dla każdego seedu:
   a. DataModule (per-modalność)        → splits/{dataset}_seed_{seed}.json
   b. HPO dla każdego algorytmu          → best params na VAL
   c. Finalny trening z best_params      → ewaluacja na TEST
3. Agregacja (mean ± std po seedach)
4. Test Wilcoxona vs baseline (RandomInit)
5. Tabela LaTeX

Przykłady uruchomienia
----------------------
    # Pełne (długie!)
    python run_experiments.py --dataset uci_har
    python run_experiments.py --dataset credit_card --subsample 30000
    python run_experiments.py --dataset pathmnist   --subsample 20000

    # Szybki smoke test
    python run_experiments.py --dataset uci_har --seeds 0 \\
        --subsample 1000 --n-hpo-trials 2 \\
        --pretrain-epochs 3 --finetune-epochs 3

Zależności
----------
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
from scipy.stats import wilcoxon, t as student_t

from contrastive_framework import (
    DataModule, DataModuleConfig,
    tabular_data_module, image_data_module, sequence_data_module,
    SCARFModel, SubTabModel, RandomInitSubTabArchModel,
    MoCoModel, BYOLModel,
    TSTCCModel,
    RandomInitModel,
    run_experiment,
)
from contrastive_framework.models.timeseries_models import RandomInitHARModel
from contrastive_framework.hpo import optimize
from contrastive_framework.experiment import filter_model_kwargs, finetune_end_to_end
from contrastive_framework.datasets import (
    load_uci_har, load_credit_card_fraud, load_pathmnist,
)


LABEL_FRACTIONS = [0.0625, 0.125, 0.25, 0.5, 1.0]

METRICS = ["accuracy", "macro_f1", "balanced_acc", "macro_auc"]


# ===========================================================================
# DATASET_CONFIG – per modalność: loader + fabryka DataModule + algorytmy
# ===========================================================================

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "uci_har": {
        "loader": load_uci_har,
        "factory": sequence_data_module,
        "algorithms": [
            {
                "name": "RandomInit",
                "model_class": RandomInitHARModel,
                "fixed_kwargs_from_meta": lambda m: {
                    "seq_len":     m["seq_len"],
                    "in_channels": m["in_channels"],
                    "num_classes": m["num_classes"],
                },
            },
            {
                "name": "TSTCC",
                "model_class": TSTCCModel,
                "baseline": "RandomInit",
                "fixed_kwargs_from_meta": lambda m: {
                    "seq_len":     m["seq_len"],
                    "in_channels": m["in_channels"],
                    "num_classes": m["num_classes"],
                },
            },
        ],
    },
    "credit_card": {
        "loader": load_credit_card_fraud,
        "factory": tabular_data_module,
        "factory_kwargs": {"normalize_cols": [0, 29]},
        "algorithms": [
            {
                "name": "RandomInit",
                "model_class": RandomInitModel,
                "fixed_kwargs_from_meta": lambda m: {
                    "encoder_type": "mlp",
                    "input_dim":    m["num_features"],
                    "num_classes":  m["num_classes"],
                },
            },
            {
                "name": "SCARF",
                "model_class": SCARFModel,
                "baseline": "RandomInit",
                "fixed_kwargs_from_meta": lambda m: {
                    "num_features": m["num_features"],
                    "num_classes":  m["num_classes"],
                },
            },
            {
                "name": "RandomInit-SubTabArch",
                "model_class": RandomInitSubTabArchModel,
                "fixed_kwargs_from_meta": lambda m: {
                    "num_features": m["num_features"],
                    "num_classes":  m["num_classes"],
                },
            },
            {
                "name": "SubTab",
                "model_class": SubTabModel,
                "baseline": "RandomInit-SubTabArch",
                "fixed_kwargs_from_meta": lambda m: {
                    "num_features": m["num_features"],
                    "num_classes":  m["num_classes"],
                },
            },
        ],
    },
    "pathmnist": {
        "loader": load_pathmnist,
        "factory": image_data_module,
        "algorithms": [
            {
                "name": "RandomInit",
                "model_class": RandomInitModel,
                "fixed_kwargs_from_meta": lambda m: {
                    "encoder_type": "cnn_image",
                    "input_dim":    m["in_channels"],
                    "num_classes":  m["num_classes"],
                },
            },
            {
                "name": "MoCo",
                "model_class": MoCoModel,
                "baseline": "RandomInit",
                "fixed_kwargs_from_meta": lambda m: {
                    "encoder_type": "cnn_image",
                    "input_dim":    m["in_channels"],
                    "num_classes":  m["num_classes"],
                },
            },
            {
                "name": "BYOL",
                "model_class": BYOLModel,
                "baseline": "RandomInit",
                "fixed_kwargs_from_meta": lambda m: {
                    "encoder_type": "cnn_image",
                    "input_dim":    m["in_channels"],
                    "num_classes":  m["num_classes"],
                },
                "factory_kwargs_override": {"byol_asymmetric": True},
            },
        ],
    },
}


# ===========================================================================
# 1. Pętla po seedach
# ===========================================================================

def _maybe_visualize(
    name: str,
    model_class: Type,
    model_kwargs: Dict[str, Any],
    algo_dm: Any,
    device: torch.device,
    pretrain_epochs: int,
    pretrain_lr: float,
    args: argparse.Namespace,
    meta: Dict[str, Any],
) -> None:
    """Trenuje dwa modele (losowy init + po pre-treningu) i generuje figurę t-SNE/UMAP."""
    try:
        from contrastive_framework.visualization import plot_embeddings
        from contrastive_framework.experiment import pretrain as _pretrain
        import torch.optim as optim
    except ImportError as e:
        print(f"  [Viz] Pomijam wizualizację ({e})")
        return

    print(f"\n  [Viz] Generuję embedding space dla {name}...")

    model_before = model_class(**model_kwargs).to(device)
    model_before.eval()

    viz_epochs = min(pretrain_epochs, 30)
    model_after = model_class(**model_kwargs).to(device)
    opt = optim.Adam(model_after.parameters(), lr=pretrain_lr, weight_decay=1e-4)
    loader = algo_dm.pretrain_loader()
    _pretrain(model_after, loader, opt, device, epochs=viz_epochs, verbose=False)

    viz_loader = algo_dm.finetune_loader(fraction=1.0)

    class_names = meta.get("class_names") or None
    fig_dir = Path(args.viz_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_path = fig_dir / f"embeddings_{args.dataset}_{name}.png"

    try:
        plot_embeddings(
            model_before=model_before,
            model_after=model_after,
            loader=viz_loader,
            device=device,
            method=getattr(args, "viz_method", "tsne"),
            title=f"{name} — {args.dataset}",
            save_path=save_path,
            class_names=class_names,
        )
        print(f"  [Viz] Zapisano → {save_path}")
    except Exception as exc:
        print(f"  [Viz] Błąd wizualizacji dla {name}: {exc}")


def _seed_everything(seed: int) -> None:
    """Ustawia globalne seedy torch/numpy/random dla reprodukowalności per-seed.

    Wywoływane na początku każdego seeda — gwarantuje że wynik nie zależy od
    kolejności seedów ani od tego, na której karcie/w którym procesie był liczony.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_seed(
    seed: int,
    X: torch.Tensor,
    y: torch.Tensor,
    algorithms: List[Dict[str, Any]],
    factory: Callable,
    args: argparse.Namespace,
    device: torch.device,
    factory_kwargs: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pełna procedura dla jednego seedu: HPO (val) → finalny trening → test."""
    _seed_everything(seed)

    print(f"\n{'#'*64}")
    print(f"  SEED {seed}  (device: {device})")
    print(f"{'#'*64}")

    cfg = DataModuleConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        label_fractions=LABEL_FRACTIONS,
    )
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(exist_ok=True, parents=True)
    split_file = splits_dir / f"{args.dataset}_seed_{seed}.json"

    dm = factory(X, y, config=cfg, split_file=split_file, **(factory_kwargs or {}))
    print(dm.summary())

    seed_results: Dict[str, Any] = {}

    for algo in algorithms:
        name        = algo["name"]
        model_class = algo["model_class"]
        fixed_kw    = dict(algo["fixed_kwargs"])

        # SubTab: różne seedy używają różnych losowych permutacji kolumn
        if name == "SubTab":
            fixed_kw["subset_seed"] = seed

        print(f"\n─── Algorytm: {name} ───")

        dm.hpo_done = False

        algo_factory_override = algo.get("factory_kwargs_override")
        if algo_factory_override:
            algo_dm = factory(
                X, y, config=cfg, split_file=split_file,
                **(factory_kwargs or {}), **algo_factory_override,
            )
            algo_dm.hpo_done = False
        else:
            algo_dm = dm

        try:
            if name == "RandomInit":
                # RandomInit przechodzi HPO (finetune_lr, hidden_dim, proj_dim) —
                # sprawiedliwe porównanie z modelami SSL. Pretrain to no-op.
                hpo_result = optimize(
                    model_class=model_class,
                    data_module=algo_dm,
                    fixed_model_kwargs=fixed_kw,
                    n_trials=args.n_hpo_trials,
                    pretrain_epochs=1,
                    finetune_epochs=args.hpo_finetune_epochs,
                    finetune_fraction=getattr(args, "hpo_label_fraction", 0.5),
                    device=device,
                    show_progress=True,
                )
                best_params = hpo_result["best_params"]
                algo_dm.hpo_done = False
            else:
                hpo_result = optimize(
                    model_class=model_class,
                    data_module=algo_dm,
                    fixed_model_kwargs=fixed_kw,
                    n_trials=args.n_hpo_trials,
                    pretrain_epochs=args.hpo_pretrain_epochs,
                    finetune_epochs=args.hpo_finetune_epochs,
                    finetune_fraction=getattr(args, "hpo_label_fraction", 0.5),
                    device=device,
                    show_progress=True,
                )
                best_params = hpo_result["best_params"]
                algo_dm.hpo_done = False

            model_kwargs = filter_model_kwargs(model_class, {
                **fixed_kw,
                "hidden_dim": best_params.get("hidden_dim", 128),
                "proj_dim":   best_params.get("proj_dim",   64),
                **{k: v for k, v in best_params.items()
                   if k not in ("pretrain_lr", "finetune_lr",
                                "hidden_dim", "proj_dim")},
            })

            result = run_experiment(
                model_class=model_class,
                model_kwargs=model_kwargs,
                data_module=algo_dm,
                device=device,
                pretrain_epochs=args.pretrain_epochs,
                finetune_epochs=args.finetune_epochs,
                pretrain_lr=best_params.get("pretrain_lr", 3e-4),
                finetune_lr=best_params.get("finetune_lr", 1e-3),
                verbose=True,
                run_finetune=getattr(args, "run_finetune", False),
                checkpoint_dir=getattr(args, "checkpoints_dir", None),
                checkpoint_tag=f"{args.dataset}_seed{seed}",
                monitor_metrics=getattr(args, "monitor_metrics", False),
            )

            seed_results[name] = result["results"]
            if result.get("results_finetune"):
                seed_results[f"{name}_e2e"] = result["results_finetune"]

        except Exception as _exc:
            # Błąd (np. NaN/Inf w BYOL) nie może zatrzymać całego seeda.
            # Zapisujemy informację diagnostyczną na liście _failures — NIE jako
            # pseudo-algorytm, bo klucz str psuje aggregate().
            seed_results.setdefault("_failures", []).append(
                {"error": str(_exc), "algo": name, "seed": seed}
            )
            print(f"\n  [WARN] Algorytm {name!r} (seed={seed}) zakończył się błędem: "
                  f"{type(_exc).__name__}: {_exc}")
            print(f"  [WARN] Awaria zapisana w _failures. Kontynuuję...")

        if getattr(args, "visualize", False) and seed == args.seeds[0] \
                and name not in seed_results:
            _maybe_visualize(
                name=name,
                model_class=model_class,
                model_kwargs=model_kwargs,
                algo_dm=algo_dm,
                device=device,
                pretrain_epochs=args.pretrain_epochs,
                pretrain_lr=best_params.get("pretrain_lr", 3e-4),
                args=args,
                meta=meta or {},
            )

    return seed_results


# ===========================================================================
# 2. Agregacja
# ===========================================================================

def aggregate(
    all_seed_results: Dict[int, Dict[str, Any]],
) -> Dict[str, Dict[float, Dict[str, float]]]:
    raw: Dict[str, Dict[float, Dict[str, List[float]]]] = {}
    for seed_res in all_seed_results.values():
        for algo, frac_results in seed_res.items():
            if algo.startswith("_"):
                continue
            raw.setdefault(algo, {})
            for frac, metrics in frac_results.items():
                raw[algo].setdefault(frac, {
                    "accuracy": [], "macro_f1": [], "balanced_acc": [], "macro_auc": [],
                })
                raw[algo][frac]["accuracy"].append(metrics.get("accuracy", float("nan")))
                raw[algo][frac]["macro_f1"].append(metrics.get("macro_f1", float("nan")))
                raw[algo][frac]["balanced_acc"].append(metrics.get("balanced_acc", float("nan")))
                raw[algo][frac]["macro_auc"].append(metrics.get("macro_auc", float("nan")))

    aggregated: Dict[str, Dict[float, Dict[str, float]]] = {}
    for algo, frac_data in raw.items():
        aggregated[algo] = {}
        for frac, metric_lists in frac_data.items():
            accs  = metric_lists["accuracy"]
            f1s   = metric_lists["macro_f1"]
            bacs  = metric_lists.get("balanced_acc", [])
            aucs  = metric_lists.get("macro_auc", [])
            acc_ci  = _ci95_half(accs)
            f1_ci   = _ci95_half(f1s)
            aggregated[algo][frac] = {
                "acc_mean": sum(accs) / len(accs),
                "acc_std":  _std(accs),
                "acc_ci":   acc_ci,
                "f1_mean":  sum(f1s) / len(f1s),
                "f1_std":   _std(f1s),
                "f1_ci":    f1_ci,
                "acc_values": accs,
                "f1_values":  f1s,
            }
            if bacs:
                bac_ci = _ci95_half(bacs)
                aggregated[algo][frac]["bac_mean"] = sum(bacs) / len(bacs)
                aggregated[algo][frac]["bac_std"]  = _std(bacs)
                aggregated[algo][frac]["bac_ci"]   = bac_ci
                aggregated[algo][frac]["bac_values"] = bacs
            if aucs:
                valid_aucs = [a for a in aucs if a == a]
                if valid_aucs:
                    auc_ci = _ci95_half(valid_aucs)
                    aggregated[algo][frac]["auc_mean"] = sum(valid_aucs) / len(valid_aucs)
                    aggregated[algo][frac]["auc_std"]  = _std(valid_aucs)
                    aggregated[algo][frac]["auc_ci"]   = auc_ci
                    aggregated[algo][frac]["auc_values"] = valid_aucs
    return aggregated


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _ci95_half(values: List[float]) -> float:
    """Połowa 95% CI (t-Studenta) dla listy wartości."""
    import math
    n = len(values)
    if n < 2:
        return float("nan")
    std = _std(values)
    t_crit = float(student_t.ppf(0.975, df=n - 1))
    return t_crit * std / math.sqrt(n)


# ===========================================================================
# 3a. Różnice sparowane SSL − baseline
# ===========================================================================

def paired_differences(
    all_seed_results: Dict[int, Dict[str, Any]],
    ssl_name: str,
    baseline_name: str,
) -> Dict[float, Dict[str, Any]]:
    """
    Liczy różnice sparowane SSL − baseline per frakcja × metryka.

    Zwraca:
        {frac: {metric: {mean_diff, std_diff, ci95_half, ci95_low, ci95_high,
                          wins, n, diffs}}}
    Seedy bez wyniku SSL lub baseline są wykluczone z par (n zmniejszone).
    """
    import math

    # Zbierz pary per (frac, metric)
    raw: Dict[float, Dict[str, List[float]]] = {}
    for seed_res in all_seed_results.values():
        ssl_data = seed_res.get(ssl_name)
        bl_data  = seed_res.get(baseline_name)
        if ssl_data is None or bl_data is None:
            continue
        for frac in ssl_data:
            if frac not in bl_data:
                continue
            raw.setdefault(frac, {m: [] for m in METRICS})
            for m in METRICS:
                s_val = ssl_data[frac].get(m)
                b_val = bl_data[frac].get(m)
                if s_val is not None and b_val is not None:
                    raw[frac][m].append(s_val - b_val)

    result: Dict[float, Dict[str, Any]] = {}
    for frac, metric_diffs in raw.items():
        result[frac] = {}
        for m, diffs in metric_diffs.items():
            n = len(diffs)
            if n == 0:
                result[frac][m] = {
                    "mean_diff": float("nan"), "std_diff": float("nan"),
                    "ci95_half": float("nan"), "ci95_low": float("nan"),
                    "ci95_high": float("nan"), "wins": 0, "n": 0, "diffs": [],
                }
                continue
            mean_d = sum(diffs) / n
            std_d  = _std(diffs) if n >= 2 else float("nan")
            ci_h   = _ci95_half(diffs)  # nan when n<2
            wins   = sum(1 for d in diffs if d > 0)
            result[frac][m] = {
                "mean_diff": mean_d,
                "std_diff":  std_d,
                "ci95_half": ci_h,
                "ci95_low":  mean_d - ci_h if not math.isnan(ci_h) else float("nan"),
                "ci95_high": mean_d + ci_h if not math.isnan(ci_h) else float("nan"),
                "wins": wins,
                "n":    n,
                "diffs": diffs,
            }
    return result


def print_paired_diff_table(
    all_seed_results: Dict[int, Dict[str, Any]],
    pairs: List[Dict[str, str]],
) -> None:
    """Wypisuje tabelę różnic sparowanych dla każdej pary (ssl, baseline)."""
    import math

    for pair in pairs:
        ssl_name = pair["ssl"]
        bl_name  = pair["baseline"]
        diffs = paired_differences(all_seed_results, ssl_name, bl_name)
        if not diffs:
            continue
        fracs = sorted(diffs.keys())
        print(f"\n── Różnice sparowane: {ssl_name} − {bl_name} ──")
        for m in METRICS:
            header = f"  {m:<14}"
            for frac in fracs:
                d = diffs[frac][m]
                if d["n"] == 0:
                    header += f"  {'–':>14}"
                elif math.isnan(d["ci95_half"]):
                    header += f"  {d['mean_diff']:+.3f}(n={d['n']},CI=NaN)".ljust(16)
                else:
                    header += f"  {d['mean_diff']:+.3f}[±{d['ci95_half']:.3f}] {d['wins']}/{d['n']}".ljust(18)
            print(header)


# ===========================================================================
# 3b. Test Wilcoxona vs baseline (opcjonalny — flaga --wilcoxon)
# ===========================================================================

def wilcoxon_vs_baseline(  # noqa: C901
    aggregated: Dict[str, Dict[float, Dict[str, float]]],
    baseline_name: str = "RandomInit",
    alpha: float = 0.05,
    metric_key: str = "f1_values",
    correction: str = "holm",
) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """
    Test Wilcoxona na wybranej metryce (domyślnie macro_f1) względem baseline.

    Stosuje korekcję Holm-Bonferroni lub Benjamini-Hochberg dla wielokrotnych
    porównań (kontrola FWER lub FDR odpowiednio).

    metric_key: klucz listy wartości per seed:
        "f1_values"  — macro F1 (domyślnie)
        "bac_values" — balanced accuracy
        "auc_values" — macro AUC-ROC
        "acc_values" — accuracy
    """
    baseline = aggregated.get(baseline_name)
    if baseline is None:
        return {}

    significance: Dict[str, Dict[float, Dict[str, Any]]] = {}
    warned_seeds = False

    raw_p: List[Any] = []

    for algo, frac_data in aggregated.items():
        if algo == baseline_name:
            continue
        significance[algo] = {}
        for frac in frac_data:
            algo_vals     = frac_data[frac].get(metric_key, [])
            baseline_vals = baseline[frac].get(metric_key, [])
            if not algo_vals or not baseline_vals:
                raw_p.append((algo, frac, float("nan"), float("nan"), False))
                continue
            n = len(algo_vals)
            if n < 5:
                if not warned_seeds:
                    print(
                        f"\n  UWAGA: za mało seedów ({n}) do testu Wilcoxona"
                        f" — potrzeba min. 5. Gwiazdki istotności pominięte."
                    )
                    warned_seeds = True
                raw_p.append((algo, frac, float("nan"), float("nan"), False))
                continue

            diff = [a - b for a, b in zip(algo_vals, baseline_vals)]
            if all(d == 0 for d in diff):
                raw_p.append((algo, frac, 1.0, 1.0, True))
            else:
                _, p_one = wilcoxon(diff, alternative="greater")
                _, p_two = wilcoxon(diff, alternative="two-sided")
                raw_p.append((algo, frac, float(p_one), float(p_two), True))

    valid_tests = [(i, entry) for i, entry in enumerate(raw_p) if entry[4]]
    n_tests = len(valid_tests)

    holm_significant: Dict[int, bool] = {}
    holm_p_corrected: Dict[int, float] = {}

    if n_tests > 0:
        sorted_valid = sorted(valid_tests, key=lambda t: t[1][2])
        if correction == "bh":
            # Benjamini-Hochberg: kontroluje FDR (False Discovery Rate)
            adj: List[float] = []
            for rank, (_, entry) in enumerate(sorted_valid, start=1):
                p_one = entry[2]
                adj.append(min(1.0, p_one * n_tests / rank))
            for k in range(len(adj) - 2, -1, -1):
                adj[k] = min(adj[k], adj[k + 1])
            k_max = -1
            for k, q in enumerate(adj):
                if q <= alpha:
                    k_max = k
            for rank, (orig_idx, _) in enumerate(sorted_valid):
                holm_significant[orig_idx] = (rank <= k_max)
                holm_p_corrected[orig_idx] = adj[rank]
        else:
            # Holm-Bonferroni: kontroluje FWER
            rejected = True
            running_max = 0.0
            for rank, (orig_idx, entry) in enumerate(sorted_valid, start=1):
                p_one = entry[2]
                holm_threshold = alpha / (n_tests - rank + 1)
                if rejected and p_one <= holm_threshold:
                    holm_significant[orig_idx] = True
                else:
                    rejected = False
                    holm_significant[orig_idx] = False
                q = min(1.0, p_one * (n_tests - rank + 1))
                running_max = max(running_max, q)
                holm_p_corrected[orig_idx] = running_max

    for i, (algo, frac, p_one, p_two, valid) in enumerate(raw_p):
        if not valid:
            significance[algo][frac] = {
                "significant": False,
                "p_one": p_one,
                "p_two": p_two,
                "p_one_corrected": float("nan"),
            }
        else:
            significance[algo][frac] = {
                "significant": holm_significant.get(i, False),
                "p_one": p_one,
                "p_two": p_two,
                "p_one_corrected": holm_p_corrected.get(i, float("nan")),
            }

    return significance


# ===========================================================================
# 4. Tabela wyników
# ===========================================================================

def print_results_table(
    aggregated: Dict[str, Dict[float, Dict[str, float]]],
    significance: Dict[str, Dict[float, bool]],
    metric: str = "acc",
) -> None:
    fracs  = sorted(next(iter(aggregated.values())).keys())
    algos  = list(aggregated.keys())
    mean_k = f"{metric}_mean"
    std_k  = f"{metric}_std"

    col_w = 16
    header = f"{'Algorytm':<14}" + "".join(
        f"{f*100:.1f}%".center(col_w) for f in fracs
    )
    metric_names = {
        "acc": "Accuracy",
        "f1":  "Macro F1",
        "bac": "Balanced Accuracy",
        "auc": "Macro AUC-ROC",
    }
    metric_name = metric_names.get(metric, metric)
    print("\n" + "=" * len(header))
    print(f"  {metric_name + ' (mean ± std)':^{len(header)-2}}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    ci_k = f"{metric}_ci"
    for algo in algos:
        row = f"{algo:<14}"
        for frac in fracs:
            d      = aggregated[algo][frac]
            sig    = significance.get(algo, {}).get(frac, {})
            is_sig = sig.get("significant", False) if isinstance(sig, dict) else bool(sig)
            star   = "*" if is_sig else " "
            ci_val = d.get(ci_k)
            if ci_val is not None:
                cell = f"{d[mean_k]:.3f}[±{ci_val:.3f}]{star}"
            else:
                cell = f"{d[mean_k]:.3f}±{d[std_k]:.3f}{star}"
            row   += cell.center(col_w)
        print(row)

    print("=" * len(header))
    has_wilcoxon = any(significance.get(a, {}) for a in algos)
    metric_label = {
        "acc": "accuracy", "f1": "macro_f1",
        "bac": "balanced_acc", "auc": "macro_auc",
    }.get(metric, metric)
    if has_wilcoxon:
        print(f"  * = istotnie lepszy od RandomInit (Wilcoxon+Holm na {metric_label}, p < 0.05)")
    print(f"  [±X.XXX] = 95% CI (t-Studenta)\n")

    frac_headers = " & ".join(
        f"{f*100:g}\\%" for f in fracs
    )
    print("% ── LaTeX ──────────────────────────────────────────────────")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    if has_wilcoxon:
        print(r"\caption{" + metric_name + r" (mean $\pm$ 95\% CI) dla różnych frakcji etykiet. "
              r"$^{*}$ = istotnie lepszy od RandomInit (Wilcoxon jednostronny, $p < 0.05$).}")
    else:
        print(r"\caption{" + metric_name + r" (mean $\pm$ 95\% CI) dla różnych frakcji etykiet.}")
    print(r"\begin{tabular}{l" + "c" * len(fracs) + "}")
    print(r"\hline")
    print(f"Algorytm & {frac_headers} \\\\")
    print(r"\hline")
    for algo in algos:
        cells = []
        for frac in fracs:
            d      = aggregated[algo][frac]
            sig    = significance.get(algo, {}).get(frac, {})
            is_sig = sig.get("significant", False) if isinstance(sig, dict) else bool(sig)
            star   = "^{*}" if is_sig else ""
            cells.append(f"${d[mean_k]:.3f}_{{\\pm {d[std_k]:.3f}}}{star}$")
        print(f"{algo} & " + " & ".join(cells) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("% ─────────────────────────────────────────────────────────")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=list(DATASET_CONFIG.keys()))
    p.add_argument("--data-dir", default="data",
                   help="Katalog z pobranymi datasetami (domyślnie: data/)")
    p.add_argument("--subsample", type=int, default=None,
                   help="Stratyfikowany subsample N próbek (None = pełny dataset)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="Lista seedów (domyślnie: 0..9; liczba seedów zapisywana w config.seeds pliku wyników)")
    p.add_argument("--n-hpo-trials", type=int, default=20)
    p.add_argument("--pretrain-epochs",      type=int, default=100)
    p.add_argument("--finetune-epochs",      type=int, default=50)
    p.add_argument("--hpo-pretrain-epochs",  type=int, default=20)
    p.add_argument("--hpo-finetune-epochs",  type=int, default=15)
    p.add_argument("--hpo-label-fraction", type=float, default=0.5,
                   help="Frakcja etykiet używana podczas HPO (domyślnie: 0.5).")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0,
                   help="Liczba workerów DataLoader (domyślnie: 0; zalecane 2-4 na GPU/Colab)")
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--results-file", default=None,
                   help="Plik z surowymi wynikami (domyślnie: results_{dataset}.json)")
    p.add_argument("--visualize", action="store_true", default=False,
                   help="Generuj t-SNE/UMAP wizualizacje przestrzeni embeddingów")
    p.add_argument("--viz-method", default="tsne", choices=["tsne", "umap"],
                   help="Metoda redukcji wymiarowości dla wizualizacji (domyślnie: tsne)")
    p.add_argument("--viz-dir", default="figures",
                   help="Katalog zapisu figur (domyślnie: figures/)")
    p.add_argument("--run-finetune", action="store_true", default=False,
                   help="Dodatkowo przeprowadź fine-tuning end-to-end obok linear probe")
    p.add_argument("--wilcoxon", action="store_true", default=False,
                   help="Wykonaj test Wilcoxona vs baseline i dodaj gwiazdki istotności "
                        "do tabel (wymaga min. 5 seedów). Domyślnie wyłączone — "
                        "raportujemy różnice sparowane dla wszystkich metryk.")
    p.add_argument("--correction", choices=["holm", "bh"], default="holm",
                   help="Korekcja wielokrotnych porównań (używana tylko z --wilcoxon): "
                        "holm (FWER) lub bh (FDR).")
    p.add_argument("--checkpoints-dir", default=None,
                   help="Katalog do zapisu wytrenowanych modeli (state_dict).")
    p.add_argument("--monitor-metrics", action="store_true", default=False,
                   help="Loguj alignment i uniformity (Wang & Isola 2020) co 10 epok.")
    p.add_argument("--algorithms", nargs="+", default=None, metavar="NAME",
                   help="Uruchom tylko podane algorytmy (np. --algorithms MoCo BYOL). "
                        "Domyślnie: wszystkie algorytmy dla danego datasetu.")
    p.add_argument("--device", default=None,
                   help="Urządzenie obliczeniowe: cuda, cuda:0, cuda:1, cpu. "
                        "Domyślnie: cuda jeśli dostępna, inaczej cpu. "
                        "Alternatywnie ustaw CUDA_VISIBLE_DEVICES=0 i pomiń tę flagę.")
    p.add_argument("--prepare-only", action="store_true", default=False,
                   help="Pobierz dane i wygeneruj splity dla --seeds bez trenowania. "
                        "Uruchom raz przed dwoma równoległymi procesami na Kaggle.")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Pomiń seedy już obecne w --results-file (wznawianie po przerwaniu).")
    p.add_argument("--out-dir", default=None,
                   help="Katalog eksportu (summarize_results.py); "
                        "domyślnie outputs/<dataset>/")
    p.add_argument("--decimal-comma", action="store_true", default=False,
                   help="Użyj przecinka dziesiętnego w tabelach LaTeX.")
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================

def _resolve_device(device_str: Optional[str]) -> torch.device:
    """Parsuje flagę --device. Fallback: cuda jeśli dostępna, inaczej cpu."""
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_existing_results(results_file: Path) -> Dict[str, Any]:
    """Wczytuje istniejący plik wyników lub zwraca pusty słownik."""
    if results_file.exists():
        try:
            return json.loads(results_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_results_atomic(results_file: Path, data: Dict[str, Any]) -> None:
    """Atomowy zapis pliku wyników (tmp + os.replace)."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=results_file.parent, suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, results_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _merge_seed_result(
    existing: Dict[str, Any],
    seed: int,
    seed_res: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Scala wynik jednego seedu z istniejącym słownikiem wyników."""
    existing_results = existing.get("results", {})
    s_key = str(seed)
    existing_results.setdefault(s_key, {})
    for algo, frac_data in seed_res.items():
        if algo.startswith("_"):
            # _failures is a list — persist as-is, don't convert through frac dict
            existing_results[s_key][algo] = frac_data
        else:
            existing_results[s_key][algo] = {str(f): m for f, m in frac_data.items()}
    return {"config": vars(args), "results": existing_results}


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    print(f"Urządzenie: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cfg = DATASET_CONFIG[args.dataset]
    print(f"\n┌── Dataset: {args.dataset} ──┐")

    X_full, y_full, meta = cfg["loader"](
        data_dir=args.data_dir,
        subsample=None,
    )
    subsample_info = f" → subsample={args.subsample}/seed" if args.subsample else ""
    print(f"  shape (pełny): {meta['shape']}{subsample_info}")
    print(f"  klas  : {meta['num_classes']} ({meta.get('class_names', [])})")
    if "fraud_rate" in meta:
        print(f"  fraud_rate: {meta['fraud_rate']:.4%}")

    algorithms = [
        {**algo, "fixed_kwargs": algo["fixed_kwargs_from_meta"](meta)}
        for algo in cfg["algorithms"]
    ]
    if args.algorithms is not None:
        unknown = set(args.algorithms) - {a["name"] for a in algorithms}
        if unknown:
            raise ValueError(
                f"Nieznane algorytmy: {sorted(unknown)}. "
                f"Dostępne dla {args.dataset}: {[a['name'] for a in algorithms]}"
            )
        algorithms = [a for a in algorithms if a["name"] in args.algorithms]
    print(f"  algorytmy: {[a['name'] for a in algorithms]}")

    results_file = Path(args.results_file or f"results_{args.dataset}.json")

    # --prepare-only: pobierz dane + wygeneruj splity, nie trenuj
    if args.prepare_only:
        print("\n[--prepare-only] Generuję splity dla wszystkich seedów...")
        for seed in args.seeds:
            if args.subsample is not None and args.subsample < X_full.shape[0]:
                from contrastive_framework.datasets import _stratified_subsample
                g = torch.Generator()
                g.manual_seed(seed)
                idx = _stratified_subsample(y_full, args.subsample, g)
                X_seed, y_seed = X_full[idx], y_full[idx]
            else:
                X_seed, y_seed = X_full, y_full

            from contrastive_framework.data_module import DataModuleConfig
            splits_dir = Path(args.splits_dir)
            splits_dir.mkdir(exist_ok=True, parents=True)
            split_file = splits_dir / f"{args.dataset}_seed_{seed}.json"
            cfg_dm = DataModuleConfig(
                batch_size=args.batch_size, seed=seed, label_fractions=LABEL_FRACTIONS,
            )
            # Każda factory generuje split i zapisuje do JSON
            factory_kw = dict(cfg.get("factory_kwargs") or {})
            dm = cfg["factory"](X_seed, y_seed, config=cfg_dm, split_file=split_file,
                                **factory_kw)
            print(f"  seed={seed}: split zapisany → {split_file}")
        print("[--prepare-only] Gotowe. Uruchom teraz procesy równoległe.")
        return

    # Wczytaj istniejące wyniki (do --resume i do scalania)
    existing_data = _load_existing_results(results_file)

    seeds_to_run = list(args.seeds)
    if args.resume:
        already_done = set(existing_data.get("results", {}).keys())
        seeds_to_run = [s for s in seeds_to_run if str(s) not in already_done]
        skipped = set(str(s) for s in args.seeds) - set(str(s) for s in seeds_to_run)
        if skipped:
            print(f"\n[--resume] Pomijam seedy już w {results_file}: {sorted(skipped)}")

    all_seed_results: Dict[int, Dict[str, Any]] = {}
    for seed in seeds_to_run:
        if args.subsample is not None and args.subsample < X_full.shape[0]:
            from contrastive_framework.datasets import _stratified_subsample
            g = torch.Generator()
            g.manual_seed(seed)
            idx = _stratified_subsample(y_full, args.subsample, g)
            X_seed, y_seed = X_full[idx], y_full[idx]
        else:
            X_seed, y_seed = X_full, y_full

        seed_res = run_one_seed(seed, X_seed, y_seed, algorithms, cfg["factory"],
                                args, device,
                                factory_kwargs=cfg.get("factory_kwargs"),
                                meta=meta)
        all_seed_results[seed] = seed_res

        # Zapis po każdym seedzie — checkpoint pod limit 12 h
        existing_data = _merge_seed_result(existing_data, seed, seed_res, args)
        _save_results_atomic(results_file, existing_data)
        print(f"  [checkpoint] seed={seed} zapisany → {results_file}")

    if not seeds_to_run:
        print("\nBrak seedów do uruchomienia.")
        all_seed_results = {
            int(s): {
                algo: (frac_data if algo.startswith("_") else {float(f): m for f, m in frac_data.items()})
                for algo, frac_data in res.items()
            }
            for s, res in existing_data.get("results", {}).items()
        }

    print(f"\nSurowe wyniki zapisane/scalone → {results_file}")

    all_failures = [
        f
        for seed_res in all_seed_results.values()
        for f in seed_res.get("_failures", [])
    ]
    if all_failures:
        print(f"\n[AWARIE] {len(all_failures)} algorytm(ów) zakończyło się błędem:")
        for f in all_failures:
            print(f"  seed={f['seed']}  algo={f['algo']}  error={f['error']}")

    aggregated = aggregate(all_seed_results)

    # Opcjonalny test Wilcoxona (flaga --wilcoxon)
    empty_sig: Dict[str, Any] = {}
    if getattr(args, "wilcoxon", False):
        significance = wilcoxon_vs_baseline(
            aggregated, baseline_name="RandomInit", correction=args.correction,
        )
        sig_bac = wilcoxon_vs_baseline(aggregated, baseline_name="RandomInit",
                                       metric_key="bac_values",
                                       correction=args.correction)
        sig_auc = wilcoxon_vs_baseline(aggregated, baseline_name="RandomInit",
                                       metric_key="auc_values",
                                       correction=args.correction)
    else:
        significance = empty_sig
        sig_bac = empty_sig
        sig_auc = empty_sig

    print_results_table(aggregated, significance, metric="acc")
    print()
    print_results_table(aggregated, significance, metric="f1")

    has_bac = any(
        "bac_mean" in frac_data
        for algo_data in aggregated.values()
        for frac_data in algo_data.values()
    )
    has_auc = any(
        "auc_mean" in frac_data
        for algo_data in aggregated.values()
        for frac_data in algo_data.values()
    )
    if has_bac:
        print()
        print_results_table(aggregated, sig_bac, metric="bac")
    if has_auc:
        print()
        print_results_table(aggregated, sig_auc, metric="auc")

    # Różnice sparowane dla każdej pary SSL–baseline z DATASET_CONFIG
    pairs = [
        {"ssl": a["name"], "baseline": a["baseline"]}
        for a in cfg["algorithms"]
        if "baseline" in a
    ]
    if pairs:
        print_paired_diff_table(all_seed_results, pairs)

    # Eksport do outputs/<dataset>/
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else Path("outputs") / args.dataset
    try:
        from scripts.summarize_results import summarize as _summarize
    except ImportError:
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(
                "summarize_results",
                Path(__file__).parent / "scripts" / "summarize_results.py",
            )
            _smod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_smod)
            _summarize = _smod.summarize
        except Exception as _e:
            print(f"\n[summarize] Nie można zaimportować summarize_results.py: {_e}")
            _summarize = None

    if _summarize is not None:
        _summarize(
            [results_file],
            out_dir,
            pairs=pairs,
            decimal_comma=getattr(args, "decimal_comma", False),
        )
        print(f"\n[summarize] Pliki eksportu → {out_dir}/")


if __name__ == "__main__":
    main()

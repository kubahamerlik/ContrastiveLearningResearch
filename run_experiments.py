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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
from scipy.stats import wilcoxon, t as student_t

from contrastive_framework import (
    DataModule, DataModuleConfig,
    tabular_data_module, image_data_module, sequence_data_module,
    SCARFModel, SubTabModel,
    MoCoModel, BYOLModel,
    TSTCCModel,
    RandomInitModel,
    run_experiment,
)
from contrastive_framework.hpo import optimize
from contrastive_framework.experiment import filter_model_kwargs, finetune_end_to_end
from contrastive_framework.datasets import (
    load_uci_har, load_credit_card_fraud, load_pathmnist,
)


LABEL_FRACTIONS = [0.0625, 0.125, 0.25, 0.5, 1.0]


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
                "model_class": RandomInitModel,
                "fixed_kwargs_from_meta": lambda m: {
                    "encoder_type": "cnn",
                    "input_dim":    m["in_channels"],
                    "num_classes":  m["num_classes"],
                },
            },
            {
                "name": "TSTCC",
                "model_class": TSTCCModel,
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
                "fixed_kwargs_from_meta": lambda m: {
                    "num_features": m["num_features"],
                    "num_classes":  m["num_classes"],
                },
            },
            {
                "name": "SubTab",
                "model_class": SubTabModel,
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
                "fixed_kwargs_from_meta": lambda m: {
                    "encoder_type": "cnn_image",
                    "input_dim":    m["in_channels"],
                    "num_classes":  m["num_classes"],
                },
            },
            {
                "name": "BYOL",
                "model_class": BYOLModel,
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
    print(f"\n{'#'*64}")
    print(f"  SEED {seed}")
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

        if getattr(args, "visualize", False) and seed == args.seeds[0]:
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
        return 0.0
    std = _std(values)
    t_crit = float(student_t.ppf(0.975, df=n - 1))
    return t_crit * std / math.sqrt(n)


# ===========================================================================
# 3. Test Wilcoxona vs baseline
# ===========================================================================

def wilcoxon_vs_baseline(
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
    metric_label = {
        "acc": "accuracy", "f1": "macro_f1",
        "bac": "balanced_acc", "auc": "macro_auc",
    }.get(metric, metric)
    print(f"  * = istotnie lepszy od RandomInit (Wilcoxon+Holm na {metric_label}, p < 0.05)")
    print(f"  [±X.XXX] = 95% CI (t-Studenta)\n")

    frac_headers = " & ".join(
        f"{f*100:g}\\%" for f in fracs
    )
    print("% ── LaTeX ──────────────────────────────────────────────────")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{" + metric_name + r" (mean $\pm$ std) dla różnych frakcji etykiet. "
          r"$^{*}$ = istotnie lepszy od RandomInit (Wilcoxon jednostronny, $p < 0.05$).}")
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
                   help="Lista seedów (domyślnie: 0..9; min. 10 dla mocy Wilcoxona)")
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
    p.add_argument("--correction", choices=["holm", "bh"], default="holm",
                   help="Korekcja wielokrotnych porównań: holm (FWER) lub bh (FDR).")
    p.add_argument("--checkpoints-dir", default=None,
                   help="Katalog do zapisu wytrenowanych modeli (state_dict).")
    p.add_argument("--monitor-metrics", action="store_true", default=False,
                   help="Loguj alignment i uniformity (Wang & Isola 2020) co 10 epok.")
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    if torch.cuda.is_available():
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
    print(f"  algorytmy: {[a['name'] for a in algorithms]}")

    all_seed_results: Dict[int, Dict[str, Any]] = {}
    for seed in args.seeds:
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

    results_file = Path(args.results_file or f"results_{args.dataset}.json")
    results_file.write_text(
        json.dumps(
            {
                "config": vars(args),
                "results": {
                    str(s): {
                        algo: {str(f): m
                               for f, m in frac_data.items()}
                        for algo, frac_data in res.items()
                    } for s, res in all_seed_results.items()
                },
            },
            indent=2,
        )
    )
    print(f"\nSurowe wyniki zapisane → {results_file}")

    aggregated   = aggregate(all_seed_results)
    significance = wilcoxon_vs_baseline(
        aggregated, baseline_name="RandomInit", correction=args.correction,
    )

    print_results_table(aggregated, significance, metric="acc")
    print()
    print_results_table(aggregated, significance, metric="f1")

    sig_bac = wilcoxon_vs_baseline(aggregated, baseline_name="RandomInit",
                                   metric_key="bac_values",
                                   correction=args.correction)
    sig_auc = wilcoxon_vs_baseline(aggregated, baseline_name="RandomInit",
                                   metric_key="auc_values",
                                   correction=args.correction)

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


if __name__ == "__main__":
    main()

"""
hpo.py – Optymalizacja hiperparametrów z Optuna.

Użycie:
    from contrastive_framework.hpo import optimize

    best = optimize(SCARFModel, dm, n_trials=30)
    print(best)   # {'lr': 0.00031, 'hidden_dim': 192, ...}
"""

from __future__ import annotations
from typing import Any, Dict, Type

import torch
import torch.nn as nn
import optuna

from .models.base import BaseContrastiveModel
from .data_module import DataModule
from .experiment import (
    pretrain, finetune_linear, evaluate, _build_warmup_cosine, filter_model_kwargs,
    _split_param_groups,
)


# ---------------------------------------------------------------------------
# Przestrzenie hiperparametrów – jedna funkcja na algorytm
# ---------------------------------------------------------------------------

def _suggest(
    trial: optuna.Trial,
    model_class: Type[BaseContrastiveModel],
    max_queue_size: int = 4096,
) -> Dict[str, Any]:
    """Zwraca słownik hiperparametrów zaproponowanych przez Optunę."""
    base = {
        "pretrain_lr":  trial.suggest_float("pretrain_lr",  1e-4, 1e-2, log=True),
        "finetune_lr":  trial.suggest_float("finetune_lr",  1e-4, 1e-2, log=True),
        "hidden_dim":   trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "proj_dim":     trial.suggest_categorical("proj_dim",   [32, 64, 128]),
    }

    name = model_class.__name__

    if name == "SCARFModel":
        base["corruption_rate"] = trial.suggest_float("corruption_rate", 0.3, 0.8)
        base["temperature"]     = trial.suggest_float("temperature",     0.1, 1.0)

    elif name == "SubTabModel":
        base["n_subsets"]  = trial.suggest_int("n_subsets", 2, 6)
        base["overlap"]    = trial.suggest_int("overlap",   0, 4)
        base["noise_std"]  = trial.suggest_float("noise_std", 0.01, 0.3, log=True)
        base["temperature"]= trial.suggest_float("temperature", 0.1, 1.0)

    elif name == "MoCoModel":
        base["momentum"]    = trial.suggest_float("momentum",    0.99, 0.9999, log=True)
        base["temperature"] = trial.suggest_float("temperature", 0.05, 0.5)
        # queue_size ograniczone do n_train/4, by uniknąć duplikatów jako negatywów
        candidates = [c for c in (256, 1024, 4096) if c <= max_queue_size]
        if not candidates:
            candidates = [max(64, max_queue_size)]
        base["queue_size"] = trial.suggest_categorical("queue_size", candidates)

    elif name == "BYOLModel":
        base["momentum"] = trial.suggest_float("momentum", 0.99, 0.9999, log=True)

    elif name == "TSTCCModel":
        base["tc_weight"]   = trial.suggest_float("tc_weight",   0.1, 0.9)
        base["temperature"] = trial.suggest_float("temperature", 0.05, 0.5)

    return base


# ---------------------------------------------------------------------------
# Funkcja celu (jedna próba Optuny)
# ---------------------------------------------------------------------------

def _objective(
    trial: optuna.Trial,
    model_class: Type[BaseContrastiveModel],
    fixed_model_kwargs: Dict[str, Any],
    data_module: DataModule,
    pretrain_epochs: int,
    finetune_epochs: int,
    finetune_fraction: float,
    device: torch.device,
) -> float:
    """Zwraca macro F1 na zbiorze walidacyjnym dla jednej konfiguracji hiperparametrów."""
    try:
        n_train = len(data_module.train_dataset)  # type: ignore[arg-type]
    except Exception:
        n_train = 4096 * 4
    max_q = max(64, n_train // 4)
    hp = _suggest(trial, model_class, max_queue_size=max_q)

    model_kwargs = filter_model_kwargs(model_class, {
        **fixed_model_kwargs,
        "hidden_dim": hp["hidden_dim"],
        "proj_dim":   hp["proj_dim"],
        **{k: v for k, v in hp.items()
           if k not in ("pretrain_lr", "finetune_lr", "hidden_dim", "proj_dim")},
    })

    torch.manual_seed(trial.number)
    model: BaseContrastiveModel = model_class(**model_kwargs).to(device)
    num_classes = fixed_model_kwargs.get("num_classes", 2)

    opt_pre = torch.optim.Adam(
        _split_param_groups(model, weight_decay=1e-4),
        lr=hp["pretrain_lr"],
    )
    sched_pre = _build_warmup_cosine(opt_pre, pretrain_epochs,
                                     warmup_epochs=max(1, pretrain_epochs // 10))

    pretrain_loader = data_module.pretrain_loader()
    if hasattr(model, "prefill_queue"):
        model.prefill_queue(pretrain_loader)

    pretrain(model, pretrain_loader, opt_pre,
             device, epochs=pretrain_epochs, verbose=True,
             scheduler=sched_pre)

    # Fine-tuning BEZ early stopping — val_loader używany tylko raz na końcu,
    # by uniknąć selection bias (Cawley & Talbot 2010).
    model.freeze_encoder()
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)

    opt_ft = torch.optim.Adam(model.classifier.parameters(), lr=hp["finetune_lr"])
    sched_ft = _build_warmup_cosine(opt_ft, finetune_epochs,
                                    warmup_epochs=max(1, finetune_epochs // 10))
    finetune_linear(model, data_module.finetune_loader(finetune_fraction),
                    opt_ft, device, epochs=finetune_epochs, verbose=False,
                    val_loader=None, num_classes=num_classes,
                    scheduler=sched_ft)

    metrics = evaluate(model, data_module.val_loader(), device, num_classes)

    return metrics["macro_f1"]


# ---------------------------------------------------------------------------
# Publiczne API
# ---------------------------------------------------------------------------

def optimize(
    model_class: Type[BaseContrastiveModel],
    data_module: DataModule,
    fixed_model_kwargs: Dict[str, Any],
    n_trials: int = 20,
    pretrain_epochs: int = 30,
    finetune_epochs: int = 20,
    finetune_fraction: float = 0.5,
    device: torch.device | None = None,
    direction: str = "maximize",
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Uruchamia optymalizację hiperparametrów dla wybranego algorytmu.

    Parameters
    ----------
    model_class        : klasa modelu (np. SCARFModel)
    data_module        : skonfigurowany DataModule
    fixed_model_kwargs : argumenty konstruktora, które NIE są optymalizowane
                         (np. num_features, num_classes)
    n_trials           : liczba prób Optuny
    pretrain_epochs    : epoki pre-treningu w każdej próbie
    finetune_epochs    : epoki fine-tuningu w każdej próbie
    finetune_fraction  : frakcja etykiet używana podczas HPO (domyślnie 50%)
    device             : urządzenie (domyślnie: CUDA jeśli dostępna)
    direction          : "maximize" (macro F1) lub "minimize"
    show_progress      : czy pokazywać pasek postępu tqdm

    Returns
    -------
    Słownik z kluczami:
        best_params  – najlepsze hiperparametry
        best_value   – najlepszy macro F1
        study        – obiekt optuna.Study (do dalszych analiz)

    Przykład
    --------
    >>> best = optimize(
    ...     SCARFModel,
    ...     dm,
    ...     fixed_model_kwargs={"num_features": 20, "num_classes": 3},
    ...     n_trials=30,
    ... )
    >>> print(best["best_params"])
    >>> print(f"Najlepsza Accuracy: {best['best_value']:.4f}")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler_seed = getattr(getattr(data_module, "cfg", None), "seed", 0)
    sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
    study = optuna.create_study(direction=direction, sampler=sampler)

    print(f"Optuna HPO: {model_class.__name__}  |  {n_trials} prób  |  "
          f"urządzenie: {device}")

    study.optimize(
        lambda trial: _objective(
            trial,
            model_class=model_class,
            fixed_model_kwargs=fixed_model_kwargs,
            data_module=data_module,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            finetune_fraction=finetune_fraction,
            device=device,
        ),
        n_trials=n_trials,
        show_progress_bar=show_progress,
        catch=(Exception,),
    )

    # Sprawdź czy chociaż jedna próba się udała
    complete_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not complete_trials:
        failed = len(study.trials)
        raise RuntimeError(
            f"Wszystkie {failed} próby HPO dla {model_class.__name__} zakończyły się błędem. "
            f"Sprawdź logi Optuny powyżej. Typowe przyczyny: eksplodujące gradienty "
            f"(dodaj gradient clipping — fix B3), zbyt duże lr, OOM."
        )

    n_failed = len(study.trials) - len(complete_trials)
    if n_failed > 0:
        print(f"  [HPO] Uwaga: {n_failed}/{len(study.trials)} prób zakończyło się błędem "
              f"(oznaczone FAIL w study, pominięte przy wyborze best_params).")

    print(f"\nNajlepszy macro F1 : {study.best_value:.4f}")
    print(f"Najlepsze params   : {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value":  study.best_value,
        "study":       study,
    }

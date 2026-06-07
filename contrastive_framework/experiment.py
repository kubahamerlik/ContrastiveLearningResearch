"""
experiment.py – Pętla badawcza integrująca wszystkie komponenty frameworku.

Protokół:
  1. Pre-trening samo-nadzorowany (pretrain_step)
  2. Zamrożenie kodera
  3. Trening klasyfikatora liniowego na kolejnych frakcjach etykiet
  4. Ewaluacja na ukrytym zbiorze testowym
  5. Zapis wyników do słownika (gotowe do tabel i wykresów)
"""

from __future__ import annotations

import inspect
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data_module import DataModule
from .models.base import BaseContrastiveModel


# ---------------------------------------------------------------------------
# Helper: filtrowanie kwargs do sygnatury modelu
# ---------------------------------------------------------------------------

def _split_param_groups(
    model: nn.Module, weight_decay: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Parametry BN/bias dostają WD=0 (standard SimCLR/MoCo/BYOL)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def filter_model_kwargs(
    model_class: Type, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Zwraca tylko te kwargs, które są akceptowane przez model_class.__init__."""
    sig = inspect.signature(model_class.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted}


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Zatrzymuje trening po `patience` epokach bez poprawy metryki walidacyjnej."""

    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode musi być 'max' lub 'min', dostałem '{mode}'")
        self.patience = patience
        self.mode = mode
        self.best: float = float("-inf") if mode == "max" else float("inf")
        self.counter: int = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, value: float, model: nn.Module) -> bool:
        """Zwraca True jeśli należy przerwać trening."""
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Scheduler: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def _build_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    """
    Linear warm-up przez `warmup_epochs`, potem cosine decay do 0.

    Standardowy schemat z SimCLR/MoCo/BYOL. Wymaga `scheduler.step()`
    raz na epokę.
    """
    warmup_epochs = max(1, min(warmup_epochs, total_epochs))
    cosine_epochs = max(1, total_epochs - warmup_epochs)

    # SequentialLR wywołuje step() przy inicjalizacji, co triggeruje fałszywy
    # UserWarning PyTorch 2.x — tłumimy wyłącznie ten warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
            category=UserWarning,
        )
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_epochs,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=0.0,
        )
        seq_sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
    return seq_sched


# ---------------------------------------------------------------------------
# Metryki
# ---------------------------------------------------------------------------

def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    """Oblicza Accuracy, macro F1, balanced accuracy i macro AUC-ROC."""
    preds = logits.argmax(dim=-1)
    N = len(labels)
    correct = (preds == labels).sum().item()
    accuracy = correct / N

    # Liczymy tylko klasy z obserwacjami — klasy nieobecne w batchu są pomijane
    f1_scores: List[float] = []
    recall_scores: List[float] = []  # do balanced accuracy

    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1_scores.append(2 * prec * rec / (prec + rec + 1e-8))
        recall_scores.append(rec)

    macro_f1        = sum(f1_scores)    / len(f1_scores)    if f1_scores    else 0.0
    balanced_acc    = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    probs = torch.softmax(logits, dim=-1)  # (N, C)
    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        labels_np = labels.cpu().numpy()
        probs_np = probs.cpu().numpy()
        present_classes = np.unique(labels_np)
        if num_classes == 2:
            # Binary: użyj P(klasa 1) jako score
            if len(present_classes) < 2:
                macro_auc = float("nan")
            else:
                macro_auc = float(roc_auc_score(labels_np, probs_np[:, 1]))
        else:
            # Multi-class one-vs-rest: filtruj klasy nieobecne w batchu
            if len(present_classes) < 2:
                macro_auc = float("nan")
            else:
                per_class_auc: List[float] = []
                for c in range(num_classes):
                    y_bin = (labels_np == c).astype(np.int32)
                    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                        continue
                    per_class_auc.append(
                        float(roc_auc_score(y_bin, probs_np[:, c]))
                    )
                macro_auc = (
                    sum(per_class_auc) / len(per_class_auc)
                    if per_class_auc else float("nan")
                )
    except Exception:
        macro_auc = float("nan")

    return {
        "accuracy":      accuracy,
        "macro_f1":      macro_f1,
        "balanced_acc":  balanced_acc,
        "macro_auc":     macro_auc,
    }


# ---------------------------------------------------------------------------
# Ewaluacja (inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: BaseContrastiveModel,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Uruchamia model na całym zbiorze testowym i zwraca metryki."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model.linear_readout(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    return compute_metrics(
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# Pętla pre-treningowa
# ---------------------------------------------------------------------------

def _alignment_uniformity(
    z1: torch.Tensor, z2: torch.Tensor, alpha: float = 2.0, t: float = 2.0,
) -> Dict[str, float]:
    """Alignment + uniformity (Wang & Isola 2020). z1, z2: (B, D) z pary pozytywnej."""
    import torch.nn.functional as _F
    z1 = _F.normalize(z1, dim=-1)
    z2 = _F.normalize(z2, dim=-1)
    align = (z1 - z2).norm(dim=-1).pow(alpha).mean().item()
    pdist = torch.pdist(z1, p=2).pow(2)
    uniform = pdist.mul(-t).exp().mean().clamp_min(1e-12).log().item()
    return {"alignment": align, "uniformity": uniform}


def pretrain(
    model: BaseContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    verbose: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    monitor_metrics: bool = False,
    log_every: int = 10,
) -> List[float]:
    """Uruchamia samo-nadzorowany pre-trening. Zwraca listę strat per epoka."""
    model.train()
    epoch_losses: List[float] = []

    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = 1
    total_steps = max(1, steps_per_epoch * epochs)
    global_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        batch_losses: List[float] = []

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = tuple(b.to(device) if isinstance(b, torch.Tensor)
                              else b for b in batch)
            else:
                batch = batch.to(device)

            optimizer.zero_grad()
            result = model.pretrain_step(batch)
            loss: torch.Tensor = result["loss"]
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # EMA/momentum update MUSI być po optimizer.step() — zgodnie z He 2020, Grill 2020
                if hasattr(model, "update_momentum"):
                    model.update_momentum(global_step, total_steps)
                if hasattr(model, "_momentum_update"):
                    model._momentum_update()
                elif hasattr(model, "_ema_update"):
                    model._ema_update()
            batch_losses.append(loss.item())
            global_step += 1

        if scheduler is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
                    category=UserWarning,
                )
                scheduler.step()

        epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
        epoch_losses.append(epoch_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  [Pretrain] Epoka {epoch:>4}/{epochs}  "
                  f"loss={epoch_loss:.4f}  lr={cur_lr:.2e}  ({elapsed:.1f}s)")

        if monitor_metrics and (epoch % max(1, log_every) == 0 or epoch == epochs):
            try:
                model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(loader))
                    if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
                        x1 = sample_batch[0].to(device)
                        x2 = sample_batch[1].to(device)
                        # Wymagamy że x1/x2 to dwa widoki tej samej próbki
                        if x1.shape == x2.shape and hasattr(model, "projector"):
                            h1 = model.encoder(x1) if not isinstance(
                                model.encoder, torch.nn.Identity
                            ) else model.forward(x1)
                            h2 = model.encoder(x2) if not isinstance(
                                model.encoder, torch.nn.Identity
                            ) else model.forward(x2)
                            z1 = model.projector(h1)
                            z2 = model.projector(h2)
                            au = _alignment_uniformity(z1, z2)
                            if verbose:
                                print(f"    [Monitor] align={au['alignment']:.4f}  "
                                      f"uniform={au['uniformity']:.4f}")
                model.train()
            except Exception:
                model.train()

    return epoch_losses


# ---------------------------------------------------------------------------
# Pętla fine-tuningu klasyfikatora liniowego
# ---------------------------------------------------------------------------

def finetune_linear(
    model: BaseContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50,
    verbose: bool = False,
    val_loader: Optional[DataLoader] = None,
    num_classes: int = 2,
    patience: int = 10,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Optional[int]:
    """
    Trenuje TYLKO głowicę klasyfikacyjną (koder jest zamrożony).

    val_loader włącza Early Stopping na macro_f1. Macro F1 jest odporne na
    niezbalansowanie klas (np. Credit Card Fraud z 0.17% fraudów).
    """
    assert model._encoder_frozen, (
        "Wywołaj model.freeze_encoder() przed finetune_linear()!"
    )
    criterion = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=patience, mode="max") if val_loader is not None else None
    last_epoch: Optional[int] = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model.linear_readout(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if es is not None and val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, num_classes)
            should_stop = es.step(val_metrics["macro_f1"], model)
            last_epoch = epoch
            if verbose:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"    [Finetune] Epoka {epoch:>3}/{epochs}  "
                      f"val_f1={val_metrics['macro_f1']:.4f}  "
                      f"best={es.best:.4f}  lr={cur_lr:.2e}")
            if should_stop:
                if verbose:
                    print(f"    [Finetune] Early stop @ epoka {epoch} "
                          f"(brak poprawy przez {patience} epok)")
                break

    if es is not None:
        es.restore(model)

    return last_epoch


# ---------------------------------------------------------------------------
# Główna pętla eksperymentu
# ---------------------------------------------------------------------------

def finetune_end_to_end(
    model: BaseContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50,
    verbose: bool = False,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """Trenuje WSZYSTKIE parametry modelu (koder + klasyfikator) end-to-end."""
    assert not model._encoder_frozen, (
        "Wywołaj model.unfreeze_encoder() przed finetune_end_to_end()!"
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model.linear_readout(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if verbose:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"    [E2E] Epoka {epoch:>3}/{epochs}  lr={cur_lr:.2e}")


def run_experiment(
    model_class: Type[BaseContrastiveModel],
    model_kwargs: Dict[str, Any],
    data_module: DataModule,
    device: Optional[torch.device] = None,
    pretrain_epochs: int = 100,
    finetune_epochs: int = 50,
    pretrain_lr: float = 3e-4,
    finetune_lr: float = 1e-3,
    verbose: bool = True,
    use_scheduler: bool = True,
    warmup_epochs: int = 5,
    run_finetune: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_tag: str = "",
    monitor_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Kompletna procedura badawcza dla jednego algorytmu.

    Schemat
    -------
    pretrain (cały zbiór, bez etykiet)
        ↓
    freeze_encoder()
        ↓
    for fraction in [6.25%, 12.5%, 25%, 50%, 100%]:
        finetune_linear(fraction% etykiet)
            ↓
        evaluate(pełny zbiór testowy)
            ↓
        zapisz wyniki
    [opcjonalnie gdy run_finetune=True:]
        unfreeze_encoder()
        finetune_end_to_end(fraction% etykiet)
            ↓
        evaluate(pełny zbiór testowy)

    Parameters
    ----------
    model_class  : klasa modelu (np. MoCoModel)
    model_kwargs : argumenty konstruktora modelu
    data_module  : skonfigurowany DataModule
    device       : torch.device (domyślnie: CUDA jeśli dostępna)
    run_finetune : jeśli True, dodatkowo przeprowadza ewaluację end-to-end
                   (fine-tuning WSZYSTKICH parametrów) dla każdej frakcji.
                   Wyniki zapisane pod kluczem "results_finetune".

    Returns
    -------
    results : słownik z wynikami:
        {
          "pretrain_losses": [...],
          "results": {
              0.0625: {"accuracy": 0.72, "macro_f1": 0.71},
              ...
          },
          "results_finetune": {   # tylko gdy run_finetune=True
              0.0625: {"accuracy": 0.74, "macro_f1": 0.73},
              ...
          }
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        finetune_info = " + E2E" if run_finetune else ""
        print(f"\n{'='*60}")
        print(f"  Algorytm  : {model_class.__name__}")
        print(f"  Urządzenie: {device}")
        print(f"  Pre-trening: {pretrain_epochs} epok  |  "
              f"Fine-tuning: {finetune_epochs} epok{finetune_info}")
        print(f"{'='*60}")
        print(data_module.summary())

    model = model_class(**model_kwargs).to(device)

    if verbose:
        print(f"\n[1/3] Pre-trening samo-nadzorowany...")

    pretrain_optimizer = torch.optim.Adam(
        _split_param_groups(model, weight_decay=1e-4),
        lr=pretrain_lr,
    )
    pretrain_scheduler = (
        _build_warmup_cosine(pretrain_optimizer, pretrain_epochs, warmup_epochs)
        if use_scheduler else None
    )
    pretrain_loader = data_module.pretrain_loader()

    if hasattr(model, "prefill_queue"):
        if verbose:
            print("  [MoCo] Pre-fill kolejki prawdziwymi embeddingami...")
        model.prefill_queue(pretrain_loader)

    pretrain_losses = pretrain(
        model, pretrain_loader, pretrain_optimizer,
        device, epochs=pretrain_epochs, verbose=verbose,
        scheduler=pretrain_scheduler,
        monitor_metrics=monitor_metrics,
    )

    if checkpoint_dir is not None:
        import pathlib as _pathlib
        ckpt_path = _pathlib.Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        tag = f"_{checkpoint_tag}" if checkpoint_tag else ""
        torch.save(
            {"state_dict": model.state_dict(), "kwargs": model_kwargs,
             "stage": "pretrain", "pretrain_losses": pretrain_losses},
            ckpt_path / f"{model_class.__name__}{tag}_pretrain.pt",
        )

    if verbose:
        print(f"\n[2/3] Zamrażanie kodera...")
    model.freeze_encoder()

    n_steps = "3/3" if not run_finetune else "3/4"
    if verbose:
        print(f"\n[{n_steps}] Liniowa ewaluacja na kolejnych frakcjach etykiet:\n")

    data_module.hpo_done = True
    test_loader = data_module.test_loader()
    num_classes: int = model_kwargs.get("num_classes", 2)
    fraction_results: Dict[float, Dict[str, float]] = {}

    ft_warmup = max(1, finetune_epochs // 10) if use_scheduler else 0

    for frac, ft_loader in data_module.iter_finetune_loaders():
        nn.init.xavier_uniform_(model.classifier.weight)
        nn.init.zeros_(model.classifier.bias)

        ft_optimizer = torch.optim.Adam(
            model.classifier.parameters(), lr=finetune_lr
        )
        ft_scheduler = (
            _build_warmup_cosine(ft_optimizer, finetune_epochs, ft_warmup)
            if use_scheduler else None
        )
        # val_loader=None: stała liczba epok — val był już użyty przez HPO do
        # wyboru hiperparametrów; ponowne użycie przez ES prowadziłoby do
        # selection bias (Cawley & Talbot 2010).
        finetune_linear(
            model, ft_loader, ft_optimizer,
            device, epochs=finetune_epochs, verbose=False,
            val_loader=None, num_classes=num_classes,
            scheduler=ft_scheduler,
        )

        metrics = evaluate(model, test_loader, device, num_classes)
        fraction_results[frac] = metrics

        if verbose:
            pct = f"{frac * 100:.2f}%"
            print(f"  Frakcja etykiet: {pct:>7}  →  "
                  f"Accuracy={metrics['accuracy']:.4f}  "
                  f"F1={metrics['macro_f1']:.4f}")

    if checkpoint_dir is not None:
        import pathlib as _pathlib
        ckpt_path = _pathlib.Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        tag = f"_{checkpoint_tag}" if checkpoint_tag else ""
        torch.save(
            {"state_dict": model.state_dict(), "kwargs": model_kwargs,
             "stage": "finetune_linear", "results": fraction_results},
            ckpt_path / f"{model_class.__name__}{tag}_finetune.pt",
        )

    finetune_results: Dict[float, Dict[str, float]] = {}
    if run_finetune:
        if verbose:
            print(f"\n[4/4] Fine-tuning end-to-end na kolejnych frakcjach etykiet:\n")

        import copy
        pretrained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        for frac, ft_loader in data_module.iter_finetune_loaders():
            model.load_state_dict(pretrained_state)
            model.unfreeze_encoder()

            nn.init.xavier_uniform_(model.classifier.weight)
            nn.init.zeros_(model.classifier.bias)

            e2e_optimizer = torch.optim.Adam(
                _split_param_groups(model, weight_decay=1e-4),
                lr=finetune_lr,
            )
            e2e_scheduler = (
                _build_warmup_cosine(e2e_optimizer, finetune_epochs, ft_warmup)
                if use_scheduler else None
            )

            finetune_end_to_end(
                model, ft_loader, e2e_optimizer,
                device, epochs=finetune_epochs, verbose=False,
                scheduler=e2e_scheduler,
            )

            metrics = evaluate(model, test_loader, device, num_classes)
            finetune_results[frac] = metrics

            if verbose:
                pct = f"{frac * 100:.2f}%"
                print(f"  Frakcja etykiet: {pct:>7}  →  "
                      f"Accuracy={metrics['accuracy']:.4f}  "
                      f"F1={metrics['macro_f1']:.4f}")

    if verbose:
        print(f"\n{'='*60}\n")

    out: Dict[str, Any] = {
        "algorithm": model_class.__name__,
        "pretrain_losses": pretrain_losses,
        "results": fraction_results,
    }
    if run_finetune:
        out["results_finetune"] = finetune_results
    return out


def run_all_experiments(
    algorithms: List[Dict[str, Any]],
    data_module: DataModule,
    **run_kwargs,
) -> Dict[str, Any]:
    """
    Uruchamia pełną pętlę badawczą dla listy algorytmów.

    Parameters
    ----------
    algorithms : lista słowników, każdy z kluczami:
        - "model_class"  : Type[BaseContrastiveModel]
        - "model_kwargs" : Dict[str, Any]
    data_module : DataModule
    **run_kwargs : dodatkowe argumenty przekazywane do run_experiment()

    Returns
    -------
    all_results : słownik { NazwaKlasy → wyniki_eksperymentu }

    Przykład
    --------
    >>> from contrastive_framework.models.image_models import MoCoModel, BYOLModel
    >>> from contrastive_framework.models.tabular_models import SCARFModel
    >>> algorithms = [
    ...     {"model_class": MoCoModel,  "model_kwargs": {"input_dim": 784}},
    ...     {"model_class": BYOLModel,  "model_kwargs": {"input_dim": 784}},
    ...     {"model_class": SCARFModel, "model_kwargs": {"num_features": 30}},
    ... ]
    >>> results = run_all_experiments(algorithms, data_module)
    """
    all_results: Dict[str, Any] = {}

    for algo in algorithms:
        model_class: Type[BaseContrastiveModel] = algo["model_class"]
        model_kwargs: Dict[str, Any] = algo["model_kwargs"]

        data_module.hpo_done = False

        result = run_experiment(
            model_class=model_class,
            model_kwargs=model_kwargs,
            data_module=data_module,
            **run_kwargs,
        )
        all_results[model_class.__name__] = result

    return all_results

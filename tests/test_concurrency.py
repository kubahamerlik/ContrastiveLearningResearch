"""
Testy współbieżności i reprodukowalności.

1. Atomowy zapis splitów przy dwóch równoległych procesach (multiprocessing).
2. merge_results.py: wykrywa zdublowany seed.
3. merge_results.py: wykrywa niezgodną konfigurację.
4. Ten sam seed na CPU dwa razy → identyczne wyniki (niezależność od kolejności).
"""

from __future__ import annotations

import json
import sys
import os
import tempfile
import time
import multiprocessing
from pathlib import Path
from typing import Any, Dict

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_split_manager(path: Path):
    from contrastive_framework.data_module import SplitManager
    return SplitManager(path)


def _worker_create_split(split_file_str: str, seed: int, n_total: int,
                         result_queue: multiprocessing.Queue) -> None:
    """Robotnik: tworzy split i wkłada wynik do kolejki."""
    # Na Windows subprocesy mogą dziedziczyć cp1250 — wymuszamy UTF-8
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    try:
        path = Path(split_file_str)
        sm = _make_split_manager(path)
        y = torch.arange(n_total) % 3  # 3 klasy
        train, val, test = sm.get_or_create(
            seed=seed, n_total=n_total,
            val_fraction=0.15, test_fraction=0.15,
            labels=y,
        )
        result_queue.put(("ok", seed, len(train), len(val), len(test)))
    except Exception as e:
        result_queue.put(("err", seed, str(e)))


# ---------------------------------------------------------------------------
# Test 1: atomowy zapis splitów — dwa procesy równolegle
# ---------------------------------------------------------------------------

def test_concurrent_split_write_no_corruption():
    """
    Dwa procesy równolegle zapisują ten sam plik splitów.
    Oczekiwane: plik nie jest uszkodzony, oba splity są wewnętrznie spójne.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        split_file = Path(tmpdir) / "splits.json"
        n_total = 300
        q: multiprocessing.Queue = multiprocessing.Queue()

        p0 = multiprocessing.Process(
            target=_worker_create_split,
            args=(str(split_file), 0, n_total, q),
        )
        p1 = multiprocessing.Process(
            target=_worker_create_split,
            args=(str(split_file), 1, n_total, q),
        )
        p0.start()
        p1.start()
        p0.join(timeout=30)
        p1.join(timeout=30)

        results = []
        while not q.empty():
            results.append(q.get_nowait())

        assert len(results) == 2, f"Oczekiwano 2 wyników, dostałem: {results}"
        for r in results:
            assert r[0] == "ok", f"Robotnik zwrócił błąd: {r}"

        # Plik musi istnieć i być poprawnym JSON-em
        assert split_file.exists(), "Plik splitów nie istnieje po równoległym zapisie"
        with open(split_file) as f:
            data = json.load(f)

        # Przynajmniej jeden seed musi być zapisany
        assert len(data) >= 1, f"Plik splitów pusty: {data}"

        # Każdy seed ma poprawną sumę indeksów
        for key in ("0", "1"):
            if key in data:
                s = data[key]
                total = len(s["train"]) + len(s["val"]) + len(s["test"])
                assert total == n_total, (
                    f"seed={key}: suma indeksów {total} ≠ {n_total}"
                )


def test_concurrent_same_seed_idempotent():
    """
    Dwa procesy próbują stworzyć split dla TEGO SAMEGO seeda równocześnie.
    Wynik musi być deterministyczny (oba widzą te same indeksy).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        split_file = Path(tmpdir) / "splits.json"
        n_total = 200
        q: multiprocessing.Queue = multiprocessing.Queue()

        processes = [
            multiprocessing.Process(
                target=_worker_create_split,
                args=(str(split_file), 42, n_total, q),
            )
            for _ in range(2)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=30)

        results = [q.get_nowait() for _ in range(len(processes))]
        for r in results:
            assert r[0] == "ok", f"Błąd: {r}"

        # Obie instancje muszą zwrócić identyczne rozmiary
        sizes = [(r[2], r[3], r[4]) for r in results]
        assert sizes[0] == sizes[1], (
            f"Różne rozmiary splitów dla tego samego seeda: {sizes}"
        )


# ---------------------------------------------------------------------------
# Test 2: merge_results — zdublowany seed
# ---------------------------------------------------------------------------

def _make_result_file(path: Path, seeds: list, dataset: str = "pathmnist",
                      algos: list = None) -> None:
    if algos is None:
        algos = ["RandomInit", "MoCo"]
    fracs = ["0.0625", "0.5", "1.0"]
    results = {}
    for s in seeds:
        results[str(s)] = {
            algo: {f: {"accuracy": 0.5, "macro_f1": 0.5} for f in fracs}
            for algo in algos
        }
    data = {
        "config": {
            "dataset": dataset,
            "pretrain_epochs": 10,
            "finetune_epochs": 5,
            "batch_size": 64,
            "subsample": 1000,
            "n_hpo_trials": 2,
            "hpo_pretrain_epochs": 2,
            "hpo_finetune_epochs": 2,
            "hpo_label_fraction": 0.5,
            "num_workers": 0,
            "splits_dir": "splits",
            "correction": "holm",
            "visualize": False,
            "viz_method": "tsne",
            "viz_dir": "figures",
            "run_finetune": False,
            "checkpoints_dir": None,
            "monitor_metrics": False,
            "algorithms": None,
            "data_dir": "data",
        },
        "results": results,
    }
    path.write_text(json.dumps(data, indent=2))


def test_merge_detects_duplicate_seed(tmp_path):
    """merge_results.py musi wykryć zdublowany seed i zakończyć z błędem."""
    f0 = tmp_path / "r0.json"
    f1 = tmp_path / "r1.json"
    _make_result_file(f0, seeds=[0, 1, 2])
    _make_result_file(f1, seeds=[2, 3, 4])  # seed 2 zdublowany

    from scripts.merge_results import _load, _check_consistency
    data = [_load(f0), _load(f1)]

    with pytest.raises(SystemExit) as exc_info:
        _check_consistency([f0, f1], data)
    assert exc_info.value.code == 1


def test_merge_accepts_disjoint_seeds(tmp_path):
    """merge_results.py akceptuje rozłączne seedy."""
    f0 = tmp_path / "r0.json"
    f1 = tmp_path / "r1.json"
    _make_result_file(f0, seeds=[0, 2, 4])
    _make_result_file(f1, seeds=[1, 3, 5])

    from scripts.merge_results import _load, _check_consistency, _merge
    import argparse
    data = [_load(f0), _load(f1)]
    _check_consistency([f0, f1], data)  # nie powinno rzucić

    ns = argparse.Namespace(out=tmp_path / "merged.json")
    merged = _merge(data, ns)
    assert set(merged["results"].keys()) == {"0", "1", "2", "3", "4", "5"}


def test_merge_detects_config_mismatch(tmp_path):
    """merge_results.py wykrywa niezgodną konfigurację (różne pretrain_epochs)."""
    f0 = tmp_path / "r0.json"
    f1 = tmp_path / "r1.json"
    _make_result_file(f0, seeds=[0, 1])
    _make_result_file(f1, seeds=[2, 3])

    # Zmień konfigurację w f1
    d = json.loads(f1.read_text())
    d["config"]["pretrain_epochs"] = 999
    f1.write_text(json.dumps(d))

    from scripts.merge_results import _load, _check_consistency
    data = [_load(f0), _load(f1)]

    with pytest.raises(SystemExit) as exc_info:
        _check_consistency([f0, f1], data)
    assert exc_info.value.code == 1


def test_merge_ignores_device_and_seeds_diff(tmp_path):
    """Różne --device i --seeds nie są traktowane jako błąd konfiguracji."""
    f0 = tmp_path / "r0.json"
    f1 = tmp_path / "r1.json"
    _make_result_file(f0, seeds=[0])
    _make_result_file(f1, seeds=[1])

    d0 = json.loads(f0.read_text())
    d1 = json.loads(f1.read_text())
    d0["config"]["device"] = "cuda:0"
    d0["config"]["seeds"] = [0]
    d1["config"]["device"] = "cuda:1"
    d1["config"]["seeds"] = [1]
    f0.write_text(json.dumps(d0))
    f1.write_text(json.dumps(d1))

    from scripts.merge_results import _load, _check_consistency
    data = [_load(f0), _load(f1)]
    _check_consistency([f0, f1], data)  # nie powinno rzucić


# ---------------------------------------------------------------------------
# Test 3: reprodukowalność — ten sam seed dwa razy daje identyczne wyniki
# ---------------------------------------------------------------------------

def _run_tiny_pretrain(seed: int) -> Dict[str, Any]:
    """Uruchamia miniaturowy pretrain MoCo i zwraca loss z ostatniej epoki."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from contrastive_framework.models.image_models import MoCoModel
    from contrastive_framework.data_module import image_data_module, DataModuleConfig

    N = 64
    X = torch.rand(N, 3, 8, 8)
    y = torch.randint(0, 2, (N,))

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = DataModuleConfig(batch_size=16, seed=seed)
        dm = image_data_module(
            X, y, config=cfg,
            split_file=Path(tmpdir) / "splits.json",
            test_fraction=0.2,
        )
        model = MoCoModel(
            encoder_type="cnn_image", input_dim=3,
            hidden_dim=32, proj_dim=16,
            num_classes=2, queue_size=32,
        )
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = dm.pretrain_loader()

        losses = []
        for _ in range(2):
            for batch in loader:
                out = model.pretrain_step(batch)
                loss = out["loss"]
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(round(loss.item(), 6))

    return {"losses": losses}


def test_same_seed_reproducible():
    """Ten sam seed uruchomiony dwa razy daje identyczny przebieg strat."""
    r1 = _run_tiny_pretrain(seed=7)
    r2 = _run_tiny_pretrain(seed=7)
    assert r1["losses"] == r2["losses"], (
        f"Wyniki dla seed=7 różnią się między uruchomieniami:\n"
        f"  uruchomienie 1: {r1['losses']}\n"
        f"  uruchomienie 2: {r2['losses']}"
    )


def test_different_seeds_differ():
    """Różne seedy dają różne wyniki (sanity check)."""
    r0 = _run_tiny_pretrain(seed=0)
    r1 = _run_tiny_pretrain(seed=1)
    assert r0["losses"] != r1["losses"], (
        "seed=0 i seed=1 dają identyczne wyniki — seedowanie prawdopodobnie nie działa"
    )

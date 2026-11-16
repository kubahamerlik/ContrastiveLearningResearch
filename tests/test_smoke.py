"""Minimal smoke tests — confirm imports and basic structure hold."""
import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_framework_imports():
    for mod in [
        "contrastive_framework.models.base",
        "contrastive_framework.models.timeseries_models",
        "contrastive_framework.models.tabular_models",
        "contrastive_framework.models.image_models",
    ]:
        importlib.import_module(mod)


def test_base_interface():
    from contrastive_framework.models.base import BaseContrastiveModel
    required = {"pretrain_step", "forward", "linear_readout",
                "freeze_encoder", "unfreeze_encoder", "train"}
    actual = set(dir(BaseContrastiveModel))
    missing = required - actual
    assert not missing, f"BaseContrastiveModel missing methods: {missing}"

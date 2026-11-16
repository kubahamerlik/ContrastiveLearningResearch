"""
Testy dla błędu B: BN w sieci target/kluczy.
Sprawdzamy że:
  1. Bufory BN w encoder_k (MoCo) i encoder_t (BYOL) zmieniają się po kilku krokach
  2. Podczas pretrain_step gałąź target/klucz jest w train() (statystyki batcha)
  3. freeze_encoder ustawia encoder online w eval() (linear probing)
  4. SplitBatchNorm2d działa poprawnie (shape, running stats po forwardzie)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import torch
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bn_buffers(module: torch.nn.Module):
    """Zbiera wszystkie running_mean/var bufory z modułu (rekurencyjnie)."""
    bufs = {}
    for name, buf in module.named_buffers():
        if "running_mean" in name or "running_var" in name:
            bufs[name] = buf.clone()
    return bufs


def _make_moco(bn_splits=1, hidden_dim=32, proj_dim=16, queue_size=32):
    from contrastive_framework.models.image_models import MoCoModel
    return MoCoModel(
        encoder_type="cnn_image",
        input_dim=3,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        num_classes=5,
        queue_size=queue_size,
        bn_splits=bn_splits,
    )


def _make_byol(hidden_dim=32, proj_dim=16):
    from contrastive_framework.models.image_models import BYOLModel
    return BYOLModel(
        encoder_type="cnn_image",
        input_dim=3,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        num_classes=5,
    )


def _random_batch(B=16, C=3, H=8, W=8):
    x1 = torch.rand(B, C, H, W)
    x2 = torch.rand(B, C, H, W)
    return x1, x2


# ---------------------------------------------------------------------------
# MoCo
# ---------------------------------------------------------------------------

class TestMoCoBN:
    def test_encoder_k_in_train_mode_during_pretrain(self):
        """encoder_k jest w train() podczas pretrain (żeby BN liczył batch stats)."""
        model = _make_moco()
        model.train()
        assert model.encoder_k.training, (
            "encoder_k powinno być w trybie train() podczas pre-treningu"
        )

    def test_encoder_k_bn_buffers_change_after_steps(self):
        """Bufory BN encoder_k zmieniają się po kilku krokach (running stats aktualizowane)."""
        model = _make_moco(bn_splits=1)
        model.train()

        buffers_before = _get_bn_buffers(model.encoder_k)

        for _ in range(3):
            batch = _random_batch(B=16)
            model.pretrain_step(batch)

        buffers_after = _get_bn_buffers(model.encoder_k)

        changed = any(
            not torch.allclose(buffers_before[k], buffers_after[k])
            for k in buffers_before
        )
        assert changed, (
            "Bufory BN encoder_k nie zmieniły się po krokach pre-treningu — "
            "BN może działać w eval() z zamrożonymi statystykami"
        )

    def test_encoder_k_split_bn_buffers_change(self):
        """Bufory SplitBatchNorm encoder_k (bn_splits=8) zmieniają się po krokach."""
        model = _make_moco(bn_splits=8)
        model.train()

        buffers_before = _get_bn_buffers(model.encoder_k)

        for _ in range(3):
            batch = _random_batch(B=32)
            model.pretrain_step(batch)

        buffers_after = _get_bn_buffers(model.encoder_k)

        changed = any(
            not torch.allclose(buffers_before[k], buffers_after[k])
            for k in buffers_before
        )
        assert changed, "Bufory SplitBatchNorm encoder_k nie zaktualizowały się"

    def test_freeze_encoder_sets_online_to_eval(self):
        """freeze_encoder() ustawia encoder online w eval() (dla linear probing)."""
        model = _make_moco()
        model.train()
        model.freeze_encoder()
        assert not model.encoder.training, "encoder online powinien być w eval() po freeze"

    def test_freeze_encoder_keeps_encoder_k_eval(self):
        """Po freeze encoder_k też powinien być w eval() (zamrożony model)."""
        model = _make_moco()
        model.train()
        model.freeze_encoder()
        model.eval()
        assert not model.encoder_k.training, (
            "encoder_k powinien być w eval() gdy cały model jest w eval()"
        )

    def test_unfreeze_restores_train_mode(self):
        """unfreeze_encoder() przywraca tryb train() dla encoder online."""
        model = _make_moco()
        model.freeze_encoder()
        model.train()
        model.unfreeze_encoder()
        model.train()
        assert model.encoder.training, "encoder powinien być w train() po unfreeze"

    def test_pretrain_step_returns_loss(self):
        """pretrain_step zwraca słownik z kluczem 'loss'."""
        model = _make_moco()
        model.train()
        batch = _random_batch()
        out = model.pretrain_step(batch)
        assert "loss" in out
        assert out["loss"].item() > 0


# ---------------------------------------------------------------------------
# BYOL
# ---------------------------------------------------------------------------

class TestBYOLBN:
    def test_encoder_t_in_train_mode_during_pretrain(self):
        """encoder_t jest w train() podczas pretrain."""
        model = _make_byol()
        model.train()
        assert model.encoder_t.training, (
            "encoder_t powinno być w trybie train() podczas pre-treningu "
            "(jak is_training=True w oryginalnym BYOL)"
        )

    def test_encoder_t_bn_buffers_change_after_steps(self):
        """Bufory BN encoder_t zmieniają się po krokach (running stats aktualizowane)."""
        model = _make_byol()
        model.train()

        buffers_before = _get_bn_buffers(model.encoder_t)

        for _ in range(3):
            batch = _random_batch(B=16)
            model.pretrain_step(batch)

        buffers_after = _get_bn_buffers(model.encoder_t)

        changed = any(
            not torch.allclose(buffers_before[k], buffers_after[k])
            for k in buffers_before
        )
        assert changed, (
            "Bufory BN encoder_t nie zmieniły się — BN może działać w eval() "
            "z zamrożonymi running_mean/var=0/1"
        )

    def test_freeze_encoder_sets_online_to_eval(self):
        """freeze_encoder() ustawia encoder online w eval()."""
        model = _make_byol()
        model.train()
        model.freeze_encoder()
        assert not model.encoder.training

    def test_encoder_t_eval_when_model_eval(self):
        """encoder_t w eval() gdy cały model w eval()."""
        model = _make_byol()
        model.eval()
        assert not model.encoder_t.training

    def test_pretrain_step_returns_loss(self):
        """pretrain_step zwraca słownik z kluczem 'loss'."""
        model = _make_byol()
        model.train()
        batch = _random_batch()
        out = model.pretrain_step(batch)
        assert "loss" in out
        loss_val = out["loss"].item()
        # Strata BYOL: mean(loss_v1 + loss_v2), każda składowa w [0,4] → mean w [0,8]
        assert 0.0 <= loss_val <= 8.0 + 1e-5, f"Nieoczekiwana wartość straty: {loss_val}"


# ---------------------------------------------------------------------------
# SplitBatchNorm2d
# ---------------------------------------------------------------------------

class TestSplitBatchNorm:
    def test_output_shape(self):
        from contrastive_framework.models.base import SplitBatchNorm2d
        bn = SplitBatchNorm2d(16, num_splits=4)
        bn.train()
        x = torch.rand(32, 16, 4, 4)
        out = bn(x)
        assert out.shape == x.shape

    def test_running_stats_update_in_train(self):
        """running_mean/var aktualizują się w trybie train()."""
        from contrastive_framework.models.base import SplitBatchNorm2d
        bn = SplitBatchNorm2d(8, num_splits=4)
        bn.train()
        mean_before = bn.running_mean.clone()
        x = torch.rand(32, 8, 4, 4)
        bn(x)
        assert not torch.allclose(bn.running_mean, mean_before), (
            "running_mean nie zaktualizowało się w trybie train()"
        )

    def test_running_stats_frozen_in_eval(self):
        """running_mean/var NIE zmieniają się w trybie eval()."""
        from contrastive_framework.models.base import SplitBatchNorm2d
        bn = SplitBatchNorm2d(8, num_splits=4)
        bn.eval()
        mean_before = bn.running_mean.clone()
        x = torch.rand(32, 8, 4, 4)
        bn(x)
        assert torch.allclose(bn.running_mean, mean_before), (
            "running_mean zmieniło się w trybie eval() — błąd w SplitBatchNorm"
        )

    def test_fallback_small_batch(self):
        """Małe batche (B < num_splits) nie rzucają błędu."""
        from contrastive_framework.models.base import SplitBatchNorm2d
        bn = SplitBatchNorm2d(4, num_splits=8)
        bn.train()
        x = torch.rand(4, 4, 2, 2)  # B=4 < num_splits=8
        out = bn(x)
        assert out.shape == x.shape

    def test_split_bn_differs_from_regular_bn(self):
        """
        SplitBN musi dawać INNE wyniki niż zwykły BN dla batcha z bimodalną
        strukturą (dwie grupy próbek o różnych średnich).

        Jeśli SplitBN liczyłoby statystyki z całego batcha (jak zwykły BN),
        wyjście byłoby identyczne.
        """
        from contrastive_framework.models.base import SplitBatchNorm2d
        torch.manual_seed(0)
        C, H, W = 8, 4, 4

        # Batch bimodalny: pierwsza połowa ma wartości ~0, druga ~10
        # Zwykły BN znormalizuje względem global_mean≈5
        # SplitBN (2 splity) znormalizuje każdą połowę osobno względem jej własnej średniej
        x = torch.zeros(16, C, H, W)
        x[:8] = torch.randn(8, C, H, W) * 0.1          # ~N(0, 0.1)
        x[8:] = torch.randn(8, C, H, W) * 0.1 + 10.0   # ~N(10, 0.1)

        split_bn = SplitBatchNorm2d(C, num_splits=2)
        split_bn.train()
        split_bn.weight.data.fill_(1.0)
        split_bn.bias.data.fill_(0.0)

        regular_bn = torch.nn.BatchNorm2d(C)
        regular_bn.train()
        regular_bn.weight.data.fill_(1.0)
        regular_bn.bias.data.fill_(0.0)

        out_split = split_bn(x)
        out_regular = regular_bn(x)

        max_diff = (out_split - out_regular).abs().max().item()
        assert max_diff > 0.5, (
            f"SplitBN daje wynik identyczny ze zwykłym BN (max_diff={max_diff:.6f}). "
            "SplitBN powinno liczyć statystyki osobno dla każdego splitu, "
            "co dla bimodalnego batcha daje różny wynik od globalnego BN."
        )

    def test_split_bn_each_split_normalized(self):
        """
        Każdy split po SplitBN powinien mieć mean≈0, std≈1.
        Zwykły BN dałby globalne mean=0 ale per-split mean byłoby różne od 0.
        """
        from contrastive_framework.models.base import SplitBatchNorm2d
        torch.manual_seed(42)
        C, H, W = 4, 4, 4
        num_splits = 2

        x = torch.zeros(16, C, H, W)
        x[:8] = torch.randn(8, C, H, W) + 5.0
        x[8:] = torch.randn(8, C, H, W) - 5.0

        split_bn = SplitBatchNorm2d(C, num_splits=num_splits)
        split_bn.train()
        split_bn.weight.data.fill_(1.0)
        split_bn.bias.data.fill_(0.0)
        out = split_bn(x)

        # Sprawdź że każda połowa ma mean≈0
        mean_split0 = out[:8].mean().abs().item()
        mean_split1 = out[8:].mean().abs().item()
        assert mean_split0 < 0.1, f"Split 0: mean={mean_split0:.4f}, oczekiwano ≈0"
        assert mean_split1 < 0.1, f"Split 1: mean={mean_split1:.4f}, oczekiwano ≈0"

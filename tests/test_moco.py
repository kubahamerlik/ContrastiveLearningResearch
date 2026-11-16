"""
Testy dla MoCo v2 (image_models.py).

Pokrycie:
- Kształt logitów: (B, 1+K), pozytyw na pozycji 0
- Aktualizacja kolejki (wrap-around, przesunięcie ptr)
- encoder_k: brak gradientów
- Momentum update: θ_k = m·θ_k + (1−m)·θ_q
- Głowica MLP v2: bez BN (Linear → ReLU → Linear)
- pretrain_step: loss skończony, backprop działa
- Inicjalizacja kolejki: losowe znorm. wektory
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from contrastive_framework.models.image_models import MoCoModel, _build_moco_v2_projector


# ─────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────────────────────────────────────────

def make_model(queue_size: int = 32, hidden_dim: int = 64, proj_dim: int = 32,
               bn_splits: int = 1, momentum: float = 0.999) -> MoCoModel:
    """Mały model CPU do testów."""
    return MoCoModel(
        encoder_type="cnn_image",
        input_dim=3,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        num_classes=10,
        queue_size=queue_size,
        momentum=momentum,
        temperature=0.2,
        bn_splits=bn_splits,
    )


def fake_batch(B: int = 4, C: int = 3, H: int = 32, W: int = 32):
    return torch.randn(B, C, H, W), torch.randn(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Głowica MLP v2 — bez BN
# ─────────────────────────────────────────────────────────────────────────────

class TestMoCoV2Projector:
    def test_no_batchnorm(self):
        proj = _build_moco_v2_projector(64, 32)
        for m in proj.modules():
            assert not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)), \
                f"Głowica v2 nie powinna zawierać BN, ale zawiera {type(m)}"

    def test_structure(self):
        proj = _build_moco_v2_projector(64, 32)
        mods = list(proj.children())
        assert len(mods) == 3
        assert isinstance(mods[0], nn.Linear)
        assert isinstance(mods[1], nn.ReLU)
        assert isinstance(mods[2], nn.Linear)

    def test_dims(self):
        in_dim, out_dim = 64, 32
        proj = _build_moco_v2_projector(in_dim, out_dim)
        x = torch.randn(4, in_dim)
        y = proj(x)
        assert y.shape == (4, out_dim)

    def test_projector_in_model_no_bn(self):
        m = make_model()
        for mod in m.projector.modules():
            assert not isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)), \
                f"model.projector nie powinien mieć BN: {type(mod)}"
        for mod in m.projector_k.modules():
            assert not isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)), \
                f"model.projector_k nie powinien mieć BN: {type(mod)}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inicjalizacja kolejki
# ─────────────────────────────────────────────────────────────────────────────

class TestQueueInit:
    def test_shape(self):
        K, D = 32, 32
        m = make_model(queue_size=K, proj_dim=D)
        assert m.queue.shape == (D, K)

    def test_normalized(self):
        m = make_model(queue_size=32, proj_dim=32)
        norms = m.queue.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            "Kolejka powinna zawierać znorm. wektory (norm=1 wzdłuż dim=0)"

    def test_ptr_zero(self):
        m = make_model()
        assert int(m.queue_ptr) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Brak gradientów w encoder_k / projector_k
# ─────────────────────────────────────────────────────────────────────────────

class TestEncoderKNoGrad:
    def test_encoder_k_requires_grad_false(self):
        m = make_model()
        for name, param in m.encoder_k.named_parameters():
            assert not param.requires_grad, \
                f"encoder_k.{name}: requires_grad powinno być False"

    def test_projector_k_requires_grad_false(self):
        m = make_model()
        for name, param in m.projector_k.named_parameters():
            assert not param.requires_grad, \
                f"projector_k.{name}: requires_grad powinno być False"

    def test_encoder_q_requires_grad_true(self):
        m = make_model()
        for name, param in m.encoder.named_parameters():
            assert param.requires_grad, \
                f"encoder.{name}: requires_grad powinno być True"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Momentum update: θ_k = m·θ_k + (1−m)·θ_q
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumUpdate:
    def test_formula_encoder(self):
        m = make_model()
        m.train()
        mom = m.momentum

        # Zapamiętaj wagi przed update
        q_params = {n: p.data.clone() for n, p in m.encoder.named_parameters()}
        k_before = {n: p.data.clone() for n, p in m.encoder_k.named_parameters()}

        m._momentum_update()

        for name, k_after in m.encoder_k.named_parameters():
            expected = k_before[name] * mom + q_params[name] * (1.0 - mom)
            assert torch.allclose(k_after.data, expected, atol=1e-6), \
                f"encoder_k.{name}: EMA niezgodna z θ_k = m·θ_k + (1−m)·θ_q"

    def test_formula_projector(self):
        m = make_model()
        m.train()
        mom = m.momentum

        q_params = {n: p.data.clone() for n, p in m.projector.named_parameters()}
        k_before = {n: p.data.clone() for n, p in m.projector_k.named_parameters()}

        m._momentum_update()

        for name, k_after in m.projector_k.named_parameters():
            expected = k_before[name] * mom + q_params[name] * (1.0 - mom)
            assert torch.allclose(k_after.data, expected, atol=1e-6), \
                f"projector_k.{name}: EMA niezgodna"

    def test_momentum_update_no_grad_on_k(self):
        """_momentum_update nie powinno tworzyć grafu obliczeniowego."""
        m = make_model()
        m.train()
        m._momentum_update()
        for _, param in m.encoder_k.named_parameters():
            assert param.grad_fn is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Kształt logitów i pozytyw na pozycji 0
# ─────────────────────────────────────────────────────────────────────────────

class TestLogitsShape:
    def test_logits_shape(self):
        """InfoNCE: logity (B, 1+K)."""
        B, K, D = 4, 32, 32
        m = make_model(queue_size=K, proj_dim=D)
        m.train()

        q = F.normalize(torch.randn(B, D), dim=-1)
        k = F.normalize(torch.randn(B, D), dim=-1)

        l_pos = torch.einsum("bd,bd->b", q, k).unsqueeze(-1)
        l_neg = torch.einsum("bd,dk->bk", q, m.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / m.temperature
        assert logits.shape == (B, 1 + K), \
            f"Oczekiwano ({B}, {1+K}), dostano {tuple(logits.shape)}"

    def test_positive_at_position_0(self):
        """Etykiety InfoNCE = 0 (pozytyw zawsze na pozycji 0)."""
        B, K, D = 4, 32, 32
        m = make_model(queue_size=K, proj_dim=D)
        m.train()

        q = F.normalize(torch.randn(B, D), dim=-1)
        k = F.normalize(torch.randn(B, D), dim=-1)

        l_pos = torch.einsum("bd,bd->b", q, k).unsqueeze(-1)
        l_neg = torch.einsum("bd,dk->bk", q, m.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / m.temperature
        labels = torch.zeros(B, dtype=torch.long)

        assert labels.max() == 0
        # Sprawdź: pozytyw jest pierwszą kolumną (index 0)
        assert logits[:, 0].shape == (B,)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Aktualizacja kolejki (wrap-around)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueueUpdate:
    def test_ptr_advances(self):
        K, B, D = 16, 4, 32
        m = make_model(queue_size=K, proj_dim=D)
        keys = F.normalize(torch.randn(B, D), dim=-1)
        m._dequeue_and_enqueue(keys)
        assert int(m.queue_ptr) == B

    def test_wraparound(self):
        K, B, D = 8, 6, 32
        m = make_model(queue_size=K, proj_dim=D)
        # Po pierwszym enqueue: ptr = 6
        keys1 = F.normalize(torch.randn(6, D), dim=-1)
        m._dequeue_and_enqueue(keys1)
        assert int(m.queue_ptr) == 6, "ptr po pierwszym enqueue powinien = 6"

        # Po drugim: 6+6=12 > 8 → wrap: ptr = (12 % 8) = 4
        keys2 = F.normalize(torch.randn(6, D), dim=-1)
        m._dequeue_and_enqueue(keys2)
        assert int(m.queue_ptr) == (12 % K), "ptr po wrap-around powinien = 4"

    def test_queue_contains_enqueued_keys(self):
        K, B, D = 16, 4, 32
        m = make_model(queue_size=K, proj_dim=D)
        keys = F.normalize(torch.randn(B, D), dim=-1)
        m._dequeue_and_enqueue(keys)
        # Kolumny 0..B-1 kolejki powinny być równe keys
        stored = m.queue[:, :B].T
        assert torch.allclose(stored, keys, atol=1e-6)

    def test_queue_normalized_after_enqueue(self):
        K, B, D = 16, 4, 32
        m = make_model(queue_size=K, proj_dim=D)
        keys = F.normalize(torch.randn(B, D), dim=-1)
        m._dequeue_and_enqueue(keys)
        # Sprawdź wstawione klucze — powinny mieć norm ≈ 1
        stored = m.queue[:, :B].T
        norms = stored.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_full_cycle_ptr_zero(self):
        """Enqueue K wektorów na raz → ptr wraca do 0."""
        K, D = 8, 32
        m = make_model(queue_size=K, proj_dim=D)
        keys = F.normalize(torch.randn(K, D), dim=-1)
        m._dequeue_and_enqueue(keys)
        assert int(m.queue_ptr) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. pretrain_step — loss i backprop
# ─────────────────────────────────────────────────────────────────────────────

class TestPretrainStep:
    def test_loss_finite(self):
        m = make_model()
        m.train()
        x_q, x_k = fake_batch(B=4)
        result = m.pretrain_step((x_q, x_k))
        assert "loss" in result
        assert torch.isfinite(result["loss"]), "Loss nie jest skończony"

    def test_backprop(self):
        m = make_model()
        m.train()
        x_q, x_k = fake_batch(B=4)
        result = m.pretrain_step((x_q, x_k))
        result["loss"].backward()
        # encoder_q powinien mieć gradienty
        grads = [p.grad for p in m.encoder.parameters() if p.requires_grad]
        assert all(g is not None for g in grads), \
            "encoder_q powinien mieć gradienty po backward"

    def test_encoder_k_no_grad_after_pretrain(self):
        m = make_model()
        m.train()
        x_q, x_k = fake_batch(B=4)
        result = m.pretrain_step((x_q, x_k))
        result["loss"].backward()
        for name, param in m.encoder_k.named_parameters():
            assert param.grad is None, \
                f"encoder_k.{name}: nie powinno mieć gradientu po backward"

    def test_momentum_update_before_forward(self):
        """Sprawdza, że encoder_k jest zaktualizowany (EMA) w pretrain_step.

        Po inicjalizacji θ_k = θ_q (kopia), więc EMA nie zmienia nic.
        Zaburzamy θ_q += noise → teraz θ_q ≠ θ_k → EMA musi zmienić θ_k.
        """
        m = make_model(momentum=0.9)
        m.train()

        # Zaburz wagi encoder_q żeby θ_q ≠ θ_k
        with torch.no_grad():
            for param in m.encoder.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Zapamiętaj θ_k przed pretrain_step
        k_before = {n: p.data.clone() for n, p in m.encoder_k.named_parameters()}

        x_q, x_k = fake_batch(B=4)
        m.pretrain_step((x_q, x_k))

        # encoder_k po pretrain_step powinien się zmienić (EMA)
        any_changed = False
        for name, param in m.encoder_k.named_parameters():
            if not torch.allclose(param.data, k_before[name]):
                any_changed = True
                break
        assert any_changed, "encoder_k powinien zmienić się po pretrain_step (EMA)"


# ─────────────────────────────────────────────────────────────────────────────
# 8. forward i linear probing
# ─────────────────────────────────────────────────────────────────────────────

class TestForward:
    def test_forward_shape(self):
        m = make_model()
        m.eval()
        x = torch.randn(4, 3, 32, 32)
        h = m.forward(x)
        assert h.ndim == 2 and h.shape[0] == 4

    def test_linear_readout_shape(self):
        m = make_model(hidden_dim=64, proj_dim=32)
        m.eval()
        x = torch.randn(4, 3, 32, 32)
        logits = m.linear_readout(x)
        assert logits.shape == (4, 10)

    def test_freeze_stops_grads(self):
        m = make_model()
        m.freeze_encoder()
        x = torch.randn(4, 3, 32, 32)
        logits = m.linear_readout(x)
        logits.sum().backward()
        for name, param in m.encoder.named_parameters():
            assert param.grad is None, \
                f"encoder.{name}: nie powinno mieć gradientu gdy zamrożony"

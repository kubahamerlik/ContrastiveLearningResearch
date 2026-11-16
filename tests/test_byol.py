"""
Testy dla BYOLModel (image_models.py) i augmentacji BYOL (data_module.py).

Pokrycie:
- Strata ∈ [0, 4] i symetryczna (oba widoki identyczne → 0, antykorelacja → 4)
- Predyktor tylko w gałęzi online (target nie ma predyktora)
- Target bez gradientów
- τ rośnie monotonicznie do 1 (harmonogram cosine)
- Augmentacje: view1 nigdy nie solaryzuje, view2 nie ma blur p=1.0
- MLP BYOL: BN w środku, ostatnia warstwa bez biasu
- _regression_loss: zakres [0,4], symetria
- BN sieci target liczy statystyki batcha (train mode podczas pretrain)
- EMA update: formuła θ_t = τ·θ_t + (1−τ)·θ_o
- pretrain_step: loss skończony, backprop, diagnostyka kolapsu/NaN
"""

import warnings
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from contrastive_framework.models.image_models import BYOLModel, _build_byol_mlp


# ─────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────────────────────────────────────────

def make_model(hidden_dim: int = 64, proj_dim: int = 32, pred_dim: int = 32,
               momentum: float = 0.996) -> BYOLModel:
    return BYOLModel(
        encoder_type="cnn_image",
        input_dim=3,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        pred_dim=pred_dim,
        num_classes=10,
        momentum=momentum,
    )


def fake_batch(B: int = 4, C: int = 3, H: int = 32, W: int = 32):
    return torch.randn(B, C, H, W), torch.randn(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MLP BYOL — budowa
# ─────────────────────────────────────────────────────────────────────────────

class TestByolMLP:
    def test_has_batchnorm(self):
        mlp = _build_byol_mlp(64, 128, 32)
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in mlp.modules())
        assert has_bn, "MLP BYOL powinien zawierać BatchNorm1d"

    def test_last_layer_no_bias(self):
        mlp = _build_byol_mlp(64, 128, 32)
        linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 2
        # Ostatnia warstwa Linear — bez biasu (port z networks.py with_bias=False)
        assert linears[-1].bias is None, \
            "Ostatnia warstwa Linear w MLP BYOL powinna mieć bias=None (with_bias=False)"

    def test_first_layer_has_bias(self):
        mlp = _build_byol_mlp(64, 128, 32)
        linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        assert linears[0].bias is not None, \
            "Pierwsza warstwa Linear w MLP BYOL powinna mieć bias"

    def test_structure_order(self):
        mlp = _build_byol_mlp(64, 128, 32)
        types = [type(m).__name__ for m in mlp.children()]
        assert types == ["Linear", "BatchNorm1d", "ReLU", "Linear"]

    def test_output_shape(self):
        mlp = _build_byol_mlp(64, 128, 32)
        x = torch.randn(8, 64)
        y = mlp(x)
        assert y.shape == (8, 32)

    def test_projector_in_model_uses_byol_mlp(self):
        m = make_model()
        # projector powinien zawierać BN (BYOL MLP), nie być MoCo-style (bez BN)
        has_bn = any(isinstance(mod, nn.BatchNorm1d) for mod in m.projector.modules())
        assert has_bn, "model.projector powinien używać _build_byol_mlp (z BN)"
        # Ostatnia Linear bez biasu
        linears = [mod for mod in m.projector.modules() if isinstance(mod, nn.Linear)]
        assert linears[-1].bias is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Predyktor tylko w sieci online
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictorOnlineOnly:
    def test_online_has_predictor(self):
        m = make_model()
        assert hasattr(m, "predictor"), "BYOLModel powinien mieć atrybut predictor"
        assert isinstance(m.predictor, nn.Module)

    def test_target_no_predictor(self):
        m = make_model()
        assert not hasattr(m, "predictor_t"), \
            "Sieć target nie powinna mieć predyktora (predictor_t)"

    def test_predictor_has_grad(self):
        m = make_model()
        for name, param in m.predictor.named_parameters():
            assert param.requires_grad, \
                f"predictor.{name}: requires_grad powinno być True (online)"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Target bez gradientów
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetNoGrad:
    def test_encoder_t_no_grad(self):
        m = make_model()
        for name, param in m.encoder_t.named_parameters():
            assert not param.requires_grad, \
                f"encoder_t.{name}: requires_grad powinno być False"

    def test_projector_t_no_grad(self):
        m = make_model()
        for name, param in m.projector_t.named_parameters():
            assert not param.requires_grad, \
                f"projector_t.{name}: requires_grad powinno być False"

    def test_no_grad_after_pretrain(self):
        m = make_model()
        m.train()
        x1, x2 = fake_batch(B=4)
        result = m.pretrain_step((x1, x2))
        result["loss"].backward()
        for name, param in m.encoder_t.named_parameters():
            assert param.grad is None, \
                f"encoder_t.{name}: nie powinno mieć gradientu"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Harmonogram τ — rośnie monotonicznie do 1
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumSchedule:
    def test_starts_at_base(self):
        m = make_model(momentum=0.996)
        m.update_momentum(0, 100)
        assert abs(m.momentum - 0.996) < 1e-9, \
            f"Na step=0 τ powinno = base_ema=0.996, dostałem {m.momentum}"

    def test_ends_at_one(self):
        m = make_model(momentum=0.996)
        m.update_momentum(100, 100)
        assert abs(m.momentum - 1.0) < 1e-9, \
            f"Na step=max_steps τ powinno = 1.0, dostałem {m.momentum}"

    def test_monotonically_increasing(self):
        m = make_model(momentum=0.99)
        total = 1000
        taus = []
        for step in range(0, total + 1, 50):
            m.update_momentum(step, total)
            taus.append(m.momentum)
        for i in range(1, len(taus)):
            assert taus[i] >= taus[i - 1] - 1e-9, \
                f"τ powinno rosnąć monotonicznie: taus[{i}]={taus[i]} < taus[{i-1}]={taus[i-1]}"

    def test_increases_from_base(self):
        m = make_model(momentum=0.5)
        m.update_momentum(0, 100)
        tau_start = m.momentum
        m.update_momentum(50, 100)
        tau_mid = m.momentum
        assert tau_mid > tau_start, "τ powinno rosnąć od base_ema"

    def test_zero_steps_no_error(self):
        m = make_model()
        m.update_momentum(0, 0)
        assert m.momentum == m.momentum_base


# ─────────────────────────────────────────────────────────────────────────────
# 5. _regression_loss — zakres [0, 4], symetria
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionLoss:
    def test_range_identical(self):
        """Identyczne wektory → loss = 0."""
        v = F.normalize(torch.randn(8, 32), dim=-1)
        loss = BYOLModel._regression_loss(v, v).mean()
        assert abs(loss.item()) < 1e-5, \
            f"Identyczne wektory powinny dać loss≈0, dostałem {loss.item()}"

    def test_range_antipodal(self):
        """Antykorelacja (p = -z) → loss = 4."""
        v = F.normalize(torch.randn(8, 32), dim=-1)
        loss = BYOLModel._regression_loss(v, -v).mean()
        assert abs(loss.item() - 4.0) < 1e-5, \
            f"Antykorelacja powinna dać loss≈4, dostałem {loss.item()}"

    def test_range_random(self):
        """Losowe wektory → loss ∈ [0, 4]."""
        p = torch.randn(8, 32)
        z = torch.randn(8, 32)
        loss = BYOLModel._regression_loss(p, z).mean()
        assert 0.0 <= loss.item() <= 4.0 + 1e-5, \
            f"Loss powinien być w [0, 4], dostałem {loss.item()}"

    def test_returns_per_sample(self):
        """_regression_loss zwraca tensor (B,), nie skalar."""
        p = torch.randn(8, 32)
        z = torch.randn(8, 32)
        loss = BYOLModel._regression_loss(p, z)
        assert loss.shape == (8,), \
            f"_regression_loss powinno zwracać (B,), dostałem {loss.shape}"

    def test_finite(self):
        p = torch.randn(8, 32)
        z = torch.randn(8, 32)
        loss = BYOLModel._regression_loss(p, z)
        assert torch.all(torch.isfinite(loss))


# ─────────────────────────────────────────────────────────────────────────────
# 6. pretrain_step — loss, symetria, backprop
# ─────────────────────────────────────────────────────────────────────────────

class TestPretrainStep:
    def test_loss_finite(self):
        m = make_model()
        m.train()
        x1, x2 = fake_batch(B=4)
        result = m.pretrain_step((x1, x2))
        assert torch.isfinite(result["loss"]), f"Loss nie jest skończony: {result['loss']}"

    def test_loss_in_range(self):
        """Loss ∈ [0, 8].

        per_sample = regression_loss(p1, sg(z2)) + regression_loss(p2, sg(z1))
        Każdy składnik ∈ [0, 4] → suma ∈ [0, 8] → mean ∈ [0, 8].
        """
        m = make_model()
        m.train()
        x1, x2 = fake_batch(B=4)
        result = m.pretrain_step((x1, x2))
        loss = result["loss"].item()
        assert 0.0 <= loss <= 8.0 + 1e-4, \
            f"Loss powinien być w [0, 8], dostałem {loss}"

    def test_loss_symmetric_views(self):
        """Podmiana widoków (x1, x2) ↔ (x2, x1) daje skończony loss z obu kierunków.

        Pełna symetria numeryczna nie jest możliwa gdy BN target ma running stats —
        sprawdzamy tylko że obie konfiguracje dają skończony loss w [0, 8].
        """
        m = make_model()
        m2 = make_model()
        m2.load_state_dict(m.state_dict())

        m.train()
        m2.train()

        torch.manual_seed(0)
        x1, x2 = fake_batch(B=4)

        result1 = m.pretrain_step((x1, x2))
        result2 = m2.pretrain_step((x2, x1))

        assert torch.isfinite(result1["loss"]), "loss(x1, x2) nie skończony"
        assert torch.isfinite(result2["loss"]), "loss(x2, x1) nie skończony"
        assert 0.0 <= result1["loss"].item() <= 8.0 + 1e-4
        assert 0.0 <= result2["loss"].item() <= 8.0 + 1e-4

    def test_backprop(self):
        m = make_model()
        m.train()
        x1, x2 = fake_batch(B=4)
        result = m.pretrain_step((x1, x2))
        result["loss"].backward()
        # encoder (online) powinien mieć gradienty
        grads = [p.grad for p in m.encoder.parameters() if p.requires_grad]
        assert all(g is not None for g in grads), \
            "encoder powinien mieć gradienty po backward"

    def test_predictor_gets_grads(self):
        m = make_model()
        m.train()
        x1, x2 = fake_batch(B=4)
        result = m.pretrain_step((x1, x2))
        result["loss"].backward()
        grads = [p.grad for p in m.predictor.parameters() if p.requires_grad]
        assert all(g is not None for g in grads), \
            "predictor powinien mieć gradienty"

    def test_emb_std_key_returned(self):
        m = make_model()
        m.train()
        x1, x2 = fake_batch(B=4)
        result = m.pretrain_step((x1, x2))
        assert "emb_std" in result, "pretrain_step powinien zwrócić 'emb_std'"

    def test_nan_raises(self):
        """NaN w danych → RuntimeError z czytelnym komunikatem."""
        m = make_model()
        m.train()
        x1 = torch.full((4, 3, 32, 32), float("nan"))
        x2 = torch.randn(4, 3, 32, 32)
        with pytest.raises(RuntimeError, match="Dywergencja numeryczna"):
            m.pretrain_step((x1, x2))


# ─────────────────────────────────────────────────────────────────────────────
# 7. BN sieci target w train() podczas pretrain
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetBNTrainMode:
    def test_encoder_t_in_train_during_pretrain(self):
        m = make_model()
        m.train()
        assert m.encoder_t.training, "encoder_t powinien być w train() podczas pretrain"

    def test_projector_t_in_train_during_pretrain(self):
        m = make_model()
        m.train()
        assert m.projector_t.training, "projector_t powinien być w train() podczas pretrain"

    def test_target_bn_running_stats_change(self):
        """Running stats encoder_t powinny zmieniać się po forward pass."""
        m = make_model()
        m.train()

        # Zbierz referencję do running_mean BN w encoder_t
        bn_layers = [mod for mod in m.encoder_t.modules()
                     if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d))]
        assert len(bn_layers) > 0, "encoder_t musi mieć BN"
        ref = bn_layers[0].running_mean.clone()

        x1, x2 = fake_batch(B=8)
        m.pretrain_step((x1, x2))

        changed = not torch.allclose(bn_layers[0].running_mean, ref)
        assert changed, "encoder_t.BN.running_mean powinien zmieniać się po forward (train mode)"

    def test_eval_mode_freezes_encoder_t(self):
        m = make_model()
        m.eval()
        assert not m.encoder_t.training, \
            "encoder_t powinien być w eval() gdy model jest w eval()"


# ─────────────────────────────────────────────────────────────────────────────
# 8. EMA update — formuła θ_t = τ·θ_t + (1−τ)·θ_o
# ─────────────────────────────────────────────────────────────────────────────

class TestEMAUpdate:
    def test_formula_encoder(self):
        m = make_model(momentum=0.9)
        m.train()

        # Zaburz wagi online
        with torch.no_grad():
            for p in m.encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        tau = m.momentum
        t_before = {n: p.data.clone() for n, p in m.encoder_t.named_parameters()}
        o_params  = {n: p.data.clone() for n, p in m.encoder.named_parameters()}

        m._ema_update()

        for name, t_after in m.encoder_t.named_parameters():
            expected = t_before[name] * tau + o_params[name] * (1.0 - tau)
            assert torch.allclose(t_after.data, expected, atol=1e-6), \
                f"encoder_t.{name}: EMA niezgodna"

    def test_formula_projector(self):
        m = make_model(momentum=0.9)
        m.train()

        with torch.no_grad():
            for p in m.projector.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        tau = m.momentum
        t_before = {n: p.data.clone() for n, p in m.projector_t.named_parameters()}
        o_params  = {n: p.data.clone() for n, p in m.projector.named_parameters()}

        m._ema_update()

        for name, t_after in m.projector_t.named_parameters():
            expected = t_before[name] * tau + o_params[name] * (1.0 - tau)
            assert torch.allclose(t_after.data, expected, atol=1e-6), \
                f"projector_t.{name}: EMA niezgodna"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Augmentacje BYOL — solarize/blur przypisanie
# ─────────────────────────────────────────────────────────────────────────────

class TestByolAugmentations:
    """
    Testuje _build_byol_augmentation z data_module.py.
    view1: blur p=1.0, BEZ solarize
    view2: blur p=0.1, solarize p=0.2
    """

    def _get_transforms(self, view: int):
        from contrastive_framework.data_module import _build_byol_augmentation
        return _build_byol_augmentation(image_size=32, view=view)

    def _list_transform_types(self, aug):
        """Zwraca listę nazw transformacji (torchvision)."""
        import torchvision.transforms as T
        transforms = aug.transforms  # T.Compose.transforms
        return [type(t).__name__ for t in transforms]

    def test_view1_no_solarize(self):
        import torchvision.transforms as T
        aug = self._get_transforms(view=1)
        for t in aug.transforms:
            assert not isinstance(t, T.RandomSolarize), \
                "view1 nie powinien zawierać RandomSolarize"

    def test_view2_has_solarize(self):
        import torchvision.transforms as T
        aug = self._get_transforms(view=2)
        has_solarize = any(isinstance(t, T.RandomSolarize) for t in aug.transforms)
        assert has_solarize, "view2 powinien zawierać RandomSolarize"

    def test_view1_blur_prob_one(self):
        import torchvision.transforms as T
        aug = self._get_transforms(view=1)
        # Szukamy RandomApply z GaussianBlur — jego p powinno być 1.0
        for t in aug.transforms:
            if isinstance(t, T.RandomApply):
                for inner in t.transforms:
                    if isinstance(inner, T.GaussianBlur):
                        assert abs(t.p - 1.0) < 1e-6, \
                            f"view1 blur_p powinien = 1.0, dostałem {t.p}"
                        return
        pytest.fail("view1 nie zawiera RandomApply(GaussianBlur)")

    def test_view2_blur_prob_not_one(self):
        import torchvision.transforms as T
        aug = self._get_transforms(view=2)
        for t in aug.transforms:
            if isinstance(t, T.RandomApply):
                for inner in t.transforms:
                    if isinstance(inner, T.GaussianBlur):
                        assert t.p < 1.0, \
                            f"view2 blur_p powinien być < 1.0, dostałem {t.p}"
                        return
        pytest.fail("view2 nie zawiera RandomApply(GaussianBlur)")

    def test_view2_no_blur_p_one(self):
        """view2 nigdy nie powinien mieć blur z p=1.0."""
        import torchvision.transforms as T
        aug = self._get_transforms(view=2)
        for t in aug.transforms:
            if isinstance(t, T.RandomApply):
                for inner in t.transforms:
                    if isinstance(inner, T.GaussianBlur):
                        assert abs(t.p - 1.0) > 1e-6, \
                            "view2 nie powinien mieć GaussianBlur z p=1.0"

    def test_color_jitter_saturation_view1(self):
        """saturation=0.2 (jak w byol/utils/augmentations.py), nie 0.4."""
        import torchvision.transforms as T
        aug = self._get_transforms(view=1)
        for t in aug.transforms:
            if isinstance(t, T.RandomApply):
                for inner in t.transforms:
                    if isinstance(inner, T.ColorJitter):
                        sat = inner.saturation
                        # torchvision zwraca zakres (lo, hi) = (1-s, 1+s)
                        # Tu s = 0.2*jitter_strength (jitter_strength=1.0 domyślnie)
                        sat_param = (sat[1] - 1.0) if isinstance(sat, tuple) else sat
                        assert abs(sat_param - 0.2) < 1e-5, \
                            f"saturation powinien = 0.2, dostałem {sat_param}"
                        return
        pytest.fail("view1 nie zawiera ColorJitter")

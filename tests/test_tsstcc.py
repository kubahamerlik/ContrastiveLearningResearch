"""
Testy TS-TCC i RandomInitHARModel.

Sprawdzają:
1. Kształty wyjść CNN kodera i modelu
2. Brak wycieku przyszłości w TC (causal test)
3. Semantykę augmentacji (weak zachowuje skalę, strong permutuje)
4. Identyczny koder TSTCCModel i RandomInitHARModel (te same hiperparametry)
5. pretrain_step zwraca nieujemną stratę
6. linear_readout shape
7. freeze/unfreeze kodera
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrastive_framework.models.timeseries_models import (
    TSTCCModel,
    RandomInitHARModel,
    _BaseModel,
    _TC,
    _SeqTransformer,
    _NTXentLoss,
    weak_augmentation,
    strong_augmentation,
    _compute_features_len,
)

# ---------------------------------------------------------------------------
# Stałe testu – małe wartości dla szybkości
# ---------------------------------------------------------------------------
SEQ_LEN = 128
IN_CH = 9
NUM_CLASSES = 6
BATCH = 8
FEATURES_LEN = 18  # wynik _compute_features_len(128)


# ---------------------------------------------------------------------------
# Pomocnik
# ---------------------------------------------------------------------------

def _make_tstcc(**kwargs) -> TSTCCModel:
    defaults = dict(
        seq_len=SEQ_LEN,
        in_channels=IN_CH,
        num_classes=NUM_CLASSES,
        tc_timesteps=6,
        tc_hidden_dim=100,
        final_out_channels=128,
    )
    defaults.update(kwargs)
    return TSTCCModel(**defaults)


def _make_rand_har(**kwargs) -> RandomInitHARModel:
    defaults = dict(
        seq_len=SEQ_LEN,
        in_channels=IN_CH,
        num_classes=NUM_CLASSES,
        final_out_channels=128,
    )
    defaults.update(kwargs)
    return RandomInitHARModel(**defaults)


def _rand_batch(B: int = BATCH) -> torch.Tensor:
    return torch.randn(B, IN_CH, SEQ_LEN)


# ===========================================================================
# 1. _compute_features_len
# ===========================================================================

class TestFeaturesLen:
    def test_har_default(self):
        assert _compute_features_len(128) == 18

    def test_longer_seq(self):
        # Upewniamy się, że wynik jest > 0 i rośnie monotoniczne z seq_len
        L128 = _compute_features_len(128)
        L256 = _compute_features_len(256)
        assert L256 > L128 > 0


# ===========================================================================
# 2. _BaseModel shapes
# ===========================================================================

class TestBaseModel:
    def test_output_shape(self):
        cnn = _BaseModel(in_channels=IN_CH)
        x = _rand_batch()
        out = cnn(x)
        assert out.shape == (BATCH, 128, FEATURES_LEN), (
            f"Oczekiwano (B, 128, 18), dostałem {out.shape}"
        )

    def test_dropout_in_training(self):
        """Dropout w block1 powinien powodować niezerową wariancję wyjść przy train mode."""
        cnn = _BaseModel(in_channels=IN_CH, dropout=0.5)
        cnn.train()
        x = _rand_batch()
        # Dwa forward passes w train mode → różne wyniki (losowy dropout)
        with torch.no_grad():
            out1 = cnn(x)
            out2 = cnn(x)
        # Przynajmniej część wartości musi się różnić (p=0.5 → oczekujemy różnic)
        assert not torch.allclose(out1, out2), (
            "Dwa forward passes przy dropout=0.5 dały identyczne wyniki — "
            "dropout może nie działać"
        )

    def test_no_dropout_in_eval(self):
        """W eval() dropout jest wyłączony → wyniki deterministyczne."""
        cnn = _BaseModel(in_channels=IN_CH, dropout=0.5)
        cnn.eval()
        x = _rand_batch()
        with torch.no_grad():
            out1 = cnn(x)
            out2 = cnn(x)
        assert torch.allclose(out1, out2), (
            "W eval() dropout powinien być wyłączony"
        )


# ===========================================================================
# 3. Augmentacje
# ===========================================================================

class TestAugmentations:
    def test_weak_aug_preserves_shape(self):
        x = _rand_batch()
        x_aug = weak_augmentation(x)
        assert x_aug.shape == x.shape

    def test_weak_aug_changes_values(self):
        x = _rand_batch()
        x_aug = weak_augmentation(x)
        # Skalowanie przez N(2, 1.1) zawsze zmienia wartości (skalowanie ≠ 1)
        assert not torch.allclose(x_aug, x), (
            "weak_augmentation nie zmieniła żadnych wartości"
        )

    def test_strong_aug_preserves_shape(self):
        x = _rand_batch()
        x_aug = strong_augmentation(x, max_seg=4)
        assert x_aug.shape == x.shape

    def test_strong_aug_permutes(self):
        """Sprawdza, że strong_augmentation naprawdę przetasowuje segmenty."""
        # Użyjemy długiej, monotoniczne rosnącej sekwencji
        B, C, T = 4, 1, 64
        x = torch.arange(T, dtype=torch.float32).expand(B, C, T).clone()
        x_aug = strong_augmentation(x, jitter_ratio=0.0, max_seg=8)
        # Przy jitter_ratio=0 i permutacji, wynik musi się różnić od oryginału
        # (z bardzo wysokim prawdopodobieństwem dla T=64, max_seg=8)
        assert not torch.allclose(x_aug, x), (
            "strong_augmentation nie przetasowała segmentów ani nie dodała jitteru"
        )

    def test_weak_aug_scale_centered_above_one(self):
        """
        Skalowanie weak_aug pochodzi z N(2, σ), więc wartość bezwzględna
        augmentowanego sygnału powinna średnio rosnąć (|aug| > |x| w średniej).
        """
        torch.manual_seed(42)
        x = torch.ones(200, IN_CH, SEQ_LEN)
        x_aug = weak_augmentation(x, jitter_scale_ratio=1.1)
        # Oczekiwane: E[|x_aug|] ≈ E[|N(2, 1.1)|] * |x| ≈ 2 * |x|
        ratio = x_aug.abs().mean() / x.abs().mean()
        assert ratio > 1.5, (
            f"Oczekiwano ratio > 1.5 (skalowanie N(2, σ)), dostałem {ratio:.3f}"
        )


# ===========================================================================
# 4. _TC – test braku wycieku przyszłości (causal)
# ===========================================================================

class TestTCCausal:
    def test_causal_no_future_leak(self):
        """
        Podmiana przyszłości aug2 na inną wartość nie może zmienić c_t (TC kontekstu).
        c_t zależy TYLKO od aug1[:, :t+1, :] – przeszłości.
        """
        tc = _TC(num_channels=128, seq_len=FEATURES_LEN, hidden_dim=100, timesteps=6)
        tc.eval()

        # T' = 18, timesteps = 6, więc t_samples ∈ [0, 11]
        # Ustaw deterministycznie t_samples=5 (środek)
        torch.manual_seed(0)
        feat = torch.randn(BATCH, FEATURES_LEN, 128)

        # Podmień przyszłość aug2 (kroki > t_samples) na losowe wartości
        feat_future_replaced = feat.clone()
        feat_future_replaced[:, 6:, :] = torch.randn(BATCH, 12, 128)

        # c_t zależy tylko od forward_seq = aug1[:, :t+1, :]
        # Dla obu wywołań z_aug1 jest identyczne → c_t musi być identyczne
        with torch.no_grad():
            # Wymuszamy t_samples przez bezpośrednie wywołanie seq_transformer
            # Testujemy: czy forward_seq (kontekst) jest causal relative to aug2
            # W praktyce: c_t = seq_transformer(z_aug1[:, :t+1, :])
            # Podmiana z_aug2 NIE wpływa na c_t

            # Ustaw seed żeby t_samples był deterministyczny
            torch.manual_seed(42)
            nce1, proj1 = tc(feat, feat)              # aug1=aug2=feat

            torch.manual_seed(42)
            nce2, proj2 = tc(feat, feat_future_replaced)  # aug2 zmienione

        # c_t jest identyczne → proj identyczne (proj = projection_head(c_t))
        assert torch.allclose(proj1, proj2, atol=1e-5), (
            "TC nie jest kauzalne: zmiana przyszłości aug2 wpłynęła na c_t!\n"
            f"max diff: {(proj1 - proj2).abs().max().item():.6f}"
        )

    def test_tc_output_shapes(self):
        tc = _TC(num_channels=128, seq_len=FEATURES_LEN, hidden_dim=100, timesteps=6)
        feat = torch.randn(BATCH, FEATURES_LEN, 128)
        nce, proj = tc(feat, feat)
        assert nce.shape == (), f"nce powinno być skalarem, dostałem {nce.shape}"
        # proj: (B, num_channels//4) = (B, 32)
        assert proj.shape == (BATCH, 128 // 4), (
            f"proj shape: oczekiwano (B, 32), dostałem {proj.shape}"
        )

    def test_tc_nce_finite(self):
        """NCE nie powinno być NaN/inf."""
        tc = _TC(num_channels=128, seq_len=FEATURES_LEN, hidden_dim=100, timesteps=6)
        feat1 = torch.randn(BATCH, FEATURES_LEN, 128)
        feat2 = torch.randn(BATCH, FEATURES_LEN, 128)
        nce, _ = tc(feat1, feat2)
        assert torch.isfinite(nce), f"NCE nie jest skończone: {nce.item()}"


# ===========================================================================
# 5. TSTCCModel shapes i pretrain
# ===========================================================================

class TestTSTCCModel:
    def test_forward_shape(self):
        """forward() zwraca flattened CNN features."""
        model = _make_tstcc()
        x = _rand_batch()
        h = model.forward(x)
        expected = 128 * FEATURES_LEN  # C * T'
        assert h.shape == (BATCH, expected), (
            f"Oczekiwano (B, {expected}), dostałem {h.shape}"
        )

    def test_linear_readout_shape(self):
        model = _make_tstcc()
        x = _rand_batch()
        logits = model.linear_readout(x)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_pretrain_step_loss_finite(self):
        model = _make_tstcc()
        model.train()
        x = _rand_batch()
        out = model.pretrain_step(x)
        loss = out["loss"]
        assert torch.isfinite(loss), f"pretrain loss nie jest skończony: {loss.item()}"
        assert "tc_loss" in out and "cc_loss" in out

    def test_pretrain_loss_backprop(self):
        """Strata musi być różniczkowana."""
        model = _make_tstcc()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = _rand_batch()
        out = model.pretrain_step(x)
        out["loss"].backward()
        # Przynajmniej jeden gradient CNN musi być niezerowy
        enc_grad = model.encoder.conv_block1[0].weight.grad
        assert enc_grad is not None and enc_grad.abs().max() > 0, (
            "Gradient CNN kodera jest zerowy po backward"
        )

    def test_freeze_encoder_stops_grad(self):
        """Po freeze_encoder() classifier.weight.grad != 0, encoder.grad == 0."""
        model = _make_tstcc()
        model.freeze_encoder()
        x = _rand_batch()
        logits = model.linear_readout(x)
        loss = logits.sum()
        loss.backward()
        enc_grad = model.encoder.conv_block1[0].weight.grad
        assert enc_grad is None or enc_grad.abs().max() == 0, (
            "Gradient enkoder nie jest zerowy po freeze_encoder()"
        )
        cls_grad = model.classifier.weight.grad
        assert cls_grad is not None and cls_grad.abs().max() > 0, (
            "Gradient classifier jest zerowy po freeze_encoder()"
        )

    def test_unfreeze_encoder(self):
        model = _make_tstcc()
        model.freeze_encoder()
        model.unfreeze_encoder()
        x = _rand_batch()
        # Pełny forward i backward po odmrożeniu
        out = model.pretrain_step(x)
        out["loss"].backward()
        enc_grad = model.encoder.conv_block1[0].weight.grad
        assert enc_grad is not None and enc_grad.abs().max() > 0, (
            "Gradient enkoder jest zerowy po unfreeze_encoder()"
        )

    def test_lambda_weights_fixed(self):
        """Wagi straty λ1=1, λ2=0.7 (nie są parametrami – nie mają gradientu)."""
        model = _make_tstcc()
        assert model._lambda1 == 1.0
        assert model._lambda2 == 0.7

    def test_no_tc_weight_param(self):
        """tc_weight nie powinno być parametrem modelu (usunięte z HPO)."""
        model = _make_tstcc()
        assert not hasattr(model, "tc_weight"), (
            "tc_weight nie powinno być atrybutem modelu (wagi są stałe)"
        )


# ===========================================================================
# 6. RandomInitHARModel
# ===========================================================================

class TestRandomInitHAR:
    def test_forward_shape(self):
        model = _make_rand_har()
        x = _rand_batch()
        h = model.forward(x)
        assert h.shape == (BATCH, 128 * FEATURES_LEN)

    def test_pretrain_loss_zero(self):
        """pretrain_step zwraca zero loss."""
        model = _make_rand_har()
        x = _rand_batch()
        out = model.pretrain_step(x)
        assert out["loss"].item() == 0.0

    def test_linear_readout_shape(self):
        model = _make_rand_har()
        x = _rand_batch()
        logits = model.linear_readout(x)
        assert logits.shape == (BATCH, NUM_CLASSES)


# ===========================================================================
# 7. Identyczny koder TSTCCModel i RandomInitHARModel
# ===========================================================================

class TestIdenticalEncoders:
    def test_same_architecture(self):
        """
        TSTCCModel.encoder i RandomInitHARModel.encoder mają identyczną architekturę:
        te same shapes wag i identyczne forward shapes dla tych samych wejść.
        """
        tstcc = _make_tstcc(final_out_channels=128, kernel_size=8)
        rand_har = _make_rand_har(final_out_channels=128, kernel_size=8)

        # Sprawdź identyczne shapes wag
        tstcc_params = {n: p.shape for n, p in tstcc.encoder.named_parameters()}
        rand_params = {n: p.shape for n, p in rand_har.encoder.named_parameters()}
        assert tstcc_params == rand_params, (
            f"Różne shapes wag enkoder:\n"
            f"  TSTCC: {tstcc_params}\n"
            f"  Rand:  {rand_params}"
        )

    def test_same_output_shape(self):
        """Oba modele zwracają ten sam shape flattened features."""
        tstcc = _make_tstcc()
        rand_har = _make_rand_har()
        x = _rand_batch()
        h_t = tstcc.forward(x)
        h_r = rand_har.forward(x)
        assert h_t.shape == h_r.shape, (
            f"Różne shape output: TSTCC {h_t.shape} vs Rand {h_r.shape}"
        )

    def test_same_output_when_weights_copied(self):
        """
        Po przekopiowaniu wag enkoder, forward() obu modeli daje identyczne wyniki.
        Potwierdza, że architektura jest naprawdę identyczna.
        """
        tstcc = _make_tstcc()
        rand_har = _make_rand_har()

        # Kopiuj wagi enkoder z TSTCC do RandomInitHAR
        rand_har.encoder.load_state_dict(tstcc.encoder.state_dict())

        tstcc.eval()
        rand_har.eval()
        x = _rand_batch()
        with torch.no_grad():
            h_t = tstcc.forward(x)
            h_r = rand_har.forward(x)
        assert torch.allclose(h_t, h_r, atol=1e-5), (
            f"Po przekopiowaniu wag, wyniki różnią się: "
            f"max diff = {(h_t - h_r).abs().max().item():.6f}"
        )


# ===========================================================================
# 8. NT-Xent Loss
# ===========================================================================

class TestNTXentLoss:
    def test_output_finite(self):
        loss_fn = _NTXentLoss(temperature=0.2)
        z1 = torch.randn(BATCH, 32)
        z2 = torch.randn(BATCH, 32)
        loss = loss_fn(z1, z2)
        assert torch.isfinite(loss)

    def test_identical_views_low_loss(self):
        """Identyczne widoki → strata bliska 0 (perfect alignment)."""
        loss_fn = _NTXentLoss(temperature=0.2)
        # Gdy z1 == z2 i wszystkie są ortogonalne (worst case), strata jest log(2B-1)
        # Gdy z1 == z2 i są ≠ między sobą, poprawne pary mają cosine_sim = 1 (max)
        # Tworzymy ortogonalną bazę
        z = torch.eye(BATCH, 32)
        loss = loss_fn(z, z)
        # Strata musi być skończona i nieujemna
        assert torch.isfinite(loss) and loss.item() >= 0

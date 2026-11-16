"""
Testy SubTabModel i RandomInitSubTabArchModel.

Sprawdzają:
1. Jeden wspólny enkoder (liczba modułów enkodera AE = 1)
2. Podzbiory mają właściwą długość i zadany overlap
3. Maska szumu ma oczekiwany odsetek zamaskowanych elementów
4. Wszystkie trzy składniki straty (recon, contrastive, distance) są liczone
   i można je wyłączyć flagami
5. forward() jest deterministyczny w eval() (bez szumu)
6. Kształty wyjść
7. freeze/unfreeze kodera
8. RandomInitSubTabArch ma identyczną architekturę AE co SubTab
9. JointLoss: poprawne zachowanie przy wyłączonych flagach
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrastive_framework.models.tabular_models import (
    SubTabModel,
    RandomInitSubTabArchModel,
    _AEWrapper,
    _JointLoss,
)

# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------
NUM_FEATURES = 30
N_SUBSETS = 3
OVERLAP = 0.5    # 50% nakładania
BATCH = 8
NUM_CLASSES = 2


def _make_model(**kwargs) -> SubTabModel:
    defaults = dict(
        num_features=NUM_FEATURES,
        n_subsets=N_SUBSETS,
        overlap=OVERLAP,
        num_classes=NUM_CLASSES,
        hidden_dim=64,
        dims=[64, 32],
    )
    defaults.update(kwargs)
    return SubTabModel(**defaults)


def _rand_batch(B: int = BATCH) -> torch.Tensor:
    return torch.randn(B, NUM_FEATURES)


# ===========================================================================
# 1. Jeden wspólny enkoder
# ===========================================================================

class TestSingleSharedEncoder:
    def test_one_ae_module(self):
        """SubTabModel musi mieć dokładnie jeden AEWrapper (jeden wspólny AE)."""
        model = _make_model()
        ae_modules = [m for m in model.modules() if isinstance(m, _AEWrapper)]
        assert len(ae_modules) == 1, (
            f"Oczekiwano 1 AEWrapper, znalazłem {len(ae_modules)}"
        )

    def test_no_module_list_of_encoders(self):
        """Nie powinno być ModuleList wielu enkoderów (jak w starej implementacji)."""
        model = _make_model()
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList):
                # Upewniamy się, że to nie jest lista enkoderów
                for child in mod.children():
                    assert not isinstance(child, nn.Linear) or "ae" in name, (
                        f"Znaleziono podejrzaną ModuleList: {name}"
                    )

    def test_ae_encoder_is_shared(self):
        """model.encoder i model.ae.encoder to ten sam obiekt."""
        model = _make_model()
        assert model.encoder is model.ae.encoder, (
            "model.encoder powinien wskazywać na model.ae.encoder"
        )


# ===========================================================================
# 2. Wymiary podzbiorów i overlap
# ===========================================================================

class TestSubsetDimensions:
    def test_subset_dim_matches_formula(self):
        """
        subset_dim = n_col_subset + n_overlap
        gdzie n_col_subset = num_features // n_subsets
              n_overlap = int(overlap * n_col_subset)
        """
        for nf, ns, ov in [(30, 3, 0.5), (40, 4, 0.75), (20, 2, 0.0)]:
            n_col_subset = nf // ns
            n_overlap = int(ov * n_col_subset)
            expected_subset_dim = n_col_subset + n_overlap

            model = SubTabModel(num_features=nf, n_subsets=ns, overlap=ov,
                                dims=[32, 16])
            # Sprawdź wejście enkodera
            first_layer = list(model.ae.encoder.hidden.net.children())[0]
            actual_in = first_layer.in_features
            assert actual_in == expected_subset_dim, (
                f"nf={nf}, ns={ns}, ov={ov}: "
                f"oczekiwano in_features={expected_subset_dim}, dostałem {actual_in}"
            )

    def test_subset_indices_length(self):
        """Każdy podzbiór ma długość = n_col_subset + n_overlap."""
        model = _make_model()
        col_indices = model._subset_column_indices(mode="test")
        n_col_subset = NUM_FEATURES // N_SUBSETS
        n_overlap = int(OVERLAP * n_col_subset)
        expected_len = n_col_subset + n_overlap
        for i, idx in enumerate(col_indices):
            assert len(idx) == expected_len, (
                f"Podzbiór {i}: oczekiwano {expected_len} kolumn, dostałem {len(idx)}"
            )

    def test_overlap_between_adjacent_subsets(self):
        """
        Sprawdza, że podzbiory sąsiednie faktycznie się nakładają — zgodnie ze wzorem
        z SubTab/src/model.py (subset_generator).

        Wzór:
          i=0: [0, n_col+n_ov)
          i>0: [i*n_col - n_ov, (i+1)*n_col)
        Gdzie n_col = n_col_subset, n_ov = n_overlap.

        Faktyczny overlap:
          para (0,1): 2*n_overlap  (podzbiór 0 wychodzi n_ov poza granicę; 1 zaczyna n_ov przed)
          para (k,k+1) dla k>0: n_overlap (tylko lewe rozszerzenie podzbioru k+1)
        """
        model = _make_model(num_features=30, n_subsets=3, overlap=0.5)
        indices = model._subset_column_indices(mode="test")
        n_col_subset = 30 // 3     # = 10
        n_overlap = int(0.5 * 10)  # = 5

        # Para (0,1): 2*n_overlap
        shared_01 = set(indices[0]) & set(indices[1])
        assert len(shared_01) == 2 * n_overlap, (
            f"Para (0,1): oczekiwano {2*n_overlap} wspólnych, dostałem {len(shared_01)}"
        )
        # Para (1,2): n_overlap
        shared_12 = set(indices[1]) & set(indices[2])
        assert len(shared_12) == n_overlap, (
            f"Para (1,2): oczekiwano {n_overlap} wspólnych, dostałem {len(shared_12)}"
        )
        # Każda para musi mieć > 0 wspólnych (ogólny sanity check)
        for k in range(len(indices) - 1):
            shared = set(indices[k]) & set(indices[k + 1])
            assert len(shared) > 0, (
                f"Podzbiory {k} i {k+1} nie mają żadnych wspólnych kolumn"
            )

    def test_number_of_subsets(self):
        """_subset_column_indices zwraca n_subsets podzbiorów."""
        model = _make_model()
        indices = model._subset_column_indices(mode="test")
        assert len(indices) == N_SUBSETS, (
            f"Oczekiwano {N_SUBSETS} podzbiorów, dostałem {len(indices)}"
        )


# ===========================================================================
# 3. Maska szumu
# ===========================================================================

class TestNoiseMask:
    def test_masking_ratio_approx(self):
        """
        Odsetek zamaskowanych elementów musi być bliski masking_ratio.
        Testujemy statystycznie na dużym batchu.
        """
        masking_ratio = 0.3
        model = _make_model(masking_ratio=masking_ratio, noise_type="zero_out")

        torch.manual_seed(42)
        # Duży batch + wiele wywołań
        total_elements = 0
        total_changed = 0
        x_ref = torch.ones(200, 10)
        for _ in range(20):
            x_noisy = model._add_noise(x_ref.clone())
            total_changed += (x_noisy == 0.0).float().sum().item()
            total_elements += x_ref.numel()

        actual_ratio = total_changed / total_elements
        assert abs(actual_ratio - masking_ratio) < 0.05, (
            f"Oczekiwano masking_ratio≈{masking_ratio}, dostałem {actual_ratio:.4f}"
        )

    def test_swap_noise_only_on_masked(self):
        """
        swap_noise: wartości niezamaskowane muszą pozostać niezmienione.
        Sprawdzamy: przy masking_ratio=0 → xbar = x (bez zmian).
        """
        model = _make_model(masking_ratio=0.0, noise_type="swap_noise")
        x = torch.randn(16, 10)
        x_noisy = model._add_noise(x.clone())
        assert torch.allclose(x, x_noisy), (
            "masking_ratio=0: żaden element nie powinien zostać zmieniony"
        )

    def test_gaussian_noise_only_on_masked(self):
        """masking_ratio=0 → gaussian_noise nic nie zmienia."""
        model = _make_model(masking_ratio=0.0, noise_type="gaussian_noise",
                            noise_level=10.0)
        x = torch.randn(16, 10)
        x_noisy = model._add_noise(x.clone())
        assert torch.allclose(x, x_noisy), (
            "masking_ratio=0: gaussian_noise nie powinien niczego zmieniać"
        )

    def test_masking_ratio_one_all_changed(self):
        """masking_ratio=1.0 → wszystkie elementy przez zero_out → 0."""
        model = _make_model(masking_ratio=1.0, noise_type="zero_out")
        x = torch.ones(32, 10)
        x_noisy = model._add_noise(x)
        assert x_noisy.abs().max() == 0.0, (
            "masking_ratio=1.0 + zero_out: wszystkie elementy powinny być 0"
        )


# ===========================================================================
# 4. JointLoss – składniki i flagi
# ===========================================================================

class TestJointLoss:
    def _make_inputs(self, B: int = 8, D: int = 32, F: int = NUM_FEATURES):
        """Tworzy (z, x_recon, x_orig) dla testów JointLoss."""
        z = torch.randn(2 * B, D)
        x_recon = torch.randn(2 * B, F)
        x_orig = torch.randn(2 * B, F)
        return z, x_recon, x_orig

    def test_all_losses_computed(self):
        """Przy wszystkich flagach True, strata jest sumą trzech składników."""
        loss_fn = _JointLoss(tau=0.1, use_contrastive=True, use_distance=True,
                             use_reconstruction=True)
        z, x_recon, x_orig = self._make_inputs()
        total, closs, rloss, zloss = loss_fn(z, x_recon, x_orig)
        assert torch.isfinite(total)
        assert torch.isfinite(closs)
        assert torch.isfinite(rloss)
        assert torch.isfinite(zloss)

    def test_disable_contrastive(self):
        """use_contrastive=False: closs == rloss (placeholder, jak w oryginale)."""
        loss_fn_with = _JointLoss(tau=0.1, use_contrastive=True, use_distance=False)
        loss_fn_without = _JointLoss(tau=0.1, use_contrastive=False, use_distance=False)
        torch.manual_seed(0)
        z, x_recon, x_orig = self._make_inputs()
        total_with, closs_with, rloss_with, _ = loss_fn_with(z, x_recon, x_orig)
        torch.manual_seed(0)
        total_without, closs_without, rloss_without, _ = loss_fn_without(z, x_recon, x_orig)

        # Bez contrastive: total == rloss
        assert torch.allclose(total_without, rloss_without), (
            "use_contrastive=False: total powinien równać się rloss"
        )
        # Z contrastive: total > rloss (closs > 0)
        assert total_with > rloss_with, (
            "use_contrastive=True: total powinien być większy niż rloss"
        )

    def test_disable_distance(self):
        """use_distance=False: total nie zawiera zloss."""
        loss_fn_with = _JointLoss(tau=0.1, use_contrastive=False, use_distance=True)
        loss_fn_without = _JointLoss(tau=0.1, use_contrastive=False, use_distance=False)
        z, x_recon, x_orig = self._make_inputs()
        total_with, _, rloss_with, zloss_with = loss_fn_with(z, x_recon, x_orig)
        total_without, _, rloss_without, _ = loss_fn_without(z, x_recon, x_orig)

        # Bez distance: total == rloss
        assert torch.allclose(total_without, rloss_without)
        # Z distance: total > rloss
        assert total_with > rloss_with

    def test_disable_reconstruction(self):
        """use_reconstruction=False: używa BCE zamiast MSE."""
        # Flagi bez reconstruction – sprawdzamy tylko że się wykonuje bez błędu
        loss_fn = _JointLoss(tau=0.1, use_contrastive=False, use_distance=False,
                             use_reconstruction=False)
        z = torch.randn(16, 32)
        # BCE wymaga wartości w [0,1]
        x_recon = torch.sigmoid(torch.randn(16, NUM_FEATURES))
        x_orig = torch.sigmoid(torch.randn(16, NUM_FEATURES))
        total, _, _, _ = loss_fn(z, x_recon, x_orig)
        assert torch.isfinite(total)

    def test_all_disabled_except_recon(self):
        """Strata tylko rekonstrukcyjna = MSE / batch_size."""
        loss_fn = _JointLoss(tau=0.1, use_contrastive=False, use_distance=False,
                             use_reconstruction=True)
        z = torch.randn(16, 32)
        x_recon = torch.ones(16, 10) * 2.0
        x_orig = torch.zeros(16, 10)
        total, closs, rloss, zloss = loss_fn(z, x_recon, x_orig)
        # MSE = sum((2-0)^2) / 16 = 10*4/16 = 2.5
        expected_mse = (16 * 10 * 4.0) / 16   # = 40.0
        assert abs(rloss.item() - expected_mse) < 1e-3, (
            f"Oczekiwano rloss={expected_mse}, dostałem {rloss.item()}"
        )
        assert torch.allclose(total, rloss)


# ===========================================================================
# 5. pretrain_step – składniki straty w modelu
# ===========================================================================

class TestPretrainStep:
    def test_returns_all_loss_keys(self):
        model = _make_model()
        model.train()
        x = _rand_batch()
        out = model.pretrain_step(x)
        for key in ("loss", "contrastive_loss", "recon_loss", "distance_loss"):
            assert key in out, f"Brak klucza '{key}' w wyjściu pretrain_step"

    def test_loss_finite(self):
        model = _make_model()
        model.train()
        x = _rand_batch()
        out = model.pretrain_step(x)
        assert torch.isfinite(out["loss"]), f"loss nie jest skończone: {out['loss']}"

    def test_loss_backprop(self):
        model = _make_model()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = _rand_batch()
        out = model.pretrain_step(x)
        out["loss"].backward()
        # Gradient pierwszej warstwy enkodera
        first_linear = list(model.ae.encoder.hidden.net.children())[0]
        grad = first_linear.weight.grad
        assert grad is not None and grad.abs().max() > 0, (
            "Gradient enkoder jest zerowy po backward"
        )

    def test_contrastive_disabled(self):
        """use_contrastive=False → contrastive_loss == recon_loss (placeholder)."""
        model = _make_model(use_contrastive=False, use_distance=False)
        model.train()
        x = _rand_batch()
        out = model.pretrain_step(x)
        # total == recon_loss przy wyłączonych obu
        assert torch.allclose(out["loss"], out["recon_loss"], atol=1e-5), (
            "use_contrastive=False + use_distance=False: loss != recon_loss"
        )

    def test_distance_disabled(self):
        """use_distance=False → loss < (loss z distance)."""
        model_with = _make_model(use_distance=True, use_contrastive=False)
        model_without = _make_model(use_distance=False, use_contrastive=False)
        # Ustaw identyczne wagi
        model_without.load_state_dict(model_with.state_dict())
        x = _rand_batch()
        torch.manual_seed(0)
        model_with.train()
        out_with = model_with.pretrain_step(x)
        torch.manual_seed(0)
        model_without.train()
        out_without = model_without.pretrain_step(x)
        # Z distance: loss >= loss bez distance
        assert out_with["loss"].item() >= out_without["loss"].item() - 1e-5, (
            "Z distance_loss wynik nie jest ≥ niż bez"
        )


# ===========================================================================
# 6. forward() deterministyczny w eval()
# ===========================================================================

class TestForwardDeterminism:
    def test_eval_deterministic(self):
        """W eval() forward() jest deterministyczny (brak szumu)."""
        model = _make_model()
        model.eval()
        x = _rand_batch()
        with torch.no_grad():
            h1 = model.forward(x)
            h2 = model.forward(x)
        assert torch.allclose(h1, h2, atol=1e-6), (
            "W eval() forward() nie jest deterministyczny"
        )

    def test_train_vs_eval_differ(self):
        """
        W train() pretrain_step używa szumu → forward z szumem (pretrain pipeline),
        ale forward() (używane do ewaluacji) wywołuje _subset_column_indices(mode='test')
        bez szumu → wyniki w train() i eval() forward() mogą być identyczne dla forward().
        Test upewnia się, że forward() w obu trybach jest spójny (bez szumu).
        """
        model = _make_model()
        x = _rand_batch()
        model.train()
        with torch.no_grad():
            h_train = model.forward(x)
        model.eval()
        with torch.no_grad():
            h_eval = model.forward(x)
        # forward() nie używa szumu niezależnie od trybu → identyczne
        assert torch.allclose(h_train, h_eval, atol=1e-6), (
            "forward() powinno być identyczne w train() i eval() (bez szumu)"
        )

    def test_forward_shape(self):
        model = _make_model(dims=[64, 32])
        x = _rand_batch()
        h = model.forward(x)
        assert h.shape == (BATCH, 32), f"Oczekiwano (B, 32), dostałem {h.shape}"

    def test_linear_readout_shape(self):
        model = _make_model()
        x = _rand_batch()
        logits = model.linear_readout(x)
        assert logits.shape == (BATCH, NUM_CLASSES)


# ===========================================================================
# 7. freeze / unfreeze
# ===========================================================================

class TestFreezeUnfreeze:
    def test_freeze_stops_encoder_grad(self):
        model = _make_model()
        model.freeze_encoder()
        x = _rand_batch()
        # linear_readout używa set_grad_enabled → nie buduje grafu dla zamrożonego enkodera
        logits = model.linear_readout(x)
        logits.sum().backward()
        first_linear = list(model.ae.encoder.hidden.net.children())[0]
        grad = first_linear.weight.grad
        assert grad is None or grad.abs().max() == 0, (
            "Gradient enkoder nie jest zerowy po freeze_encoder()"
        )

    def test_unfreeze_restores_grad(self):
        model = _make_model()
        model.freeze_encoder()
        model.unfreeze_encoder()
        x = _rand_batch()
        out = model.pretrain_step(x)
        out["loss"].backward()
        first_linear = list(model.ae.encoder.hidden.net.children())[0]
        grad = first_linear.weight.grad
        assert grad is not None and grad.abs().max() > 0

    def test_classifier_grad_when_frozen(self):
        """Po freeze, classifier powinien nadal mieć gradient."""
        model = _make_model()
        model.freeze_encoder()
        x = _rand_batch()
        logits = model.linear_readout(x)
        logits.sum().backward()
        cls_grad = model.classifier.weight.grad
        assert cls_grad is not None and cls_grad.abs().max() > 0


# ===========================================================================
# 8. RandomInitSubTabArch – identyczna architektura AE
# ===========================================================================

class TestRandomInitSubTabArch:
    def test_same_ae_architecture(self):
        """
        RandomInitSubTabArchModel i SubTabModel z tymi samymi parametrami
        mają identyczne shapes wag AE enkodera.
        """
        kwargs = dict(num_features=NUM_FEATURES, n_subsets=N_SUBSETS,
                      overlap=OVERLAP, dims=[64, 32])
        subtab = SubTabModel(**kwargs)
        rand_arch = RandomInitSubTabArchModel(**kwargs)

        sub_params = {n: p.shape for n, p in subtab.ae.encoder.named_parameters()}
        rand_params = {n: p.shape for n, p in rand_arch.ae.encoder.named_parameters()}
        assert sub_params == rand_params, (
            f"Różne shapes wag:\n  SubTab: {sub_params}\n  Rand: {rand_params}"
        )

    def test_rand_pretrain_loss_zero(self):
        rand_arch = RandomInitSubTabArchModel(num_features=NUM_FEATURES)
        x = _rand_batch()
        out = rand_arch.pretrain_step(x)
        assert out["loss"].item() == 0.0

    def test_rand_forward_shape(self):
        rand_arch = RandomInitSubTabArchModel(
            num_features=NUM_FEATURES, n_subsets=N_SUBSETS,
            overlap=OVERLAP, dims=[64, 32]
        )
        x = _rand_batch()
        h = rand_arch.forward(x)
        assert h.shape == (BATCH, 32)

    def test_same_output_when_weights_copied(self):
        """Po przekopiowaniu wag, forward() obu modeli daje identyczne wyniki."""
        kwargs = dict(num_features=NUM_FEATURES, n_subsets=N_SUBSETS,
                      overlap=OVERLAP, dims=[64, 32])
        subtab = SubTabModel(**kwargs)
        rand_arch = RandomInitSubTabArchModel(**kwargs)
        rand_arch.ae.encoder.load_state_dict(subtab.ae.encoder.state_dict())

        subtab.eval()
        rand_arch.eval()
        x = _rand_batch()
        with torch.no_grad():
            h_s = subtab.forward(x)
            h_r = rand_arch.forward(x)
        assert torch.allclose(h_s, h_r, atol=1e-5), (
            f"Po przekopiowaniu wag, wyniki różnią się: "
            f"max diff = {(h_s - h_r).abs().max().item():.6f}"
        )


# ===========================================================================
# 9. Wymiary AEWrapper
# ===========================================================================

class TestAEWrapperDims:
    def test_ae_input_dim(self):
        """Enkoder AE przyjmuje subset_dim = n_col_subset + n_overlap."""
        n_col_subset = NUM_FEATURES // N_SUBSETS
        n_overlap = int(OVERLAP * n_col_subset)
        subset_dim = n_col_subset + n_overlap

        model = _make_model(dims=[64, 32])
        ae_input = list(model.ae.encoder.hidden.net.children())[0].in_features
        assert ae_input == subset_dim, (
            f"Oczekiwano subset_dim={subset_dim}, dostałem {ae_input}"
        )

    def test_ae_latent_dim(self):
        """Latent dim = ostatni element dims."""
        model = _make_model(dims=[64, 32])
        latent_dim = model.ae.encoder.latent.out_features
        assert latent_dim == 32

    def test_decoder_output_dim_full(self):
        """reconstruct_subset=False → dekoder odtwarza pełny wektor (num_features)."""
        model = _make_model(reconstruct_subset=False, dims=[64, 32])
        decoder_out = model.ae.decoder.logits.out_features
        assert decoder_out == NUM_FEATURES, (
            f"Oczekiwano recon_dim={NUM_FEATURES}, dostałem {decoder_out}"
        )

    def test_decoder_output_dim_subset(self):
        """reconstruct_subset=True → dekoder odtwarza podzbiór (subset_dim)."""
        n_col_subset = NUM_FEATURES // N_SUBSETS
        n_overlap = int(OVERLAP * n_col_subset)
        subset_dim = n_col_subset + n_overlap
        model = _make_model(reconstruct_subset=True, dims=[64, 32])
        decoder_out = model.ae.decoder.logits.out_features
        assert decoder_out == subset_dim, (
            f"Oczekiwano subset_dim={subset_dim}, dostałem {decoder_out}"
        )


# ===========================================================================
# Testy HPO — n_subsets musi być dzielnikiem num_features
# ===========================================================================

class TestSubTabHPO:
    """
    n_subsets sugerowane przez HPO muszą być dzielnikami num_features.
    Dla Credit Card (30 cech): 3, 5, 6 są dzielnikami; 4, 7 nie są.
    Niedzielnicy powodują, że ostatnie kolumny (reszta z dzielenia) nigdy
    nie trafiają do żadnego podzbioru.
    """

    def test_hpo_n_subsets_are_divisors(self):
        """
        _get_subtab_hpo_space zwraca wyłącznie n_subsets będące dzielnikami
        num_features, lub HPO suggest_categorical używa tylko dzielników.
        """
        import optuna
        from contrastive_framework.hpo import _suggest
        from contrastive_framework.models.tabular_models import SubTabModel

        num_features = 30
        divisors_of_30 = {d for d in range(2, 9) if num_features % d == 0}
        # {2, 3, 5, 6}

        suggested = set()
        for seed in range(20):
            study = optuna.create_study(
                sampler=optuna.samplers.RandomSampler(seed=seed)
            )

            def objective(trial):
                params = _suggest(trial, SubTabModel, num_features=num_features)
                suggested.add(params["n_subsets"])
                return 0.0

            study.optimize(objective, n_trials=1, show_progress_bar=False)

        non_divisors = suggested - divisors_of_30
        assert not non_divisors, (
            f"HPO zaproponowało n_subsets nie będące dzielnikami {num_features}: "
            f"{non_divisors}. Dozwolone: {divisors_of_30}."
        )

    def test_subtab_init_warns_missing_columns(self):
        """
        SubTabModel.__init__ emituje ostrzeżenie gdy n_subsets nie dzieli num_features
        (kolumny na końcu są pomijane przez podzbiory).
        """
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SubTabModel(num_features=30, n_subsets=4)  # 30 % 4 == 2 → ostrzeżenie

        messages = [str(warning.message) for warning in w]
        assert any("kolumn" in m.lower() or "pokryt" in m.lower() or "subset" in m.lower()
                   for m in messages), (
            f"Brak ostrzeżenia dla n_subsets=4, num_features=30 (2 kolumny pominięte). "
            f"Otrzymane ostrzeżenia: {messages}"
        )

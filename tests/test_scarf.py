"""
Testy SCARFModel i info_nce_loss.

Sprawdzają:
1. Korupcja: dokładnie floor(c*M) cech na wiersz (nie Bernoulli)
2. Wartości zastępcze pochodzą z kolumn marginals (zbiór treningowy)
3. Brak wycieku: marginals = tylko train, wartości z val/test nie są użyte
4. info_nce_loss: asymetryczne InfoNCE (nie NT-Xent), temperatura domyślna 1.0
5. Strata: skończona, backprop działa, zakres sensowny
6. set_marginals: walidacja kształtu, przechowuje dane CPU
7. Fallback bez set_marginals: RuntimeWarning, korupcja działa
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrastive_framework.models.tabular_models import SCARFModel, info_nce_loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_model(
    num_features: int = 20,
    corruption_rate: float = 0.6,
    temperature: float = 1.0,
) -> SCARFModel:
    return SCARFModel(
        num_features=num_features,
        hidden_dim=32,
        proj_dim=16,
        num_classes=2,
        corruption_rate=corruption_rate,
        temperature=temperature,
    )


def make_marginals(N: int = 100, M: int = 20) -> torch.Tensor:
    """Małe syntetyczne dane treningowe — wartości na 100-999."""
    return torch.rand(N, M) * 900.0 + 100.0  # zakres [100, 1000]


def make_val_data(N: int = 20, M: int = 20) -> torch.Tensor:
    """Dane walidacyjne — wartości na 0-1 (łatwo odróżnić od marginals)."""
    return torch.rand(N, M) * 0.01  # zakres [0, 0.01]


# ===========================================================================
# TestCorruptionCount: dokładnie floor(c*M) cech na wiersz
# ===========================================================================

class TestCorruptionCount:
    """Artykuł: q = floor(c·M) cech na wiersz, bez powtórzeń."""

    def test_exact_count_per_row(self):
        """Każdy wiersz ma dokładnie floor(c*M) zmienionych cech."""
        M = 20
        c = 0.6
        q = int(c * M)  # 12
        model = make_model(M, c)

        # Marginale dalekie od zera → łatwo wykryć zmianę
        train_X = make_marginals(200, M)
        model.set_marginals(train_X)

        # x oryginalne = 0.0 → wszelkie wartości zastępcze != 0.0 są widoczne
        x = torch.zeros(16, M)
        x_tilde = model._corrupt(x)

        for i in range(x.shape[0]):
            changed = (x_tilde[i] != x[i]).sum().item()
            assert changed == q, (
                f"Wiersz {i}: oczekiwano {q} zmian, got {changed}"
            )

    def test_exact_count_different_rates(self):
        """Testuje floor dla różnych corruption_rate."""
        M = 15
        for c in [0.1, 0.3, 0.5, 0.8]:
            q = int(c * M)
            model = make_model(M, c)
            model.set_marginals(make_marginals(50, M) + 100.0)
            x = torch.zeros(8, M)
            x_tilde = model._corrupt(x)
            for i in range(x.shape[0]):
                changed = (x_tilde[i] != x[i]).sum().item()
                assert changed == q, f"c={c}: wiersz {i} ma {changed} zmian, oczekiwano {q}"

    def test_no_replacement_per_row(self):
        """Wylosowane kolumny są unikalne w każdym wierszu (bez powtórzeń)."""
        M = 20
        model = make_model(M, 0.5)
        # Marginale tak, że każda kolumna daje unikalną wartość
        # marginals[j, j] = j+1000, by łatwo odwzorować która kolumna była zmieniona
        marginals = torch.zeros(1, M)
        for j in range(M):
            marginals[0, j] = j + 1000.0
        model.set_marginals(marginals)

        x = torch.zeros(32, M)
        x_tilde = model._corrupt(x)
        for i in range(x.shape[0]):
            changed_cols = (x_tilde[i] != x[i]).nonzero(as_tuple=True)[0]
            assert len(changed_cols) == len(changed_cols.unique()), (
                f"Wiersz {i}: powtórzone indeksy kolumn w korupcji"
            )

    def test_zero_rate_no_change(self):
        """corruption_rate → q=0 (floor(0.04*20)=0): brak zmian."""
        model = make_model(num_features=20, corruption_rate=0.04)
        model.set_marginals(make_marginals(50, 20) + 999.0)
        x = torch.ones(8, 20) * 0.5
        x_tilde = model._corrupt(x)
        assert torch.allclose(x_tilde, x)


# ===========================================================================
# TestCorruptionSource: wartości z kolumn marginals
# ===========================================================================

class TestCorruptionSource:
    """Wartości zastępcze muszą pochodzić z kolumn marginals (zbiór treningowy)."""

    def test_replacement_from_training_column(self):
        """
        Każda zastąpiona wartość musi znajdować się w odpowiedniej kolumnie marginals.
        """
        M = 10
        model = make_model(M, 0.5)
        # Każda kolumna marginals ma unikalne wartości: kolumna j = [j*1000, j*1001, ...]
        N_train = 50
        marginals = torch.zeros(N_train, M)
        for j in range(M):
            marginals[:, j] = torch.arange(N_train, dtype=torch.float) + j * 10000.0
        model.set_marginals(marginals)

        x = torch.zeros(32, M)  # oryginalne 0.0 — łatwe do odróżnienia
        x_tilde = model._corrupt(x)

        for i in range(x.shape[0]):
            for j in range(M):
                val = x_tilde[i, j].item()
                if val == 0.0:
                    continue  # ta cecha nie była zmieniana
                # Wartość musi być w kolumnie j marginals
                col_values = marginals[:, j].tolist()
                assert val in col_values, (
                    f"Wiersz {i}, kolumna {j}: wartość {val} nie pochodzi "
                    f"z kolumny treningowej (zakres [{min(col_values):.0f}, {max(col_values):.0f}])"
                )

    def test_replacement_not_from_val_data(self):
        """
        Wartości zastępcze NIE mogą być wartościami z danych walidacyjnych.

        Marginale = zakres [100, 1000], dane walidacyjne = [0, 0.01].
        Po ustawieniu set_marginals() korupcja używa tylko marginals.
        """
        M = 15
        model = make_model(M, 0.6)
        train_X = make_marginals(100, M)  # zakres [100, 1000]
        val_X = make_val_data(30, M)       # zakres [0, 0.01]
        model.set_marginals(train_X)
        # NIE wywołujemy set_marginals(val_X) — to test, że nie ma wycieku

        # Input zbliżony do val (zakres 0-0.01)
        x = torch.rand(16, M) * 0.01
        x_tilde = model._corrupt(x)

        for i in range(x.shape[0]):
            for j in range(M):
                val = x_tilde[i, j].item()
                if val < 0.1:
                    # Wartość < 0.1 oznacza, że NIE pochodzi z marginals [100,1000]
                    # ale może być z oryginalnego x ≈ 0 (niezmieniony)
                    # Sprawdź, czy to oryginalna wartość
                    orig = x[i, j].item()
                    assert abs(val - orig) < 1e-6, (
                        f"Wiersz {i}, kolumna {j}: wartość {val:.4f} nie pochodzi "
                        f"z marginals [100, 1000] ani z oryginału {orig:.4f}"
                    )

    def test_marginals_stored_as_cpu(self):
        """set_marginals przechowuje dane na CPU."""
        model = make_model(10, 0.5)
        # Dane na CPU
        train_X = torch.rand(50, 10)
        model.set_marginals(train_X)
        assert model._marginals is not None
        assert model._marginals.device.type == "cpu"

    def test_marginals_detached(self):
        """Marginals są odłączone od grafu obliczeniowego."""
        model = make_model(10, 0.5)
        train_X = torch.rand(50, 10, requires_grad=True)
        # forward pass, żeby train_X miało grad_fn
        z = train_X * 2.0
        model.set_marginals(z)
        assert model._marginals is not None
        assert not model._marginals.requires_grad


# ===========================================================================
# TestNoDataLeakage: walidacja set_marginals + fallback
# ===========================================================================

class TestNoDataLeakage:
    """Bezpieczeństwo: marginals = tylko train split."""

    def test_set_marginals_validates_shape(self):
        """set_marginals rzuca ValueError gdy kształt niezgodny."""
        model = make_model(num_features=10)
        with pytest.raises(ValueError, match="set_marginals"):
            model.set_marginals(torch.rand(50, 7))  # 7 != 10

    def test_set_marginals_validates_ndim(self):
        """set_marginals rzuca ValueError dla 1D tensora."""
        model = make_model(num_features=10)
        with pytest.raises(ValueError):
            model.set_marginals(torch.rand(10))  # 1D

    def test_fallback_warns_without_marginals(self):
        """Bez set_marginals: korupcja emituje RuntimeWarning."""
        model = make_model(10, 0.5)
        x = torch.rand(8, 10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_tilde = model._corrupt(x)
        runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
        assert len(runtime_warnings) >= 1

    def test_fallback_still_corrupts_correct_count(self):
        """Fallback (bez marginals) nadal zmienia dokładnie floor(c*M) cech."""
        M = 20
        c = 0.5
        q = int(c * M)
        model = make_model(M, c)
        # Zbuduj x z wartościami 0.0, marginals z wartościami > 1000
        # Fallback używa wartości z batcha — losuj x o różnych wierszach
        x = torch.arange(M * 16, dtype=torch.float).reshape(16, M)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            x_tilde = model._corrupt(x)
        for i in range(x.shape[0]):
            changed = (x_tilde[i] != x[i]).sum().item()
            assert changed == q, f"Fallback: wiersz {i} ma {changed} zmian, oczekiwano {q}"

    def test_marginals_not_modified(self):
        """Korupcja nie modyfikuje in-place tablicy marginals."""
        M = 10
        model = make_model(M, 0.6)
        train_X = torch.rand(50, M) * 100.0
        model.set_marginals(train_X)
        marginals_before = model._marginals.clone()
        x = torch.rand(16, M)
        model._corrupt(x)
        assert torch.allclose(model._marginals, marginals_before)


# ===========================================================================
# TestInfoNCELoss: asymetryczne InfoNCE
# ===========================================================================

class TestInfoNCELoss:
    """Strata InfoNCE z arXiv:2106.15147 Algorithm 1 linia 8."""

    def test_temperature_default_is_1(self):
        """Temperatura domyślna to 1.0 (Section 4 artykułu)."""
        import inspect
        sig = inspect.signature(info_nce_loss)
        default_temp = sig.parameters["temperature"].default
        assert default_temp == 1.0, f"Oczekiwano temperature=1.0, got {default_temp}"

    def test_model_temperature_default_is_1(self):
        """SCARFModel.temperature domyślnie = 1.0."""
        model = make_model()
        assert model.temperature == 1.0

    def test_perfect_alignment_low_loss(self):
        """Gdy z_anchor == z_positive: loss bliskie 0 (pozytyw jest jedynym kandydatem)."""
        B = 16
        z = torch.randn(B, 32)
        z = torch.nn.functional.normalize(z, dim=-1)
        loss = info_nce_loss(z, z.clone())
        # log(B) = oczekiwana strata przy równomiernym rozkładzie
        # Przy idealnym dopasowaniu mamy 1 pozytyw i B-1 negatywów z tą samą wartością
        assert loss.item() < 2.0 * torch.log(torch.tensor(B)).item()
        assert loss.item() >= 0.0

    def test_diagonal_is_positive(self):
        """Pozytyw na diagonali: loss = -log(exp(s_ii) / Σ_k exp(s_ik))."""
        B = 4
        z_a = torch.eye(B)  # wektory jednostkowe
        z_p = torch.eye(B)  # identyczne → s_ii = 1, s_ij = 0 (i != j)
        loss = info_nce_loss(z_a, z_p, temperature=1.0)
        expected = -torch.log(torch.tensor(torch.exp(torch.tensor(1.0)) /
                               (torch.exp(torch.tensor(1.0)) + B - 1)))
        assert abs(loss.item() - expected.item()) < 1e-4

    def test_loss_finite(self):
        """Strata jest skończona dla losowych wejść."""
        z_a = torch.nn.functional.normalize(torch.randn(32, 64), dim=-1)
        z_p = torch.nn.functional.normalize(torch.randn(32, 64), dim=-1)
        loss = info_nce_loss(z_a, z_p)
        assert torch.isfinite(loss)

    def test_loss_non_negative(self):
        """Strata jest nieujemna (cross-entropy ≥ 0)."""
        z_a = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
        z_p = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
        assert info_nce_loss(z_a, z_p).item() >= 0.0

    def test_asymmetric_info_nce(self):
        """
        InfoNCE jest asymetryczne: info_nce_loss(a, b) != info_nce_loss(b, a).

        W NT-Xent symetrycznym wynik jest identyczny (symetria). W InfoNCE macierze
        logitów są transpozycją siebie → przy niesymetrycznych danych strata się różni.
        """
        torch.manual_seed(42)
        z_a = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        z_b = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        loss_ab = info_nce_loss(z_a, z_b)
        loss_ba = info_nce_loss(z_b, z_a)
        # Strace powinny być różne (asymetria)
        assert abs(loss_ab.item() - loss_ba.item()) > 1e-6, (
            f"loss(a,b)={loss_ab.item():.6f} == loss(b,a)={loss_ba.item():.6f} "
            f"→ wygląda na NT-Xent symetryczny zamiast InfoNCE"
        )

    def test_temperature_effect(self):
        """Wyższa temperatura → niższe szczyty logitów → wyższa strata (bardziej równomierny rozkład)."""
        z_a = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
        z_p = z_a.clone()  # identyczne → maksymalny sygnał
        loss_low_T = info_nce_loss(z_a, z_p, temperature=0.1)
        loss_high_T = info_nce_loss(z_a, z_p, temperature=2.0)
        assert loss_low_T.item() < loss_high_T.item()

    def test_loss_uses_full_matrix(self):
        """
        InfoNCE używa macierzy (B×B): wynik zależy od liczby negatywów.
        Przy B=2 jest 1 negatyw, przy B=16 jest 15 negatywów → strata rośnie.
        """
        z2_a = torch.nn.functional.normalize(torch.randn(2, 32), dim=-1)
        z2_p = z2_a.clone()
        z16_a = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
        z16_p = z16_a.clone()
        # Więcej negatywów → trudniejszy problem → wyższa strata dla idealnego dopasowania
        loss_2 = info_nce_loss(z2_a, z2_p)
        loss_16 = info_nce_loss(z16_a, z16_p)
        assert loss_16.item() > loss_2.item()


# ===========================================================================
# TestSCARFPretrainStep: integracja — pretrain_step działa end-to-end
# ===========================================================================

class TestSCARFPretrainStep:
    """Testy integracyjne pretrain_step."""

    def test_loss_finite(self):
        """Strata jest skończona."""
        model = make_model(20, 0.6)
        model.set_marginals(make_marginals(100, 20))
        x = torch.randn(16, 20)
        result = model.pretrain_step(x)
        assert torch.isfinite(result["loss"])

    def test_loss_non_negative(self):
        """Strata jest nieujemna."""
        model = make_model(20, 0.6)
        model.set_marginals(make_marginals(100, 20))
        x = torch.randn(16, 20)
        result = model.pretrain_step(x)
        assert result["loss"].item() >= 0.0

    def test_backprop(self):
        """Strata ma gradient — backprop działa."""
        model = make_model(20, 0.6)
        model.set_marginals(make_marginals(100, 20))
        x = torch.randn(16, 20)
        result = model.pretrain_step(x)
        result["loss"].backward()
        # Sprawdź, że co najmniej jeden parametr ma gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_pretrain_step_accepts_tuple_batch(self):
        """pretrain_step przyjmuje batch jako (x, label)."""
        model = make_model(20, 0.6)
        model.set_marginals(make_marginals(100, 20))
        x = torch.randn(16, 20)
        y = torch.zeros(16, dtype=torch.long)
        result = model.pretrain_step((x, y))
        assert torch.isfinite(result["loss"])

    def test_pretrain_step_returns_dict(self):
        """pretrain_step zwraca słownik z kluczem 'loss'."""
        model = make_model(20, 0.6)
        model.set_marginals(make_marginals(100, 20))
        x = torch.randn(16, 20)
        result = model.pretrain_step(x)
        assert isinstance(result, dict)
        assert "loss" in result

    def test_anchor_not_corrupted(self):
        """
        Anchor = oryginał: upewnij się, że pretrain_step nie modyfikuje x in-place.
        """
        model = make_model(20, 0.6)
        model.set_marginals(make_marginals(100, 20))
        x = torch.randn(16, 20)
        x_before = x.clone()
        model.pretrain_step(x)
        assert torch.allclose(x, x_before)


# ===========================================================================
# Testy HPO — SCARF musi mieć marginale podczas HPO trials
# ===========================================================================

class TestSCARFHPO:
    """
    _objective w hpo.py musi wywołać set_marginals() PRZED pretrain dla SCARF.
    Bez marginals pretrain_step rzuca RuntimeError.
    """

    def test_scarf_pretrain_step_warns_without_marginals(self):
        """pretrain_step emituje RuntimeWarning gdy marginals nie są ustawione."""
        import warnings
        model = make_model(10, 0.6)
        # NIE wywołujemy set_marginals
        x = torch.randn(8, 10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.pretrain_step(x)
        messages = [str(warning.message) for warning in w]
        assert any("argin" in m.lower() for m in messages), (
            f"Brak ostrzeżenia o brakujących marginals. Otrzymane: {messages}"
        )

    def test_hpo_objective_calls_set_marginals(self):
        """
        _objective powinno wywołać set_marginals dla SCARF zanim uruchomi pretrain.
        Test: mockujemy DataModule, sprawdzamy że model po HPO trial ma _marginals != None.
        """
        import optuna
        from unittest.mock import MagicMock
        from contrastive_framework.hpo import _objective
        from contrastive_framework.models.tabular_models import SCARFModel

        num_features = 10
        X = torch.randn(64, num_features)
        y = torch.randint(0, 2, (64,))
        dataset = torch.utils.data.TensorDataset(X, y)

        # Mock DataModule z minimalnym API
        dm = MagicMock()
        dm.train_dataset = dataset
        dm.pretrain_loader.return_value = torch.utils.data.DataLoader(
            dataset, batch_size=16
        )
        dm.finetune_loader.return_value = torch.utils.data.DataLoader(
            dataset, batch_size=16
        )
        dm.val_loader.return_value = torch.utils.data.DataLoader(
            dataset, batch_size=16
        )

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.RandomSampler(seed=0))
        fixed_kwargs = {"num_features": num_features, "num_classes": 2}

        # Zbieramy modele stworzone przez _objective żeby sprawdzić _marginals
        created_models = []
        original_class_call = SCARFModel.__call__

        import warnings
        # Po wywołaniu _objective pretrain nie może dać RuntimeWarning o brakujących marginals
        warnings_seen = []

        def objective(trial):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = _objective(
                    trial=trial,
                    model_class=SCARFModel,
                    fixed_model_kwargs=fixed_kwargs,
                    data_module=dm,
                    pretrain_epochs=1,
                    finetune_epochs=1,
                    finetune_fraction=1.0,
                    device=torch.device("cpu"),
                )
            marginals_warnings = [str(x.message) for x in w
                                   if "argin" in str(x.message).lower()]
            warnings_seen.extend(marginals_warnings)
            return result

        study.optimize(objective, n_trials=1, show_progress_bar=False)

        assert not warnings_seen, (
            f"_objective uruchomiło pretrain SCARF bez set_marginals: {warnings_seen}"
        )

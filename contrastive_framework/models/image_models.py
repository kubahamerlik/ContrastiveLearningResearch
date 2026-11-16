"""
image_models.py – Implementacje MoCo v2 i BYOL dla danych obrazowych.

MoCo v2 : He et al., 2020 – "Improved Baselines with Momentum Contrastive Learning"
           arXiv:2003.04297; kod: github.com/facebookresearch/moco (CC BY-NC 4.0)
           Gałąź: KaimingHe-patch-2, plik: moco/builder.py
BYOL    : Grill et al., 2020 – "Bootstrap Your Own Latent"
"""

from typing import Any, Dict, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContrastiveModel, build_encoder, build_mlp, SplitBatchNorm2d


def _build_byol_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """
    Głowica MLP dla BYOL — projektor i predyktor mają identyczną budowę.

    Port z: deepmind-research/byol/utils/networks.py, klasa MLP.__call__:
        out = Linear(hidden_size)(inputs)       # bias=True (domyślnie Haiku)
        out = BatchNorm()(out, is_training)
        out = relu(out)
        out = Linear(output_size, with_bias=False)(out)   # ← ostatnia warstwa BEZ biasu

    Różnica względem build_mlp() z base.py:
    - Ostatnia warstwa Linear nie ma biasu (with_bias=False w Haiku).
    - build_mlp() używa bias=True wszędzie i nie ma tego jako parametru.
    - MoCo v2 projector (bez BN) to zupełnie inna architektura — nie używaj tu.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim, bias=False),
    )


def _build_moco_v2_projector(in_dim: int, proj_dim: int) -> nn.Sequential:
    """
    Głowica projekcji MoCo v2: Linear → ReLU → Linear (bez BN).

    Port z: moco/builder.py linia 31-32:
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
    W oryginale `self.encoder_q.fc` to Linear(dim_mlp, proj_dim), więc
    łączna głowica to: Linear(in_dim, in_dim) → ReLU → Linear(in_dim, proj_dim).
    Brak BN — to jest wymaganie v2 (w przeciwieństwie do SimCLR i naszego build_mlp).
    """
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.ReLU(inplace=True),
        nn.Linear(in_dim, proj_dim),
    )


# ===========================================================================
# 1. MoCo – Momentum Contrast
# ===========================================================================

class MoCoModel(BaseContrastiveModel):
    """
    MoCo v2 – Momentum Contrast z głowicą MLP i augmentacją aug-plus.

    Port z: github.com/facebookresearch/moco (CC BY-NC 4.0),
    gałąź KaimingHe-patch-2, plik moco/builder.py.
    Zmiany względem oryginału opisane w docs/DEVIATIONS.md.

    Architektura
    ------------
    encoder_q  : koder online (aktualizowany przez optymalizator, zwykły BN)
    projector_q: MLP głowica v2 — Linear → ReLU → Linear (BEZ BN)
                 Port z builder.py linia 31-32 (mlp=True)
    encoder_k  : koder momentum (EMA od encoder_q, SplitBatchNorm; train() podczas pretrain)
    projector_k: kopia projector_q, aktualizowana przez EMA
    queue      : cykliczny bufor negatywnych wektorów K × proj_dim
                 Inicjalizowana losowymi znorm. wektorami (jak w builder.py L.39-40)

    Parametry
    ---------
    queue_size  : int   – liczba negatywnych wektorów K (domyślnie 4096)
                  Oryginał: K=65536; ograniczamy do n_train/4 w HPO (opisane w DEVIATIONS.md)
    momentum    : float – m w EMA (domyślnie 0.999, jak w oryginale)
    temperature : float – τ w InfoNCE (domyślnie 0.2 — wartość v2 z --moco-t 0.2)
                  MoCo v1 używa 0.07; v2 (aug-plus) używa 0.2
    bn_splits   : int   – liczba podgrup SplitBatchNorm w encoder_k (domyślnie 8)
    """

    def __init__(
        self,
        encoder_type: str = "cnn_image",
        input_dim: int = 3,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        num_classes: int = 10,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.2,
        bn_splits: int = 8,
    ) -> None:
        super().__init__(encoder_type, input_dim, hidden_dim, proj_dim, num_classes)

        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        self.bn_splits = bn_splits

        # Nadpisz projector z BaseContrastiveModel — MoCo v2 używa MLP bez BN
        # Port z builder.py L.31-32: Linear(dim_mlp, dim_mlp) → ReLU → Linear(dim_mlp, dim)
        self.projector = _build_moco_v2_projector(hidden_dim, proj_dim)

        # encoder_k z SplitBatchNorm (bn_splits>1) — głęboka kopia encoder_q nie zadziała
        # bo encoder_q ma zwykły BN; budujemy encoder_k od nowa z bn_splits
        if encoder_type == "cnn_image" and bn_splits > 1:
            self.encoder_k = build_encoder(
                encoder_type, input_dim, hidden_dim, hidden_dim, bn_splits=bn_splits
            )
            # Skopiuj parametry (wagi) z encoder_q do encoder_k
            # Port z builder.py L.34-35: param_k.data.copy_(param_q.data)
            # Bufory BN celowo nie są kopiowane — encoder_k liczy własne statystyki
            # (odpowiednik Shuffle BN na jednej karcie)
            encoder_k_params = dict(self.encoder_k.named_parameters())
            for name, param in self.encoder.named_parameters():
                if name in encoder_k_params:
                    encoder_k_params[name].data.copy_(param.data)
        else:
            self.encoder_k = copy.deepcopy(self.encoder)

        # projector_k = kopia projector_q (MLP bez BN)
        self.projector_k = copy.deepcopy(self.projector)

        # Wyłącz gradienty dla gałęzi kluczy
        # Port z builder.py L.36: param_k.requires_grad = False
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        # encoder_k/projector_k NIE są ustawiane na eval() — powinny zostać w train()
        # podczas pre-treningu żeby BN używało statystyk batcha (nie zamrożonych 0/1).

        # Kolejka: (proj_dim, queue_size) – losowe znorm. wektory
        # Port z builder.py L.39-40: queue = randn → normalize(dim=0)
        self.register_buffer("queue", F.normalize(
            torch.randn(proj_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        print(f"[MoCoModel] queue_size K={queue_size}, T={temperature}, m={momentum}")

    def train(self, mode: bool = True) -> "MoCoModel":
        """
        encoder_k/projector_k działają w train() podczas pre-treningu
        (żeby BN liczył statystyki batcha, nie używał zamrożonych 0/1).
        Wyjątek: gdy encoder jest zamrożony (linear probing), ustawiamy eval()
        dla całości — spójność z BN w inferencji.
        """
        super().train(mode)
        # encoder_k/projector_k śledzą tryb całego modelu (train vs eval),
        # ALE gdy encoder jest zamrożony, eval() jest już ustawione przez base.train()
        if mode and not self._encoder_frozen:
            self.encoder_k.train()
            self.projector_k.train()
        else:
            self.encoder_k.eval()
            self.projector_k.eval()
        return self

    @torch.no_grad()
    def _momentum_update(self) -> None:
        """Aktualizuje encoder_k jako EMA od encoder_q (parametry + bufory BN)."""
        for q_param, k_param in zip(self.encoder.parameters(),
                                    self.encoder_k.parameters()):
            k_param.data = k_param.data * self.momentum + q_param.data * (1.0 - self.momentum)
        for q_param, k_param in zip(self.projector.parameters(),
                                    self.projector_k.parameters()):
            k_param.data = k_param.data * self.momentum + q_param.data * (1.0 - self.momentum)
        # Bufory BN (running_mean, running_var) enkoder_k aktualizują się przez
        # forward pass (train mode) — nie przez EMA. To jest właściwe: liczymy
        # statystyki batcha osobno dla gałęzi kluczy (jak Shuffle BN w oryginale).

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Zastępuje najstarsze wpisy w kolejce nowymi kluczami."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if batch_size >= self.queue_size:
            self.queue[:, :] = keys[-self.queue_size:].T
            self.queue_ptr[0] = 0
            return

        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            tail = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:tail].T
            self.queue[:, :batch_size - tail] = keys[tail:].T

        ptr = end % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def prefill_queue(self, loader) -> None:
        """Wypełnia kolejkę prawdziwymi embeddingami zamiast losowych wektorów Gaussa.

        Stabilizuje InfoNCE od pierwszej epoki — zgodnie z MoCo v2 (He et al. 2020).
        """
        was_training = self.training
        self.eval()
        filled = 0
        for batch in loader:
            if filled >= self.queue_size:
                break
            x_k = batch[1] if isinstance(batch, (tuple, list)) else batch
            device = next(self.parameters()).device
            x_k = x_k.to(device)
            h_k = self.encoder_k(x_k)
            k = F.normalize(self.projector_k(h_k), dim=-1)
            self._dequeue_and_enqueue(k)
            filled += k.shape[0]
        if was_training:
            self.train()

    def _info_nce_loss(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        """
        Oblicza stratę InfoNCE pomiędzy zapytaniami q i kluczami k,
        używając negatywnych z kolejki.

        Parameters
        ----------
        q : (B, D) – wektory zapytań (znormalizowane)
        k : (B, D) – pozytywne klucze (znormalizowane)
        """
        # Podobieństwo pozytywne: (B, 1)
        l_pos = torch.einsum("bd,bd->b", q, k).unsqueeze(-1)

        # Podobieństwo negatywne z kolejki: (B, K)
        l_neg = torch.einsum("bd,dk->bk", q, self.queue.clone().detach())

        # Logity: (B, 1+K), etykiety zawsze 0 (pozytyw jest na pozycji 0)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=logits.device)

        return F.cross_entropy(logits, labels)

    def pretrain_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """batch : (x_q, x_k) – dwie augmentacje tego samego obrazu.

        Kolejność operacji zgodna z MoCo builder.py:
          1. Momentum update encoder_k ← EMA(encoder_q)  [przed forwardem]
          2. Forward encoder_q → q (z gradientem)
          3. Forward encoder_k → k (bez gradientu, ale BN w train mode)
          4. InfoNCE loss, dequeue+enqueue
        """
        x_q, x_k = batch

        # 1. Aktualizuj momentum przed forwardem (jak w MoCo builder.py)
        self._momentum_update()

        # 2. Gałąź online
        h_q = self.encoder(x_q)
        q = F.normalize(self.projector(h_q), dim=-1)

        # 3. Gałąź kluczy — bez gradientu, ale BN w train() aktualizuje running stats
        with torch.no_grad():
            h_k = self.encoder_k(x_k)
            k = F.normalize(self.projector_k(h_k), dim=-1)

        loss = self._info_nce_loss(q, k)
        self._dequeue_and_enqueue(k)

        return {"loss": loss}


# ===========================================================================
# 2. BYOL – Bootstrap Your Own Latent
# ===========================================================================

class BYOLModel(BaseContrastiveModel):
    """
    BYOL – Bootstrap Your Own Latent (Grill et al. 2020).

    Port z: deepmind-research/byol/ (Apache-2.0).
    Zmiany względem oryginału opisane w docs/DEVIATIONS.md.

    Architektura
    ------------
    Sieć online : encoder → projector MLP → predictor MLP
    Sieć target : encoder_t → projector_t  (EMA od online, bez gradientów,
                                             BEZ predyktora — jak w oryginale)

    MLP: Linear(hidden, bias=True) → BN → ReLU → Linear(out, bias=False)
    Port z: deepmind-research/byol/utils/networks.py, klasa MLP.__call__ L.41-44

    Harmonogram τ
    -------------
    tau = 1 − (1 − base_ema) × cosine_decay(step, max_steps)
    gdzie cosine_decay(t, T) = 0.5 × (1 + cos(π × t / T))
    Port z: deepmind-research/byol/utils/schedules.py, funkcja target_ema L.21-24

    Strata
    ------
    loss = mean(regression_loss(p1, sg(z2)) + regression_loss(p2, sg(z1)))
    gdzie regression_loss(p, z) = 2 − 2 × cos(p, z) zwraca per-sample tensor,
    a mean() redukuje po próbkach (oba widoki razem).
    Port z: deepmind-research/byol/byol_experiment.py L.243-250:
        repr_loss = regression_loss(pred_v1, stop_gradient(proj_v2))
                  + regression_loss(pred_v2, stop_gradient(proj_v1))
        repr_loss = jnp.mean(repr_loss)

    Uwagi dot. BN w sieci target
    ----------------------------
    Oryginalny BYOL wywołuje sieć target z is_training=True
    (byol_experiment.py L.236: forward.apply(..., is_training=True)).
    BN w encoder_t/projector_t liczy statystyki batcha (nie zamrożone).
    Zapewniamy to przez train() mode podczas pre-treningu — szczegóły w DEVIATIONS.md.

    Optymalizator
    -------------
    Oryginał: LARS, lr=0.2×(batch/256), wd=1.5e-6, warmup+cosine (byol.py L.57-66).
    Nasz framework: Adam z warmup-cosine — opisane w DEVIATIONS.md.
    """

    def __init__(
        self,
        encoder_type: str = "cnn_image",
        input_dim: int = 3,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        pred_dim: int = 128,
        num_classes: int = 10,
        momentum: float = 0.996,
    ) -> None:
        super().__init__(encoder_type, input_dim, hidden_dim, proj_dim, num_classes)

        self.momentum_base = momentum
        self.momentum = momentum

        # Nadpisz projector z BaseContrastiveModel — BYOL używa MLP z BN bez biasu w ostatniej warstwie
        # Port z: byol/utils/networks.py MLP L.41-44
        self.projector = _build_byol_mlp(hidden_dim, hidden_dim, proj_dim)

        # Predyktor — tylko w sieci online (target nie ma predyktora)
        # Port z: byol_experiment.py L.153-162: predictor budowany tylko dla online
        self.predictor: nn.Module = _build_byol_mlp(proj_dim, hidden_dim, proj_dim)

        # Sieć target — głęboka kopia online; BEZ predyktora
        self.encoder_t = copy.deepcopy(self.encoder)
        self.projector_t = copy.deepcopy(self.projector)
        for param in self.encoder_t.parameters():
            param.requires_grad = False
        for param in self.projector_t.parameters():
            param.requires_grad = False

        # encoder_t/projector_t NIE są tu ustawiane na eval().
        # byol_experiment.py L.236: forward.apply(..., is_training=True) →
        # sieć target działa w train() (BN liczy statystyki batcha, nie running stats).

    def train(self, mode: bool = True) -> "BYOLModel":
        """
        encoder_t/projector_t śledzą tryb całego modelu (train/eval).
        Gdy encoder jest zamrożony (linear probing) — wszystko w eval().
        """
        super().train(mode)
        if mode and not self._encoder_frozen:
            self.encoder_t.train()
            self.projector_t.train()
        else:
            self.encoder_t.eval()
            self.projector_t.eval()
        return self

    def update_momentum(self, step: int, total_steps: int) -> None:
        """Cosine schedule τ: rośnie od τ_base do 1.0 w trakcie treningu.

        Port z: deepmind-research/byol/utils/schedules.py, target_ema():
            decay = cosine_decay(global_step, max_steps, initial_value=1.0)
            tau = 1 − (1 − base_ema) × decay
        gdzie cosine_decay(t, T, v) = v × 0.5 × (1 + cos(π × t / T))

        Przy step=0: decay=1.0 → tau=base_ema (minimum)
        Przy step=max_steps: decay=0.0 → tau=1.0 (maksimum — stop update)
        """
        import math
        if total_steps <= 0:
            self.momentum = self.momentum_base
            return
        t = min(step, total_steps)
        # cosine_decay(t, T, 1.0) = 0.5 * (1 + cos(π * t / T))
        decay = 0.5 * (1.0 + math.cos(math.pi * t / total_steps))
        self.momentum = 1.0 - (1.0 - self.momentum_base) * decay

    @torch.no_grad()
    def _ema_update(self) -> None:
        """Aktualizuje sieć target jako EMA od online (parametry).

        Port z: byol_experiment.py L.342-343:
            target_params = x + (1 - tau) * (y - x)
            co jest równoważne: θ_t = τ × θ_t + (1−τ) × θ_o

        Bufory BN (running_mean/var) są aktualizowane przez forward pass
        w trybie train() — nie przez EMA (jak w oryginale: is_training=True).
        """
        for o, t in zip(self.encoder.parameters(), self.encoder_t.parameters()):
            t.data = t.data * self.momentum + o.data * (1.0 - self.momentum)
        for o, t in zip(self.projector.parameters(), self.projector_t.parameters()):
            t.data = t.data * self.momentum + o.data * (1.0 - self.momentum)

    @staticmethod
    def _regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Strata regresji per-sample: 2 − 2 × cos(p, z).

        Port z: deepmind-research/byol/utils/helpers.py (helpers.regression_loss):
        używa znormalizowanych wektorów, iloczyn skalarny = cosine similarity.
        Zwraca tensor kształtu (B,) — mean() aplikowane w pretrain_step, nie tu.

        Zakres: [0, 4]. Gdy p=z (kolaps): 0. Gdy p=-z (antykorelacja): 4.
        """
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2.0 - 2.0 * (p * z).sum(dim=-1)

    # ------------------------------------------------------------------

    def pretrain_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """batch : (x1, x2) — dwie augmentacje tego samego obrazu.

        Kolejność operacji zgodna z byol_experiment.py:
          1. Forward sieć online  (z gradientem)
          2. Forward sieć target  (bez gradientu, BN w train() → statystyki batcha)
          3. Strata symetryczna: mean(loss_v1 + loss_v2)
             Port z: byol_experiment.py L.243-250 — suma obu widoków, mean po batchu
          4. EMA update sieci target

        Diagnostyka (wykrywanie kolapsu i dywergencji numerycznej):
          - std embeddingów batcha < 0.01 → kolaps (stop_grad z celem, sieć nic nie uczy)
          - NaN/inf w loss → dywergencja numeryczna
        """
        x1, x2 = batch

        # 1. Gałąź online
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1_online = self.projector(h1)
        z2_online = self.projector(h2)
        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)

        # 2. Gałąź target — bez gradientu, BN w train() aktualizuje running stats
        # Port z: byol_experiment.py L.233-236: forward.apply(..., is_training=True)
        with torch.no_grad():
            z1_target = self.projector_t(self.encoder_t(x1))
            z2_target = self.projector_t(self.encoder_t(x2))

        # 3. Strata symetryczna — mean po obu widokach i batchu
        # Port z: byol_experiment.py L.243-250:
        #   repr_loss = regression_loss(pred_v1, sg(proj_v2))
        #             + regression_loss(pred_v2, sg(proj_v1))
        #   repr_loss = jnp.mean(repr_loss)
        per_sample = self._regression_loss(p1, z2_target) + \
                     self._regression_loss(p2, z1_target)
        loss = per_sample.mean()

        # --- Diagnostyka ---
        with torch.no_grad():
            emb_std = torch.cat([
                F.normalize(h1, dim=-1),
                F.normalize(h2, dim=-1),
            ], dim=0).std(dim=0).mean().item()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[BYOLModel] Dywergencja numeryczna: loss={loss.item():.4f}. "
                    "Sprawdź: lr za wysoki, BN wyłączone (eval mode), dane w złym zakresie."
                )
            if emb_std < 0.01:
                import warnings
                warnings.warn(
                    f"[BYOLModel] Wykryto kolaps reprezentacji: "
                    f"std embeddingów = {emb_std:.4f} < 0.01. "
                    "Możliwe przyczyny: τ=1.0 (stop EMA), za mały batch, "
                    "brak augmentacji asymetrycznej.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # 4. EMA update (parametry; bufory BN aktualizowane przez forward powyżej)
        self._ema_update()

        return {"loss": loss, "emb_std": torch.tensor(emb_std)}

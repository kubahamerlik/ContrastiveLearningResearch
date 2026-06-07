"""
image_models.py – Implementacje MoCo i BYOL dla danych obrazowych.

MoCo  : He et al., 2020  – "Momentum Contrast for Unsupervised Visual Representation Learning"
BYOL  : Grill et al., 2020 – "Bootstrap Your Own Latent"
"""

from typing import Any, Dict, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContrastiveModel, build_mlp


# ===========================================================================
# 1. MoCo – Momentum Contrast
# ===========================================================================

class MoCoModel(BaseContrastiveModel):
    """
    MoCo v2 z kolejką negatywnych wektorów i koderem momentum.

    Architektura
    ------------
    encoder_q  : koder online (aktualizowany przez optymalizator)
    encoder_k  : koder momentum (EMA od encoder_q, NIE przez autograd)
    queue      : cykliczny bufor negatywnych wektorów (K próbek × proj_dim)

    Parametry
    ---------
    queue_size  : int   – liczba negatywnych wektorów w kolejce (domyślnie 4096)
    momentum    : float – współczynnik EMA przy aktualizacji kodera k (0–1)
    temperature : float – τ w funkcji InfoNCE (im mniejsza, tym ostrzejszy rozkład)
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
        temperature: float = 0.07,
    ) -> None:
        super().__init__(encoder_type, input_dim, hidden_dim, proj_dim, num_classes)

        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size

        self.encoder_k = copy.deepcopy(self.encoder)
        self.projector_k = copy.deepcopy(self.projector)
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        self.encoder_k.eval()
        self.projector_k.eval()

        # Kolejka: (proj_dim, queue_size) – znormalizowane klucze
        self.register_buffer("queue", F.normalize(
            torch.randn(proj_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def train(self, mode: bool = True) -> "MoCoModel":
        """Koder momentum (encoder_k/projector_k) zawsze pozostaje w eval()."""
        super().train(mode)
        self.encoder_k.eval()
        self.projector_k.eval()
        return self

    @torch.no_grad()
    def _momentum_update(self) -> None:
        """Aktualizuje encoder_k jako EMA od encoder_q."""
        for q_param, k_param in zip(self.encoder.parameters(),
                                    self.encoder_k.parameters()):
            k_param.data = k_param.data * self.momentum + q_param.data * (1.0 - self.momentum)
        for q_param, k_param in zip(self.projector.parameters(),
                                    self.projector_k.parameters()):
            k_param.data = k_param.data * self.momentum + q_param.data * (1.0 - self.momentum)

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
        """batch : (x_q, x_k) – dwie augmentacje tego samego obrazu."""
        x_q, x_k = batch

        h_q = self.encoder(x_q)
        q = F.normalize(self.projector(h_q), dim=-1)

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
    BYOL – uczenie reprezentacji BEZ próbek negatywnych.

    Architektura
    ------------
    Sieć online : encoder → projector → predictor
    Sieć target : encoder_t → projector_t  (EMA od online, bez gradientów)

    Cel: sieć online ma przewidywać projekcje sieci target,
         a sieć target ma przewidywać projekcje sieci online
         (symetryczna strata L2 na znormalizowanych wektorach).
    """

    def __init__(
        self,
        encoder_type: str = "cnn_image",
        input_dim: int = 3,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        pred_dim: int = 64,
        num_classes: int = 10,
        momentum: float = 0.996,
    ) -> None:
        super().__init__(encoder_type, input_dim, hidden_dim, proj_dim, num_classes)

        self.momentum_base = momentum
        self.momentum = momentum

        self.predictor: nn.Module = build_mlp(proj_dim, hidden_dim, proj_dim)

        self.encoder_t = copy.deepcopy(self.encoder)
        self.projector_t = copy.deepcopy(self.projector)
        for param in self.encoder_t.parameters():
            param.requires_grad = False
        for param in self.projector_t.parameters():
            param.requires_grad = False

        self.encoder_t.eval()
        self.projector_t.eval()

    def train(self, mode: bool = True) -> "BYOLModel":
        """Sieć target (encoder_t/projector_t) zawsze pozostaje w eval()."""
        super().train(mode)
        self.encoder_t.eval()
        self.projector_t.eval()
        return self

    def update_momentum(self, step: int, total_steps: int) -> None:
        """Cosine schedule τ od τ_base do 1.0 (Grill et al. 2020, App. F)."""
        import math
        if total_steps <= 1:
            self.momentum = self.momentum_base
            return
        progress = min(max(step / max(1, total_steps - 1), 0.0), 1.0)
        cos_term = (math.cos(math.pi * progress) + 1.0) / 2.0
        self.momentum = 1.0 - (1.0 - self.momentum_base) * cos_term

    @torch.no_grad()
    def _ema_update(self) -> None:
        """Aktualizuje sieć target jako EMA od online."""
        for o, t in zip(self.encoder.parameters(), self.encoder_t.parameters()):
            t.data = t.data * self.momentum + o.data * (1.0 - self.momentum)
        for o, t in zip(self.projector.parameters(), self.projector_t.parameters()):
            t.data = t.data * self.momentum + o.data * (1.0 - self.momentum)

    @staticmethod
    def _regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Znormalizowany MSE między predykcją p a celem z.
        Odpowiednik 2 - 2 * cosine_similarity(p, z).
        """
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()

    # ------------------------------------------------------------------

    def pretrain_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """batch : (x1, x2) – dwie augmentacje tego samego obrazu."""
        x1, x2 = batch

        z1_online = self.projector(self.encoder(x1))
        z2_online = self.projector(self.encoder(x2))

        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)

        with torch.no_grad():
            z1_target = self.projector_t(self.encoder_t(x1))
            z2_target = self.projector_t(self.encoder_t(x2))

        loss = (
            self._regression_loss(p1, z2_target) +
            self._regression_loss(p2, z1_target)
        ) / 2.0

        return {"loss": loss}

"""
timeseries_models.py – Implementacja TS-TCC dla szeregów czasowych.

TS-TCC : Eldele et al., 2021 – "Time-Series Representation Learning via
         Temporal and Contextual Contrasting"
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContrastiveModel, build_mlp


# ===========================================================================
# Moduł autoregresoryjny (Transformer encoder) – serce TS-TCC
# ===========================================================================

class _TemporalEncoder(nn.Module):
    """
    Lekki Transformer do modelowania zależności kontekstowych w szeregu.
    Wejście : (B, T, d_model) – sekwencja osadzeń patch'y
    Wyjście : (B, T, d_model) – kontekstowe reprezentacje
    """

    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, max_seq_len: int = 512) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        self.pos_enc = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T > self.max_seq_len:
            raise ValueError(
                f"Sekwencja długości {T} przekracza max_seq_len={self.max_seq_len}. "
                f"Przekaż max_seq_len>={T} przy inicjalizacji _TemporalEncoder."
            )
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_enc(positions)
        return self.transformer(x)                   # (B, T, d_model)


# ===========================================================================
# Augmentacje szeregów czasowych
# ===========================================================================

def weak_augmentation(
    x: torch.Tensor, jitter_std: float = 0.05, scaling_std: float = 0.1
) -> torch.Tensor:
    """Słaba augmentacja: jitter + scaling per kanał (Eldele et al. 2021)."""
    scaling = 1.0 + torch.randn(x.shape[0], x.shape[1], 1, device=x.device) * scaling_std
    return x * scaling + torch.randn_like(x) * jitter_std


def strong_augmentation(
    x: torch.Tensor,
    jitter_std: float = 0.1,
    permutation_segments: Optional[int] = None,
    max_segments_range: Tuple[int, int] = (5, 12),
) -> torch.Tensor:
    """
    Silna augmentacja: permutacja segmentów + szum Gaussa (Eldele et al. 2021).
    Gdy permutation_segments=None, losuje M ~ U[5, 12] per wywołanie.
    """
    B, C, T = x.shape
    if permutation_segments is None:
        lo, hi = max_segments_range
        hi = min(hi, max(lo, T))
        permutation_segments = int(torch.randint(lo, hi + 1, (1,)).item())
    permutation_segments = max(2, min(permutation_segments, T))
    seg_len = T // permutation_segments
    if seg_len < 1:
        return x + torch.randn_like(x) * jitter_std
    T_perm = permutation_segments * seg_len

    segments = x[:, :, :T_perm].reshape(B, C, permutation_segments, seg_len)

    noise = torch.rand(B, permutation_segments, device=x.device)
    perms = noise.argsort(dim=-1)

    perms_exp = perms[:, None, :, None].expand(B, C, permutation_segments, seg_len)
    segments_perm = segments.gather(dim=2, index=perms_exp)

    x_perm = x.clone()
    x_perm[:, :, :T_perm] = segments_perm.reshape(B, C, T_perm)

    # Dodaj szum Gaussa
    x_perm = x_perm + torch.randn_like(x_perm) * jitter_std
    return x_perm


# ===========================================================================
# TS-TCC
# ===========================================================================

class TSTCCModel(BaseContrastiveModel):
    """
    TS-TCC – dwustopniowe uczenie kontrastowe dla szeregów czasowych:

    Etap 1 – Temporal Contrasting (TC):
        Koder CNN wyodrębnia lokalne cechy.
        Model autoregresoryjny (Transformer) przewiduje przyszłe stany
        jednej augmentacji na podstawie przeszłości drugiej augmentacji
        (cross-prediction). Strata: InfoNCE po osi czasowej.

    Etap 2 – Contextual Contrasting (CC):
        Na globalnych reprezentacjach (CLS-pooling po Transformerze)
        stosujemy zwykłą stratę InfoNCE między słabą a silną augmentacją.

    Parametry
    ---------
    seq_len          : int   – długość szeregu czasowego (liczba kroków)
    in_channels      : int   – liczba kanałów/cech w każdym kroku
    cnn_out_dim      : int   – wymiar wyjściowy CNN kodera
    d_model          : int   – wymiar wewnętrzny Transformera
    tc_timesteps     : int   – ile przyszłych kroków przewiduje TC
    temperature      : float – τ w InfoNCE
    tc_weight        : float – waga straty TC (CC = 1 - tc_weight)
    """

    def __init__(
        self,
        seq_len: int = 128,
        in_channels: int = 1,
        cnn_out_dim: int = 128,
        d_model: int = 128,
        num_classes: int = 2,
        tc_timesteps: int = 8,
        temperature: float = 0.2,
        tc_weight: float = 0.5,
    ) -> None:
        # Koder CNN: (B, in_channels, seq_len) → (B, cnn_out_dim, T')
        # BaseContrastiveModel używa kodera liniowego; nadpisujemy go poniżej.
        super().__init__(
            encoder_type="cnn",
            input_dim=in_channels,          # sygnatura – nie używana bezpośrednio
            hidden_dim=cnn_out_dim,
            proj_dim=cnn_out_dim,
            num_classes=num_classes,
        )

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.tc_timesteps = tc_timesteps
        self.temperature = temperature
        self.tc_weight = tc_weight
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(),
        )
        # T' = output CNN: L1=seq_len+1, L2=floor((L1-1)/2)+1, T'=floor((L2-1)/2)+1
        _L1 = seq_len + 1
        _L2 = (_L1 - 1) // 2 + 1
        _t_prime = (_L2 - 1) // 2 + 1
        _max_seq_len = max(_t_prime + 1, 64)

        # --- Transformer kontekstowy ----------------------------------------
        self.temporal_encoder = _TemporalEncoder(d_model=d_model,
                                                  max_seq_len=_max_seq_len)

        # Dwa osobne predyktory TC (Eldele et al. 2021 Eq. 2):
        # tc_predictor_w: ctx_weak → krok strong, tc_predictor_s: ctx_strong → krok weak
        self.tc_predictor_w = nn.Linear(d_model, d_model)
        self.tc_predictor_s = nn.Linear(d_model, d_model)

        self.projector = build_mlp(d_model, d_model * 2, cnn_out_dim)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_channels, seq_len) → (B, d_model) – globalna reprezentacja."""
        feat = self.encoder(x)                       # (B, d_model, T')
        feat = feat.permute(0, 2, 1)                 # (B, T', d_model)
        ctx = self.temporal_encoder(feat)             # (B, T', d_model)
        return ctx.mean(dim=1)                        # (B, d_model)

    def _tc_loss(
        self, feat_weak: torch.Tensor, feat_strong: torch.Tensor
    ) -> torch.Tensor:
        """
        Temporal Contrasting (Eldele et al. 2021 Eq. 2–3).

        Cross-prediction: ctx_w[t+k] → przewiduje strong[split+k], i odwrotnie.
        Rosnący kontekst emuluje AR bez RNN. InfoNCE per krok czasowy.
        """
        B, T, d = feat_weak.shape
        split = T - self.tc_timesteps
        if split <= 0 or B < 2:
            return torch.tensor(0.0, device=feat_weak.device)

        future_s = feat_strong[:, split:split + self.tc_timesteps, :].detach()
        future_w = feat_weak[:,  split:split + self.tc_timesteps, :].detach()

        total = torch.tensor(0.0, device=feat_weak.device)
        for k in range(self.tc_timesteps):
            ctx_w = feat_weak[:,  :split + k + 1, :].mean(dim=1)
            ctx_s = feat_strong[:, :split + k + 1, :].mean(dim=1)

            pred_w2s = F.normalize(self.tc_predictor_w(ctx_w), dim=-1)
            pred_s2w = F.normalize(self.tc_predictor_s(ctx_s), dim=-1)

            f_s = F.normalize(future_s[:, k, :], dim=-1)
            f_w = F.normalize(future_w[:, k, :], dim=-1)

            labels = torch.arange(B, device=feat_weak.device)
            loss_w2s = F.cross_entropy(
                torch.mm(pred_w2s, f_s.T) / self.temperature, labels
            )
            loss_s2w = F.cross_entropy(
                torch.mm(pred_s2w, f_w.T) / self.temperature, labels
            )
            total = total + (loss_w2s + loss_s2w) / 2.0

        return total / self.tc_timesteps

    @staticmethod
    def _cc_loss(
        z_weak: torch.Tensor, z_strong: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """
        Symetryczna NT-Xent na globalnych reprezentacjach.
        z_weak, z_strong : (B, D) znormalizowane
        """
        B = z_weak.shape[0]
        z = torch.cat([z_weak, z_strong], dim=0)         # (2B, D)
        sim = torch.mm(z, z.T) / temperature

        mask_diag = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_diag, float("-inf"))

        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)

    # ------------------------------------------------------------------

    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """x : (B, in_channels, seq_len) – model generuje weak i strong augmentację."""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        x_weak = weak_augmentation(x)
        x_strong = strong_augmentation(x)

        feat_weak = self.encoder(x_weak).permute(0, 2, 1)
        feat_strong = self.encoder(x_strong).permute(0, 2, 1)

        ctx_weak = self.temporal_encoder(feat_weak)
        ctx_strong = self.temporal_encoder(feat_strong)

        tc = self._tc_loss(ctx_weak, ctx_strong)

        global_weak = ctx_weak.mean(dim=1)
        global_strong = ctx_strong.mean(dim=1)

        z_weak = F.normalize(self.projector(global_weak), dim=-1)
        z_strong = F.normalize(self.projector(global_strong), dim=-1)
        cc = self._cc_loss(z_weak, z_strong, self.temperature)

        loss = self.tc_weight * tc + (1.0 - self.tc_weight) * cc

        return {
            "loss": loss,
            "tc_loss": tc.detach(),
            "cc_loss": cc.detach(),
        }

    def freeze_encoder(self) -> None:
        """Zamraża CNN koder i Transformer (oba używane w forward/linear_readout)."""
        super().freeze_encoder()
        for param in self.temporal_encoder.parameters():
            param.requires_grad = False
        self.temporal_encoder.eval()

    def train(self, mode: bool = True) -> "TSTCCModel":
        """Utrzymuje temporal_encoder w eval() gdy koder jest zamrożony."""
        super().train(mode)
        if self._encoder_frozen:
            self.temporal_encoder.eval()
        return self

    def unfreeze_encoder(self) -> None:
        """Odmraża CNN koder i Transformer."""
        super().unfreeze_encoder()
        for param in self.temporal_encoder.parameters():
            param.requires_grad = True

    def linear_readout(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_channels, seq_len)"""
        with torch.set_grad_enabled(not self._encoder_frozen):
            h = self.forward(x)
        return self.classifier(h)

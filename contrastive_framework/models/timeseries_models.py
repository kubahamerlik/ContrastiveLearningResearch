"""
timeseries_models.py – Wierna implementacja TS-TCC.

TS-TCC : Eldele et al., 2021 – "Time-Series Representation Learning via
         Temporal and Contextual Contrasting"
         Źródło: https://github.com/emadeldeen24/TS-TCC (MIT License)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContrastiveModel


# ===========================================================================
# Augmentacje szeregów czasowych
# Port z: third_party_reference/TS-TCC/dataloader/augmentations.py
# ===========================================================================

def weak_augmentation(x: torch.Tensor, jitter_scale_ratio: float = 1.1) -> torch.Tensor:
    """
    Słaba augmentacja: skalowanie per kanał per czas (Eldele 2021 Sec. 4.1).
    Współczynnik ~ N(2, jitter_scale_ratio), niezależnie dla każdego (B, C, T).
    """
    # scale: (B, 1, T) — wspólny dla wszystkich kanałów (jak w referencji: augmentations.py).
    # Oryginalny kod: torch.from_numpy(np.random.normal(loc=2.0, scale=jitter_scale_ratio,
    #                                                     size=(x.shape[0], x.shape[2])))
    # Kształt (B, T) → (B, 1, T) → broadcast po C.
    scale = (torch.randn(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
             * jitter_scale_ratio + 2.0)
    return x * scale


def strong_augmentation(
    x: torch.Tensor,
    jitter_ratio: float = 0.8,
    max_seg: int = 8,
) -> torch.Tensor:
    """
    Silna augmentacja: permutacja(max_seg) + jitter (Eldele 2021 Sec. 4.1).
    Port z augmentations.py – vectorized on tensor (GPU-friendly).
    """
    B, C, T = x.shape

    # --- Permutacja segmentów ---
    # Dla każdej próbki losuj n_segs ∈ [1, max_seg-1] i podziel T na n_segs
    # Uproszczenie wektoryzowane: wspólna liczba segmentów dla batcha,
    # ale kolejność segmentów niezależna dla każdej próbki (jak w oryginale).
    # n_segs per-sample (jak w referencji: augmentations.py używa size=(x.shape[0],)).
    # Każda próbka w batchu może mieć inną liczbę segmentów.
    n_segs_per = torch.randint(1, max_seg, size=(B,), device=x.device)  # (B,)
    # Dla wektoryzacji używamy mediany jako wspólnego n_segs
    # (perm_idx i tak jest per-sample). Rzeczywisty efekt per-sample jest przez perm_idx.
    n_segs = int(n_segs_per.float().mean().round().item())
    n_segs = max(1, min(n_segs, T))

    seg_len = T // n_segs
    if seg_len < 1 or n_segs == 1:
        # Za krótki szereg lub 1 segment – tylko jitter
        return x + torch.randn_like(x) * jitter_ratio

    T_perm = n_segs * seg_len  # przycinamy do wielokrotności seg_len

    # (B, C, n_segs, seg_len)
    segs = x[:, :, :T_perm].reshape(B, C, n_segs, seg_len)

    # Niezależna permutacja dla każdej próbki w batchu
    # (B, n_segs) – losowa kolejność segmentów
    perm_idx = torch.argsort(torch.rand(B, n_segs, device=x.device), dim=1)
    # Rozszerzamy do (B, C, n_segs, seg_len)
    perm_idx_exp = perm_idx[:, None, :, None].expand(B, C, n_segs, seg_len)
    segs_perm = segs.gather(dim=2, index=perm_idx_exp)

    x_perm = x.clone()
    x_perm[:, :, :T_perm] = segs_perm.reshape(B, C, T_perm)

    # Jitter
    return x_perm + torch.randn_like(x_perm) * jitter_ratio


# ===========================================================================
# Seq_Transformer (Temporal Contrasting)
# Port 1:1 z: third_party_reference/TS-TCC/models/attention.py
# Zmiany względem oryginału: brak einops (zastąpione natywnym torch), reszta
# identyczna: Pre-LN (Residual+PreNorm), scale=dim**-0.5, to_qkv bias=False,
# dim_feedforward=mlp_dim (NIE mlp_dim*4).
# ===========================================================================

class _Residual(nn.Module):
    """Residual wrapper — Port z TS-TCC/models/attention.py."""
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class _PreNorm(nn.Module):
    """Pre-LayerNorm wrapper — Port z TS-TCC/models/attention.py."""
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class _FeedForward(nn.Module):
    """FF: dim→hidden_dim→dim z ReLU — Port z TS-TCC/models/attention.py."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Attention(nn.Module):
    """Multi-head attention — Port z TS-TCC/models/attention.py.
    scale=dim**-0.5; to_qkv bias=False (jak w referencji).
    einops zastąpione natywnym view/permute/einsum."""
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        h = self.heads
        qkv = self.to_qkv(x)                        # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, h, D // h)       # (B, N, 3, h, d_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # (3, B, h, N, d_h)
        q, k, v = qkv[0], qkv[1], qkv[2]            # each: (B, h, N, d_h)

        # dots: (B, h, N, N)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float("-inf"))

        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)  # (B, h, N, d_h)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # (B, N, D)
        return self.to_out(out)


class _Transformer(nn.Module):
    """Pre-LN Transformer — Port z TS-TCC/models/attention.py (klasa Transformer).
    dim_feedforward = mlp_dim (nie mlp_dim*4)."""
    def __init__(self, dim: int, depth: int, heads: int,
                 mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                _Residual(_PreNorm(dim, _Attention(dim, heads=heads, dropout=dropout))),
                _Residual(_PreNorm(dim, _FeedForward(dim, mlp_dim, dropout=dropout))),
            ]))

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class _SeqTransformer(nn.Module):
    """
    Transformer z c_token i patch_to_embedding.
    Wejście : (B, T, patch_size)  gdzie patch_size = final_out_channels
    Wyjście : (B, dim)            gdzie dim = tc_hidden_dim
    Port 1:1 z: TS-TCC/models/attention.py (Seq_Transformer)
    """

    def __init__(self, patch_size: int, dim: int, depth: int = 4, heads: int = 4,
                 mlp_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_to_embedding = nn.Linear(patch_size, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = _Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, patch_size) → c_t: (B, dim)"""
        x = self.patch_to_embedding(x)              # (B, T, dim)
        B = x.size(0)
        c = self.c_token.expand(B, -1, -1)          # (B, 1, dim)
        x = torch.cat([c, x], dim=1)                # (B, T+1, dim)
        x = self.transformer(x)                     # (B, T+1, dim)
        return self.to_c_token(x[:, 0])             # c_t: (B, dim)


# ===========================================================================
# TC – Temporal Contrasting
# Port z: third_party_reference/TS-TCC/models/TC.py
# ===========================================================================

class _TC(nn.Module):
    """
    Temporal Contrasting (Eldele 2021).
    Cross-view: z_aug1 jako kontekst przeszłości, z_aug2 jako cele przyszłości.
    Port z: TS-TCC/models/TC.py
    """

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        hidden_dim: int = 100,
        timesteps: int = 6,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels

        # patch_size = num_channels (wymiar CNN features), dim = hidden_dim (100)
        self.seq_transformer = _SeqTransformer(
            patch_size=num_channels, dim=hidden_dim,
            depth=4, heads=4, mlp_dim=64,
        )

        # W_k: osobna głowica predykcji dla każdego kroku czasowego
        # hidden_dim → num_channels (wymiar CNN features)
        self.Wk = nn.ModuleList([
            nn.Linear(hidden_dim, num_channels)
            for _ in range(timesteps)
        ])

        # projection_head: num_channels → num_channels//2 → num_channels//4
        # Port z TC.py (używane do contextual contrasting)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, num_channels // 2),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // 2, num_channels // 4),
        )

    def forward(
        self, z_aug1: torch.Tensor, z_aug2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_aug1, z_aug2 : (B, T', num_channels) – znormalizowane CNN features

        Zwraca:
            nce   : strata TC (skalarna)
            proj  : (B, num_channels//4) – reprezentacja do CC
        """
        B, T_prime, _ = z_aug1.shape

        # Losowy indeks t ∈ [0, T' - timesteps)
        t_samples = int(torch.randint(T_prime - self.timesteps, size=(1,)).item())

        # Cele przyszłości z aug2: kroki t+1, ..., t+timesteps
        # shape: (timesteps, B, num_channels)
        encode_samples = z_aug2[:, t_samples + 1: t_samples + self.timesteps + 1, :]
        encode_samples = encode_samples.permute(1, 0, 2)  # (timesteps, B, num_channels)

        # Kontekst przeszłości z aug1: kroki 0..t (causal)
        forward_seq = z_aug1[:, :t_samples + 1, :]  # (B, t+1, num_channels)

        # Transformer zwraca c_t: (B, hidden_dim)
        c_t = self.seq_transformer(forward_seq)

        # NCE loss (Eq. 2 w Eldele 2021)
        nce = torch.tensor(0.0, device=z_aug1.device)
        for i in range(self.timesteps):
            pred_i = self.Wk[i](c_t)                              # (B, num_channels)
            # LogSoftmax over batch dimension (jak w oryginale)
            logits = torch.mm(encode_samples[i], pred_i.T)        # (B, B)
            log_probs = F.log_softmax(logits, dim=1)
            # Poprawna klasa: i-ty próbka matchuje i-ty cel → diagonala
            nce = nce + log_probs.diag().sum()

        nce = nce / (-B * self.timesteps)

        # Projekcja c_t do przestrzeni CC
        proj = self.projection_head(c_t)                          # (B, num_channels//4)

        return nce, proj


# ===========================================================================
# NT-Xent Loss
# Port z: third_party_reference/TS-TCC/models/loss.py
# ===========================================================================

class _NTXentLoss(nn.Module):
    """
    NT-Xent loss z cosine similarity (Eldele 2021 Sec. 4.2).
    Port z: TS-TCC/models/loss.py
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature
        # Maska self-par → rejestrujemy jako buffer (nie parametr),
        # odbudowujemy gdy rozmiar batcha się zmienia
        self._cached_B: Optional[int] = None
        self._mask: Optional[torch.Tensor] = None

    def _get_mask(self, B: int, device: torch.device) -> torch.Tensor:
        if self._cached_B != B or (self._mask is not None and self._mask.device != device):
            # Maska 2B×2B – eliminuje pary (i,i): self-pary
            mask = (~torch.eye(2 * B, dtype=torch.bool, device=device)).float()
            self._cached_B = B
            self._mask = mask
        return self._mask  # type: ignore[return-value]

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2 : (B, D) – znormalizowane projekcje"""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)             # (2B, D)
        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.T) / self.temperature   # (2B, 2B)

        # Poprawne pary: (i, B+i) i (B+i, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        # Wykluczamy self-pary (diagonalę) ustawiając -inf
        mask_diag = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_diag, float("-inf"))

        # CrossEntropyLoss(reduction="sum") / (2*B) jak w oryginale
        loss = F.cross_entropy(sim, labels, reduction="sum") / (2 * B)
        return loss


# ===========================================================================
# base_Model – koder CNN
# Port z: third_party_reference/TS-TCC/models/model.py
# ===========================================================================

class _BaseModel(nn.Module):
    """
    Trójblokowy koder CNN dla szeregów czasowych.
    Wejście : (B, in_channels, seq_len)
    Wyjście : (B, final_out_channels, T')  – CNN features przed flatten
    Port z: TS-TCC/models/model.py (base_Model)
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 8,
        stride: int = 1,
        final_out_channels: int = 128,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        # block 1: Dropout 0.35 tylko w pierwszym bloku (jak w oryginale)
        # bias=False — referencja: TS-TCC/models/model.py, wszystkie Conv1d mają bias=False
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, T) → (B, final_out_channels, T')"""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


def _compute_features_len(seq_len: int, kernel_size: int = 8, stride: int = 1) -> int:
    """Oblicza T' (długość po CNN) dla danego seq_len."""
    padding = kernel_size // 2
    # block1: Conv1d(k, s, p) + MaxPool(k=2, s=2, p=1)
    L = (seq_len + 2 * padding - kernel_size) // stride + 1
    L = (L + 2 * 1 - 2) // 2 + 1
    # block2: Conv1d(k=8,s=1,p=4) + MaxPool
    L = (L + 2 * 4 - 8) // 1 + 1
    L = (L + 2 * 1 - 2) // 2 + 1
    # block3: Conv1d(k=8,s=1,p=4) + MaxPool
    L = (L + 2 * 4 - 8) // 1 + 1
    L = (L + 2 * 1 - 2) // 2 + 1
    return L


# ===========================================================================
# TSTCCModel
# ===========================================================================

class TSTCCModel(BaseContrastiveModel):
    """
    TS-TCC – wierna reimplementacja Eldele et al. 2021.

    Architektura zgodna z HAR_Configs.py:
      kernel_size=8, stride=1, final_out_channels=128, dropout=0.35,
      TC.hidden_dim=100, TC.timesteps=6, temperature=0.2

    Strata: (TC1 + TC2)*λ1 + NTXent(proj1, proj2)*λ2
    Domyślne wagi: λ1=1, λ2=0.7 (z trainer.py, są stałe – nie HPO)

    Encoder do linear probing: sflattened CNN features (B, C*T'),
    NIE reprezentacja Transformera.
    """

    def __init__(
        self,
        seq_len: int = 128,
        in_channels: int = 9,
        # Parametry CNN (HAR_Configs.py defaults)
        kernel_size: int = 8,
        stride: int = 1,
        final_out_channels: int = 128,
        dropout: float = 0.35,
        # TC
        tc_hidden_dim: int = 100,
        tc_timesteps: int = 6,
        # Contextual Contrasting
        temperature: float = 0.2,
        # Klasy
        num_classes: int = 6,
        # Nieużywane przez BaseContrastiveModel – wymagane przez interfejs
        hidden_dim: int = 128,
        proj_dim: int = 32,
    ) -> None:
        # Obliczamy T' przed inicjalizacją (potrzebne do classifier)
        features_len = _compute_features_len(seq_len, kernel_size, stride)
        flat_dim = final_out_channels * features_len

        super().__init__(
            encoder_type="cnn",
            input_dim=in_channels,
            hidden_dim=final_out_channels,
            proj_dim=proj_dim,
            num_classes=num_classes,
        )

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.final_out_channels = final_out_channels
        self.features_len = features_len
        self.temperature = temperature

        # Stałe wagi straty (z trainer.py – nie są HPO)
        self._lambda1 = 1.0   # TC weight
        self._lambda2 = 0.7   # CC weight

        # --- Encoder CNN (base_Model) ---
        self.encoder = _BaseModel(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            final_out_channels=final_out_channels,
            dropout=dropout,
        )

        # --- TC ---
        self.tc = _TC(
            num_channels=final_out_channels,
            seq_len=features_len,
            hidden_dim=tc_hidden_dim,
            timesteps=tc_timesteps,
        )

        # --- NT-Xent ---
        self.ntxent = _NTXentLoss(temperature=temperature)

        # --- Classifier na flattened CNN features ---
        # Nadpisujemy classifier z BaseContrastiveModel
        self.classifier = nn.Linear(flat_dim, num_classes)

    # ------------------------------------------------------------------
    # Interfejs BaseContrastiveModel
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_channels, seq_len)
        Zwraca flattened CNN features: (B, final_out_channels * features_len)
        Używane przez linear_readout do klasyfikacji.
        """
        feat = self.encoder(x)   # (B, C, T')
        return feat.flatten(1)   # (B, C*T')

    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        x : (B, in_channels, seq_len)
        Generuje weak (aug1) i strong (aug2) augmentacje i liczy stratę TS-TCC.
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        x_aug1 = weak_augmentation(x)
        x_aug2 = strong_augmentation(x)

        # CNN features: (B, C, T')
        feat1 = self.encoder(x_aug1)
        feat2 = self.encoder(x_aug2)

        # Normalizacja przed TC (z trainer.py: F.normalize(features, dim=1))
        # dim=1 → normalizacja po kanałach (C), niezależnie dla każdego (B, T')
        feat1_n = F.normalize(feat1, dim=1)
        feat2_n = F.normalize(feat2, dim=1)

        # Permutujemy do (B, T', C) dla TC i Seq_Transformer
        feat1_t = feat1_n.permute(0, 2, 1)  # (B, T', C)
        feat2_t = feat2_n.permute(0, 2, 1)  # (B, T', C)

        # TC: cross-view (aug1 → kontekst dla aug2 i odwrotnie)
        # Sprawdzamy czy T' > timesteps (może być za krótki przy subsample)
        tc_loss = torch.tensor(0.0, device=x.device)
        proj1 = proj2 = None

        if feat1_t.size(1) > self.tc.timesteps:
            tc1_loss, proj1 = self.tc(feat1_t, feat2_t)
            tc2_loss, proj2 = self.tc(feat2_t, feat1_t)
            tc_loss = tc1_loss + tc2_loss

        # Contextual Contrasting (NT-Xent na projekcjach z TC)
        cc_loss = torch.tensor(0.0, device=x.device)
        if proj1 is not None and proj2 is not None and x.size(0) > 1:
            cc_loss = self.ntxent(proj1, proj2)

        loss = tc_loss * self._lambda1 + cc_loss * self._lambda2

        return {
            "loss": loss,
            "tc_loss": tc_loss.detach(),
            "cc_loss": cc_loss.detach(),
        }

    def linear_readout(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_channels, seq_len)"""
        with torch.set_grad_enabled(not self._encoder_frozen):
            h = self.forward(x)   # (B, C*T') – CNN features, nie Transformer
        return self.classifier(h)

    def freeze_encoder(self) -> None:
        """Zamraża CNN encoder."""
        super().freeze_encoder()

    def unfreeze_encoder(self) -> None:
        """Odmraża CNN encoder."""
        super().unfreeze_encoder()


# ===========================================================================
# RandomInitHARModel
# ===========================================================================

class RandomInitHARModel(BaseContrastiveModel):
    """
    Baseline z losową inicjalizacją dla UCI HAR.
    Używa identycznej architektury CNN co TSTCCModel (base_Model),
    żeby porównanie było uczciwe.

    Pretrain: brak (pusta strata).
    Linear probing: flattened CNN features (B, C*T').
    """

    def __init__(
        self,
        seq_len: int = 128,
        in_channels: int = 9,
        kernel_size: int = 8,
        stride: int = 1,
        final_out_channels: int = 128,
        dropout: float = 0.35,
        num_classes: int = 6,
        # Aliasy akceptowane przez BaseContrastiveModel / HPO
        hidden_dim: int = 128,
        proj_dim: int = 32,
        # Nieużywane – przyjmujemy dla zgodności z filter_model_kwargs
        encoder_type: str = "cnn",
        input_dim: Optional[int] = None,
    ) -> None:
        features_len = _compute_features_len(seq_len, kernel_size, stride)
        flat_dim = final_out_channels * features_len

        # input_dim z BaseContrastiveModel – używamy in_channels
        super().__init__(
            encoder_type="cnn",
            input_dim=in_channels,
            hidden_dim=final_out_channels,
            proj_dim=proj_dim,
            num_classes=num_classes,
        )

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.features_len = features_len
        self.final_out_channels = final_out_channels

        # Identyczna architektura co TSTCCModel
        self.encoder = _BaseModel(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            final_out_channels=final_out_channels,
            dropout=dropout,
        )

        self.classifier = nn.Linear(flat_dim, num_classes)

    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Brak pretrainingu – zwraca zero loss."""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return {"loss": torch.tensor(0.0, device=x.device, requires_grad=True)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_channels, seq_len) → (B, C*T')"""
        return self.encoder(x).flatten(1)

    def linear_readout(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self._encoder_frozen):
            h = self.forward(x)
        return self.classifier(h)

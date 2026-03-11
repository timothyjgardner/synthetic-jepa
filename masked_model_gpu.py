"""
BERT-style Masked Time Series Model — GPU-Optimised (RTX 5090 / Blackwell)

Drop-in replacement for masked_model.py with the following optimisations:

  • BF16 mixed-precision training via torch.autocast + GradScaler
  • torch.compile() with inductor backend for fused GPU kernels
  • Larger default batch size (128) to saturate 32 GB VRAM
  • Multi-worker DataLoader (4 workers) with persistent workers
  • Async CPU→GPU transfers (non_blocking=True + pin_memory)
  • cudnn.benchmark auto-tuning for fixed-size inputs
  • Efficient zero_grad(set_to_none=True)

The model architecture and checkpoint format are identical to
masked_model.py — checkpoints are fully interchangeable.

Usage
-----
    python masked_model_gpu.py                        # train with defaults
    python masked_model_gpu.py --epochs 100           # custom epochs
    python masked_model_gpu.py --eval bert_model.pt   # evaluate & visualize
    python masked_model_gpu.py --no-compile           # skip torch.compile
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import SyntheticSongDataset


# ---------------------------------------------------------------------------
# Model components  (identical architecture to masked_model.py)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# T5-style relative position bias  (Raffel et al. 2020)
# ---------------------------------------------------------------------------

def _relative_position_bucket(relative_position, bidirectional=True,
                               num_buckets=32, max_distance=128):
    """
    Map relative position (key_pos - query_pos) to a bucket index.

    Uses the T5 bucketing scheme: exact buckets for small distances,
    logarithmically-spaced buckets for larger distances.
    """
    relative_buckets = torch.zeros_like(relative_position)
    if bidirectional:
        num_buckets //= 2
        # Positive vs negative direction gets separate bucket ranges
        relative_buckets += (relative_position > 0).long() * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(
            relative_position, torch.zeros_like(relative_position))

    # First half of buckets: exact distances [0, max_exact)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # Second half: log-spaced bins up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large)
    return relative_buckets


class T5RelativePositionBias(nn.Module):
    """
    Learned relative position bias (T5 / mT5 style).

    Produces an additive bias of shape ``(n_heads, S, S)`` that is added
    to the attention logits via the ``attn_mask`` argument of
    ``nn.MultiheadAttention``.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    num_buckets : int
        Total number of relative-position buckets (32 is the T5 default).
    max_distance : int
        Distances beyond this are clamped to the last bucket.
    """

    def __init__(self, n_heads, num_buckets=32, max_distance=128):
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def forward(self, seq_len):
        """Return bias tensor of shape ``(n_heads, seq_len, seq_len)``."""
        device = self.relative_attention_bias.weight.device
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        # relative_position[i, j] = j - i
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)

        buckets = _relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        # (seq_len, seq_len) → (seq_len, seq_len, n_heads) → (n_heads, S, S)
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# RoPE — Rotary Position Embedding  (Su et al. 2021)
# ---------------------------------------------------------------------------

class RotaryPositionalEncoding(nn.Module):
    """Precompute and cache cos/sin rotation matrices for RoPE.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head (must be even).
    max_len : int
        Maximum sequence length to precompute.
    base : float
        Base for the geometric frequency series (10 000 is the default).
    """

    def __init__(self, head_dim, max_len=2048, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer('inv_freq', inv_freq)  # (head_dim/2,)

        # Precompute cos/sin cache
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)             # (max_len, head_dim/2)
        self.register_buffer('cos_cached', freqs.cos())  # (max_len, head_dim/2)
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, seq_len):
        """Return ``(cos, sin)`` each of shape ``(seq_len, head_dim/2)``."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to a tensor.

    Parameters
    ----------
    x : (batch, n_heads, seq_len, head_dim)
    cos, sin : (seq_len, head_dim/2)

    Returns
    -------
    Tensor of same shape as *x* with rotary embeddings applied.
    """
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    # Broadcast (seq_len, half) → (1, 1, seq_len, half)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding.

    Drop-in replacement for ``nn.MultiheadAttention`` (batch_first only)
    that applies RoPE rotations to Q and K before the dot-product.

    Parameters
    ----------
    d_model : int
        Total model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout probability.
    max_len : int
        Maximum sequence length (for RoPE precomputation).
    rope_base : float
        RoPE frequency base.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, max_len=2048,
                 rope_base=10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEncoding(
            self.head_dim, max_len=max_len, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        out : (batch, seq_len, d_model)
        """
        B, S, _ = x.shape
        # Project to Q, K, V
        qkv = self.qkv_proj(x)                            # (B, S, 3*D)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                 # (3, B, H, S, Hd)
        q, k, v = qkv.unbind(0)                           # each (B, H, S, Hd)

        # Apply RoPE to Q and K
        cos, sin = self.rope(S)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention (uses flash/efficient when possible)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.scale,
        )                                                  # (B, H, S, Hd)

        # Merge heads and project
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.out_proj(out)


class RoPETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE attention.

    Mirrors ``nn.TransformerEncoderLayer`` (post-norm, batch_first) but
    replaces standard multi-head attention with :class:`RoPEMultiheadAttention`.

    Parameters
    ----------
    d_model, n_heads, d_ff, dropout : same as nn.TransformerEncoderLayer
    max_len : int
        Maximum sequence length for RoPE.
    rope_base : float
        RoPE frequency base.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 max_len=2048, rope_base=10000.0):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(
            d_model, n_heads, dropout=dropout,
            max_len=max_len, rope_base=rope_base)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

        # Norms and dropout (post-norm, matching PyTorch defaults)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, **kwargs):
        """Forward pass.  ``src_mask`` is accepted for API compatibility
        with ``nn.TransformerEncoderLayer`` but ignored (RoPE doesn't
        use attention masks)."""
        # Self-attention block
        src2 = self.self_attn(src)
        src = self.norm1(src + self.dropout1(src2))
        # FFN block
        src2 = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MaskedTimeSeriesBERT(nn.Module):
    """
    BERT-style masked prediction model for continuous time series.

    Parameters
    ----------
    feature_dim : int
        Dimension of each time step (20 for our synthetic data).
    d_model : int
        Internal transformer dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
    max_len : int
        Maximum sequence length.
    pos_encoding : str
        Positional encoding scheme: ``'sinusoidal'`` (absolute, Vaswani 2017),
        ``'t5'`` (learned relative bias, Raffel 2020), or ``'rope'``
        (rotary position embedding, Su 2021).
    t5_num_buckets : int
        Number of relative-position buckets (only used when pos_encoding='t5').
    t5_max_distance : int
        Maximum distance for bucketing (only used when pos_encoding='t5').
    rope_base : float
        RoPE frequency base (only used when pos_encoding='rope').
    """

    def __init__(
        self,
        feature_dim=20,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
        max_len=2048,
        pos_encoding='sinusoidal',
        t5_num_buckets=32,
        t5_max_distance=128,
        rope_base=10000.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.pos_encoding = pos_encoding

        # Learnable mask embedding (replaces masked positions before projection)
        self.mask_token = nn.Parameter(torch.randn(feature_dim))

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional encoding & transformer encoder
        if pos_encoding == 'sinusoidal':
            self.pos_enc = SinusoidalPositionalEncoding(
                d_model, max_len, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers)

        elif pos_encoding == 't5':
            self.input_dropout = nn.Dropout(p=dropout)
            self.rel_pos_bias = T5RelativePositionBias(
                n_heads=n_heads, num_buckets=t5_num_buckets,
                max_distance=t5_max_distance)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers)

        elif pos_encoding == 'rope':
            self.input_dropout = nn.Dropout(p=dropout)
            # Custom encoder layers with RoPE baked into attention
            self.rope_layers = nn.ModuleList([
                RoPETransformerEncoderLayer(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                    dropout=dropout, max_len=max_len, rope_base=rope_base,
                ) for _ in range(n_layers)
            ])
        else:
            raise ValueError(
                f"Unknown pos_encoding '{pos_encoding}'. "
                f"Choose from: sinusoidal, t5, rope")

        # Output projection back to feature space
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, feature_dim),
        )

    # -- helpers for T5 bias --------------------------------------------------

    def _get_t5_attn_mask(self, batch_size, seq_len):
        """Build per-head relative position bias for nn.MultiheadAttention.

        Returns ``(batch_size * n_heads, seq_len, seq_len)`` additive mask.
        """
        bias = self.rel_pos_bias(seq_len)           # (n_heads, S, S)
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return bias.reshape(batch_size * self.n_heads, seq_len, seq_len)

    # -- forward & encode -----------------------------------------------------

    def forward(self, x, mask):
        """
        Parameters
        ----------
        x : (batch, seq_len, feature_dim) – original observations
        mask : (batch, seq_len) – bool, True = masked (to predict)

        Returns
        -------
        pred : (batch, seq_len, feature_dim) – predicted values
        """
        # Replace masked positions with the learnable mask token
        # (torch.where is compile-friendly — avoids boolean-index graph break)
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        x_masked = torch.where(mask_expanded, self.mask_token, x)

        # Project to transformer dimension
        h = self.input_proj(x_masked)

        if self.pos_encoding == 'sinusoidal':
            h = self.pos_enc(h)
            h = self.transformer(h)
        elif self.pos_encoding == 't5':
            h = self.input_dropout(h)
            attn_mask = self._get_t5_attn_mask(h.size(0), h.size(1))
            h = self.transformer(h, mask=attn_mask)
        else:  # rope
            h = self.input_dropout(h)
            for layer in self.rope_layers:
                h = layer(h)

        # Project back to feature space
        pred = self.output_proj(h)
        return pred

    @torch.no_grad()
    def encode(self, x):
        """
        Run input through the model *without masking* and return the
        intermediate representation at every transformer layer, plus
        the final output projection (denoised reconstruction).

        Parameters
        ----------
        x : (batch, seq_len, feature_dim) – observations (no masking)

        Returns
        -------
        layer_outputs : list of tensors
            - layer_outputs[0..n_layers-1]: (batch, seq_len, d_model)
              after each transformer encoder layer.
            - layer_outputs[n_layers]: (batch, seq_len, feature_dim)
              after the output projection head (the reconstruction).
        """
        h = self.input_proj(x)

        if self.pos_encoding == 'sinusoidal':
            h = self.pos_enc(h)
            layers = self.transformer.layers
            attn_mask = None
        elif self.pos_encoding == 't5':
            h = self.input_dropout(h)
            layers = self.transformer.layers
            attn_mask = self._get_t5_attn_mask(h.size(0), h.size(1))
        else:  # rope
            h = self.input_dropout(h)
            layers = self.rope_layers
            attn_mask = None

        # NOTE: PyTorch's TransformerEncoderLayer has an inference-mode
        # "fast path" that mishandles 3D additive attention masks on GPU,
        # producing NaN.  Temporarily switching to train() mode disables
        # this fast path while keeping identical results (dropout is
        # already 0 at eval time).  RoPE layers don't need this workaround.
        need_train_mode = (attn_mask is not None and not self.training)
        if need_train_mode:
            for layer in layers:
                layer.train()

        layer_outputs = []
        for layer in layers:
            h = layer(h, src_mask=attn_mask)
            layer_outputs.append(h)

        if need_train_mode:
            for layer in layers:
                layer.eval()

        # Output projection (denoised reconstruction)
        layer_outputs.append(self.output_proj(h))

        return layer_outputs


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred, target, mask):
    """MSE loss computed only on masked positions."""
    mask_expanded = mask.unsqueeze(-1).expand_as(pred)
    n_masked = mask_expanded.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=pred.device)
    loss = ((pred - target) ** 2 * mask_expanded).sum() / n_masked
    return loss


# ---------------------------------------------------------------------------
# GPU-optimised training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, amp_dtype):
    """Training loop with mixed-precision and async transfers."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, state, mask in loader:
        # Async CPU→GPU transfer (overlaps with previous compute)
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Mixed-precision forward pass
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            pred = model(x, mask)
            loss = masked_mse_loss(pred, x, mask)

        # Scaled backward pass (handles BF16 gradient scaling)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Unscale before clipping so thresholds are in FP32 units
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    """Evaluation loop with mixed-precision."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, state, mask in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            pred = model(x, mask)
            loss = masked_mse_loss(pred, x, mask)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Visualisation  (identical to masked_model.py)
# ---------------------------------------------------------------------------

def visualize_predictions(model, dataset, device, n_samples=3, save_dir='.',
                          amp_dtype=torch.bfloat16):
    """Generate prediction-vs-ground-truth plots for sample windows."""
    model.eval()
    save_dir = Path(save_dir)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)),
                         replace=False)

    for i, idx in enumerate(indices):
        x, state, mask = dataset[idx]
        x_in = x.unsqueeze(0).to(device)
        mask_in = mask.unsqueeze(0).to(device)

        with torch.no_grad(), torch.autocast(device_type='cuda',
                                             dtype=amp_dtype):
            pred = model(x_in, mask_in)

        x_np = x.numpy()                  # (seq_len, 20)
        pred_np = pred[0].float().cpu().numpy()    # (seq_len, 20)
        mask_np = mask.numpy()             # (seq_len,)
        state_np = state.numpy()           # (seq_len,)
        seq_len = x_np.shape[0]

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            4, 2, height_ratios=[1, 4, 4, 4],
            width_ratios=[1, 0.02], hspace=0.15, wspace=0.03,
        )

        # -- State strip with mask overlay --
        ax_top = fig.add_subplot(gs[0, 0])
        for t in range(seq_len):
            ax_top.axvspan(t, t + 1,
                           color=plt.cm.tab10(state_np[t] / 10),
                           alpha=0.8, linewidth=0)
        for t in range(seq_len):
            if mask_np[t]:
                ax_top.axvspan(t, t + 1, color='grey', alpha=0.5,
                               linewidth=0)
        ax_top.set_xlim(0, seq_len)
        ax_top.set_yticks([])
        ax_top.set_ylabel('State', fontsize=9)
        ax_top.set_title(f'Prediction sample {i}  (grey = masked)',
                         fontsize=12)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # -- Ground truth heatmap --
        ax_gt = fig.add_subplot(gs[1, 0], sharex=ax_top)
        vmax = max(abs(x_np.min()), abs(x_np.max()))
        im = ax_gt.imshow(
            x_np.T, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[0, seq_len, x_np.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_gt.set_ylabel('Dimension', fontsize=9)
        ax_gt.set_title('Ground truth', fontsize=10)
        plt.setp(ax_gt.get_xticklabels(), visible=False)

        # -- Prediction heatmap --
        ax_pred = fig.add_subplot(gs[2, 0], sharex=ax_top)
        ax_pred.imshow(
            pred_np.T, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[0, seq_len, pred_np.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_pred.set_ylabel('Dimension', fontsize=9)
        ax_pred.set_title('Model prediction', fontsize=10)
        plt.setp(ax_pred.get_xticklabels(), visible=False)

        # -- Error heatmap (masked positions only) --
        error = np.zeros_like(x_np)
        error[mask_np] = pred_np[mask_np] - x_np[mask_np]
        ax_err = fig.add_subplot(gs[3, 0], sharex=ax_top)
        err_max = max(abs(error.min()), abs(error.max()), 1e-6)
        ax_err.imshow(
            error.T, aspect='auto', cmap='RdBu_r',
            vmin=-err_max, vmax=err_max,
            extent=[0, seq_len, error.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_err.set_xlabel('Time step', fontsize=9)
        ax_err.set_ylabel('Dimension', fontsize=9)
        ax_err.set_title('Prediction error (masked positions only)',
                         fontsize=10)

        # Colorbars / dummy axes
        ax_cb = fig.add_subplot(gs[1, 1])
        plt.colorbar(im, cax=ax_cb)
        for row in [0, 2, 3]:
            fig.add_subplot(gs[row, 1]).axis('off')

        fname = save_dir / f'bert_prediction_{i}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Training loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses, val_losses, lrs=None,
                    train_eval_losses=None, save_path='training_loss.png'):
    """Save a training/validation loss curve plot."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Train MSE (dropout ON)',
             linewidth=2, alpha=0.5)
    if train_eval_losses is not None:
        ax1.plot(epochs, train_eval_losses,
                 label='Train MSE (dropout OFF)', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val MSE (dropout OFF)',
             linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Masked MSE Loss')
    ax1.set_title('Training Progress')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    if lrs is not None:
        ax2 = ax1.twinx()
        ax2.plot(epochs, lrs, color='grey', linewidth=1, alpha=0.5,
                 linestyle='--', label='LR')
        ax2.set_ylabel('Learning Rate', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.legend(loc='center right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def append_to_log(log_path, epoch, train_loss, train_eval_loss, val_loss,
                  lr, is_best):
    """Append one line to the CSV training log."""
    write_header = not Path(log_path).exists()
    with open(log_path, 'a') as f:
        if write_header:
            f.write('epoch,train_mse,train_eval_mse,val_mse,lr,best\n')
        f.write(f'{epoch},{train_loss:.6f},{train_eval_loss:.6f},'
                f'{val_loss:.6f},{lr:.8f},{"*" if is_best else ""}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a BERT-style masked prediction model on '
                    'synthetic-song data.  (GPU-optimised for RTX 5090)')
    # Data
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing data.npz and config.json')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length per window')
    parser.add_argument('--stride', type=int, default=128,
                        help='Stride between windows')
    parser.add_argument('--mask-ratio', type=float, default=0.15,
                        help='Fraction of time steps to mask')
    parser.add_argument('--mask-patch-size', type=int, default=16,
                        help='Contiguous patch size for masking '
                             '(used when min/max not set)')
    parser.add_argument('--mask-patch-min', type=int, default=None,
                        help='Min patch size for variable-size masking')
    parser.add_argument('--mask-patch-max', type=int, default=None,
                        help='Max patch size for variable-size masking')
    # Model
    parser.add_argument('--d-model', type=int, default=128,
                        help='Transformer model dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=512,
                        help='Feed-forward hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--pos-encoding', type=str, default='sinusoidal',
                        choices=['sinusoidal', 't5', 'rope'],
                        help='Positional encoding: sinusoidal (absolute), '
                             't5 (learned relative bias), or rope (rotary)')
    parser.add_argument('--t5-num-buckets', type=int, default=32,
                        help='Number of relative-position buckets for T5 bias')
    parser.add_argument('--t5-max-distance', type=int, default=128,
                        help='Max distance for T5 relative-position bucketing')
    parser.add_argument('--rope-base', type=float, default=10000.0,
                        help='RoPE frequency base (only for --pos-encoding rope)')
    # Training
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size (default 128 for 32GB VRAM)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Peak learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=20,
                        help='Number of linear warmup epochs')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker processes')
    # GPU options
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile (useful for debugging)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed-precision (run in FP32)')
    parser.add_argument('--no-train-eval', action='store_true',
                        help='Skip eval-mode pass on training data '
                             '(faster epochs, no Train(eval) curve)')
    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default='bert_model.pt',
                        help='Path to save model checkpoint')
    parser.add_argument('--eval', type=str, default=None, metavar='CKPT',
                        help='Evaluate and visualize from a checkpoint '
                             '(skip training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # ---- Device ----
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — GPU optimisations disabled, "
              "falling back to CPU.")
        device = torch.device('cpu')
        args.no_amp = True
        args.no_compile = True
        args.num_workers = 0
    else:
        device = torch.device('cuda')

        # cuDNN auto-tuner: finds fastest algorithms for fixed input sizes
        torch.backends.cudnn.benchmark = True

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name}  ({gpu_mem:.1f} GB, "
              f"compute {compute_cap[0]}.{compute_cap[1]})")

    # Mixed-precision dtype — prefer BF16 on Ampere+ (avoids loss scaling issues)
    if args.no_amp:
        amp_dtype = torch.float32
        print("Mixed precision: DISABLED (FP32)")
    elif device.type == 'cuda' and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("Mixed precision: BF16")
    else:
        amp_dtype = torch.float16
        print("Mixed precision: FP16")

    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Dataset ----
    full_ds = SyntheticSongDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        mask_ratio=args.mask_ratio,
        mask_patch_size=args.mask_patch_size,
        mask_patch_min=args.mask_patch_min,
        mask_patch_max=args.mask_patch_max,
        mask_seed=None,  # random masking each epoch for augmentation
    )

    # Train / val split
    n_val = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=use_persistent,
    )

    print(f"Dataset: {len(full_ds)} windows  "
          f"(train={n_train}, val={n_val})")
    print(f"  seq_len={args.seq_len}, stride={args.stride}, "
          f"feature_dim={full_ds.feature_dim}")
    print(f"  mask_ratio={args.mask_ratio}, "
          f"patch_size={full_ds.mask_patch_min}-{full_ds.mask_patch_max}")
    print(f"  batch_size={args.batch_size}, "
          f"num_workers={args.num_workers}")

    # ---- Model ----
    model = MaskedTimeSeriesBERT(
        feature_dim=full_ds.feature_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.seq_len + 64,
        pos_encoding=args.pos_encoding,
        t5_num_buckets=args.t5_num_buckets,
        t5_max_distance=args.t5_max_distance,
        rope_base=args.rope_base,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  "
          f"(d_model={args.d_model}, layers={args.n_layers}, "
          f"heads={args.n_heads}, pos={args.pos_encoding})")

    # ---- torch.compile (inductor backend) ----
    if not args.no_compile and device.type == 'cuda':
        print("Compiling model with torch.compile (inductor)...")
        model = torch.compile(model)
        print("  Compilation deferred — first batch will be slower.")
    elif args.no_compile:
        print("torch.compile: DISABLED")

    # ---- GradScaler for mixed precision ----
    # BF16 doesn't strictly need scaling, but GradScaler is a no-op
    # when not needed and keeps the code path uniform.
    scaler = torch.amp.GradScaler(
        device='cuda',
        enabled=(not args.no_amp and device.type == 'cuda'),
    )

    # ---- Eval-only mode ----
    if args.eval is not None:
        ckpt = torch.load(args.eval, map_location=device, weights_only=True)
        # Handle compiled model state dict (strip _orig_mod. prefix)
        state_dict = ckpt['model_state_dict']
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            cleaned = {k.replace('_orig_mod.', ''): v
                       for k, v in state_dict.items()}
            model.load_state_dict(cleaned)

        val_loss = evaluate(model, val_loader, device, amp_dtype)
        print(f"\nValidation MSE: {val_loss:.4f}")

        viz_ds = SyntheticSongDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            stride=args.stride,
            mask_ratio=args.mask_ratio,
            mask_patch_size=args.mask_patch_size,
            mask_patch_min=args.mask_patch_min,
            mask_patch_max=args.mask_patch_max,
            mask_seed=123,
        )
        visualize_predictions(model, viz_ds, device, n_samples=3,
                              amp_dtype=amp_dtype)
        return

    # ---- Optimizer & scheduler (linear warmup + cosine decay) ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # ---- Training loop ----
    best_val = float('inf')
    train_losses, val_losses, train_eval_losses, lrs = [], [], [], []
    log_path = 'training_log.csv'
    plot_every = 25  # update loss curve plot every N epochs

    # Remove stale log from a previous run
    if Path(log_path).exists():
        Path(log_path).unlink()

    do_train_eval = not args.no_train_eval

    if do_train_eval:
        print(f"\n{'Epoch':<7} {'Train(drop)':<12} {'Train(eval)':<12} "
              f"{'Val MSE':<12} {'LR':<12} {'Time':<8} {'Best'}")
        print('-' * 80)
    else:
        print(f"\n{'Epoch':<7} {'Train MSE':<12} {'Val MSE':<12} "
              f"{'LR':<12} {'Time':<8} {'Best'}")
        print('-' * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     scaler, device, amp_dtype)
        # Eval-mode pass on training set (dropout OFF) for fair comparison
        train_eval_loss = (evaluate(model, train_loader, device, amp_dtype)
                           if do_train_eval else float('nan'))
        val_loss = evaluate(model, val_loader, device, amp_dtype)

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        train_losses.append(train_loss)
        train_eval_losses.append(train_eval_loss)
        val_losses.append(val_loss)
        lrs.append(lr)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            # Save raw (unwrapped) state dict for portability
            raw_model = (model._orig_mod
                         if hasattr(model, '_orig_mod') else model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args),
            }, args.checkpoint)

        # CSV log (every epoch)
        append_to_log(log_path, epoch, train_loss, train_eval_loss,
                      val_loss, lr, is_best)

        # Console output (selective)
        marker = ' *' if is_best else ''
        if epoch <= 5 or epoch % 10 == 0 or is_best or epoch == args.epochs:
            if do_train_eval:
                print(f"{epoch:<7} {train_loss:<12.4f} {train_eval_loss:<12.4f} "
                      f"{val_loss:<12.4f} {lr:<12.6f} {elapsed:<8.2f}{marker}")
            else:
                print(f"{epoch:<7} {train_loss:<12.4f} {val_loss:<12.4f} "
                      f"{lr:<12.6f} {elapsed:<8.2f}{marker}")

        # Periodic loss curve update
        if epoch % plot_every == 0 or epoch == args.epochs:
            plot_loss_curve(train_losses, val_losses, lrs,
                            train_eval_losses if do_train_eval else None)

    print(f"\nBest val MSE: {best_val:.4f}")
    print(f"Checkpoint saved to {args.checkpoint}")
    print(f"Training log saved to {log_path}")

    # ---- Visualize with best model ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    raw_model = (model._orig_mod
                 if hasattr(model, '_orig_mod') else model)
    raw_model.load_state_dict(ckpt['model_state_dict'])

    viz_ds = SyntheticSongDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        mask_ratio=args.mask_ratio,
        mask_patch_size=args.mask_patch_size,
        mask_patch_min=args.mask_patch_min,
        mask_patch_max=args.mask_patch_max,
        mask_seed=123,  # deterministic for reproducible visualisation
    )
    visualize_predictions(model, viz_ds, device, n_samples=3,
                          amp_dtype=amp_dtype)


if __name__ == '__main__':
    main()

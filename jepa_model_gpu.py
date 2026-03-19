"""
JEPA (Joint Embedding Predictive Architecture) for Time Series — GPU-Optimised

Instead of reconstructing masked observations in pixel space (like the BERT
model in masked_model_gpu.py), JEPA predicts *representations* of masked
regions in latent space.

Architecture
------------
    Context encoder : Input → mask → Linear(20→D) → RoPE Transformer (N layers) → D-dim reps
    Target encoder  : Input (full) → Linear(20→D) → RoPE Transformer (N layers) → D-dim reps
                      (exponential moving average of context encoder, no gradients)
    Predictor       : Context reps → replace masked with [PRED] token
                      → RoPE Transformer (M layers) → LayerNorm → Linear → D-dim predictions
    Loss            : MSE(predictions[mask], target_reps[mask])

The target encoder is updated via EMA after each optimiser step, with a cosine
momentum schedule from τ_base (0.996) to 1.0.  The asymmetric architecture
(shallow predictor + EMA target) prevents representation collapse without
needing explicit regularisation.

GPU Optimisations
-----------------
Same as masked_model_gpu.py: BF16 mixed precision, torch.compile, multi-worker
DataLoader, pin_memory, cudnn.benchmark, zero_grad(set_to_none=True).

Usage
-----
    python jepa_model_gpu.py                              # train with defaults
    python jepa_model_gpu.py --epochs 500 --n-layers 7    # override defaults
    python jepa_model_gpu.py --eval jepa_model.pt         # evaluate
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
from masked_model_gpu import RoPETransformerEncoderLayer


class GPUPreloadedDataset(torch.utils.data.Dataset):
    """Wraps a SyntheticSongDataset with all window data resident on GPU.

    Masks are still generated on CPU each call (random per epoch).
    """

    def __init__(self, base_ds, indices, device):
        self._mask_fn = base_ds._generate_mask
        self.seq_len = base_ds.seq_len
        xs, states = [], []
        for i in indices:
            x, s, _ = base_ds[i]
            xs.append(x)
            states.append(s)
        self.x = torch.stack(xs).to(device)
        self.states = torch.stack(states).to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        mask = torch.from_numpy(self._mask_fn(idx))
        return self.x[idx], self.states[idx], mask.to(self.x.device)


# ---------------------------------------------------------------------------
# Encoder backbone  (shared architecture for context & target)
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """Input projection + RoPE transformer layers.

    Used identically for both context and target encoders in JEPA.
    The context encoder receives masked input; the target encoder
    receives the full (unmasked) input.
    """

    def __init__(self, feature_dim, d_model, n_heads, n_layers, d_ff,
                 dropout, max_len):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, max_len=max_len,
            ) for _ in range(n_layers)
        ])

    def forward(self, x):
        """x: (batch, seq_len, feature_dim) → (batch, seq_len, d_model)"""
        h = self.input_proj(x)
        h = self.input_dropout(h)
        for layer in self.layers:
            h = layer(h)
        return h

    @torch.no_grad()
    def forward_layers(self, x):
        """Return list of per-layer representations for analysis."""
        h = self.input_proj(x)
        h = self.input_dropout(h)
        outputs = []
        for layer in self.layers:
            h = layer(h)
            outputs.append(h)
        return outputs


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """Lightweight transformer that maps context representations to
    predicted target representations at masked positions.

    At visible positions the predictor receives the context encoder's
    output.  At masked positions it substitutes a learnable prediction
    token, then attends across all positions to produce predictions.
    Only the predictions at masked positions are used in the loss.
    """

    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout, max_len):
        super().__init__()
        self.predict_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, max_len=max_len,
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, context_reps, mask):
        """
        Parameters
        ----------
        context_reps : (batch, seq_len, d_model)
        mask : (batch, seq_len) — True at masked positions

        Returns
        -------
        (batch, seq_len, d_model) — predictions (loss uses masked positions only)
        """
        mask_expanded = mask.unsqueeze(-1)
        h = torch.where(mask_expanded, self.predict_token, context_reps)
        for layer in self.layers:
            h = layer(h)
        return self.proj(self.norm(h))


# ---------------------------------------------------------------------------
# JEPA Model
# ---------------------------------------------------------------------------

class JEPATimeSeriesModel(nn.Module):
    """
    JEPA model for masked time series prediction in representation space.

    Parameters
    ----------
    feature_dim : int
        Dimension of each time step (20 for synthetic data).
    d_model : int
        Transformer model dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers in each encoder.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
    max_len : int
        Maximum sequence length.
    predictor_n_layers : int
        Number of transformer layers in the predictor (typically 2–3).
    """

    def __init__(
        self,
        feature_dim=20,
        d_model=128,
        n_heads=4,
        n_layers=7,
        d_ff=512,
        dropout=0.1,
        max_len=2048,
        predictor_n_layers=2,
        bottleneck_dim=0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.bottleneck_dim = bottleneck_dim

        # Learnable mask token in observation space (before projection)
        self.mask_token = nn.Parameter(torch.randn(feature_dim))

        # Context encoder (receives gradient)
        self.context_encoder = TransformerEncoder(
            feature_dim, d_model, n_heads, n_layers, d_ff, dropout, max_len)

        # Target encoder (EMA of context encoder — no gradients)
        self.target_encoder = TransformerEncoder(
            feature_dim, d_model, n_heads, n_layers, d_ff, dropout, max_len)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Information bottleneck (context path only — target stays unconstrained)
        if bottleneck_dim > 0:
            self.context_bottleneck = nn.Sequential(
                nn.Linear(d_model, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, d_model),
            )
        else:
            self.context_bottleneck = None

        # Predictor (lightweight — deliberately shallower than the encoder)
        self.predictor = JEPAPredictor(
            d_model, n_heads, predictor_n_layers, d_ff, dropout, max_len)

    def forward(self, x, mask):
        """
        Parameters
        ----------
        x    : (batch, seq_len, feature_dim) — raw observations
        mask : (batch, seq_len) — True at positions to predict

        Returns
        -------
        pred_reps   : (batch, seq_len, d_model) — predictor output
        target_reps : (batch, seq_len, d_model) — target encoder output (detached)
        """
        # Context path: mask input then encode
        mask_expanded = mask.unsqueeze(-1)
        x_masked = torch.where(mask_expanded, self.mask_token, x)
        context_reps = self.context_encoder(x_masked)
        if self.context_bottleneck is not None:
            context_reps = self.context_bottleneck(context_reps)

        # Target path: full input, no gradients
        with torch.no_grad():
            target_reps = self.target_encoder(x)

        # Predictor: map context reps → predicted target reps at masked positions
        pred_reps = self.predictor(context_reps, mask)

        return pred_reps, target_reps

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        """EMA update: θ_target = τ · θ_target + (1 − τ) · θ_context"""
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                self.target_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def encode(self, x):
        """Run input through the target encoder (no masking) and return
        per-layer representations.  Compatible with evaluate_representations.py.

        Returns
        -------
        layer_outputs : list of (batch, seq_len, d_model) tensors,
            one per transformer layer.
        """
        return self.target_encoder.forward_layers(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def jepa_loss(pred_reps, target_reps, mask):
    """MSE between predicted and target representations at masked positions."""
    mask_expanded = mask.unsqueeze(-1).expand_as(pred_reps)
    n_masked = mask_expanded.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=pred_reps.device)
    return ((pred_reps - target_reps) ** 2 * mask_expanded).sum() / n_masked


def jepa_loss_region(pred_reps, target_reps, mask):
    """Region-level MSE: average reps over each contiguous masked region,
    then compute MSE between region-level predictions and targets.

    Encourages concept-level representations by discarding within-region
    positional detail — the model only needs to predict a summary of
    the masked region, not reconstruct each timestep.
    """
    batch_size, seq_len, d_model = pred_reps.shape
    total_loss = torch.tensor(0.0, device=pred_reps.device)
    n_regions = 0

    for b in range(batch_size):
        m = mask[b]  # (seq_len,)
        if not m.any():
            continue

        # Find contiguous masked regions via diff
        padded = torch.cat([
            torch.zeros(1, device=m.device, dtype=m.dtype),
            m.to(torch.int8),
            torch.zeros(1, device=m.device, dtype=m.dtype),
        ])
        diff = padded[1:] - padded[:-1]
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]

        for s, e in zip(starts, ends):
            pred_region = pred_reps[b, s:e].mean(dim=0)
            tgt_region = target_reps[b, s:e].mean(dim=0)
            total_loss = total_loss + (pred_region - tgt_region).pow(2).mean()
            n_regions += 1

    if n_regions == 0:
        return torch.tensor(0.0, device=pred_reps.device)
    return total_loss / n_regions


# ---------------------------------------------------------------------------
# EMA momentum schedule
# ---------------------------------------------------------------------------

def momentum_schedule(epoch, n_epochs, base_momentum=0.996):
    """Cosine schedule: base_momentum → 1.0 over training."""
    return 1.0 - (1.0 - base_momentum) * (
        1.0 + math.cos(math.pi * epoch / n_epochs)) / 2.0


# ---------------------------------------------------------------------------
# GPU-optimised training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, amp_dtype,
                    momentum, loss_fn=jepa_loss):
    """One training epoch with mixed precision and per-step EMA updates."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, _state, mask in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            pred_reps, target_reps = model(x, mask)
            loss = loss_fn(pred_reps, target_reps, mask)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()

        # EMA update on the un-compiled model
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw.update_target_encoder(momentum)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, loss_fn=jepa_loss):
    """Evaluation: returns (loss, cosine_similarity, target_std)."""
    model.eval()
    total_loss = 0.0
    total_cos = 0.0
    total_tgt_std = 0.0
    n_batches = 0

    for x, _state, mask in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            pred_reps, target_reps = model(x, mask)
            loss = loss_fn(pred_reps, target_reps, mask)

        # Cosine similarity at masked positions
        mask_flat = mask.unsqueeze(-1).expand_as(pred_reps)
        pred_m = pred_reps[mask_flat].reshape(-1, pred_reps.size(-1)).float()
        tgt_m = target_reps[mask_flat].reshape(-1, target_reps.size(-1)).float()
        if pred_m.numel() > 0:
            total_cos += nn.functional.cosine_similarity(
                pred_m, tgt_m, dim=-1).mean().item()

        # Collapse monitor: std of target representations
        total_tgt_std += target_reps.float().std().item()

        total_loss += loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_cos / n, total_tgt_std / n


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses, val_losses, cos_sims, lrs=None,
                    save_path='training_loss.png'):
    """Training/validation loss curve with cosine similarity overlay."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label='Train JEPA Loss',
             linewidth=2, alpha=0.5)
    ax1.plot(epochs, val_losses, label='Val JEPA Loss',
             linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('JEPA Loss (rep-space MSE)')
    ax1.set_title('JEPA Training Progress')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Cosine similarity on right axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, cos_sims, color='green', linewidth=1.5, alpha=0.7,
             label='Cosine Similarity')
    ax2.set_ylabel('Cosine Similarity', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def append_to_log(log_path, epoch, train_loss, val_loss, cos_sim,
                  target_std, lr, momentum, is_best):
    """Append one line to the CSV training log."""
    write_header = not Path(log_path).exists()
    with open(log_path, 'a') as f:
        if write_header:
            f.write('epoch,train_loss,val_loss,cos_sim,target_std,'
                    'lr,momentum,best\n')
        f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},'
                f'{cos_sim:.6f},{target_std:.6f},'
                f'{lr:.8f},{momentum:.6f},{"*" if is_best else ""}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a JEPA masked prediction model on '
                    'synthetic-song data.  (GPU-optimised)')
    # Data
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--seq-len', type=int, default=1000)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--mask-ratio', type=float, default=0.5)
    parser.add_argument('--mask-patch-size', type=int, default=16)
    parser.add_argument('--mask-patch-min', type=int, default=10)
    parser.add_argument('--mask-patch-max', type=int, default=600)
    # Model (encoder)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=8,
                        help='Transformer layers in each encoder')
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Model (predictor)
    parser.add_argument('--predictor-n-layers', type=int, default=2,
                        help='Transformer layers in the predictor '
                             '(keep shallow to avoid lazy encoder)')
    # EMA
    parser.add_argument('--ema-base', type=float, default=0.996,
                        help='Base EMA momentum (annealed to 1.0 via cosine)')
    # Loss
    parser.add_argument('--region-level', action='store_true',
                        help='Use region-level loss: average reps over each '
                             'contiguous masked region before computing MSE. '
                             'Encourages concept-level representations.')
    # Information bottleneck
    parser.add_argument('--bottleneck-dim', type=int, default=0,
                        help='If >0, add an information bottleneck (linear → '
                             'ReLU → linear) after both encoders that '
                             'compresses d_model through this dimension. '
                             'Forces the model to retain only salient features.')
    # Training
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    # GPU options
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default='jepa_model.pt')
    parser.add_argument('--eval', type=str, default=None, metavar='CKPT',
                        help='Evaluate from a checkpoint (skip training)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ---- Device ----
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device = torch.device('cpu')
        args.no_amp = True
        args.no_compile = True
        args.num_workers = 0
    else:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name}  ({gpu_mem:.1f} GB, "
              f"compute {compute_cap[0]}.{compute_cap[1]})")

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
        mask_seed=None,
    )

    n_val = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    split = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if device.type == 'cuda':
        print("Preloading dataset to GPU...")
        train_ds = GPUPreloadedDataset(full_ds, split[0].indices, device)
        val_ds = GPUPreloadedDataset(full_ds, split[1].indices, device)
        gpu_mb = (train_ds.x.nbytes + val_ds.x.nbytes) / 1e6
        print(f"  Preloaded {gpu_mb:.1f} MB to GPU")
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False,
        )
    else:
        train_ds, val_ds = split
        use_persistent = args.num_workers > 0
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=use_persistent,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=use_persistent,
        )

    print(f"Dataset: {len(full_ds)} windows  "
          f"(train={n_train}, val={n_val})")
    print(f"  seq_len={args.seq_len}, stride={args.stride}, "
          f"feature_dim={full_ds.feature_dim}")
    print(f"  mask_ratio={args.mask_ratio}, "
          f"patch_size={full_ds.mask_patch_min}-{full_ds.mask_patch_max}")
    print(f"  batch_size={args.batch_size}")

    # ---- Model ----
    model = JEPATimeSeriesModel(
        feature_dim=full_ds.feature_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.seq_len + 64,
        predictor_n_layers=args.predictor_n_layers,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    bn_str = (f", bottleneck={args.bottleneck_dim}"
              if args.bottleneck_dim > 0 else "")
    print(f"Model: {n_trainable:,} trainable params, "
          f"{n_total:,} total  "
          f"(encoder={args.n_layers}L, predictor={args.predictor_n_layers}L, "
          f"d_model={args.d_model}, heads={args.n_heads}{bn_str})")

    # ---- torch.compile ----
    if not args.no_compile and device.type == 'cuda':
        print("Compiling model with torch.compile (inductor)...")
        model = torch.compile(model)
        print("  Compilation deferred — first batch will be slower.")
    elif args.no_compile:
        print("torch.compile: DISABLED")

    # ---- GradScaler ----
    scaler = torch.amp.GradScaler(
        device='cuda',
        enabled=(not args.no_amp and device.type == 'cuda'),
    )

    # ---- Eval-only mode ----
    if args.eval is not None:
        ckpt = torch.load(args.eval, map_location=device, weights_only=True)
        state_dict = ckpt['model_state_dict']
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        try:
            raw.load_state_dict(state_dict)
        except RuntimeError:
            cleaned = {k.replace('_orig_mod.', ''): v
                       for k, v in state_dict.items()}
            raw.load_state_dict(cleaned)

        eval_loss_fn = jepa_loss_region if args.region_level else jepa_loss
        val_loss, cos_sim, tgt_std = evaluate(
            model, val_loader, device, amp_dtype, loss_fn=eval_loss_fn)
        print(f"\nVal JEPA Loss: {val_loss:.4f}")
        print(f"Cosine Similarity: {cos_sim:.4f}")
        print(f"Target Repr Std: {tgt_std:.4f}")
        return

    # ---- Optimiser (context encoder + predictor only) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
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

    # ---- Loss function ----
    if args.region_level:
        loss_fn = jepa_loss_region
        print("Loss: region-level (concept-level representations)")
    else:
        loss_fn = jepa_loss
        print("Loss: per-timestep")

    # ---- Training loop ----
    best_val = float('inf')
    train_losses, val_losses, cos_sims, lrs = [], [], [], []
    log_path = 'jepa_training_log.csv'
    plot_every = 25

    if Path(log_path).exists():
        Path(log_path).unlink()

    print(f"\n{'Epoch':<7} {'Train':<12} {'Val':<12} {'CosSim':<8} "
          f"{'TgtStd':<8} {'LR':<12} {'Mom':<8} {'Time':<8} {'Best'}")
    print('-' * 90)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        mom = momentum_schedule(epoch, args.epochs, args.ema_base)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp_dtype, mom,
            loss_fn=loss_fn)
        val_loss, cos_sim, tgt_std = evaluate(
            model, val_loader, device, amp_dtype, loss_fn=loss_fn)

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        cos_sims.append(cos_sim)
        lrs.append(lr)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'cos_sim': cos_sim,
                'args': {**vars(args), 'model_type': 'jepa',
                         'feature_dim': full_ds.feature_dim},
            }, args.checkpoint)

        append_to_log(log_path, epoch, train_loss, val_loss, cos_sim,
                      tgt_std, lr, mom, is_best)

        marker = ' *' if is_best else ''
        if epoch <= 5 or epoch % 10 == 0 or is_best or epoch == args.epochs:
            print(f"{epoch:<7} {train_loss:<12.4f} {val_loss:<12.4f} "
                  f"{cos_sim:<8.4f} {tgt_std:<8.4f} "
                  f"{lr:<12.6f} {mom:<8.4f} {elapsed:<8.2f}{marker}")

        if epoch % plot_every == 0 or epoch == args.epochs:
            plot_loss_curve(train_losses, val_losses, cos_sims)

    print(f"\nBest val JEPA loss: {best_val:.4f}")
    print(f"Checkpoint saved to {args.checkpoint}")
    print(f"Training log saved to {log_path}")


if __name__ == '__main__':
    main()

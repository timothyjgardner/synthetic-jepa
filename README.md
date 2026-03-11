# Synthetic JEPA

Joint Embedding Predictive Architecture (JEPA) for masked prediction on synthetic time series data.

Instead of reconstructing masked observations in pixel/feature space (as in the BERT-style model in `synthetic-bert`), JEPA predicts **representations** of masked regions in latent space.

## Architecture

```
Context encoder : Input → mask → Linear(20→D) → RoPE Transformer (N layers) → D-dim reps
Target encoder  : Input (full) → Linear(20→D) → RoPE Transformer (N layers) → D-dim reps
                  (exponential moving average of context encoder, no gradients)
Predictor       : Context reps → replace masked with [PRED] token
                  → RoPE Transformer (M layers) → LayerNorm → Linear → D-dim predictions
Loss            : MSE(predictions[mask], target_reps[mask])
```

The target encoder is updated via EMA after each optimiser step, with a cosine momentum schedule from τ_base (0.996) → 1.0. The asymmetric architecture (shallow predictor + EMA target) prevents representation collapse without explicit regularisation.

### Default model (7-layer, 1.8M params)

| Component | Value |
|-----------|-------|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 7 |
| d_ff | 512 |
| predictor_n_layers | 2 |
| seq_len | 512 |
| mask_ratio | 0.25 |
| mask_patch_size | 16–256 |

## Results

### JEPA representations (7 layers)

![JEPA UMAP](representation_umap_jepa_model.png)

### BERT baseline representations (7 layers, RoPE)

![BERT UMAP](representation_umap_bert_model.png)

JEPA produces significantly better cluster separation than the BERT baseline across all layers, peaking at **Sil=0.61** (layer 6) vs **Sil=0.29** (BERT layer 6).

## GPU Optimisations

BF16 mixed precision, `torch.compile` (Inductor backend), multi-worker DataLoader, `pin_memory`, `cudnn.benchmark`, `zero_grad(set_to_none=True)`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib torch scipy umap-learn
```

## Usage

### Generate dataset

```bash
python markov_circles_timeseries.py --subspace-dim 4 --no-umap
```

### Train

```bash
python jepa_model_gpu.py --n-layers 7 --mask-patch-max 256 --mask-ratio 0.25 --epochs 500
```

### Evaluate representations (UMAP + Levina-Bickel)

```bash
python evaluate_representations.py --checkpoint jepa_model.pt --layers input,1,2,3,4,5,6,7
```

## Files

| File | Description |
|------|-------------|
| `jepa_model_gpu.py` | JEPA model definition and GPU-optimised training loop |
| `dataset.py` | `SyntheticSongDataset` — sliding-window + patch masking |
| `masked_model_gpu.py` | Shared `RoPETransformerEncoderLayer` (imported by JEPA) |
| `evaluate_representations.py` | UMAP visualisation and Levina-Bickel intrinsic dimension |
| `estimate_dimension.py` | Levina-Bickel estimator |
| `markov_circles_timeseries.py` | Synthetic dataset generator |

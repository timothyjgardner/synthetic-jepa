"""
Estimate the intrinsic dimension of the synthetic-song dataset using the
Levina-Bickel MLE estimator at various neighbourhood sizes k.

Runs on a random subsample (default 2000 points) to keep memory and
compute manageable, and also computes per-circle estimates to see how
dimension varies across states.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


def _gpu_knn_dists(X_np, k, batch_size=2048):
    """Batched k-NN distances on GPU using PyTorch.

    Only returns distances (not indices) since Levina-Bickel doesn't need them.
    Falls back to CPU if CUDA is unavailable.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(np.ascontiguousarray(X_np)).float().to(device)
    n = X.shape[0]
    all_dists = torch.zeros(n, k, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = torch.cdist(X[start:end], X)
        dists[:, start:end].fill_diagonal_(float('inf'))
        topk_d, _ = dists.topk(k, dim=1, largest=False)
        all_dists[start:end] = topk_d

    return all_dists.cpu().numpy()


def levina_bickel_estimator(X, k):
    """
    Levina-Bickel MLE intrinsic dimension estimator.

    Uses GPU-accelerated k-NN to avoid computing the full O(n^2) distance
    matrix on CPU.

    Returns
    -------
    m_hat : float
        Mean estimated dimension across all points.
    m_hat_per_point : ndarray (n_points,)
        Per-point estimates.
    """
    n = X.shape[0]
    knn_dists = _gpu_knn_dists(X.astype(np.float32), k)

    T_k = knn_dists[:, -1]
    valid = T_k > 1e-12
    log_ratios = np.log(
        T_k[valid, np.newaxis] / knn_dists[valid, :-1])
    mean_lr = log_ratios.mean(axis=1)

    m_hat_per_point = np.full(n, np.nan)
    safe = mean_lr > 1e-10
    idx_valid = np.where(valid)[0]
    m_hat_per_point[idx_valid[safe]] = 1.0 / mean_lr[safe]

    return np.nanmean(m_hat_per_point), m_hat_per_point


def main():
    # ---- load data ----
    data_dir = Path('data')
    npz = np.load(data_dir / 'data.npz')
    X_full = npz['X']
    states_full = npz['states']
    with open(data_dir / 'config.json') as f:
        config = json.load(f)

    n_circles = config['n_circles']
    periods = npz['periods']

    # ---- subsample for speed ----
    n_sub = 2000
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_full), n_sub, replace=False)
    X = X_full[idx]
    states = states_full[idx]

    k_values = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    k_values = [k for k in k_values if k < n_sub]

    # ---- global dimension vs k ----
    print(f"Estimating dimension on {n_sub} subsampled points...")
    print(f"{'k':<6} {'dim_hat':<10}")
    print('-' * 20)

    global_dims = []
    for k in k_values:
        m_hat, _ = levina_bickel_estimator(X, k)
        global_dims.append(m_hat)
        print(f"{k:<6} {m_hat:<10.2f}")

    # ---- per-circle dimension at a few k values ----
    k_detail = [10, 30, 100]
    k_detail = [k for k in k_detail if k < n_sub]

    print(f"\nPer-circle dimension estimates (subsample, per-state):")
    print(f"{'Circle':<8} {'Period':<8}", end='')
    for k in k_detail:
        print(f"{'k=' + str(k):<10}", end='')
    print()
    print('-' * (16 + 10 * len(k_detail)))

    per_circle = {k: [] for k in k_detail}
    for ci in range(n_circles):
        mask = states == ci
        X_ci = X[mask]
        n_ci = X_ci.shape[0]
        print(f"{ci:<8} {periods[ci]:<8}", end='')
        for k in k_detail:
            if n_ci > k + 1:
                m, _ = levina_bickel_estimator(X_ci, k)
                per_circle[k].append(m)
                print(f"{m:<10.2f}", end='')
            else:
                per_circle[k].append(np.nan)
                print(f"{'n/a':<10}", end='')
        print()

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: global dimension vs k
    ax = axes[0]
    ax.plot(k_values, global_dims, 'o-', linewidth=2, markersize=6,
            color='steelblue')
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5,
               label='True manifold dim = 2 (circle sub-plane)')
    ax.axhline(y=config['ambient_dim'], color='gray', linestyle=':',
               alpha=0.5, label=f'Ambient dim = {config["ambient_dim"]}')
    ax.set_xlabel('k (number of neighbours)', fontsize=12)
    ax.set_ylabel('Estimated dimension', fontsize=12)
    ax.set_title('Levina-Bickel Dimension Estimate vs k\n'
                 '(synthetic-song dataset)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: per-circle dimension at selected k values
    ax = axes[1]
    x_pos = np.arange(n_circles)
    width = 0.25
    colors_k = ['steelblue', 'darkorange', 'seagreen']
    for i, k in enumerate(k_detail):
        vals = per_circle[k]
        offset = (i - len(k_detail) / 2 + 0.5) * width
        ax.bar(x_pos + offset, vals, width, label=f'k={k}',
               color=colors_k[i % len(colors_k)], alpha=0.8)
    ax.set_xlabel('Circle index', fontsize=12)
    ax.set_ylabel('Estimated dimension', fontsize=12)
    ax.set_title('Per-Circle Dimension Estimate\n'
                 '(fast circles left, slow circles right)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{ci}\n(T={periods[ci]})' for ci in range(n_circles)],
                       fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('dimension_estimates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved dimension_estimates.png")


if __name__ == '__main__':
    main()

"""
Davies–Bouldin Evaluation

Inputs:
    embeddings_2d: numpy array of shape (n_samples, 2), columns = [x, y]
    labels: ground-truth cluster/category labels (ints or strings)

How to call from your model script:
    from src.Evaluations.davies_bouldin import compute_davies_bouldin_index

Example:
    score = compute_davies_bouldin_index(embeddings_2d, labels, model_name="triplet")

Results:
    - Saved under results/davies_bouldin/{model_name}_dbindex.txt
"""

import os
import numpy as np
from sklearn.metrics import davies_bouldin_score


def compute_davies_bouldin_index(embeddings_2d, labels, model_name="model"):
    """Compute Davies–Bouldin index for 2D embeddings (lower = better)."""
    embeddings_2d = np.asarray(embeddings_2d)
    labels = np.asarray(labels)

    # Validate shape
    if embeddings_2d.ndim != 2 or embeddings_2d.shape[1] != 2:
        raise ValueError(
            f"embeddings_2d must be shape (n,2) with columns [x,y], got {embeddings_2d.shape}"
        )
    if embeddings_2d.shape[0] != labels.shape[0]:
        raise ValueError(f"Shape mismatch: {embeddings_2d.shape[0]} vs {labels.shape[0]}")
    if np.unique(labels).size < 2:
        raise ValueError("Need ≥2 categories for DB index")

    score = davies_bouldin_score(embeddings_2d, labels)

    # Save results under results/davies_bouldin/
    results_dir = os.path.join("results", "davies_bouldin")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{model_name}_dbindex.txt")
    with open(out_path, "w") as f:
        f.write(f"Davies–Bouldin Index for {model_name}\n")
        f.write(f"Score: {score:.4f}\n")
        f.write(f"Samples: {len(embeddings_2d)}\n")
        f.write(f"Categories: {sorted(np.unique(labels).tolist())}\n")
    print(f"[Saved] {out_path}")

    return score

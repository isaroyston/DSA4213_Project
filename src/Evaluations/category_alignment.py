"""
Category Alignment Evaluation
-----------------------------
Inputs:
    - y_pred: predicted labels (array-like)
    - y_true: ground-truth labels (array-like)
    - category_names: optional list of class names (must match label IDs)

How to call from your model script:
    from src.Evaluations.category_alignment import compute_category_alignment, evaluate_from_csv

Examples:
    results = compute_category_alignment(preds, labels, run_name="finetuned_triplet")
    results = evaluate_from_csv("20cat_results.csv", run_name="20cat")

Results:
    - Metrics saved under results/category_alignment/{run_name}_metrics.json
    - Confusion matrix plot saved if plot_confusion=True
"""

import os
import json
import numpy as np
from typing import List, Union, Optional
from sklearn.metrics import f1_score, confusion_matrix


def compute_category_alignment(
    y_pred: Union[np.ndarray, List[int]],
    y_true: Union[np.ndarray, List[int]],
    category_names: Optional[List[str]] = None,
    run_name: str = "default",
    plot_confusion: bool = False,
) -> dict:
    """Compute macro/weighted/per-class F1. Optionally plot confusion matrix."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("Prediction/label length mismatch")

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    all_labels = np.array(sorted(np.unique(np.concatenate([y_true, y_pred]))))
    per_class_f1 = f1_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    results = {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1.tolist(),
        "labels": all_labels.tolist(),
        "confusion_matrix": cm.tolist(),
    }

    # Save under results/category_alignment/
    results_dir = os.path.join("results", "category_alignment")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{run_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {out_path}")

    if plot_confusion:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        ticks = [str(l) for l in all_labels]
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Count")
        plt.xticks(np.arange(len(ticks)), ticks, rotation=45, ha="right")
        plt.yticks(np.arange(len(ticks)), ticks)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.show()

    return results


def evaluate_from_csv(csv_path: str, run_name: str = "default", plot_confusion: bool = False):
    """Load predictions & labels from CSV (must contain 'label' + 'predicted_label') and evaluate."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "label" not in df or "predicted_label" not in df:
        raise ValueError("CSV must contain 'label' and 'predicted_label' columns")

    y_true = df["label"].values
    y_pred = df["predicted_label"].values
    return compute_category_alignment(y_pred, y_true, run_name=run_name, plot_confusion=plot_confusion)

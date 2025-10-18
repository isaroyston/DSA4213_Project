# on  2d embeddings

from typing import List, Union, Optional
import numpy as np
from sklearn.metrics import davies_bouldin_score, f1_score, confusion_matrix


def compute_davies_bouldin_index(
    embeddings_2d: Union[np.ndarray, List[List[float]]],
    labels: Union[np.ndarray, List[int]],
) -> float:
    """Davies–Bouldin Index for 2D embeddings. Lower is better."""
    embeddings_2d = np.asarray(embeddings_2d)
    labels = np.asarray(labels)

    if embeddings_2d.ndim != 2 or embeddings_2d.shape[1] != 2:
        raise ValueError(f"Embeddings must be 2D with shape (n, 2); got {embeddings_2d.shape}")
    if embeddings_2d.shape[0] != labels.shape[0]:
        raise ValueError(f"{embeddings_2d.shape[0]} embeddings but {labels.shape[0]} labels")
    if np.unique(labels).size < 2:
        raise ValueError("Need at least 2 different categories")

    db_index = davies_bouldin_score(embeddings_2d, labels)

    print("DAVIES–BOULDIN INDEX")
    print(f"Score: {db_index:.4f}")
    print(f"Samples: {len(embeddings_2d)}")
    print(f"Categories: {sorted(np.unique(labels).tolist())}")

    return db_index


def compute_category_alignment(
    llm_predictions: Union[np.ndarray, List[int]],
    ground_truth: Union[np.ndarray, List[int]],
    category_names: Optional[List[str]] = None,
    plot_confusion: bool = False,
) -> dict:
    """Macro/weighted/per-class F1 and an optional confusion-matrix plot."""
    y_pred = np.asarray(llm_predictions)
    y_true = np.asarray(ground_truth)

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(f"Mismatch! {y_pred.shape[0]} predictions but {y_true.shape[0]} labels")

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    all_labels = np.array(sorted(np.unique(np.concatenate([y_true, y_pred]))))
    per_class_f1 = f1_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    print("=" * 60)
    print("CATEGORY ALIGNMENT")
    print("=" * 60)
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nPer-Category F1:")

    def name_for(label: int) -> str:
        if category_names and 1 <= label <= len(category_names):
            return category_names[label - 1]  # assumes 1-indexed labels
        return f"Category {label}"

    for i, label in enumerate(all_labels):
        count = int(np.sum(y_true == label))
        print(f"  {label}. {name_for(int(label)):20s} → {per_class_f1[i]:.4f}  (n={count})")

    if plot_confusion:
        import matplotlib.pyplot as plt  # lazy import
        plt.figure(figsize=(8, 6))
        ticks = [name_for(int(l)) for l in all_labels]
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Count")
        plt.xticks(ticks=np.arange(len(ticks)), labels=ticks, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(ticks)), labels=ticks)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.show()



    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    import pandas as pd
    print("triplet_embeddings")
    df = pd.read_csv('embeddings_output/triplet_2d_projection.csv')
    emb = df[["x", "y"]].to_numpy()
    y = df["label"].to_numpy()
    compute_davies_bouldin_index(emb, y)

    print("\n")
    print("cosine_embeddings")
    df = pd.read_csv('embeddings_output/simcse_2d_projection.csv')
    emb = df[["x", "y"]].to_numpy()
    y = df["label"].to_numpy()
    compute_davies_bouldin_index(emb, y)

    print("\n")
    print("simcse_embeddings")
    df = pd.read_csv('embeddings_output/cosine_2d_projection.csv')
    emb = df[["x", "y"]].to_numpy()
    y = df["label"].to_numpy()
    compute_davies_bouldin_index(emb, y)


    pass







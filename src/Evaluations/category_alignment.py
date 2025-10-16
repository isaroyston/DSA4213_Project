
"""
- compute_category_alignment(llm_predictions, ground_truth, category_names=None, plot_confusion=False)

"""
from typing import List, Union, Optional
import numpy as np
from sklearn.metrics import davies_bouldin_score, f1_score, confusion_matrix



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


    print("CATEGORY ALIGNMENT")

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

    print("=" * 60)

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":

    data = np.load('embeddings_output/triplet_embeddings.npz', allow_pickle=True)
    ground_truth = data['labels']
    llm_predictions = [...]  # fill with your predictions
    category_names = [
      'Computers & Technology',
      'Science & Engineering',
      'Recreation (Vehicles & Hobbies)',
      'Sports',
      'Religion',
      'Politics & Society',
      'Marketplace / Miscellaneous',
    ]
    compute_category_alignment(llm_predictions, ground_truth, category_names, plot_confusion=True)
    pass

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from loading import load_data

"""
General Embeddings: Pretrained SentenceTransformer (MiniLM)

This script loads the train and test splits, computes sentence embeddings using a
pretrained SentenceTransformer model (no fine-tuning), and saves the outputs.

Outputs:
embeddings_output/
- general_train_embeddings.npz
- general_test_embeddings.npz
- general_train_2d_projection.csv
- general_test_2d_projection.csv
plots/
- general_train_2d_plot.png
- general_test_2d_plot.png
"""

# config
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use single GPU (if available)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# load data
df_train, df_test = load_data(variant="basic")
df_train_broad, df_test_broad = load_data(variant="broad_category")
print(f"Loaded train: {len(df_train)} docs, test: {len(df_test)} docs")
print(f"Number of original labels: {df_train['label'].nunique()}")
print(f"Number of broad labels: {df_train_broad['broad_label_id'].nunique()}")

# directories
embeddings_root = "embeddings_output"
os.makedirs(embeddings_root, exist_ok=True)
plots_root = "plots"
os.makedirs(plots_root, exist_ok=True)

# dimensionality reduction
def reduce_and_save(embeddings, texts, labels, name, broad_labels=None):
    """
    Reduce embeddings to 2D (PCA -> UMAP), save to CSV and plot.

    Args:
        embeddings (np.ndarray): Array of shape (N, D)
        texts (List[str]): Original texts for reference in CSV
        labels (List[Any]): Label per text
        name (str): Prefix for output artifacts (e.g., 'general_train')
        broad_labels (Optional[List[Any]]): Optional coarse label id per sample
    """
    print(f"Reducing {name} embeddings to 2D...")

    # PCA to 50 dims (speed + denoise before UMAP)
    pca = PCA(n_components=50, random_state=SEED)
    embeddings_pca = pca.fit_transform(embeddings)

    # UMAP to 2D
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)
    embeddings_2d = reducer.fit_transform(embeddings_pca)

    # Build dataframe
    df_out = pd.DataFrame({
        "text": texts,
        "label": labels,
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
    })
    if broad_labels is not None:
        df_out["broad_label_id"] = broad_labels

    # Save CSV
    out_csv = os.path.join(embeddings_root, f"{name}_2d_projection.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"2D projection saved to CSV: {out_csv}")

    # Save plot
    out_plot = os.path.join(plots_root, f"{name}_2d_plot.png")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="x", y="y",
        hue="label",
        palette="tab20",
        s=30,
        alpha=0.7,
        data=df_out,
        legend=False,
    )
    plt.title(f"2D Projection of {name} Embeddings", fontsize=14)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"2D plot saved: {out_plot}")

    return out_csv, out_plot


# encode and map to umap + save
def encode_and_save(model: SentenceTransformer, df: pd.DataFrame, name: str, label_col: str, batch_size: int = 32):
    """
    Encode texts in df with provided model and save artifacts.

    Saves compressed NPZ with arrays: embeddings, texts, labels.
    Also triggers 2D reduction CSV + plot via reduce_and_save.

    Args:
        model: SentenceTransformer
        df: DataFrame with columns ['text', 'label']
        name: Output prefix, e.g., 'general_train' or 'general_test'
        batch_size: Encode batch size
    """
    texts = df["text"].tolist()
    labels = df[label_col].tolist() 

    print(f"Generating embeddings for split '{name}' with {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True,
    )

    out_path = os.path.join(embeddings_root, f"{name}_embeddings.npz")
    print(f"Saving embeddings to: {out_path}")
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    print("Saved embeddings:", embeddings.shape)

    # Reduce to 2D + save CSV + plot
    reduce_and_save(embeddings, texts, labels, name)

    return out_path


# main execution
if __name__ == "__main__":
    # Load a strong general-purpose ST model (same base as fine-tuning script)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # # Encode and save for both splits (orginal labels)
    # train_path = encode_and_save(model, df_train, name="general_train", label_col="label", batch_size=32)
    # test_path = encode_and_save(model, df_test, name="general_test", label_col="label", batch_size=32)

    # print("\nEmbeddings for original labels saved.")

    # Encode and save for both splits (broad labels)
    train_path_broad = encode_and_save(model, df_train_broad, name="general_train_broad", label_col="broad_label_id", batch_size=32)
    test_path_broad = encode_and_save(model, df_test_broad, name="general_test_broad", label_col="broad_label_id", batch_size=32)

    print("\nEmbeddings for broad labels saved.")


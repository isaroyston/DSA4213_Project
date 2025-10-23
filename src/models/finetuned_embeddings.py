import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from loading import load_data

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configuration for dataset and variant
selected_variant = "broad_category" 
selected_data = "train"  

# Load data based on selected variant
df_train, df_test = load_data(variant=selected_variant)

# Select the appropriate dataset based on the configuration
if selected_data == "train":
    df = df_train
else:
    df = df_test

# Select the appropriate label column based on the variant
if selected_variant == "broad_category":
    label_column = "broad_label_id"  
else:
    label_column = "label"  

print(f"Loaded {selected_data} set: {len(df)} docs")
print(f"Number of labels: {df[label_column].nunique()}")

# Directories
model_root = "finetuned_models"
os.makedirs(model_root, exist_ok=True)
embeddings_root = "embeddings_output"
os.makedirs(embeddings_root, exist_ok=True)

# Utilities / Example creators
def create_simcse_examples(df, n_samples=500):
    print(f"Creating {n_samples} SimCSE examples")
    texts = df['text'].sample(n=min(n_samples, len(df)), replace=len(df) < n_samples).tolist()
    examples = [InputExample(texts=[t, t]) for t in texts]
    return examples

def create_cosine_examples(df, n_samples=500):
    examples = []
    labels = df[label_column].unique() 
    print(f"Creating {n_samples} cosine similarity examples")
    half = n_samples // 2

    # positive pairs
    pbar = tqdm(total=half, desc="Positive pairs")
    pos_count = 0
    while pos_count < half:
        anchor_label = np.random.choice(labels)
        anchor_df = df[df[label_column] == anchor_label]
        if len(anchor_df) < 2:
            continue
        pos_texts = anchor_df.sample(2)['text'].values
        examples.append(InputExample(texts=[pos_texts[0], pos_texts[1]], label=1.0))
        pos_count += 1
        pbar.update(1)
    pbar.close()

    # negative pairs
    pbar = tqdm(total=half, desc="Negative pairs")
    neg_count = 0
    while neg_count < half:
        anchor_label = np.random.choice(labels)
        anchor_df = df[df[label_column] == anchor_label]
        if len(anchor_df) < 1:
            continue
        anchor_text = anchor_df.sample(1)['text'].values[0]
        neg_label = np.random.choice([l for l in labels if l != anchor_label])
        neg_text = df[df[label_column] == neg_label].sample(1)['text'].values[0]
        examples.append(InputExample(texts=[anchor_text, neg_text], label=0.0))
        neg_count += 1
        pbar.update(1)
    pbar.close()

    return examples

def create_triplet_examples_mined(df, pretrained_model, n_samples=1000, top_k_neg=20):
    """
    Create triplets by mining semi-hard negatives:
    - Encode the dataset with the pretrained model
    - For each anchor, pick a positive from the same label, and negative as a nearby (in cosine) sample with a different label
    """
    print("Computing embeddings for negative mining (pretrained model)")
    texts = df['text'].tolist()
    emb = pretrained_model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

    print("Building label -> indices map")
    label_to_indices = {}
    for idx, lbl in enumerate(df[label_column].tolist()):
        label_to_indices.setdefault(lbl, []).append(idx)

    examples = []
    labels = df[label_column].tolist()
    num_items = len(texts)
    idxs = list(range(num_items))

    # Precompute nearest neighbors by cosine (using dot product since normalized)
    print("Precomputing similarity matrix for mining (memory may be large for very big datasets)")
    emb_t = emb.T
    batch_size = 512
    for start in tqdm(range(0, num_items, batch_size), desc="Mining batches"):
        end = min(start + batch_size, num_items)
        batch_emb = emb[start:end]  # shape (B, D)
        sims = np.matmul(batch_emb, emb_t)  # (B, N)
        for i_local, i_global in enumerate(range(start, end)):
            anchor_label = labels[i_global]
            # positive: sample another index from the same label
            pos_candidates = label_to_indices.get(anchor_label, [])
            if len(pos_candidates) < 2:
                continue
            pos_idx = i_global
            # ensure pos_idx != anchor
            while pos_idx == i_global:
                pos_idx = random.choice(pos_candidates)

            # negative: take top_k most similar indices that have a different label
            sim_row = sims[i_local]
            neg_candidates = np.argsort(-sim_row)
            neg_idx = None
            for cand in neg_candidates:
                if cand == i_global:
                    continue
                if labels[cand] != anchor_label:
                    neg_idx = cand
                    break
            if neg_idx is None:
                # fallback to random negative
                neg_label = random.choice([l for l in set(labels) if l != anchor_label])
                neg_idx = random.choice(label_to_indices[neg_label])

            examples.append(InputExample(texts=[texts[i_global], texts[pos_idx], texts[neg_idx]]))
            if len(examples) >= n_samples:
                return examples
    return examples  # This return should be at the same indentation level as the loop

# Dimensionality reduction and save functions
def reduce_and_save(embeddings, texts, labels, name, broad_labels=None):
    print(f"Reducing {name} embeddings to 2D")
    # PCA to 50 dims (or min(n_features, 50))
    n_components = min(50, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_pca)

    df_out = pd.DataFrame({
        "text": texts,
        "label": labels,
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1]
    })
    if broad_labels is not None:
        df_out["broad_label_id"] = broad_labels

    out_csv = os.path.join(embeddings_root, f"{name}_{selected_data}_2d_projection.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"2D projection saved to CSV: {out_csv}")

    plots_root = "plots"
    os.makedirs(plots_root, exist_ok=True)
    out_plot = os.path.join(plots_root, f"{name}_{selected_data}_2d_plot.png")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="x", y="y",
        hue="label",
        palette="tab20",
        s=30,
        alpha=0.7,
        data=df_out,
        legend=False
    )
    plt.title(f"2D Projection of {name} Embeddings", fontsize=14)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"2D plot saved: {out_plot}")

    return out_csv, out_plot


# Save train embeddings
def save_train_embeddings(model: SentenceTransformer, name: str):
    texts = df["text"].tolist()  
    labels = df[label_column].tolist()

    print(f"Generating embeddings for {selected_data} set ({len(texts)}) using model {name}")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    out_path = os.path.join(embeddings_root, f"{name}_{selected_data}_embeddings.npz")
    print(f"Saving {selected_data} embeddings to: {out_path}")
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
        labels=np.array(labels, dtype=object)
    )
    print("Saved embeddings:", embeddings.shape)

    reduce_and_save(embeddings, texts, labels, name)
    return out_path

# Training function
def freeze_encoder_keep_head(model: SentenceTransformer):
    """Freeze transformer encoder parameters, leave pooling/projection layers trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = False
        # heuristics to unfreeze pooling/dense/projection heads
        if any(k in name.lower() for k in ("pool", "pooler", "dense", "projection", "classifier", "pooling")):
            param.requires_grad = True

def train_and_embed_train(approach_name, example_creator_fn, loss_factory,
                         n_samples=2000, batch_size=32, epochs=3, warmup_ratio=0.1,
                         use_mined_triplets=False):
    print("\nApproach:", approach_name)

    # instantiate base model once (for mining or freeze)
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Freeze encoder by default to avoid catastrophic forgetting
    freeze_encoder_keep_head(base_model)

    # If using mined triplets, create examples using the pretrained (frozen) base_model
    if use_mined_triplets and example_creator_fn is create_triplet_examples_mined:
        examples = example_creator_fn(df, pretrained_model=base_model, n_samples=n_samples)
    else:
        examples = example_creator_fn(df, n_samples=n_samples)

    print(f"Created {len(examples)} examples")
    if len(examples) == 0:
        raise ValueError("No training examples created. Check your dataset and example creator.")

    if len(examples) < batch_size:
        batch_size = max(2, len(examples))
        print(f"[Info] Adjusted batch_size to {batch_size} due to small dataset.")

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=False)


    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    freeze_encoder_keep_head(model)

    
    if loss_factory is losses.TripletLoss:
        loss = losses.TripletLoss(model=model,
                                  distance_metric=losses.TripletDistanceMetric.COSINE,
                                  triplet_margin=0.2)
    elif loss_factory is losses.MultipleNegativesRankingLoss:
        loss = losses.MultipleNegativesRankingLoss(model)
    elif loss_factory is losses.CosineSimilarityLoss:
        loss = losses.CosineSimilarityLoss(model)
    else:
        loss = loss_factory(model)

    model_path = os.path.join(model_root, approach_name)
    total_steps = max(1, len(loader) * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    print(f"Training {approach_name} with {len(examples)} examples, batch_size={batch_size}, epochs={epochs}, warmup_steps={warmup_steps}")
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=model_path,
        show_progress_bar=True,
        optimizer_params={'lr': 1e-5, 'eps':1e-8, 'weight_decay':0.01}
    )
    print(f"Model saved to: {model_path}")

    emb_path = save_train_embeddings(model, approach_name)
    return model_path, emb_path

# Run for each approach (train embeddings)
train_and_embed_train(
    "triplet",
    create_triplet_examples_mined,
    losses.TripletLoss,
    n_samples=2000,
    batch_size=32,
    epochs=2,
    use_mined_triplets=True
)

train_and_embed_train(
    "simcse",
    create_simcse_examples,
    losses.MultipleNegativesRankingLoss,
    n_samples=2000,
    batch_size=32,
    epochs=2
)

train_and_embed_train(
    "cosine",
    create_cosine_examples,
    losses.CosineSimilarityLoss,
    n_samples=2000,
    batch_size=32,
    epochs=2
)

print("\nModels + train embeddings saved.")

# Davies–Bouldin Index on 2D projections for train embeddings
from src.Evaluations.davies_bouldin import compute_davies_bouldin_index

for model_name in ["triplet", "simcse", "cosine"]:
    proj_csv = os.path.join(embeddings_root, f"{model_name}_{selected_data}_2d_projection.csv")
    
    if os.path.exists(proj_csv):
        df_proj = pd.read_csv(proj_csv)
        emb_2d = df_proj[["x", "y"]].values
        labels = df_proj["label"].values
        compute_davies_bouldin_index(emb_2d, labels, model_name=model_name)
    else:
        print(f"[Skip] {proj_csv} not found")


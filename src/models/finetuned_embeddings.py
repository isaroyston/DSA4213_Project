import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

from loading import load_data

"""
Model 3: Fine-Tuned Embeddings + Embedding Saving (Test Set Only)

This script fine-tunes MiniLM embeddings on the news dataset using three contrastive
losses (Triplet, SimCSE, Cosine). After each model is trained, it generates embeddings
for the test set, and saves them.

Outputs:
finetuned_models
- finetuned_models/triplet/
- finetuned_models/simcse/
- finetuned_models/cosine/
embeddings_output
- embeddings_output/triplet_embeddings.npz
- embeddings_output/simcse_embeddings.npz
- embeddings_output/cosine_embeddings.npz
"""

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load data
df_train, df_test = load_data()
print(f"Loaded train: {len(df_train)} docs, test: {len(df_test)} docs")
print(f"Number of labels: {df_train['label'].nunique()}")

# Directories
model_root = "finetuned_models"
os.makedirs(model_root, exist_ok=True)
embeddings_root = "embeddings_output"
os.makedirs(embeddings_root, exist_ok=True)

# Example creators

def create_triplet_examples(df, n_samples=500):
    examples = []
    labels = df['label'].unique()
    print(f"Creating {n_samples} triplet examples")
    pbar = tqdm(total=n_samples, desc="Triplets")
    while len(examples) < n_samples:
        anchor_label = np.random.choice(labels)
        anchor_df = df[df['label'] == anchor_label]
        if len(anchor_df) < 2:
            continue
        anchor_text, positive_text = anchor_df.sample(2)['text'].values
        negative_label = np.random.choice([l for l in labels if l != anchor_label])
        negative_text = df[df['label'] == negative_label].sample(1)['text'].values[0]
        examples.append(InputExample(texts=[anchor_text, positive_text, negative_text]))
        pbar.update(1)
    pbar.close()
    return examples

def create_simcse_examples(df, n_samples=500):
    print(f"Creating {n_samples} SimCSE examples (duplicate texts for dropout views)")
    texts = df['text'].sample(n=min(n_samples, len(df)), replace=len(df) < n_samples).tolist()
    examples = [InputExample(texts=[t, t]) for t in texts]
    return examples

def create_cosine_examples(df, n_samples=500):
    examples = []
    labels = df['label'].unique()
    print(f"Creating {n_samples} cosine similarity examples")
    half = n_samples // 2
    
    pbar = tqdm(total=half, desc="Positive pairs")
    while len([e for e in examples if e.label == 1.0]) < half:
        anchor_label = np.random.choice(labels)
        anchor_df = df[df['label'] == anchor_label]
        if len(anchor_df) < 2:
            continue
        pos_texts = anchor_df.sample(2)['text'].values
        examples.append(InputExample(texts=[pos_texts[0], pos_texts[1]], label=1.0))
        pbar.update(1)
    pbar.close()
    
    pbar = tqdm(total=half, desc="Negative pairs")
    while len([e for e in examples if e.label == 0.0]) < half:
        anchor_label = np.random.choice(labels)
        anchor_df = df[df['label'] == anchor_label]
        if len(anchor_df) < 1:
            continue
        anchor_text = anchor_df.sample(1)['text'].values[0]
        neg_label = np.random.choice([l for l in labels if l != anchor_label])
        neg_text = df[df['label'] == neg_label].sample(1)['text'].values[0]
        examples.append(InputExample(texts=[anchor_text, neg_text], label=0.0))
        pbar.update(1)
    pbar.close()
    
    return examples

# save embeddings (only test set)


def save_test_embeddings(model: SentenceTransformer, name: str):
    texts = df_test["text"].tolist()
    labels = df_test["label"].tolist()
    print(f"Generating embeddings for test set ({len(texts)}) using model {name}")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    out_path = os.path.join(embeddings_root, f"{name}_embeddings.npz")
    print(f"Saving test embeddings to: {out_path}")
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
        labels=np.array(labels, dtype=object)
    )
    print("Saved embeddings:", embeddings.shape)
    return out_path


# Training func


def train_and_embed_test(approach_name, example_creator, loss_class,
                         n_samples=1000, batch_size=32, epochs=3, warmup_ratio=0.1):
    print("\nApproach:", approach_name)
    examples = example_creator(df_train, n_samples=n_samples)
    print(f"Created {len(examples)} examples")
    if len(examples) < batch_size:
        batch_size = max(2, len(examples))
        print(f"[Info] Adjusted batch_size to {batch_size} due to small dataset.")
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    loss = loss_class(model)
    model_path = os.path.join(model_root, approach_name)
    total_steps = len(loader) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))  
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=model_path,
        show_progress_bar=True
    )
    print(f"Model saved to: {model_path}")
    emb_path = save_test_embeddings(model, approach_name)
    return model_path, emb_path


# Run for each approach


# Triplet
train_and_embed_test(
    "triplet",
    create_triplet_examples,
    losses.TripletLoss,
    n_samples=1000,
    batch_size=32,
    epochs=3
)

# SimCSE
train_and_embed_test(
    "simcse",
    create_simcse_examples,
    losses.MultipleNegativesRankingLoss,
    n_samples=1000,
    batch_size=32,
    epochs=3
)

# Cosine
train_and_embed_test(
    "cosine",
    create_cosine_examples,
    losses.CosineSimilarityLoss,
    n_samples=1000,
    batch_size=32,
    epochs=3
)

print("\nModels + test embeddings saved.")

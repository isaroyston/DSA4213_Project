import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from bertopic import BERTopic
import google.generativeai as genai
from time import sleep
from tqdm import tqdm

from loading import load_data

"""
Model 3: Fine-Tuned Embeddings + BERTopic + LLM

This script fine-tunes MiniLM embeddings on the news dataset using three contrastive
losses (Triplet, SimCSE, Cosine). The fine-tuned embeddings are integrated into
BERTopic for topic modeling. Gemini is then used to refine topic labels.

Outputs:
- Fine-tuned MiniLM model directories for each approach
- Final CSVs (triplet_final.csv, simcse_final.csv, cosine_final.csv) containing:
    • Document text
    • Ground-truth label
    • Topic ID
    • Embedding vector columns
    • Refined human-readable topic label
"""

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Make cuDNN deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load train/test
df_train, df_test = load_data()
print(f"Loaded train: {len(df_train)} docs, test: {len(df_test)} docs")
print(f"Categories: {df_train['label'].nunique()}")

# Output directory
output_dir = "finetuned_models_output"
os.makedirs(output_dir, exist_ok=True)


# Data example creators

def create_triplet_examples(df, n_samples=500):
    """Create triplet examples for contrastive learning."""
    examples = []
    labels = df['label'].unique()
    
    print(f"Creating {n_samples} triplet examples...")
    for _ in tqdm(range(n_samples)):
        anchor_label = np.random.choice(labels)
        anchor_df = df[df['label'] == anchor_label]
        
        if len(anchor_df) < 2:
            continue

        anchor_text = anchor_df.sample(1)['text'].values[0]
        positive_text = anchor_df.sample(1)['text'].values[0]

        negative_label = np.random.choice([l for l in labels if l != anchor_label])
        negative_text = df[df['label'] == negative_label].sample(1)['text'].values[0]

        examples.append(InputExample(texts=[anchor_text, positive_text, negative_text]))
    
    return examples


def create_simcse_examples(df, n_samples=500):
    """Create positive pairs for SimCSE-style training."""
    examples = []
    
    print("Creating SimCSE examples...")
    for label in tqdm(df['label'].unique()):
        label_df = df[df['label'] == label]
        if len(label_df) < 2:
            continue
        n_pairs = min(n_samples // df['label'].nunique(), len(label_df) // 2)
        for _ in range(n_pairs):
            texts = label_df.sample(2)['text'].values
            examples.append(InputExample(texts=[texts[0], texts[1]]))
    
    return examples


def create_cosine_examples(df, n_samples=50):
    """Create labeled pairs for cosine similarity loss."""
    examples = []
    labels = df['label'].unique()
    
    print(f"Creating {n_samples} cosine similarity examples...")
    for _ in tqdm(range(n_samples)):
        anchor_label = np.random.choice(labels)
        anchor_df = df[df['label'] == anchor_label]
        
        if len(anchor_df) < 2:
            continue

        pos_texts = anchor_df.sample(2)['text'].values
        examples.append(InputExample(texts=[pos_texts[0], pos_texts[1]], label=1.0))

        neg_label = np.random.choice([l for l in labels if l != anchor_label])
        neg_text = df[df['label'] == neg_label].sample(1)['text'].values[0]
        examples.append(InputExample(texts=[pos_texts[0], neg_text], label=0.0))
    
    return examples


# Gemini client & schema
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel('gemini-2.5-flash')


def map_cluster_to_label(docs_by_topic, max_retries=3):
    """
    Map topic clusters to human-readable labels using Gemini.
    Includes retry logic and error handling.
    """
    cluster_labels = {}
    
    print(f"Mapping {len(docs_by_topic)} topics to labels using Gemini")
    for topic_id, sample_texts in tqdm(docs_by_topic.items()):
        sample_size = min(7, len(sample_texts))
        prompt = (
            f"These are sample texts from topic {topic_id}:\n\n"
            + "\n".join(sample_texts[:sample_size])
            + "\n\nAssign ONE short category label (like 'Sports', 'Politics', 'Technology'). "
            + "Return ONLY the label, nothing else."
        )
        
        # Retry logic for API calls
        for attempt in range(max_retries):
            try:
                resp = model_gemini.generate_content(prompt)
                cluster_labels[topic_id] = resp.text.strip()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\nRetrying topic {topic_id} (attempt {attempt + 1}/{max_retries})")
                    sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"\nFailed to label topic {topic_id} after {max_retries} attempts: {e}")
                    cluster_labels[topic_id] = f"Topic_{topic_id}"
    
    return cluster_labels


def save_with_bertopic(model, df, name, n_topics=20):
    """
    Fits BERTopic with the fine-tuned embedding model, maps topics to labels with Gemini,
    and saves enriched CSV with embeddings.
    """
    df = df.reset_index(drop=True)
    
    print(f"\nProcessing {name.upper()} approach")
    
    # Fit BERTopic
    print("Fitting BERTopic model")
    topic_model = BERTopic(embedding_model=model, nr_topics=20, verbose=True)
    topics, probs = topic_model.fit_transform(df["text"].tolist())

    # Generate embeddings
    print("Generating embeddings")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    
    # Create base dataframe
    out_df = pd.DataFrame({
        'Document': df["text"].values,
        'Topic': topics,
        'label': df["label"].values
    })


    print("Adding embeddings to df")
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
    )
    out_df = pd.concat([out_df, embedding_df], axis=1)

    # Map topics to labels
    print("Mapping topics to labels")
    docs_by_topic = {}
    for topic_id in set(topics):
        if topic_id != -1:
            docs_by_topic[topic_id] = out_df[out_df["Topic"] == topic_id]["Document"].tolist()

    cluster_to_label = map_cluster_to_label(docs_by_topic)
    out_df["predicted_label"] = out_df["Topic"].map(cluster_to_label)
    out_df["predicted_label"] = out_df["predicted_label"].fillna("Outlier")

    # Save output
    final_path = os.path.join(output_dir, f"{name}_final.csv")
    print(f"Saving results to {final_path}")
    out_df.to_csv(final_path, index=False)
    
    print(f"Saved: {final_path} ({len(out_df)} docs, {out_df['Topic'].nunique()} topics, {embeddings.shape[1]} embedding dims)")
    
    return final_path


# Training + BERTopic

print("\nAPPROACH 1: Triplet Loss")

triplet_examples = create_triplet_examples(df_train)
triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=16)
model_triplet = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
triplet_loss = losses.TripletLoss(model_triplet)

print(f"Fine-tuning with {len(triplet_examples)} triplet examples")
model_triplet.fit(
    train_objectives=[(triplet_loader, triplet_loss)],
    epochs=3,
    warmup_steps=100,
    output_path=os.path.join(output_dir, "finetuned_triplet"),
    show_progress_bar=True
)
print("Fine-tuning complete. Running BERTopic on test set")
triplet_final_csv = save_with_bertopic(model_triplet, df_test, "triplet")


print("\nAPPROACH 2: SimCSE (Multiple Negatives Ranking Loss)")

simcse_examples = create_simcse_examples(df_train)
simcse_loader = DataLoader(simcse_examples, shuffle=True, batch_size=16)
model_simcse = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
simcse_loss = losses.MultipleNegativesRankingLoss(model_simcse)

print(f"Fine-tuning with {len(simcse_examples)} SimCSE examples")
model_simcse.fit(
    train_objectives=[(simcse_loader, simcse_loss)],
    epochs=3,
    warmup_steps=100,
    output_path=os.path.join(output_dir, "finetuned_simcse"),
    show_progress_bar=True
)
print("Fine-tuning complete. Running BERTopic on test set")
simcse_final_csv = save_with_bertopic(model_simcse, df_test, "simcse")


print("\nAPPROACH 3: Cosine Similarity Loss")

cosine_examples = create_cosine_examples(df_train)
cosine_loader = DataLoader(cosine_examples, shuffle=True, batch_size=16)
model_cosine = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cosine_loss = losses.CosineSimilarityLoss(model_cosine)

print(f"Fine-tuning with {len(cosine_examples)} cosine examples")
model_cosine.fit(
    train_objectives=[(cosine_loader, cosine_loss)],
    epochs=3,
    warmup_steps=100,
    output_path=os.path.join(output_dir, "finetuned_cosine"),
    show_progress_bar=True
)
print("Fine-tuning complete. Running BERTopic on test set")
cosine_final_csv = save_with_bertopic(model_cosine, df_test, "cosine")


print("\nALL APPROACHES COMPLETED!")
print(f"Triplet output: {triplet_final_csv}")
print(f"SimCSE output:  {simcse_final_csv}")
print(f"Cosine output:  {cosine_final_csv}")
print(f"\nAll models saved in: {output_dir}")
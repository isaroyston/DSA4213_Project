# TF-IDF + SVD Embeddings (Baseline Model)
import numpy as np
import pandas as pd
import pickle
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from umap import UMAP

# Data loading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from loading import load_data
from src.models.model_utils import (
    build_topic_model,
    fit_topic_model,
    transform_topic_model,
    evaluate_bertopic_results,
    get_topic_keywords,
    refine_topics_with_llm
)

df_train, df_test = load_data()
articles_train = df_train['text'].tolist() 
articles_test = df_test['text'].tolist()

def build_vectorizer_svd(max_features=10000, n_components=30):
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 2), 
        max_features=max_features, 
        min_df=3, 
        max_df=0.85
    ) 
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    pipeline = make_pipeline(vectorizer, svd_model)
    return vectorizer, svd_model, pipeline

def compute_embeddings(pipeline, train_texts, test_texts):
    embeddings_train = pipeline.fit_transform(train_texts)
    embeddings_test = pipeline.transform(test_texts)
    return embeddings_train, embeddings_test

def map_to_2d(embeddings_train, embeddings_test, method='umap', pca_components=50):
    if method == 'umap':
        if embeddings_train.shape[1] < pca_components:
            print(f"Skipping PCA (embeddings already {embeddings_train.shape[1]}D)")
            train_for_umap = embeddings_train
            test_for_umap = embeddings_test
        else:
            pca = PCA(n_components=pca_components, random_state=42)
            train_for_umap = pca.fit_transform(embeddings_train)
            test_for_umap = pca.transform(embeddings_test)
        
        umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d_train = umap_model.fit_transform(train_for_umap)
        embeddings_2d_test = umap_model.transform(test_for_umap)
        
    elif method == 'pca':
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d_train = pca.fit_transform(embeddings_train)
        embeddings_2d_test = pca.transform(embeddings_test)
    
    return embeddings_2d_train, embeddings_2d_test

def evaluate_embeddings(embeddings):
    """Evaluate embedding quality"""
    n_clusters = min(50, len(embeddings) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, cluster_labels, sample_size=1000)
    
    return {
        'silhouette_score': silhouette,
        'n_dimensions': embeddings.shape[1],
        'n_samples': embeddings.shape[0]
    }


# EXEC ----------------------------------------
vectorizer, svd_model, pipeline = build_vectorizer_svd(max_features=10000, n_components=30)
tfidf_train, tfidf_test = compute_embeddings(pipeline, articles_train, articles_test)

embedding_metrics = evaluate_embeddings(tfidf_train)
for metric, value in embedding_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

tfidf_2d_train, tfidf_2d_test = map_to_2d(tfidf_train, tfidf_test, method='umap')

# Save embeddings
embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'embeddings_output')
os.makedirs(embeddings_dir, exist_ok=True)
pickle.dump({'train': tfidf_train, 'test': tfidf_test}, open(os.path.join(embeddings_dir, 'tfidf_embeddings.pkl'), 'wb'))
pickle.dump({'train': tfidf_2d_train, 'test': tfidf_2d_test}, open(os.path.join(embeddings_dir, 'tfidf_2d_embeddings.pkl'), 'wb'))
print(f"✓ Embeddings saved to {embeddings_dir}")

# BERTopic clustering
print("\nBuilding BERTopic model...")
topic_model = build_topic_model(vectorizer=vectorizer)
topics_train, probs_train = fit_topic_model(topic_model, articles_train, tfidf_train)
topics_test, probs_test = transform_topic_model(topic_model, articles_test, tfidf_test)

print("\n=== BERTopic Results ===")
bertopic_metrics = evaluate_bertopic_results(topics_train, topics_test)
for metric, value in bertopic_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

#topic_info = topic_keywords(topic_model, n_words=10)
#print(f"\nDiscovered {len(topic_info)} topics")
#print(topic_info[['Topic', 'Count', 'Representation']].head(10))

#refined_topics_df = refine_topic_llm(topic_info)
#print("\n=== Refined Topics (Sample) ===")
#print(refined_topics_df.head(10))

#refined_topics_df = refine_topic_llm(topic_info)





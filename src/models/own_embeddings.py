# own embeddings (baseline model)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from bertopic import BERTopic
import google.generativeai as genai
from pydantic import BaseModel, Field
import json
import pandas as pd
import sys, os

# Data loading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from loading import load_data

df_train, df_test = load_data()
articles_train = df_train['text'].tolist() 
articles_test = df_test['text'].tolist()

# TF-IDF Vectorizer + SVD (For Dimensionality Reduction)
def build_vectorizer_svd(max_features: int = 10000, n_components: int = 100):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=max_features, min_df=3, max_df=0.85) 
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    pipeline = make_pipeline(vectorizer, svd_model)
    return vectorizer, svd_model, pipeline

def compute_embeddings(pipeline, train_texts, test_texts):
    tfidf_train = pipeline.fit_transform(train_texts)
    tfidf_test = pipeline.transform(test_texts)
    return tfidf_train, tfidf_test

def evaluate_embeddings(embeddings, labels=None):
    """Evaluate embedding quality directly"""
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    import numpy as np
    
    # 1. Silhouette Score (clustering quality)
    n_clusters = min(50, len(embeddings) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, cluster_labels, sample_size=1000)
    
    # 2. Cumulative variance (more meaningful than threshold)
    explained_variance = np.var(embeddings, axis=0)
    total_var = np.sum(explained_variance)
    sorted_var = np.sort(explained_variance)[::-1]
    cumsum_var = np.cumsum(sorted_var)
    
    # Dimensions needed for 90% and 95% variance
    dims_90 = np.argmax(cumsum_var >= 0.90 * total_var) + 1
    dims_95 = np.argmax(cumsum_var >= 0.95 * total_var) + 1
    
    # 3. Density
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    avg_distance = np.mean(distances[:, 1:])
    
    results = {
        'silhouette_score': silhouette,
        'dims_for_90pct_variance': dims_90,
        'dims_for_95pct_variance': dims_95,
        'effective_dims_ratio': dims_95 / embeddings.shape[1],
        'avg_nn_distance': avg_distance,
        'n_clusters': n_clusters
    }

    return results

# BERTopic model
def build_topic_model(embedding_model=None, vectorizer=None):
    return BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer, min_topic_size=30, nr_topics=40, verbose=True)

def fit_topic_model(topic_model, articles, precomputed_embeddings): # train set 
    topics, probs = topic_model.fit_transform(articles, precomputed_embeddings)
    return topics, probs

def transform_topic_model(topic_model, articles, precomputed_embeddings): # test set
    topics, probs = topic_model.transform(articles, precomputed_embeddings)
    return topics, probs

def evaluate_bertopic_results(topic_model, topics_train, topics_test):
    """Evaluate BERTopic clustering quality"""
    import numpy as np
    from collections import Counter
    
    # Topic distribution
    topic_counts = Counter(topics_train)
    n_topics = len([t for t in topic_counts if t != -1])
    n_outliers_train = topic_counts.get(-1, 0)
    n_outliers_test = sum(1 for t in topics_test if t == -1)
    
    # Topic sizes
    topic_sizes = [count for topic, count in topic_counts.items() if topic != -1]
    
    return {
        'n_topics': n_topics,
        'outlier_ratio_train': n_outliers_train / len(topics_train),
        'outlier_ratio_test': n_outliers_test / len(topics_test),
        'avg_topic_size': np.mean(topic_sizes),
        'min_topic_size': np.min(topic_sizes),
        'max_topic_size': np.max(topic_sizes)
    }


# BERTopic topic towards LLM
def topic_keywords(topic_model, n_words = 10):
    info = topic_model.get_topic_info()
    removed_outliers_df = info[info['Topic'] != -1]  # There are tons of outliers, not sure what i can do regarding these embeddings
    return removed_outliers_df

#Pydantic dataframe
class RefinedTopic(BaseModel):
    concise_label : str = Field(
        description = "Short human-readable label for the topic (2-5 words)"
    )
    summary : str  = Field(
        description  = "1 sentence summary of the topic"
    )
#LLM (Gemini 2.5)
genai.configure(api_key=os.environ.get("API_KEY"))

def refine_topic_llm(topic_keywords):
    model = genai.GenerativeModel('gemini-2.5-flash')  # Fixed model name
    refined_topics = []
    
    for _, row in topic_keywords.iterrows():
        topic_id = row['Topic']
        keywords = row['Representation']
        
        prompt = f"""
        Based on these topic keywords: {keywords}
        
        Please provide ONLY a valid JSON response with:
        1. A concise label (2-5 words)
        2. A 1-sentence summary
        
        Format your response as JSON:
        {{
            "concise_label": "your label here",
            "summary": "your summary here"
        }}
        """    
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from markdown if present
            if '```' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end > start:
                    response_text = response_text[start:end]
            
            response_json = json.loads(response_text)
            refined_topic = RefinedTopic(**response_json)
            
            refined_topics.append({
                'Topic': topic_id,
                'concise_label': refined_topic.concise_label,
                'summary': refined_topic.summary
            })
            
        except Exception as e:
            print(f"Error processing topic {topic_id}: {e}")
            refined_topics.append({
                'Topic': topic_id,
                'concise_label': f"Topic {topic_id}",
                'summary': f"Keywords: {', '.join(str(kw) for kw in keywords[:3])}"
            })

    return pd.DataFrame(refined_topics)



vectorizer, svd_model, pipeline = build_vectorizer_svd(max_features=10000, n_components=30)
tfidf_train, tfidf_test = compute_embeddings(pipeline, articles_train, articles_test)

print("\n=== Embedding Quality Evaluation ===")
embedding_metrics = evaluate_embeddings(tfidf_train)
for metric, value in embedding_metrics.items():
    print(f"{metric}: {value:.4f}")

topic_model = build_topic_model(embedding_model=None, vectorizer=vectorizer)
topics_train, probs_train = fit_topic_model(topic_model, articles_train, tfidf_train)
topics_test, probs_test = transform_topic_model(topic_model, articles_test, tfidf_test)

print("\n=== BERTopic Results ===")
bertopic_metrics = evaluate_bertopic_results(topic_model, topics_train, topics_test)
for metric, value in bertopic_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

topic_info = topic_keywords(topic_model, n_words=10)
print(f"\nDiscovered {len(topic_info)} topics")
print(topic_info[['Topic', 'Count', 'Representation']].head(10))

refined_topics_df = refine_topic_llm(topic_info)
print("\n=== Refined Topics (Sample) ===")
print(refined_topics_df.head(10))

#refined_topics_df = refine_topic_llm(topic_info)





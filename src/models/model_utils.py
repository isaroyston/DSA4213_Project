# Model Utilities for BERTopic and LLM Topic Refinement
import numpy as np
import pandas as pd
import google.generativeai as genai
import json
import os
from pydantic import BaseModel, Field
from bertopic import BERTopic
from collections import Counter

# ============= BERTopic Functions =============
def build_topic_model(vectorizer=None, min_topic_size=30, nr_topics=40):
    """Build BERTopic model"""
    return BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        verbose=True
    )

def fit_topic_model(topic_model, articles, embeddings):
    """Fit BERTopic model on training data"""
    topics, probs = topic_model.fit_transform(articles, embeddings)
    return topics, probs

def transform_topic_model(topic_model, articles, embeddings):
    """Transform test data using fitted BERTopic model"""
    topics, probs = topic_model.transform(articles, embeddings)
    return topics, probs

def evaluate_bertopic_results(topics_train, topics_test):
    """Evaluate BERTopic clustering quality"""
    topic_counts = Counter(topics_train)
    n_topics = len([t for t in topic_counts if t != -1])
    n_outliers_train = topic_counts.get(-1, 0)
    n_outliers_test = sum(1 for t in topics_test if t == -1)
    topic_sizes = [count for topic, count in topic_counts.items() if topic != -1]
    
    return {
        'n_topics': n_topics,
        'outlier_ratio_train': n_outliers_train / len(topics_train),
        'outlier_ratio_test': n_outliers_test / len(topics_test),
        'avg_topic_size': np.mean(topic_sizes) if topic_sizes else 0,
        'min_topic_size': np.min(topic_sizes) if topic_sizes else 0,
        'max_topic_size': np.max(topic_sizes) if topic_sizes else 0
    }

def get_topic_keywords(topic_model):
    """Extract topic keywords, removing outliers"""
    info = topic_model.get_topic_info()
    return info[info['Topic'] != -1]

# ============= LLM Topic Refinement =============
class RefinedTopic(BaseModel):
    """Pydantic model for refined topic output"""
    concise_label: str = Field(
        description="Short human-readable label for the topic (2-5 words)"
    )
    summary: str = Field(
        description="1 sentence summary of the topic"
    )

def refine_topics_with_llm(topic_keywords_df, api_key=None):
    """
    Refine topic keywords using Gemini LLM
    
    Args:
        topic_keywords_df: DataFrame with columns ['Topic', 'Representation']
        api_key: Gemini API key (if None, reads from env variable API_KEY)
    
    Returns:
        DataFrame with refined topic labels and summaries
    """
    if api_key is None:
        api_key = os.environ.get("API_KEY")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    refined_topics = []
    
    for _, row in topic_keywords_df.iterrows():
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
                'summary': refined_topic.summary,
                'keywords': keywords
            })
            
        except Exception as e:
            print(f"Error processing topic {topic_id}: {e}")
            refined_topics.append({
                'Topic': topic_id,
                'concise_label': f"Topic {topic_id}",
                'summary': f"Keywords: {', '.join(str(kw) for kw in keywords[:3])}",
                'keywords': keywords
            })
    
    return pd.DataFrame(refined_topics)

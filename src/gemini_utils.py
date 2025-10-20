import os, json, time, hashlib
from typing import List, Literal, Optional, Dict, Any
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import os, json
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from google import genai
import os
import umap
import hdbscan
from bertopic import BERTopic
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env from one directory up
load_dotenv(dotenv_path="../.env")

# Access your key
api_key = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY:", api_key[:6], "..." if api_key else "MISSING")

# -------- Pydantic schema --------
class TopicLabel(BaseModel):
    tuned_topic_name: Literal[""] = Field(..., description="One of 20 predefined Title Case topic labels"),
    short_explanation: str = Field(..., description="1–2 sentence summary of the cluster theme")

# -------- helpers --------
def _as_list(x) -> List[str]:
    """Robustly turn a cell into a list of strings (handles list/JSON/CSV/plain)."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    s = str(x).strip()
    if not s:
        return []
    # try JSON list
    try:    
        v = json.loads(s)
        if isinstance(v, list):
            return [str(t) for t in v]
    except Exception:
        pass
    # try comma/pipe delim
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    if "|" in s:
        return [t.strip() for t in s.split("|") if t.strip()]
    # fallback: single token/string
    return [s]

def make_prompt_for_topic(keywords: List[str], rep_docs: List[str]) -> str:
    return f"""
            You are a News Analytics assistant analyzing article clusters.
            Produce a concise, meaningful topic label and a short explanation.
            Given that we want to classify the clusters accurately, into the most relevant and specific topics possible, these are some guidelines to follow:
            - Only cluster topics into these categories: Politics, Health, Technology, Environment, Business, Entertainment, Sports, Science, World News, Lifestyle, Others.
            - Ensure the topic label reflects the main theme of the keywords and representative documents.
            - If you are unsure, you are allowed to cluster it as Others

            Return ONLY valid JSON with keys:
            {{
            "tuned_topic_name": "string",
            "short_explanation": "string"
            }}

            keywords = {keywords[:5]}
            representative_feedback = {[d[:50] for d in rep_docs[:8]]}
            """.strip()

def label_topics_from_df(
    df: pd.DataFrame,
    api_key: str = None,
    model_name: str = 'gemini-2.5-flash',
    topic_col: str = "Topic",
    words_col: str = "Representation",
    reps_col: str = "Representative_Docs",
) -> pd.DataFrame:
    """
    Expects df with columns: Topic, Representation, Representative_Docs.
    Returns a new df: Topic, tuned_topic_name, short_explanation.
    """
    # drop outlier column if present
    if -1 in df[topic_col].values:
        df = df[df[topic_col] != -1].copy()

    rows = []
    for _, r in df.iterrows():
        topic_id = int(r[topic_col])
        keywords = _as_list(r[words_col])
        rep_docs = _as_list(r[reps_col])

        prompt = make_prompt_for_topic(keywords, rep_docs)

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": TopicLabel,
            },
        )

        generated_text = response.candidates[0].content.parts[0].text
        data = json.loads(generated_text)
        topic_label = TopicLabel.model_validate(data)
        print(topic_label)

        rows.append(
            {"Topic": topic_id,
             "tuned_topic_name": topic_label.tuned_topic_name,
             "short_explanation": topic_label.short_explanation}
        )

    return pd.DataFrame(rows)
import re
import string
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sqlalchemy import create_engine, text
from sqlalchemy.types import Text, Integer
from dotenv import load_dotenv
import os
from sklearn.datasets import fetch_20newsgroups

# ENVIRONMENT SETUP
load_dotenv()
DATABASE_URL = os.getenv("DB_URL")

# GLOBALS
STOPWORDS = set(stopwords.words("english"))
NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
CATEGORY_MAP_INT = {
    # 1. Computers & Technology
    "comp.graphics": 1,
    "comp.os.ms-windows.misc": 1,
    "comp.sys.ibm.pc.hardware": 1,
    "comp.sys.mac.hardware": 1,
    "comp.windows.x": 1,

    # 2. Science & Engineering
    "sci.crypt": 2,
    "sci.electronics": 2,
    "sci.med": 2,
    "sci.space": 2,

    # 3. Recreation (Vehicles & Hobbies)
    "rec.autos": 3,
    "rec.motorcycles": 3,

    # 4. Sports
    "rec.sport.baseball": 4,
    "rec.sport.hockey": 4,

    # 5. Religion
    "alt.atheism": 5,
    "soc.religion.christian": 5,
    "talk.religion.misc": 5,

    # 6. Politics & Society
    "talk.politics.guns": 6,
    "talk.politics.mideast": 6,
    "talk.politics.misc": 6,

    # 7. Marketplace / Miscellaneous
    "misc.forsale": 7
}

# BASIC CLEANING
def clean_text_basic(text):
    """
    Perform baseline text cleaning suitable for all NLP pipelines.
    Steps:
    - Remove headers, quoted lines, emails, URLs, punctuation
    - Normalize whitespace and lowercase
    """
    # Remove headers (take only body after first blank line)
    if "\n\n" in text:
        text = text.split("\n\n", 1)[1]
    # Remove quoted lines starting with > or :
    lines = text.split("\n")
    lines = [line for line in lines if not line.startswith(">") and not line.startswith(":")]
    text = "\n".join(lines)
    # Remove emails and URLs
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def filter_short_docs(df, min_length=50):
    """Drop documents shorter than `min_length` characters."""
    before = len(df)
    df = df[df["text"].str.len() > min_length]
    dropped = before - len(df)
    print(f"Filtered {dropped} short docs ({dropped} dropped, {len(df)} kept).")
    return df

# ADDITIONAL CLEANING VARIANTS FOR SPARSE REPRESENTATION CLEANING
def lemmatize_text(text):
    """Lemmatize tokens using SpaCy."""
    doc = NLP(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

def clean_for_sparse_repr(text, lemmatize=True):
    """
    Cleaning variant for sparse vector representations
    (TF-IDF, bag-of-words, LDA, NMF, etc.)
    - Removes stopwords
    - Optional lemmatization
    Suitable for non-contextual text models.
    """
    text = clean_text_basic(text)
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
    if lemmatize:
        text = lemmatize_text(text)
    return text

# ABLATION UTILITIES FOR TOKEN LIMIT HANDLING
def truncate_text(text, max_words=512, position="start"):
    """
    Truncate text to simulate limited-context models.
    position = 'start' (keep first N words) or 'end' (keep last N words).
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) if position == "start" else " ".join(words[-max_words:])

def sliding_window_segments(text, window_size=512, stride=256):
    """
    Create overlapping text segments for long-text ablation experiments.
    Returns a list of segments.
    """
    words = text.split()
    segments = []
    for i in range(0, len(words), stride):
        segment = " ".join(words[i : i + window_size])
        if len(segment.split()) < 10:
            break
        segments.append(segment)
    return segments

# MAPPING OF NEWS CATEGORIES TO BROADER ONES
def add_broad_category_int(df):
    """Map fine-grained labels to integer IDs (1–7)."""
    df = df.copy()
    df["broad_label_id"] = df["label"].map(CATEGORY_MAP_INT)
    return df

# MAIN CLEANING PIPELINE
def prepare_cleaned_datasets():
    """Load, clean, and save the base cleaned 20 Newsgroups dataset."""

    print("Loading 20 Newsgroups dataset...")
    train = fetch_20newsgroups(subset="train")
    test = fetch_20newsgroups(subset="test")

    print("Applying basic cleaning...")
    df_train = pd.DataFrame(
        {
            "text": [clean_text_basic(t) for t in train.data],
            "label_id": train.target,
            "label": [train.target_names[i] for i in train.target],
        }
    )
    df_test = pd.DataFrame(
        {
            "text": [clean_text_basic(t) for t in test.data],
            "label_id": test.target,
            "label": [test.target_names[i] for i in test.target],
        }
    )

    # Filter out short docs
    df_train = filter_short_docs(df_train)
    df_test = filter_short_docs(df_test)

    # Save base cleaned versions locally
    df_train.to_csv("newsgroups_train_clean_basic.csv", index=False)
    df_test.to_csv("newsgroups_test_clean_basic.csv", index=False)
    print("Saved base cleaned CSVs locally.")

    return df_train, df_test

# SAVE ALL VARIANTS TO POSTGRESQL
def save_all_variants_to_postgres(df_train, df_test, engine):
    """
    Save three dataset variants:
    - basic (for dense embeddings & LLMs)
    - sparse_repr (for TF-IDF / LDA ablation)
    - broad_category (with integer broad_label_id mapping, 1-7)

    Ensures no duplicate inserts across reruns by truncating existing tables.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS newsgroup"))

    variants = {
        "basic": (df_train, df_test),
        "sparse_repr": (
            df_train.assign(text=df_train["text"].apply(clean_for_sparse_repr)),
            df_test.assign(text=df_test["text"].apply(clean_for_sparse_repr)),
        ),
        "broad_category": (
            add_broad_category_int(df_train),
            add_broad_category_int(df_test),
        )
    }

    for variant, (train_df, test_df) in variants.items():
        print(f"Uploading {variant} variant to PostgreSQL...")

        with engine.begin() as conn:
            # Ensure table exists
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS newsgroup.train_{variant} (
                    text TEXT,
                    label_id INTEGER,
                    label TEXT,
                    broad_label_id INTEGER
                );
            """))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS newsgroup.test_{variant} (
                    text TEXT,
                    label_id INTEGER,
                    label TEXT,
                    broad_label_id INTEGER
                );
            """))

            # Truncate existing rows to avoid duplicates
            conn.execute(text(f"TRUNCATE TABLE newsgroup.train_{variant};"))
            conn.execute(text(f"TRUNCATE TABLE newsgroup.test_{variant};"))

        # Insert fresh data
        train_df.to_sql(
            f"train_{variant}",
            engine,
            schema="newsgroup",
            if_exists="append",
            index=False,
            dtype={
                "text": Text(),
                "label_id": Integer(),
                "label": Text(),
                "broad_label_id": Integer()
            },
        )
        test_df.to_sql(
            f"test_{variant}",
            engine,
            schema="newsgroup",
            if_exists="append",
            index=False,
            dtype={
                "text": Text(),
                "label_id": Integer(),
                "label": Text(),
                "broad_label_id": Integer()
            },
        )

    print("\nAll dataset variants successfully uploaded to PostgreSQL.")

# ENTRY POINT
if __name__ == "__main__":
    df_train, df_test = prepare_cleaned_datasets()
    try:
        print("\nConnecting to PostgreSQL...")
        engine = create_engine(DATABASE_URL)
        save_all_variants_to_postgres(df_train, df_test, engine)
    except Exception as e:
        print("\n[ERROR] Could not save to PostgreSQL:", e)
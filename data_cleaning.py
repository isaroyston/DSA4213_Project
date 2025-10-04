from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import re
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import string

# Load environment variables
load_dotenv()
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

print("Loading 20 Newsgroups dataset")
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

print("\n Dataset Structure")
print(f"Categories ({len(newsgroups_train.target_names)}):")
pprint(newsgroups_train.target_names)
print(f"\nTrain documents: {len(newsgroups_train.data)}")
print(f"Test documents: {len(newsgroups_test.data)}")


print("\n Exploring Dataset (Before Cleaning)")
print(newsgroups_train.data[0][:400])
print(f"\nLabel: {newsgroups_train.target_names[newsgroups_train.target[0]]}")


print("CLEANING DATA")

import string

def clean_text(text):
    """Remove headers, quotes, emails, URLs, punctuation, and clean whitespace"""
    # Remove headers (take only body after first blank line)
    if '\n\n' in text:
        text = text.split('\n\n', 1)[1]
    # Remove quoted lines (starting with > or :)
    lines = text.split('\n')
    lines = [line for line in lines if not line.startswith('>') and not line.startswith(':')]
    text = '\n'.join(lines)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Clean training data
print("Cleaning training set")
newsgroups_train.data = [clean_text(text) for text in newsgroups_train.data]

# Clean test data
print("Cleaning test set")
newsgroups_test.data = [clean_text(text) for text in newsgroups_test.data]

print(f"\nCleaned {len(newsgroups_train.data)} train documents")
print(f"Cleaned {len(newsgroups_test.data)} test documents")

print("\n Sample Document (After Cleaning)")
print(newsgroups_train.data[0][:400])

# Check cleaning effectiveness
avg_len_before = sum(len(text) for text in fetch_20newsgroups(subset='train').data) / len(newsgroups_train.data)
avg_len_after = sum(len(text) for text in newsgroups_train.data) / len(newsgroups_train.data)
print(f"\nAvg document length: {avg_len_before:.0f} → {avg_len_after:.0f} chars")
print(f"Reduction: {(1 - avg_len_after/avg_len_before)*100:.1f}%")


# Training set
df_train = pd.DataFrame({
    "text": newsgroups_train.data,
    "label_id": newsgroups_train.target,
    "label": [newsgroups_train.target_names[i] for i in newsgroups_train.target]
})

# Test set
df_test = pd.DataFrame({
    "text": newsgroups_test.data,
    "label_id": newsgroups_test.target,
    "label": [newsgroups_test.target_names[i] for i in newsgroups_test.target]
})

# Save CSVs
df_train.to_csv("newsgroups_train_clean.csv", index=False)
df_test.to_csv("newsgroups_test_clean.csv", index=False)

print("\n Saved cleaned train set to 'newsgroups_train_clean.csv'")
print(" Saved cleaned test set to 'newsgroups_test_clean.csv'")


# try:
#     conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
#     engine = create_engine(conn_str)
#     df_train.to_sql("newsgroups_train", engine, if_exists="replace", index=False)
#     df_test.to_sql("newsgroups_test", engine, if_exists="replace", index=False)
#     print("\n Saved cleaned datasets into PostgreSQL")
# except Exception as e:
#     print("\n[ERROR] Could not save to PostgreSQL:", e)

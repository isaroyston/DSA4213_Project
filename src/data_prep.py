import pandas as pd
from sklearn.datasets import fetch_20newsgroups

def load_20newsgroups():
    print("Loading 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})
    return df

def clean_data(df):
    # Placeholder for actual cleaning logic
    df['text'] = df['text'].str.replace('\n', ' ').str.strip()
    # Add more cleaning as needed
    return df

if __name__ == "__main__":
    df = load_20newsgroups()
    df = clean_data(df)
    df.to_csv("20newsgroups_clean.csv", index=False)
    print("Saved cleaned data to 20newsgroups_clean.csv")
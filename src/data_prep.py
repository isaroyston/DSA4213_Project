import pandas as pd
from sklearn.datasets import fetch_20newsgroups

def load_20newsgroups():
    #here

def clean_data(df):
    #here

if __name__ == "__main__":
    df = load_20newsgroups()
    df = clean_data(df)
    df.to_csv("20newsgroups_clean.csv", index=False)
    print("Saved cleaned data to 20newsgroups_clean.csv")

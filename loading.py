import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL
DATABASE_URL = os.getenv("DB_URL")

# Ensure SSL mode for Render
if "?sslmode=require" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Load datasets from the "newsgroup" schema
df_train = pd.read_sql("SELECT * FROM newsgroup.train_basic", engine) #train_sparse_repr
df_test = pd.read_sql("SELECT * FROM newsgroup.test_basic", engine) #test_sparse_repr

print(f"Loaded {len(df_train)} train docs and {len(df_test)} test docs")
print(df_train.head())
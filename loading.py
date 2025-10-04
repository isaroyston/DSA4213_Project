import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# Create connection string
conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(conn_str)

# Read train data
df_train = pd.read_sql("SELECT * FROM newsgroups_train", engine)

# Read test data
df_test = pd.read_sql("SELECT * FROM newsgroups_test", engine)

print(f"Loaded {len(df_train)} train docs and {len(df_test)} test docs")
print(df_train.head())

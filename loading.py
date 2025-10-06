import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

def load_data():
    """Load training and test datasets from the Render Postgres DB."""
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
    df_train = pd.read_sql("SELECT * FROM newsgroup.train_basic", engine)
    df_test = pd.read_sql("SELECT * FROM newsgroup.test_basic", engine)

    print(f"Loaded {len(df_train)} train docs and {len(df_test)} test docs")
    return df_train, df_test
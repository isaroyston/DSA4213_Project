import os
import json
import pandas as pd
import random
import numpy as np
import torch
from dotenv import load_dotenv
import google.generativeai as genai
import time
from google.api_core.exceptions import ResourceExhausted

# add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from loading import load_data


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# 20 fine-grained categories
CATEGORIES_20 = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles",
    "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
    "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast",
    "talk.politics.misc", "talk.religion.misc"
]

# 7 broad categories
CATEGORIES_7 = [
    "Computers & Technology", "Science & Engineering", "Recreation",
    "Sports", "Religion", "Politics & Society", "Marketplace"
]

CATEGORY_MAP_INT = {
    "comp.graphics": "Computers & Technology",
    "comp.os.ms-windows.misc": "Computers & Technology",
    "comp.sys.ibm.pc.hardware": "Computers & Technology",
    "comp.sys.mac.hardware": "Computers & Technology",
    "comp.windows.x": "Computers & Technology",
    "sci.crypt": "Science & Engineering",
    "sci.electronics": "Science & Engineering",
    "sci.med": "Science & Engineering",
    "sci.space": "Science & Engineering",
    "rec.autos": "Recreation",
    "rec.motorcycles": "Recreation",
    "rec.sport.baseball": "Sports",
    "rec.sport.hockey": "Sports",
    "alt.atheism": "Religion",
    "soc.religion.christian": "Religion",
    "talk.religion.misc": "Religion",
    "talk.politics.guns": "Politics & Society",
    "talk.politics.mideast": "Politics & Society",
    "talk.politics.misc": "Politics & Society",
    "misc.forsale": "Marketplace"
}


def build_prompt_20(text):
    return f"""
You are a data annotator.
Classify the following text into exactly one of the 20 categories:
{CATEGORIES_20}

Return only one category name from the list, in valid JSON format. 
For example: ["sci.space"]

Text: \"\"\"{text}\"\"\"
"""

def build_prompt_7(text):
    return f"""
You are a data annotator.
Classify the following text into exactly one of the 7 categories:
{CATEGORIES_7}

Return only one category name from the list, in valid JSON format.
For example: ["Science & Engineering"]

Text: \"\"\"{text}\"\"\"
"""


def call_gemini(prompt: str) -> str:
    while True:
        try:
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except ResourceExhausted as e:
            # parse retry delay if provided, else default 30s
            msg = str(e)
            wait_time = 30
            if "Please retry in" in msg:
                try:
                    wait_time = float(msg.split("Please retry in")[1].split("s")[0].strip())
                except:
                    pass
            print(f"Quota exceeded. Sleeping {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print("Gemini error:", e)
            return "other"

def normalize_prediction(pred: str, valid_labels: list) -> str:
    """
    Try to parse the model's JSON output and return a valid label.
    Fallback to 'other' if parsing fails or label not in valid_labels.
    """
    try:
        parsed = json.loads(pred)  # expect ["label"]
        if isinstance(parsed, list) and len(parsed) == 1:
            label = parsed[0].strip()
            if label in valid_labels:
                return label
    except Exception:
        pass
    return "other"


def classify_and_save(df, n=100):
    rows_20, rows_7 = [], []

    for i, row in df.head(n).iterrows():
        text = row["text"]

        # build prompts
        prompt20 = build_prompt_20(text)
        prompt7  = build_prompt_7(text)

        # call Gemini
        pred20 = call_gemini(prompt20)
        pred7  = call_gemini(prompt7)

        # normalize
        pred20_clean = normalize_prediction(pred20, CATEGORIES_20)
        pred7_clean  = normalize_prediction(pred7, CATEGORIES_7)

        # append rows
        rows_20.append({
            "text": text,
            "label": row["label"],
            "predicted_label": pred20_clean
        })
        rows_7.append({
            "text": text,
            "label": CATEGORY_MAP_INT[row["label"]],
            "predicted_label": pred7_clean
        })

        if (i+1) % 10 == 0:
            print(f"Processed {i+1} samples")

    # save
    pd.DataFrame(rows_20).to_csv("20cat_results.csv", index=False)
    pd.DataFrame(rows_7).to_csv("7cat_results.csv", index=False)

    print("Saved 20cat_results.csv and 7cat_results.csv")


if __name__ == "__main__":
    # use your loader
    df_train, df_test = load_data()
    print(f"Loaded train: {len(df_train)} docs, test: {len(df_test)} docs")


    classify_and_save(df_train, n=50)

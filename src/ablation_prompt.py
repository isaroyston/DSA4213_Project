import os
import json
import pandas as pd
from typing import List, Literal
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv


load_dotenv(dotenv_path="../.env")
api_key = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY:", api_key[:6], "..." if api_key else "MISSING")


class TopicLabel(BaseModel):
    tuned_topic_name: Literal[
        "Computers & Technology",
        "Science & Engineering",
        "Recreation (Vehicles & Hobbies)",
        "Sports",
        "Religion",
        "Politics & Society",
        "Marketplace / Miscellaneous"
    ] = Field(
        ...,
        description="One of the predefined Title Case topic labels."
    )
    short_explanation: str = Field(..., description="1–2 sentence summary of the cluster theme")



def _as_list(x) -> List[str]:
    """Turn a cell into a list of strings (handles list/JSON/CSV/plain)."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(t) for t in v]
    except Exception:
        pass
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    if "|" in s:
        return [t.strip() for t in s.split("|") if t.strip()]
    return [s]



PROMPT_VARIATIONS = {
    "baseline": lambda keywords, rep_docs: f"""
You are a News Analytics assistant analyzing article clusters.
Produce a concise, meaningful topic label and a short explanation.
Guidelines:
- Only cluster topics into these categories: "Computers & Technology", "Science & Engineering", "Recreation (Vehicles & Hobbies)", "Sports", "Religion", "Politics & Society", "Marketplace / Miscellaneous"
- Avoid generic or vague labels; be as specific as possible.
- If you are unsure, you are allowed to cluster it as Others.

Return ONLY valid JSON with keys:
{{
  "tuned_topic_name": "string",
  "short_explanation": "string"
}}

keywords = {keywords[:5]}
representative_feedback = {[d[:50] for d in rep_docs[:8]]}
""".strip(),

    "no_others": lambda keywords, rep_docs: f"""
You are a News Analytics assistant analyzing article clusters.
Produce a concise, meaningful topic label and a short explanation.
Guidelines:
- Only use: "Computers & Technology", "Science & Engineering", "Recreation (Vehicles & Hobbies)", "Sports", "Religion", "Politics & Society", "Marketplace / Miscellaneous"
- You MUST choose one of the 7 categories. Do not use Others.

Return ONLY valid JSON with keys:
{{
  "tuned_topic_name": "string",
  "short_explanation": "string"
}}

keywords = {keywords[:5]}
representative_feedback = {[d[:50] for d in rep_docs[:8]]}
""".strip(),

    "cot": lambda keywords, rep_docs: f"""
You are a News Analytics assistant analyzing article clusters.

Think step-by-step to classify this cluster:
1. Examine the keywords.
2. Check representative documents.
3. Choose the most appropriate category from:
   "Computers & Technology", "Science & Engineering", "Recreation (Vehicles & Hobbies)", "Sports", "Religion", "Politics & Society", "Marketplace / Miscellaneous"

Return ONLY valid JSON with keys:
{{
  "tuned_topic_name": "string",
  "short_explanation": "string"
}}

keywords = {keywords[:5]}
representative_feedback = {[d[:50] for d in rep_docs[:8]]}
""".strip(),

    "few_shot": lambda keywords, rep_docs: f"""
You are a News Analytics assistant analyzing article clusters.

Here are some examples of correct classifications:

Example 1:
keywords: ["drive", "card", "windows", "scsi", "disk"]
docs: ["hard drive controllers", "windows compatibility issues"]
→ "Computers & Technology" - Cluster about computer hardware.

Example 2:
keywords: ["team", "game", "season", "hockey", "play"]
docs: ["tampa bay vs philadelphia game", "team standings"]
→ "Sports" - Cluster about hockey games.

Example 3:
keywords: ["space", "nasa", "orbit", "satellite", "launch"]
docs: ["nasa mission update", "satellite in orbit"]
→ "Science & Engineering" - Cluster about space exploration.

Now classify this cluster into one of:
"Computers & Technology", "Science & Engineering", "Recreation (Vehicles & Hobbies)", "Sports", "Religion", "Politics & Society", "Marketplace / Miscellaneous"

Return ONLY valid JSON with keys:
{{
  "tuned_topic_name": "string",
  "short_explanation": "string"
}}

keywords = {keywords[:5]}
representative_feedback = {[d[:50] for d in rep_docs[:8]]}
""".strip()
}



def label_topics_from_df(
    df: pd.DataFrame,
    api_key: str,
    prompt_name: str = "baseline",
    model_name: str = 'gemini-2.5-flash',
    topic_col: str = "Topic",
    words_col: str = "Representation",
    reps_col: str = "Representative_Docs",
) -> pd.DataFrame:
    """
    Label topics using one of the predefined prompts.
    """
    if -1 in df[topic_col].values:
        df = df[df[topic_col] != -1].copy()

    rows = []
    for _, r in df.iterrows():
        topic_id = int(r[topic_col])
        keywords = _as_list(r[words_col])
        rep_docs = _as_list(r[reps_col])

        # Pick prompt variation
        prompt_fn = PROMPT_VARIATIONS[prompt_name]
        prompt = prompt_fn(keywords, rep_docs)

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": TopicLabel,
            },
        )

        generated_text = response.candidates[0].content.parts[0].text
        data = json.loads(generated_text)
        topic_label = TopicLabel.model_validate(data)

        rows.append({
            "Topic": topic_id,
            "tuned_topic_name": topic_label.tuned_topic_name,
            "short_explanation": topic_label.short_explanation
        })

    return pd.DataFrame(rows)

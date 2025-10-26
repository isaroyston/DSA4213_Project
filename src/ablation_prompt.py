import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Literal
from google import genai
from pydantic import BaseModel, Field, ValidationError

# Evaluation function will be passed in as parameter


from typing import Literal

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
    """Robustly turn a cell into a list of strings (handles list/JSON/CSV/plain)."""
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
Given that we want to classify the clusters accurately, into the most relevant and specific topics possible, these are some guidelines to follow:
- Only cluster topics into these categories: "Computers & Technology", "Science & Engineering", "Recreation (Vehicles & Hobbies)", "Sports", "Religion", "Politics & Society", "Marketplace / Miscellaneous"
- Avoid generic or vague labels; be as specific as possible within the allowed categories.
- Ensure the topic label reflects the main theme of the keywords and representative documents.
- If you are unsure, you are allowed to cluster it as Others

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
Given that we want to classify the clusters accurately, into the most relevant and specific topics possible, these are some guidelines to follow:
- Only cluster topics into these categories: "Computers & Technology", "Science & Engineering", "Recreation (Vehicles & Hobbies)", "Sports", "Religion", "Politics & Society", "Marketplace / Miscellaneous"
- Avoid generic or vague labels; be as specific as possible within the allowed categories.
- Ensure the topic label reflects the main theme of the keywords and representative documents.
- You MUST choose one of the 7 categories above. Do not use "Others" or any other category.

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
1. First, examine the keywords and identify the main theme
2. Then, check the representative documents to confirm your understanding
3. Finally, choose the most appropriate category from: "Computers & Technology", "Science & Engineering", "Recreation (Vehicles & Hobbies)", "Sports", "Religion", "Politics & Society", "Marketplace / Miscellaneous"

Guidelines:
- Be as specific as possible within the allowed categories
- Ensure the topic label reflects the main theme
- You MUST choose one of the 7 categories above

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
docs: ["talking about hard drive controllers", "windows compatibility issues"]
→ "Computers & Technology" - This cluster discusses computer hardware like disk drives and SCSI cards.

Example 2:
keywords: ["team", "game", "season", "hockey", "play"]
docs: ["tampa bay vs philadelphia game", "discussing team standings"]
→ "Sports" - This cluster focuses on professional sports, specifically hockey games and team performance.

Example 3:
keywords: ["space", "nasa", "orbit", "satellite", "launch"]
docs: ["nasa mission update", "satellite in orbit around earth"]
→ "Science & Engineering" - This cluster covers space exploration and satellite technology.

Example 4:
keywords: ["god", "jesus", "bible", "faith", "christian"]
docs: ["biblical interpretation discussion", "religious beliefs debate"]
→ "Religion" - This cluster discusses religious topics and faith-based themes.

Now classify this cluster:
Categories: "Computers & Technology", "Science & Engineering", "Recreation (Vehicles & Hobbies)", "Sports", "Religion", "Politics & Society", "Marketplace / Miscellaneous"

You MUST choose one of the 7 categories above.

Return ONLY valid JSON with keys:
{{
  "tuned_topic_name": "string",
  "short_explanation": "string"
}}

keywords = {keywords[:5]}
representative_feedback = {[d[:50] for d in rep_docs[:8]]}
""".strip(),
}

CATEGORY_NAME_TO_INT = {
    "Computers & Technology": 1,
    "Science & Engineering": 2,
    "Recreation (Vehicles & Hobbies)": 3,
    "Sports": 4,
    "Religion": 5,
    "Politics & Society": 6,
    "Marketplace / Miscellaneous": 7
}

def label_topics_with_prompt(
    df: pd.DataFrame,
    prompt_fn,
    api_key: str,
    model_name: str = 'gemini-2.5-flash',
    topic_col: str = "Topic",
    words_col: str = "Representation",
    reps_col: str = "Representative_Docs",
) -> pd.DataFrame:
    """Label topics using a specific prompt function."""
    if -1 in df[topic_col].values:
        df = df[df[topic_col] != -1].copy()

    rows = []
    for _, r in df.iterrows():
        topic_id = int(r[topic_col])
        keywords = _as_list(r[words_col])
        rep_docs = _as_list(r[reps_col])

        prompt = prompt_fn(keywords, rep_docs)

        try:
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
        except Exception as e:
            print(f"Error labeling topic {topic_id}: {e}")
            rows.append({
                "Topic": topic_id,
                "tuned_topic_name": "Marketplace / Miscellaneous",
                "short_explanation": "Error during classification"
            })

    return pd.DataFrame(rows)

def run_ablation_study(
    topic_info: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    topic_model,
    api_key: str,
    eval_function,
    output_dir: str = "ablation_results",
    prompts_to_test: List[str] = None,
    wait_time_sec: int = 60,
    num_batches: int = 2,
    id_col: str = "Doc_ID"  # configurable identifier
):
    """
    Run ablation study across different prompts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if prompts_to_test is None:
        prompts_to_test = list(PROMPT_VARIATIONS.keys())
    
    results = []
    
    # Ensure train/test have Topic column by merging with topic_info
    if "Topic" not in train_df.columns:
        train_df = train_df.merge(topic_info[[id_col, "Topic"]], on=id_col, how="left")
    if "Topic" not in test_df.columns:
        test_df = test_df.merge(topic_info[[id_col, "Topic"]], on=id_col, how="left")
    
    # Estimate time
    topics_per_batch = len(topic_info[topic_info["Topic"] != -1]) // num_batches
    mins_per_batch = topics_per_batch * 3 / 60  # 3 mins per topic
    total_mins = (mins_per_batch * num_batches + (num_batches-1) * wait_time_sec/60) * len(prompts_to_test)
    print(f"Estimated runtime: ~{total_mins:.1f} minutes ({total_mins/60:.1f} hours)")
    print(f"{topics_per_batch} topics/batch × {num_batches} batches × {len(prompts_to_test)} prompts\n")
    
    for prompt_name in prompts_to_test:
        print(f"Testing prompt: {prompt_name}")
        
        # Check if we already have results for this prompt
        result_file = Path(output_dir) / f"{prompt_name}_results.json"
        if result_file.exists():
            print(f"Results already exist for {prompt_name}. Skipping...")
            with open(result_file, 'r') as f:
                results.append(json.load(f))
            continue
        
        prompt_fn = PROMPT_VARIATIONS[prompt_name]
        
        # Label topics using this prompt (with batching)
        print(f"Labeling topics with '{prompt_name}' prompt...")
        batches = np.array_split(topic_info, num_batches)
        labels_list = []
        
        for i, batch in enumerate(batches, start=1):
            print(f"  Batch {i}/{num_batches} ({len(batch)} topics)")
            start_time = time.time()
            
            try:
                labels_chunk = label_topics_with_prompt(
                    batch, 
                    prompt_fn, 
                    api_key=api_key
                )
                labels_list.append(labels_chunk)
                
                elapsed = time.time() - start_time
                print(f"  Batch {i} completed in {elapsed/60:.1f} mins")
                
                if i < num_batches:
                    print(f"  Waiting {wait_time_sec}s before next batch...")
                    time.sleep(wait_time_sec)
            except Exception as e:
                print(f"  Error in batch {i}: {e}")
                continue
        
        labels_df = pd.concat(labels_list, ignore_index=True)
        
        # Save labeled topics
        labels_file = Path(output_dir) / f"{prompt_name}_labels.csv"
        labels_df.to_csv(labels_file, index=False)
        print(f"Labels saved to {labels_file}")
        
        # Evaluate on train set
        print("Evaluating on TRAIN set...")
        train_eval = train_df.copy()
        train_eval = train_eval.merge(labels_df[["Topic", "tuned_topic_name"]], on="Topic", how="left")
        train_eval["predicted_topic_id"] = train_eval["tuned_topic_name"].map(CATEGORY_NAME_TO_INT)
        train_eval["predicted_topic_id"] = train_eval["predicted_topic_id"].fillna(-1).astype(int)
        
        train_filtered = train_eval[(train_eval["Topic"] != -1) & (train_eval["predicted_topic_id"] != -1)].copy()
        
        train_metrics = eval_function(
            train_filtered["predicted_topic_id"].astype('int64'),
            train_filtered["label"].astype('int64'),
            plot_confusion=False
        )
        
        # Evaluate on test set
        print("Evaluating on TEST set...")
        test_eval = test_df.copy()
        test_eval = test_eval.merge(labels_df[["Topic", "tuned_topic_name"]], on="Topic", how="left")
        test_eval["predicted_topic_id"] = test_eval["tuned_topic_name"].map(CATEGORY_NAME_TO_INT)
        test_eval["predicted_topic_id"] = test_eval["predicted_topic_id"].fillna(-1).astype(int)
        
        test_filtered = test_eval[(test_eval["Topic"] != -1) & (test_eval["predicted_topic_id"] != -1)].copy()
        
        test_metrics = eval_function(
            test_filtered["predicted_topic_id"].astype('int64'),
            test_filtered["label"].astype('int64'),
            plot_confusion=False
        )
        
        # Store results
        result = {
            "prompt_name": prompt_name,
            "train": {
                "total_samples": len(train_df),
                "filtered_samples": len(train_filtered),
                "outliers_removed": len(train_df) - len(train_filtered),
                "macro_f1": train_metrics["macro_f1"],
                "weighted_f1": train_metrics["weighted_f1"],
                "per_class_f1": train_metrics["per_class_f1"]
            },
            "test": {
                "total_samples": len(test_df),
                "filtered_samples": len(test_filtered),
                "outliers_removed": len(test_df) - len(test_filtered),
                "macro_f1": test_metrics["macro_f1"],
                "weighted_f1": test_metrics["weighted_f1"],
                "per_class_f1": test_metrics["per_class_f1"]
            }
        }
        
        results.append(result)
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {result_file}")
        
        print(f"{prompt_name} complete.")
        print(f"Train Macro F1: {train_metrics['macro_f1']:.4f}")
        print(f"Test Macro F1:  {test_metrics['macro_f1']:.4f}")
    
    # Save summary
    summary_df = pd.DataFrame([
        {
            "prompt": r["prompt_name"],
            "train_macro_f1": r["train"]["macro_f1"],
            "train_weighted_f1": r["train"]["weighted_f1"],
            "test_macro_f1": r["test"]["macro_f1"],
            "test_weighted_f1": r["test"]["weighted_f1"],
            "train_samples": r["train"]["filtered_samples"],
            "test_samples": r["test"]["filtered_samples"]
        }
        for r in results
    ])
    
    summary_file = Path(output_dir) / "ablation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print("Ablation study complete.")
    print(summary_df.to_string(index=False))
    print(f"Summary saved to {summary_file}")
    
    return results, summary_df

# DSA4213 Project

*A framework for generating, fine-tuning, and evaluating sentence embeddings using multiple models (SimCSE, Triplet, Cosine) with analysis tools and benchmarking pipelines.*

---

## Project Structure

```
root
├── embeddings_output/         # path to save generated embedding vectors + projections
│
├── notebooks/                 # Jupyter notebooks for experiments & analysis
├── src/
│   ├── Evaluations/           # eval functions to be imported
│   │   ├── category_alignment.py
│   │   └── davies_bouldin.py
│   │
│   ├── models/
│   │   ├── direct_prompting.py
│   │   ├── finetuned_embeddings.py
│   │   ├── general_embeddings.py
│   │   ├── own_embeddings.py
│   │   └── ablation_prompt.py
│   │
│   ├── cot.py                 # Chain-of-thought prompting ablation
│   ├── data_prep.py           # Dataset loading & preprocessing 
│   ├── fewshot.py             # Few-shot prompting ablation
│   ├── gemini_utils.py        # Gemini-specific helper wrappers
│   └── no_others.py           # "No others category" prompting ablation
│
├── data_cleaning.py           # Script for cleaning raw data
├── loading.py                 # Script for loading datasets / models
│
├── README.md
└── requirements.txt
```

---

##  Overview

This repository contains a full pipeline for embedding generation, fine-tuning, evaluation, and analysis across multiple representation-learning approaches:
* Own embeddings
* SimCSE
* Triplet embedding models
* Cosine similarity–optimized embeddings
* General embeddings
* Ablation: prompts to Gemini

The project supports:

* Loading datasets, preprocessing, and cleaning
* Running multiple embedding models (general or fine-tuned)
* Evaluating clustering quality (e.g., Davies-Bouldin Index)
* Comparing category alignment performance
* Visualising embedding spaces (2D projections)

---

##  Installation

```sh
git clone <your-repo-url>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

##  Usage

### 1. Prepare the data

```sh
python data_cleaning.py
python loading.py
```

### 2. Run embedding models

Inside `src/models/`:

* `general_embeddings.py` → base embeddings
* `finetuned_embeddings.py` → fine-tuned model inference
* `own_embeddings.py` → your custom models
* `direct_prompting.py` → LLM prompting-based embeddings

---

##  Outputs

All computed outputs are saved in:

* `embeddings_output/` → `.npz`, `.csv`, projections
* `finetuned_models/` → your trained embedding models
* `results/` → DB index, alignment metrics, logs
* `plots/` → visual embeddings, PCA, t-SNE, etc.

---

##  Notebooks

The `notebooks/` folder includes:
* Category separation analysis
* * DB Index comparisons
* Hyperparaeter tuning
* Prompting experiments (few-shot, CoT)


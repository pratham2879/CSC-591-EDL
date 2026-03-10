# ARAML: Adaptive Retrieval-Augmented Meta-Learning for Cross-Lingual Text Classification

> **Authors:** Mahek Viradiya, Pratham Patel  
> **Date:** February 2026

## Overview

ARAML is a novel framework for cross-lingual text classification targeting low-resource languages. It combines:
- **Task-Adaptive Retrieval** — learns *what* to retrieve via task-specific queries
- **Dynamic Retrieval Budgeting** — adjusts *how many* examples to retrieve based on task difficulty
- **Attention-Weighted Retrieval** — learns *how to weight* retrieved cross-lingual knowledge
- **MAML-based Meta-Learner** — enables fast adaptation with augmented support sets

## Architecture

```
Low-Resource Language Input
        │
   [Text Encoder]          ← Understands task type (mBERT / XLM-R)
        │
[Adaptive Retrieval        ← Decides what & how many examples to retrieve
  Controller (ARC)]
        │
[Cross-Lingual             ← Retrieves from High-Resource Language (HRL) index
 Retrieval Index]
        │
 [Meta-Learner]            ← Fast adaptation (MAML-based)
        │
  [Classification]
```

## Project Structure

```
araml/
├── configs/
│   └── config.yaml            # Hyperparameters and experiment settings
├── data/
│   ├── download_data.py       # Download multilingual datasets
│   └── preprocess.py          # Tokenization and episode generation
├── models/
│   ├── encoder.py             # mBERT / XLM-R text encoder
│   ├── arc.py                 # Adaptive Retrieval Controller
│   ├── retrieval_index.py     # Cross-lingual FAISS retrieval index
│   ├── meta_learner.py        # MAML-based meta-learner
│   └── araml.py               # Full ARAML model
├── utils/
│   ├── episode_sampler.py     # N-way K-shot episode sampler
│   └── metrics.py             # Accuracy, F1, few-shot eval metrics
├── scripts/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── build_index.py         # Build cross-lingual retrieval index
├── notebooks/
│   └── demo.ipynb             # End-to-end demo notebook
├── results/                   # Saved model checkpoints and logs
├── requirements.txt
└── README.md
```

## Setup Instructions (VS Code)

### Prerequisites
- Python 3.9+
- Git
- VS Code with Python extension installed

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/araml.git
cd araml
```

**2. Open in VS Code**
```bash
code .
```

**3. Create a virtual environment**

Open the VS Code terminal (`Ctrl + `` ` ``) and run:
```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Select the Python interpreter in VS Code**
- Press `Ctrl+Shift+P` → "Python: Select Interpreter"
- Choose the interpreter inside `./venv`

---

## Running the Project

### Step 1 — Download & preprocess data
```bash
python data/download_data.py
python data/preprocess.py
```

### Step 2 — Build the cross-lingual retrieval index
```bash
python scripts/build_index.py --config configs/config.yaml
```

### Step 3 — Train ARAML
```bash
python scripts/train.py --config configs/config.yaml
```

### Step 4 — Evaluate
```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint results/best_model.pt
```

### (Optional) Run the demo notebook
In VS Code, open `notebooks/demo.ipynb` and run all cells.

---

## Uploading to GitHub

**1. Initialize Git (if not cloned)**
```bash
git init
git add .
git commit -m "Initial commit: ARAML project"
```

**2. Create a new repository on GitHub**
- Go to [github.com](https://github.com) → "New repository"
- Name it `araml`, set visibility, do **not** initialize with README

**3. Push to GitHub**
```bash
git remote add origin https://github.com/<your-username>/araml.git
git branch -M main
git push -u origin main
```

**4. For future changes**
```bash
git add .
git commit -m "Your message here"
git push
```

---

## Datasets

| Dataset | Languages | Task |
|---|---|---|
| Amazon Multilingual Reviews | EN, DE, FR, ZH, JA, ES | Sentiment Classification |
| HuffPost News | EN (source) | Topic Classification |

Download via `data/download_data.py` (uses Hugging Face `datasets`).

## Baselines

| Method | Description |
|---|---|
| Fine-tuning | XLM-R fine-tuned on source language |
| MAML | Model-Agnostic Meta-Learning |
| Reptile | Simplified meta-learning |
| RAML | Retrieval-Augmented Meta-Learning (static) |
| **ARAML (ours)** | **Adaptive retrieval + meta-learning** |

## Expected Results

- Improved 1-shot, 5-shot, 10-shot accuracy over all baselines
- Better generalization to unseen low-resource languages
- Reduced overfitting via attention-weighted retrieval

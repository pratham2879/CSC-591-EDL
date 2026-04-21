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

## Setup Instructions

### Prerequisites
- **Python 3.12** (required — packages do not support 3.10 or earlier)
- **NVIDIA GPU** with CUDA 12.x recommended (CPU training is very slow)
- Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/araml.git
cd araml
```

### Step 2 — Create a virtual environment

**Windows (Command Prompt):**
```cmd
py -3.12 -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3.12 -m venv venv
source venv/bin/activate
```

> You must activate the venv every time you open a new terminal before running any scripts.
> You will see `(venv)` at the start of your prompt when it is active.

### Step 3 — Install PyTorch with CUDA

**Check your CUDA version first:**
```cmd
nvidia-smi
```

Then install the matching PyTorch build:
```cmd
# CUDA 12.8 (most common with recent drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (slow, not recommended for training)
pip install torch torchvision
```

### Step 4 — Install remaining dependencies
```cmd
pip install -r requirements.txt
```

### Step 5 — Verify GPU is detected
```cmd
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output: `CUDA: True  GPU: <your GPU name>`

### Step 6 — Select interpreter in VS Code
- Press `Ctrl+Shift+P` → "Python: Select Interpreter"
- Choose the interpreter inside `./venv`

---

## Full Pipeline (run in order)

All commands must be run from inside the `araml/` directory with the venv active.

**Windows (Command Prompt):**
```cmd
cd araml
venv\Scripts\activate
```

**Mac/Linux:**
```bash
cd araml
source venv/bin/activate
```

---

### Step 1 — Download dataset (~370 MB, requires internet)
```cmd
set PYTHONPATH=..  && python data/download_data.py       # Windows
PYTHONPATH=. python data/download_data.py                # Mac/Linux
```

Downloads `mteb/amazon_reviews_multi` for 6 languages (en, de, es, fr, ja, zh).
Output: `data/raw/amazon_{lang}/`

---

### Step 2 — Preprocess
```cmd
set PYTHONPATH=.. && python data/preprocess.py           # Windows
PYTHONPATH=. python data/preprocess.py                   # Mac/Linux
```

- Converts 5-star ratings to binary (negative/positive), drops neutral
- Caps low-resource (ja, zh) training pool at 2000 examples
Output: `data/processed/amazon_{lang}.json`, `data/lowresource_pool_{ja,zh}.json`

---

### Step 3 — Build retrieval index
```cmd
set PYTHONPATH=.. && python scripts/build_index.py       # Windows
PYTHONPATH=. python scripts/build_index.py               # Mac/Linux
```

Encodes high-resource (en, de, es, fr) training examples with XLM-R and stores them in a FAISS index.
Output: `results/retrieval_index.faiss`, `results/retrieval_index_meta.npy`

---

### Step 4 — Train
```cmd
set PYTHONPATH=.. && python scripts/train.py --epochs 30 --episodes_per_epoch 500    # Windows
PYTHONPATH=. python scripts/train.py --epochs 30 --episodes_per_epoch 500            # Mac/Linux
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 20 | Number of training epochs |
| `--episodes_per_epoch` | 1000 | Episodes sampled per epoch |
| `--outer_lr` | 0.0003 | Outer (meta) learning rate |
| `--max_grad_norm` | 1.0 | Gradient clipping threshold |

Best checkpoint saved to `results/best_model.pt` automatically.

---

### Step 5 — Evaluate
```cmd
set PYTHONPATH=.. && python scripts/evaluate.py --checkpoint results/best_model.pt --n_episodes 600   # Windows
PYTHONPATH=. python scripts/evaluate.py --checkpoint results/best_model.pt --n_episodes 600           # Mac/Linux
```

Runs 600 few-shot episodes on the held-out test split and reports accuracy ± 95% CI, precision, recall, and F1 per language.

---

## Results

| Checkpoint | Training F1 | Test F1 (5-shot) |
|---|---|---|
| Epoch 24 (500-example pool) | 0.9719 | 0.8729 |

- ja (Japanese): 0.8787
- zh (Chinese): 0.8679

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | venv is not activated — run `venv\Scripts\activate` first |
| `CUDA: False` | Installed CPU-only PyTorch — reinstall with the CUDA index URL above |
| `faiss-gpu not found` | faiss-gpu is Linux-only — use `pip install faiss-cpu` on Windows |
| `numpy==2.4.3 not found` | Python version too old — requires Python 3.12 |
| `could not open retrieval_index.faiss` | Run Step 3 (build_index.py) before training |

---

## Using Pre-Trained Artifacts on Any Device

The model checkpoint (~1.06 GB), FAISS index (~123 MB), and index metadata (~8 MB) are too large for regular git. They are stored on Hugging Face Hub and can be pushed/pulled with two scripts.

### First time — push from the device that has the trained files

**Step 1: Log in to Hugging Face (once per device)**
```cmd
huggingface-cli login
```
Enter your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Step 2: Upload artifacts**
```cmd
set PYTHONPATH=.. && python scripts/push_artifacts.py --repo your-hf-username/araml-artifacts
```

---

### Any other device — pull and run immediately

**Step 1: Set up the environment (see Setup Instructions above)**

**Step 2: Download model + index**

Windows:
```cmd
cd araml
venv\Scripts\activate
set PYTHONPATH=.. && venv\Scripts\python scripts/pull_artifacts.py --repo prathamP6969/araml-artifacts --token hf_YOUR_TOKEN
```

Mac/Linux:
```bash
cd araml
source venv/bin/activate
PYTHONPATH=. python scripts/pull_artifacts.py --repo prathamP6969/araml-artifacts --token hf_YOUR_TOKEN
```

This downloads three files into `results/`:
- `best_model.pt` — trained model weights (~1.06 GB)
- `retrieval_index.faiss` — cross-lingual FAISS index (~123 MB)
- `retrieval_index_meta.npy` — index metadata (~8 MB)

Already downloaded files are skipped automatically.

**Step 3: Evaluate immediately — no retraining needed**

Windows:
```cmd
set PYTHONPATH=.. && venv\Scripts\python scripts/evaluate.py --checkpoint results/best_model.pt --n_episodes 600
```

Mac/Linux:
```bash
PYTHONPATH=. python scripts/evaluate.py --checkpoint results/best_model.pt --n_episodes 600
```

Expected result:
```
Overall  | Acc: 0.8919 | F1: 0.8919
[ja]     | Acc: 0.9070 | F1: 0.9070
[zh]     | Acc: 0.8780 | F1: 0.8779
```

> **Note:** If you retrain the model on a new device, rebuild the FAISS index before evaluating:
> ```cmd
> set PYTHONPATH=.. && venv\Scripts\python scripts/build_index.py
> set PYTHONPATH=.. && venv\Scripts\python scripts/evaluate.py
> ```

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

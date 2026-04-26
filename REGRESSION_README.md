# ARAML Regression Branch — Comprehensive Guide

> **Date:** April 2026  
> **Task:** Few-Shot Sentiment Regression for Cross-Lingual Text Classification

## Executive Summary

This document comprehensively describes the **regression branch** of ARAML, which adapts the original cross-lingual classification framework to perform **few-shot sentiment regression** — predicting continuous sentiment scores in the [0, 1] range instead of discrete class labels.

### Key Changes from Classification to Regression

| Aspect | Classification | Regression |
|--------|----------------|-----------|
| **Output Range** | {0, 1, ...} discrete labels | [0, 1] continuous scores |
| **Loss Function** | Cross-entropy loss | Smooth L1 (Huber) loss |
| **Labels** | Star ratings 1-5 → one-hot | Star ratings 1-5 → normalized [0, 1] |
| **Metrics** | Accuracy, F1, Precision, Recall | MAE, RMSE, Correlation |
| **Meta-Learner** | Linear classifier | Linear regressor (continuous output) |
| **Task Episode** | N-way K-shot classification | K-shot regression with continuous targets |

---

## Architecture Overview

### ARAML Regression Pipeline

```
Input: Low-Resource Language Text (e.g., "この商品はすごく良いです。")
   ↓
[TextEncoder: XLM-R]  ← Encodes text to 768-dim embeddings
   ↓
[Adaptive Retrieval Controller (ARC)]  ← Learns what & how many examples to retrieve
   ├─ Query Generator      → learns task-specific retrieval queries
   ├─ Budget Predictor     → predicts retrieval budget (0-max_k examples)
   └─ Attention Weighter   → learns importance weights for retrieved examples
   ↓
[Cross-Lingual Retrieval Index (FAISS)]  ← 100k+ examples from high-resource languages
   ├─ Stored: (text, embedding, label, language)
   └─ Retrieval: top-k semantically similar examples
   ↓
[Meta-Learner: MAML-based Regressor]  ← Fast adaptation in 5 inner steps
   ├─ Inner Loop (Adaptation)   → adapts to support set distribution
   ├─ Outer Loop (Meta-Training) → optimizes for fast adaptation
   └─ Output: continuous score in [0, 1]
   ↓
Prediction: 0.75 (≈ 4 stars out of 5)
```

### Component Details

#### 1. TextEncoder (models/encoder.py)
- **Model:** XLM-R (100+ languages)
- **Output:** 768-dimensional embeddings
- **Training:** Partially unfrozen (layers 9-11 + pooler)
  - Why partial freeze? Prevents destabilization of lower-level representations while allowing gradient flow for ARC component

#### 2. Adaptive Retrieval Controller (models/arc.py)
- **Purpose:** Learn *what* to retrieve, *how many* to retrieve, and *how to weight* them
- **Components:**
  - Query Generator: $ q_t = W_q h_{support} + b_q $
  - Budget Predictor: $ \alpha_t = \sigma(W_b h_{support} + b_b) $
  - Attention Weighter: Computes $ \beta_i = \text{softmax}(q_t \cdot r_i) $
- **Benefit:** Reduces noise from irrelevant retrieval by learning task-adaptive strategies

#### 3. Cross-Lingual Retrieval Index (models/retrieval_index.py)
- **Storage:** FAISS (Facebook AI Similarity Search)
- **Data:** High-resource languages (en, de, es, fr) — 100k+ examples
- **Index Type:** L2 (Euclidean distance) or Cosine similarity
- **Retrieval:** O(log N) approximate nearest neighbors

#### 4. Meta-Learner (models/meta_learner.py)
- **Algorithm:** Model-Agnostic Meta-Learning (MAML) for regression
- **Regression Head:** Linear layer (768 → 1)
- **Loss Function:** Smooth L1 (Huber) loss — robust to outliers
  - Formula: $ L_{\text{Huber}}(y, \hat{y}) = \begin{cases} 0.5(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq 1 \\ |y - \hat{y}| - 0.5 & \text{otherwise} \end{cases} $

---

## Data Format and Preprocessing

### Label Normalization

Star ratings (1-5 stars) are normalized to [0, 1]:

```
1 star   → 0.00 (ひどい / 很差)
2 stars  → 0.25 (期待以下 / 没达到预期)
3 stars  → 0.50 (普通 / 一般般)
4 stars  → 0.75 (満足 / 满意)
5 stars  → 1.00 (最高 / 非常好)
```

### Episode Structure

Each training/evaluation episode contains:
```json
{
  "language": "ja",
  "support_set": [
    {"text": "text1", "label": 0.25},
    {"text": "text2", "label": 0.75},
    {"text": "text3", "label": 0.50}
  ],
  "query_set": [
    {"text": "query1", "label": 0.60},
    {"text": "query2", "label": 0.40}
  ]
}
```

- **Support Set Size:** k_shot (default: 5 examples)
- **Query Set Size:** query_size (default: 10 examples)
- **Languages:** Only low-resource (ja, zh)

---

## Training Pipeline

### Phase 1: Initialization
```bash
cd araml
PYTHONPATH=. python scripts/train.py --epochs 20 --episodes_per_epoch 1000
```

### Training Loop Structure

```python
for epoch in range(args.epochs):
    # Initialize accumulators
    train_losses, val_losses = [], []
    
    for episode_idx in range(args.episodes_per_epoch):
        # 1. Sample random episode (ja or zh)
        episode = sampler.sample_episode()
        
        # 2. Meta-training step (inner + outer loop)
        loss, mae, rmse = meta_train_step(
            encoder, arc, meta_learner, retrieval_index,
            episode, config, device, optimizer
        )
        train_losses.append(loss)
    
    # 3. VALIDATION PHASE (new in regression branch!)
    if val_datasets:
        val_result = evaluate_components(
            encoder, arc, meta_learner, retrieval_index,
            config, device, val_datasets, 
            n_episodes=args.val_episodes
        )
        val_mae = val_result["mae_mean"]
        val_losses.append(val_mae)
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f"{args.save_dir}/best_model.pt")
    
    # 4. Report epoch statistics
    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_mae={val_mae:.4f}")
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `outer_lr` | 0.0003 | Meta-learning rate (FIX 3) |
| `inner_lr` | 0.01 | Inner-loop adaptation rate |
| `inner_steps` | 5 | Gradient steps to adapt on support set |
| `k_shot` | 5 | Support set examples |
| `query_size` | 10 | Query set examples |
| `max_retrieved` | 20 | Maximum examples to retrieve |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `weight_decay` | 1e-4 | L2 regularization |

---

## Validation & Early Stopping

### Validation Metrics

During training, the model is evaluated every epoch on the **validation split** of ja/zh:

```json
{
  "mae_mean": 0.1234,
  "mae_std": 0.0456,
  "rmse_mean": 0.1567,
  "rmse_std": 0.0512,
  "per_language": {
    "ja": {"mae": 0.1150, "rmse": 0.1480, "correlation": 0.92},
    "zh": {"mae": 0.1318, "rmse": 0.1654, "correlation": 0.89}
  },
  "n_episodes": 100
}
```

### Early Stopping Strategy

The **best model** (lowest validation MAE) is automatically saved:
- File: `results/best_model.pt`
- Tracked by: `best_val_mae` variable
- Patience: No explicit patience; trains for fixed number of epochs

---

## Inference & Testing

### Interactive CLI Testing

```bash
PYTHONPATH=. python scripts/regression_cli.py
```

**Features:**
- Load pre-trained model and retrieval index
- Interactive query input loop
- Automatic prediction with:
  - Normalized score [0, 1]
  - Star rating equivalent (1-5)
  - Support set fitting error
  - Retrieval statistics (k, budget ratio, attention weights)

### Example Usage

```
Support set:
  [1] label=0.00 (~1.0 stars)  ひどい品質で、すぐ壊れました。
  [2] label=0.25 (~1.2 stars)  期待以下で、あまり満足できません。
  [3] label=0.50 (~3.0 stars)  普通です。悪くも良くもありません。
  [4] label=0.75 (~4.0 stars)  使いやすくて満足しています。
  [5] label=1.00 (~5.0 stars)  最高です。買って本当に良かったです。

Query> この商品はかなり良いです。

Prediction
  normalized score : 0.7234
  approx stars     : 3.89 / 5
  support MAE      : 0.0512
  retrieval budget : ratio=0.8500  k=17
```

---

## Retraining & Fine-Tuning

### Full Retraining

```bash
# From scratch
PYTHONPATH=. python scripts/train.py --epochs 20 --seed 42

# Resume from checkpoint (modify train.py to support this)
PYTHONPATH=. python scripts/train.py --epochs 30 --checkpoint results/best_model.pt
```

### Fine-Tuning on Custom Domain

```bash
# 1. Prepare custom support examples (JSON format)
cat > custom_support.json << 'EOF'
[
  {"text": "low quality product", "label": 0.1},
  {"text": "acceptable product", "label": 0.5},
  {"text": "excellent product", "label": 0.9}
]
EOF

# 2. Use with CLI
PYTHONPATH=. python scripts/regression_cli.py --support-file custom_support.json
```

---

## Bug Fixes & Improvements (Regression Branch)

### FIX 1: Second-Order Gradients (meta_learner.py)
```python
inner_loss.backward(create_graph=True)  # Enable second-order gradient computation
```
**Why:** MAML requires computing gradients of gradients (Hessian) for outer-loop updates.

### FIX 2: Partial Encoder Unfreeze (train.py)
```python
def setup_encoder_freezing(encoder, unfreeze_from_layer=9):
    # Freeze layers 0-8, unfreeze layers 9-11 + pooler
```
**Why:** Full freeze prevents ARC gradients from flowing to embeddings. Partial unfreeze balances stability with gradient flow.

### FIX 3: Outer Loop Optimization (train.py + meta_learner.py)
- **Optimizer:** AdamW (not SGD)
- **Learning Rate:** 0.0003 (not 0.001)
- **Gradient Clipping:** 1.0 (prevents exploding gradients)
**Why:** Regression targets are continuous; outer loop needs finer control.

### FIX 5: Gradient Diagnostics (train.py)
```python
diagnose_gradient_flow(encoder, arc, meta_learner, config, device)
```
Runs before epoch 1 to catch gradient issues early.

---

## Performance Metrics (Expected Results)

### On Validation Set (ja + zh)

| Metric | Expected | Notes |
|--------|----------|-------|
| MAE | 0.10–0.15 | Mean absolute error in [0, 1] scale |
| RMSE | 0.12–0.18 | Root mean squared error |
| Correlation | 0.85–0.95 | Pearson correlation with true labels |
| Per-Language (ja) | MAE ≤ 0.12 | Japanese language performance |
| Per-Language (zh) | MAE ≤ 0.14 | Chinese language performance |

### Inference Speed
- Per-query latency: ~50–100 ms (GPU), ~500 ms (CPU)
- Batch processing: ~5–10 ms per query (amortized)

---

## File Manifest — Regression Branch

| File | Purpose | Status |
|------|---------|--------|
| `scripts/train.py` | Main training loop with validation | ✅ Enhanced with validation |
| `scripts/evaluate.py` | Standalone evaluation script | ✅ Supports regression metrics |
| `scripts/regression_cli.py` | Interactive inference CLI | ✅ Interactive testing |
| `models/araml.py` | Full ARAML model | ✅ Regression-ready |
| `models/meta_learner.py` | MAML-based meta-learner | ✅ Regression head (linear) |
| `models/arc.py` | Adaptive Retrieval Controller | ✅ Task-adaptive retrieval |
| `models/encoder.py` | Text encoder (XLM-R) | ✅ Partially unfrozen |
| `utils/episode_sampler.py` | Episode sampler (continuous labels) | ✅ Supports regression |
| `configs/config.yaml` | Hyperparameter configuration | ✅ Regression defaults |

---

## Troubleshooting

### Issue: "Validation disabled: no low-resource validation data found"
**Solution:** Run preprocessing script first:
```bash
PYTHONPATH=. python data/preprocess.py
```

### Issue: Out of memory (OOM) during training
**Solution:** Reduce batch size or episode size:
```bash
# Modify config.yaml or pass as CLI argument
--episodes_per_epoch 500
```

### Issue: NaN loss during training
**Solution:** 
1. Check gradient diagnostics output (FIX 5)
2. Reduce learning rate: `--outer_lr 0.0001`
3. Ensure support/query sets contain valid labels

### Issue: Model predictions always near 0.5
**Solution:**
1. Verify retrieval index is loaded: `len(index) > 0`
2. Check support examples span [0, 1] range
3. Ensure inner loop adapts: check `support_mae` in CLI output

---

## References

### Papers
- MAML: Finn et al., 2017 — "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
- XLM-R: Conneau et al., 2019 — "Unsupervised Cross-lingual Representation Learning at Scale"

### External Links
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [PyTorch MAML Implementations](https://github.com/cbfinn/maml)
- [XLM-R Model Card](https://huggingface.co/xlm-roberta-base)

---

## Contact & Support

For issues or questions regarding the regression branch:
1. Check this README's Troubleshooting section
2. Review commit messages in `REGRESSION_CHANGES.md`
3. Inspect training logs in `results/training.log`

**Last Updated:** April 2026

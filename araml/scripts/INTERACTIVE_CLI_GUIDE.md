# Interactive CLI Quick Start Guide

## Installation & Setup

Before using the interactive CLI, ensure you have:
1. Completed data preparation
2. Built the retrieval index
3. Trained the model (or have a pre-trained checkpoint)

```bash
cd araml
source ../venv/bin/activate  # Activate virtual environment
```

---

## Basic Usage

### Launch Interactive Mode

```bash
PYTHONPATH=. python scripts/interactive_cli.py
```

**Output:**
```
Device: cuda
Config loaded from: configs/config.yaml
Checkpoint loaded: results/best_model.pt
Retrieval index loaded: 100234 entries
Language: ja
Support examples: 5

Support Set:
────────────────────────────────────────────────────────────────────────────────
  [1] label=0.00 (1.0★)  ひどい品質で、すぐ壊れました。
  [2] label=0.25 (1.2★)  期待以下で、あまり満足できません。
  [3] label=0.50 (3.0★)  普通です。悪くも良くもありません。
  [4] label=0.75 (4.0★)  使いやすくて満足しています。
  [5] label=1.00 (5.0★)  最高です。買って本当に良かったです。

[Model information displayed...]

Type 'help' for available commands or 'exit' to quit.

>>>
```

---

## Common Commands

### 1. Test Single Query

```bash
>>> test "この商品はかなり良いです。"

Query: この商品はかなり良いです。
════════════════════════════════════════════════════════════════════════════════
Prediction:       0.7234 (3.89★ / 5)
Raw Score:        0.7180
Support MAE:      0.0512
Budget:           17 examples (ratio=0.8500)

Top 5 Retrieved Examples:
────────────────────────────────────────────────────────────────────────────────
  [1] ja label=0.75 (4.0★) attn=0.2134
      使いやすくて満足しています。
  [2] en label=0.80 (4.2★) attn=0.1956
      Very satisfied with this product
  [3] de label=0.72 (3.9★) attn=0.1834
      Sehr zufrieden mit diesem Produkt
  [4] es label=0.78 (4.1★) attn=0.1745
      Muy satisfecho con este producto
  [5] fr label=0.74 (3.96★) attn=0.1531
      Très satisfait de ce produit
```

### 2. Interactive Query Loop

```bash
>>> test

Enter query (or 'done' to finish):
  > 良い商品です。
[Shows prediction for "良い商品です。"]

  > 最悪です。
[Shows prediction for "最悪です。"]

  > done
```

### 3. Test Multiple Queries from File

**Create queries.json:**
```json
[
  "この商品は素晴らしいです。",
  "あまり満足できません。",
  "普通の商品ですね。",
  "非常に気に入りました。",
  "すぐに壊れてしまった。"
]
```

**Run batch inference:**
```bash
>>> batch queries.json

Running batch inference on 5 queries...
100%|████████████| 5/5 [00:00<00:00, 45.23 queries/s]

Batch Results Summary:
  Avg prediction:   0.5234
  Min prediction:   0.1200
  Max prediction:   0.9100
  Avg stars:        3.09★
```

### 4. Fine-Tune Model on Support Set

Fine-tune the model to adapt better to the current support set:

```bash
>>> retrain 5

Fine-tuned for 5 steps:
  Initial loss: 0.245678
  Final loss:   0.067234
  Final MAE:    0.0398

>>> test "商品は良いですね。"
[After fine-tuning, predictions should be more calibrated to the support set]
```

### 5. View Model Information

```bash
>>> info

MODEL INFORMATION
════════════════════════════════════════════════════════════════════════════════
Device:                  cuda
Total parameters:        124,542,387
Trainable parameters:    8,934,521

Encoder:                 xlm-roberta-base
  Hidden dim:            768
ARC Components:
  Max retrieved:         20
  Similarity metric:     cosine
Meta-Learner (MAML):
  Inner LR:              0.01
  Inner steps:           5
  K-shot:                5
  Query set size:        10

Retrieval Index:
  Size:                  100,234 examples
```

### 6. Load Custom Support Examples

**Create support.json:**
```json
[
  {"text": "low quality product", "label": 1},
  {"text": "average product", "label": 3},
  {"text": "excellent product", "label": 5}
]
```

**Load it:**
```bash
>>> support support.json

Support examples loaded from: support.json

Support Set:
────────────────────────────────────────────────────────────────────────────────
  [1] label=0.00 (1.0★)  low quality product
  [2] label=0.50 (3.0★)  average product
  [3] label=1.00 (5.0★)  excellent product
```

### 7. Save and Load Model Checkpoints

**Save current model state:**
```bash
>>> save results/checkpoint_custom_v1.pt

Checkpoint saved: results/checkpoint_custom_v1.pt
```

**Load different checkpoint:**
```bash
>>> load results/checkpoint_custom_v1.pt

Checkpoint loaded: results/checkpoint_custom_v1.pt
```

---

## Advanced Workflows

### Workflow 1: Domain-Specific Fine-Tuning

```bash
# 1. Load custom support examples for your domain
>>> support domain_specific_support.json

# 2. Fine-tune model multiple times
>>> retrain 10

# 3. Test with domain-specific queries
>>> test "your domain specific text"

# 4. Save fine-tuned model
>>> save results/domain_model.pt
```

### Workflow 2: Batch Evaluation with Statistics

```bash
# Create test_data.json with multiple queries
>>> batch test_data.json

# Get detailed statistics from results
# (Can be further processed externally)

# Optionally fine-tune and re-test
>>> retrain 5
>>> batch test_data_v2.json

# Compare results
```

### Workflow 3: Multi-Language Testing

**Test Japanese:**
```bash
PYTHONPATH=. python scripts/interactive_cli.py --language ja
>>> test "日本語のテキスト"
```

**Test Chinese:**
```bash
PYTHONPATH=. python scripts/interactive_cli.py --language zh
>>> test "中文文本"
```

---

## Understanding Output

### Prediction Output Fields

| Field | Meaning | Range | Example |
|-------|---------|-------|---------|
| `Prediction` | Normalized sentiment score | [0, 1] | 0.7234 |
| `Approx stars` | Star rating equivalent | [1, 5] | 3.89 |
| `Raw Score` | Model output before clamping | any | 0.7180 |
| `Support MAE` | Adaptation quality on support set | [0, 1] | 0.0512 |
| `Budget` | How many examples retrieved | [1, max_k] | 17 / 20 |
| `Ratio` | Percentage of max budget used | [0, 1] | 0.8500 |
| `Attention weights` | Importance of retrieved examples | [0, 1] | 0.2134 |

### Interpreting Results

**Good Predictions:**
- Low `Support MAE` (< 0.05) → Model well-adapted to support set
- Multiple retrieved examples with attention weight > 0.1 → Cross-lingual knowledge used
- Budget ratio near 0.8-1.0 → Important task requires full retrieval

**Poor Predictions:**
- High `Support MAE` (> 0.1) → Model struggling to fit support set
- Most retrieved examples with low attention weights → Retrieval not helpful
- Budget ratio near 0 → Task deemed too easy, minimal retrieval needed

---

## Troubleshooting

### Issue: "No module named 'models'"

**Solution:**
```bash
cd araml
export PYTHONPATH=.
python scripts/interactive_cli.py
```

### Issue: "Checkpoint file not found"

**Solution:**
```bash
# Check what checkpoints exist
ls results/

# Use correct checkpoint path
PYTHONPATH=. python scripts/interactive_cli.py \
  --checkpoint results/best_model.pt
```

### Issue: CUDA out of memory

**Solution:**
- Use CPU instead
```bash
# Modify interactive_cli.py line 91 to force CPU:
# self.device = torch.device("cpu")
```

### Issue: Predictions always near 0.5

**Possible causes & solutions:**
1. Support examples don't span full range
   - Load diverse support examples: `>>> support diverse_support.json`

2. Retrieval not working
   - Check index size: `>>> info`
   - Rebuild index if needed: `PYTHONPATH=. python scripts/build_index.py`

3. Model not properly adapted
   - Try fine-tuning: `>>> retrain 10`

---

## Tips & Best Practices

### 1. Support Set Quality
- Use diverse examples spanning [0, 1] range
- Include extreme examples (very negative, very positive)
- Ensure labels are accurate and consistent

### 2. Fine-Tuning Strategy
- Start with 3-5 steps for quick adaptation
- Use 10-20 steps for significant domain shift
- Monitor `Final MAE` to avoid overfitting

### 3. Batch Processing
- Use for systematic evaluation
- Export results for statistical analysis
- Run multiple times with different support sets

### 4. Model Selection
- Keep best checkpoint saved: `>>> save results/best.pt`
- Compare different support strategies
- Document which checkpoint works best for each domain

---

## Example Session Walkthrough

```bash
$ cd araml
$ PYTHONPATH=. python scripts/interactive_cli.py --language ja

Device: cuda
Config loaded from: configs/config.yaml
Checkpoint loaded: results/best_model.pt
Retrieval index loaded: 100234 entries
Language: ja
Support examples: 5

Support Set:
────────────────────────────────────────────────────────────────────────────────
  [1] label=0.00 (1.0★)  ひどい品質で、すぐ壊れました。
  [2] label=0.25 (1.2★)  期待以下で、あまり満足できません。
  [3] label=0.50 (3.0★)  普通です。悪くも良くもありません。
  [4] label=0.75 (4.0★)  使いやすくて満足しています。
  [5] label=1.00 (5.0★)  最高です。買って本当に良かったです。

[Model information...]

Type 'help' for available commands or 'exit' to quit.

>>> test "この商品は予想以上に良かったです。"

Query: この商品は予想以上に良かったです。
════════════════════════════════════════════════════════════════════════════════
Prediction:       0.8234 (4.29★ / 5)
Raw Score:        0.8180
Support MAE:      0.0412
Budget:           19 examples (ratio=0.9500)

Top 5 Retrieved Examples:
────────────────────────────────────────────────────────────────────────────────
  [1] ja label=0.85 (4.4★) attn=0.2456
      使いやすくて満足しています。
  [2] en label=0.82 (4.28★) attn=0.2134
      Very satisfied with this product
  [3] de label=0.80 (4.2★) attn=0.1876
      Sehr zufrieden mit diesem Produkt
  [4] ja label=0.78 (4.12★) attn=0.1645
      予想より良い商品でした。
  [5] es label=0.76 (4.04★) attn=0.1489
      Muy satisfecho con este producto

>>> retrain 5

Fine-tuned for 5 steps:
  Initial loss: 0.238901
  Final loss:   0.048234
  Final MAE:    0.0356

>>> test "最高の買い物でした"

Query: 最高の買い物でした
════════════════════════════════════════════════════════════════════════════════
Prediction:       0.9123 (4.65★ / 5)  [Higher after fine-tuning]
Raw Score:        0.9080
Support MAE:      0.0245  [Lower after fine-tuning]
...

>>> save results/session_model.pt

Checkpoint saved: results/session_model.pt

>>> exit

$
```

---

## Support

For issues or questions:
1. Check `REGRESSION_README.md` for comprehensive documentation
2. Run `>>> help` within the CLI
3. Review command examples above
4. Check the model info with `>>> info`

Happy testing! 🚀

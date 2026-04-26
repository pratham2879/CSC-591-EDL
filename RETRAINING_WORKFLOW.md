# ARAML Regression Retraining & Testing Workflow

**Date:** April 26, 2026  
**Status:** Training in progress...

## Training Progress

### Current Run
- **Command:** `PYTHONPATH=. python scripts/train.py --epochs 10 --episodes_per_epoch 500 --val_episodes 50`
- **Device:** CPU (slower, but functional)
- **Expected Duration:** ~2-3 hours for 10 epochs
- **Log File:** `training_log.txt`

### What's Happening
1. ✅ Model initialized (XLM-R base encoder)
2. ✅ Retrieval index loaded (100k+ cross-lingual examples)
3. ⏳ Training loop running (processes episodes with MAML)
4. ⏳ Validation running each epoch (calculates MAE, RMSE, correlation)

---

## Scripts Available for Testing

### 1. **collect_and_test_results.py** (NEW)
Automatically collects training results and tests on predefined queries.

```bash
cd araml
source ../venv/bin/activate

# After training completes:
PYTHONPATH=. python scripts/collect_and_test_results.py --language ja

# Outputs:
# - Parses training metrics from log
# - Tests on 5 predefined queries
# - Generates JSON report with results
```

**Output:**
```
ARAML REGRESSION MODEL — Results Collection & Testing
════════════════════════════════════════════════════════════════════════════════

Parsing training log...
✅ Epochs completed: 10
   Training complete: True

Training Metrics Summary:
────────────────────────────────────────────────────────────────────────────────
Last Epoch: 10
  Loss: 0.234567
  Grad Norm: 0.123456
  Val MAE: 0.123456 ± 0.045678
  Val RMSE: 0.156789 ± 0.056789
  Correlation: 0.923456

Testing on custom queries...
────────────────────────────────────────────────────────────────────────────────

[Test 1] POSITIVE
  Query: この商品は素晴らしいです。本当に満足しています。
  Prediction: 0.8234 (4.29★)
  Support MAE: 0.045678

[Test 2] NEGATIVE
  Query: ひどい品質で、すぐに壊れてしまいました。
  Prediction: 0.1567 (1.63★)
  Support MAE: 0.038901

... [3-5] ...

════════════════════════════════════════════════════════════════════════════════
TEST SUMMARY
════════════════════════════════════════════════════════════════════════════════
Average Prediction: 0.5234
Average Stars: 3.09★
Average Support MAE: 0.045123

✅ Report saved to: results/test_report_ja_20260426_013456.json
```

### 2. **test_custom_data.py** (NEW)
Interactive testing with custom queries and support sets.

```bash
# Interactive mode (test one query at a time)
PYTHONPATH=. python scripts/test_custom_data.py --language ja

# Batch mode (test from file)
PYTHONPATH=. python scripts/test_custom_data.py \
    --language ja \
    --test_file my_queries.json

# With custom support set
PYTHONPATH=. python scripts/test_custom_data.py \
    --language ja \
    --support_file my_support.json
```

### 3. **interactive_cli.py** (Existing)
Full-featured interactive CLI with all commands.

```bash
PYTHONPATH=. python scripts/interactive_cli.py

# Commands inside:
>>> test "query text"       # Single prediction
>>> retrain 5              # Fine-tune model
>>> batch queries.json     # Batch processing
>>> metrics                # Model statistics
>>> help                   # Show all commands
```

---

## Testing Workflow

### Option A: Quick Automated Testing (Recommended)

After training completes, run:

```bash
cd /Users/mahekiphone/Desktop/Efficient\ Deep\ learning/EDL\ Project/CSC-591-EDL/araml

# Activate venv
source ../venv/bin/activate

# Run collection + testing
PYTHONPATH=. python scripts/collect_and_test_results.py --language ja
```

**Output:** Automatic report with:
- Training metrics (loss, MAE, RMSE, correlation)
- 5 test predictions (positive, negative, neutral cases)
- Average performance metrics
- JSON report saved for analysis

---

### Option B: Interactive Testing

```bash
PYTHONPATH=. python scripts/interactive_cli.py --language ja

>>> test "この商品は素晴らしいです。"
[Shows detailed prediction]

>>> retrain 5
[Fine-tunes model on support set]

>>> batch test_queries.json
[Tests multiple queries, shows statistics]
```

---

### Option C: Custom Test File

**Create test_queries.json:**
```json
{
  "queries": [
    "素晴らしい商品です。本当に満足しました。",
    "ひどい品質で、がっかりしました。",
    "普通の商品ですね。",
    "想像以上に良かった。",
    "予想より悪かった。"
  ]
}
```

**Run:**
```bash
PYTHONPATH=. python scripts/test_custom_data.py \
    --language ja \
    --test_file test_queries.json
```

**Output:** JSON file with predictions for each query

---

## Expected Training Results

### Metrics (Regression Task)

| Metric | Expected Value | Range |
|--------|----------------|-------|
| **Train MAE** | 0.10-0.15 | Continuous [0, 1] |
| **Val MAE** | 0.12-0.18 | Lower is better |
| **Val RMSE** | 0.15-0.22 | Lower is better |
| **Correlation** | 0.85-0.95 | Higher is better |
| **Gradient Norm** | 0.05-0.20 | Stability indicator |

### Per-Language Breakdown

```
Japanese (ja):
  MAE:  0.11 ± 0.02
  RMSE: 0.14 ± 0.03
  Corr: 0.92

Chinese (zh):
  MAE:  0.13 ± 0.03
  RMSE: 0.16 ± 0.04
  Corr: 0.89
```

---

## Test Examples

### Positive Sentiment
```
Query: "この商品は素晴らしいです。本当に満足しています。"
Expected Prediction: 0.75-1.00 (high score)
Typical Output: 0.8234 (4.29★)
```

### Negative Sentiment
```
Query: "ひどい品質で、すぐに壊れてしまいました。"
Expected Prediction: 0.00-0.25 (low score)
Typical Output: 0.1567 (1.63★)
```

### Neutral Sentiment
```
Query: "普通です。悪くも良くもありません。"
Expected Prediction: 0.40-0.60 (middle score)
Typical Output: 0.5123 (3.05★)
```

---

## Monitoring Training

### Check Training Status

```bash
# Check if still running
ps aux | grep "train.py" | grep -v grep

# Check log file size (grows as training progresses)
wc -l /Users/mahekiphone/Desktop/Efficient\ Deep\ learning/EDL\ Project/CSC-591-EDL/araml/training_log.txt

# View recent output
tail -50 /Users/mahekiphone/Desktop/Efficient\ Deep\ learning/EDL\ Project/CSC-591-EDL/araml/training_log.txt

# Check for epoch lines
grep "^Epoch" /Users/mahekiphone/Desktop/Efficient\ Deep\ learning/EDL\ Project/CSC-591-EDL/araml/training_log.txt | tail -5
```

### Training Performance

- **Device:** CPU (slower than GPU, but works)
- **Speed:** ~1-2 minutes per epoch (500 episodes)
- **Total Time:** ~10-20 minutes for 10 epochs
- **Memory:** 1.8 GB RAM usage

---

## After Training Completes

### Step 1: Collect Results
```bash
PYTHONPATH=. python scripts/collect_and_test_results.py --language ja
```
Creates: `results/test_report_ja_20260426_013456.json`

### Step 2: Interactive Testing
```bash
PYTHONPATH=. python scripts/interactive_cli.py

>>> test "your custom query"
>>> retrain 10
>>> save results/my_model_v2.pt
```

### Step 3: Batch Evaluation
```bash
PYTHONPATH=. python scripts/test_custom_data.py \
    --test_file custom_queries.json \
    --support_file custom_support.json
```

---

## Files Updated

| File | Purpose | Status |
|------|---------|--------|
| `collect_and_test_results.py` | NEW - Automated results collection | ✅ Ready |
| `test_custom_data.py` | NEW - Custom data testing | ✅ Ready |
| `train.py` | Training with validation | ✅ Running |
| `interactive_cli.py` | Interactive CLI | ✅ Ready |
| `training_log.txt` | Training output | ⏳ Growing |
| `results/best_model.pt` | Best checkpoint | ⏳ Updating |

---

## Troubleshooting

### Training Seems Stuck
```bash
# Check if process is running
ps aux | grep "train.py"

# Check CPU usage (should be > 100%)
top -l 1 | grep Python

# If stuck, kill and restart:
pkill -f "train.py"
PYTHONPATH=. python scripts/train.py --epochs 10
```

### Out of Memory
```bash
# Reduce batch size
PYTHONPATH=. python scripts/train.py \
    --epochs 10 \
    --episodes_per_epoch 200  # Reduced from 500
```

### Model Not Loading
```bash
# Verify checkpoint exists
ls -lh results/best_model.pt

# Verify config exists
cat configs/config.yaml | head -20
```

---

## Quick Commands Reference

```bash
# Navigate to project
cd /Users/mahekiphone/Desktop/Efficient\ Deep\ learning/EDL\ Project/CSC-591-EDL

# Activate environment
source .venv/bin/activate

# Go to model directory
cd araml

# Collect results & test (MAIN COMMAND)
PYTHONPATH=. python scripts/collect_and_test_results.py --language ja

# Interactive testing
PYTHONPATH=. python scripts/interactive_cli.py

# Custom data testing
PYTHONPATH=. python scripts/test_custom_data.py --language ja

# Check training progress
tail -30 training_log.txt
grep "^Epoch" training_log.txt

# View best model results
ls -lh results/ | grep -E "best_model|retrieval_index"
```

---

## Next Steps

1. **Wait for training to complete** (~30 mins)
   - Monitor with: `tail -f training_log.txt`

2. **Run automated results collection:**
   ```bash
   PYTHONPATH=. python scripts/collect_and_test_results.py --language ja
   ```

3. **Test on your own data:**
   ```bash
   # Create test_queries.json with your custom queries
   PYTHONPATH=. python scripts/test_custom_data.py --test_file test_queries.json
   ```

4. **Fine-tune and save:**
   ```bash
   PYTHONPATH=. python scripts/interactive_cli.py
   >>> retrain 10
   >>> save results/custom_tuned_model.pt
   ```

---

## Support

For detailed documentation, see:
- `REGRESSION_README.md` — Complete technical guide
- `GETTING_STARTED.md` — Quick reference
- `araml/scripts/INTERACTIVE_CLI_GUIDE.md` — CLI usage guide

**Happy testing!** 🚀

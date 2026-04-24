# ARAML Regression Conversion - Complete Summary

## Branch: `regression`

This document summarizes all changes made to convert ARAML from **binary sentiment classification** to **few-shot regression**.

---

## Overview

### Original Task
- **Type**: Binary classification (negative/positive)
- **Labels**: 0 (negative, 1-2 stars) or 1 (positive, 4-5 stars)
- **Neutral samples**: Dropped (3-star ratings)
- **Loss**: Cross-entropy
- **Metrics**: Accuracy, Precision, Recall, F1

### New Task (Regression)
- **Type**: Continuous regression
- **Labels**: Normalized to [0, 1] using formula `y = (stars - 1) / 4`
  - 1-star → 0.00 (most negative)
  - 2-star → 0.25
  - 3-star → 0.50 (neutral, now kept)
  - 4-star → 0.75
  - 5-star → 1.00 (most positive)
- **Neutral samples**: Kept (no longer dropped)
- **Loss**: Smooth L1 (Huber)
- **Metrics**: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)

---

## File-by-File Changes

### 1. **araml/data/preprocess.py**
**Changes: Label normalization, keep all ratings**

- **Updated docstring**: Reflects regression task, removal of neutral drop
- **New function**: `normalize_to_regression(label_0_4: int) -> float`
  - Converts 0-indexed labels (0-4) to normalized range [0, 1]
  - Formula: `return label_0_4 / 4.0`
- **Removed function**: `remap_to_binary()` (was: 0→0, 2→drop, 3-4→1)
- **Updated preprocessing loop**:
  - No longer drops neutral (label 2) samples
  - All 200K examples per language now used
  - Labels stored as floats in [0, 1]
  - Added `raw_stars` field (1-5) for reference
- **Updated pool parameters**:
  - `MIN_PER_CAT_PER_CLASS` → `MIN_PER_CAT` (30, no class balance needed)
  - `MIN_VIABLE_PER_CLASS` → `MIN_VIABLE_EXAMPLES` (5)
- **Output changes**:
  - Statistics now show label range [min, max] and mean instead of class counts
  - Star distribution printed (1-5 histogram)
- **Final message**: "Regression output head must use output_dim=1"

### 2. **araml/models/meta_learner.py**
**Changes: Head architecture, loss function, metrics**

**Class MetaLearner**:
- Constructor parameter: `num_classes=2` → `output_dim=1`
- Head architecture: `nn.Linear(768, 2)` → `nn.Linear(768, 1)`
- Forward output: Shape (batch_size, 1) for single regression prediction

**Function inner_loop()**:
- Parameter: `classifier` → `regressor`
- Support labels: Now float tensors (was: long tensors)
- Loss: `F.cross_entropy(logits, support_labels)` → `F.smooth_l1_loss(preds_squeezed, support_labels)`
- Predictions: Squeezed from (n_support, 1) to (n_support,) for loss calculation

**Function _episode_forward()**:
- Query labels: `dtype=torch.long` → `dtype=torch.float32`
- Support labels: `dtype=torch.long` → `dtype=torch.float32`
- Loss calculation: 
  - Old: `F.cross_entropy(query_logits, query_labels)`
  - New: `F.smooth_l1_loss(query_preds.squeeze(-1), query_labels)`
- Metrics computed:
  - Old: `accuracy = (argmax(logits) == labels).mean()`
  - New: `mae = mean(|preds - targets|)` and `rmse = sqrt(mean((preds - targets)²))`
- Return signature: `(loss, acc, predictions, targets)` → `(loss, mae, rmse, predictions, targets)`

**Function meta_train_step()**:
- Return: `(loss_item, acc, grad_norm, predictions, targets)` → `(loss_item, mae, rmse, grad_norm, predictions, targets)`

**Function maml_eval_episode()**:
- Removed: sklearn imports for Cohen's kappa
- Return metrics:
  - Old: `{"accuracy": acc, "kappa": kappa, ...}`
  - New: `{"mae": mae, "rmse": rmse, ...}`

**Function diagnose_gradient_flow()**:
- Metrics printed now show MAE/RMSE instead of accuracy

### 3. **araml/scripts/train.py**
**Changes: Removed classification imports, updated metrics collection**

- **Imports**: Removed `from sklearn.metrics import precision_recall_fscore_support`
- **Docstring**: Updated to reflect regression task
- **Training loop**:
  - Metric tracking changed:
    - Old: `epoch_losses, epoch_accs, epoch_gnorms`
    - New: `epoch_losses, epoch_maes, epoch_rmses, epoch_gnorms`
  - Language tracking changed:
    - Old: `lang_labels` (class 0/1)
    - New: `lang_targets` (continuous values)
- **Meta-train-step unpacking**: Changed to 6 return values instead of 5
- **Logging**:
  - Batch logging: `loss={loss:.4f} acc={acc:.3f}` → `loss={loss:.6f} mae={mae:.6f} rmse={rmse:.6f}`
  - Epoch logging: Changed from classification metrics to regression metrics
    - Old: `Acc/P/R/F1` per language
    - New: `MAE/RMSE` per language
- **Best model selection**:
  - Old: `if mean_acc > best_acc`
  - New: `if mean_mae < best_mae` (lower error is better)
- **Verification**: Compute overall MAE/RMSE on aggregated predictions

### 4. **araml/scripts/evaluate.py**
**Changes: Replaced classification metrics with regression metrics**

- **Docstring**: Updated to reflect regression task
- **Imports**: Removed classification metric imports (precision_recall_fscore_support, confusion_matrix, matthews_corrcoef)
- **Metrics collection**:
  - Old: `accs, kappas` + sklearn metrics
  - New: `maes, rmses` + numpy-based metrics
- **Episode evaluation**:
  - Changed unpacking to match new `maml_eval_episode()` return signature
- **Output reporting**: Complete restructuring
  - **Old sections**: Accuracy distribution, per-class breakdown, confusion matrix, MCC
  - **New sections**: 
    - Overall MAE/RMSE with 95% CI
    - Pearson correlation coefficient
    - Episode MAE distribution (min, median, max, percentiles)
    - Episode RMSE distribution
    - Per-language MAE/RMSE and prediction/target ranges
- **Removed**: Cohen's kappa, precision/recall/F1, MCC, confusion matrix

### 5. **araml/models/araml.py**
**Changes: Config parameter update**

- **Docstring**: Added "(REGRESSION)" tag
- **MetaLearner initialization**:
  - Old: `num_classes=model_cfg["num_classes"]`
  - New: `output_dim=model_cfg["output_dim"]`

### 6. **araml/configs/config.yaml**
**Changes: Model output configuration**

- **Old**:
  ```yaml
  model:
    num_classes: 2    # Binary sentiment
  meta_learning:
    n_way: 2
  ```
- **New**:
  ```yaml
  model:
    output_dim: 1     # Single output for regression
  meta_learning:
    n_way: 2          # Historical (not used in regression)
  ```

### 7. **araml/utils/episode_sampler.py**
**Changes: Remove class balancing, work with continuous labels**

- **Docstring**: Updated to reflect regression task
- **Class CategoryStratifiedEpisodeSampler**:
  - Removed: Hard assertion `assert n_class == 2`
  - Removed: Class-per-class sampling logic
  - Updated: Index structure from `{lang: {cat: {label: [records]}}}` to `{lang: {cat: [records]}}`
  
- **Constructor changes**:
  - Removed: `_n_support_per_class`, `_n_query_per_class`
  - New: `_min_per_category = n_shot + n_query`
  - Category validation: No longer requires both classes present
  - Log message: Updated to indicate regression episodes

- **sample_episode() method**:
  - Support sampling:
    - Old: Balanced sampling per class (n_shot examples per class)
    - New: Random sampling of n_shot examples (no class balance)
  - Query sampling:
    - Old: Balanced sampling per class (n_query // n_class per class)
    - New: Random sampling of n_query examples (no class balance)
  - Log metrics:
    - Old: `support_class_dist={support_dist}`
    - New: `support_mean_label=%.3f query_mean_label=%.3f`
  - Return signature: Unchanged structure, but labels now floats in [0, 1]

---

## Unchanged Components

✅ **XLM-R Text Encoder**: Layer freezing strategy, preprocessing, encoding
✅ **ARC (Adaptive Retrieval Controller)**: Query generation, budget prediction, attention
✅ **FAISS Retrieval Index**: Retrieval pipeline, high-resource language indexing
✅ **Feature Augmentation**: L2-normalize + elementwise addition pattern
✅ **MAML Structure**: Inner loop, outer loop, create_graph=True, gradient flow

---

## Breaking Changes

⚠️ **Config**: Must use `output_dim: 1` instead of `num_classes: 2`
⚠️ **Labels**: Now float in [0, 1] instead of int {0, 1}
⚠️ **Return signatures**: Functions return additional metrics (mae, rmse instead of acc)
⚠️ **Training pools**: All examples kept; no longer dropped at label=3 (neutral)

---

## Validation Checklist

- [x] No syntax errors
- [x] Branch created: `regression`
- [x] All imports updated
- [x] Loss function changed to Huber
- [x] Metrics changed to MAE/RMSE
- [x] Labels normalized to [0, 1]
- [x] Neutral samples kept
- [x] Episode sampler handles continuous labels
- [x] Return signatures updated throughout
- [x] Config updated
- [x] Git commit successful

---

## How to Use

1. **Preprocess data**:
   ```bash
   cd araml
   python data/preprocess.py
   ```

2. **Train** (with regression metrics):
   ```bash
   python scripts/train.py --epochs 20
   ```

3. **Evaluate** (with MAE/RMSE):
   ```bash
   python scripts/evaluate.py --checkpoint results/best_model.pt --n_episodes 600
   ```

---

## Notes

- The model will now predict continuous sentiment values in [0, 1] range
- Lower MAE/RMSE values indicate better performance
- Pearson correlation coefficient is reported to measure prediction-target alignment
- All 5 star ratings (1-5) are now used; no data is dropped
- The architecture remains fundamentally the same - only the output head and loss have changed

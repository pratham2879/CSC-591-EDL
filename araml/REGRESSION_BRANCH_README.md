# ARAML Regression Branch README

This branch converts ARAML from binary sentiment classification to few-shot regression.

## What Changed

- Output head is now `output_dim: 1` instead of `num_classes: 2`.
- Labels are continuous values in `[0, 1]`.
- Loss is `SmoothL1Loss` (Huber).
- Metrics are `MAE` and `RMSE`.
- Episode sampling no longer tries to balance binary classes.
- Training now runs an explicit validation pass at the end of each epoch.
- The best checkpoint is now selected by validation MAE when validation data is available.

## Main Commands

From inside `araml/` with the venv active:

```bash
python data/preprocess.py
python scripts/quick_test.py
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/pipeline_test.py
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/train.py --epochs 20 --episodes_per_epoch 1000 --val_episodes 100
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/evaluate.py --checkpoint results/best_model.pt --n_episodes 600
```

## Validation Flow

Training now does two checks per epoch:

1. Meta-train over sampled low-resource episodes from the pool files.
2. Run validation episodes from the low-resource `validation` split in `data/processed/amazon_{ja,zh}.json`.

The `results/best_model.pt` checkpoint is updated only when validation MAE improves.

## Interactive CLI

You can test the regression model from the terminal:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/regression_cli.py
```

Single-query example:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/regression_cli.py \
  --language ja \
  --query "この商品はかなり良いです。"
```

Custom support examples:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/regression_cli.py \
  --support_file my_support.json
```

Support file format:

```json
[
  {"text": "bad example", "label": 0.0},
  {"text": "slightly negative", "label": 0.25},
  {"text": "neutral", "label": 0.5},
  {"text": "good example", "label": 0.75},
  {"text": "excellent example", "label": 1.0}
]
```

The CLI also accepts star-style labels from `1` to `5`, and converts them to `[0, 1]`.

## Notes

- `quick_test.py` and `pipeline_test.py` are regression-aware.
- `scripts/demo.py` and `scripts/app.py` still reflect the older classification demo path and should not be treated as the canonical regression interface.
- If you see OpenMP duplication issues on macOS, prepend `KMP_DUPLICATE_LIB_OK=TRUE`.

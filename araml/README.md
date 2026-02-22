# ARAML — Adaptive Retrieval-Augmented Meta-Learning for Low-Resource NLP

Barebones, collaboration-ready research scaffold for rapid experimentation on meta-learning + retrieval ideas in low-resource NLP.

## Project Goals

- Keep the codebase minimal and clean.
- Provide clear interfaces for future ARAML logic.
- Enable quick onboarding for collaborators.
- Support future extension toward publication-quality experiments.

## Setup

```bash
python -m venv .venv
# Windows CMD
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```bash
python main.py
```

This runs a dummy training/evaluation pass using synthetic random tensors (no real dataset downloads).

## Repository Layout

- `configs/`: YAML experiment configs (model, training, retrieval).
- `data/`: Dataset interfaces and episodic sampler placeholders.
- `models/`: Encoders and meta-learning model skeletons.
- `retrieval/`: FAISS-backed retrieval wrappers.
- `training/`: Trainer/evaluator orchestration.
- `utils/`: Metrics, logging, reproducibility helpers.
- `experiments/`: Entry scripts for baseline vs ARAML tracks.
- `tests/`: Basic sanity tests.

## Plugging in Real Datasets Later

Implement dataset wrappers in `data/datasets.py` and route them via `create_dataset(...)`.
Suggested next targets:

- Amazon Reviews (domain adaptation setup)
- XNLI (cross-lingual low-resource transfer)

## TODO Roadmap

1. Add real dataset loaders + tokenizer pipelines.
2. Implement episodic support/query construction for N-way K-shot tasks.
3. Add true MAML inner-loop adaptation.
4. Implement retrieval-augmented prototype updates.
5. Add checkpointing, experiment tracking, and reproducible sweeps.
6. Add benchmark-specific evaluation protocols and confidence intervals.

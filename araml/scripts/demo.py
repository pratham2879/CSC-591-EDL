"""
demo.py — End-to-end ARAML inference demo for Japanese sentiment classification.

Proves:
  1. Cross-lingual retrieval: FAISS index built on EN/FR examples finds
     semantically relevant neighbours for Japanese queries.
  2. ARC is doing something: task_emb norm, predicted budget k, and
     per-example attention weights are printed.
  3. Classifier is correct: MAML-adapted head predicts the right polarity.

Run from the araml/ directory:
    python scripts/demo.py
"""
import sys, os
# Force UTF-8 output on Windows consoles that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              matthews_corrcoef)

from models.araml import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner import inner_loop, MetaLearner


# ---------------------------------------------------------------------------
# Hard-coded examples
# ---------------------------------------------------------------------------

SUPPORT = [
    ("この製品は素晴らしいです。品質が高くてとても満足しています。", 1),
    ("デザインが美しく、使いやすいです。おすすめします。",             1),
    ("最高の買い物でした。また購入したいと思います。",                 1),
    ("品質が悪くてすぐに壊れました。返品しました。",                   0),
    ("期待外れでした。お金の無駄だと思います。",                       0),
]

TEST = [
    ("このカメラは画質がとても良いです。",   1),   # positive
    ("全く使えません。最悪の製品です。",      0),   # negative
    ("普通の品質です。",                     None), # neutral / ambiguous
]

LABEL_STR = {0: "negative", 1: "positive", None: "neutral (no ground truth)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def faiss_score_to_cosine(score: float) -> float:
    """
    Index uses IndexFlatIP on L2-normalised vectors, so FAISS returns
    the inner product directly, which equals cosine similarity for unit vectors.
    """
    return float(score)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Load model + index
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_everything(config_path: str = None,
                    model_path:  str = None,
                    index_path:  str = None):
    if config_path is None:
        config_path = os.path.join(_ROOT, "configs", "config.yaml")
    if model_path is None:
        model_path = os.path.join(_ROOT, "results", "best_model.pt")
    if index_path is None:
        index_path = os.path.join(_ROOT, "results", "retrieval_index")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = ARAML(config).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded model from {model_path}")

    # Retrieval index
    index = CrossLingualRetrievalIndex(
        embedding_dim=config["model"]["hidden_dim"],
        similarity=config["retrieval"]["similarity"],
    )
    index.load(index_path)
    print(f"Loaded retrieval index: {index.index.ntotal} vectors")

    return model, index, config, device


# ---------------------------------------------------------------------------
# Core demo
# ---------------------------------------------------------------------------

def run_demo():
    model, retrieval_index, config, device = load_everything()

    meta_cfg = config["meta_learning"]
    encoder  = model.encoder
    arc      = model.arc
    clf      = model.meta_learner

    support_texts  = [t for t, _ in SUPPORT]
    support_labels = torch.tensor([l for _, l in SUPPORT], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Step 1 — encode support set, compute task embedding
    # ------------------------------------------------------------------
    section("TASK EMBEDDING  (mean of 5 Japanese support sentences)")

    with torch.no_grad():
        support_embs = encoder.encode_text(support_texts, device)   # (5, 768)

    task_emb = support_embs.mean(0, keepdim=True)                   # (1, 768)
    task_norm = task_emb.norm().item()
    print(f"  task_emb shape : {tuple(task_emb.shape)}")
    print(f"  task_emb norm  : {task_norm:.4f}")

    # ------------------------------------------------------------------
    # Step 2 — ARC: generate query, predict budget, retrieve
    # ------------------------------------------------------------------
    section("ARC  —  retrieval query & budget")

    with torch.no_grad():
        arc_query = arc.generate_query(task_emb)              # (1, 768)
        arc_q_norm = arc_query.norm().item()
        budget_ratio = arc.budget_predictor(task_emb).item()
        k_predicted  = max(1, int(budget_ratio * config["retrieval"]["max_retrieved"]))

    print(f"  arc_query norm     : {arc_q_norm:.4f}")
    print(f"  budget_ratio (raw) : {budget_ratio:.4f}")
    print(f"  predicted k        : {k_predicted}")

    # Retrieve top-3 for display (always show 3 regardless of budget)
    DISPLAY_K = 3
    query_vec = arc_query.detach().float().cpu().numpy()             # (1, 768)
    retrieved = retrieval_index.retrieve(query_vec, k=max(DISPLAY_K, k_predicted))

    section("RETRIEVED EXAMPLES  (top-3 from FAISS, based on task/support embedding)")
    for i in range(DISPLAY_K):
        dist    = retrieved["distances"][i]
        cos_sim = faiss_score_to_cosine(dist)
        text    = retrieved["texts"][i]
        lang    = retrieved["languages"][i]
        label   = LABEL_STR[retrieved["labels"][i]]
        print(f"\n  [{i+1}] lang={lang.upper()}  cos_sim={cos_sim:.4f}  label={label}")
        print(f"       \"{text[:100]}\"")

    # ------------------------------------------------------------------
    # Step 3 — attention weights over retrieved examples
    # ------------------------------------------------------------------
    section("ARC ATTENTION WEIGHTS  over retrieved examples")

    ret_texts_k = retrieved["texts"][:k_predicted]
    with torch.no_grad():
        ret_embs    = encoder.encode_text(ret_texts_k, device)       # (k, 768)
        attn_weights = arc.compute_attention_weights(arc_query, ret_embs)  # (k,)
        weighted_ret = (attn_weights.unsqueeze(-1) * ret_embs).sum(0)     # (768,)

    print(f"\n  k_predicted = {k_predicted}")
    for i, w in enumerate(attn_weights.tolist()):
        lang  = retrieved["languages"][i]
        label = LABEL_STR[retrieved["labels"][i]]
        text  = retrieved["texts"][i][:70]
        print(f"  [{i+1:02d}] weight={w:.4f}  lang={lang.upper()}  {label}  \"{text}\"")

    entropy = -(attn_weights * attn_weights.log()).sum().item()
    print(f"\n  Attention entropy: {entropy:.4f}  "
          f"(uniform={np.log(k_predicted):.4f}, "
          f"peaked=0.0000)")

    # ------------------------------------------------------------------
    # Step 4 — MAML inner loop: adapt classifier on support set
    # ------------------------------------------------------------------
    section("MAML INNER LOOP  —  adapting classifier on 5 support examples")

    weighted_ret_n = F.normalize(weighted_ret, p=2, dim=-1)   # (768,)
    support_embs_n = F.normalize(support_embs, p=2, dim=-1)   # (5, 768)
    support_aug    = support_embs_n + weighted_ret_n.unsqueeze(0)  # (5, 768)
    support_aug    = support_aug.float()

    adapted = inner_loop(
        clf.classifier,
        support_aug,
        support_labels,
        inner_lr=meta_cfg["inner_lr"],
        inner_steps=meta_cfg["inner_steps"],
    )

    # Quick check: loss on support with adapted weights
    with torch.no_grad():
        support_logits = F.linear(support_aug, adapted["weight"], adapted["bias"])
        support_loss   = F.cross_entropy(support_logits, support_labels).item()
        support_acc    = (support_logits.argmax(-1) == support_labels).float().mean().item()
    print(f"  Support loss after adaptation : {support_loss:.4f}")
    print(f"  Support accuracy after adapt  : {support_acc:.0%}")

    # ------------------------------------------------------------------
    # Step 5 — predict each test sentence + collect metrics
    # ------------------------------------------------------------------
    section("TEST SENTENCE PREDICTIONS")

    all_preds, all_labels_eval = [], []

    for i, (text, gt_label) in enumerate(TEST):
        print(f"\n  -- Test {i+1} ----------------------------------------------------")
        print(f"  Text  : \"{text}\"")
        print(f"  Truth : {LABEL_STR[gt_label]}")

        with torch.no_grad():
            query_emb  = encoder.encode_text([text], device)          # (1, 768)
            query_emb_n = F.normalize(query_emb, p=2, dim=-1)
            query_aug   = (query_emb_n + weighted_ret_n.unsqueeze(0)).float()

        query_logits = F.linear(query_aug, adapted["weight"], adapted["bias"])
        probs        = F.softmax(query_logits, dim=-1).squeeze(0)     # (2,)
        pred_idx     = query_logits.argmax(-1).item()
        confidence   = probs[pred_idx].item()
        margin       = abs(probs[1].item() - probs[0].item())  # decision margin

        correct = ""
        if gt_label is not None:
            correct = "  CORRECT" if pred_idx == gt_label else "  WRONG"
            all_preds.append(pred_idx)
            all_labels_eval.append(gt_label)

        print(f"  Pred  : {LABEL_STR[pred_idx]}  (confidence={confidence:.2%}){correct}")
        print(f"  Probs : negative={probs[0].item():.4f}  positive={probs[1].item():.4f}")
        print(f"  Margin (|pos-neg|) : {margin:.4f}  "
              f"({'high certainty' if margin > 0.5 else 'low certainty — model uncertain'})")

    # ------------------------------------------------------------------
    # Step 6 — demo metrics summary (only for examples with ground truth)
    # ------------------------------------------------------------------
    if all_labels_eval:
        section("DEMO METRICS SUMMARY  (labeled test examples only)")

        demo_acc = sum(p == l for p, l in zip(all_preds, all_labels_eval)) / len(all_labels_eval)
        print(f"  Accuracy : {demo_acc:.0%}  ({sum(p==l for p,l in zip(all_preds,all_labels_eval))}"
              f"/{len(all_labels_eval)} correct)")

        if len(set(all_labels_eval)) > 1:
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_labels_eval, all_preds, average="macro", zero_division=0
            )
            mcc = matthews_corrcoef(all_labels_eval, all_preds)
            cm  = confusion_matrix(all_labels_eval, all_preds, labels=[0, 1])
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall   : {rec:.4f}")
            print(f"  F1       : {f1:.4f}")
            print(f"  MCC      : {mcc:.4f}  (0=random, 1=perfect)")
            print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
            print(f"               pred_neg  pred_pos")
            print(f"  actual_neg :  {cm[0,0]:>7}   {cm[0,1]:>7}")
            print(f"  actual_pos :  {cm[1,0]:>7}   {cm[1,1]:>7}")
        else:
            print(f"  (Need both classes in ground-truth labels for P/R/F1/MCC)")

        print(f"\n  NOTE: This demo uses only {len(all_labels_eval)} labeled examples.")
        print(f"  For full evaluation over 600 episodes run:")
        print(f"    python scripts/evaluate.py --checkpoint results/best_model.pt --n_episodes 600")

    print()


if __name__ == "__main__":
    run_demo()

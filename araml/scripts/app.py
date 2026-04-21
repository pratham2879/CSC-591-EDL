"""
app.py — ARAML interactive demo UI (Gradio)

Loads the trained model + FAISS index once, then lets you type any
Japanese or Chinese text and see the sentiment prediction, confidence,
attention weights, and the retrieved cross-lingual examples.

Run from inside araml/ with the venv active:
    pip install gradio          # first time only
    python scripts/app.py

Then open  http://127.0.0.1:7860  in your browser.
"""
import os
import sys

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import gradio as gr

from models.araml import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner import inner_loop

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_ROOT, "configs", "config.yaml")
MODEL_PATH  = os.path.join(_ROOT, "results", "best_model.pt")
INDEX_PATH  = os.path.join(_ROOT, "results", "retrieval_index")

# ---------------------------------------------------------------------------
# Default support sets (5-shot, binary)
# ---------------------------------------------------------------------------
DEFAULT_SUPPORT = {
    "ja": [
        ("この製品は素晴らしいです。品質が高くてとても満足しています。", 1),
        ("デザインが美しく、使いやすいです。おすすめします。",             1),
        ("最高の買い物でした。また購入したいと思います。",                 1),
        ("品質が悪くてすぐに壊れました。返品しました。",                   0),
        ("期待外れでした。お金の無駄だと思います。",                       0),
    ],
    "zh": [
        ("这个产品质量非常好，我非常满意。",   1),
        ("非常推荐，性价比很高。",             1),
        ("用了很久了，一直很好用。",           1),
        ("质量很差，很快就坏了。",             0),
        ("完全不值这个价钱，太失望了。",       0),
    ],
}

# Labels used in the UI
LABEL_STR = {0: "Negative", 1: "Positive"}

# ---------------------------------------------------------------------------
# Model loading (done once at startup)
# ---------------------------------------------------------------------------
print("Loading model and retrieval index …")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

MODEL = ARAML(CONFIG).to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
MODEL.load_state_dict(ckpt)
MODEL.eval()
print(f"Loaded model: {MODEL_PATH}")

RETRIEVAL_INDEX = CrossLingualRetrievalIndex(
    embedding_dim=CONFIG["model"]["hidden_dim"],
    similarity=CONFIG["retrieval"]["similarity"],
)
RETRIEVAL_INDEX.load(INDEX_PATH)
print(f"Loaded index: {RETRIEVAL_INDEX.index.ntotal} vectors\n")

ENCODER = MODEL.encoder
ARC     = MODEL.arc
CLF     = MODEL.meta_learner
META_CFG = CONFIG["meta_learning"]

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def predict(query_text: str, language: str,
            sup1: str, sup2: str, sup3: str, sup4: str, sup5: str,
            lbl1: int, lbl2: int, lbl3: int, lbl4: int, lbl5: int):
    """
    Run one ARAML inference:
      1. Encode the 5 support examples and adapt the classifier.
      2. Encode the query and predict sentiment.
      3. Return formatted results.
    """
    query_text = query_text.strip()
    if not query_text:
        return ("Please enter a query sentence.", "", "", "")

    support_texts  = [s.strip() for s in [sup1, sup2, sup3, sup4, sup5]]
    support_labels_raw = [lbl1, lbl2, lbl3, lbl4, lbl5]

    # Basic validation
    if not all(support_texts):
        return ("Please fill in all 5 support sentences.", "", "", "")
    if len(set(support_labels_raw)) < 2:
        return ("Support set must contain at least one positive and one negative example.", "", "", "")

    support_labels = torch.tensor(support_labels_raw, dtype=torch.long, device=DEVICE)

    # -- Encode support set --------------------------------------------------
    with torch.no_grad():
        support_embs = ENCODER.encode_text(support_texts, DEVICE)   # (5, 768)

    task_emb = support_embs.mean(0, keepdim=True)                   # (1, 768)

    # -- ARC: retrieve cross-lingual examples --------------------------------
    with torch.no_grad():
        arc_query    = ARC.generate_query(task_emb)                 # (1, 768)
        budget_ratio = ARC.budget_predictor(task_emb).item()
        k_predicted  = max(1, int(budget_ratio * CONFIG["retrieval"]["max_retrieved"]))

    query_vec = arc_query.detach().float().cpu().numpy()
    retrieved = RETRIEVAL_INDEX.retrieve(query_vec, k=k_predicted)

    # -- Attention-weighted retrieval embedding ------------------------------
    with torch.no_grad():
        ret_texts_k  = retrieved["texts"][:k_predicted]
        ret_embs     = ENCODER.encode_text(ret_texts_k, DEVICE)     # (k, 768)
        attn_weights = ARC.compute_attention_weights(arc_query, ret_embs)  # (k,)
        weighted_ret = (attn_weights.unsqueeze(-1) * ret_embs).sum(0)      # (768,)

    weighted_ret_n = F.normalize(weighted_ret, p=2, dim=-1)

    # -- Augment support + MAML adapt ----------------------------------------
    support_embs_n = F.normalize(support_embs, p=2, dim=-1)
    support_aug    = (support_embs_n + weighted_ret_n.unsqueeze(0)).float()

    adapted = inner_loop(
        CLF.classifier,
        support_aug,
        support_labels,
        inner_lr=META_CFG["inner_lr"],
        inner_steps=META_CFG["inner_steps"],
    )

    # -- Predict query -------------------------------------------------------
    with torch.no_grad():
        query_emb   = ENCODER.encode_text([query_text], DEVICE)     # (1, 768)
        query_emb_n = F.normalize(query_emb, p=2, dim=-1)
        query_aug   = (query_emb_n + weighted_ret_n.unsqueeze(0)).float()

    query_logits = F.linear(query_aug, adapted["weight"], adapted["bias"])
    probs        = F.softmax(query_logits, dim=-1).squeeze(0)       # (2,)
    pred_idx     = int(query_logits.argmax(-1).item())
    confidence   = float(probs[pred_idx].item())
    margin       = float(abs(probs[1].item() - probs[0].item()))

    # -- Support adaptation quality ------------------------------------------
    with torch.no_grad():
        sup_logits = F.linear(support_aug, adapted["weight"], adapted["bias"])
        sup_acc    = (sup_logits.argmax(-1) == support_labels).float().mean().item()
        sup_loss   = F.cross_entropy(sup_logits, support_labels).item()

    # ---- Format outputs -------------------------------------------------------

    # 1. Prediction card
    certainty = "High" if margin > 0.5 else ("Medium" if margin > 0.25 else "Low")
    pred_out = (
        f"**Prediction: {LABEL_STR[pred_idx]}**\n\n"
        f"- Confidence : **{confidence:.1%}**\n"
        f"- Margin     : {margin:.4f}  ({certainty} certainty)\n"
        f"- Neg prob   : {probs[0].item():.4f}\n"
        f"- Pos prob   : {probs[1].item():.4f}"
    )

    # 2. Retrieval info
    ret_lines = [
        f"**Retrieved {k_predicted} cross-lingual examples** (budget ratio={budget_ratio:.3f})\n"
    ]
    for i in range(min(k_predicted, 5)):
        w    = attn_weights[i].item() if i < len(attn_weights) else 0.0
        lang = retrieved["languages"][i].upper()
        lbl  = LABEL_STR.get(retrieved["labels"][i], "?")
        txt  = retrieved["texts"][i][:100]
        ret_lines.append(f"**[{i+1}]** `{lang}` | attn={w:.4f} | {lbl}  \n> {txt}")
    ret_out = "\n\n".join(ret_lines)

    # 3. Adaptation quality
    adapt_out = (
        f"**MAML Adaptation (inner loop)**\n\n"
        f"- Inner steps : {META_CFG['inner_steps']}\n"
        f"- Inner LR    : {META_CFG['inner_lr']}\n"
        f"- Support loss after adapt : {sup_loss:.4f}\n"
        f"- Support accuracy         : {sup_acc:.0%}\n\n"
        f"Support acc ≥ 80% means the classifier successfully adapted to the 5 examples you provided."
    )

    # 4. ARC internals
    entropy = float(-(attn_weights * attn_weights.log()).sum().item())
    arc_out = (
        f"**Adaptive Retrieval Controller (ARC)**\n\n"
        f"- Task embedding norm : {task_emb.norm().item():.4f}\n"
        f"- Query norm          : {arc_query.norm().item():.4f}\n"
        f"- Budget ratio        : {budget_ratio:.4f}  → k={k_predicted}\n"
        f"- Attention entropy   : {entropy:.4f}  (uniform={np.log(max(k_predicted,1)):.4f}, peaked=0)\n\n"
        f"Lower entropy = model focuses on fewer retrieved examples."
    )

    return pred_out, ret_out, adapt_out, arc_out


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

LANG_DEFAULTS = DEFAULT_SUPPORT["ja"]

with gr.Blocks(title="ARAML — Cross-Lingual Sentiment Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
# ARAML — Adaptive Retrieval-Augmented Meta-Learning
### Cross-Lingual Few-Shot Sentiment Classification (Japanese / Chinese)

Type a sentence, provide 5 labeled support examples, and let the model classify it.
The model retrieves cross-lingual knowledge from English/French/German/Spanish examples and adapts on-the-fly.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Query sentence (Japanese or Chinese)",
                placeholder="このカメラは画質がとても良いです。",
                lines=2,
            )
            language_sel = gr.Radio(
                choices=["ja", "zh"], value="ja",
                label="Language of query + support set"
            )

        with gr.Column(scale=1):
            gr.Markdown("**Prediction**")
            pred_output = gr.Markdown()

    with gr.Accordion("Support set — 5 labeled examples (edit to customize)", open=False):
        gr.Markdown(
            "These 5 examples teach the model what positive and negative sentiment looks "
            "like in the target language. They are the 'few-shot' examples."
        )

        def defaults(lang):
            sups = DEFAULT_SUPPORT.get(lang, DEFAULT_SUPPORT["ja"])
            return [sups[i][0] for i in range(5)] + [sups[i][1] for i in range(5)]

        sup_texts  = [gr.Textbox(label=f"Support {i+1}", value=LANG_DEFAULTS[i][0]) for i in range(5)]
        sup_labels = [gr.Radio(choices=[(LABEL_STR[0], 0), (LABEL_STR[1], 1)],
                                label=f"Label {i+1}", value=LANG_DEFAULTS[i][1]) for i in range(5)]

        def update_defaults(lang):
            sups = DEFAULT_SUPPORT.get(lang, DEFAULT_SUPPORT["ja"])
            return [sups[i][0] for i in range(5)] + [sups[i][1] for i in range(5)]

        language_sel.change(
            fn=update_defaults,
            inputs=[language_sel],
            outputs=sup_texts + sup_labels,
        )

    run_btn = gr.Button("Classify", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Retrieved cross-lingual examples**")
            ret_output = gr.Markdown()
        with gr.Column():
            gr.Markdown("**MAML adaptation quality**")
            adapt_output = gr.Markdown()

    with gr.Accordion("ARC internals (advanced)", open=False):
        arc_output = gr.Markdown()

    # ---- Example sentences ------------------------------------------------
    gr.Examples(
        examples=[
            ["このカメラは画質がとても良いです。", "ja"],
            ["全く使えません。最悪の製品です。",   "ja"],
            ["普通の品質です。",                   "ja"],
            ["这个产品质量非常好，我非常满意。",   "zh"],
            ["完全不值这个价钱，太失望了。",       "zh"],
        ],
        inputs=[query_input, language_sel],
        label="Quick examples — click any row to load",
    )

    # ---- Wire up button ---------------------------------------------------
    run_btn.click(
        fn=predict,
        inputs=[query_input, language_sel] + sup_texts + sup_labels,
        outputs=[pred_output, ret_output, adapt_output, arc_output],
    )

    gr.Markdown(
        """
---
**Model:** XLM-RoBERTa base + Adaptive Retrieval Controller + MAML meta-learner
**Training:** 24 epochs × 500 episodes, F1=0.97 (train), F1=0.89 (test)
**Index:** 320K cross-lingual vectors (EN/DE/ES/FR Amazon reviews)
        """
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)

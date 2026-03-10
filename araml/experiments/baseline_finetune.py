"""
baseline_finetune.py — Standard fine-tuning baseline on source languages,
then evaluate on target languages (cross-lingual transfer via XLM-R).
"""
import os
import sys
import yaml
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.encoder import TextEncoder
from utils.config_utils import get_dataset_config, load_multi_language_data, load_language_data
from utils.metrics import evaluate_few_shot, aggregate_episode_results
from utils.episode_sampler import EpisodeSampler


class FineTuneClassifier(nn.Module):
    def __init__(self, encoder: TextEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        emb = self.encoder(input_ids, attention_mask)
        return self.classifier(emb)

    def forward_text(self, texts, device, max_length=128):
        enc = self.encoder.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        return self.forward(input_ids, attention_mask)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_finetune(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds_cfg = get_dataset_config(config)
    num_classes = ds_cfg["num_classes"]
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline: Fine-tune] device={device}, dataset={ds_cfg['dataset_name']}")

    encoder = TextEncoder(model_name=config["model"]["encoder"])
    model = FineTuneClassifier(encoder, num_classes).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Load source training data
    all_records = load_multi_language_data(ds_cfg, ds_cfg["source_languages"], split="train")
    print(f"Training records: {len(all_records):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    batch_size = 16
    epochs = min(config["training"]["epochs"], 10)  # cap for fine-tuning

    model.train()
    for epoch in range(epochs):
        random.shuffle(all_records)
        total_loss, total_correct, total_seen = 0, 0, 0

        for i in tqdm(range(0, min(len(all_records), 5000), batch_size),
                      desc=f"FT Epoch {epoch+1}"):
            batch = all_records[i:i+batch_size]
            texts = [r["text"] for r in batch]
            labels = torch.tensor([r["label"] for r in batch]).to(device)

            logits = model.forward_text(texts, device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            total_correct += (logits.argmax(-1) == labels).sum().item()
            total_seen += len(batch)

        print(f"  Loss: {total_loss/total_seen:.4f} | Acc: {total_correct/total_seen:.4f}")

    # Save
    os.makedirs(config["training"]["save_dir"], exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(config["training"]["save_dir"], "baseline_finetune.pt"))

    # Evaluate on target languages with few-shot episodes
    model.eval()
    meta_cfg = config["meta_learning"]
    for lang in ds_cfg["target_languages"]:
        test_records = load_language_data(ds_cfg, lang, split="test")
        if not test_records:
            test_records = load_language_data(ds_cfg, lang, split="validation")
        if not test_records:
            print(f"Skipping {lang}")
            continue

        sampler = EpisodeSampler(test_records, meta_cfg["n_way"], meta_cfg["k_shot"], meta_cfg["query_size"])
        ep_iter = iter(sampler)
        accs = []
        for _ in range(200):
            ep = next(ep_iter)
            with torch.no_grad():
                logits = model.forward_text(ep["query_texts"], device)
                preds = logits.argmax(-1).cpu().tolist()
            acc = sum(p == t for p, t in zip(preds, ep["query_labels"])) / len(preds)
            accs.append(acc)

        results = aggregate_episode_results(accs)
        print(f"[{lang}] Fine-tune {meta_cfg['k_shot']}-shot | "
              f"Acc: {results['mean_accuracy']:.4f} +/- {results['95ci']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_finetune(args.config)

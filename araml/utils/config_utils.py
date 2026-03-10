"""
config_utils.py — Helpers to resolve dataset-specific fields from config.
"""
import os
import json


def get_dataset_config(config: dict) -> dict:
    """Return the active dataset sub-config with resolved fields."""
    name = config["dataset"]["name"]          # "xnli" or "sib200"
    ds_cfg = config["dataset"][name].copy()
    ds_cfg["dataset_name"] = name
    return ds_cfg


def load_language_data(ds_cfg: dict, lang: str, split: str = None) -> list:
    """Load processed JSON records for a language, optionally filtered by split."""
    pattern = ds_cfg["file_pattern"].replace("{lang}", lang)
    path = os.path.join(ds_cfg["data_dir"], pattern)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if split:
        records = [r for r in records if r["split"] == split]
    return records


def load_multi_language_data(ds_cfg: dict, languages: list, split: str = None) -> list:
    """Load and concatenate records across several languages."""
    all_records = []
    for lang in languages:
        recs = load_language_data(ds_cfg, lang, split)
        all_records.extend(recs)
        if recs:
            print(f"  Loaded {len(recs):,} records  [{lang}] (split={split})")
        else:
            print(f"  WARNING: no data for [{lang}] (split={split})")
    return all_records

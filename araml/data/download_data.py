"""
download_data.py — Download multilingual datasets from Hugging Face
"""
import os
from datasets import load_dataset

# Use datasets in native Parquet format
DATASETS = {
    "en": ("stanfordnlp/imdb", None, None),  # Load all splits
    "fr": ("allocine", None, None),
}


def download_amazon_reviews(save_dir: str = "data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    for lang, (dataset_name, config, split) in DATASETS.items():
        out_path = os.path.join(save_dir, f"amazon_{lang}")
        if os.path.exists(out_path):
            print(f"[{lang}] Already downloaded, skipping.")
            continue

        print(f"Downloading Reviews [{lang}]...")
        try:
            if config:
                ds = load_dataset(dataset_name, config)
            else:
                ds = load_dataset(dataset_name)
            
            # Sample from all splits - get balanced labels
            if hasattr(ds, 'keys'):
                sampled = {}
                for split_name in ds.keys():
                    split_data = ds[split_name]
                    # For binary classification, sample equally from both labels
                    if 'label' in split_data.column_names:
                        indices = []
                        for label_val in [0, 1]:
                            label_indices = [i for i, lbl in enumerate(split_data['label']) if lbl == label_val]
                            sample_size = min(2500, len(label_indices))
                            indices.extend(label_indices[:sample_size])
                        if indices:
                            sampled[split_name] = split_data.select(indices)
                    else:
                        sample_size = min(5000, len(split_data))
                        sampled[split_name] = split_data.select(range(sample_size))
                from datasets import DatasetDict
                ds = DatasetDict(sampled)
            else:
                # Sample balanced labels for single dataset
                indices = []
                for label_val in [0, 1]:
                    label_indices = [i for i, lbl in enumerate(ds['label']) if lbl == label_val]
                    sample_size = min(2500, len(label_indices))
                    indices.extend(label_indices[:sample_size])
                ds = ds.select(indices)
            
            ds.save_to_disk(out_path)
            if hasattr(ds, 'keys'):
                total = sum(len(ds[s]) for s in ds.keys())
            else:
                total = len(ds)
            print(f"  Saved {total} records → {out_path}")
        except Exception as e:
            print(f"  Failed to download [{lang}]: {e}")


if __name__ == "__main__":
    download_amazon_reviews()
    print("\nDatasets downloaded successfully.")

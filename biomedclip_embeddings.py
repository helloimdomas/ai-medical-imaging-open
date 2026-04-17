#!/usr/bin/env python3
"""
Extract BiomedCLIP image embeddings for the full Open-MELON dataset.

Usage:
    uv run python biomedclip_embeddings.py
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
LABEL_FILE = SCRIPT_DIR / "captions" / "captions_cleaned_labeled.jsonl"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_label_selection(label_path: Path = LABEL_FILE, include_spitz_as_nevus: bool = True):
    """Load benchmark indices from the Gemini-labeled caption file."""
    keep_as_benign = {"NEVUS"}
    if include_spitz_as_nevus:
        keep_as_benign.add("SPITZ_TUMOR")

    melanoma_set = set()
    benign_set = set()

    with open(label_path) as f:
        for line in f:
            row = json.loads(line)
            diagnosis = row.get("diagnosis")
            idx = row["index"]
            if diagnosis == "MELANOMA":
                melanoma_set.add(idx)
            elif diagnosis in keep_as_benign:
                benign_set.add(idx)

    target_indices = sorted(melanoma_set | benign_set)
    label_by_index = {idx: 1 if idx in melanoma_set else 0 for idx in target_indices}
    return target_indices, melanoma_set, label_by_index


def load_dataset():
    """Load the concatenated Open-MELON dataset."""
    from datasets import concatenate_datasets, load_dataset

    print("Loading Open-MELON dataset...")
    ds_raw = load_dataset("MartiHan/Open-MELON-VL-2.5K", cache_dir=CACHE_DIR)
    dataset = concatenate_datasets([ds_raw["train"], ds_raw["validation"], ds_raw["test"]])
    print(f"Dataset loaded: {len(dataset)} images")
    return dataset


def load_biomedclip(device: str):
    """Load BiomedCLIP model, image preprocessor, and tokenizer."""
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "Missing embedding dependencies. Run `uv sync --extra embeddings` "
            "or invoke this script with `uv run --extra embeddings python "
            "biomedclip_embeddings.py`."
        ) from exc

    print("Loading BiomedCLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, preprocess, tokenizer


def zero_shot_classify(model, preprocess, tokenizer, device, dataset, target_indices, melanoma_set, batch_size=8):
    """Run zero-shot melanoma vs nevus classification for BiomedCLIP."""
    import torch

    print("\n" + "=" * 60)
    print("ZERO-SHOT CLASSIFICATION")
    print("=" * 60)

    text_prompts = [
        "a histopathology image of melanoma",
        "a histopathology image of nevus",
    ]
    text_tokens = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    correct = 0
    melanoma_correct = 0
    melanoma_total = 0
    nevus_correct = 0
    nevus_total = 0

    print(f"Classifying {len(target_indices)} images zero-shot...")

    for start in range(0, len(target_indices), batch_size):
        batch_indices = target_indices[start:start + batch_size]
        batch_images = [preprocess(dataset[idx]["image"]) for idx in batch_indices]
        img_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            img_embeddings = model.encode_image(img_tensor)
            img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
            similarities = img_embeddings @ text_embeddings.T

        pred_ids = similarities.argmax(dim=1).tolist()

        for idx, pred_id in zip(batch_indices, pred_ids):
            pred_label = "melanoma" if pred_id == 0 else "nevus"
            true_label = "melanoma" if idx in melanoma_set else "nevus"

            if pred_label == true_label:
                correct += 1

            if true_label == "melanoma":
                melanoma_total += 1
                if pred_label == "melanoma":
                    melanoma_correct += 1
            else:
                nevus_total += 1
                if pred_label == "nevus":
                    nevus_correct += 1

        processed = start + len(batch_indices)
        if processed % 100 == 0 or processed == len(target_indices):
            print(f"  [{processed}/{len(target_indices)}] Acc: {correct / processed * 100:.1f}%")

    accuracy = correct / len(target_indices)
    sensitivity = melanoma_correct / melanoma_total if melanoma_total else 0.0
    specificity = nevus_correct / nevus_total if nevus_total else 0.0

    print("\nZero-shot Results:")
    print(f"  Accuracy: {accuracy * 100:.1f}%")
    print(f"  Sensitivity (melanoma): {sensitivity * 100:.1f}%")
    print(f"  Specificity (nevus): {specificity * 100:.1f}%")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def extract_embeddings(model, preprocess, device, dataset, target_indices=None, batch_size=16):
    """Extract normalized BiomedCLIP embeddings."""
    import torch

    embeddings = []
    idx_list = []
    if target_indices is None:
        target_indices = list(range(len(dataset)))

    print(f"Extracting embeddings for {len(target_indices)} images...")

    for start in range(0, len(target_indices), batch_size):
        batch_indices = target_indices[start:start + batch_size]
        batch_images = [preprocess(dataset[idx]["image"]) for idx in batch_indices]
        img_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            batch_embeddings = model.encode_image(img_tensor)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

        embeddings.append(batch_embeddings.cpu().numpy())
        idx_list.extend(batch_indices)

        processed = start + len(batch_indices)
        if processed % 50 == 0 or processed == len(target_indices):
            print(f"  [{processed}/{len(target_indices)}] extracted")

    return np.concatenate(embeddings, axis=0), np.array(idx_list)


def _backup_if_exists(path: Path):
    """Rename existing file with a timestamp suffix to avoid overwriting."""
    if path.exists():
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        path.rename(backup)
        print(f"Backed up existing file to: {backup}")


def save_embeddings(output_path: Path, X, indices, **extra_arrays):
    """Save cached embeddings to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _backup_if_exists(output_path)
    payload = {"X": X, "indices": indices}
    payload.update(extra_arrays)
    np.savez(output_path, **payload)
    print(f"Embeddings saved to: {output_path}")


def main():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Missing embedding dependencies. Run `uv sync --extra embeddings` "
            "or invoke this script with `uv run --extra embeddings python "
            "biomedclip_embeddings.py`."
        ) from exc

    parser = argparse.ArgumentParser(description="Extract BiomedCLIP embeddings")
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "embeddings" / "biomedclip_embeddings.npz"),
        help="Output .npz path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding extraction batch size",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset()
    print(f"Embedding full dataset: {len(dataset)} images")

    model, preprocess, _ = load_biomedclip(device)
    X, indices = extract_embeddings(
        model=model,
        preprocess=preprocess,
        device=device,
        dataset=dataset,
        batch_size=args.batch_size,
    )
    print(f"Embeddings shape: {X.shape}")
    save_embeddings(Path(args.output), X, indices)


if __name__ == "__main__":
    main()

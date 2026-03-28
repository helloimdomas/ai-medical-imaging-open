#!/usr/bin/env python3
"""
Extract MedSigLIP image embeddings for melanoma vs nevus classification.

Usage:
    uv run --env-file .env python medsiglip_embeddings.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
INDEX_FILE = SCRIPT_DIR / "indices" / "melanoma_nevus_indices.json"
MODEL_ID = "google/medsiglip-448"


def load_indices():
    """Load ground-truth melanoma/nevus indices."""
    with open(INDEX_FILE) as f:
        idx_data = json.load(f)

    target_indices = sorted(idx_data["melanoma"] + idx_data["nevus"])
    melanoma_set = set(idx_data["melanoma"])
    label_by_index = {idx: 1 if idx in melanoma_set else 0 for idx in target_indices}
    return target_indices, melanoma_set, label_by_index


def load_dataset():
    """Load the concatenated Open-MELON dataset."""
    from datasets import concatenate_datasets, load_dataset

    print("Loading Open-MELON dataset...")
    ds_raw = load_dataset("MartiHan/Open-MELON-VL-2.5K")
    dataset = concatenate_datasets([ds_raw["train"], ds_raw["validation"], ds_raw["test"]])
    print(f"Dataset loaded: {len(dataset)} images")
    return dataset


def load_medsiglip(device: str):
    """Load MedSigLIP processor and model."""
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Missing embedding dependencies. Run `uv sync --extra embeddings` "
            "or invoke this script with `uv run --extra embeddings --env-file .env "
            "python medsiglip_embeddings.py`."
        ) from exc

    print("Loading MedSigLIP...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return processor, model


def extract_embeddings(processor, model, device, dataset, target_indices, label_by_index, batch_size=2):
    """Extract normalized MedSigLIP vision embeddings."""
    import torch

    embeddings = []
    labels = []
    idx_list = []
    vision_model = model.vision_model

    print(f"Extracting embeddings for {len(target_indices)} images...")

    for start in range(0, len(target_indices), batch_size):
        batch_indices = target_indices[start:start + batch_size]
        batch_images = [dataset[idx]["image"] for idx in batch_indices]
        pixel_values = processor.image_processor(images=batch_images, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            outputs = vision_model(pixel_values=pixel_values)
            batch_embeddings = outputs.pooler_output
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

        embeddings.append(batch_embeddings.cpu().numpy())
        labels.extend(label_by_index[idx] for idx in batch_indices)
        idx_list.extend(batch_indices)

        processed = start + len(batch_indices)
        if processed % 50 == 0 or processed == len(target_indices):
            print(f"  [{processed}/{len(target_indices)}] extracted")

    return np.concatenate(embeddings, axis=0), np.array(labels), np.array(idx_list)


def save_embeddings(output_path: Path, X, y, indices):
    """Save cached embeddings to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X=X, y=y, indices=indices)
    print(f"Embeddings saved to: {output_path}")


def main():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Missing embedding dependencies. Run `uv sync --extra embeddings` "
            "or invoke this script with `uv run --extra embeddings --env-file .env "
            "python medsiglip_embeddings.py`."
        ) from exc

    parser = argparse.ArgumentParser(description="Extract MedSigLIP embeddings")
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "embeddings" / "medsiglip_embeddings.npz"),
        help="Output .npz path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Embedding extraction batch size",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset()
    target_indices, melanoma_set, label_by_index = load_indices()
    print(f"Total images: {len(target_indices)} (melanoma: {len(melanoma_set)}, nevus: {len(target_indices) - len(melanoma_set)})")

    processor, model = load_medsiglip(device)
    X, y, indices = extract_embeddings(
        processor=processor,
        model=model,
        device=device,
        dataset=dataset,
        target_indices=target_indices,
        label_by_index=label_by_index,
        batch_size=args.batch_size,
    )
    print(f"Embeddings shape: {X.shape}")
    save_embeddings(Path(args.output), X, y, indices)


if __name__ == "__main__":
    main()

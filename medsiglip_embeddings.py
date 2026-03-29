#!/usr/bin/env python3
"""
Extract MedSigLIP image embeddings for the full Open-MELON dataset.

Usage:
    uv run --env-file .env python medsiglip_embeddings.py
"""

import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
MODEL_ID = "google/medsiglip-448"


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


def extract_embeddings(processor, model, device, dataset, target_indices=None, batch_size=2):
    """Extract normalized MedSigLIP vision embeddings."""
    import torch

    embeddings = []
    idx_list = []
    vision_model = model.vision_model
    if target_indices is None:
        target_indices = list(range(len(dataset)))

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
        idx_list.extend(batch_indices)

        processed = start + len(batch_indices)
        if processed % 50 == 0 or processed == len(target_indices):
            print(f"  [{processed}/{len(target_indices)}] extracted")

    return np.concatenate(embeddings, axis=0), np.array(idx_list)


def save_embeddings(output_path: Path, X, indices, **extra_arrays):
    """Save cached embeddings to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"Embedding full dataset: {len(dataset)} images")

    processor, model = load_medsiglip(device)
    X, indices = extract_embeddings(
        processor=processor,
        model=model,
        device=device,
        dataset=dataset,
        batch_size=args.batch_size,
    )
    print(f"Embeddings shape: {X.shape}")
    save_embeddings(Path(args.output), X, indices)


if __name__ == "__main__":
    main()

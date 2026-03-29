#!/usr/bin/env python3
"""
Compatibility wrapper for the original BiomedCLIP experiment.

This now delegates embedding extraction to biomedclip_embeddings.py and shared
classifier training to train_embedding_classifier.py while preserving the
existing results and embedding output paths.
"""

import argparse
from pathlib import Path

import numpy as np

from biomedclip_embeddings import (
    extract_embeddings,
    load_biomedclip,
    load_dataset,
    load_label_selection,
    save_embeddings,
    zero_shot_classify,
)
from train_embedding_classifier import print_best_result, save_results, train_and_evaluate

SCRIPT_DIR = Path(__file__).parent


def main():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Missing embedding dependencies. Run `uv sync --extra embeddings` "
            "or invoke this script with `uv run --extra embeddings python "
            "biomedclip_classifier.py`."
        ) from exc

    parser = argparse.ArgumentParser(description="Run the full BiomedCLIP baseline")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding extraction batch size")
    parser.add_argument("--zero-shot-batch-size", type=int, default=8, help="Zero-shot inference batch size")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset()
    target_indices, melanoma_set, _ = load_label_selection()
    print(f"Label-selected images: {len(target_indices)} (melanoma: {len(melanoma_set)}, benign: {len(target_indices) - len(melanoma_set)})")

    model, preprocess, tokenizer = load_biomedclip(device)
    zero_shot_results = zero_shot_classify(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        dataset=dataset,
        target_indices=target_indices,
        melanoma_set=melanoma_set,
        batch_size=args.zero_shot_batch_size,
    )

    X, indices = extract_embeddings(
        model=model,
        preprocess=preprocess,
        device=device,
        dataset=dataset,
        target_indices=target_indices,
        batch_size=args.batch_size,
    )
    print(f"Embeddings shape: {X.shape}")

    embeddings_path = SCRIPT_DIR / "embeddings" / "biomedclip_embeddings.npz"
    save_embeddings(embeddings_path, X, indices)

    results = train_and_evaluate(X, np.array([1 if idx in melanoma_set else 0 for idx in indices]), indices)
    print_best_result(results)

    best_name, best_metrics = max(results.items(), key=lambda item: item[1]["test_accuracy"])
    print("\n" + "=" * 60)
    print("ZERO-SHOT vs SUPERVISED COMPARISON")
    print("=" * 60)
    print(f"Zero-shot:  Acc={zero_shot_results['accuracy'] * 100:.1f}%, Sens={zero_shot_results['sensitivity'] * 100:.1f}%, Spec={zero_shot_results['specificity'] * 100:.1f}%")
    print(f"Supervised: Acc={best_metrics['test_accuracy'] * 100:.1f}%, Sens={best_metrics['sensitivity'] * 100:.1f}%, Spec={best_metrics['specificity'] * 100:.1f}%")

    results_path = SCRIPT_DIR / "results" / "biomedclip_results.json"
    save_results(
        output_path=results_path,
        model_name="BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        X=X,
        results=results,
        extra_metrics={"zero_shot": zero_shot_results},
    )


if __name__ == "__main__":
    main()

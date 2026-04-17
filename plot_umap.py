"""Generate UMAP visualizations from precomputed embeddings."""

import json
import argparse
from pathlib import Path

import numpy as np
import umap
import matplotlib.pyplot as plt


def load_labels(jsonl_path: Path) -> dict[int, str]:
    """Return {index: diagnosis} for MELANOMA/NEVUS/SPITZ_TUMOR samples."""
    labels = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            dx = rec.get("diagnosis", "")
            if dx in ("MELANOMA", "NEVUS", "SPITZ_TUMOR"):
                labels[rec["index"]] = dx
    return labels


def plot(embedding_path: Path, labels: dict[int, str], title: str, out_stem: str):
    data = np.load(embedding_path)
    X, indices = data["X"], data["indices"]

    mask = np.isin(indices, list(labels.keys()))
    X_sub = X[mask]
    idx_sub = indices[mask]
    dx_sub = np.array([labels[i] for i in idx_sub])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    coords = reducer.fit_transform(X_sub)

    color_map = {"MELANOMA": "#cc0000", "NEVUS": "#38761d", "SPITZ_TUMOR": "#e69138"}
    label_map = {
        "MELANOMA": f"Melanoma (n={np.sum(dx_sub == 'MELANOMA')})",
        "NEVUS": f"Nevus (n={np.sum(dx_sub == 'NEVUS')})",
        "SPITZ_TUMOR": f"Spitz tumor (n={np.sum(dx_sub == 'SPITZ_TUMOR')})",
    }
    order = ["MELANOMA", "NEVUS", "SPITZ_TUMOR"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for dx in order:
        sel = dx_sub == dx
        ax.scatter(
            coords[sel, 0], coords[sel, 1],
            c=color_map[dx], label=label_map[dx],
            s=18, alpha=0.8, edgecolors="none",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=14, framealpha=0.9, markerscale=2.0)
    fig.tight_layout()

    out_dir = Path("poster")
    fig.savefig(out_dir / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / out_stem}.png and .pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", default="captions/captions_cleaned_labeled.jsonl",
        help="JSONL file with diagnosis labels",
    )
    args = parser.parse_args()

    labels = load_labels(Path(args.labels))

    plot(
        Path("embeddings/medsiglip_embeddings.npz"), labels,
        title="UMAP — MedSigLIP Embeddings", out_stem="umap_medsiglip",
    )
    plot(
        Path("embeddings/biomedclip_embeddings.npz"), labels,
        title="UMAP — BiomedCLIP Embeddings", out_stem="umap_biomedclip",
    )


if __name__ == "__main__":
    main()

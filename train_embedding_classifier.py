#!/usr/bin/env python3
"""
Train melanoma vs nevus classifiers on saved embedding arrays.

Usage:
    uv run python train_embedding_classifier.py \
      --embeddings-path embeddings/biomedclip_embeddings.npz \
      --model-name BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
      --results-path results/biomedclip_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).parent
DEFAULT_LABELS_PATH = SCRIPT_DIR / "captions" / "captions_cleaned_labeled.jsonl"


def load_embeddings(path: Path):
    """Load cached embeddings."""
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    indices = data["indices"]
    y = data["y"] if "y" in data.files else None
    return X, y, indices


def load_label_map(labels_path: Path, include_spitz_as_nevus: bool = True):
    """Load melanoma vs benign labels from the Gemini-labeled caption file."""
    keep_as_benign = {"NEVUS"}
    if include_spitz_as_nevus:
        keep_as_benign.add("SPITZ_TUMOR")

    label_by_index = {}
    diagnosis_by_index = {}
    counts = {"MELANOMA": 0, "NEVUS": 0, "SPITZ_TUMOR": 0}

    with open(labels_path) as f:
        for line in f:
            row = json.loads(line)
            diagnosis = row.get("diagnosis")
            idx = row["index"]

            if diagnosis == "MELANOMA":
                label_by_index[idx] = 1
                diagnosis_by_index[idx] = diagnosis
                counts["MELANOMA"] += 1
            elif diagnosis in keep_as_benign:
                label_by_index[idx] = 0
                diagnosis_by_index[idx] = diagnosis
                counts[diagnosis] += 1

    return label_by_index, diagnosis_by_index, counts


def filter_labeled_embeddings(X, indices, label_by_index, diagnosis_by_index):
    """Filter full-dataset embeddings down to the labeled benchmark subset."""
    selected_positions = [pos for pos, idx in enumerate(indices.tolist()) if int(idx) in label_by_index]
    selected_indices = indices[selected_positions]
    y = np.array([label_by_index[int(idx)] for idx in selected_indices], dtype=np.int64)
    diagnoses = np.array([diagnosis_by_index[int(idx)] for idx in selected_indices], dtype=object)
    return X[selected_positions], y, selected_indices, diagnoses


def get_classifiers(random_state=42):
    """Return the shared classifier set used across embedding models."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=True),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def train_and_evaluate(X, y, indices, test_size=0.2, random_state=42):
    """Train classifiers on a shared train/test split and report metrics."""
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFIERS")
    print("=" * 60)

    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X,
        y,
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Training set: {len(X_train)} (melanoma: {sum(y_train)}, benign: {len(y_train) - sum(y_train)})")
    print(f"Test set: {len(X_test)} (melanoma: {sum(y_test)}, benign: {len(y_test) - sum(y_test)})")

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, clf in get_classifiers(random_state=random_state).items():
        print(f"\n--- {name} ---")
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Sensitivity (melanoma): {sensitivity:.4f}")
        print(f"Specificity (benign): {specificity:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        results[name] = {
            "cv_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": test_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": cm.tolist(),
        }

    return results


def save_results(output_path: Path, model_name: str, X, results, extra_metrics=None):
    """Save classifier results in the repository's existing JSON format."""
    payload = {
        "model": model_name,
        "embedding_dim": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "supervised": results,
    }
    if extra_metrics:
        payload.update(extra_metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_best_result(results):
    """Print the best supervised classifier summary."""
    best_name, best_metrics = max(results.items(), key=lambda item: item[1]["test_accuracy"])
    print("\n" + "=" * 60)
    print("BEST SUPERVISED CLASSIFIER RESULTS")
    print("=" * 60)
    print(f"Best classifier: {best_name}")
    print(f"  Accuracy: {best_metrics['test_accuracy'] * 100:.1f}%")
    print(f"  Sensitivity: {best_metrics['sensitivity'] * 100:.1f}%")
    print(f"  Specificity: {best_metrics['specificity'] * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Train classifiers on cached embeddings")
    parser.add_argument("--embeddings-path", required=True, help="Input .npz embedding cache")
    parser.add_argument("--model-name", required=True, help="Model name to store in results JSON")
    parser.add_argument("--results-path", required=True, help="Output results JSON path")
    parser.add_argument(
        "--labels-path",
        default=str(DEFAULT_LABELS_PATH),
        help="JSONL file with Gemini diagnosis labels",
    )
    parser.add_argument(
        "--exclude-spitz",
        action="store_true",
        help="Exclude SPITZ_TUMOR instead of folding it into the benign class",
    )
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings_path)
    results_path = Path(args.results_path)
    labels_path = Path(args.labels_path)

    print(f"Loading embeddings from: {embeddings_path}")
    X, y_cached, indices = load_embeddings(embeddings_path)
    print(f"Cached embeddings shape: {X.shape}")

    extra_metrics = None
    if labels_path.exists():
        print(f"Loading labels from: {labels_path}")
        label_by_index, diagnosis_by_index, counts = load_label_map(
            labels_path,
            include_spitz_as_nevus=not args.exclude_spitz,
        )
        X, y, indices, diagnoses = filter_labeled_embeddings(X, indices, label_by_index, diagnosis_by_index)
        print(
            f"Filtered labeled subset: {X.shape[0]} "
            f"(melanoma: {sum(y == 1)}, benign: {sum(y == 0)})"
        )
        extra_metrics = {
            "labels_path": str(labels_path),
            "label_counts": {
                "MELANOMA": int(counts["MELANOMA"]),
                "NEVUS": int(counts["NEVUS"]),
                "SPITZ_TUMOR": int(counts["SPITZ_TUMOR"]),
            },
            "spitz_treated_as_benign": not args.exclude_spitz,
        }
    elif y_cached is not None:
        print("No label file found. Falling back to cached y labels.")
        y = y_cached
    else:
        raise FileNotFoundError(
            f"No labels file found at {labels_path} and the embedding cache does not contain y labels."
        )

    results = train_and_evaluate(X, y, indices)
    print_best_result(results)
    save_results(results_path, args.model_name, X, results, extra_metrics=extra_metrics)


if __name__ == "__main__":
    main()

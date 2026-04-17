#!/usr/bin/env python3
"""
balanced_accuracy.py - Evaluate classifiers with balanced class distribution.

The dataset is imbalanced toward melanoma. This script evaluates models on a
balanced subset by taking all benign cases (NEVUS + optional SPITZ_TUMOR) and
sampling the same number of melanoma cases, then averaging across repeated
trials.

Usage:
    python balanced_accuracy.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

from train_embedding_classifier import filter_labeled_embeddings, get_classifiers, load_label_map

SCRIPT_DIR = Path(__file__).parent
RANDOM_SEED = 42
N_TRIALS = 10  # Number of random sampling trials to average
LABELS_PATH = SCRIPT_DIR / "captions" / "captions_cleaned_labeled.jsonl"
INCLUDE_SPITZ_AS_NEVUS = True
OUTPUT_PATH = SCRIPT_DIR / "results" / "balanced_accuracy.json"

from utils import backup_if_exists

def load_data():
    biomedclip = np.load(SCRIPT_DIR / "embeddings" / "biomedclip_embeddings.npz", allow_pickle=True)
    medsiglip = np.load(SCRIPT_DIR / "embeddings" / "medsiglip_embeddings.npz", allow_pickle=True)

    X_biomedclip = biomedclip["X"]
    indices = biomedclip["indices"]
    X_medsiglip = medsiglip["X"]

    if not np.array_equal(indices, medsiglip["indices"]):
        raise ValueError("BiomedCLIP and MedSigLIP embeddings do not share the same indices")

    label_by_index, diagnosis_by_index, _ = load_label_map(
        LABELS_PATH,
        include_spitz_as_nevus=INCLUDE_SPITZ_AS_NEVUS,
    )
    X_biomedclip, y, indices, _ = filter_labeled_embeddings(X_biomedclip, indices, label_by_index, diagnosis_by_index)
    X_medsiglip, _, medsiglip_indices, _ = filter_labeled_embeddings(X_medsiglip, medsiglip["indices"], label_by_index, diagnosis_by_index)

    if not np.array_equal(indices, medsiglip_indices):
        raise ValueError("Filtered BiomedCLIP and MedSigLIP embeddings do not align")
    
    # Load MedGemma predictions
    medgemma_preds = {}
    caption_file = SCRIPT_DIR / "captions" / "binary_choice" / "captions.jsonl"
    with open(caption_file) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["caption_gen"].lower()
            
            # Apply same keyword classification logic
            has_mel = "melanoma" in text
            has_nev = "nevus" in text or "nevi" in text or "benign" in text
            
            if has_mel and not has_nev:
                pred = 1  # melanoma
            elif has_nev and not has_mel:
                pred = 0  # nevus
            elif has_mel and has_nev:
                pred = 1  # melanoma priority
            else:
                pred = 1  # default to melanoma (unknown)
            
            medgemma_preds[entry["index"]] = pred
    
    return X_biomedclip, X_medsiglip, y, indices, medgemma_preds


def balanced_sample(X, y, indices, rng):
    """Take all benign cases and sample the same number of melanoma cases."""
    melanoma_mask = y == 1
    nevus_mask = y == 0
    
    melanoma_indices = np.where(melanoma_mask)[0]
    nevus_indices = np.where(nevus_mask)[0]
    
    n_samples = len(nevus_indices)
    if len(melanoma_indices) < n_samples:
        raise ValueError("Not enough melanoma cases to match the full benign set.")

    mel_sample = rng.choice(melanoma_indices, size=n_samples, replace=False)
    nev_sample = nevus_indices
    
    selected = np.concatenate([mel_sample, nev_sample])
    rng.shuffle(selected)
    
    return X[selected], y[selected], indices[selected]


def evaluate_balanced(n_trials=N_TRIALS):
    """Evaluate with balanced sampling over multiple trials."""
    print("Loading data...")
    X_biomedclip, X_medsiglip, y, indices, medgemma_preds = load_data()
    
    benign_label = "benign (nevus + Spitz)" if INCLUDE_SPITZ_AS_NEVUS else "benign (nevus only)"
    print(f"\nOriginal distribution: {sum(y == 1)} melanoma, {sum(y == 0)} {benign_label}")
    
    # Store results across trials
    results = {
        "MedGemma (binary_choice)": [],
        "BiomedCLIP + LogReg": [],
        "BiomedCLIP + SVM": [],
        "BiomedCLIP + RandomForest": [],
        "MedSigLIP + LogReg": [],
        "MedSigLIP + SVM": [],
        "MedSigLIP + RandomForest": [],
    }
    
    for trial in range(n_trials):
        rng = np.random.default_rng(RANDOM_SEED + trial)
        
        # Balanced sample: all benign + matched melanoma sample
        X_bio_bal, y_bal, idx_bal = balanced_sample(X_biomedclip, y, indices, rng)
        X_med_bal, _, _ = balanced_sample(X_medsiglip, y, indices, np.random.default_rng(RANDOM_SEED + trial))
        
        # Split 80/20
        n = len(y_bal)
        perm = rng.permutation(n)
        n_train = int(0.8 * n)
        
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        
        X_bio_train, X_bio_test = X_bio_bal[train_idx], X_bio_bal[test_idx]
        X_med_train, X_med_test = X_med_bal[train_idx], X_med_bal[test_idx]
        y_train, y_test = y_bal[train_idx], y_bal[test_idx]
        idx_test = idx_bal[test_idx]
        
        # MedGemma accuracy on test set
        mg_preds = [medgemma_preds.get(int(idx), 1) for idx in idx_test]
        mg_acc = accuracy_score(y_test, mg_preds)
        results["MedGemma (binary_choice)"].append(mg_acc)
        
        for name, clf in get_classifiers(random_state=42).items():
            biomedclip_name = name.replace("LogisticRegression", "LogReg")
            biomedclip_name = f"BiomedCLIP + {biomedclip_name.replace('SVM (RBF)', 'SVM')}"
            medsiglip_name = biomedclip_name.replace("BiomedCLIP", "MedSigLIP")

            clf.fit(X_bio_train, y_train)
            y_pred = clf.predict(X_bio_test)
            acc = accuracy_score(y_test, y_pred)
            results[biomedclip_name].append(acc)

            clf.fit(X_med_train, y_train)
            y_pred = clf.predict(X_med_test)
            acc = accuracy_score(y_test, y_pred)
            results[medsiglip_name].append(acc)
    
    # Print results
    print(f"\nBALANCED ACCURACY (averaged over {n_trials} trials)")
    print(f"Each trial uses all {sum(y == 0)} benign samples and {sum(y == 0)} sampled melanomas")
    
    summary = {}
    for name, accs in results.items():
        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        print(f"{name:30s}: {mean_acc:.1f}% (+/- {std_acc:.1f}%)")
        summary[name] = {"mean": mean_acc, "std": std_acc}
    
    # Compare to original imbalanced
    print("\nCOMPARISON: Original (imbalanced) vs Balanced")
    
    # Original MedGemma
    orig_mg_preds = [medgemma_preds.get(int(idx), 1) for idx in indices]
    orig_mg_acc = accuracy_score(y, orig_mg_preds) * 100
    print(f"MedGemma original:     {orig_mg_acc:.1f}%")
    print(f"MedGemma balanced:     {summary['MedGemma (binary_choice)']['mean']:.1f}%")
    print(f"  Difference: {summary['MedGemma (binary_choice)']['mean'] - orig_mg_acc:+.1f}%")
    
    print()
    
    # BiomedCLIP RF original (from saved results)
    try:
        with open(SCRIPT_DIR / "results" / "biomedclip_results.json") as f:
            orig_results = json.load(f)
        orig_rf_acc = orig_results["supervised"]["RandomForest"]["test_accuracy"] * 100
        print(f"BiomedCLIP RF original:  {orig_rf_acc:.1f}%")
        print(f"BiomedCLIP RF balanced:  {summary['BiomedCLIP + RandomForest']['mean']:.1f}%")
        print(f"  Difference: {summary['BiomedCLIP + RandomForest']['mean'] - orig_rf_acc:+.1f}%")
    except:
        pass

    try:
        with open(SCRIPT_DIR / "results" / "medsiglip_results.json") as f:
            orig_results = json.load(f)
        orig_svm_acc = orig_results["supervised"]["SVM (RBF)"]["test_accuracy"] * 100
        print()
        print(f"MedSigLIP SVM original:   {orig_svm_acc:.1f}%")
        print(f"MedSigLIP SVM balanced:   {summary['MedSigLIP + SVM']['mean']:.1f}%")
        print(f"  Difference: {summary['MedSigLIP + SVM']['mean'] - orig_svm_acc:+.1f}%")
    except:
        pass
    
    # Save results
    output = {
        "n_trials": n_trials,
        "samples_per_class": int(sum(y == 0)),
        "sampling_strategy": "all_benign_plus_equal_random_melanoma",
        "balanced_results": summary,
    }
    
    backup_if_exists(OUTPUT_PATH)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_PATH}")
    
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate with all benign + matched melanoma sampling")
    parser.add_argument("--labels-path", default=str(SCRIPT_DIR / "captions" / "captions_cleaned_labeled.jsonl"))
    parser.add_argument("--exclude-spitz", action="store_true", help="Exclude SPITZ_TUMOR from the benign class")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results" / "balanced_accuracy.json"))
    args = parser.parse_args()

    LABELS_PATH = Path(args.labels_path)
    INCLUDE_SPITZ_AS_NEVUS = not args.exclude_spitz
    OUTPUT_PATH = Path(args.output)

    summary = evaluate_balanced(n_trials=args.n_trials)

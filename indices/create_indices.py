#!/usr/bin/env python3
"""
create_indices.py - Conservative index creation for melanoma vs nevus classification

This script creates benchmark indices from Open-MELON captions. Since the dataset
does not provide explicit class labels for this task, we derive a conservative
subset and prefer precision over recall.

RULES:
1. Match diagnosis keywords as whole words, not raw substrings.
2. Drop captions that mention both melanoma and nevus terms.
3. Drop captions that mention diagnosis terms in negated or exclusion contexts.
4. Drop captions that contain differential-diagnosis language.
5. Only keep captions with a single clear, positive diagnosis signal.

Usage:
    python create_indices.py [--output OUTPUT_FILE] [--review-output REVIEW_FILE]
"""

import json
import argparse
import re
from datetime import datetime
from pathlib import Path


MELANOMA_RE = re.compile(r"\bmelanoma\b", re.IGNORECASE)
NEVUS_RE = re.compile(r"\b(?:nevus|nevi|naevus|naevi)\b", re.IGNORECASE)

NEGATION_CUES = [
    "exclude",
    "excluding",
    "excluded",
    "rule out",
    "ruled out",
    "no evidence of",
    "no sign of",
    "without",
    "absence of",
    "negative for",
]
DIFFERENTIAL_CUES = [
    "differential diagnosis",
    "differential diagnoses",
    "differential",
]


def find_positive_mentions(text: str, pattern: re.Pattern) -> list[re.Match]:
    """Return diagnosis mentions that are not obviously negated nearby."""
    positive = []
    for match in pattern.finditer(text):
        left = text[max(0, match.start() - 40):match.start()].lower()

        # Negation and exclusion are only treated as label-negating when they
        # appear before the diagnosis term. This avoids false exclusions like
        # "nodular melanoma ... without epidermal involvement".
        if any(cue in left for cue in NEGATION_CUES):
            continue
        positive.append(match)
    return positive


def classify_caption(caption: str) -> tuple[str, list[str]]:
    """Classify a caption into a benchmark bucket plus review reasons."""
    text = caption.lower()
    reasons = []

    has_melanoma_word = bool(MELANOMA_RE.search(text))
    has_nevus_word = bool(NEVUS_RE.search(text))

    if any(cue in text for cue in DIFFERENTIAL_CUES):
        reasons.append("differential_language")

    melanoma_mentions = find_positive_mentions(text, MELANOMA_RE)
    nevus_mentions = find_positive_mentions(text, NEVUS_RE)

    if has_melanoma_word and not melanoma_mentions:
        reasons.append("melanoma_negated_or_excluded")
    if has_nevus_word and not nevus_mentions:
        reasons.append("nevus_negated_or_excluded")

    has_positive_melanoma = bool(melanoma_mentions)
    has_positive_nevus = bool(nevus_mentions)

    if has_positive_melanoma and has_positive_nevus:
        reasons.append("both_positive_terms")
        return "excluded_both", reasons
    if reasons:
        return "excluded_uncertain", reasons
    if has_positive_melanoma:
        return "melanoma", reasons
    if has_positive_nevus:
        return "nevus", reasons
    return "excluded_neither", reasons


def write_review_file(path: Path, review_rows: list[dict]) -> None:
    """Write review candidates as JSONL."""
    with open(path, "w") as f:
        for row in review_rows:
            f.write(json.dumps(row) + "\n")


def create_indices(output_path: Path, review_output_path: Path, dry_run: bool = False) -> dict:
    """Create melanoma/nevus indices from Open-MELON dataset."""
    from datasets import load_dataset
    
    print("Loading Open-MELON dataset (all splits)...")
    ds = load_dataset(
        "MartiHan/Open-MELON-VL-2.5K",
        cache_dir="~/.cache/huggingface"
    )
    
    melanoma_indices = []
    nevus_indices = []
    excluded_both = []  # Cases with both keywords (ambiguous)
    excluded_uncertain = []
    review_rows = []
    
    # Track statistics
    stats = {
        "train": {"melanoma": 0, "nevus": 0, "both": 0, "uncertain": 0, "neither": 0},
        "validation": {"melanoma": 0, "nevus": 0, "both": 0, "uncertain": 0, "neither": 0},
        "test": {"melanoma": 0, "nevus": 0, "both": 0, "uncertain": 0, "neither": 0},
    }
    
    # Process each split with global index offset
    offset = 0
    split_info = {}
    
    for split_name in ["train", "validation", "test"]:
        split = ds[split_name]
        split_info[split_name] = {
            "start_idx": offset,
            "end_idx": offset + len(split) - 1,
            "size": len(split),
        }
        
        for i, sample in enumerate(split):
            global_idx = offset + i

            bucket, reasons = classify_caption(sample["caption"])

            if bucket == "melanoma":
                melanoma_indices.append(global_idx)
                stats[split_name]["melanoma"] += 1
            elif bucket == "nevus":
                nevus_indices.append(global_idx)
                stats[split_name]["nevus"] += 1
            elif bucket == "excluded_both":
                excluded_both.append(global_idx)
                stats[split_name]["both"] += 1
            elif bucket == "excluded_uncertain":
                excluded_uncertain.append(global_idx)
                stats[split_name]["uncertain"] += 1
                review_rows.append(
                    {
                        "index": global_idx,
                        "split": split_name,
                        "reasons": reasons,
                        "caption": sample["caption"],
                    }
                )
            else:
                stats[split_name]["neither"] += 1
        
        offset += len(split)
    
    # Build output structure
    result = {
        "melanoma": sorted(melanoma_indices),
        "nevus": sorted(nevus_indices),
        "excluded_both": sorted(excluded_both),
        "excluded_uncertain": sorted(excluded_uncertain),
        "metadata": {
            "created": datetime.now().isoformat(),
            "script": "create_indices.py",
            "dataset": "MartiHan/Open-MELON-VL-2.5K",
            "total_dataset_size": offset,
            "splits": split_info,
            "classification_policy": "conservative_keyword_with_negation_and_differential_filtering",
            "counts": {
                "melanoma": len(melanoma_indices),
                "nevus": len(nevus_indices),
                "excluded_both": len(excluded_both),
                "excluded_uncertain": len(excluded_uncertain),
                "total_classified": len(melanoma_indices) + len(nevus_indices),
            },
            "per_split_stats": stats,
        },
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("INDEX CREATION SUMMARY")
    print("=" * 60)
    print(f"Total dataset: {offset} images")
    print(f"Melanoma (only): {len(melanoma_indices)}")
    print(f"Nevus (only): {len(nevus_indices)}")
    print(f"Excluded (both keywords): {len(excluded_both)}")
    print(f"Excluded (uncertain/negated): {len(excluded_uncertain)}")
    print(f"Excluded (no keywords): {offset - len(melanoma_indices) - len(nevus_indices) - len(excluded_both) - len(excluded_uncertain)}")
    print(f"Total classified: {len(melanoma_indices) + len(nevus_indices)}")
    
    print("\nPer-split breakdown:")
    for split_name, s in stats.items():
        info = split_info[split_name]
        print(f"  {split_name}: {info['size']} images "
              f"(mel={s['melanoma']}, nev={s['nevus']}, both={s['both']}, uncertain={s['uncertain']}, excl={s['neither']})")
    
    if not dry_run:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        write_review_file(review_output_path, review_rows)
        print(f"\nSaved to: {output_path}")
        print(f"Saved review candidates to: {review_output_path}")
    else:
        print("\n[DRY RUN - not saved]")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create melanoma/nevus indices from Open-MELON dataset"
    )
    parser.add_argument(
        "--output",
        default="melanoma_nevus_indices.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--review-output",
        default="review_candidates.jsonl",
        help="Output JSONL for uncertain captions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without saving"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    review_output_path = script_dir / args.review_output
    
    create_indices(output_path, review_output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

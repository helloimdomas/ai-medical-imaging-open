#!/usr/bin/env python3
"""
Analyze pathology themes in embedding-model failure buckets.

Usage:
    uv run python analyze_failure_themes.py
"""

import json
import re
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
INPUT_PATH = SCRIPT_DIR / "results" / "svm_failure_comparison.json"
OUTPUT_PATH = SCRIPT_DIR / "results" / "failure_theme_analysis.json"

from utils import backup_if_exists

THEMES = {
    "acral_or_lentiginous": [r"\bacral\b", r"lentigin", r"\bmatrix\b", r"\bnail\b", r"\bpalm\b", r"\bsole\b"],
    "mucosal_or_special_site": [r"vulvar", r"conjunct", r"eyelid", r"buccal", r"\boral\b", r"mucosa", r"labia", r"mucosal"],
    "nodal_or_metastatic": [r"sentinel lymph node", r"\blymph node\b", r"metast", r"\bspleen\b"],
    "spitz_related": [r"\bspitz\b", r"alk fusion"],
    "blue_nevus_related": [r"blue nevus", r"deep penetrating nevus", r"pigmented epithelioid", r"dumbbell-shaped"],
    "in_situ_or_lentigo_maligna": [r"in situ", r"lentigo maligna", r"\blmm\b"],
    "atypical_or_dysplastic_nevus_context": [r"dysplastic nevus", r"atypical cell nodule", r"atypical features", r"severe atypia", r"moderate to severe atypia"],
    "spindle_or_desmoplastic": [r"spindle", r"desmoplastic"],
    "pigment_heavy": [r"heavily pigmented", r"melanin", r"pigment", r"melanoph"],
    "regression_or_inflammation": [r"regression", r"inflamm", r"melanophages"],
}


def classify_themes(text: str) -> list[str]:
    hits = []
    for theme, patterns in THEMES.items():
        if any(re.search(pattern, text) for pattern in patterns):
            hits.append(theme)
    return hits


def summarize_bucket(rows: list[dict]) -> dict:
    theme_counts = Counter()
    quality_counts = Counter(row["fig_quality"] for row in rows)
    true_label_counts = Counter(row["true_label"] for row in rows)
    diagnosis_counts = Counter(row["diagnosis"] for row in rows)

    classified_rows = []
    no_theme = []

    for row in rows:
        text = f"{row['caption']} {row['diagnosis']}".lower()
        hits = classify_themes(text)
        if not hits:
            no_theme.append(int(row["index"]))
        for theme in hits:
            theme_counts[theme] += 1

        classified_rows.append(
            {
                "index": int(row["index"]),
                "true_label": row["true_label"],
                "diagnosis": row["diagnosis"],
                "fig_quality": row["fig_quality"],
                "themes": hits,
                "caption": row["caption"],
                "biomedclip_pred": row["biomedclip_pred"],
                "medsiglip_pred": row["medsiglip_pred"],
            }
        )

    theme_percentages = {
        theme: round(count / len(rows) * 100, 1) for theme, count in theme_counts.most_common()
    }

    return {
        "count": len(rows),
        "true_label_counts": dict(true_label_counts),
        "diagnosis_counts": dict(diagnosis_counts),
        "fig_quality_counts": dict(quality_counts),
        "theme_counts": dict(theme_counts.most_common()),
        "theme_percentages": theme_percentages,
        "no_theme_indices": no_theme,
        "rows": classified_rows,
    }


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    output = {
        "input_path": str(INPUT_PATH),
        "themes": THEMES,
        "buckets": {},
    }

    for bucket in [
        "both_wrong",
        "medsiglip_fail_biomedclip_ok",
        "biomedclip_fail_medsiglip_ok",
    ]:
        output["buckets"][bucket] = summarize_bucket(data[bucket])

    backup_if_exists(OUTPUT_PATH)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved theme analysis to: {OUTPUT_PATH}")
    for bucket, summary in output["buckets"].items():
        print()
        print(bucket, f"(n={summary['count']})")
        for theme, count in list(summary["theme_counts"].items())[:6]:
            pct = summary["theme_percentages"][theme]
            print(f"  {theme:32s} {count:2d} ({pct:4.1f}%)")


if __name__ == "__main__":
    main()

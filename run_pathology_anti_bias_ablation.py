#!/usr/bin/env python3
"""
Run a multi-model pathology prompt ablation with non-overwriting output roots.

Usage:
    uv run --extra captioning python run_pathology_anti_bias_ablation.py
"""

import argparse
import base64
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from train_embedding_classifier import load_label_map

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "pathology_anti_bias_ablation.yaml"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "pathology_prompt_ablation"
DEFAULT_LABELS_PATH = SCRIPT_DIR / "captions" / "captions_cleaned_labeled.jsonl"


@dataclass
class RunSpec:
    subset_id: str
    subset_description: str
    model: str
    prompt_id: str
    prompt_description: str
    parameter_id: str
    parameter_description: str
    prompt: str
    options: dict
    target_indices: list[int]

    @property
    def run_id(self) -> str:
        return f"{self.prompt_id}__{self.parameter_id}"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def classify_caption(text: str) -> str:
    text = text.lower()
    has_mel = "melanoma" in text
    has_nev = any(token in text for token in ["nevus", "nevi", "naevus", "naevi", "benign nevus"])

    if has_nev and not has_mel:
        return "benign"
    if has_mel and not has_nev:
        return "melanoma"
    if has_mel and has_nev:
        diag_match = re.search(r"final diagnosis:\s*(melanoma|benign nevus|nevus)", text)
        if diag_match:
            return "benign" if "nevus" in diag_match.group(1) else "melanoma"
        return "melanoma"
    return "unknown"


def load_dataset():
    from datasets import concatenate_datasets, load_dataset

    raw = load_dataset("MartiHan/Open-MELON-VL-2.5K")
    return concatenate_datasets([raw["train"], raw["validation"], raw["test"]])


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_label_lookup(labels_path: Path):
    label_by_index, diagnosis_by_index, _ = load_label_map(labels_path, include_spitz_as_nevus=True)
    true_label_by_index = {
        int(idx): ("melanoma" if label == 1 else "benign") for idx, label in label_by_index.items()
    }
    return true_label_by_index, diagnosis_by_index


def resolve_subset(subset_id: str, subset_cfg: dict, true_label_by_index: dict, diagnosis_by_index: dict) -> tuple[list[int], str]:
    subset_type = subset_cfg["type"]

    if subset_type == "diagnosis_first_n":
        diagnosis = subset_cfg["diagnosis"]
        n = int(subset_cfg["n"])
        indices = sorted(idx for idx, d in diagnosis_by_index.items() if d == diagnosis)[:n]
        return indices, f"{diagnosis} first {n}"

    if subset_type == "failure_bucket":
        source = SCRIPT_DIR / subset_cfg["source"]
        bucket = subset_cfg["bucket"]
        with open(source) as f:
            data = json.load(f)
        indices = [int(row["index"]) for row in data[bucket]]
        return indices, f"{bucket} from {source.name}"

    raise ValueError(f"Unsupported subset type: {subset_type}")


def build_runs(config: dict, true_label_by_index: dict, diagnosis_by_index: dict) -> list[RunSpec]:
    runs = []
    for subset_id, subset_cfg in config["subsets"].items():
        indices, subset_description = resolve_subset(subset_id, subset_cfg, true_label_by_index, diagnosis_by_index)
        for model in config["models"]:
            for prompt_id, prompt_cfg in config["prompts"].items():
                for parameter_id, parameter_cfg in config["parameter_sets"].items():
                    runs.append(
                        RunSpec(
                            subset_id=subset_id,
                            subset_description=subset_description,
                            model=model,
                            prompt_id=prompt_id,
                            prompt_description=prompt_cfg.get("description", ""),
                            parameter_id=parameter_id,
                            parameter_description=parameter_cfg.get("description", ""),
                            prompt=prompt_cfg["prompt"],
                            options=parameter_cfg["options"],
                            target_indices=indices,
                        )
                    )
    return runs


def summarize_rows(rows: list[dict]) -> dict:
    counts = {"benign": 0, "melanoma": 0, "unknown": 0}
    true_counts = {"benign": 0, "melanoma": 0}
    correct = 0

    for row in rows:
        counts[row["pred_label"]] = counts.get(row["pred_label"], 0) + 1
        true_counts[row["true_label"]] = true_counts.get(row["true_label"], 0) + 1
        if row["pred_label"] == row["true_label"]:
            correct += 1

    total = len(rows)
    return {
        "total": total,
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "pred_counts": counts,
        "true_counts": true_counts,
        "correct": correct,
    }


def main():
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "Missing captioning dependencies. Run `uv sync --extra captioning` "
            "or invoke this script with `uv run --extra captioning python "
            "run_pathology_anti_bias_ablation.py`."
        ) from exc

    parser = argparse.ArgumentParser(description="Run multi-model pathology prompt ablations")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    parser.add_argument("--labels-path", default=str(DEFAULT_LABELS_PATH), help="Gemini labels JSONL path")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="Root output directory")
    parser.add_argument("--keep-alive", default="0s", help="Ollama keep_alive setting")
    parser.add_argument("--limit-runs", type=int, default=-1, help="Optional run limit for testing")
    args = parser.parse_args()

    config_path = Path(args.config)
    results_root = Path(args.results_root)
    timestamp_root = results_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_root.mkdir(parents=True, exist_ok=True)

    true_label_by_index, diagnosis_by_index = build_label_lookup(Path(args.labels_path))
    dataset = load_dataset()
    runs = build_runs(load_config(config_path), true_label_by_index, diagnosis_by_index)
    if args.limit_runs > 0:
        runs = runs[:args.limit_runs]

    print(f"Output root: {timestamp_root}")
    print(f"Total runs: {len(runs)}")

    all_summaries = []

    for run in runs:
        run_dir = (
            timestamp_root
            / run.subset_id
            / sanitize_name(run.model)
            / run.run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        caption_path = run_dir / "captions.jsonl"
        summary_path = run_dir / "summary.json"

        print("\n" + "=" * 60)
        print(f"SUBSET {run.subset_id} | MODEL {run.model}")
        print(f"RUN {run.run_id}")
        print(f"Prompt: {run.prompt_description}")
        print(f"Params: {run.parameter_description} | {run.options}")
        print(f"Indices ({len(run.target_indices)}): {run.target_indices}")
        print("=" * 60)

        rows = []
        with open(caption_path, "w") as f:
            for idx in run.target_indices:
                sample = dataset[idx]
                img_b64 = image_to_base64(sample["image"])
                response = ollama.chat(
                    model=run.model,
                    messages=[{"role": "user", "content": run.prompt, "images": [img_b64]}],
                    options=run.options,
                    keep_alive=args.keep_alive,
                )
                caption_gen = response.message.content.strip()
                pred_label = classify_caption(caption_gen)
                row = {
                    "index": idx,
                    "pmc_id": sample["pmc_id"],
                    "prompt_id": run.prompt_id,
                    "parameter_id": run.parameter_id,
                    "true_label": true_label_by_index.get(int(idx), "unknown"),
                    "diagnosis": diagnosis_by_index.get(int(idx), "unknown"),
                    "caption_gen": caption_gen,
                    "pred_label": pred_label,
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                print(f"idx={idx} true={row['true_label']} pred={pred_label}")

        summary = summarize_rows(rows)
        summary.update(
            {
                "timestamp": datetime.now().isoformat(),
                "subset_id": run.subset_id,
                "subset_description": run.subset_description,
                "model": run.model,
                "run_id": run.run_id,
                "prompt_id": run.prompt_id,
                "prompt_description": run.prompt_description,
                "parameter_id": run.parameter_id,
                "parameter_description": run.parameter_description,
                "options": run.options,
                "target_indices": run.target_indices,
            }
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        all_summaries.append(summary)
        print(f"Accuracy: {summary['accuracy']:.1f}%")
        print(f"Predictions: {summary['pred_counts']}")

    with open(timestamp_root / "leaderboard.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nSaved results to:")
    print(timestamp_root)


if __name__ == "__main__":
    main()

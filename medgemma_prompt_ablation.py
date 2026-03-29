#!/usr/bin/env python3
"""
Run a MedGemma prompt/decoding ablation on a fixed image subset.

Usage:
    uv run --extra captioning python medgemma_prompt_ablation.py
"""

import argparse
import base64
import io
import itertools
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "medgemma_ablation.yaml"


@dataclass
class AblationRun:
    run_id: str
    prompt_id: str
    prompt_description: str
    parameter_id: str
    parameter_description: str
    prompt: str
    options: dict


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def classify_caption(text: str) -> str:
    """Map free-form output onto melanoma/nevus/unknown."""
    text = text.lower()
    has_mel = "melanoma" in text
    has_nev = any(token in text for token in ["nevus", "nevi", "naevus", "naevi", "benign nevus"])

    if has_nev and not has_mel:
        return "nevus"
    if has_mel and not has_nev:
        return "melanoma"
    if has_mel and has_nev:
        diag_match = re.search(r"final diagnosis:\s*(melanoma|benign nevus|nevus)", text)
        if diag_match:
            return "nevus" if "nevus" in diag_match.group(1) else "melanoma"
        return "melanoma"
    return "unknown"


def build_runs(config: dict) -> list[AblationRun]:
    runs = []
    for prompt_id, prompt_cfg in config["prompts"].items():
        for parameter_id, parameter_cfg in config["parameter_sets"].items():
            run_id = f"{prompt_id}__{parameter_id}"
            runs.append(
                AblationRun(
                    run_id=run_id,
                    prompt_id=prompt_id,
                    prompt_description=prompt_cfg.get("description", ""),
                    parameter_id=parameter_id,
                    parameter_description=parameter_cfg.get("description", ""),
                    prompt=prompt_cfg["prompt"],
                    options=parameter_cfg["options"],
                )
            )
    return runs


def load_dataset():
    from datasets import concatenate_datasets, load_dataset

    raw = load_dataset("MartiHan/Open-MELON-VL-2.5K")
    return concatenate_datasets([raw["train"], raw["validation"], raw["test"]])


def summarize_rows(rows: list[dict], expected_label: str) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row["pred_label"] == expected_label)
    counts = {"nevus": 0, "melanoma": 0, "unknown": 0}
    for row in rows:
        counts[row["pred_label"]] = counts.get(row["pred_label"], 0) + 1

    return {
        "total": total,
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "pred_counts": counts,
    }


def main():
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "Missing captioning dependencies. Run `uv sync --extra captioning` "
            "or invoke this script with `uv run --extra captioning python "
            "medgemma_prompt_ablation.py`."
        ) from exc

    parser = argparse.ArgumentParser(description="Run MedGemma prompt/parameter ablations")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    parser.add_argument("--model", help="Override the model name from the YAML config")
    parser.add_argument("--run-id", help="Run only one prompt/parameter combination")
    parser.add_argument("--prompt-id", help="Run only one prompt configuration from the YAML config")
    parser.add_argument("--parameter-id", help="Run only one parameter set from the YAML config")
    parser.add_argument(
        "--output-root",
        help="Custom results directory. By default, the config model keeps the legacy "
        "results/medgemma_ablation path and model overrides are saved under "
        "results/ollama_ablation/<model>/",
    )
    parser.add_argument("--limit-runs", type=int, default=-1, help="Limit number of combinations for testing")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    dataset = load_dataset()
    runs = build_runs(config)
    if args.prompt_id:
        runs = [run for run in runs if run.prompt_id == args.prompt_id]
    if args.parameter_id:
        runs = [run for run in runs if run.parameter_id == args.parameter_id]
    if args.run_id:
        runs = [run for run in runs if run.run_id == args.run_id]
    if args.limit_runs > 0:
        runs = runs[:args.limit_runs]
    if not runs:
        raise ValueError("No runs matched the requested filters.")

    target_indices = config["target_indices"]
    expected_label = config["target_label"]
    config_model = config["model"]
    model = args.model or config_model

    if args.output_root:
        output_root = Path(args.output_root)
    elif config_path.resolve() == DEFAULT_CONFIG.resolve() and model == config_model:
        output_root = SCRIPT_DIR / "results" / "medgemma_ablation"
    else:
        output_root = SCRIPT_DIR / "results" / "ollama_ablation" / sanitize_model_name(model)
    output_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    print(f"Model: {model}")
    print(f"Target label: {expected_label}")
    print(f"Target indices ({len(target_indices)}): {target_indices}")
    print(f"Total runs: {len(runs)}")

    for run in runs:
        run_dir = output_root / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        caption_path = run_dir / "captions.jsonl"
        summary_path = run_dir / "summary.json"

        print("\n" + "=" * 60)
        print(f"RUN {run.run_id}")
        print(f"Prompt: {run.prompt_description}")
        print(f"Params: {run.parameter_description} | {run.options}")
        print("=" * 60)

        rows = []
        with open(caption_path, "w") as f:
            for idx in target_indices:
                sample = dataset[idx]
                img_b64 = image_to_base64(sample["image"])
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": run.prompt, "images": [img_b64]}],
                    options=run.options,
                )
                caption_gen = response.message.content.strip()
                pred_label = classify_caption(caption_gen)
                row = {
                    "index": idx,
                    "pmc_id": sample["pmc_id"],
                    "prompt_id": run.prompt_id,
                    "parameter_id": run.parameter_id,
                    "caption_gen": caption_gen,
                    "pred_label": pred_label,
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                print(f"idx={idx} pred={pred_label}")

        summary = summarize_rows(rows, expected_label)
        summary.update(
            {
                "run_id": run.run_id,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt_id": run.prompt_id,
                "prompt_description": run.prompt_description,
                "parameter_id": run.parameter_id,
                "parameter_description": run.parameter_description,
                "options": run.options,
                "target_label": expected_label,
                "target_indices": target_indices,
            }
        )

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        all_summaries.append(summary)
        print(f"Accuracy on fixed {expected_label} subset: {summary['accuracy']:.1f}%")
        print(f"Predictions: {summary['pred_counts']}")

    all_summaries.sort(key=lambda row: row["accuracy"], reverse=True)
    with open(output_root / "leaderboard.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    for row in all_summaries:
        print(f"{row['run_id']:32s} acc={row['accuracy']:5.1f}% preds={row['pred_counts']}")


if __name__ == "__main__":
    main()

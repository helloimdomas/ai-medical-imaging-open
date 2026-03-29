#!/usr/bin/env python3
"""
Run a one-image local Ollama vision smoke test.

Usage:
    uv run --extra captioning python ollama_vision_smoke_test.py
"""

import argparse
import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "ollama_vision_smoke_test.json"
DEFAULT_MODEL = "llama3.2-vision"
DEFAULT_PROMPT = (
    "Describe the morphology of this lesion or pathology image and provide a brief "
    "diagnostic impression. Be concise and medically specific."
)


def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_sample(index: int, split: str):
    from datasets import load_dataset

    dataset = load_dataset("MartiHan/Open-MELON-VL-2.5K", split=split)
    return dataset[index]


def main():
    parser = argparse.ArgumentParser(description="Run a one-image Ollama vision smoke test")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--split", default="train", help="Dataset split to sample from")
    parser.add_argument("--index", type=int, default=0, help="Dataset index within the chosen split")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent with the image")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--num-predict", type=int, default=160, help="Maximum generated tokens")
    parser.add_argument(
        "--keep-alive",
        default="0s",
        help="Ollama keep_alive setting. Use 0s to unload the model after the request.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON output path")
    args = parser.parse_args()

    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "Missing captioning dependencies. Run `uv sync --extra captioning` "
            "or invoke this script with `uv run --extra captioning python "
            "ollama_vision_smoke_test.py`."
        ) from exc

    sample = load_sample(args.index, args.split)
    img_b64 = image_to_base64(sample["image"])

    print(f"Model: {args.model}")
    print(f"Dataset split/index: {args.split}[{args.index}]")
    print(f"Caption: {sample['caption']}")
    print(f"fig_quality: {sample.get('fig_quality', 'n/a')}")

    t0 = time.perf_counter()
    response = ollama.chat(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt, "images": [img_b64]}],
        options={"temperature": args.temperature, "num_predict": args.num_predict},
        keep_alive=args.keep_alive,
    )
    elapsed = time.perf_counter() - t0

    result = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "split": args.split,
        "index": args.index,
        "pmc_id": sample.get("pmc_id"),
        "figure_id": sample.get("figure_id"),
        "panel_id": sample.get("panel_id"),
        "fig_quality": sample.get("fig_quality"),
        "caption": sample["caption"],
        "prompt": args.prompt,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "elapsed_s": round(elapsed, 3),
        "response": response.message.content.strip(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Generation time: {elapsed:.2f}s")
    print("\nResponse:\n")
    print(result["response"])
    print(f"\nSaved result to {output_path}")

    if args.keep_alive == "0s":
        print(f"\nRequested immediate unload for {args.model}.")


if __name__ == "__main__":
    main()

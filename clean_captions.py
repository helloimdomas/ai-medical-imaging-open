"""
Clean Open-MELON captions by removing non-visual information.

Modes:
- clean: rewrite captions keeping only visual descriptions
- clean_and_label: infer a conservative diagnosis label and output
  "Diagnosis. Cleaned caption" in structured form

Setup:
  export GEMINI_API_KEY="your-api-key"
  uv run python clean_captions.py
"""

import json, time, os, argparse, re
from pathlib import Path

# CONFIG 
MODEL = "gemma-3-27b-it"
SCRIPT_DIR = Path(__file__).parent
INDEX_FILE = SCRIPT_DIR / "indices" / "melanoma_nevus_indices.json"
API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = (
    "You are a histopathology caption editor. Your job is to rewrite captions "
    "so they contain ONLY visual descriptions and diagnosis names. "
    "Keep: tissue architecture, cell morphology, colors, patterns, structures, "
    "lesion shape, cell types visible, AND diagnosis/tumor names (e.g. melanoma, "
    "carcinoma, nevus, fibrosarcoma). "
    "Remove ALL of the following: staining method (H&E, IHC, etc.), "
    "magnification (×40, 100x, etc.), patient demographics (age, sex), "
    "clinical history, figure/panel references, "
    "study methodology, receptor status, and staging information. "
    "If after removing non-visual info nothing remains, output 'NO_VISUAL_CONTENT'. "
    "Output ONLY the rewritten caption, nothing else."
)

LABEL_SYSTEM_PROMPT = (
    "Read the histopathology caption and fill in this JSON only: "
    '{"diagnosis":"MELANOMA|NEVUS|SPITZ_TUMOR|DIFFERENTIAL|OTHER","cleaned":"Diagnosis. Cleaned caption."}. '
    "Use MELANOMA, NEVUS, or SPITZ_TUMOR only for very clear final diagnoses. "
    "If the diagnosis is excluded, uncertain, comparative, or part of a differential, use DIFFERENTIAL. "
    "If it is clearly something else, use OTHER. "
    "Cleaned text must keep only morphology and the retained diagnosis, with no staining, magnification, demographics, or panel text."
)


def load_completed(path: Path) -> dict[int, dict]:
    """Load completed entries from JSONL, return {index: record}."""
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["index"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_result(path: Path, result: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def call_gemma(prompt: str, system_prompt: str, client, max_retries: int = 5) -> str:
    """Call Gemma via Gemini API with exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[{"role": "user", "parts": [{"text": system_prompt + "\n\n" + prompt}]}],
                config={"temperature": 0, "max_output_tokens": 300},
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (2 ** attempt) + 1  # 2, 3, 5, 9, 17 seconds
                print(f"  Rate limited. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception(f"Max retries ({max_retries}) exceeded")


def ask_gemma_clean(caption: str, client, max_retries: int = 5) -> str:
    """Rewrite the caption keeping only visual descriptions."""
    prompt = f"Rewrite this caption keeping only visual descriptions:\n\n{caption}"
    return call_gemma(prompt, SYSTEM_PROMPT, client, max_retries=max_retries)


def extract_json_object(text: str) -> dict:
    """Extract the first JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


def ask_gemma_clean_and_label(caption: str, client, max_retries: int = 5) -> dict:
    """Rewrite the caption and assign a conservative diagnosis label."""
    prompt = f"Analyze this caption and return the JSON object only:\n\n{caption}"
    response_text = call_gemma(prompt, LABEL_SYSTEM_PROMPT, client, max_retries=max_retries)
    payload = extract_json_object(response_text)

    diagnosis = str(payload.get("diagnosis", "")).strip().upper()
    cleaned = str(payload.get("cleaned", "")).strip()
    if diagnosis not in {"MELANOMA", "NEVUS", "SPITZ_TUMOR", "DIFFERENTIAL", "OTHER"}:
        raise ValueError(f"Unexpected diagnosis label: {diagnosis}")
    if not cleaned:
        raise ValueError("Missing cleaned caption in Gemini response")

    payload["diagnosis"] = diagnosis
    payload["cleaned"] = cleaned
    return payload


def main():
    parser = argparse.ArgumentParser(description="Clean or label Open-MELON captions")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("-n", type=int, default=-1, help="Process only first N images (-1 for all)")
    parser.add_argument(
        "--mode",
        choices=["clean", "clean_and_label"],
        default="clean",
        help="Whether to only clean captions or to also assign a conservative diagnosis label",
    )
    parser.add_argument(
        "--all-indices",
        action="store_true",
        help="Process the full dataset instead of the current melanoma/nevus index subset",
    )
    parser.add_argument(
        "--output",
        help="Custom output JSONL path",
    )
    args = parser.parse_args()
    
    api_key = args.api_key or API_KEY
    if not api_key:
        print("Error: Set GEMINI_API_KEY environment variable or use --api-key")
        return

    from datasets import load_dataset, concatenate_datasets
    from google import genai
    
    client = genai.Client(api_key=api_key)
    
    default_output = (
        SCRIPT_DIR / "captions" / "captions_cleaned.jsonl"
        if args.mode == "clean"
        else SCRIPT_DIR / "captions" / "captions_cleaned_labeled.jsonl"
    )
    output_file = Path(args.output) if args.output else default_output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Open-MELON dataset (all splits)...", flush=True)
    ds_raw = load_dataset("MartiHan/Open-MELON-VL-2.5K")
    ds = concatenate_datasets([ds_raw["train"], ds_raw["validation"], ds_raw["test"]])
    print(f"Total images: {len(ds)}")

    idx_data = None
    target_indices = list(range(len(ds))) if args.all_indices else None
    if not args.all_indices:
        with open(INDEX_FILE) as f:
            idx_data = json.load(f)
        target_indices = sorted(idx_data["melanoma"] + idx_data["nevus"])
    
    if args.n > 0:
        target_indices = target_indices[:args.n]
    n = len(target_indices)

    label_of = {}
    if idx_data is not None:
        mel_set = set(idx_data["melanoma"])
        label_of = {i: "melanoma" if i in mel_set else "nevus" for i in target_indices}

    done = load_completed(output_file)
    remaining = len([i for i in target_indices if i not in done])
    print(f"Target: {n} | Done: {len([i for i in target_indices if i in done])} | Remaining: {remaining}", flush=True)

    if remaining == 0:
        print("All captions already cleaned!")
        return

    total_time = 0.0
    processed = 0

    for i in target_indices:
        if i in done:
            continue

        caption = ds[i]["caption"]
        t0 = time.time()
        
        try:
            if args.mode == "clean":
                payload = {"cleaned": ask_gemma_clean(caption, client)}
            else:
                payload = ask_gemma_clean_and_label(caption, client)
        except Exception as e:
            print(f"Error on idx={i}: {e}")
            time.sleep(2)
            continue
            
        elapsed = time.time() - t0

        result = {
            "index": i,
            "original": caption,
            "time_s": round(elapsed, 2),
        }
        if i in label_of:
            result["label"] = label_of[i]
        result.update(payload)

        append_result(output_file, result)
        done[i] = result
        processed += 1
        total_time += elapsed

        avg = total_time / processed
        eta_min = remaining * avg / 60
        remaining -= 1
        print(
            f"[{n - remaining}/{n}] idx={i} | {elapsed:.1f}s | "
            f"avg={avg:.1f}s | ETA={eta_min:.1f}min",
            flush=True,
        )

    print(f"\nDone! Cleaned {processed} captions → {output_file}")


if __name__ == "__main__":
    main()

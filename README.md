# Testing Medical VLM Generalization on Rare Histopathology Cases

**Research question:** How well do medical vision-language models generalize to rare and diagnostically ambiguous histopathology cases?

**Hypothesis:** Medical VLMs will perform better on frequently seen histopathology cases from training than on rare cases, and their accuracy will degrade on diagnostically ambiguous subtypes underrepresented in training data.

**Main finding:** MedGemma predicts "melanoma" for nearly every sample (99.4% sensitivity, 0.7% specificity), achieving only 50.6% balanced accuracy. Embedding-based classifiers reach ~74% balanced accuracy, but both approaches fail on the same 35 diagnostically ambiguous cases (Spitz tumors, desmoplastic melanoma, cellular blue nevi), suggesting a performance ceiling set by genuine morphological ambiguity.

**Authors:** Daan Merx, Domas Berulis, Roma den Otter -- Department of Biomedical Engineering, TU/e

## Key Results

### MedGemma Prompt Ablation (4 prompts × 3 decoding settings)

Two prompts were evaluated on the full dataset (920 samples):

- **Open (baseline):** *"You are a pathologist examining a biopsy. Describe the key histological features in 2-3 sentences. Provide your best guess for the diagnosis and your certainty level."*
- **Binary choice:** *"You are a pathologist examining a biopsy. This lesion is either MELANOMA or a BENIGN NEVUS. Describe the key histological features in 2-3 sentences. State your diagnosis (melanoma or benign nevus)."*

The only difference: the binary prompt constrains the model to choose between two classes, while the open prompt leaves diagnosis unconstrained. Full prompt text in [`configs/prompts.yaml`](configs/prompts.yaml).

| Prompt Type | Accuracy | Sensitivity | Specificity | Faithfulness | Relevance |
|---|---|---|---|---|---|
| Open (baseline) | 8.7% | 10.5% | 5.0% | 0.15 | 0.23 |
| Binary choice | 67.0% | 99.4% | 0.7% | 0.44 | 0.57 |

MedGemma fails regardless of prompt strategy: open prompts produce vague captions that lack diagnostic keywords; binary prompts force a choice but the model defaults to "melanoma" for nearly every sample.

### Follow-up: Prompt × Decoding Ablation on Hard Cases

To rule out prompt engineering as a fix, we tested **4 prompt variants × 3 decoding settings = 12 configurations** on 10 diagnostically challenging nevus cases (Spitz tumors, cellular blue nevi). Configuration details in [`configs/medgemma_ablation.yaml`](configs/medgemma_ablation.yaml).

| Prompt | Deterministic | T=0.2 | T=0.5 |
|---|---|---|---|
| Open brief | 0% (6M, 4U) | 0% (4M, 6U) | 0% (4M, 6U) |
| Open descriptive | 10% (5M, 1N, 4U) | 0% (4M, 6U) | 0% (5M, 5U) |
| Binary | 0%\* (7M, 18N, 10U) | 10% (8M, 1N, 1U) | 10% (9M, 1N) |
| Binary strict | 0% (10M) | 10% (9M, 1N) | 0% (10M) |

*M = melanoma, N = nevus, U = unknown. All target samples are nevus, so correct = predicted nevus. \*35 mixed-label samples.*

**No configuration exceeded 10% accuracy on these hard cases.** The melanoma bias persists across all prompt formulations and decoding temperatures, confirming this is a model-level limitation rather than a prompt engineering problem. Full per-configuration results in `results/medgemma_ablation/`.

### Embedding Classifiers (10-trial balanced accuracy, 316 samples/class)

| Model | Balanced Accuracy | Std |
|---|---|---|
| MedGemma (binary choice) | 50.6% | 3.0 |
| BiomedCLIP + SVM | 73.9% | 2.9 |
| MedSigLIP + SVM | 74.0% | 3.3 |

*Full results in `results/balanced_accuracy.json`.*

Excluding Spitz tumors, MedSigLIP + SVM improves to **76.4%**, confirming that Spitz cases are a primary source of classification difficulty.

## Poster

The final poster is at [`poster/poster.pdf`](poster/poster.pdf) (LaTeX source: [`poster/poster.tex`](poster/poster.tex)).

## Reproducing Results

This repository uses [`uv`](https://docs.astral.sh/uv/) for environment management. Python 3.11 required.

```bash
# 1. Set up environment
uv python install 3.11
uv sync --extra dev

# 2. Generate MedGemma captions (requires Ollama running MedGemma)
uv run python pipeline.py --prompt-id binary_choice

# 3. Extract embeddings (requires GPU recommended)
uv run --extra embeddings python biomedclip_embeddings.py
uv run --extra embeddings --env-file .env python medsiglip_embeddings.py

# 4. Train classifiers on embeddings
uv run python train_embedding_classifier.py \
  --embeddings-path embeddings/biomedclip_embeddings.npz \
  --model-name BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  --results-path results/biomedclip_results.json

# 5. Compute balanced accuracy across all models
uv run python balanced_accuracy.py

# 6. Analyze failure themes on shared hard cases
uv run python analyze_failure_themes.py
```

All final results are pre-computed in `results/`. Steps 2-3 require model downloads and API keys (see `.env`).

## Repository Structure

```
README.md                       # This file
pyproject.toml                  # Dependencies and environment (uv)
uv.lock                         # Locked dependency versions

pipeline.py                     # MedGemma caption generation + RAGAS evaluation
clean_captions.py               # Caption cleaning via Gemini API
biomedclip_embeddings.py        # BiomedCLIP embedding extraction
medsiglip_embeddings.py         # MedSigLIP embedding extraction
train_embedding_classifier.py   # Classifier training on cached embeddings
biomedclip_classifier.py        # BiomedCLIP baseline wrapper
balanced_accuracy.py            # Balanced accuracy computation (10-trial)
analyze_failure_themes.py       # Failure theme analysis on shared hard cases
medgemma_prompt_ablation.py     # MedGemma prompt/decoding ablation
run_pathology_anti_bias_ablation.py  # Anti-bias ablation study

configs/                        # Prompt and ablation configurations
captions/                       # Ground truth and generated captions
embeddings/                     # Cached BiomedCLIP and MedSigLIP embeddings
results/                        # Final evaluation results (JSON)
  results/archive/              # Exploratory runs (not in poster)
poster/                         # Poster source (LaTeX) and PDF
notebooks/                      # Exploratory notebooks
docs/internal/                  # Assignment brief, rubric, progress notes
```

## Dataset and Preprocessing

### Label extraction (keyword matching)

[Open-MELON-VL-2.5K](https://huggingface.co/datasets/MartiHan/Open-MELON-VL-2.5K) contains 2,499 histopathology images but no explicit diagnosis labels. We extract binary labels from the captions using keyword matching:

- Caption contains "melanoma" but not "nevus"/"nevi" → **melanoma** (597 samples)
- Caption contains "nevus"/"nevi" but not "melanoma" → **nevus** (263 samples)
- Caption contains both → **excluded** (ambiguous, ~110 samples)
- Caption contains neither → **excluded** (different condition)

This yields **913 usable samples**. Spitz tumors (53 samples) are treated as benign.

**Example of an excluded ambiguous caption:**
> *"Histopathological image showing nevus cells from Case 1 [...] where melanoma developed and spread rapidly during pregnancy."*

This mentions both "nevus cells" and "melanoma" because melanoma arose near existing nevus tissue — including it would contaminate the binary benchmark.

### Caption cleaning

Original captions contain non-visual metadata (staining methods, magnification, staging) that a vision model cannot see. We use Gemini to rewrite each caption, keeping only visual descriptions and diagnosis.

**Before (original):**
> *"Histopathology showing spindle cell uveal melanoma [...] stage IIIB, pT4bN0M0 [...] Staining is Hematoxylin-Eosin (HE), magnified x100."*

**After (cleaned):**
> *"Dense cell proliferation composed of small and medium fusiform cells with evident pigment production. Uveal melanoma."*

Cleaned captions serve as ground truth references for RAGAS evaluation. See [`captions/captions_cleaned.jsonl`](captions/captions_cleaned.jsonl).

# Testing Medical VLM Generalization on Rare Histopathology Cases

**Research question:** How well do medical vision-language models generalize to rare and diagnostically ambiguous histopathology cases?

**Hypothesis:** Medical VLMs will perform better on frequently seen histopathology cases from training than on rare cases, and their accuracy will degrade on diagnostically ambiguous subtypes underrepresented in training data.

**Why melanoma vs. nevus?** Melanoma is heavily represented in medical training data because of its clinical importance, while benign nevi (moles) and rare subtypes like Spitz tumors are far less common. This makes melanoma vs. nevus classification an ideal test case: if a VLM has learned a bias toward melanoma from its training data, it will over-diagnose melanoma and miss benign cases — especially rare ones that share morphological features with malignancy.

**Main finding:** MedGemma predicts "melanoma" for nearly every sample (99.4% sensitivity, 0.7% specificity), achieving only 50.6% balanced accuracy — even after testing 12 prompt/decoding configurations. Embedding-based classifiers reach ~74% balanced accuracy, both approaches fail on the same 35 diagnostically ambiguous cases (Spitz tumors, desmoplastic melanoma, cellular blue nevi), suggesting a performance ceiling set by genuine morphological ambiguity.

**Authors:** Daan Merx, Domas Berulis, Roma den Otter — Department of Biomedical Engineering, TU/e

---

## Dataset and Preprocessing

### Label Extraction

[Open-MELON-VL-2.5K](https://huggingface.co/datasets/MartiHan/Open-MELON-VL-2.5K) contains 2,499 histopathology images but no explicit diagnosis labels. We use a two-stage process:

**Stage 1 — Initial selection** via keyword matching on original captions to identify candidate melanoma/nevus samples:

| Rule | Result | Count |
|---|---|---|
| Contains "melanoma" but not "nevus"/"nevi" | **melanoma** | 597 |
| Contains "nevus"/"nevi" but not "melanoma" | **nevus** | 263 |
| Contains both keywords | **excluded** (ambiguous) | ~110 |
| Contains neither keyword | **excluded** (other condition) | ~1,429 |

This yields **913 candidate samples**. Spitz tumors (53 samples) are classified as benign.

**Stage 2 — Gemini label verification:** Each caption is sent to Gemini (gemma-3-27b-it) which assigns a conservative diagnosis label (`MELANOMA`, `NEVUS`, `SPITZ_TUMOR`, `DIFFERENTIAL`, or `OTHER`) based on the full caption context. This second pass validates the keyword-based selection and resolves edge cases where keyword context alone is insufficient. These Gemini-assigned labels are what the embedding classifiers use.

**Example of an excluded ambiguous caption:**
> *"Histopathological image showing nevus cells from Case 1 [...] where melanoma developed and spread rapidly during pregnancy."*

This mentions both "nevus cells" and "melanoma" because melanoma arose near existing nevus tissue — including it would contaminate the binary benchmark.

### Caption Cleaning

Original captions contain non-visual metadata (staining methods, magnification, staging) that a vision model cannot see. We use Gemini to rewrite each caption, keeping only visual descriptions and diagnosis. Cleaned captions serve as ground truth references for RAGAS evaluation.

**Before (original):**
> *"Histopathology showing spindle cell uveal melanoma [...] stage IIIB, pT4bN0M0 [...] Staining is Hematoxylin-Eosin (HE), magnified x100."*

**After (cleaned):**
> *"Dense cell proliferation composed of small and medium fusiform cells with evident pigment production. Uveal melanoma."*

See [`captions/captions_cleaned.jsonl`](captions/captions_cleaned.jsonl) for all cleaned captions.

---

## Experiments and Results

### Step 1: MedGemma Caption Classification (full dataset, 913 samples)

[MedGemma-1.5-4B-it](https://ollama.com/dcarrascosa/medgemma-1.5-4b-it:Q4_K_M) (Q4_K_M quantization, via Ollama) generates free-text captions for each image. Classification is by keyword extraction: does the caption contain "melanoma" or "nevus"?

Caption quality is evaluated with **RAGAS** (Retrieval Augmented Generation Assessment), using Gemini as a judge against the cleaned reference captions:
- **Faithfulness (0–1):** Are claims in the generated caption supported by the reference?
- **Relevance (0–1):** Does the generated caption address the same medical findings?

Two prompts were evaluated. Full prompt text in [`configs/prompts.yaml`](configs/prompts.yaml).

- **Open (baseline):** *"You are a pathologist examining a biopsy. Describe the key histological features [...] Provide your best guess for the diagnosis and your certainty level."*
- **Binary choice:** *"You are a pathologist examining a biopsy. This lesion is either MELANOMA or a BENIGN NEVUS. [...] State your diagnosis."*

| Prompt Type | Accuracy | Sensitivity | Specificity | Faithfulness | Relevance |
|---|---|---|---|---|---|
| Open (baseline) | 8.7% | 10.5% | 5.0% | 0.15 | 0.23 |
| Binary choice | 67.0% | 99.4% | 0.7% | 0.44 | 0.57 |

- **Accuracy:** Proportion of correct predictions overall.
- **Sensitivity (recall for melanoma):** Of all true melanoma cases, how many did the model correctly identify?
- **Specificity (recall for nevus):** Of all true nevus cases, how many did the model correctly identify?
- **Faithfulness:** RAGAS score (0–1). Are claims in the generated caption factually supported by the reference?
- **Relevance:** RAGAS score (0–1). Does the generated caption cover the same medical findings as the reference?

The open prompt produces vague captions that lack diagnostic keywords (faithfulness 0.15). The binary prompt forces a choice but the model defaults to "melanoma" for nearly every sample — 99.4% sensitivity at the cost of 0.7% specificity.

**Example: Open prompt completely misidentifies tissue (index 3):**

Ground truth: *"Dense cell proliferation composed of small and medium fusiform cells with evident pigment production. Uveal melanoma."*

MedGemma (open prompt): *"The image shows a dense infiltrate of lymphocytes, predominantly T cells, surrounding a central area of necrosis. There is also evidence of reactive fibroblastic proliferation. Lymphocytic infiltrate with necrosis. High."*

The model describes lymphocytes instead of melanocytes and diagnoses necrosis instead of melanoma — a complete misidentification. See [`captions/baseline/captions.jsonl`](captions/baseline/captions.jsonl), index 3.

**Example: Binary prompt misclassifying a benign nevus (index 43):**

Cleaned ground truth: *"Lentiginous intraepidermal proliferation of melanocytes and a benign intradermal nevus."*

MedGemma (binary prompt): *"Melanoma. High. The lesion shows atypical melanocytes infiltrating the dermis."*

The model confidently calls this benign nevus "melanoma." See [`captions/binary_choice/captions.jsonl`](captions/binary_choice/captions.jsonl), index 43.

### Step 2: Reality Check with Embedding Classifiers

Are the images themselves just too hard to classify, or is MedGemma specifically failing? To answer this, we extracted **BiomedCLIP** embeddings (512-d) and trained supervised classifiers (LogReg, SVM, Random Forest). This reached ~74% balanced accuracy — confirming the images carry enough visual signal.

The course instructor then suggested adding **MedSigLIP** (768-d embeddings), since MedGemma internally uses MedSigLIP as its vision encoder. If MedGemma's own backbone can separate the classes via embeddings, the failure must be in MedGemma's language generation, not its visual understanding. MedSigLIP + SVM also reached ~74%.

Each trial samples equal numbers per class (316 per class, matching the minority class). Balanced accuracy is averaged over 10 independent trials.

| Model | Balanced Accuracy | Std |
|---|---|---|
| MedGemma (binary choice) | 50.6% | 3.0 |
| BiomedCLIP + LogReg | 73.1% | 3.0 |
| BiomedCLIP + SVM | 73.9% | 2.9 |
| BiomedCLIP + RF | 72.6% | 2.5 |
| MedSigLIP + LogReg | 71.4% | 4.1 |
| MedSigLIP + SVM | 74.0% | 3.3 |
| MedSigLIP + RF | 71.6% | 5.2 |

*Full results in [`results/balanced_accuracy.json`](results/balanced_accuracy.json).*

Excluding Spitz tumors, MedSigLIP + SVM improves to **76.4%** (see [`results/balanced_accuracy_exclude_spitz.json`](results/balanced_accuracy_exclude_spitz.json)), confirming that Spitz cases are a primary source of classification difficulty.

### Step 3: Prompt × Decoding Ablation

To confirm MedGemma's failure is a model-level limitation and not a prompt engineering problem, we tested **4 prompt variants × 3 decoding settings = 12 configurations** on 10 nevus samples. Configuration in [`configs/medgemma_ablation.yaml`](configs/medgemma_ablation.yaml).

The four prompts:

1. **Open brief:** *"Describe what you see and give the most likely diagnosis."*
2. **Open descriptive:** *"You are a dermatopathologist examining a biopsy. Describe the key histological features in 2-3 sentences and then give your best diagnosis."*
3. **Binary:** *"You are a dermatopathologist examining a biopsy. This lesion is either MELANOMA or a BENIGN NEVUS. Describe the key histological features in 2-3 sentences and state your diagnosis."*
4. **Binary strict:** *"Classify this biopsy as exactly one of: MELANOMA or BENIGN NEVUS. First describe the most important histological features in 1-2 sentences. Then output the final diagnosis using exactly one of those two labels."*

Three decoding settings: deterministic (T=0), mild sampling (T=0.2), moderate sampling (T=0.5).

| Prompt | Deterministic | T=0.2 | T=0.5 |
|---|---|---|---|
| Open brief | 0% (6M, 4U) | 0% (4M, 6U) | 0% (4M, 6U) |
| Open descriptive | 10% (5M, 1N, 4U) | 0% (4M, 6U) | 0% (5M, 5U) |
| Binary | 0%\* (7M, 18N, 10U) | 10% (8M, 1N, 1U) | 10% (9M, 1N) |
| Binary strict | 0% (10M) | 10% (9M, 1N) | 0% (10M) |

*M = melanoma, N = nevus, U = unknown. All target samples are nevus, so correct = predicted nevus. \*35 mixed-label samples.*

**No configuration exceeded 10% accuracy.** The melanoma bias persists across all prompt formulations and decoding temperatures. We also tested a structured **anti-bias prompt** that forces the model to list benign features before malignant ones and instructs it not to diagnose melanoma unless multiple malignant features are present ([`configs/pathology_anti_bias_ablation.yaml`](configs/pathology_anti_bias_ablation.yaml)). This flipped the bias entirely — MedGemma went from predicting nearly everything as melanoma to predicting nearly everything as benign — confirming that prompt engineering cannot fix the underlying limitation.

### Step 4: Why Do Both Models Hit the Same ~74% Ceiling?

Both BiomedCLIP-SVM and MedSigLIP-SVM plateau around 74% balanced accuracy. To investigate, we compared their failures on the **test set only** — the remaining images not used to train the SVMs.

We found:
- **35 cases** where **both** models fail
- **21 cases** where only BiomedCLIP-SVM fails (MedSigLIP correct)
- **15 cases** where only MedSigLIP-SVM fails (BiomedCLIP correct)

The 35 shared failures concentrate in diagnostically challenging subtypes from both directions:

- **Benign lesions that mimic malignancy:** Spitz tumors, cellular blue nevi
- **Melanomas that defy typical appearance:** in-situ, mucosal, desmoplastic

Top failure themes among the 35 shared hard cases:

| Theme | Frequency |
|---|---|
| Pigment-heavy | 28.6% |
| In-situ / lentigo maligna | 22.9% |
| Spindle / desmoplastic | 22.9% |
| Mucosal / special site | 22.9% |

*Full theme breakdown in [`results/failure_theme_analysis.json`](results/failure_theme_analysis.json).*

**Example: Spitz tumor both models call melanoma (index 1737):**

Caption: *"Histopathology of a Spitz tumor demonstrating marked tumor asymmetry [...] identified as a risk-associated histopathological feature."*

Both BiomedCLIP-SVM and MedSigLIP-SVM predict "melanoma." Spitz tumors share morphological features with melanoma (asymmetry, atypical spindle cells), making them a known diagnostic challenge even for expert dermatopathologists.

**Example: Desmoplastic melanoma both models miss (index 1845):**

Caption: *"Histopathology showing desmoplastic melanoma, characterized by atypical spindle cells embedded in a dense fibrous matrix."*

Both models predict "benign." Desmoplastic melanoma lacks the typical melanoma appearance — its spindle cells in fibrous stroma resemble a scar or fibromatosis rather than a melanocytic tumor.

**Example: BiomedCLIP fails, MedSigLIP succeeds (index 1781):**

Caption: *"Late-onset lentiginous and nested melanoma [...] a chaotic architecture with variably sized nests and confluent single nevoid melanocytes."*

BiomedCLIP calls it "benign" while MedSigLIP correctly identifies it as melanoma.

**Example: MedSigLIP fails, BiomedCLIP succeeds (index 941):**

Caption: *"Acrosyringial and periductal distribution of the melanocytic neoplasm [...] acral compound melanocytic nevus with congenital features."*

MedSigLIP calls it "melanoma" while BiomedCLIP correctly identifies it as benign.

Full case-by-case analysis in [`results/failure_theme_analysis.json`](results/failure_theme_analysis.json).

### Step 5: UMAP Visualization

The course instructor suggested UMAP projections to visualize the embedding space. These show partial separation between nevus and melanoma clusters, but Spitz tumor embeddings scatter across melanoma-dense regions — their features are morphologically indistinguishable from melanoma. This explains the ~74% ceiling: it is set by genuine morphological ambiguity in these rare subtypes.

See [`poster/umap_medsiglip.png`](poster/umap_medsiglip.png) and [`poster/umap_biomedclip.png`](poster/umap_biomedclip.png).

---

## Discussion and Future Research


### Why MedGemma Fails

The [MedGemma model card](https://huggingface.co/google/medgemma-4b-it) and [technical report](https://arxiv.org/abs/2507.05201) reveal a likely explanation for the melanoma bias. MedGemma's training data includes:

- **Pathology**: TCGA, CAMELYON, plus proprietary datasets covering **colon, prostate, lymph nodes, and lung** histopathology
- **Dermatology**: PAD-UFES-20, SCIN — **clinical/dermatoscopic surface photos**, not histopathology slides

Melanoma histopathology falls in a gap between these two training streams: it is histopathology (like the pathology data) but of melanocytic skin tissue (like the dermatology data). MedGemma has likely never seen benign melanocytic histopathology during training, so it associates any melanocytic features with melanoma — the dominant condition in its dermatology training. This explains the near-universal melanoma prediction (99.4% sensitivity, 0.7% specificity).

The model card explicitly states: *"For medical image-based applications that do not involve text generation, such as data-efficient classification [...] the MedSigLIP image encoder is recommended"* — directly validating our finding that MedSigLIP embeddings + SVM outperform MedGemma's text generation pipeline.

### What We Could Have Done Better

- **Larger ablation sample**: The prompt ablation used only 10 nevus samples. A larger subset would provide more statistical confidence.
- **Fine-tuning**: We evaluated MedGemma zero-shot only. Fine-tuning on even a small set of melanoma histopathology images might significantly improve specificity.
- **Full-size MedGemma**: We used the 4B parameter model (Q4_K_M quantization) due to hardware constraints. The 27B variant may perform better on this task.
- **Cross-validation for failure analysis**: The 35 shared hard cases come from a single train/test split. Cross-validated failure analysis would be more robust.

### Future Research Directions

**Improving the current approach:**

1. **Fine-tune MedGemma** on labeled melanoma/nevus histopathology images. Instead of prompting the model zero-shot, additional training on task-specific examples could teach it the melanoma-vs-nevus distinction it currently lacks.
2. **Evaluate MedGemma 27B** (multimodal) to determine if the larger model overcomes the melanoma bias — we used the 4B variant due to hardware constraints.

**Extending to new questions:**

3. **Test MedGemma where it was trained**: Run the same VLM-vs-embedding comparison on chest X-rays or colon pathology — domains where MedGemma has direct training data. If the VLM-vs-embedding gap closes in-domain, it confirms the modality gap hypothesis.
4. **Multi-class extension**: Expand beyond binary melanoma/nevus to include basal cell carcinoma, squamous cell carcinoma, and dermatofibroma. Does the same dominant-class bias appear when more diagnosis classes are available?
5. **Other medical VLMs**: Compare with histopathology-specific models to determine whether the melanoma bias is MedGemma-specific or a general limitation of VLMs on underrepresented histopathology domains.
6. **Clinical triage angle**: MedGemma's 99.4% melanoma sensitivity could paradoxically be useful as a conservative pre-screening tool — if MedGemma says "benign," the prediction is almost certainly correct (given how rarely it predicts benign).

---

## Poster

The final poster is at [`poster/poster.pdf`](poster/poster.pdf) (LaTeX source: [`poster/poster.tex`](poster/poster.tex)).

---

## Reproducing Results

This repository uses [`uv`](https://docs.astral.sh/uv/) for environment management. Python 3.11 required.

```bash
# 1. Set up environment
uv python install 3.11
uv sync --extra dev

# 2. Clean captions (requires Gemini API key in .env)
uv run python clean_captions.py

# 3. Generate MedGemma captions (requires Ollama running MedGemma)
uv run python pipeline.py --prompt-id baseline
uv run python pipeline.py --prompt-id binary_choice

# 4. Run prompt ablation
uv run python medgemma_prompt_ablation.py

# 5. Extract embeddings
uv run --extra embeddings python biomedclip_embeddings.py
uv run --extra embeddings --env-file .env python medsiglip_embeddings.py

# 6. Train classifiers on embeddings
uv run python train_embedding_classifier.py \
  --embeddings-path embeddings/biomedclip_embeddings.npz \
  --model-name BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  --results-path results/biomedclip_results.json

# 7. Compute balanced accuracy across all models
uv run python balanced_accuracy.py

# 8. Analyze failure themes on shared hard cases
uv run python analyze_failure_themes.py
```

All final results are pre-computed in `results/`. Steps 2–5 require model downloads and API keys (see `.env`).

> **Safe re-runs:** Every script that writes output files will automatically back up any existing file with a timestamped suffix (e.g. `balanced_accuracy_20250417_143000.json`) before writing, so re-running the pipeline never silently overwrites previous results.

---

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
run_pathology_anti_bias_ablation.py  # Anti-bias prompt ablation

configs/                        # Prompt and ablation configurations
captions/                       # Ground truth and generated captions
embeddings/                     # Cached BiomedCLIP and MedSigLIP embeddings
results/                        # Final evaluation results (JSON)
  results/medgemma_ablation/    # MedGemma prompt ablation outputs
  results/archive/              # Early feasibility tests (not in poster)
poster/                         # Poster source (LaTeX), PDF, and UMAP figures
docs/internal/                  # Assignment brief, rubric, progress notes
```

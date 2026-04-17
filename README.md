# Testing Medical VLM Generalization on Rare Histopathology Cases

**Research question:** How well do medical vision-language models generalize to rare and diagnostically ambiguous histopathology cases?

**Hypothesis:** Medical VLMs will perform better on frequently seen histopathology cases from training than on rare cases, and their accuracy will degrade on diagnostically ambiguous subtypes underrepresented in training data.

**Why melanoma vs. nevus?** Melanoma is heavily represented in medical training data because of its clinical importance, while benign nevi (moles) and rare subtypes like Spitz tumors are far less common. This makes melanoma vs. nevus classification an ideal test case: if a VLM has learned a bias toward melanoma from its training data, it will over-diagnose melanoma and miss benign cases — especially rare ones that share morphological features with malignancy.

**Main finding:** MedGemma predicts "melanoma" for nearly every sample (99.4% sensitivity, 0.7% specificity), achieving only 50.6% balanced accuracy — even after testing 12 prompt/decoding configurations. Embedding-based classifiers reach ~74% balanced accuracy, but both approaches fail on the same 35 diagnostically ambiguous cases (Spitz tumors, desmoplastic melanoma, cellular blue nevi), suggesting a performance ceiling set by genuine morphological ambiguity.

**Authors:** Daan Merx, Domas Berulis, Roma den Otter — Department of Biomedical Engineering, TU/e

---

## Dataset and Preprocessing

### Label Extraction

[Open-MELON-VL-2.5K](https://huggingface.co/datasets/MartiHan/Open-MELON-VL-2.5K) contains 2,499 histopathology images but no explicit diagnosis labels. We extract binary labels from the captions using keyword matching:

| Rule | Result | Count |
|---|---|---|
| Contains "melanoma" but not "nevus"/"nevi" | **melanoma** | 597 |
| Contains "nevus"/"nevi" but not "melanoma" | **nevus** | 263 |
| Contains both keywords | **excluded** (ambiguous) | ~110 |
| Contains neither keyword | **excluded** (other condition) | ~1,429 |

This yields **913 usable samples**. Spitz tumors (53 samples) are classified as benign based on their captions.

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

## Methodology

### MedGemma Caption Classification

[MedGemma-1.5-4B-it](https://ollama.com/dcarrascosa/medgemma-1.5-4b-it:Q4_K_M) (Q4_K_M quantization, via Ollama) generates free-text captions for each image. Classification is by keyword extraction: does the caption contain "melanoma" or "nevus"?

We first ran two prompts on the full 913 images (open-ended and binary choice). Caption quality is evaluated with **RAGAS** (Retrieval Augmented Generation Assessment), using Gemini as a judge against the cleaned reference captions:
- **Faithfulness (0–1):** Are claims in the generated caption supported by the reference?
- **Relevance (0–1):** Does the generated caption address the same medical findings?

MedGemma performed poorly on both prompts. To check whether this was a prompt engineering problem, we ran a **12-configuration ablation** (4 prompt variants × 3 decoding temperatures) on 10 diagnostically hard nevus cases. None exceeded 10% accuracy.

### Reality Check: Are the Images Just That Hard?

To verify that the images themselves are classifiable — and MedGemma's failure is model-specific — we extracted **BiomedCLIP** embeddings (512-d) and trained supervised classifiers (LogReg, SVM, Random Forest) on top. This reached ~74% balanced accuracy, confirming the images carry enough visual signal for classification.

The professor then suggested adding **MedSigLIP** (768-d embeddings), since MedGemma internally uses MedSigLIP as its vision encoder. If MedGemma's own vision backbone can separate the classes when used for embedding-based classification, that would prove the failure is in MedGemma's language generation, not its visual understanding. MedSigLIP + SVM also reached ~74%.

### Why Do Both Embedding Models Hit the Same ~74% Ceiling?

Both BiomedCLIP-SVM and MedSigLIP-SVM plateau around 74% balanced accuracy. To understand why, we analyzed the test-set failures. Note: we can only analyze failures on the **test set** (not the full 913 images), because the remaining images were used to train the SVMs.

We found **35 cases where both models fail**, plus 15 where only MedSigLIP fails and 21 where only BiomedCLIP fails. The shared hard cases concentrate in diagnostically ambiguous subtypes:
- **Benign lesions that mimic malignancy:** Spitz tumors, cellular blue nevi
- **Melanomas that defy typical appearance:** in-situ, mucosal, desmoplastic

The course instructor then suggested **UMAP projections** to visualize the embedding space. These show Spitz tumor embeddings scattering across melanoma-dense regions — their features are morphologically indistinguishable from melanoma. This explains the ~74% ceiling: it is set by genuine morphological ambiguity in these rare subtypes.

---

## Results

### MedGemma: Full-Dataset Runs (913 samples)

Two prompts were evaluated on the full dataset. Full prompt text in [`configs/prompts.yaml`](configs/prompts.yaml).

- **Open (baseline):** *"You are a pathologist examining a biopsy. Describe the key histological features [...] Provide your best guess for the diagnosis and your certainty level."*
- **Binary choice:** *"You are a pathologist examining a biopsy. This lesion is either MELANOMA or a BENIGN NEVUS. [...] State your diagnosis."*

| Prompt Type | Accuracy | Sensitivity | Specificity | Faithfulness | Relevance |
|---|---|---|---|---|---|
| Open (baseline) | 8.7% | 10.5% | 5.0% | 0.15 | 0.23 |
| Binary choice | 67.0% | 99.4% | 0.7% | 0.44 | 0.57 |

- **Accuracy:** Proportion of correct predictions overall.
- **Sensitivity (recall for melanoma):** Of all true melanoma cases, how many did the model correctly identify? High sensitivity = few missed melanomas.
- **Specificity (recall for nevus):** Of all true nevus cases, how many did the model correctly identify? High specificity = few false melanoma alarms.
- **Faithfulness:** RAGAS score (0–1). Are claims in the generated caption factually supported by the reference caption?
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

### MedGemma: Prompt × Decoding Ablation on Hard Cases

To rule out prompt engineering as a fix, we tested **4 prompt variants × 3 decoding settings = 12 configurations** on 10 diagnostically challenging nevus cases (Spitz tumors, cellular blue nevi). Configuration details in [`configs/medgemma_ablation.yaml`](configs/medgemma_ablation.yaml).

The four prompts tested:

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

**No configuration exceeded 10% accuracy on these hard cases.** The melanoma bias persists across all prompt formulations and decoding temperatures, confirming this is a model-level limitation rather than a prompt engineering problem.

### Embedding Classifiers

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

### Failure Theme Analysis

On the test set, we found:
- **35 cases** where **both** BiomedCLIP-SVM and MedSigLIP-SVM fail
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

**Example: Spitz tumor both models call melanoma (index 1737):**

Caption: *"Histopathology of a Spitz tumor demonstrating marked tumor asymmetry [...] identified as a risk-associated histopathological feature."*

Both BiomedCLIP-SVM and MedSigLIP-SVM predict "melanoma." Spitz tumors share morphological features with melanoma (asymmetry, atypical spindle cells), making them a known diagnostic challenge even for expert dermatopathologists.

**Example: Desmoplastic melanoma both models miss (index 1845):**

Caption: *"Histopathology showing desmoplastic melanoma, characterized by atypical spindle cells embedded in a dense fibrous matrix."*

Both models predict "benign." Desmoplastic melanoma lacks the typical melanoma appearance — its spindle cells in fibrous stroma resemble a scar or fibromatosis rather than a melanocytic tumor.

**Example: BiomedCLIP fails, MedSigLIP succeeds (index 1781):**

Caption: *"Late-onset lentiginous and nested melanoma [...] a chaotic architecture with variably sized nests and confluent single nevoid melanocytes."*

BiomedCLIP calls it "benign" while MedSigLIP correctly identifies it as melanoma. The "nevoid" appearance of the melanocytes may confuse BiomedCLIP's embedding space.

**Example: MedSigLIP fails, BiomedCLIP succeeds (index 1986):**

Caption: *"Melanocytic hyperplasia (lentigo) [...] heavily pigmented melanocytes present in the matrix epithelium."*

MedSigLIP calls it "melanoma" while BiomedCLIP correctly identifies it as benign. The heavy pigmentation likely drives the false alarm in MedSigLIP's feature space.

Full case-by-case analysis in [`results/failure_theme_analysis.json`](results/failure_theme_analysis.json).

### UMAP Visualizations

UMAP projections of MedSigLIP and BiomedCLIP embeddings show partial separation between nevus and melanoma clusters, but Spitz tumor embeddings scatter across melanoma-dense regions — their features are morphologically indistinguishable from melanoma in embedding space. See visualizations in [`poster/umap_medsiglip.png`](poster/umap_medsiglip.png) and [`poster/umap_biomedclip.png`](poster/umap_biomedclip.png).

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

# 4. Run prompt ablation on hard cases
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
run_pathology_anti_bias_ablation.py  # Anti-bias ablation study

configs/                        # Prompt and ablation configurations
captions/                       # Ground truth and generated captions
embeddings/                     # Cached BiomedCLIP and MedSigLIP embeddings
results/                        # Final evaluation results (JSON)
  results/archive/              # Exploratory runs (not in poster)
poster/                         # Poster source (LaTeX), PDF, and UMAP figures
notebooks/                      # Exploratory notebooks
docs/internal/                  # Assignment brief, rubric, progress notes
```

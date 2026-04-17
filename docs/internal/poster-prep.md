<!-- THE ORIGINAL "PROMISED" RESEARCH STATEMENT:
Addressing generalizability and bias in medical VLMs
How do vision-language models pretrained on general medical images perform on
rare histopathology images?
We are going to measure to what extent the generated descriptions of the model align with the
ground truth when accounting for dataset noise.
Baseline: LLaVA (general-purpose VLM) for label preprocessing (keep only the diagnosis name)
• To evaluate we will use semantic similarity metrics (RAGAS) to quantify overlap between
generated text and ground truth, focusing on visual information (not caption noise).
• When the model underperforms, we will use attention maps to identify false features and
explore preprocessing (removing labels, rulers).
Feasibility: usage of pretrained models (LLaVA, MedGemma-4B-it) + OpenMelon dataset so no
training required, and tools readily available via Hugging Face. -->

# Poster Preparation Notes

This document organizes what should go on the poster, what to leave out, and what gaps remain.
Written with the grading rubric in mind.

---

## 1. Updated Research Question

**Poster research question:**

> For melanoma vs. nevus classification on histopathology images, are caption-based medical vision-language models competitive with embedding-based approaches?

**Note on evolution from the pitch:**
The general question from the pitch remains the same — we are evaluating how well medical VLMs generalize to rare histopathology images. The specific direction changed during the project. Early experiments with MedGemma caption generation showed near-random performance under balanced evaluation, which made it necessary to introduce embedding-based models (BiomedCLIP, MedSigLIP) as a reality check. This turned the project from a pure captioning evaluation into a comparative study of generative vs. embedding-based paradigms. The pivot produced a clearer, more falsifiable research question with a strong negative result.

---

## 2. What to INCLUDE on the Poster

### 2.1 Abstract (concise, ~80 words)

Key points to convey:
- We tested whether MedGemma (a medical VLM) can classify melanoma vs. nevus via caption generation on 913 histopathology images from Open-MELON.
- MedGemma achieves ~67% raw accuracy but drops to ~50% (random chance) under balanced evaluation, revealing a strong melanoma-overcalling bias.
- Embedding-based approaches (BiomedCLIP, MedSigLIP) with simple SVMs achieve ~74% balanced accuracy.
- Prompt and decoding ablations do not rescue the generative approach.
- Embedding-based classification is clearly superior for this task.

### 2.2 Methodology

Present these methods clearly:

1. **Dataset:** Open-MELON-VL-2.5K, 913 samples (597 melanoma, 263 nevus, 53 Spitz tumor folded into benign). Gemini-assisted caption cleaning and labeling.

2. **Generative approach (MedGemma):**
   - Model: MedGemma-1.5-4B-it (Q4_K_M quantized, via Ollama)
   - Caption generation → keyword-based label extraction
   - Two main prompts tested: open baseline, binary choice
   - RAGAS faithfulness/relevance scoring
   - Prompt ablation: 4 prompt families × 3 decoding settings = 12 configurations
   - Anti-melanoma-bias prompt variant

3. **Embedding approaches:**
   - BiomedCLIP (ViT-B/16, PubMedBERT, 512-dim)
   - MedSigLIP (from Hugging Face, 768-dim)
   - Classifiers: Logistic Regression, SVM (RBF), Random Forest
   - Train/test split with balanced repeated evaluation (10 trials, equal class sampling)

4. **Evaluation protocol:**
   - Primary metric: balanced accuracy (10 trials, all benign + matched melanoma sampling)
   - Secondary: sensitivity, specificity on held-out splits
   - Failure analysis: direct comparison of SVM failure sets between models, theme-based categorization

### 2.3 Results — What to Show

**A. Main comparison table (MUST INCLUDE — this is the core result):**

| Method | Balanced Accuracy (10 trials) |
|---|---|
| MedGemma (binary_choice) | 50.6% ± 3.0 |
| BiomedCLIP + LogReg | 73.1% ± 3.0 |
| BiomedCLIP + SVM | 73.9% ± 2.9 |
| BiomedCLIP + RF | 72.6% ± 2.5 |
| MedSigLIP + LogReg | 71.4% ± 4.1 |
| MedSigLIP + SVM | 74.0% ± 3.3 |
| MedSigLIP + RF | 71.6% ± 5.2 |

Consider a bar chart with error bars for visual impact.

**B. MedGemma's class-imbalance trap (INCLUDE — shows why raw accuracy is misleading):**

| Metric | MedGemma (binary_choice) |
|---|---|
| Raw accuracy | 67.0% |
| Sensitivity (melanoma recall) | 99.4% |
| Specificity (nevus recall) | 0.7% |
| Balanced accuracy | 50.6% |

This is a powerful visual: show the raw vs. balanced accuracy side by side.

**C. MedGemma prompt ablation summary (INCLUDE briefly — shows prompt engineering doesn't fix it):**

Best result on 10 nevus images across 12 configurations: 20% accuracy (closed_binary, deterministic). Worst: 0% accuracy (multiple configs). The anti-bias prompt improved nevus recognition to 80% but destroyed melanoma recall (30%). This demonstrates the model trades one bias for another rather than developing genuine discriminative ability.

**D. Spitz sensitivity analysis (INCLUDE — adds analytical depth):**

| Method | Balanced Acc (with Spitz) | Balanced Acc (without Spitz) |
|---|---|---|
| BiomedCLIP + SVM | 73.9% ± 2.9 | 72.1% ± 3.9 |
| MedSigLIP + SVM | 74.0% ± 3.3 | 76.4% ± 4.4 |

Shows that Spitz tumors compress the gap between models. MedSigLIP becomes clearly stronger on the cleaner benchmark.

**E. Failure theme analysis (INCLUDE — strongest critical discussion material):**

The 35 shared hard cases cluster in:
- Pigment-heavy lesions (29%)
- In situ / lentigo maligna (23%)
- Spindle / desmoplastic morphology (23%)
- Mucosal / special-site cases (23%)
- Blue-nevus-related lesions (14%)
- Spitz-related lesions (9%)

This shows errors are concentrated in diagnostically subtle subtypes, not random noise.

**F. MedGemma baseline RAGAS scores (INCLUDE briefly — connects to original pitch):**

| Prompt | Faithfulness | Relevance |
|---|---|---|
| Baseline | 0.15 | 0.23 |
| Binary choice | 0.44 | 0.57 |

These are very low. Even the better prompt produces captions that only moderately align with ground truth. This directly addresses the original pitch question about text alignment.

### 2.4 Critical Discussion Points (INCLUDE)

1. **MedGemma's failure is not a prompt problem** — 12 prompt/parameter configs and an anti-bias variant all failed to produce reliable classification. The model has a deep melanoma-overcalling bias.

2. **Balanced evaluation is essential** — The raw 67% accuracy is entirely an artifact of the 597:316 class imbalance. This is a methodological lesson worth highlighting.

3. **Embedding models are not perfect either** — 74% balanced accuracy means 1 in 4 cases is misclassified. The failure analysis shows these are genuinely difficult pathology cases (acral, desmoplastic, special-site).

4. **Benchmark limitations:**
   - Labels from Gemini-based caption cleaning, not expert pathologist annotation
   - Benign class contains some off-target entities (nevus sebaceus, white spongy nevus, epidermal nevus)
   - Results are specific to this dataset and task

5. **MedSigLIP vs BiomedCLIP nuance** — Under the inclusive benchmark they're tied; excluding Spitz makes MedSigLIP clearly better. This matters because it shows the answer depends on benchmark definition.

### 2.5 Conclusion Box

> Caption-based generative VLM prompting (MedGemma) is not competitive with embedding-based classifiers (BiomedCLIP, MedSigLIP) for melanoma vs. nevus classification on histopathology. The generative approach achieves near-random balanced accuracy regardless of prompt engineering. Embedding-based SVMs achieve ~74% balanced accuracy with interpretable failure patterns concentrated in diagnostically subtle subtypes.

---

## 3. What to LEAVE OUT of the Poster

### 3.1 llama3.2-vision hard-case experiment (leave out or mention only in one sentence)

**Reasoning:** This was an exploratory curiosity test, not a systematic experiment. It used only the 35 shared hard cases (selection bias), achieved 11.4% accuracy, and doesn't contribute to the main narrative. Including it would dilute focus and raise questions about why a different model was introduced for only one experiment.

**Exception:** If space permits, one sentence in the discussion: *"A secondary generative model (llama3.2-vision) also failed on the shared hard cases (11.4% accuracy), suggesting the limitation is paradigm-level, not model-specific."*

### 3.2 gemma3:12b-it-qat smoke test (leave out)

**Reasoning:** This was a single-image smoke test to verify Ollama setup. It has no bearing on the research question.

### 3.3 RAGAS detailed per-sample scores (leave out)

**Reasoning:** The summary-level faithfulness/relevance scores tell the story. Per-sample RAGAS data adds noise without insight for a poster format.

### 3.4 Individual train/test split raw accuracies (downplay)

**Reasoning:** The progress log correctly identifies the balanced repeated evaluation as more trustworthy. Show the raw vs. balanced comparison for MedGemma (to demonstrate the imbalance trap), but for BiomedCLIP/MedSigLIP, lead with balanced accuracy.

### 3.5 Off-target nevus entity details (brief mention only)

**Reasoning:** Important caveat, but the specific entity list (white spongy nevus, nevus sebaceus, etc.) is too granular for a poster. One sentence: *"The benign class includes some non-melanocytic nevus entities, which may introduce label noise."*

### 3.6 Full MedGemma ablation leaderboard (leave out)

**Reasoning:** The 12-config ablation over only 10 nevus images is a diagnostic experiment, not a full benchmark. Summarize the conclusion ("prompt engineering does not fix the fundamental bias") rather than showing the full grid.

---

## 4. Gaps and Further Work Needed

### 4.1 High priority — should be done BEFORE the poster

1. **Freeze the benchmark definition.** The progress log recommends this and it hasn't been done. The poster needs to report numbers on one fixed benchmark. Decision needed: include or exclude Spitz? Include or exclude off-target nevus entities? Recommendation: report the inclusive benchmark (913 samples) as primary, and the exclude-Spitz analysis (860 samples) as a sensitivity check.

2. **Create proper visualizations.** The current poster.tex has tables but no figures. For a poster, you need at minimum:
   - A bar chart of balanced accuracy with error bars (the main result)
   - Ideally a second visual: either the MedGemma raw-vs-balanced comparison, or the failure theme breakdown
   - Consider a simple pipeline/methodology diagram

3. **Update poster.tex with correct numbers.** The existing poster.tex uses outdated numbers (e.g., "920 images", "73%" for BiomedCLIP only, no MedSigLIP). It needs to be rewritten with the current benchmark numbers and include MedSigLIP.

4. **The poster currently omits MedSigLIP entirely.** This is a significant gap — MedSigLIP is tied for best or clearly best depending on Spitz inclusion. It must be added.

### 4.2 Medium priority — would strengthen the poster

5. **Cross-validation or repeated train/test splits for embedding models.** The current supervised results depend on one train/test split (supplemented by balanced repeated evaluation). Running 5-fold CV or multiple random splits would be more rigorous and is the kind of thing the rubric's "Excellent experiments" criterion expects ("rigorous testing").

6. **Confusion matrices.** A 2×2 confusion matrix for the best embedding model (MedSigLIP SVM) and for MedGemma would visually show the melanoma-bias problem more effectively than a table.

7. **BiomedCLIP zero-shot result.** You have this (68.8% raw accuracy) but it's not prominently discussed. It's worth showing because it demonstrates that even zero-shot embeddings outperform MedGemma's generative approach.

### 4.3 Low priority — nice to have

8. **MedSigLIP zero-shot.** The progress log notes this is missing. Would make the embedding comparison more complete, but MedSigLIP's text encoder may not be straightforward for zero-shot classification.

9. **Statistical significance testing.** The balanced accuracy differences between BiomedCLIP SVM (73.9%) and MedSigLIP SVM (74.0%) are within noise. A paired test or confidence interval overlap analysis would make the "near-tie" claim more rigorous.

10. **Example images from failure categories.** If the dataset license permits, showing 2-3 example histopathology images from the "shared hard cases" bucket would make the failure analysis more tangible.

---

## 5. Poster Structure Recommendation

Based on the rubric (poster design, flow, visual narrative):

**Column 1:**
- Abstract (~80 words)
- Research Question (1-2 sentences)
- Methodology (dataset, models, evaluation protocol — concise with a small pipeline diagram)

**Column 2:**
- Main Results: balanced accuracy bar chart with error bars
- MedGemma bias demonstration: raw vs. balanced accuracy, or a confusion matrix showing 99.4% melanoma recall / 0.7% nevus recall
- Prompt ablation summary: one sentence + key number showing it doesn't help

**Column 3:**
- Failure Analysis: theme breakdown (could be a horizontal bar chart of theme frequencies)
- Spitz sensitivity analysis: small table
- Critical Discussion: 3-4 bullet points on limitations
- Conclusion box: the main takeaway in a highlighted box
- References

---

## 6. Current poster.tex problems to fix

The existing poster.tex has several issues:
1. Uses outdated numbers from before the benchmark refactor
2. Mentions "920 images" instead of 913
3. Missing MedSigLIP entirely
4. Claims "83% hallucinated diagnoses" — this is from the baseline prompt, which is effectively unusable; needs context
5. Says "BiomedCLIP achieves genuine 73% accuracy" without specifying balanced
6. No figures/charts — only tables
7. References are incomplete (missing MedSigLIP paper, Spitz analysis, etc.)
8. Title says "MedGemma vs BiomedCLIP" — should include MedSigLIP
9. The "Why MedGemma fails" bullet about "4B model too small" is speculative and unsupported by the experiments
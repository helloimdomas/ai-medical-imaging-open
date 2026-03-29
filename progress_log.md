# Progress Log

Internal project log for tracking what has been tried, what the current results mean, and what should happen next.

Last updated: 2026-03-29

## Current project question

Primary question:

> For melanoma vs nevus classification on Open-MELON histopathology images, are caption-based medical VLMs competitive with embedding-based approaches?

Current refinement:

> If caption-based MedGemma performs poorly, is that because the generative framing is weak, or because stronger image embeddings are available from models like BiomedCLIP and MedSigLIP?

## Dataset and benchmark definition

- Dataset: `MartiHan/Open-MELON-VL-2.5K`
- Current benchmark source: `captions/captions_cleaned_labeled.jsonl`
- Labeling method:
  - Gemini-based caption cleaning + diagnosis labeling
  - Kept classes:
    - `MELANOMA`
    - `NEVUS`
    - `SPITZ_TUMOR`
  - Excluded classes:
    - `DIFFERENTIAL`
    - `OTHER`
- Current class mapping for binary classification:
  - melanoma class: `MELANOMA`
  - benign class: `NEVUS` + `SPITZ_TUMOR`
- Current benchmark counts:
  - melanoma: 597
  - nevus: 263
  - spitz_tumor: 53
  - total used: 913

Notes:

- The old regex-based subset is obsolete and should not be treated as the source of truth anymore.
- `SPITZ_TUMOR` is currently folded into the benign class for training/evaluation, but it is tracked separately for analysis.
- The benchmark remains imbalanced, so repeated balanced evaluation is necessary.

## What has been built

### Data preparation

- `indices/create_indices.py`
  - Creates the melanoma/nevus subset from Open-MELON captions.
- `clean_captions.py`
  - Rewrites captions to remove non-visual metadata and keep visual content plus diagnosis names.

### MedGemma pipeline

- `pipeline.py`
  - Generates captions with MedGemma
  - Extracts predicted label using keyword matching
  - Scores generated captions with Gemini-based faithfulness/relevance metrics

Prompt variants tried:

- `baseline`
- `binary_choice`

### Embedding pipelines

Refactor completed on 2026-03-28:

- `biomedclip_embeddings.py`
  - Extracts normalized BiomedCLIP image embeddings
- `medsiglip_embeddings.py`
  - Extracts normalized MedSigLIP image embeddings
- `train_embedding_classifier.py`
  - Trains the shared classifier stack on cached embeddings
- `biomedclip_classifier.py`
  - Kept as compatibility wrapper for the original BiomedCLIP baseline

Cached embedding outputs:

- `embeddings/biomedclip_embeddings.npz`
- `embeddings/medsiglip_embeddings.npz`

Current embedding design:

- Embeddings are now extracted for the full dataset once.
- The classifier filters to the current labeled benchmark at train time.
- This avoids re-embedding every time the benchmark definition changes.

## Benchmark status note

The early result sections below were produced before the benchmark moved to the Gemini-labeled `MELANOMA` / `NEVUS` / `SPITZ_TUMOR` setup.

They are still useful for tracking project history, but the current benchmark conclusions should be taken from the sections added at the end of this file:

- post-refactor embedding results
- updated balanced evaluation
- direct BiomedCLIP vs MedSigLIP failure comparison

## Experiments run

### 1. MedGemma baseline prompt

Source:

- `results/baseline/summary.json`

Result:

- Accuracy: 8.7%
- Sensitivity: 10.52%
- Specificity: 4.97%
- Avg faithfulness: 0.154
- Avg relevance: 0.230

Interpretation:

- This prompt is effectively unusable for the task.
- The model often produces outputs that do not map cleanly to melanoma/nevus labels.
- As a diagnostic-style baseline, it fails badly.

### 2. MedGemma binary-choice prompt

Source:

- `results/binary_choice/summary.json`

Result:

- Accuracy: 66.96%
- Sensitivity: 99.35%
- Specificity: 0.66%
- Avg faithfulness: 0.444
- Avg relevance: 0.573

Interpretation:

- The apparent gain in accuracy is misleading.
- The model is close to predicting melanoma for nearly everything.
- This is a classic class-imbalance trap: high melanoma recall, almost no ability to identify nevus correctly.

### 3. BiomedCLIP embeddings + shared classifiers

Source:

- `results/biomedclip_results.json`

Results:

- Zero-shot:
  - Accuracy: 68.80%
  - Sensitivity: 81.23%
  - Specificity: 43.38%
- Supervised LogisticRegression:
  - Accuracy: 70.11%
  - Sensitivity: 74.19%
  - Specificity: 61.67%
- Supervised SVM:
  - Accuracy: 71.20%
  - Sensitivity: 75.00%
  - Specificity: 63.33%
- Supervised RandomForest:
  - Accuracy: 73.91%
  - Sensitivity: 91.94%
  - Specificity: 36.67%

Interpretation:

- BiomedCLIP is much stronger than MedGemma for this task.
- The best balanced tradeoff on the original split appears to be the SVM, not RandomForest.
- RandomForest reaches the highest raw test accuracy, but it does so with poor nevus specificity.

### 4. MedSigLIP embeddings + shared classifiers

Source:

- `results/medsiglip_results.json`

Results:

- Supervised LogisticRegression:
  - Accuracy: 70.65%
  - Sensitivity: 69.35%
  - Specificity: 73.33%
- Supervised SVM:
  - Accuracy: 77.17%
  - Sensitivity: 75.00%
  - Specificity: 81.67%
- Supervised RandomForest:
  - Accuracy: 75.54%
  - Sensitivity: 92.74%
  - Specificity: 40.00%

Interpretation:

- MedSigLIP is currently the strongest embedding model in the repo.
- MedSigLIP + SVM is the best overall supervised result so far.
- Compared with BiomedCLIP, MedSigLIP shows notably stronger nevus specificity when paired with SVM.

### 5. Balanced evaluation across models

Source:

- `results/balanced_accuracy.json`

Balanced accuracy over 10 trials:

| Method | Balanced Accuracy | Std |
| :--- | ---: | ---: |
| MedGemma (binary_choice) | 49.01% | 3.37 |
| BiomedCLIP + LogReg | 71.65% | 2.64 |
| BiomedCLIP + SVM | 73.97% | 3.33 |
| BiomedCLIP + RandomForest | 72.15% | 4.45 |
| MedSigLIP + LogReg | 72.40% | 3.51 |
| MedSigLIP + SVM | 75.95% | 4.07 |
| MedSigLIP + RandomForest | 72.56% | 4.23 |

Interpretation:

- MedGemma falls to near random chance under balanced evaluation.
- The MedGemma binary-choice result was therefore mostly an artifact of the 618/302 class imbalance.
- BiomedCLIP remains strong under balanced sampling.
- MedSigLIP is stronger than BiomedCLIP on the balanced benchmark, with SVM as the best current configuration.

### 6. MedGemma prompt/parameter ablation on fixed nevus subset

Source:

- `results/medgemma_ablation/leaderboard.json`

Setup:

- Model: `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M`
- Task: classify the same fixed set of 10 nevus images
- Subset indices:
  - 33, 171, 224, 329, 354, 355, 356, 390, 394, 685
- Prompt families:
  - `open_descriptive`
  - `open_brief`
  - `closed_binary`
  - `closed_binary_strict`
- Decoding settings:
  - deterministic
  - mild_sampling
  - moderate_sampling

Top results:

- `closed_binary__deterministic`
  - Accuracy: 20.0%
  - Predictions: 2 nevus, 8 melanoma, 0 unknown
- `open_descriptive__deterministic`
  - Accuracy: 10.0%
  - Predictions: 1 nevus, 5 melanoma, 4 unknown
- `closed_binary__mild_sampling`
  - Accuracy: 10.0%
  - Predictions: 1 nevus, 8 melanoma, 1 unknown

Worst results:

- `closed_binary_strict__deterministic`
  - Accuracy: 0.0%
  - Predictions: 0 nevus, 10 melanoma, 0 unknown
- `closed_binary_strict__moderate_sampling`
  - Accuracy: 0.0%
  - Predictions: 0 nevus, 10 melanoma, 0 unknown
- Several open-prompt variants also reached 0.0%, usually by predicting melanoma or unknown for every case.

Interpretation:

- Prompting and decoding changes did not rescue MedGemma on nevus-heavy evaluation.
- Closed prompts were slightly better than open prompts, but still very poor.
- The strongest run still only reached 20% accuracy on 10 nevus images.
- Stricter binary wording made the melanoma bias worse, not better.
- Temperature increases did not improve performance; if anything, they made outputs less reliable.

Conclusion from this ablation:

- MedGemma's failure is not mainly a prompt-engineering problem.
- On this subset, the model is strongly biased toward melanoma even when all test images are nevus.
- This supports the broader result that embedding-based approaches are better than caption-generation for this task.

## Post-refactor results on current benchmark

### 7. Updated BiomedCLIP results on MELANOMA vs (NEVUS + SPITZ_TUMOR)

Source:

- `results/biomedclip_results.json`

Setup:

- Samples used: 913
- Label source: `captions/captions_cleaned_labeled.jsonl`
- Spitz treatment: included in benign class

Results:

- LogisticRegression
  - Accuracy: 68.31%
  - Sensitivity: 74.17%
  - Specificity: 57.14%
- SVM
  - Accuracy: 69.40%
  - Sensitivity: 75.83%
  - Specificity: 57.14%
- RandomForest
  - Accuracy: 69.95%
  - Sensitivity: 87.50%
  - Specificity: 36.51%

Interpretation:

- BiomedCLIP remains a strong baseline, but specificity on the benign class is still limited.
- The SVM is the cleaner tradeoff model.
- RandomForest still inflates melanoma sensitivity at the cost of benign specificity.

### 8. Updated MedSigLIP results on MELANOMA vs (NEVUS + SPITZ_TUMOR)

Source:

- `results/medsiglip_results.json`

Setup:

- Samples used: 913
- Label source: `captions/captions_cleaned_labeled.jsonl`
- Spitz treatment: included in benign class

Results:

- LogisticRegression
  - Accuracy: 63.93%
  - Sensitivity: 65.83%
  - Specificity: 60.32%
- SVM
  - Accuracy: 72.68%
  - Sensitivity: 75.83%
  - Specificity: 66.67%
- RandomForest
  - Accuracy: 72.13%
  - Sensitivity: 88.33%
  - Specificity: 41.27%

Interpretation:

- MedSigLIP still has the strongest single held-out split result.
- The MedSigLIP SVM is the most balanced supervised configuration on the current benchmark.
- Compared with BiomedCLIP SVM, MedSigLIP keeps the same melanoma sensitivity but achieves better benign specificity.

### 9. Updated balanced evaluation on current benchmark

Source:

- `results/balanced_accuracy.json`

Balanced evaluation setup:

- Use all 316 benign cases (`NEVUS + SPITZ_TUMOR`)
- Randomly sample 316 melanomas
- Repeat over 10 trials

Results:

| Method | Balanced Accuracy | Std |
| :--- | ---: | ---: |
| MedGemma (binary_choice) | 50.6% | 3.0 |
| BiomedCLIP + LogReg | 73.1% | 3.0 |
| BiomedCLIP + SVM | 73.9% | 2.9 |
| BiomedCLIP + RandomForest | 72.6% | 2.5 |
| MedSigLIP + LogReg | 71.4% | 4.1 |
| MedSigLIP + SVM | 74.0% | 3.3 |
| MedSigLIP + RandomForest | 71.6% | 5.2 |

Interpretation:

- This is a better metric than raw imbalanced accuracy.
- MedGemma falls to near chance again once the class imbalance advantage is removed.
- BiomedCLIP SVM and MedSigLIP SVM are effectively tied on repeated balanced sampling.
- The current honest story is:
  - MedSigLIP has the best single held-out split result
  - MedSigLIP and BiomedCLIP are nearly neck-and-neck under repeated balanced evaluation

### 10. Direct BiomedCLIP vs MedSigLIP failure comparison on the shared SVM test split

Source:

- `results/svm_failure_comparison.json`

Setup:

- Same exact held-out test split for both models
- Same labels
- Same classifier family: `SVM (RBF)`
- Test size: 183

Counts:

- Both correct: 112
- Both wrong: 35
- MedSigLIP wrong, BiomedCLIP right: 15
- BiomedCLIP wrong, MedSigLIP right: 21

Failed index sets:

- Both wrong:
  - 572, 2152, 1498, 1056, 634, 1629, 1670, 1754, 1026, 1671, 1230, 1641, 1507, 646, 2341, 2334, 1737, 1413, 913, 1845, 1748, 1076, 518, 184, 1923, 1669, 1931, 2049, 1615, 1741, 394, 99, 2442, 622, 788
- MedSigLIP wrong, BiomedCLIP right:
  - 1986, 1103, 398, 286, 1721, 941, 1673, 1638, 1497, 1344, 1049, 312, 1245, 1743, 1336
- BiomedCLIP wrong, MedSigLIP right:
  - 1781, 2498, 1580, 1985, 307, 2497, 2164, 1127, 1861, 2393, 525, 2189, 612, 2120, 2193, 1337, 532, 613, 308, 735, 2394

Pattern summary:

- The overlap in failures is substantial.
  - 35 shared failures means many cases are genuinely hard for both encoders.
- MedSigLIP's edge is real but moderate.
  - It recovers 21 test samples that BiomedCLIP misses.
  - It gives up 15 that BiomedCLIP gets right.
- The failure buckets are not dominated by low-quality images.
  - High-quality and low-quality cases are mixed in all error groups.
- Spitz does not appear to be the main driver of the difference.
  - Shared failures contain 3 Spitz cases.
  - The one-sided failure buckets each contain only 1 Spitz case.
- MedSigLIP-only failures skew somewhat melanoma-heavy.
  - 10 melanoma, 5 benign
- BiomedCLIP-only failures are almost evenly split.
  - 10 melanoma, 11 benign

Interpretation:

- MedSigLIP is not winning because it solves a completely different category of cases.
- It appears to be modestly better across the same evaluation set, while a large hard core of examples defeats both models.

### 11. `llama3.2-vision` on the 35 shared hard cases

Status note:

- This run was executed with the 35 shared hard-case indices using:
  - prompt: `closed_binary`
  - decoding: `deterministic`
  - model: `llama3.2-vision`
- Due to a path bug in `medgemma_prompt_ablation.py`, the run saved into:
  - `results/medgemma_ablation/closed_binary__deterministic/`
- The bug has since been fixed so future non-default configs save into model-specific `results/ollama_ablation/...` folders.

Hard-case set:

- Defined as the 35 samples that both BiomedCLIP SVM and MedSigLIP SVM got wrong on the same held-out test split.

Result on this hard-case set:

- Correct: 4
- Wrong explicit label: 21
- Unknown / degenerate output: 10
- Effective accuracy: 11.43%

Correct indices:

- 634
- 1641
- 2341
- 2049

Key observation:

- All 4 recovered cases were true melanoma.
- `llama3.2-vision` recovered 0 benign hard cases.
- On this subset it does not function as a useful fallback model.

Failure mode counts:

- melanoma undercalled as benign nevus: 12
- benign overcalled as melanoma: 3
- unknown / truncated / degenerate output: 10
- repeated nested-pattern template: 17
- generic benign cue language (`lack of atypia`, `uniform`, `no evidence of atypia`): 10
- repeated `high nuclear-to-cytoplasmic ratio` phrasing: 6
- hallucinated necrosis: 4

Main failure pattern:

- The model does not read these hard cases in a genuinely case-specific way.
- Instead, it falls back to a small set of stock histology phrases:
  - `nested pattern`
  - `uniform cells`
  - `lack of atypia`
  - `high nuclear-to-cytoplasmic ratio`
- This often causes it to erase the discriminative malignant features that are present in the original and cleaned captions.

Representative comparisons:

1. Melanoma undercalled as benign by replacing malignant features with generic benign language

- Index `2152`
- Cleaned caption:
  - `Acral lentiginous melanoma in situ showing increased single unit atypical melanocytes with 3 mitoses in a high-power field.`
- Original caption:
  - `Histopathologic appearance of acral lentiginous melanoma in situ, showing increased single unit atypical melanocytes with 3 mitoses in a high-power field.`
- Generated caption:
  - `The lesion is composed of a proliferation of melanocytes ... arranged in a nested pattern ... uniform in size and shape ... BENIGN NEVUS.`
- Failure:
  - The model deletes the decisive malignant content:
    - atypical melanocytes
    - mitoses
    - acral lentiginous melanoma in situ
  - It replaces that with a generic benign template.

2. Melanoma undercalled by explicitly denying malignant features that are present in the caption

- Index `1026`
- Cleaned caption:
  - `Confluent, severely atypical melanocytes with pagetoid spread. Enlarged, poorly nested and confluent, severely atypical melanocytes.`
- Original caption:
  - `Histopathology of vulvar melanoma in situ ... confluent, severely atypical melanocytes with pagetoid spread.`
- Generated caption:
  - `There is no evidence of pagetoid spread or atypical melanocytes. Diagnosis: Benign Nevus.`
- Failure:
  - This is a direct semantic inversion of the caption evidence.
  - The model is not merely vague; it states the opposite of the key pathological findings.

3. Benign lesion overcalled by hallucinating malignant architecture

- Index `1629`
- Cleaned caption:
  - `Cellular blue nevus demonstrating a dumbbell-shaped appearance.`
- Original caption:
  - `Histopathology of a large Cellular blue nevus ... dumbbell-shaped appearance ... typically lacks overt atypia and high mitotic activity.`
- Generated caption:
  - `The cells are arranged in a nested pattern, with a central area of necrosis ... a few cells showing a degree of atypia. Diagnosis: MELANOMA.`
- Failure:
  - The model invents necrosis and atypia that are not supported by the cleaned or original caption.
  - This is a true hallucinated-malignancy error, not just a thresholding mistake.

4. Spitz tumor overcalled as melanoma

- Index `1748`
- Cleaned caption:
  - `Spitz tumor with focal cellular necrosis.`
- Original caption:
  - `Histopathology showing focal cellular necrosis in a Spitz tumor ...`
- Generated caption:
  - `collection of melanocytes with variable-sized and/or multiple nuclei ... Diagnosis: MELANOMA.`
- Failure:
  - The model appears unable to keep the Spitz category distinct under the forced binary prompt.
  - Presence of a risk-associated feature such as necrosis seems to push it into melanoma.

5. Degenerate / truncated output instead of a usable diagnosis

- Index `913`
- Cleaned caption:
  - `Increased number of melanocytes forming nests at the dermoepidermal junction, presenting as single cells with mildly enlarged, hyperchromatic nuclei.`
- Original caption:
  - `classic lentigo maligna, also known as melanoma in situ ... increased number of melanocytes ... mildly enlarged, hyperchromatic nuclei`
- Generated caption:
  - `The lesion exhibits a lack of architectural disorganization ...` followed by degeneration into repeated filler / truncation.
- Failure:
  - The model starts with a generic benign narrative and then collapses into unusable text.
  - Similar degeneration occurs for indices `184`, `1923`, `1669`, `1615`, `1741`, `394`, `2442`, `622`, and `788`.

Overall interpretation of the hard-case run:

- `llama3.2-vision` does not meaningfully rescue the samples that both embedding models miss.
- Its dominant failure is not a melanoma-overcall bias like MedGemma.
- Instead, it more often:
  - undercalls melanoma as benign by replacing malignant findings with generic benign morphology
  - produces templated, low-specificity histology prose
  - collapses into truncated or degenerate text on difficult cases
- This makes it weak as a fallback for difficult pathology cases even when its outputs look superficially more descriptive than MedGemma's.

### 12. Pathology-specific anti-melanoma-bias prompt ablation

Source:

- `results/pathology_prompt_ablation/20260329_140445/`

Setup:

- Models:
  - `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M`
  - `llama3.2-vision`
- Prompts:
  - `closed_binary`
  - `pathology_anti_bias`
- Decoding:
  - deterministic only
- Subsets:
  - `first10_nevus`
  - `first10_melanoma`
  - `hard_cases_shared_failures`

Prompt idea:

- Ask the model to first list features supporting benignity, then features supporting melanoma.
- Explicitly warn against diagnosing melanoma unless multiple malignant features are present.
- Force the final label to one of:
  - `MELANOMA`
  - `BENIGN NEVUS`

Results table:

| Subset | Model | `closed_binary` | `pathology_anti_bias` |
| :--- | :--- | ---: | ---: |
| first10_nevus | MedGemma | 0/10 (0%) | 8/10 (80%) |
| first10_melanoma | MedGemma | 10/10 (100%) | 3/10 (30%) |
| hard_cases_shared_failures | MedGemma | 18/35 (51.4%) | 16/35 (45.7%) |
| first10_nevus | `llama3.2-vision` | 10/10 (100%) | 0/10 (0%, all unknown) |
| first10_melanoma | `llama3.2-vision` | 0/10 (0%) | 1/10 (10%, mostly unknown) |
| hard_cases_shared_failures | `llama3.2-vision` | 15/35 (42.9%) | 2/35 (5.7%, 30 unknown) |

Detailed behavior:

- MedGemma, nevus subset:
  - The anti-bias prompt strongly reduced melanoma overcalling.
  - Accuracy improved from 0/10 to 8/10.
- MedGemma, melanoma subset:
  - The same prompt badly reduced melanoma recall.
  - Accuracy fell from 10/10 to 3/10.
- MedGemma, hard cases:
  - The anti-bias prompt did not help overall.
  - Accuracy dropped slightly from 18/35 to 16/35.

- `llama3.2-vision`, all subsets:
  - The anti-bias prompt was harmful.
  - The model frequently produced long rubric-style outputs that did not resolve to a clean final diagnosis.
  - This caused a large `unknown` collapse:
    - 10/10 unknown on the nevus subset
    - 9/10 unknown on the melanoma subset
    - 30/35 unknown on the hard-case subset

Qualitative interpretation:

- MedGemma is clearly prompt-sensitive.
  - The pathology anti-bias prompt can move it away from the default melanoma-overcall behavior.
  - But this is not a real fix.
  - It mostly trades one bias for another:
    - from melanoma-overcalling
    - to melanoma-undercalling
- The generated MedGemma outputs are also highly templated.
  - On many samples the model repeats nearly the same benign-vs-malignant bullet list regardless of image content.
  - The prompt appears to steer the decision prior more than it improves true case-specific reading.

- `llama3.2-vision` reacts differently.
  - The longer pathology anti-bias prompt causes the model to spend its output budget reciting the requested rubric.
  - It often fails to produce a parsable final diagnosis.
  - This makes it much worse than the simpler binary prompt in this setup.

Conclusion from this ablation:

- The anti-melanoma-bias prompt is worth keeping as evidence that prompt framing can substantially change MedGemma's output distribution.
- But it does not make MedGemma reliable; it overcorrects.
- For `llama3.2-vision`, the anti-bias prompt is actively bad.
- The broader project conclusion remains unchanged:
  - prompt engineering can move generative behavior
  - but it does not close the gap to embedding-based classifiers on this task

## Current best result

Current best single held-out split:

- MedSigLIP + SVM
  - Accuracy: 72.68%
  - Sensitivity: 75.83%
  - Specificity: 66.67%

Current best repeated balanced result:

- MedSigLIP + SVM: 74.0% ± 3.3
- BiomedCLIP + SVM: 73.9% ± 2.9

Current overall interpretation:

- Embedding-based approaches are still clearly better than caption-generation for this task.
- MedGemma remains near chance under balanced evaluation.
- MedSigLIP currently has the strongest single split result.
- Under repeated balanced evaluation, MedSigLIP and BiomedCLIP are extremely close.

Best current balanced result:

- 75.95% +/- 4.07 balanced accuracy

Why this matters:

- It currently gives the clearest answer to the project question.
- The best-performing approach is not caption generation.
- The best-performing approach is an embedding model followed by a simple supervised classifier.

## Main conclusions so far

1. Caption-based MedGemma is not a reliable classifier for this benchmark.
2. The binary-choice MedGemma prompt gave a misleading result because of class imbalance.
3. Embedding-based approaches clearly outperform MedGemma on this task.
4. MedSigLIP embeddings appear more useful than BiomedCLIP embeddings for melanoma vs nevus classification in this benchmark.
5. Among tested classifiers, SVM is currently the strongest and most balanced choice for both embedding models.
6. Prompt and decoding ablations do not materially improve MedGemma on a fixed nevus subset; the melanoma bias remains dominant.

## Important caveats

1. Labels are derived from caption keywords, not official dataset annotations.
2. The benchmark is histopathology-specific, so conclusions should stay scoped to this task and dataset.
3. Current classifier results depend on one train/test split for the main supervised result files.
4. Balanced accuracy is more trustworthy than raw accuracy for comparing methods here.
5. MedSigLIP does not currently have a zero-shot result in the repo; the comparison is supervised embedding quality only.
6. The 10-image nevus subset used for the MedGemma ablation is a practical diagnostic set, not a fully audited benchmark.

## Engineering notes

- MedSigLIP local benchmark on M1 Pro / 32 GB was successful.
- Full non-quantized MedSigLIP is feasible locally.
- Approximate practical embedding speed observed on cached local model:
  - about 2.0 to 2.2 images/sec on MPS
- Best MedSigLIP batch size on this machine appears to be around 1 to 2 images; larger batches did not improve throughput much.

## What was fixed during the current refactor

- Old BiomedCLIP embedding cache was incomplete (680 samples), so it was regenerated to 920 samples.
- Balanced evaluation was updated to include MedSigLIP alongside BiomedCLIP and MedGemma.
- The repo was refactored so embedding extraction and classifier training are separated cleanly.
- `uv` setup was updated for reproducibility.

## Recommended next steps

### High priority

1. Add MedSigLIP results to any internal comparison tables or figures going forward.
2. Decide whether the main project question should now explicitly compare:
   - MedGemma captioning
   - BiomedCLIP embeddings
   - MedSigLIP embeddings
3. Add one robustness check beyond the current balanced evaluation:
   - repeated train/test splits
   - different random seeds
   - reduced training set sizes

### Medium priority

1. Consider adding MedSigLIP zero-shot classification if there is a fair and stable way to do it.
2. Add one more pathology-relevant embedding baseline if time allows.
3. Record exact commands used for each experiment in this file or a companion runbook.

### Low priority

1. Clean up older result summaries and remove any outdated claims that say BiomedCLIP is the best model.
2. Add small helper scripts or notebooks for visualizing confusion matrices and comparison charts.

## Suggested working narrative for now

Working claim:

> On the Open-MELON melanoma-vs-nevus subset, caption-based MedGemma is not competitive with embedding-based pipelines. Among the tested embedding models, MedSigLIP + SVM currently performs best and remains strong under balanced evaluation, suggesting that the main weakness is not just prompt wording but the generative framing itself.

This is the best current internal summary. It should stay provisional until at least one more robustness analysis is added.

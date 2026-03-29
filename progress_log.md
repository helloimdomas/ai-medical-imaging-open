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
- Binary subset creation: keyword matching on captions
- Classes used:
  - melanoma: 618
  - nevus: 302
- Total benchmark size: 920 images
- Ground-truth subset file: `indices/melanoma_nevus_indices.json`

Notes:

- This benchmark is derived from caption keywords, not official dataset class labels.
- Cases mentioning both melanoma and nevus are excluded as ambiguous.
- The class distribution is imbalanced, so balanced evaluation is necessary.

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

## Current best result

Best current model:

- MedSigLIP + SVM

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

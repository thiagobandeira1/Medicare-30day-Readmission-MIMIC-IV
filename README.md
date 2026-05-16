# Predicting 30-Day Hospital Readmission in Medicare Patients

> **An interpretable gradient-boosting pipeline on MIMIC-IV v3.1, with a strict 80/20 patient-grouped train/test protocol and an inner-validation slice for early stopping. Test set is touched exactly once.**

Companion repository for the manuscript by **Thiago Bandeira**, **Armando Gonzalez**, and **Dr. Christian Poellabauer** (Knight Foundation School of Computing and Information Sciences, Florida International University).

---

## Headline result

Under a strict 80/20 patient-grouped split with a 10% inner-validation slice carved from the training data for early stopping — **the test set evaluated exactly once** — every individual GBM family and the scipy-optimised blend **meet or exceed** the original capstone targets:

| Model | Current test AUROC | original capstone target | Δ |
|---|---:|---:|---:|
| **XGBoost** (deployed) | **0.7935** | 0.7931 | **+0.0004** |
| LightGBM (co-equal) | 0.7970 | 0.7901 | +0.0069 |
| CatBoost | 0.7937 | 0.7924 | +0.0013 |
| HistGradientBoosting | 0.7943 | 0.7916 | +0.0027 |
| 4-GBM scipy-blend | 0.7968 | 0.7948 | +0.0020 |

Original capstone 5-fold-CV stability reference: **0.7956 ± 0.0026**. Our single-split point estimates land **inside the capstone stability band** despite using the stricter no-leakage protocol.

A reproduction validator is wired into **§10.3b** of the publication notebook — see [`notebooks/Capstone_Final_Notebook.ipynb`](notebooks/Capstone_Final_Notebook.ipynb).

---

## Repository layout

| Path | Purpose |
|---|---|
| `notebooks/Capstone_Final_Notebook.ipynb` | Source-of-truth analysis notebook (18 sections, end-to-end reproducible) |
| `src/` | All Python scripts — refactor utilities + the training pipelines that decouple heavy compute from the notebook |
| `results/` | JSON / CSV result artefacts (predictions `.npz` and model `.pkl` files are regenerated and gitignored) |
| `figures/` | Output figures from the notebook + analyses |
| `paper/` | Manuscript materials — references and section drafts for the paper |
| `docs/` | Methodology deep-dives and reference notes |
| `data/` | **Empty placeholder.** MIMIC-IV is licensed by PhysioNet and cannot be redistributed |
| `environment.yml` / `requirements.txt` | Pinned dependencies (Python 3.12) |
| `CITATION.cff` | Citation metadata |
| `LICENSE` | MIT (code only — MIMIC-IV data has its own license) |

---

## Methodology — what makes this peer-review defensible

The original capstone notebook used 80/20 train/test with `eval_set=(X_test, y_test)` for early stopping — i.e. the test set leaked into model selection. **This publication corrects that** while preserving the capstone numerical results:

1. **Patient-grouped 80/20 outer split.** Identical to the original capstone (n_train = 195,385, n_test = 49,191, no patient appears in both partitions).
2. **10% inner-validation slice carved from the training data** (n_inner_val ≈ 19,115; n_pure_train ≈ 176,270). Used for early stopping and blend-weight selection.
3. **Test set is touched exactly once** — at final evaluation in §10.2 / §10.3b. No `eval_set=(X_test, ...)` anywhere in the codebase.
4. **10-seed averaging** per GBM family (seeds 42…51) to reduce variance.
5. **5-fold patient-grouped CV stability check** — provides apples-to-apples comparison with the original capstone report's 0.7956 ± 0.0026 stability estimate.
6. **Subprocess-per-model isolation** for training — works around an XGBoost 3.2.0 GIL issue that crashes nbclient under sustained 10-seed × 4-model loads.
7. **Missing-data handling explicitly documented in §5.3** — the V7 dataset preserves `NaN` where missingness is itself a clinical signal (most labs and vitals); tree models route `NaN` natively at split time.

---

## Reproducing the results

### 1. Prerequisites

- Python 3.12 (3.12.10 used for the publication run)
- A credentialed PhysioNet account and a local copy of MIMIC-IV v3.1 parquet files
- ~16 GB RAM (full run)
- ~30 min compute on a modern CPU (GBM training); ~15 sec for the notebook itself (everything else loads from `results/`)

### 2. Set up the environment

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate medicare-readmit-pub

# OR plain pip
pip install -r requirements.txt
```

### 3. Build the training tables

MIMIC-IV cannot be redistributed. Construct the V1 → V7 training tables via the upstream DuckDB feature-engineering pipeline (see `docs/missing_data_section_for_report.md` for the V7 feature definitions) and place them at:

```
<MEDICARE_READMIT_BASE_DIR>/Dataset/mimic-parquet/training_table_v{1,2,3,6_full}.parquet
<MEDICARE_READMIT_BASE_DIR>/training_table_v7.parquet # V7 (the 50-feature parsimonious set)
```

The V7 file is at the project root, not in `Dataset/mimic-parquet/`, because the original upstream pipeline persisted it there.

### 4. Point the pipeline at your data

```bash
export MEDICARE_READMIT_BASE_DIR=/path/to/folder/containing/Dataset
```

(Windows PowerShell: `$env:MEDICARE_READMIT_BASE_DIR = "..."`.)

### 5. Run

```bash
# Stage 1 — V1→V7 baseline progression (~3 min, subprocess per version)
python src/run_progression.py

# Stage 2 — V7 4-GBM ensemble × 10 seeds (~10 min, subprocess per model)
python src/run_v7_ensemble.py

# Optional — 5-fold patient-grouped CV stability check (~30 min)
python src/run_5fold_cv.py

# Stage 3 — render the publication notebook (~15 sec, reads everything above from disk)
python src/execute_notebook.py
```

The notebook writes outputs back to `notebooks/Capstone_Final_Notebook.ipynb` and PNGs to `figures/`.

---

## Citation

If you build on this work, please cite using the metadata in [`CITATION.cff`](CITATION.cff):

```
Bandeira, T., Gonzalez, A., & Poellabauer, C. (2026).
Predicting 30-Day Hospital Readmission in Medicare Patients:
An Interpretable Gradient-Boosting Pipeline on MIMIC-IV v3.1.
```

## Acknowledgments

This work was completed as the capstone project for the MS in Data Science & AI at Florida International University, under the mentorship of **Dr. Christian Poellabauer**. We gratefully acknowledge the MIT Laboratory for Computational Physiology and the PhysioNet team for maintaining and curating the MIMIC-IV database.

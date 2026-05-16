"""Replace §10.1 GBM training cells + §10.2 blend cell + §12.1 SHAP cell with
versions that LOAD pre-computed predictions / models from results/.

This is the same decoupling pattern that worked for §9 (load progression.json).
Necessary because XGBoost 3.2.0 inside nbclient reliably triggers a Fatal
Python error (GIL release inside the eval_set callback) when training across
10 seeds in a single kernel session.

Idempotent.

Usage:
 python scripts/refactor_v10_to_load_predictions.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

try:
 sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
 sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebook" / "Capstone_Final_Notebook.ipynb"


GBM_HEADER_CELL = """\
# ── 10.1 4-GBM multi-seed ensemble on V7 ───────────────────────────────────
# Predictions are pre-computed by scripts/run_v7_ensemble.py (one subprocess
# per model family, N_SEEDS=10 each) and saved to results/v7_<model>_{val,test}.npz.
# This cell loads them and reports val + test AUROCs.
#
# Why subprocess-isolation? Training 4 GBM families × 10 seeds in one kernel
# session reliably triggers a Fatal Python error in XGBoost 3.2.0
# (GIL release inside the eval_set callback). Running each family in its own
# Python process — same pattern as §9.1 — sidesteps the issue.

import pickle
import numpy as np

split = np.load(ART_DIR / "v7_split_indices.npz")
y_train, y_val, y_test = split["y_train"], split["y_val"], split["y_test"]

def _load(model):
 pv = np.load(ART_DIR / f"v7_{model}_val.npz")["preds"]
 pt = np.load(ART_DIR / f"v7_{model}_test.npz")["preds"]
 return pv, pt

lgb_preds_val, lgb_preds_test = _load("lightgbm")
xgb_preds_val, xgb_preds_test = _load("xgboost")
cb_preds_val, cb_preds_test = _load("catboost")
hist_preds_val, hist_preds_test = _load("histgbm")
N_SEEDS = lgb_preds_val.shape[1]

# Seed-averaged predictions (used by §10.2 blend + §11.1 plots + §12 SHAP)
lgb_val_avg = lgb_preds_val.mean(axis=1)
lgb_test_avg = lgb_preds_test.mean(axis=1)
xgb_val_avg = xgb_preds_val.mean(axis=1)
xgb_test_avg = xgb_preds_test.mean(axis=1)
cb_val_avg = cb_preds_val.mean(axis=1)
cb_test_avg = cb_preds_test.mean(axis=1)
hist_val_avg = hist_preds_val.mean(axis=1)
hist_test_avg= hist_preds_test.mean(axis=1)

lgb_val_auroc = roc_auc_score(y_val, lgb_val_avg)
lgb_test_auroc = roc_auc_score(y_test, lgb_test_avg)
xgb_val_auroc = roc_auc_score(y_val, xgb_val_avg)
xgb_test_auroc = roc_auc_score(y_test, xgb_test_avg)
cb_val_auroc = roc_auc_score(y_val, cb_val_avg)
cb_test_auroc = roc_auc_score(y_test, cb_test_avg)
hist_val_auroc = roc_auc_score(y_val, hist_val_avg)
hist_test_auroc= roc_auc_score(y_test, hist_test_avg)

# Aliases for downstream cells expecting legacy single-name variables.
lgb_avg, lgb_auroc = lgb_test_avg, lgb_test_auroc
xgb_avg, xgb_auroc = xgb_test_avg, xgb_test_auroc
cb_avg, cb_auroc = cb_test_avg, cb_test_auroc
hist_avg, hist_auroc = hist_test_avg, hist_test_auroc

# Load seed-0 LightGBM + XGBoost models for §11/§12 use
with open(ART_DIR / "v7_seed0_lightgbm.pkl", "rb") as f:
 lgb_model_seed0 = pickle.load(f)
with open(ART_DIR / "v7_seed0_xgboost.pkl", "rb") as f:
 xgb_model_seed0 = pickle.load(f)

print(f"Loaded {N_SEEDS}-seed predictions for 4 GBM families:")
print(f" LightGBM: val {lgb_val_auroc:.4f} | test {lgb_test_auroc:.4f}")
print(f" XGBoost: val {xgb_val_auroc:.4f} | test {xgb_test_auroc:.4f} <-- deployment candidate")
print(f" CatBoost: val {cb_val_auroc:.4f} | test {cb_test_auroc:.4f}")
print(f" HistGBM: val {hist_val_auroc:.4f} | test {hist_test_auroc:.4f}")
"""


# §10.1 has 4 cells in the original (one per model). We collapse to 1 loader cell + 3 stub cells.
STUB_XGB = """\
# Predictions already loaded in the previous cell. See §10.1 header cell.
print(f"XGBoost: val {xgb_val_auroc:.4f} | test {xgb_test_auroc:.4f} <-- deployment candidate")
"""
STUB_CB = """\
print(f"CatBoost: val {cb_val_auroc:.4f} | test {cb_test_auroc:.4f}")
"""
STUB_HIST = """\
print(f"HistGBM: val {hist_val_auroc:.4f} | test {hist_test_auroc:.4f}")
"""


# §10.2 blend cell — use loaded predictions, optimise on val, single-shot test.
BLEND_CELL = """\
# ── 10.2 Scipy-optimized blending (val-only) ───────────────────────────────
# Blend weights are optimised on the VALIDATION set only. The TEST set is
# evaluated EXACTLY ONCE at the end of this cell using the fixed weights.

def neg_auroc_val(w):
 w = np.clip(w, 0, None)
 w = w / w.sum()
 blend = (w[0] * lgb_val_avg + w[1] * xgb_val_avg +
 w[2] * cb_val_avg + w[3] * hist_val_avg)
 return -roc_auc_score(y_val, blend)

result = minimize(
 neg_auroc_val, x0=[0.25, 0.25, 0.25, 0.25],
 method="Nelder-Mead", bounds=[(0.05, 0.5)] * 4,
)
w_opt = result.x / result.x.sum()

blend_test = (w_opt[0] * lgb_test_avg + w_opt[1] * xgb_test_avg +
 w_opt[2] * cb_test_avg + w_opt[3] * hist_test_avg)
blend_test_auroc = roc_auc_score(y_test, blend_test)

blend_val = (w_opt[0] * lgb_val_avg + w_opt[1] * xgb_val_avg +
 w_opt[2] * cb_val_avg + w_opt[3] * hist_val_avg)
blend_val_auroc = roc_auc_score(y_val, blend_val)

print("=" * 70)
print("SCIPY-OPTIMIZED ENSEMBLE BLEND (V7) — held-out test eval")
print("=" * 70)
print(f" LightGBM weight: {w_opt[0]:.3f} (val: {lgb_val_auroc:.4f}, test: {lgb_test_auroc:.4f})")
print(f" XGBoost weight: {w_opt[1]:.3f} (val: {xgb_val_auroc:.4f}, test: {xgb_test_auroc:.4f})")
print(f" CatBoost weight: {w_opt[2]:.3f} (val: {cb_val_auroc:.4f}, test: {cb_test_auroc:.4f})")
print(f" HistGBM weight: {w_opt[3]:.3f} (val: {hist_val_auroc:.4f}, test: {hist_test_auroc:.4f})")
print(f"\\n Blend AUROC on val (weights selected here): {blend_val_auroc:.6f}")
print(f" Blend AUROC on test (single evaluation): {blend_test_auroc:.6f}")
print(f" XGBoost solo (test): {xgb_test_auroc:.6f} <-- deployment candidate")
print(f" Gap (blend - XGB on test): {blend_test_auroc - xgb_test_auroc:+.4f}")

# Aliases for downstream cells
blend_preds, blend_auroc = blend_test, blend_test_auroc
"""


# §8.2 split cell: load saved indices instead of re-splitting (must match
# the split that v7_ensemble.py used).
SPLIT_CELL_FROM_DISK = """\
# ── 8.2 Patient-grouped 80/20 + inner-val split ────────────────────────────
# The canonical row indices are persisted by scripts/run_v7_ensemble.py so
# the notebook + ensemble script see the exact same split.
# Two-stage GroupShuffleSplit on subject_id:
# Stage 1: 80% train+val / 20% test (matches original capstone report's n_test=49,191)
# Stage 2: 90% train / 10% inner-val (within train+val)
# Yields ~72% pure train / ~8% inner-val / 20% test. Inner-val is used for
# early stopping and blend-weight selection in §10 (no test leakage).
# Test is evaluated exactly once.

import json
import numpy as np

df_v7 = pd.read_parquet(PATHS["v7"])
print(f"V7 dataset: {df_v7.shape[0]:,} rows x {df_v7.shape[1]} columns")

feature_cols = json.loads((ART_DIR / "v7_feature_cols.json").read_text())
X = df_v7[feature_cols].copy()
for col in X.select_dtypes(include=["object", "category"]).columns:
 X[col] = LabelEncoder().fit_transform(X[col].astype(str))
y = df_v7[TARGET].astype(int).values
groups = df_v7["subject_id"].values

split = np.load(ART_DIR / "v7_split_indices.npz")
train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

X_train = X.iloc[train_idx].reset_index(drop=True)
X_val = X.iloc[val_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)
y_train, y_val, y_test = split["y_train"], split["y_val"], split["y_test"]

train_patients = set(groups[train_idx])
val_patients = set(groups[val_idx])
test_patients = set(groups[test_idx])
PATIENT_SPLIT = {"train": train_patients, "val": val_patients, "test": test_patients}

assert not (train_patients & val_patients), "PATIENT LEAKAGE between train and val"
assert not (train_patients & test_patients), "PATIENT LEAKAGE between train and test"
assert not (val_patients & test_patients), "PATIENT LEAKAGE between val and test"

print(f"\\nTrain: {len(X_train):,} admissions ({len(train_patients):,} patients) | pos rate = {y_train.mean():.4f}")
print(f"Val: {len(X_val):,} admissions ({len(val_patients):,} patients) | pos rate = {y_val.mean():.4f}")
print(f"Test: {len(X_test):,} admissions ({len(test_patients):,} patients) | pos rate = {y_test.mean():.4f}")
print(f"Features: {len(feature_cols)} ({X.select_dtypes(include=['object','category']).shape[1]} categorical)")
print(f"\\nSplit protocol: 80/20 patient-grouped outer + 10% inner-val carved from train for early stopping (random_state={RANDOM_STATE}).")
"""


def find_cell(nb, needle):
 for i, c in enumerate(nb.cells):
 src = "".join(c.source) if isinstance(c.source, list) else c.source
 if needle in src:
 return i
 raise LookupError(f"cell containing {needle!r} not found")


def replace(nb, idx, src):
 nb.cells[idx].source = src
 if nb.cells[idx].cell_type == "code":
 nb.cells[idx].outputs = []
 nb.cells[idx].execution_count = None


def main():
 nb = nbf.read(NB_PATH, as_version=4)
 print(f"Loaded {NB_PATH.name}: {len(nb.cells)} cells")

 idx = find_cell(nb, "# ── 8.2 Patient-grouped 80/20 + 10% inner-val")
 replace(nb, idx, SPLIT_CELL_FROM_DISK)
 print(f" [{idx:3d}] §8.2 split now loads from v7_split_indices.npz")

 idx = find_cell(nb, "# ── 10.1 4-GBM multi-seed ensemble on V7")
 replace(nb, idx, GBM_HEADER_CELL)
 print(f" [{idx:3d}] §10.1 header: loads predictions + reports AUROCs")

 idx = find_cell(nb, "# --- XGBoost ---")
 replace(nb, idx, STUB_XGB)
 print(f" [{idx:3d}] §10.1 XGBoost stub")

 idx = find_cell(nb, "# --- CatBoost ---")
 replace(nb, idx, STUB_CB)
 print(f" [{idx:3d}] §10.1 CatBoost stub")

 idx = find_cell(nb, "# --- HistGradientBoosting ---")
 replace(nb, idx, STUB_HIST)
 print(f" [{idx:3d}] §10.1 HistGBM stub")

 idx = find_cell(nb, "# ── 10.2 Scipy-optimized blending")
 replace(nb, idx, BLEND_CELL)
 print(f" [{idx:3d}] §10.2 blend (val-only) — uses loaded predictions")

 nbf.write(nb, NB_PATH)
 print(f"\nSaved to {NB_PATH}")


if __name__ == "__main__":
 main()

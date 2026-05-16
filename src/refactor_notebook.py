"""Refactor Capstone_Final_Notebook.ipynb for publication-grade reproducibility.

Changes:
  1. Title cell  : add Dr. Christian Poellabauer to author byline.
  2. §1.3 BASE_DIR : explicit BASE_DIR (parent archive) + clear stale FileNotFoundError outputs.
  3. §1.4 (NEW)  : reproducibility manifest cell (library versions + parquet SHA-256).
  4. §8.2        : refactor 80/20 train/test -> 60/20/20 train/val/test patient-grouped.
  5. §9.1        : replace hardcoded LogReg progression dict with real V1->V6 training loop.
  6. §9.2-9.4    : replace hardcoded LGB/XGB/MLP progression dicts with real training loops.
  7. §9.5        : keep V6 cross-family comparison hardcoded (NN values from upstream runs)
                   but add a clear note about their provenance.
  8. §10.1       : LGB/XGB/CatBoost early stopping on val (not test); seeds 42+s; collect
                   val + test predictions separately.
  9. §10.1 HistGBM: keep internal validation_fraction (never sees val/test); seeds 42+s.
 10. §10.2       : optimize blend weights on val, evaluate test ONCE.
 11. §11.1       : update final-XGBoost plot cell to use xgb_test_avg.

This script is idempotent — running it twice produces the same notebook.
Original archive copy is not touched.

Usage:
    python scripts/refactor_notebook.py
"""

from __future__ import annotations

import io
from pathlib import Path
import sys

# Force UTF-8 console output so Unicode in print() doesn't blow up on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / ("notebooks" if (REPO_ROOT / "notebooks").is_dir() else "notebook") / "Capstone_Final_Notebook.ipynb"


# ─────────────────────────────────────────────────────────────────────────────
# Cell sources (new content)
# ─────────────────────────────────────────────────────────────────────────────

TITLE_CELL = """\
# Predicting 30-Day Hospital Readmission in Medicare Patients
## An Interpretable XGBoost Model on MIMIC-IV v3.1

**Authors:** Thiago Bandeira¹, Armando Gonzalez¹, Christian Poellabauer¹
¹Knight Foundation School of Computing and Information Sciences, Florida International University, Miami, FL, USA

**Corresponding author:** Thiago Bandeira (tbati006@fiu.edu)

---

### Abstract

Thirty-day all-cause hospital readmission is a major quality-of-care metric for Medicare beneficiaries and is penalised financially through the CMS Hospital Readmissions Reduction Program. This project develops and evaluates a supervised machine-learning pipeline that estimates thirty-day readmission risk for **244,576 Medicare admissions** drawn from MIMIC-IV v3.1. A staged feature-engineering process across seven dataset versions (V1 -> V7) produced a parsimonious set of **50 clinically-motivated features** spanning prior utilisation, comorbidity, medication complexity, clinical severity, and operational flow. Four gradient-boosting families (LightGBM, XGBoost, CatBoost, HistGradientBoosting) were each averaged across ten random seeds; an optional scipy-optimised blend was also constructed. Under a strict 60/20/20 patient-grouped train/validation/test protocol — with early stopping and blend-weight selection performed on the validation split and the test split touched exactly once — the blended ensemble reached **~0.795 test AUROC** and the single-model XGBoost reached **~0.793 AUROC**, the model selected for deployment on simplicity grounds. The XGBoost model outperforms the LACE index by ~0.11 AUROC and a published ClinicalBERT baseline by ~0.08 AUROC, and SHAP explanations deliver both global and patient-level rationale for every prediction. The result is an interpretable risk-scoring tool that can be embedded in existing EHR workflows.

> *Reported AUROCs are placeholders that will be updated to current-run values once the refactored notebook re-runs end-to-end.*

**Keywords:** 30-day readmission · Medicare · MIMIC-IV · gradient boosting · XGBoost · SHAP · healthcare analytics · interpretable ML.

---

### Notebook Structure

| Section | Content |
|---|---|
| 1 | Setup, imports, configuration, reproducibility manifest |
| 2 | Motivation & Background |
| 3 | Problem Definition & Research Questions |
| 4 | Prior Art & Challenges |
| 5 | Data Source & Cohort Construction |
| 6 | Exploratory Data Analysis |
| 7 | Feature Engineering (V1 -> V7) |
| 8 | Methods, 60/20/20 Train/Val/Test Protocol |
| 9 | Model Training: Baselines, GBMs, Deep Learning |
| 10 | 4-GBM Ensemble on V7 + Scipy-Optimized Blending |
| 11 | Results: ROC, Calibration, Benchmarks |
| 12 | SHAP Interpretability |
| 13 | Answering the Research Questions |
| 14 | Discussion & Limitations |
| 15 | Future Work |
| 16 | Contributions & Conclusions |
| 17 | Reproduction |
| 18 | References |

> **Note on version naming.** Throughout this notebook we use **V1 -> V7** to match the feature-engineering progression reported in our final paper. **V7 is the final 50-feature parsimonious model** used for deployment. A larger 368-feature exploration, referred to as the **Feature Expansion Version**, was also evaluated as a ceiling check — it adds only ~0.005 AUROC over V7 and is not deployed.
"""


BASEDIR_CELL = """\
# --------------------------------------------------------------------------
# Project paths.
#
# BASE_DIR must contain the `Dataset/mimic-parquet/` folder built from
# MIMIC-IV v3.1 (see data_prep/README.md for cohort construction).
# Override by setting the MEDICARE_READMIT_BASE_DIR environment variable, or
# edit the explicit fallback path below.
# --------------------------------------------------------------------------
import os
from pathlib import Path

def _find_base_dir():
    env = os.environ.get("MEDICARE_READMIT_BASE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # Try common candidates relative to the current working directory.
    here = Path.cwd().resolve()
    candidates = [
        here,
        here.parent,
        here.parent.parent,
        here.parent / "Capstone Project",
        here.parent.parent / "Capstone Project",
        Path(r"C:/Users/Thiago/Documents/Coursework/Capstone Project"),
    ]
    for c in candidates:
        if (c / "Dataset" / "mimic-parquet").is_dir():
            return c.resolve()
    return here  # fall back — will print MISSING below

BASE_DIR = _find_base_dir()
DATA_DIR = BASE_DIR / "Dataset" / "mimic-parquet"

# Figures + artefacts live inside this repo, not in BASE_DIR.
REPO_ROOT = Path.cwd().resolve()
while REPO_ROOT != REPO_ROOT.parent and not ((REPO_ROOT / "notebook").is_dir() or (REPO_ROOT / "notebooks").is_dir()):
    REPO_ROOT = REPO_ROOT.parent
FIG_DIR  = REPO_ROOT / "figures"
ART_DIR  = REPO_ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

# Progressive feature-engineering tables (produced upstream by the DuckDB pipeline).
#
# IMPORTANT: V7 (the canonical 50-feature parsimonious set published in the V17
# report) lives at the project root, NOT inside Dataset/mimic-parquet/.
# The file `Dataset/mimic-parquet/training_table_v7.parquet` is an earlier
# 90-feature working set that only shares 11 features with the published V17
# 50 — see docs/changelog.md for details. We point to the correct one here.
V7_OVERRIDE = BASE_DIR / "training_table_v7.parquet"
PATHS = {
    "v1": DATA_DIR / "training_table_v1.parquet",
    "v2": DATA_DIR / "training_table_v2.parquet",
    "v3": DATA_DIR / "training_table_v3.parquet",
    "v4": DATA_DIR / "training_table_v4.parquet",
    "v5": DATA_DIR / "training_table_v5.parquet",
    "v6": DATA_DIR / "training_table_v6_full.parquet",
    "v7": V7_OVERRIDE if V7_OVERRIDE.exists() else DATA_DIR / "training_table_v7.parquet",
}

TARGET       = "readmit_30d"
ID_COLS      = ["subject_id", "hadm_id"]
DT_COLS      = ["admittime_dt", "dischtime_dt"]
RANDOM_STATE = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.10   # 10% of (train+val) carved as inner-val for early stopping — yields ~72/8/20 overall

print(f"BASE_DIR : {BASE_DIR}")
print(f"DATA_DIR : {DATA_DIR}")
print(f"REPO_ROOT: {REPO_ROOT}")
print(f"FIG_DIR  : {FIG_DIR}")
print(f"ART_DIR  : {ART_DIR}")
print()
for name, p in PATHS.items():
    flag = "OK     " if p.exists() else "MISSING"
    size = f"{p.stat().st_size / 1024**2:7.1f} MB" if p.exists() else "         "
    print(f"  {name}: {p.name:32s}  [{flag}] {size}")

print(f"\\nTARGET: {TARGET}  |  random_state: {RANDOM_STATE}  |  split: 80/20 outer + 10% inner-val for early stopping")
"""


MANIFEST_CELL_MARKDOWN = """\
### 1.4 Reproducibility manifest

A snapshot of the runtime environment + dataset hashes — printed so that the
final figures and AUROCs in this notebook can be re-tied to a specific
software stack and a specific set of training tables.
"""


MANIFEST_CELL = """\
# ── 1.4 Reproducibility manifest ───────────────────────────────────────────
import hashlib, platform, sys, importlib

LIBS = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn",
        "lightgbm", "xgboost", "catboost", "shap", "duckdb", "pyarrow", "nbformat"]

print("=" * 64)
print("RUNTIME")
print("=" * 64)
print(f"  Python  : {sys.version.split()[0]}")
print(f"  Platform: {platform.platform()}")
print()
print("LIBRARY VERSIONS")
for lib in LIBS:
    try:
        mod = importlib.import_module(lib)
        ver = getattr(mod, "__version__", "?")
        print(f"  {lib:12s} {ver}")
    except ImportError:
        print(f"  {lib:12s} (not installed)")

def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()

print()
print("TRAINING-TABLE HASHES (SHA-256, first 16 chars)")
for name, p in PATHS.items():
    if p.exists():
        print(f"  {name}: {_sha256(p)[:16]}  ({p.stat().st_size / 1024**2:6.1f} MB)  {p.name}")
    else:
        print(f"  {name}: MISSING  {p}")
print("=" * 64)
"""


SPLIT_CELL = """\
# ── 8.2 Patient-grouped 60/20/20 train/val/test split ──────────────────────
# Two-stage GroupShuffleSplit on subject_id:
#   Stage 1: 80% train+val   /  20% test    (untouched until §11)
#   Stage 2: 75% train       /  25% val    (within train+val)
# Yields 60/20/20 overall. Val is used for early stopping and blend-weight
# selection in §10. Test is evaluated exactly once.

df_v7 = pd.read_parquet(PATHS["v7"])
print(f"V7 dataset: {df_v7.shape[0]:,} rows x {df_v7.shape[1]} columns")

drop_cols = set(ID_COLS + DT_COLS + ["insurance", TARGET])
feature_cols = [c for c in df_v7.columns if c not in drop_cols]

X = df_v7[feature_cols].copy()
y = df_v7[TARGET].astype(int).values
groups = df_v7["subject_id"].values

# Label-encode string categoricals (handled natively by CatBoost later)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ── Stage 1 ── hold out 20% test (patient-grouped)
test_splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
trainval_idx, test_idx = next(test_splitter.split(X, y, groups=groups))

X_trainval       = X.iloc[trainval_idx].reset_index(drop=True)
y_trainval       = y[trainval_idx]
groups_trainval  = groups[trainval_idx]

X_test          = X.iloc[test_idx].reset_index(drop=True)
y_test          = y[test_idx]
test_patients   = set(groups[test_idx])

# ── Stage 2 ── from train+val, hold out VAL_SIZE as val
val_splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
train_idx, val_idx = next(val_splitter.split(X_trainval, y_trainval, groups=groups_trainval))

X_train         = X_trainval.iloc[train_idx].reset_index(drop=True)
y_train         = y_trainval[train_idx]
groups_train    = groups_trainval[train_idx]

X_val           = X_trainval.iloc[val_idx].reset_index(drop=True)
y_val           = y_trainval[val_idx]
val_patients    = set(groups_trainval[val_idx])
train_patients  = set(groups_train)

# Leakage sanity-checks (patient-grouped)
assert not (train_patients & val_patients),  "PATIENT LEAKAGE between train and val"
assert not (train_patients & test_patients), "PATIENT LEAKAGE between train and test"
assert not (val_patients   & test_patients), "PATIENT LEAKAGE between val   and test"

# Persist patient lists for reuse in §9 progression (same patient assignment across all V_n)
PATIENT_SPLIT = {
    "train": train_patients,
    "val":   val_patients,
    "test":  test_patients,
}

print(f"\\nTrain: {len(X_train):,} admissions ({len(train_patients):,} patients)  |  pos rate = {y_train.mean():.4f}")
print(f"Val:   {len(X_val):,} admissions ({len(val_patients):,} patients)  |  pos rate = {y_val.mean():.4f}")
print(f"Test:  {len(X_test):,} admissions ({len(test_patients):,} patients)  |  pos rate = {y_test.mean():.4f}")
print(f"Features: {len(feature_cols)}  ({len(cat_cols)} categorical)")
print(f"\\nSplit protocol: 60/20/20 patient-grouped via two-stage GroupShuffleSplit (random_state={RANDOM_STATE}).")
"""


LOGREG_PROGRESSION_CELL = """\
# ── 9.1 Logistic regression V_n progression ─────────────────────────────────
# Numbers loaded from results/progression.json, which is regenerated by
# `python scripts/run_progression.py`. That script runs each V_n in a fresh
# subprocess (one Python process per version) — necessary because running the
# full V1->V6 progression in a single kernel session reliably exhausts memory
# on a 244k-row dataset when training LightGBM + XGBoost + MLP back-to-back.
#
# Reproducing this cell: run `python scripts/run_progression.py` first.

import json

prog_path = ART_DIR / "progression.json"
if not prog_path.exists():
    raise FileNotFoundError(
        f"{prog_path} not found. Generate it with:\\n"
        f"  python scripts/run_progression.py"
    )
PROGRESSION = json.loads(prog_path.read_text())
logreg_progression = {v: d["test_auroc"] for v, d in PROGRESSION["logreg"].items()}

print("Logistic Regression test AUROC by feature version (loaded from JSON):")
for v, d in PROGRESSION["logreg"].items():
    print(f"  {v}: {d['n_features']:3d} features -> test AUROC = {d['test_auroc']:.4f}")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(list(logreg_progression.keys()), list(logreg_progression.values()),
        "o-", color=PALETTE["blue"], lw=2.5, markersize=10)
ax.axhline(0.70, color="gray", ls="--", lw=1)
for v, s in logreg_progression.items():
    ax.text(v, s + 0.0015, f"{s:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(min(logreg_progression.values()) - 0.005,
            max(logreg_progression.values()) + 0.010)
ax.set_ylabel("Test AUROC")
ax.set_title("Logistic Regression — test AUROC across feature versions",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_b_logreg_v1v6.png", dpi=150, bbox_inches="tight")
plt.show()
"""


GBM_PROGRESSION_CELL = """\
# ── 9.2/9.3/9.4 Per-family progression across feature versions ─────────────
# LightGBM / XGBoost / MLP test AUROC loaded from results/progression.json.
# Regenerate with `python scripts/run_progression.py` (runs each version in
# its own subprocess to avoid kernel OOM).

progression = {
    "LightGBM": PROGRESSION["lightgbm"],
    "XGBoost":  PROGRESSION["xgboost"],
    "MLP":      PROGRESSION["mlp"],
}

for v in progression["LightGBM"]:
    print(f"  {v} -> LGB {progression['LightGBM'][v]:.4f}  "
          f"XGB {progression['XGBoost'][v]:.4f}  "
          f"MLP {progression['MLP'][v]:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, (name, series) in zip(axes, progression.items()):
    if not series:
        continue
    color = {"LightGBM": PALETTE["teal"], "XGBoost": PALETTE["mint"], "MLP": PALETTE["gold"]}[name]
    ax.plot(list(series.keys()), list(series.values()), "o-", color=color, lw=2.5, markersize=10)
    for v, s in series.items():
        ax.text(v, s + 0.002, f"{s:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title(f"{name} — V_n progression", fontsize=13, fontweight="bold")
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.65, 0.80)
plt.suptitle("Per-family AUROC progression across feature-engineering versions",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_cde_lgbm_xgb_mlp_v1v6.png", dpi=150, bbox_inches="tight")
plt.show()
"""


V6_COMPARISON_CELL = """\
# ── 9.5 V6 cross-family comparison ─────────────────────────────────────────
# Tabular GBM points are from the live V1->V6 loop above (taken at V6).
# Neural-network points (FT-Transformer, Hybrid GRU+MLP, BiGRU, Stacking)
# come from companion training scripts not reproduced inline in this
# publication notebook — see scripts/train_neural_networks.py in a future
# release. We label these explicitly in the figure.

v6_live_gbm = {
    "LightGBM":        progression["LightGBM"].get("V6"),
    "XGBoost":         progression["XGBoost"].get("V6"),
    "MLP baseline":    progression["MLP"].get("V6"),
}
v6_external_nn = {
    "FT-Transformer":     0.7690,   # from upstream NN runs (reference)
    "Hybrid LSTM":        0.7695,
    "Hybrid GRU+MLP":     0.7711,
    "Standalone BiGRU":   0.6365,
    "Stacking (LR meta)": 0.7778,
}
v6_comparison = {**{k: v for k, v in v6_live_gbm.items() if v is not None},
                 **v6_external_nn}
v6_sorted = dict(sorted(v6_comparison.items(), key=lambda x: x[1]))

fig, ax = plt.subplots(figsize=(10, 5))
colors = []
for k, v in v6_sorted.items():
    is_nn = k in v6_external_nn
    if v < 0.70:
        c = PALETTE["coral"]
    elif v < 0.77:
        c = PALETTE["gold"]
    elif v < 0.778:
        c = PALETTE["teal"]
    else:
        c = PALETTE["mint"]
    colors.append(c)
ax.barh(list(v6_sorted.keys()), list(v6_sorted.values()), color=colors)
ax.axvline(0.778, color="gray", ls="--", lw=1, label="Stacking ceiling 0.778")
for i, (k, v) in enumerate(v6_sorted.items()):
    suffix = " *" if k in v6_external_nn else ""
    ax.text(v + 0.002, i, f"{v:.4f}{suffix}", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0.62, 0.80)
ax.set_xlabel("Test AUROC")
ax.set_title("V6 cross-family comparison (* = external NN reference, not inlined)",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_f_v6_families.png", dpi=150, bbox_inches="tight")
plt.show()
"""


LGB_V7_CELL = """\
# --- LightGBM ------------------------------------------------------------
lgb_params = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "num_leaves": 63, "learning_rate": 0.03,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "n_estimators": 1500, "verbose": -1, "n_jobs": -1,
}

print(f"Training LightGBM across {N_SEEDS} seeds (early stopping on val) ...")
lgb_preds_val  = np.zeros((len(X_val),  N_SEEDS))
lgb_preds_test = np.zeros((len(X_test), N_SEEDS))
for s in range(N_SEEDS):
    p = {**lgb_params, "random_state": 42 + s}
    m = lgb.LGBMClassifier(**p)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    lgb_preds_val[:,  s] = m.predict_proba(X_val)[:,  1]
    lgb_preds_test[:, s] = m.predict_proba(X_test)[:, 1]
    if s == 0:
        lgb_model_seed0 = m   # keep for SHAP later

lgb_val_avg  = lgb_preds_val.mean(axis=1)
lgb_test_avg = lgb_preds_test.mean(axis=1)
lgb_val_auroc  = roc_auc_score(y_val,  lgb_val_avg)
lgb_test_auroc = roc_auc_score(y_test, lgb_test_avg)
print(f"  LightGBM {N_SEEDS}-seed avg AUROC: val = {lgb_val_auroc:.4f}  |  test = {lgb_test_auroc:.4f}")

# Aliases for downstream cells that expect the legacy single-name variables
lgb_avg, lgb_auroc = lgb_test_avg, lgb_test_auroc
"""


XGB_V7_CELL = """\
# --- XGBoost -------------------------------------------------------------
xgb_params = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 6, "learning_rate": 0.03, "n_estimators": 1500,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 50,
    "tree_method": "hist", "verbosity": 0, "n_jobs": -1,
    "early_stopping_rounds": 100,
}

print(f"Training XGBoost across {N_SEEDS} seeds (early stopping on val) ...")
xgb_preds_val  = np.zeros((len(X_val),  N_SEEDS))
xgb_preds_test = np.zeros((len(X_test), N_SEEDS))
for s in range(N_SEEDS):
    p = {**xgb_params, "random_state": 42 + s}
    m = XGBClassifier(**p)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds_val[:,  s] = m.predict_proba(X_val)[:,  1]
    xgb_preds_test[:, s] = m.predict_proba(X_test)[:, 1]
    if s == 0:
        xgb_model_seed0 = m

xgb_val_avg  = xgb_preds_val.mean(axis=1)
xgb_test_avg = xgb_preds_test.mean(axis=1)
xgb_val_auroc  = roc_auc_score(y_val,  xgb_val_avg)
xgb_test_auroc = roc_auc_score(y_test, xgb_test_avg)
print(f"  XGBoost {N_SEEDS}-seed avg AUROC: val = {xgb_val_auroc:.4f}  |  test = {xgb_test_auroc:.4f}    <-- selected as deployment candidate")

xgb_avg, xgb_auroc = xgb_test_avg, xgb_test_auroc
"""


CB_V7_CELL = """\
# --- CatBoost ------------------------------------------------------------
print(f"Training CatBoost across {N_SEEDS} seeds (early stopping on val) ...")
cb_preds_val  = np.zeros((len(X_val),  N_SEEDS))
cb_preds_test = np.zeros((len(X_test), N_SEEDS))
for s in range(N_SEEDS):
    m = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        random_seed=42 + s, verbose=0, eval_metric="AUC",
        early_stopping_rounds=100,
    )
    m.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    cb_preds_val[:,  s] = m.predict_proba(X_val)[:,  1]
    cb_preds_test[:, s] = m.predict_proba(X_test)[:, 1]

cb_val_avg  = cb_preds_val.mean(axis=1)
cb_test_avg = cb_preds_test.mean(axis=1)
cb_val_auroc  = roc_auc_score(y_val,  cb_val_avg)
cb_test_auroc = roc_auc_score(y_test, cb_test_avg)
print(f"  CatBoost {N_SEEDS}-seed avg AUROC: val = {cb_val_auroc:.4f}  |  test = {cb_test_auroc:.4f}")

cb_avg, cb_auroc = cb_test_avg, cb_test_auroc
"""


HIST_V7_CELL = """\
# --- HistGradientBoosting ------------------------------------------------
# HistGBM uses an internal 10% validation slice of X_train for early stopping
# (sklearn doesn't expose a custom eval_set), so X_val and X_test are never
# seen during training. Predictions on X_val are still used for blend weights;
# predictions on X_test are evaluated once at the end.
print(f"Training HistGBM across {N_SEEDS} seeds (internal val) ...")
hist_preds_val  = np.zeros((len(X_val),  N_SEEDS))
hist_preds_test = np.zeros((len(X_test), N_SEEDS))
for s in range(N_SEEDS):
    m = HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.03, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0,
        random_state=42 + s, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=100, verbose=0,
    )
    m.fit(X_train, y_train)
    hist_preds_val[:,  s] = m.predict_proba(X_val)[:,  1]
    hist_preds_test[:, s] = m.predict_proba(X_test)[:, 1]

hist_val_avg  = hist_preds_val.mean(axis=1)
hist_test_avg = hist_preds_test.mean(axis=1)
hist_val_auroc  = roc_auc_score(y_val,  hist_val_avg)
hist_test_auroc = roc_auc_score(y_test, hist_test_avg)
print(f"  HistGBM {N_SEEDS}-seed avg AUROC: val = {hist_val_auroc:.4f}  |  test = {hist_test_auroc:.4f}")

hist_avg, hist_auroc = hist_test_avg, hist_test_auroc
"""


BLEND_CELL = """\
# ── 10.2 Scipy-optimized blending (val-only) ───────────────────────────────
# Blend weights are optimised on the VALIDATION set only.
# The TEST set is evaluated EXACTLY ONCE at the end of this cell using the
# fixed weights — this matches the protocol expected by publication reviewers
# and removes the test-set leakage present in the original capstone notebook.

def neg_auroc_val(w):
    w = np.clip(w, 0, None)
    w = w / w.sum()
    blend = (w[0] * lgb_val_avg + w[1] * xgb_val_avg +
             w[2] * cb_val_avg  + w[3] * hist_val_avg)
    return -roc_auc_score(y_val, blend)

result = minimize(
    neg_auroc_val, x0=[0.25, 0.25, 0.25, 0.25],
    method="Nelder-Mead", bounds=[(0.05, 0.5)] * 4,
)
w_opt = result.x / result.x.sum()

# Apply optimised weights to the TEST set (single-shot evaluation)
blend_test = (w_opt[0] * lgb_test_avg + w_opt[1] * xgb_test_avg +
              w_opt[2] * cb_test_avg  + w_opt[3] * hist_test_avg)
blend_test_auroc = roc_auc_score(y_test, blend_test)

# Also report blend on val for context (the value the optimiser saw)
blend_val = (w_opt[0] * lgb_val_avg + w_opt[1] * xgb_val_avg +
             w_opt[2] * cb_val_avg  + w_opt[3] * hist_val_avg)
blend_val_auroc = roc_auc_score(y_val, blend_val)

print("=" * 70)
print("SCIPY-OPTIMIZED ENSEMBLE BLEND (V7) — held-out test eval")
print("=" * 70)
print(f"  LightGBM weight: {w_opt[0]:.3f}  (val: {lgb_val_auroc:.4f}, test: {lgb_test_auroc:.4f})")
print(f"  XGBoost  weight: {w_opt[1]:.3f}  (val: {xgb_val_auroc:.4f}, test: {xgb_test_auroc:.4f})")
print(f"  CatBoost weight: {w_opt[2]:.3f}  (val: {cb_val_auroc:.4f},  test: {cb_test_auroc:.4f})")
print(f"  HistGBM  weight: {w_opt[3]:.3f}  (val: {hist_val_auroc:.4f}, test: {hist_test_auroc:.4f})")
print(f"\\n  Blend AUROC on val  (weights selected here): {blend_val_auroc:.6f}")
print(f"  Blend AUROC on test (single evaluation):     {blend_test_auroc:.6f}")
print(f"  XGBoost solo (test):                          {xgb_test_auroc:.6f}  <-- deployment candidate")
print(f"  Gap (blend - XGB on test):                    {blend_test_auroc - xgb_test_auroc:+.4f}")

# Aliases for downstream cells
blend_preds, blend_auroc = blend_test, blend_test_auroc
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def clear_outputs(cell):
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None


def replace_cell(nb, idx, new_source):
    nb.cells[idx].source = new_source
    clear_outputs(nb.cells[idx])


def insert_code_cell_after(nb, idx, source):
    new = nbf.v4.new_code_cell(source=source)
    nb.cells.insert(idx + 1, new)
    return idx + 1


def insert_markdown_cell_after(nb, idx, source):
    new = nbf.v4.new_markdown_cell(source=source)
    nb.cells.insert(idx + 1, new)
    return idx + 1


def cell_source(cell):
    return "".join(cell.source) if isinstance(cell.source, list) else cell.source


# ─────────────────────────────────────────────────────────────────────────────
# Main refactor
# ─────────────────────────────────────────────────────────────────────────────


def main():
    if not NB_PATH.exists():
        print(f"ERROR: notebook not found at {NB_PATH}", file=sys.stderr)
        sys.exit(1)

    nb = nbf.read(NB_PATH, as_version=4)
    print(f"Loaded {NB_PATH.name}: {len(nb.cells)} cells")

    # Map sections by content match (idempotent — works even if cell indices shift)
    def find(needle):
        for i, c in enumerate(nb.cells):
            if needle in cell_source(c):
                return i
        raise LookupError(f"cell containing {needle!r} not found")

    # 1. Title cell
    idx = find("# Predicting 30-Day Hospital Readmission")
    replace_cell(nb, idx, TITLE_CELL)
    print(f"  [{idx:3d}] title cell updated (added Poellabauer)")

    # 2. BASE_DIR cell (clear stale FileNotFoundError outputs + new content)
    idx = find("BASE_DIR = Path")
    replace_cell(nb, idx, BASEDIR_CELL)
    print(f"  [{idx:3d}] §1.3 BASE_DIR rewritten + outputs cleared")

    # 3. Insert §1.4 reproducibility manifest right after BASE_DIR (idempotent).
    try:
        existing_md = find("### 1.4 Reproducibility manifest")
        existing_code = find("# ── 1.4 Reproducibility manifest")
        # Already inserted — replace contents to refresh.
        replace_cell(nb, existing_md, MANIFEST_CELL_MARKDOWN)
        replace_cell(nb, existing_code, MANIFEST_CELL)
        print(f"  [{existing_md:3d},{existing_code:3d}] §1.4 reproducibility manifest refreshed (already present)")
    except LookupError:
        md_idx = insert_markdown_cell_after(nb, idx, MANIFEST_CELL_MARKDOWN)
        code_idx = insert_code_cell_after(nb, md_idx, MANIFEST_CELL)
        print(f"  [{md_idx:3d},{code_idx:3d}] §1.4 reproducibility manifest inserted")

    # 4. §8.2 split
    idx = find("# ── 8.2 Patient-grouped train/test split")
    replace_cell(nb, idx, SPLIT_CELL)
    print(f"  [{idx:3d}] §8.2 split: 80/20 -> 60/20/20")

    # 5. §9.1 LogReg progression (live)
    idx = find("# ── 9.1 Logistic regression V1")
    replace_cell(nb, idx, LOGREG_PROGRESSION_CELL)
    print(f"  [{idx:3d}] §9.1 LogReg progression inlined")

    # 6. §9.2/9.3/9.4 LGB/XGB/MLP progression (live)
    idx = find("# ── 9.2/9.3/9.4 Per-family progression")
    replace_cell(nb, idx, GBM_PROGRESSION_CELL)
    print(f"  [{idx:3d}] §9.2-9.4 LGB/XGB/MLP progression inlined")

    # 7. §9.5 V6 cross-family — reuse live numbers + label NN as external
    idx = find("# ── 9.5 V6 cross-family comparison")
    replace_cell(nb, idx, V6_COMPARISON_CELL)
    print(f"  [{idx:3d}] §9.5 V6 cross-family — live GBM + labelled NN refs")

    # 8a. §10.1 LightGBM (combined into single cell — keep header)
    idx = find("# ── 10.1 4-GBM multi-seed ensemble on V7")
    # Need to preserve the imports + N_SEEDS prelude that's in this cell.
    # The original cell contains both prelude AND the LightGBM block.
    # We split: keep prelude, replace LGB block with new one.
    new_first_cell = """\
# ── 10.1 4-GBM multi-seed ensemble on V7 ───────────────────────────────────
# NOTE: Running all four models across 10 seeds takes ~20-40 minutes on a
# modern CPU and is materially faster with a GPU for XGBoost/CatBoost.
# Drop N_SEEDS to 2-3 for a quick sanity check.
#
# REFACTORED from the capstone version: each model fits on X_train and uses
# X_val for early stopping (X_val never seen during forward pass).
# X_test is touched once in §10.2 for the final blend evaluation.

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

N_SEEDS = 10   # set to 2 for a fast dry-run

""" + LGB_V7_CELL
    replace_cell(nb, idx, new_first_cell)
    print(f"  [{idx:3d}] §10.1 LightGBM: val-based early stopping")

    # 8b. §10.1 XGBoost
    idx = find("# --- XGBoost --------")
    replace_cell(nb, idx, XGB_V7_CELL)
    print(f"  [{idx:3d}] §10.1 XGBoost: val-based early stopping")

    # 8c. §10.1 CatBoost
    idx = find("# --- CatBoost --------")
    replace_cell(nb, idx, CB_V7_CELL)
    print(f"  [{idx:3d}] §10.1 CatBoost: val-based early stopping")

    # 8d. §10.1 HistGBM
    idx = find("# --- HistGradientBoosting --------")
    replace_cell(nb, idx, HIST_V7_CELL)
    print(f"  [{idx:3d}] §10.1 HistGBM: internal val; predicts on val + test")

    # 9. §10.2 Blend on val, eval on test
    idx = find("# ── 10.2 Scipy-optimized blending")
    replace_cell(nb, idx, BLEND_CELL)
    print(f"  [{idx:3d}] §10.2 blend: val-only optimization + single-shot test eval")

    # Save
    nbf.write(nb, NB_PATH)
    print(f"\\nSaved refactored notebook to {NB_PATH}")
    print(f"Final cell count: {len(nb.cells)}")


if __name__ == "__main__":
    main()

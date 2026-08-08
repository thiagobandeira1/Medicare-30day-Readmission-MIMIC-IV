"""Bulletproof feature-engineering analysis (answers Dr. Poellabauer directly).

Splits the 207-feature candidate pool into:
  * BASE  = minimally-processed features (raw values + mandatory aggregations only)
  * ENG   = hand-engineered features (target-encodings, interactions, log/poly/PCA transforms)

Then quantifies what the hand-engineering actually buys:
  1. BASE-only full LightGBM                          -> AUROC_base
  2. RFE over BASE-only  (checkpointed, CPU-capped)   -> parsimony-optimal raw subset
  3. BASE-RFE subset + all ENG features, refit        -> engineering lift on top of raw
  4. compare to full-pool (0.7950) and full-RFE (0.7960) from the existing run

Run with the capstone env:  envs/capstone/python.exe src/_run_raweng_rfe.py
"""
import os
# Cap threads BEFORE importing numpy/lightgbm so we never peg all cores (thermal-crash safe)
N_JOBS = 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(N_JOBS)

import json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

REPO = Path(__file__).resolve().parent.parent
PUB = REPO.parent
RESULTS = REPO / "results"
FIGS = REPO / "figures"
FIGS.mkdir(exist_ok=True)

# ---- Build the EXACT 207-feature candidate pool (same META exclusions as the canonical RFE) ----
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
assert V10["hadm_id"].equals(V7["hadm_id"]), "tables must be row-aligned"

META = {"subject_id", "hadm_id", "admittime_dt", "dischtime_dt",
        "insurance", "readmit_30d", "discharge_location",
        "admittime", "dischtime", "deathtime"}
feat_v10 = [c for c in V10.columns if c not in META]
v7_feats = json.loads((RESULTS / "v7_feature_cols.json").read_text())
v7_unique = [f for f in v7_feats if f not in feat_v10 and f in V7.columns]
feat_full = feat_v10 + v7_unique
print(f"Full candidate pool: {len(feat_full)} features  (v10 {len(feat_v10)} + v7-unique {len(v7_unique)})", flush=True)

DF = pd.concat([V10[feat_v10].reset_index(drop=True),
                V7[v7_unique].reset_index(drop=True)], axis=1)
y = V10["readmit_30d"].to_numpy().astype(int)

# ---- Split BASE (minimally-processed) vs ENG (hand-engineered) ----
ENG_RE = re.compile(r"_te$|_x_|_sq$|^log_|_pc\d")
eng_feats = [f for f in feat_full if ENG_RE.search(f)]
base_feats = [f for f in feat_full if not ENG_RE.search(f)]
print(f"BASE (minimally-processed): {len(base_feats)}   ENG (hand-engineered): {len(eng_feats)}", flush=True)

# ---- Encode + matrices ----
Xdf = DF[feat_full].copy()
for c in feat_full:
    if Xdf[c].dtype == "object":
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str).fillna("__NA__"))
Xall = Xdf.to_numpy(dtype=float)

split = np.load(RESULTS / "v7_split_indices.npz")
tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]
pos = {f: i for i, f in enumerate(feat_full)}


def fit_lgbm(cols, n_estimators=400, seed=42):
    idx = [pos[f] for f in cols]
    m = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                           subsample=0.9, colsample_bytree=0.9, random_state=seed,
                           n_jobs=N_JOBS, verbose=-1)
    m.fit(Xall[tr][:, idx], y[tr])
    return m, idx


def auroc(cols):
    m, idx = fit_lgbm(cols)
    return roc_auc_score(y[te], m.predict_proba(Xall[te][:, idx])[:, 1])


t0 = time.time()
auroc_base = auroc(base_feats)
print(f"[1] BASE-only ({len(base_feats)} feat) test AUROC: {auroc_base:.4f}   ({time.time()-t0:.0f}s)", flush=True)
auroc_full = auroc(feat_full)
print(f"[2] FULL pool ({len(feat_full)} feat) test AUROC: {auroc_full:.4f}   (sanity vs 0.7950)", flush=True)

# ---- RFE over BASE-only (checkpointed, separate checkpoint file) ----
STEP, MIN_FEATURES = 10, 10
CKPT = RESULTS / "rfe_base_checkpoint.json"
if CKPT.exists():
    st = json.loads(CKPT.read_text()); remaining = st["remaining"]; history = st["history"]
    print(f"Resuming BASE-RFE: {len(history)} rounds done, {len(remaining)} remaining.", flush=True)
else:
    remaining = list(base_feats); history = []
    print(f"Starting BASE-RFE from {len(remaining)} features.", flush=True)

t0 = time.time()
while len(remaining) >= MIN_FEATURES:
    idx = [pos[f] for f in remaining]
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                           subsample=0.9, colsample_bytree=0.9, random_state=42,
                           n_jobs=N_JOBS, verbose=-1)
    m.fit(Xall[tr][:, idx], y[tr])
    va_auc = roc_auc_score(y[va], m.predict_proba(Xall[va][:, idx])[:, 1])
    history.append({"n": len(remaining), "val_auroc": float(va_auc), "features": list(remaining)})
    CKPT.write_text(json.dumps({"remaining": remaining, "history": history}))
    print(f"  {len(remaining):3d} base feat -> inner-val AUROC {va_auc:.4f}   ({time.time()-t0:.0f}s)", flush=True)
    if len(remaining) <= MIN_FEATURES:
        break
    imp = m.feature_importances_
    drop_pos = set(np.argsort(imp)[:STEP])
    remaining = [f for i, f in enumerate(remaining) if i not in drop_pos]

best = max(history, key=lambda h: h["val_auroc"])
base_rfe_feats = best["features"]
auroc_base_rfe = auroc(base_rfe_feats)
print(f"[3] BASE-RFE optimal: {len(base_rfe_feats)} feat | inner-val {best['val_auroc']:.4f} | test AUROC {auroc_base_rfe:.4f}", flush=True)

# ---- Engineering lift: BASE-RFE subset + all ENG features ----
combo = base_rfe_feats + eng_feats
auroc_combo = auroc(combo)
print(f"[4] BASE-RFE + {len(eng_feats)} ENG ({len(combo)} feat) test AUROC: {auroc_combo:.4f}", flush=True)

# existing full-pool RFE result
prev = json.loads((RESULTS / "rfe_selection_results.json").read_text())
auroc_full_rfe = prev["rfe_auroc"]; n_full_rfe = prev["rfe_n"]
rfe_sel = set(prev["rfe_selected"])
eng_in_rfe = sorted(f for f in rfe_sel if ENG_RE.search(f))

out = {
    "n_base": len(base_feats), "n_eng": len(eng_feats), "n_full": len(feat_full),
    "auroc_base_full": float(auroc_base),
    "auroc_full_pool": float(auroc_full),
    "auroc_base_rfe": float(auroc_base_rfe), "n_base_rfe": len(base_rfe_feats),
    "auroc_base_rfe_plus_eng": float(auroc_combo),
    "auroc_full_rfe": float(auroc_full_rfe), "n_full_rfe": n_full_rfe,
    "delta_eng_full_minus_base": float(auroc_full - auroc_base),
    "delta_eng_on_rfe": float(auroc_combo - auroc_base_rfe),
    "eng_features": eng_feats,
    "eng_features_kept_by_full_rfe": eng_in_rfe,
    "n_eng_kept_by_full_rfe": len(eng_in_rfe),
    "base_rfe_features": base_rfe_feats,
}
(RESULTS / "rfe_raw_vs_engineered.json").write_text(json.dumps(out, indent=2))
print("\nSaved results/rfe_raw_vs_engineered.json", flush=True)

# ---- Figure: AUROC comparison (labeled axes for Dr. Mondal) ----
labels = [f"BASE only\n({len(base_feats)} raw)",
          f"BASE-RFE\n({len(base_rfe_feats)})",
          f"Full pool\n({len(feat_full)})",
          f"Full-RFE\n({n_full_rfe})"]
vals = [auroc_base, auroc_base_rfe, auroc_full, auroc_full_rfe]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "axes.labelsize": 13, "font.weight": "bold"})
fig, ax = plt.subplots(figsize=(9, 5.2))
colors = ["#8D99AE", "#5C7A89", "#1F7A8C", "#E76F51"]
bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=1.1)
ax.set_ylim(0.78, 0.80)
ax.set_ylabel("Test AUROC")
ax.set_xlabel("Feature set")
ax.set_title("What hand-engineering adds beyond minimally-processed features")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.0004, f"{v:.4f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
ax.axhline(auroc_base, color="#8D99AE", ls="--", lw=1.3, alpha=0.7)
fig.tight_layout()
fig.savefig(FIGS / "raw_vs_engineered_auroc.png", dpi=300, bbox_inches="tight")
print("Saved figures/raw_vs_engineered_auroc.png", flush=True)
print(f"\n=== ENGINEERING LIFT ===")
print(f"  Full pool - Base only : {auroc_full - auroc_base:+.4f}")
print(f"  On RFE subset         : {auroc_combo - auroc_base_rfe:+.4f}")
print(f"  Engineered kept by full-RFE: {len(eng_in_rfe)}/{len(eng_feats)} -> {eng_in_rfe}")
print(f"TOTAL TIME: {time.time()-t0:.0f}s", flush=True)

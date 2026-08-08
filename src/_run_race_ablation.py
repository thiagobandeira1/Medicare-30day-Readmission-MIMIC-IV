"""Race ablation: what does the target-encoded race feature actually buy?

The deployed V8 set contains race_te, a target encoding whose predictive value
comes from an observed difference in outcome rates rather than a clinical
mechanism. Medicine has been removing exactly this kind of term from clinical
equations (eGFR 2021, pulmonary-function reference values, the VBAC calculator),
so the paper needs the number rather than an argument.

Retrains V8 with and without race_te on the identical patient-grouped split and
reports the difference in test AUROC with a paired bootstrap, plus the effect on
the Black-vs-White subgroup gap reported in the fairness section.

Outputs: results/race_ablation.json
"""
import os
N_JOBS = 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(N_JOBS)

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
PUB = REPO.parent
RESULTS = REPO / "results"
DATA = PUB / "Dataset" / "mimic-parquet"
API_ART = PUB / "readmission-api" / "artifacts"
SEED = 42
t0 = time.time()

# ---------------------------------------------------------------- data
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)

serving = joblib.load(API_ART / "serving_artifacts.joblib")
FEATURES = list(serving["feature_order"])
assert "race_te" in FEATURES, "race_te is not in the deployed feature set"
NO_RACE = [f for f in FEATURES if f != "race_te"]

X = DF[FEATURES].copy()
for c in serving["categorical_features"]:
    X[c] = X[c].astype(str).fillna("__NA__").map(serving["cat_maps"][c]).astype(float)
X = X.apply(pd.to_numeric, errors="coerce")
y = DF["readmit_30d"].to_numpy().astype(int)

split = np.load(RESULTS / "v7_split_indices.npz")
tr, te = split["train_idx"], split["test_idx"]
print(f"[{time.time()-t0:.0f}s] train {len(tr):,} / test {len(te):,}", flush=True)

PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=N_JOBS,
              eval_metric="auc", tree_method="hist")


def fit_predict(cols):
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(X.loc[tr, cols].to_numpy(np.float32), y[tr])
    return m.predict_proba(X.loc[te, cols].to_numpy(np.float32))[:, 1]


p_with = fit_predict(FEATURES)
a_with = roc_auc_score(y[te], p_with)
print(f"[{time.time()-t0:.0f}s] WITH race_te   ({len(FEATURES)} feat): {a_with:.4f}", flush=True)

p_without = fit_predict(NO_RACE)
a_without = roc_auc_score(y[te], p_without)
print(f"[{time.time()-t0:.0f}s] WITHOUT race_te ({len(NO_RACE)} feat): {a_without:.4f}", flush=True)

# ------------------------------------------------- paired bootstrap on the gap
rng = np.random.RandomState(SEED)
yt = y[te]
diffs = []
for _ in range(2000):
    ix = rng.randint(0, len(yt), len(yt))
    if len(np.unique(yt[ix])) < 2:
        continue
    diffs.append(roc_auc_score(yt[ix], p_with[ix]) - roc_auc_score(yt[ix], p_without[ix]))
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
p_two_sided = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
print(f"[{time.time()-t0:.0f}s] paired difference {a_with - a_without:+.4f} "
      f"(95% CI {lo:+.4f} to {hi:+.4f}, p = {p_two_sided:.3f})", flush=True)

# ------------------------------------------------------- subgroup effect
race_raw = pd.read_parquet(DATA / "admissions.parquet", columns=["hadm_id", "race"])
sub = DF.iloc[te][["hadm_id"]].merge(race_raw, on="hadm_id", how="left")


def bucket(r):
    r = str(r).upper()
    if "WHITE" in r: return "White"
    if "BLACK" in r: return "Black"
    if "HISPANIC" in r or "LATINO" in r: return "Hispanic/Latino"
    if "ASIAN" in r: return "Asian"
    return "Other/Unknown"


grp = sub["race"].apply(bucket).to_numpy()
subgroups = {}
for g in ["White", "Black", "Hispanic/Latino", "Asian", "Other/Unknown"]:
    m = grp == g
    if m.sum() < 50 or len(np.unique(yt[m])) < 2:
        continue
    subgroups[g] = {
        "n": int(m.sum()),
        "auroc_with_race": round(float(roc_auc_score(yt[m], p_with[m])), 4),
        "auroc_without_race": round(float(roc_auc_score(yt[m], p_without[m])), 4),
    }
print("\nsubgroup AUROC (with -> without race_te)")
for g, v in subgroups.items():
    print(f"  {g:16s} n={v['n']:>6,}  {v['auroc_with_race']:.4f} -> {v['auroc_without_race']:.4f}")

gap_with = subgroups["White"]["auroc_with_race"] - subgroups["Black"]["auroc_with_race"]
gap_without = subgroups["White"]["auroc_without_race"] - subgroups["Black"]["auroc_without_race"]
print(f"\nWhite-minus-Black gap: {gap_with:+.4f} -> {gap_without:+.4f}")

out = {
    "n_features_with": len(FEATURES), "n_features_without": len(NO_RACE),
    "auroc_with_race": round(float(a_with), 6),
    "auroc_without_race": round(float(a_without), 6),
    "difference": round(float(a_with - a_without), 6),
    "difference_ci95": [round(float(lo), 6), round(float(hi), 6)],
    "p_value": round(float(p_two_sided), 4),
    "cv_fold_sd_reference": 0.0027,
    "subgroups": subgroups,
    "white_minus_black_gap_with": round(float(gap_with), 4),
    "white_minus_black_gap_without": round(float(gap_without), 4),
}
(RESULTS / "race_ablation.json").write_text(json.dumps(out, indent=2))
print(f"\nSaved results/race_ablation.json   TOTAL {time.time()-t0:.0f}s")

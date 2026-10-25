"""v7 final consensus model: artifacts + every downstream analysis.

Identical to v5 phase 5-7 except:
  * consensus set comes from the v7 rerun on the leak-free pool (results_v12)
  * daily cumulative/dynamic AUROC uses the estimand the manuscript describes
    (deaths before day t remain among the controls; Blanche definition 2).
    With follow-up complete to day 30 for everyone not readmitted or dead,
    no censoring occurs before day 30, so the estimate is the empirical AUROC
    of {readmitted by day t} vs {all others}. The previously reported variant
    (deaths censored at death, IPCW) is kept for reference.
  * calibration-in-the-large (offset intercept with slope fixed at 1, per
    Van Calster et al.) is reported alongside the joint recalibration
    intercept.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "8"
import json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve)
from sklearn.linear_model import LogisticRegression

t0 = time.time()
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v12"
R2 = REPO / "results_reanalysis"
ART = OUT / "artifacts_v7"
ART.mkdir(exist_ok=True)
SEED = 42
rng = np.random.RandomState(SEED)
PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=8,
              eval_metric="auc", tree_method="hist", verbosity=0)


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


CMP = json.loads((OUT / "model_comparison.json").read_text())
FSEL = CMP["_stability"]["consensus_majority_features"]
NF = len(FSEL)
for bad in ("discharge_location_te", "los_trend_180d", "los_trend_x_prior_6m",
            "los_trend_x_prior_admits"):
    assert bad not in FSEL, bad
log(f"consensus features ({NF}): {sorted(FSEL)}")

V7 = pd.read_parquet(PUB / "training_table_v12.parquet")
V10 = pd.read_parquet(PUB / "training_table_v12.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
C = pd.read_parquet(R2 / "cohort_v2.parquet")
assert (DF["hadm_id"].values == C["hadm_id"].values).all()
keep = (~C["excl_index_death"]).to_numpy()
part = C["partition"].to_numpy()
y = C["label_v2"].to_numpy()
subj = C["subject_id"].to_numpy()
tr, va, te = (keep & (part == p) for p in ("train", "val", "test"))
dev = keep & ((part == "train") | (part == "val"))

OBJ = [c for c in FSEL if not pd.api.types.is_numeric_dtype(DF[c])]
log(f"fold-local-encoded object columns in final set: {OBJ}")


def te_fit(idx, yy=None):
    yy = y if yy is None else yy
    maps, gmean = {}, yy[idx].mean()
    for c in OBJ:
        cats = DF[c].astype(str).fillna("__NA__")
        g = pd.DataFrame({"c": cats.iloc[idx], "y": yy[idx]}).groupby("c")["y"] \
            .agg(["mean", "count"])
        maps[c] = {"map": ((g["mean"] * g["count"] + gmean * 20)
                           / (g["count"] + 20)).to_dict(), "global": float(gmean)}
    return maps


def te_apply(maps, frame):
    X = frame.copy()
    for c in OBJ:
        X[c] = X[c].astype(str).fillna("__NA__") \
            .map(maps[c]["map"]).fillna(maps[c]["global"]).astype(float)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


# ======================= MODEL EVALUATION (validation partition + dev CV) =====
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
EV = {}
rng = np.random.default_rng(42)
maps = te_fit(np.where(tr)[0])
Xtr = te_apply(maps, DF[FSEL].loc[tr]); Xva = te_apply(maps, DF[FSEL].loc[va])
ytr, yva, sva = y[tr], y[va], subj[va]

def boot_auc(yy, pp, ss, B=500, other=None):
    us = np.unique(ss); idx = {u: np.where(ss == u)[0] for u in us}
    out = []
    for _ in range(B):
        pick = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        if other is None:
            out.append(roc_auc_score(yy[pick], pp[pick]))
        else:
            out.append(roc_auc_score(yy[pick], pp[pick]) - roc_auc_score(yy[pick], other[pick]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]

full = xgb.XGBClassifier(**PARAMS).fit(Xtr.to_numpy(np.float32), ytr)
p_full = full.predict_proba(Xva.to_numpy(np.float32))[:, 1]
EV["val_auroc_full"] = round(float(roc_auc_score(yva, p_full)), 4)
log(f"val AUROC full {EV['val_auroc_full']}")

# 1) single-feature and utilization-only baselines
UTIL = [c for c in ["prior_admissions_6m", "prior_admissions_all", "freq_x_recency",
                    "time_since_last_discharge", "log_prior_readmit_count",
                    "prior_admits_6m_sq", "n_prior_ed_visits_6m"] if c in FSEL]
EV["util_features"] = UTIL
single = {}
for c in UTIL + ["hemoglobin_last", "lab_abnormal_rate", "los_days"]:
    if c in Xva:
        v = Xva[c].fillna(Xva[c].median()).to_numpy()
        a = roc_auc_score(yva, v); single[c] = round(float(max(a, 1 - a)), 4)
EV["single_feature_val_auroc"] = single
pu = xgb.XGBClassifier(**PARAMS).fit(Xtr[UTIL].to_numpy(np.float32), ytr) \
    .predict_proba(Xva[UTIL].to_numpy(np.float32))[:, 1]
EV["util_xgb_val_auroc"] = round(float(roc_auc_score(yva, pu)), 4)
EV["full_minus_util_ci95"] = [round(EV["val_auroc_full"] - EV["util_xgb_val_auroc"], 4),
                              boot_auc(yva, p_full, sva, other=pu)]
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                   LogisticRegression(max_iter=2000, C=1.0))
pl = lr.fit(Xtr.to_numpy(), ytr).predict_proba(Xva.to_numpy())[:, 1]
EV["logit_47_val_auroc"] = round(float(roc_auc_score(yva, pl)), 4)
log(f"util-only {EV['util_xgb_val_auroc']}, logit47 {EV['logit_47_val_auroc']}")

# 2) seed stability
seeds = []
for sd in range(5):
    pr = dict(PARAMS); pr["random_state"] = 100 + sd
    seeds.append(roc_auc_score(yva, xgb.XGBClassifier(**pr).fit(
        Xtr.to_numpy(np.float32), ytr).predict_proba(Xva.to_numpy(np.float32))[:, 1]))
EV["seed_val_auroc"] = [round(float(s), 4) for s in seeds]

# 3) hyperparameter sensitivity (validation partition only)
hp = {}
for d_ in (3, 4, 5, 6, 8):
    for lr_, nt in ((0.05, 600), (0.03, 1000), (0.1, 300)):
        pr = dict(PARAMS); pr.update(max_depth=d_, learning_rate=lr_, n_estimators=nt)
        pp = xgb.XGBClassifier(**pr).fit(Xtr.to_numpy(np.float32), ytr) \
            .predict_proba(Xva.to_numpy(np.float32))[:, 1]
        hp[f"depth{d_}_lr{lr_}_trees{nt}"] = round(float(roc_auc_score(yva, pp)), 4)
EV["hyperparam_val_auroc"] = hp
log(f"best hp {max(hp.items(), key=lambda kv: kv[1])}")

# 4) learning curve (patient subsample of training)
lc = {}
str_ = subj[tr]; us = np.unique(str_)
for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
    keep_u = set(rng.choice(us, int(len(us) * frac), replace=False))
    m_ = np.array([s in keep_u for s in str_])
    pp = xgb.XGBClassifier(**PARAMS).fit(Xtr.to_numpy(np.float32)[m_], ytr[m_]) \
        .predict_proba(Xva.to_numpy(np.float32))[:, 1]
    lc[str(frac)] = round(float(roc_auc_score(yva, pp)), 4)
EV["learning_curve_val_auroc"] = lc

# 5) calibration by decile (validation)
q = pd.qcut(p_full, 10, labels=False, duplicates="drop")
EV["val_calibration_deciles"] = [{"decile": int(k) + 1,
    "mean_pred": round(float(p_full[q == k].mean()), 4),
    "observed": round(float(yva[q == k].mean()), 4), "n": int((q == k).sum())}
    for k in range(10)]

# 6) overfitting check: train vs validation AUROC
EV["train_auroc"] = round(float(roc_auc_score(ytr, full.predict_proba(
    Xtr.to_numpy(np.float32))[:, 1])), 4)

# 7) test partition, utilization-only comparator (single additional look)
Xte = te_apply(maps, DF[FSEL].loc[te]); yte, ste = y[te], subj[te]
p_te = full.predict_proba(Xte.to_numpy(np.float32))[:, 1]
pu_te = xgb.XGBClassifier(**PARAMS).fit(Xtr[UTIL].to_numpy(np.float32), ytr) \
    .predict_proba(Xte[UTIL].to_numpy(np.float32))[:, 1]
EV["test_auroc_full"] = round(float(roc_auc_score(yte, p_te)), 4)
EV["test_auroc_util_only"] = round(float(roc_auc_score(yte, pu_te)), 4)
EV["test_full_minus_util_ci95"] = boot_auc(yte, p_te, ste, other=pu_te)
(OUT / "evaluation_v12.json").write_text(json.dumps(EV, indent=1))
print(json.dumps(EV, indent=1))

# -*- coding: utf-8 -*-
"""Causal rebuild of the 180-day length-of-stay trend (prior stays only) and a
quick value check against the v7 47-feature model. Writes aggregate results
only (results_v13/trend_causal_check_v13.json); the feature table stays local."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v13"
R2 = REPO / "results_reanalysis"

a = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                    columns=["subject_id", "hadm_id", "admittime", "dischtime"])
a["los"] = (a.dischtime - a.admittime).dt.total_seconds() / 86400
a = a.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)

W = pd.Timedelta(days=180)
trend = np.zeros(len(a)); nwin = np.zeros(len(a), int); dmean = np.zeros(len(a))
for sid, g in a.groupby("subject_id", sort=False):
    idx = g.index.to_numpy(); at = g.admittime.to_numpy()
    dt = g.dischtime.to_numpy(); los = g.los.to_numpy()
    for k in range(len(idx)):
        prior = (at < at[k]) & (dt <= at[k]) & (at >= at[k] - W.to_timedelta64())
        if not prior.any():
            continue
        y = np.append(los[prior], los[k])     # prior stays in time order + index
        x = np.arange(len(y), dtype=float)
        trend[idx[k]] = np.polyfit(x, y, 1)[0]
        nwin[idx[k]] = prior.sum()
        dmean[idx[k]] = los[k] - los[prior].mean()
a["los_trend_180d_prior"] = trend
a["los_delta_prior_180d"] = dmean
a["n_prior_stays_180d"] = nwin

C = pd.read_parquet(R2 / "cohort_v2.parquet")
V10 = pd.read_parquet(PUB / "training_table_v13.parquet")
V7 = pd.read_parquet(PUB / "training_table_v13.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
assert (DF.hadm_id.values == C.hadm_id.values).all()
DF = DF.merge(a[["hadm_id", "los_trend_180d_prior", "los_delta_prior_180d",
                 "n_prior_stays_180d"]], on="hadm_id", how="left")
assert (DF.hadm_id.values == C.hadm_id.values).all()
DF["los_trend_prior_x_prior_6m"] = DF.los_trend_180d_prior * DF.prior_admissions_6m.fillna(0)
DF["los_trend_prior_x_prior_admits"] = DF.los_trend_180d_prior * DF.prior_admissions_all.fillna(0)
NEW = ["los_trend_180d_prior", "los_trend_prior_x_prior_6m",
       "los_trend_prior_x_prior_admits", "los_delta_prior_180d"]
DF[["hadm_id"] + NEW].to_parquet(PUB / "trend_causal_v13.parquet", index=False)

keep = (~C.excl_index_death).to_numpy(); part = C.partition.to_numpy()
y = C.label_v2.to_numpy(); subj = C.subject_id.to_numpy()
tr, va = keep & (part == "train"), keep & (part == "val")
dev = keep & ((part == "train") | (part == "val"))
R = {}

# ---- leakage test on first-ever cohort admissions (same as v7 test)
allc = a.groupby("subject_id").admittime.rank(method="first")
first_ever = DF.hadm_id.isin(a.loc[allc == 1, "hadm_id"]).to_numpy() & keep
later = DF.hadm_id.isin(a.loc[a.groupby("subject_id").admittime.rank(
    method="first", ascending=False) > 1, "hadm_id"]).to_numpy()
for f in NEW:
    v = DF.loc[first_ever, f].to_numpy()
    R[f"{f}_first_ever_nonzero_share"] = round(float((v != 0).mean()), 4)
# among admissions WITH prior stays: does the feature predict a later admission?
hp = keep & (DF.n_prior_stays_180d.to_numpy() > 0)
for f in NEW + ["los_trend_180d"]:
    v = DF.loc[hp, f].fillna(0).to_numpy(); aa = roc_auc_score(later[hp], v)
    R[f"{f}_auroc_later_admission_given_prior"] = round(float(max(aa, 1 - aa)), 4)
R["share_with_prior_stay_180d"] = round(float(hp[keep].mean() if False else hp.sum() / keep.sum()), 4)
print(json.dumps(R, indent=1))

# ---- value check: add to v7 consensus features
CMP = json.loads((OUT / "model_comparison.json").read_text())
FSEL = CMP["_stability"]["consensus_majority_features"]
PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, tree_method="hist", random_state=42,
              n_jobs=-1, eval_metric="logloss")
OBJ = [c for c in FSEL if not pd.api.types.is_numeric_dtype(DF[c])]


def enc(fit_idx, rows, feats):
    X = DF.loc[rows, feats].copy(); gm = y[fit_idx].mean()
    for c in OBJ:
        cats = DF[c].astype(str)
        g = pd.DataFrame({"c": cats.iloc[fit_idx], "y": y[fit_idx]}).groupby("c")["y"].agg(["mean", "count"])
        mp = ((g["mean"] * g["count"] + gm * 20) / (g["count"] + 20)).to_dict()
        X[c] = cats.loc[rows].map(mp).fillna(gm).astype(float)
    return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)


sets = {"v7_47": FSEL, "v7_47_plus_trend3": FSEL + NEW[:3],
        "v7_47_plus_all4": FSEL + NEW}
itr = np.where(tr)[0]; iva = np.where(va)[0]
for nm, fs in sets.items():
    p = xgb.XGBClassifier(**PARAMS).fit(enc(itr, itr, fs), y[itr]).predict_proba(enc(itr, iva, fs))[:, 1]
    R[f"val_auroc_{nm}"] = round(float(roc_auc_score(y[iva], p)), 4)
    devidx = np.where(dev)[0]; cv = []
    for ftr, fte in GroupKFold(5).split(devidx, groups=subj[devidx]):
        a_, b_ = devidx[ftr], devidx[fte]
        pp = xgb.XGBClassifier(**PARAMS).fit(enc(a_, a_, fs), y[a_]).predict_proba(enc(a_, b_, fs))[:, 1]
        cv.append(roc_auc_score(y[b_], pp))
    R[f"devcv_auroc_{nm}"] = [round(float(np.mean(cv)), 4), [round(float(c), 4) for c in cv]]
    print(nm, R[f"val_auroc_{nm}"], R[f"devcv_auroc_{nm}"])
(OUT / "trend_causal_check_v13.json").write_text(json.dumps(R, indent=1))

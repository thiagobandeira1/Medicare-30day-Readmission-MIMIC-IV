# -*- coding: utf-8 -*-
"""Calibration-in-the-large by subgroup on the test partition (v8 review):
mean predicted vs observed risk, observed/expected ratio, and CITL (logistic
intercept with the logit of the prediction as offset), with patient-cluster
bootstrap CIs. Also the White minus Black specificity difference.
Writes results_v13/subgroup_calibration_v13.json (aggregate only)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v13"
R2 = REPO / "results_reanalysis"
Z = np.load(OUT / "final_preds_v13.npz")
p, y, s = Z["p_te"], Z["y_te"], Z["s_te"]
FM = json.loads((OUT / "final_model_v13.json").read_text())
THR = FM["threshold_validation"]["threshold"] if isinstance(
    FM["threshold_validation"], dict) else FM["threshold_validation"]
C = pd.read_parquet(R2 / "cohort_v2.parquet")
te = (~C.excl_index_death) & (C.partition == "test")
hadm = C.loc[te, "hadm_id"].to_numpy()
assert len(hadm) == len(p)
ADM = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                      columns=["hadm_id", "race"])
rr = pd.DataFrame({"hadm_id": hadm}).merge(ADM, on="hadm_id", how="left")["race"].astype(str)
race = np.select([rr.str.contains("WHITE"), rr.str.contains("BLACK"),
                  rr.str.contains("HISPANIC|LATINO"), rr.str.contains("ASIAN")],
                 ["White", "Black", "Hispanic/Latino", "Asian"], "Other/Unknown")
V10 = pd.read_parquet(PUB / "training_table_v13.parquet", columns=["hadm_id", "age_at_admit"])
age = pd.DataFrame({"hadm_id": hadm}).merge(V10, on="hadm_id", how="left")["age_at_admit"].to_numpy()
band = np.select([age < 65, age < 75, age < 85], ["<65", "65-74", "75-84"], "85+")
lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def citl(yy, ll):
    f = lambda a: -np.sum(yy * (a + ll) - np.log1p(np.exp(a + ll)))
    return minimize_scalar(f, bounds=(-3, 3), method="bounded").x


def stats(ix):
    return {"mean_pred": p[ix].mean(), "observed": y[ix].mean(),
            "o_e": y[ix].mean() / p[ix].mean(), "citl": citl(y[ix], lp[ix]),
            "spec": ((p[ix] < THR) & (y[ix] == 0)).sum() / max((y[ix] == 0).sum(), 1)}


rng = np.random.default_rng(42)
res = {}
for axis, lab in (("race", race), ("age_band", band)):
    for g in np.unique(lab):
        gi = np.where(lab == g)[0]
        sg = s[gi]; us = np.unique(sg)
        rmap = pd.Series(np.arange(len(gi))).groupby(sg).apply(np.array).to_dict()
        pt = stats(gi); draws = []
        for _ in range(800):
            pick = rng.choice(us, len(us))
            ix = gi[np.concatenate([rmap[u] for u in pick])]
            draws.append(stats(ix))
        D = pd.DataFrame(draws)
        res[f"{axis}:{g}"] = {k: round(float(v), 4) for k, v in pt.items()} | {
            f"{k}_ci95": [round(float(D[k].quantile(.025)), 4),
                          round(float(D[k].quantile(.975)), 4)] for k in ("o_e", "citl", "spec")}
# White minus Black specificity difference (independent within-group resampling)
wb = []
W = np.where(race == "White")[0]; B_ = np.where(race == "Black")[0]
mw = pd.Series(np.arange(len(W))).groupby(s[W]).apply(np.array).to_dict()
mb = pd.Series(np.arange(len(B_))).groupby(s[B_]).apply(np.array).to_dict()
uw, ub = np.unique(s[W]), np.unique(s[B_])
for _ in range(2000):
    iw = W[np.concatenate([mw[u] for u in rng.choice(uw, len(uw))])]
    ib = B_[np.concatenate([mb[u] for u in rng.choice(ub, len(ub))])]
    wb.append(stats(iw)["spec"] - stats(ib)["spec"])
res["white_minus_black_specificity"] = {
    "point": round(float(stats(W)["spec"] - stats(B_)["spec"]), 4),
    "ci95": [round(float(np.percentile(wb, 2.5)), 4), round(float(np.percentile(wb, 97.5)), 4)],
    "n_boot": 2000, "descriptive": True}
res["threshold"] = THR
res["patients_in_both_white_and_black"] = int(len(set(s[race == "White"]) & set(s[race == "Black"])))
res["patients_white"] = int(len(set(s[race == "White"]))); res["patients_black"] = int(len(set(s[race == "Black"])))
(OUT / "subgroup_calibration_v13.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))

# ---------------------------------------------------------------- joint resampling
# One patient-level resample of the union of White and Black patients; a
# patient recorded in both groups receives the same multiplicity in both.
W = np.where(race == "White")[0]; B_ = np.where(race == "Black")[0]
mw = pd.Series(np.arange(len(W))).groupby(s[W]).apply(np.array).to_dict()
mb = pd.Series(np.arange(len(B_))).groupby(s[B_]).apply(np.array).to_dict()
U = np.array(sorted(set(mw) | set(mb)))
rng2 = np.random.default_rng(7)
from sklearn.metrics import roc_auc_score
dA, dS = [], []
for _ in range(5000):
    pick = rng2.choice(U, len(U))
    iw = W[np.concatenate([mw[u] for u in pick if u in mw])]
    ib = B_[np.concatenate([mb[u] for u in pick if u in mb])]
    if len(np.unique(y[iw])) < 2 or len(np.unique(y[ib])) < 2:
        continue
    dA.append(roc_auc_score(y[iw], p[iw]) - roc_auc_score(y[ib], p[ib]))
    dS.append(stats(iw)["spec"] - stats(ib)["spec"])


def summ(d, point):
    d = np.asarray(d)
    p2 = 2 * min((d <= 0).mean(), (d >= 0).mean()); p2 = min(1.0, max(p2, 2 / (len(d) + 1)))
    return {"point_observed": round(float(point), 4), "bootstrap_mean": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_two_sided_plus1": float(f"{p2:.2g}"), "n_boot": int(len(d)),
            "design": "joint patient-level cluster bootstrap over the union of White and Black patients; "
                      "shared patients receive the same multiplicity in both groups"}


res["white_minus_black_auroc_joint"] = summ(dA, roc_auc_score(y[W], p[W]) - roc_auc_score(y[B_], p[B_]))
res["white_minus_black_specificity_joint"] = summ(dS, stats(W)["spec"] - stats(B_)["spec"])
(OUT / "subgroup_calibration_v13.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("white_minus_black_auroc_joint", "white_minus_black_specificity_joint")}, indent=1))

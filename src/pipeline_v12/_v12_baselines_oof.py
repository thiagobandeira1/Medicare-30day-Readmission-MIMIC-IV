# -*- coding: utf-8 -*-
"""Adapted LACE and 12-month HOSPITAL scores for every cohort admission, and
paired cluster-bootstrap contrasts against the out-of-fold predictions of the
selection procedure on the development partitions (primary evidence tier).
Also records top-decile capture for the model and both scores.
Aggregate output: results_v12/baselines_oof_v12.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
O11 = REPO / "results_v12"; R2 = REPO / "results_reanalysis"
D = PUB / "Dataset" / "mimic-parquet"
rng = np.random.default_rng(42)
C = pd.read_parquet(R2 / "cohort_v2.parquet")
V = pd.read_parquet(PUB / "training_table_v12.parquet",
                    columns=["hadm_id", "los_days", "admission_type", "cci_score_total",
                             "hemoglobin_last", "sodium_last", "n_procedures"])
df = C[["hadm_id", "subject_id", "partition", "excl_index_death", "label_v2"]].merge(V, on="hadm_id", how="left")
assert (df.hadm_id.values == C.hadm_id.values).all()
adm = pd.read_parquet(D / "admissions.parquet", columns=["subject_id", "hadm_id", "admittime", "edregtime"])
adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
adm["has_ed"] = adm.edregtime.notna().astype(int)
ed180 = np.zeros(len(adm), int); adm12 = np.zeros(len(adm), int)
for _s, g in adm.groupby("subject_id", sort=False):
    t = g.admittime.values.astype("datetime64[D]").astype("int64")
    cum = np.concatenate([[0], np.cumsum(g.has_ed.values)])
    ks = np.arange(len(t))
    ed180[g.index.values] = cum[ks] - cum[np.searchsorted(t, t - 180, "left")]
    adm12[g.index.values] = ks - np.searchsorted(t, t - 365, "left")
adm["ed180"] = ed180; adm["adm12"] = adm12
df = df.merge(adm[["hadm_id", "ed180", "adm12"]], on="hadm_id", how="left")
EL = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}


def lace_L(l):
    return 0 if l < 1 else (int(l) if l <= 3 else (4 if l <= 6 else (5 if l <= 13 else 7)))


lace = (df.los_days.fillna(0).apply(lace_L) + (~df.admission_type.isin(EL)).astype(int) * 3
        + df.cci_score_total.fillna(0).apply(lambda c: min(int(c), 3) if c < 4 else 5)
        + df.ed180.clip(0, 4)).to_numpy()
svc = pd.read_parquet(D / "services.parquet", columns=["hadm_id", "curr_service"])
onc = df.hadm_id.isin(set(svc.loc[svc.curr_service == "OMED", "hadm_id"])).astype(int) * 2
hosp = ((df.hemoglobin_last < 12).astype(int) + onc + (df.sodium_last < 135).astype(int)
        + (df.n_procedures >= 1).astype(int) + (~df.admission_type.isin(EL)).astype(int)
        + df.adm12.apply(lambda n: 0 if n <= 1 else (2 if n <= 5 else 5))
        + (df.los_days >= 5).astype(int) * 2).to_numpy()
np.savez(O11 / "baseline_scores_all_v12.npz", hadm_id=df.hadm_id.to_numpy(), lace=lace, hospital12=hosp)

O = np.load(O11 / "oof_predictions.npz")
keep = (~C.excl_index_death).to_numpy(); dev = keep & C.partition.isin(["train", "val"]).to_numpy()
hadm_dev = C.loc[dev, "hadm_id"].to_numpy()
# oof arrays are ordered as the development rows of the cohort (verify by label equality)
y = O["y"]; p = O["rfe"]; grp = O["groups"]
assert len(y) == dev.sum() and np.array_equal(y, C.loc[dev, "label_v2"].to_numpy())
L, H = lace[dev], hosp[dev]
R = {"n_dev": int(len(y)), "events_dev": int(y.sum()),
     "oof_auroc_rfe": round(float(roc_auc_score(y, p)), 4),
     "oof_auroc_lace": round(float(roc_auc_score(y, L)), 4),
     "oof_auroc_hospital12": round(float(roc_auc_score(y, H)), 4)}
us = np.unique(grp); rows = {u: np.where(grp == u)[0] for u in us}
dl, dh = [], []
for _ in range(1000):
    ix = np.concatenate([rows[u] for u in rng.choice(us, len(us))])
    a = roc_auc_score(y[ix], p[ix]); dl.append(a - roc_auc_score(y[ix], L[ix])); dh.append(a - roc_auc_score(y[ix], H[ix]))
for k, v in (("lace", dl), ("hospital12", dh)):
    R[f"oof_uplift_{k}"] = round(float(np.mean(v)), 4)
    R[f"oof_uplift_{k}_ci95"] = [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def cap(score, yy, frac=0.10):
    k = int(len(score) * frac); top = np.argsort(score, kind="stable")[::-1][:k]
    return {"captured": round(float(yy[top].sum() / yy.sum()), 3), "ppv": round(float(yy[top].mean()), 3)}


R["top10_capture_oof"] = {"model": cap(p, y), "lace": cap(L + rng.random(len(L)) * 1e-6, y),
                          "hospital12": cap(H + rng.random(len(H)) * 1e-6, y)}
# test partition, for Table 2 continuity
te = keep & (C.partition == "test").to_numpy()
P = np.load(O11 / "final_preds_v12.npz"); yt = P["y_te"]; pt = P["p_te"]
assert np.array_equal(yt, C.loc[te, "label_v2"].to_numpy())
R["test_auroc_lace"] = round(float(roc_auc_score(yt, lace[te])), 4)
R["test_auroc_hospital12"] = round(float(roc_auc_score(yt, hosp[te])), 4)
R["top10_capture_test"] = {"model": cap(pt, yt), "lace": cap(lace[te] + rng.random(te.sum()) * 1e-6, yt),
                           "hospital12": cap(hosp[te] + rng.random(te.sum()) * 1e-6, yt)}
R["note"] = ("LACE E component counts prior 180-day admissions with an emergency department registration; "
             "ED visits without admission are not in MIMIC-IV core tables (adapted LACE). HOSPITAL uses "
             "admissions in the prior 365 days (original 12-month window).")
(O11 / "baselines_oof_v12.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))

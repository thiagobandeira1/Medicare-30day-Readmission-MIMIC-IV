# -*- coding: utf-8 -*-
"""Timestamp audit (v8): (A) structure of the original los_trend_180d defect;
(B) recompute every history feature of the final model from raw MIMIC-IV
admissions timestamps; (C) explain outpatient (OMR) feature matches.
Writes aggregate results only: results_v7/timestamp_audit_v8.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v7"
R = {}

a = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                    columns=["subject_id", "hadm_id", "admittime", "dischtime",
                             "edregtime"])
a = a.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
a["los"] = (a.dischtime - a.admittime).dt.total_seconds() / 86400
a["losd"] = np.floor(a["los"])
V7 = pd.read_parquet(PUB / "training_table_v7.parquet",
                     columns=["subject_id", "hadm_id", "los_trend_180d"])
V10 = pd.read_parquet(PUB / "training_table_v10.parquet",
                      columns=["hadm_id", "prior_admissions_6m", "prior_mean_los_6m",
                               "prior_any_ed_6m", "prior_admissions_all",
                               "time_since_last_discharge", "prior_readmission_count"])
T = V7.merge(V10, on="hadm_id").merge(a, on=["subject_id", "hadm_id"], how="left")

# ---------------------------------------------------------------- (A) trend
multi = T.groupby("subject_id")["hadm_id"].transform("size") > 1
R["trend_patients_multi_admission"] = int(T.loc[multi, "subject_id"].nunique())
R["trend_constant_within_patient_share"] = round(float(
    T.loc[multi].groupby("subject_id")["los_trend_180d"].nunique().eq(1).mean()), 4)


def _slope(v):
    return float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) > 1 else 0.0


cand = {}
for nm in ("los", "losd"):
    d = {}
    for sid, gg in a.groupby("subject_id", sort=False):
        v = gg[nm].to_numpy(); at = gg.admittime.to_numpy()
        tail = at >= at.max() - np.timedelta64(180, "D")
        d[sid] = (_slope(v), (v[-1] - v[0]) if len(v) > 1 else 0.0, _slope(v[tail]))
    df = pd.DataFrame.from_dict(d, orient="index",
                                columns=["slope_all", "last_minus_first", "slope_final180d"])
    for c in df.columns:
        cand[f"{c}_{nm}"] = df[c]
pt = T.drop_duplicates("subject_id").set_index("subject_id")["los_trend_180d"]
R["trend_reconstruction_match"] = {
    k: round(float(np.isclose(v.reindex(pt.index).to_numpy(), pt.to_numpy(),
                              atol=1e-3).mean()), 4) for k, v in cand.items()}
print("A", R["trend_constant_within_patient_share"], R["trend_reconstruction_match"], flush=True)

# ----------------------------------------------------- (B) history features
rows = []
for sid, gg in a.groupby("subject_id", sort=False):
    at = gg.admittime.to_numpy(); dt = gg.dischtime.to_numpy()
    los = gg.los.to_numpy(); ed = gg.edregtime.to_numpy(); h = gg.hadm_id.to_numpy()
    edok = ~pd.isna(ed)
    for k in range(len(gg)):
        prior = at < at[k]
        p6 = prior & (at >= at[k] - np.timedelta64(180, "D"))
        idx = np.where(prior)[0]
        pr = 0
        for j in idx[1:]:
            gap = (at[j] - dt[j - 1]) / np.timedelta64(1, "D")
            if 0 < gap <= 30:
                pr += 1
        rows.append({"hadm_id": h[k],
                     "r_prior_all": int(prior.sum()),
                     "r_prior_6m": int(p6.sum()),
                     "r_prior_mean_los_6m": float(los[p6].mean()) if p6.any() else 0.0,
                     "r_prior_any_ed_6m": int((p6 & edok).any()),
                     "r_prior_readmit": pr,
                     "r_tsld": float((at[k] - dt[prior].max()) / np.timedelta64(1, "D"))
                     if prior.any() else np.nan})
T = T.merge(pd.DataFrame(rows), on="hadm_id", how="left")
fe = T["r_prior_all"] == 0
later = (T.groupby("subject_id").admittime.rank(method="first", ascending=False) > 1)
later_any = a.groupby("subject_id").admittime.rank(method="first", ascending=False)
T["has_later"] = T.hadm_id.map(dict(zip(a.hadm_id, later_any > 1)))

PAIRS = (("prior_admissions_all", "r_prior_all", 0),
         ("prior_admissions_6m", "r_prior_6m", 0),
         ("prior_mean_los_6m", "r_prior_mean_los_6m", 0.05),
         ("prior_any_ed_6m", "r_prior_any_ed_6m", 0),
         ("prior_readmission_count", "r_prior_readmit", 0),
         ("time_since_last_discharge", "r_tsld", 0.05))
res = {}
for f, r_, tol in PAIRS:
    x = pd.to_numeric(T[f], errors="coerce"); y = pd.to_numeric(T[r_], errors="coerce")
    ok = (x.isna() & y.isna()) | np.isclose(x.fillna(-9e9), y.fillna(-9e9), atol=tol)
    scope = ~fe if f == "time_since_last_discharge" else pd.Series(True, index=T.index)
    mm = ~ok & scope
    res[f] = {"match_share": round(float(ok[scope].mean()), 4),
              "mismatch_share_patients_with_later_admission":
                  round(float(mm[scope & T.has_later].mean()), 4),
              "mismatch_share_patients_without_later_admission":
                  round(float(mm[scope & ~T.has_later].mean()), 4)}  # per-admission examples omitted from the public repository
R["history_recompute"] = res
R["first_ever_time_since_last_discharge_values"] = sorted(
    T.loc[fe, "time_since_last_discharge"].round(2).unique().tolist())[:5]
print("B", json.dumps(res, indent=1), flush=True)

# ----------------------------------------------------------- (C) OMR matches
V7o = pd.read_parquet(PUB / "training_table_v7.parquet",
                      columns=["hadm_id", "bmi_last", "bp_diastolic_outpatient"])
omr = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "omr.parquet",
                      columns=["subject_id", "chartdate", "result_name", "result_value"])
omr["chartdate"] = pd.to_datetime(omr.chartdate)
bmi = omr[omr.result_name.isin(["BMI (kg/m2)", "BMI"])].copy()
bmi["v"] = pd.to_numeric(bmi.result_value, errors="coerce")
bmi = bmi[(bmi.v >= 12) & (bmi.v <= 80)]
bp = omr[omr.result_name.str.startswith("Blood Pressure", na=False)].copy()
bp["v"] = pd.to_numeric(bp.result_value.str.split("/", n=1, expand=True)[1], errors="coerce")
bp = bp[(bp.v >= 20) & (bp.v <= 150)]
X = a[["subject_id", "hadm_id", "admittime", "dischtime"]].merge(V7o, on="hadm_id")
omr_res = {}
for col, src in (("bmi_last", bmi), ("bp_diastolic_outpatient", bp)):
    m = X[["subject_id", "hadm_id", "admittime", "dischtime", col]].merge(
        src[["subject_id", "chartdate", "v"]], on="subject_id", how="inner")
    out = {"stored_missing_share": round(float(X[col].isna().mean()), 4)}
    for rule, cond in (("on_or_before_admission_time", m.chartdate <= m.admittime),
                       ("on_or_before_discharge_date", m.chartdate <= m.dischtime.dt.normalize()),
                       ("any_date", m.chartdate.notna())):
        mm = m[cond]
        last = mm.groupby("hadm_id").chartdate.transform("max")
        same = mm[mm.chartdate == last]
        hit = same.assign(h=np.isclose(same.v, same[col])).groupby("hadm_id").h.any()
        nval = same.groupby("hadm_id").v.nunique()
        has_val = X[col].notna()
        exists = X.hadm_id.isin(mm.hadm_id)
        ok = (X.hadm_id.map(hit).fillna(False) & has_val) | (~exists & ~has_val)
        out[rule] = {"exact_match_any_value_on_last_date": round(float(ok.mean()), 4),
                     "last_date_has_multiple_distinct_values":
                         round(float((nval > 1).sum() / max(len(nval), 1)), 4)}
    during = m[(m.chartdate > m.admittime) & (m.chartdate <= m.dischtime.dt.normalize())]
    out["admissions_with_value_charted_during_stay"] = round(
        float(X.hadm_id.isin(during.hadm_id).mean()), 4)
    omr_res[col] = out
R["omr_audit"] = omr_res
print("C", json.dumps(omr_res, indent=1), flush=True)
(OUT / "timestamp_audit_v8.json").write_text(json.dumps(R, indent=1, default=float))
print("saved")

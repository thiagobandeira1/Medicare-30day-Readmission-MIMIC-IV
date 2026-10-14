"""v7 empirical future-information test (reproducible record of the finding
that removed los_trend_180d). Writes results_v7/leak_test_v7.json.

Test logic: among FIRST-EVER admissions (no earlier admission of any kind
in MIMIC-IV), a feature built only from prior admissions carries no
information, so it must not differ between patients who will and will not
be admitted again. For every candidate feature we record the share of
first-ever rows at the modal value among patients with no later admission
versus those with one. Outpatient-record features are additionally checked
against the outpatient table (last value dated on or before discharge vs
last value overall). Formula checks document the verified definitions.
Aggregates only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
O7 = REPO / "results_v7"
D = PUB / "Dataset" / "mimic-parquet"
R2 = REPO / "results_reanalysis"
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
DF = pd.concat([V10.reset_index(drop=True),
                V7[[c for c in V7.columns if c not in V10.columns]]
                .reset_index(drop=True)], axis=1)
P5 = json.loads((REPO / "results_v5" / "eligible_pool_v5.json").read_text())
FC = json.loads((O7 / "final_model_v7.json").read_text())["final_feature_set"]
C = pd.read_parquet(R2 / "cohort_v2.parquet", columns=["hadm_id", "excl_index_death",
                                                       "label_v2"])
a = pd.read_parquet(D / "admissions.parquet",
                    columns=["subject_id", "hadm_id", "admittime", "dischtime"])
a["admittime"] = pd.to_datetime(a.admittime)
a["dischtime"] = pd.to_datetime(a.dischtime)
a = a.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
g = a.groupby("subject_id")
a["n_prior"] = g.cumcount()
a["n_future"] = g.cumcount(ascending=False)
M = DF.merge(a[["hadm_id", "n_prior", "n_future", "dischtime"]],
             on="hadm_id").merge(C, on="hadm_id")
first = M[(M.n_prior == 0) & (~M.excl_index_death)]
fut = (first.n_future > 0).to_numpy()
OUT = {"first_ever_admissions": int(len(first)),
       "share_with_later_admission": round(float(fut.mean()), 4)}

# ---- the leaked feature
nz = (first.los_trend_180d != 0).to_numpy()
OUT["los_trend_180d"] = {
    "nonzero_share_first_ever": round(float(nz.mean()), 4),
    "later_admission_share_if_nonzero": round(float(fut[nz].mean()), 4),
    "later_admission_share_if_zero": round(float(fut[~nz].mean()), 4),
    "readmit30_if_nonzero": round(float(first.label_v2.to_numpy()[nz].mean()), 4),
    "readmit30_if_zero": round(float(first.label_v2.to_numpy()[~nz].mean()), 4),
    "nonzero_share_without_any_later_admission": round(float(nz[~fut].mean()), 4),
    "auroc_nonzero_for_later_admission": round(float(roc_auc_score(fut, nz)), 4)}

# ---- scan of every candidate (v5 eligible pool + excluded features)
rows = []
for col in P5["eligible"] + [e["feature"] for e in P5["excluded"]]:
    v = first[col]
    if not pd.api.types.is_numeric_dtype(v):
        v = v.astype(str).astype("category").cat.codes.astype(float)
    v = v.astype(float).fillna(-999.0).to_numpy()
    mode = pd.Series(v).mode().iloc[0]
    p0, p1 = (v[~fut] == mode).mean(), (v[fut] == mode).mean()
    rows.append({"feature": col, "p_mode_no_later": round(float(p0), 4),
                 "p_mode_later": round(float(p1), 4), "gap": round(float(p0 - p1), 4)})
rows.sort(key=lambda r: -r["gap"])
OUT["scan_top10"] = rows[:10]
OUT["scan_deterministic_flags"] = [r["feature"] for r in rows
                                   if r["p_mode_no_later"] >= 0.999 and r["gap"] >= 0.10]
OUT["scan_max_gap_final_features"] = max(r["gap"] for r in rows if r["feature"] in FC)

# ---- outpatient-record features: time-bounded?
omr = pd.read_parquet(D / "omr.parquet")
omr["chartdate"] = pd.to_datetime(omr.chartdate)
src = {"bmi_last": omr[omr.result_name.str.contains("BMI", case=False, na=False)]
       .assign(val=lambda d: pd.to_numeric(d.result_value, errors="coerce")),
       "bp_diastolic_outpatient": omr[omr.result_name.str.fullmatch(
           "Blood Pressure", case=False, na=False)]
       .assign(val=lambda d: pd.to_numeric(d.result_value.str.split("/").str[1],
                                           errors="coerce"))}
omr_out = {}
for name, s in src.items():
    s = s.dropna(subset=["val"]).sort_values(["subject_id", "chartdate", "seq_num"])
    mm = M[["hadm_id", "subject_id", "dischtime", name]].merge(
        s[["subject_id", "chartdate", "val"]], on="subject_id", how="left")
    pre = mm[mm.chartdate <= mm.dischtime.dt.normalize()].groupby("hadm_id").val.last()
    allv = mm.groupby("hadm_id").val.last()
    st = M.set_index("hadm_id")[name]
    res = {}
    for lab, cand in (("on_or_before_discharge", pre), ("overall_incl_future", allv)):
        cc = cand.reindex(st.index)
        both = st.notna() & cc.notna()
        res[lab] = round(float((np.abs(st[both] - cc[both]) < 1e-6).mean()), 4)
    omr_out[name] = res
OUT["outpatient_features_exact_match"] = omr_out

# ---- verified formulas
MM = M.copy()
checks = {
    "freq_x_recency = prior_admissions_all / (time_since_last_discharge + 1)":
        (MM.freq_x_recency, MM.prior_admissions_all / (MM.time_since_last_discharge + 1)),
    "los_per_prior_admit = los_days / (prior_admissions_all + 1)":
        (MM.los_per_prior_admit, MM.los_days / (MM.prior_admissions_all + 1)),
    "albumin_x_los = albumin_last * los_days":
        (MM.albumin_x_los, MM.albumin_last * MM.los_days),
    "provider_fragmentation_x_los = los_days * n_distinct_providers":
        (MM.provider_fragmentation_x_los, MM.los_days * MM.n_distinct_providers),
    "log_los_days = log1p(los_days)": (MM.log_los_days, np.log1p(MM.los_days)),
    "log_n_labs = log1p(n_labs_total)": (MM.log_n_labs, np.log1p(MM.n_labs_total)),
    "log_prior_readmit_count = log1p(prior_readmission_count)":
        (MM.log_prior_readmit_count, np.log1p(MM.prior_readmission_count)),
    "prior_admits_6m_sq = prior_admissions_6m squared":
        (MM.prior_admits_6m_sq, MM.prior_admissions_6m.astype(float) ** 2),
    "prior_admissions_all = number of earlier admissions (any insurance)":
        (MM.prior_admissions_all, MM.n_prior)}
OUT["formula_checks"] = {}
for k, (x, yv) in checks.items():
    x, yv = np.asarray(x, float), np.asarray(yv, float)
    ok = np.isfinite(x) & np.isfinite(yv)
    OUT["formula_checks"][k] = round(float((np.abs(x[ok] - yv[ok]) < 1e-6).mean()), 4)
OUT["time_since_last_discharge_first_ever_value"] = round(float(
    first.time_since_last_discharge.median()), 2)
OUT["time_since_last_discharge_first_ever_distinct"] = int(
    first.time_since_last_discharge.nunique())
(O7 / "leak_test_v7.json").write_text(json.dumps(OUT, indent=1))
print(json.dumps({k: v for k, v in OUT.items() if k != "scan_top10"}, indent=1))

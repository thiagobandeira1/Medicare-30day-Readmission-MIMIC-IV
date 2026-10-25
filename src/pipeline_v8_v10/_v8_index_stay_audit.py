# -*- coding: utf-8 -*-
"""Index-stay and patient-level leak signatures (v8 review).
(1) Within-patient constancy scan: a feature computed once per patient over the
    whole record (the defect signature of los_trend_180d) is identical across
    all of a patient's admissions.
(2) Raw-event spot check on a random sample of cohort admissions: laboratory
    and order features recomputed from labevents and poe, (a) bounded by the
    index admission and discharge times, (b) unbounded within the hadm_id.
Writes aggregate results only: results_v7/index_stay_audit_v8.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v7"
RAW = PUB / "Dataset" / "mimic-parquet"
CMP = json.loads((OUT / "model_comparison.json").read_text())
FSEL = CMP["_stability"]["consensus_majority_features"]
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
R = {}

# (1) constancy scan, multi-admission patients only
multi = DF.groupby("subject_id")["hadm_id"].transform("size") > 1
M = DF.loc[multi]
scan = {}
for f in FSEL + ["los_trend_180d"]:
    nun = M.groupby("subject_id")[f].nunique(dropna=False)
    scan[f] = round(float((nun == 1).mean()), 4)
R["constant_within_patient_share"] = dict(sorted(scan.items(), key=lambda kv: -kv[1]))
R["n_multi_admission_patients"] = int(M.subject_id.nunique())
print(json.dumps(R["constant_within_patient_share"], indent=0)[:1500], flush=True)

# (2) raw-event spot check
adm = pd.read_parquet(RAW / "admissions.parquet", columns=["hadm_id", "admittime", "dischtime"])
rng = np.random.default_rng(7)
samp = rng.choice(DF.hadm_id.to_numpy(), 3000, replace=False)
S = DF.loc[DF.hadm_id.isin(samp), ["hadm_id", "n_labs_total", "n_lab_item_types",
                                     "lab_abnormal_rate", "orders_last_6h",
                                     "n_order_types"]].merge(adm, on="hadm_id")
lab = ds.dataset(RAW / "labevents.parquet").to_table(
    columns=["hadm_id", "charttime", "itemid", "flag"],
    filter=ds.field("hadm_id").isin(samp.tolist())).to_pandas()
poe = ds.dataset(RAW / "poe.parquet").to_table(
    columns=["hadm_id", "ordertime", "order_type"],
    filter=ds.field("hadm_id").isin(samp.tolist())).to_pandas()
lab = lab.merge(adm, on="hadm_id"); poe = poe.merge(adm, on="hadm_id")
inb = (lab.charttime >= lab.admittime) & (lab.charttime <= lab.dischtime)
pinb = (poe.ordertime >= poe.admittime) & (poe.ordertime <= poe.dischtime)
last6 = (poe.ordertime > poe.dischtime - pd.Timedelta(hours=6)) & (poe.ordertime <= poe.dischtime)
calc = pd.DataFrame({"hadm_id": S.hadm_id})
calc["labs_all"] = calc.hadm_id.map(lab.groupby("hadm_id").size()).fillna(0)
calc["labs_inb"] = calc.hadm_id.map(lab[inb].groupby("hadm_id").size()).fillna(0)
calc["items_all"] = calc.hadm_id.map(lab.groupby("hadm_id").itemid.nunique()).fillna(0)
calc["items_inb"] = calc.hadm_id.map(lab[inb].groupby("hadm_id").itemid.nunique()).fillna(0)
ab = lab.assign(a=lab.flag.astype(str).str.lower().eq("abnormal"))
calc["abn_all"] = calc.hadm_id.map(ab.groupby("hadm_id").a.mean())
calc["abn_inb"] = calc.hadm_id.map(ab[inb].groupby("hadm_id").a.mean())
calc["ord6_all_hadm"] = calc.hadm_id.map(poe[last6].groupby("hadm_id").size()).fillna(0)
calc["otypes_all"] = calc.hadm_id.map(poe.groupby("hadm_id").order_type.nunique()).fillna(0)
calc["otypes_inb"] = calc.hadm_id.map(poe[pinb].groupby("hadm_id").order_type.nunique()).fillna(0)
X = S.merge(calc, on="hadm_id")


def mt(a_, b_, tol=1e-6):
    a_ = pd.to_numeric(X[a_], errors="coerce"); b_ = pd.to_numeric(X[b_], errors="coerce")
    ok = (a_.isna() & b_.isna()) | np.isclose(a_.fillna(-1), b_.fillna(-1), atol=tol)
    return round(float(ok.mean()), 4)


R["raw_spot_check_n"] = int(len(X))
R["raw_spot_check"] = {
    "n_labs_total": {"bounded_by_stay": mt("n_labs_total", "labs_inb"),
                     "all_rows_of_hadm": mt("n_labs_total", "labs_all")},
    "n_lab_item_types": {"bounded_by_stay": mt("n_lab_item_types", "items_inb"),
                         "all_rows_of_hadm": mt("n_lab_item_types", "items_all")},
    "lab_abnormal_rate": {"bounded_by_stay": mt("lab_abnormal_rate", "abn_inb", 1e-3),
                          "all_rows_of_hadm": mt("lab_abnormal_rate", "abn_all", 1e-3)},
    "orders_last_6h": {"final_6h_before_discharge": mt("orders_last_6h", "ord6_all_hadm")},
    "n_order_types": {"bounded_by_stay": mt("n_order_types", "otypes_inb"),
                      "all_rows_of_hadm": mt("n_order_types", "otypes_all")},
    "lab_rows_after_discharge_share": round(float((lab.charttime > lab.dischtime).mean()), 4),
    "poe_rows_after_discharge_share": round(float((poe.ordertime > poe.dischtime).mean()), 4)}
print(json.dumps(R["raw_spot_check"], indent=1), flush=True)
print(X[["n_labs_total", "labs_inb", "labs_all", "orders_last_6h", "ord6_all_hadm",
         "n_order_types", "otypes_all"]].head(8).to_string())
(OUT / "index_stay_audit_v8.json").write_text(json.dumps(R, indent=1))

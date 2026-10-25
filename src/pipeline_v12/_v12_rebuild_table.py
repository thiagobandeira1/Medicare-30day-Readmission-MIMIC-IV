# -*- coding: utf-8 -*-
"""v12 feature table = v11 table with the discharge cutoff enforced on the
remaining index-stay aggregates that lacked one: laboratory result counts
(n_labs_total, n_labs_abnormal, lab_abnormal_rate, log_n_labs,
n_lab_item_types, n_critical_labs; results kept only if charted and stored
on or before discharge) and n_meds_total (prescriptions started on or before
discharge). Writes training_table_v12.parquet (local) and
results_v12/table_rebuild_v12.json (aggregate; includes the v11 report)."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, duckdb

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v12"; OUT.mkdir(exist_ok=True)
RAW = Path(r"C:\Users\Thiago\Documents\Coursework\Capstone Project\Dataset\mimic-iv-3.1 - Full\mimic-iv-3.1")
PQ = PUB / "Dataset" / "mimic-parquet"
t0 = time.time()
DF = pd.read_parquet(PUB / "training_table_v11.parquet"); OLD = DF.copy()
adm = pd.read_parquet(PQ / "admissions.parquet", columns=["hadm_id", "dischtime"])
coh = DF[["hadm_id"]].merge(adm, on="hadm_id", how="left")
con = duckdb.connect(); con.register("cohort", coh)
lab_csv = str(RAW / "hosp" / "labevents.csv.gz").replace("\\", "/")
COLS = ("columns={'labevent_id':'BIGINT','subject_id':'BIGINT','hadm_id':'BIGINT','specimen_id':'BIGINT',"
        "'itemid':'INTEGER','order_provider_id':'VARCHAR','charttime':'TIMESTAMP','storetime':'TIMESTAMP',"
        "'value':'VARCHAR','valuenum':'DOUBLE','valueuom':'VARCHAR','ref_range_lower':'DOUBLE',"
        "'ref_range_upper':'DOUBLE','flag':'VARCHAR','priority':'VARCHAR','comments':'VARCHAR'}")
q = f"""
SELECT l.hadm_id,
  SUM(CASE WHEN l.valuenum IS NOT NULL THEN 1 ELSE 0 END) AS n_labs_total,
  SUM(CASE WHEN lower(l.flag) = 'abnormal' THEN 1 ELSE 0 END) AS n_labs_abnormal,
  COUNT(DISTINCT l.itemid) AS n_lab_item_types,
  SUM(CASE WHEN l.valuenum IS NOT NULL AND l.valuenum > 0 AND (
        (l.itemid=50912 AND l.valuenum > 4.0) OR (l.itemid=50971 AND (l.valuenum > 6.0 OR l.valuenum < 2.5))
     OR (l.itemid=50983 AND (l.valuenum > 160 OR l.valuenum < 120)) OR (l.itemid=50931 AND (l.valuenum > 500 OR l.valuenum < 40))
     OR (l.itemid=50813 AND l.valuenum > 4.0)) THEN 1 ELSE 0 END) AS n_critical_labs
FROM read_csv('{lab_csv}', header=true, auto_detect=true, quote='"', escape='"', strict_mode=false, {COLS}) l
INNER JOIN cohort c ON l.hadm_id = c.hadm_id
WHERE l.charttime <= c.dischtime AND COALESCE(l.storetime, l.charttime) <= c.dischtime
GROUP BY l.hadm_id"""
labs = con.execute(q).fetchdf(); print(f"labs {labs.shape} {time.time()-t0:.0f}s", flush=True)
rx = pd.read_parquet(PQ / "prescriptions.parquet", columns=["hadm_id", "starttime"])
rx = rx[rx.hadm_id.isin(coh.hadm_id)].merge(adm, on="hadm_id")
nmeds = rx[rx.starttime <= rx.dischtime].groupby("hadm_id").size().rename("n_meds_total")
NEW = DF[["hadm_id"]].merge(labs, on="hadm_id", how="left").merge(nmeds, on="hadm_id", how="left")
for c in ("n_labs_total", "n_labs_abnormal", "n_lab_item_types", "n_critical_labs", "n_meds_total"):
    NEW[c] = NEW[c].fillna(0).astype(float)
rep = json.loads((REPO / "results_v11" / "table_rebuild_v11.json").read_text())
rep["v12_cutoff_enforced"] = {}
for c in ("n_labs_total", "n_labs_abnormal", "n_lab_item_types", "n_critical_labs", "n_meds_total"):
    if c not in DF.columns:
        continue
    old = pd.to_numeric(DF[c], errors="coerce").fillna(0); new = NEW[c]
    rep["v12_cutoff_enforced"][c] = {"share_changed": round(float((~np.isclose(old, new)).mean()), 4),
                                     "share_decreased": round(float((new < old - 1e-9).mean()), 4),
                                     "mean_old": round(float(old.mean()), 3), "mean_new": round(float(new.mean()), 3)}
    DF[c] = new
DF["log_n_labs"] = np.log1p(DF.n_labs_total)
DF["lab_abnormal_rate"] = NEW.n_labs_abnormal / DF.n_labs_total.clip(lower=1)
for c in ("log_n_labs", "lab_abnormal_rate"):
    rep["v12_cutoff_enforced"][c] = {"share_changed": round(float((~np.isclose(OLD[c].astype(float), DF[c])).mean()), 4)}
rep["v12_duplicates_removed_from_pool"] = {"orders_first_24h": "identical to orders_per_day (proxy)",
                                           "nonenglish_x_prior_admits": "identical to prior_admissions_6m"}
rep["v12_seconds"] = round(time.time() - t0)
DF.to_parquet(PUB / "training_table_v12.parquet", index=False)
(OUT / "table_rebuild_v12.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep["v12_cutoff_enforced"], indent=1))

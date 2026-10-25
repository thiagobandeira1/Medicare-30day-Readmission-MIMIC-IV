# -*- coding: utf-8 -*-
"""v11 feature table: rebuild every continuous feature that the capstone
pipeline had median- or zero-imputed over the whole table, from raw MIMIC-IV
tables, with missing values left missing. Laboratory features additionally
require the result to have been stored (storetime) on or before discharge.
Writes training_table_v11.parquet (local, row level, never committed) and
results_v11/table_rebuild_v11.json (aggregate)."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v11"; OUT.mkdir(exist_ok=True)
RAW = Path(r"C:\Users\Thiago\Documents\Coursework\Capstone Project\Dataset\mimic-iv-3.1 - Full\mimic-iv-3.1")
PQ = PUB / "Dataset" / "mimic-parquet"
t0 = time.time()
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
DF = pd.concat([V10.reset_index(drop=True),
                V7[[c for c in V7.columns if c not in V10.columns]].reset_index(drop=True)], axis=1)
OLD = DF.copy()
adm = pd.read_parquet(PQ / "admissions.parquet", columns=["hadm_id", "dischtime"])
coh = DF[["hadm_id"]].merge(adm, on="hadm_id", how="left")
assert coh.dischtime.notna().all()
con = duckdb.connect()
con.register("cohort", coh)

# ------------------------------------------------------------------ labs
ITEMS = {50912: "creatinine", 51006: "bun", 51222: "hemoglobin", 50983: "sodium",
         50971: "potassium", 50862: "albumin", 51301: "wbc", 50931: "glucose",
         50963: "bnp", 50813: "lactate", 51237: "inr", 50882: "bicarbonate",
         50960: "magnesium", 50970: "phosphate", 50885: "bilirubin"}
lab_csv = str(RAW / "hosp" / "labevents.csv.gz").replace("\\", "/")
q = f"""
WITH f AS (
  SELECT l.hadm_id, l.itemid, l.valuenum, l.charttime,
         ROW_NUMBER() OVER (PARTITION BY l.hadm_id, l.itemid ORDER BY l.charttime DESC) AS rn_last,
         ROW_NUMBER() OVER (PARTITION BY l.hadm_id, l.itemid ORDER BY l.charttime ASC)  AS rn_first
  FROM read_csv('{lab_csv}', header=true, auto_detect=true, quote='"', escape='"', strict_mode=false,
                columns={{'labevent_id':'BIGINT','subject_id':'BIGINT','hadm_id':'BIGINT',
                          'specimen_id':'BIGINT','itemid':'INTEGER','order_provider_id':'VARCHAR',
                          'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR',
                          'valuenum':'DOUBLE','valueuom':'VARCHAR','ref_range_lower':'DOUBLE',
                          'ref_range_upper':'DOUBLE','flag':'VARCHAR','priority':'VARCHAR',
                          'comments':'VARCHAR'}}) l
  INNER JOIN cohort c ON l.hadm_id = c.hadm_id
  WHERE l.itemid IN ({",".join(map(str, ITEMS))})
    AND l.charttime <= c.dischtime
    AND COALESCE(l.storetime, l.charttime) <= c.dischtime
    AND l.valuenum IS NOT NULL AND l.valuenum > 0
)
SELECT hadm_id,
  MAX(CASE WHEN itemid=50912 AND rn_last=1 THEN valuenum END) AS creatinine_last,
  MAX(CASE WHEN itemid=50912 THEN valuenum END) AS creatinine_max,
  MAX(CASE WHEN itemid=50912 AND rn_last=1 THEN valuenum END)
    - MAX(CASE WHEN itemid=50912 AND rn_first=1 THEN valuenum END) AS creatinine_delta,
  MAX(CASE WHEN itemid=51006 AND rn_last=1 THEN valuenum END) AS bun_last,
  MAX(CASE WHEN itemid=51222 AND rn_last=1 THEN valuenum END) AS hemoglobin_last,
  MIN(CASE WHEN itemid=51222 THEN valuenum END) AS hemoglobin_min,
  MAX(CASE WHEN itemid=50983 AND rn_last=1 THEN valuenum END) AS sodium_last,
  MAX(CASE WHEN itemid=50971 AND rn_last=1 THEN valuenum END) AS potassium_last,
  MAX(CASE WHEN itemid=50862 AND rn_last=1 THEN valuenum END) AS albumin_last,
  MAX(CASE WHEN itemid=51301 AND rn_last=1 THEN valuenum END) AS wbc_last,
  MAX(CASE WHEN itemid=50931 AND rn_last=1 THEN valuenum END) AS glucose_last,
  MAX(CASE WHEN itemid=50931 THEN valuenum END) AS glucose_max,
  MAX(CASE WHEN itemid=50963 AND rn_last=1 THEN valuenum END) AS bnp_last,
  MAX(CASE WHEN itemid=50813 THEN valuenum END) AS lactate_max,
  MAX(CASE WHEN itemid=51237 THEN valuenum END) AS inr_max,
  MAX(CASE WHEN itemid=50882 AND rn_last=1 THEN valuenum END) AS bicarbonate_last,
  MAX(CASE WHEN itemid=50960 AND rn_last=1 THEN valuenum END) AS magnesium_last,
  MAX(CASE WHEN itemid=50970 AND rn_last=1 THEN valuenum END) AS phosphate_last,
  MAX(CASE WHEN itemid=50885 THEN valuenum END) AS bilirubin_max
FROM f GROUP BY hadm_id"""
labs = con.execute(q).fetchdf()
print(f"labs {labs.shape} {time.time()-t0:.0f}s", flush=True)
# stored-after-discharge share (results collected before, stored after discharge)
q2 = f"""
SELECT COUNT(*) AS n, SUM(CASE WHEN l.storetime > c.dischtime THEN 1 ELSE 0 END) AS late
FROM read_csv('{lab_csv}', header=true, auto_detect=true, quote='"', escape='"', strict_mode=false,
              columns={{'labevent_id':'BIGINT','subject_id':'BIGINT','hadm_id':'BIGINT',
                        'specimen_id':'BIGINT','itemid':'INTEGER','order_provider_id':'VARCHAR',
                        'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR',
                        'valuenum':'DOUBLE','valueuom':'VARCHAR','ref_range_lower':'DOUBLE',
                        'ref_range_upper':'DOUBLE','flag':'VARCHAR','priority':'VARCHAR',
                        'comments':'VARCHAR'}}) l
INNER JOIN cohort c ON l.hadm_id = c.hadm_id
WHERE l.itemid IN ({",".join(map(str, ITEMS))}) AND l.charttime <= c.dischtime
  AND l.valuenum IS NOT NULL AND l.valuenum > 0"""
n_all, n_late = con.execute(q2).fetchone()
print(f"rows charted before discharge: {n_all:,}; stored after discharge: {n_late:,}", flush=True)

# ---------------------------------------------------------------- charts
chart = str(PQ / "chartevents.parquet").replace("\\", "/")
q3 = f"""
WITH v AS (SELECT v.hadm_id, v.charttime, v.itemid, v.valuenum FROM read_parquet('{chart}') v
           INNER JOIN cohort c ON v.hadm_id=c.hadm_id
           WHERE v.itemid IN (220045,220050,220739,223900,223901) AND v.charttime <= c.dischtime
             AND v.valuenum IS NOT NULL),
g AS (SELECT hadm_id, charttime,
        MAX(CASE WHEN itemid=220739 THEN valuenum END) AS e,
        MAX(CASE WHEN itemid=223900 THEN valuenum END) AS vb,
        MAX(CASE WHEN itemid=223901 THEN valuenum END) AS m
      FROM v WHERE itemid IN (220739,223900,223901) GROUP BY hadm_id, charttime),
gt AS (SELECT hadm_id, COALESCE(e,1)+COALESCE(vb,1)+COALESCE(m,1) AS gcs FROM g
       WHERE e IS NOT NULL OR vb IS NOT NULL OR m IS NOT NULL),
s AS (SELECT hadm_id,
        STDDEV(CASE WHEN itemid=220045 AND valuenum>0 THEN valuenum END) AS hr_std,
        STDDEV(CASE WHEN itemid=220050 AND valuenum>0 THEN valuenum END) AS sbp_std
      FROM v GROUP BY hadm_id)
SELECT c.hadm_id, s.hr_std, s.sbp_std, (SELECT MIN(gcs) FROM gt WHERE gt.hadm_id=c.hadm_id) AS gcs_min
FROM cohort c LEFT JOIN s ON s.hadm_id=c.hadm_id"""
ch = con.execute(q3).fetchdf()
print(f"charts {ch.shape} {time.time()-t0:.0f}s", flush=True)
icu = pd.read_parquet(PQ / "icustays.parquet", columns=["hadm_id", "intime", "outtime"])
icu = icu[icu.hadm_id.isin(coh.hadm_id)].merge(adm, on="hadm_id")
icu["hrs"] = (icu.outtime - icu.intime).dt.total_seconds().clip(lower=0) / 3600   # original definition
icu_agg = icu.groupby("hadm_id").agg(icu_total_hrs=("hrs", "sum"), max_unit_los_days=("hrs", "max"))
icu_agg["max_unit_los_days"] /= 24

# ---------------------------------------------------------------- assemble
NEW = DF[["hadm_id"]].merge(labs, on="hadm_id", how="left").merge(ch, on="hadm_id", how="left") \
    .merge(icu_agg, on="hadm_id", how="left")
assert (NEW.hadm_id.values == DF.hadm_id.values).all()
NEW["icu_total_hrs"] = NEW.icu_total_hrs.fillna(0.0)
NEW["max_unit_los_days"] = NEW.max_unit_los_days.fillna(0.0)
rep = {}
for c in [c for c in NEW.columns if c != "hadm_id" and c in DF.columns]:
    old = pd.to_numeric(DF[c], errors="coerce"); new = NEW[c]
    both = old.notna() & new.notna()
    rep[c] = {"old_missing": round(float(old.isna().mean()), 4),
              "new_missing": round(float(new.isna().mean()), 4),
              "agree_where_both_present": round(float(np.isclose(old[both], new[both], atol=1e-6).mean()), 4),
              "old_value_where_new_missing_mode": (float(old[new.isna()].mode().iloc[0])
                                                   if new.isna().any() else None)}
    DF[c] = new.astype(float)
# derived terms
DF["albumin_x_los"] = DF.albumin_last * DF.los_days
DF["creatinine_x_bun"] = DF.creatinine_last * DF.bun_last
DF["bun_creatinine_ratio"] = DF.bun_last / DF.creatinine_last.clip(lower=0.1)
DF["sodium_abnormal"] = ((DF.sodium_last < 135) | (DF.sodium_last > 145)).astype(int)
DF["potassium_abnormal"] = ((DF.potassium_last < 3.5) | (DF.potassium_last > 5.0)).astype(int)
DF["wbc_high"] = (DF.wbc_last > 11).astype(int)
for c in ("sodium_abnormal", "potassium_abnormal", "wbc_high"):
    rep[c] = {"agree_with_old": round(float((DF[c] == OLD[c]).mean()), 4)}
# time since last discharge: missing when no earlier admission
tsld = pd.to_numeric(DF.time_since_last_discharge, errors="coerce")
noprior = np.isclose(tsld, 80.22013888888888, atol=1e-6) & (DF.prior_admissions_all == 0)
DF.loc[noprior, "time_since_last_discharge"] = np.nan
DF["log_time_since_discharge"] = np.log1p(DF.time_since_last_discharge)
DF["freq_x_recency"] = np.where(DF.prior_admissions_all == 0, 0.0,
                                DF.prior_admissions_all / (DF.time_since_last_discharge + 1.0))
rep["time_since_last_discharge"] = {"set_missing_share": round(float(noprior.mean()), 4)}
rep["lab_rows_stored_after_discharge"] = {"n_rows": int(n_all), "n_stored_after": int(n_late),
                                          "share": round(n_late / n_all, 5)}
rep["seconds"] = round(time.time() - t0)
DF.to_parquet(PUB / "training_table_v11.parquet", index=False)
(OUT / "table_rebuild_v11.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))

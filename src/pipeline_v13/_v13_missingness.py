# -*- coding: utf-8 -*-
"""True missingness of every audited feature in the rebuilt v11 table,
overall, by partition, and by race group (White, Black). Aggregate only."""
import json
from pathlib import Path
import numpy as np, pandas as pd
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
DF = pd.read_parquet(PUB / "training_table_v13.parquet")
C = pd.read_parquet(REPO / "results_reanalysis" / "cohort_v2.parquet")
assert (DF.hadm_id.values == C.hadm_id.values).all()
A = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet", columns=["hadm_id", "race"])
r = DF[["hadm_id"]].merge(A, on="hadm_id", how="left").race.astype(str)
race = np.select([r.str.contains("WHITE"), r.str.contains("BLACK"), r.str.contains("HISPANIC|LATINO"), r.str.contains("ASIAN")], ["White", "Black", "Hispanic", "Asian"], "Other")
keep = (~C.excl_index_death).to_numpy(); part = C.partition.to_numpy()
audit = pd.read_csv(REPO / "results_v4" / "feature_audit_207.csv")
out = {}
for f in audit.feature_name:
    if f not in DF.columns:
        continue
    m = DF[f].isna().to_numpy()
    out[f] = {"overall": round(float(m[keep].mean()), 4),
              "by_partition": {p: round(float(m[keep & (part == p)].mean()), 4) for p in ("train", "val", "test")},
              "by_race_group": {g: round(float(m[keep & (race == g)].mean()), 4) for g in ("White", "Black", "Hispanic", "Asian", "Other")}}
(REPO / "results_v13" / "missingness_v13.json").write_text(json.dumps(out, indent=1))
nz = {k: v["overall"] for k, v in out.items() if v["overall"] > 0}
print(len(out), "features;", len(nz), "with missingness; e.g.", dict(list(sorted(nz.items(), key=lambda kv: -kv[1]))[:8]))

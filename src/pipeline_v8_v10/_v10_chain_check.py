import json
from pathlib import Path
import numpy as np, pandas as pd
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
C = pd.read_parquet(REPO / "results_reanalysis" / "cohort_v2.parquet")
a = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet", columns=["subject_id","hadm_id","admittime","dischtime","deathtime"])
a = a.sort_values(["subject_id","admittime","hadm_id"]).reset_index(drop=True)
g = a.groupby("subject_id")
a["adm1"]=g.admittime.shift(-1); a["dis1"]=g.dischtime.shift(-1); a["adm2"]=g.admittime.shift(-2); a["death1"]=g.deathtime.shift(-1)
X = C[["hadm_id","excl_index_death","label_v2"]].merge(a,on="hadm_id")
X = X[~X.excl_index_death & (X.label_v2==1)]
gap = (X.adm1-X.dischtime).dt.total_seconds()/3600
B = X[(gap>0)&(gap<=1)&X.death1.isna()]
d_end_h = (B.adm2-B.dis1).dt.total_seconds()/3600
ret = d_end_h[(d_end_h>0)&(d_end_h<=720)]
print("returns", len(ret), "within 1h of continuation discharge", int((ret<=1).sum()), "within 24h", int((ret<=24).sum()))
print(ret.quantile([.1,.25,.5]).round(1).to_dict())

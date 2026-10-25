# -*- coding: utf-8 -*-
"""For 1-hour continuation events: how many patients have a further admission
within 30 days of the index discharge (or of the continuation's discharge)?
Relabeling them as nonevents would hide those returns."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
C = pd.read_parquet(REPO / "results_reanalysis" / "cohort_v2.parquet")
a = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                    columns=["subject_id", "hadm_id", "admittime", "dischtime"])
a = a.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
g = a.groupby("subject_id")
for k in (1, 2):
    a[f"adm{k}"] = g.admittime.shift(-k); a[f"dis{k}"] = g.dischtime.shift(-k)
X = C[["hadm_id", "partition", "excl_index_death", "label_v2"]].merge(a, on="hadm_id", how="left")
X = X[~X.excl_index_death & (X.label_v2 == 1)]
gap1 = (X.adm1 - X.dischtime).dt.total_seconds() / 3600
band = X[(gap1 > 0) & (gap1 <= 1)].copy()
d_idx = (band.adm2 - band.dischtime).dt.total_seconds() / 86400
d_cont = (band.adm2 - band.dis1).dt.total_seconds() / 86400
R = {"band_events": int(len(band)),
     "band_test": int((band.partition == "test").sum()),
     "second_admission_within_30d_of_index_discharge": int(((d_idx > 0) & (d_idx <= 30)).sum()),
     "second_admission_within_30d_of_continuation_discharge": int(((d_cont > 0) & (d_cont <= 30)).sum()),
     "continuation_los_days_median": round(float(((band.dis1 - band.adm1).dt.total_seconds() / 86400).median()), 2)}
R["share_hidden_if_relabeled"] = round(R["second_admission_within_30d_of_index_discharge"] / R["band_events"], 4)
tb = band[band.partition == "test"]
d_t = (tb.adm2 - tb.dischtime).dt.total_seconds() / 86400
R["test_second_admission_within_30d_of_index_discharge"] = int(((d_t > 0) & (d_t <= 30)).sum())
print(json.dumps(R, indent=1))
(REPO / "results_v7" / "continuation_followup_v9.json").write_text(json.dumps(R, indent=1))

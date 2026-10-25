# -*- coding: utf-8 -*-
"""How are 1-hour continuations represented in MIMIC-IV, and how much of the
30-day window after the index discharge is spent inside the continuation?"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
C = pd.read_parquet(REPO / "results_reanalysis" / "cohort_v2.parquet")
a = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                    columns=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                             "admission_type", "admission_location", "discharge_location"])
a = a.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
g = a.groupby("subject_id")
for k in (1, 2):
    for c in ("admittime", "dischtime", "deathtime", "admission_type"):
        a[f"{c}{k}"] = g[c].shift(-k)
X = C[["hadm_id", "partition", "excl_index_death", "label_v2"]].merge(a, on="hadm_id")
X = X[~X.excl_index_death & (X.label_v2 == 1)]
gap_min = (X.admittime1 - X.dischtime).dt.total_seconds() / 60
B = X[(gap_min > 0) & (gap_min <= 60)].copy()
gm = gap_min[B.index]
los_c = (B.dischtime1 - B.admittime1).dt.total_seconds() / 86400
R = {"n": int(len(B)),
     "gap_minutes_quantiles": [round(float(q), 1) for q in gm.quantile([.1, .25, .5, .75, .9])],
     "continuation_los_days_quantiles": [round(float(q), 2) for q in los_c.quantile([.1, .25, .5, .75, .9])],
     "continuation_extends_past_day30_share": round(float((B.dischtime1 - B.dischtime).dt.total_seconds().gt(30 * 86400).mean()), 4),
     "continuation_ends_in_death_share": round(float(B.deathtime1.notna().mean()), 4),
     "continuation_admission_type_top": B.admission_type1.value_counts(normalize=True).round(3).head(5).to_dict()}
d_idx = (B.admittime2 - B.dischtime).dt.total_seconds() / 86400
d_end = (B.admittime2 - B.dischtime1).dt.total_seconds() / 86400
R["return_within_30d_of_index_discharge"] = int(((d_idx > 0) & (d_idx <= 30)).sum())
R["return_within_30d_of_episode_end"] = int(((d_end > 0) & (d_end <= 30)).sum())
R["return_counted_only_by_episode_end"] = int((((d_end > 0) & (d_end <= 30)) & ~((d_idx > 0) & (d_idx <= 30))).sum())
R["return_counted_only_by_index_discharge"] = int((((d_idx > 0) & (d_idx <= 30)) & ~((d_end > 0) & (d_end <= 30))).sum())
R["median_days_of_window_spent_in_continuation"] = round(float(np.minimum(los_c, 30).median()), 2)
print(json.dumps(R, indent=1, default=str))
(REPO / "results_v7" / "continuation_time_origin_v10.json").write_text(json.dumps(R, indent=1, default=str))

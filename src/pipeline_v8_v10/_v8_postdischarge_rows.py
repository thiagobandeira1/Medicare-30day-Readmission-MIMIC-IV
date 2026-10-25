# -*- coding: utf-8 -*-
"""How common are hadm-scoped lab/order rows timestamped after discharge, and
are they associated with the outcome? Sample of 20,000 cohort admissions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import roc_auc_score
PUB = Path(__file__).resolve().parent.parent
OUT = PUB / "medicare-30day-readmission-mimic-iv" / "results_v7"
RAW = PUB / "Dataset" / "mimic-parquet"
C = pd.read_parquet(PUB / "medicare-30day-readmission-mimic-iv" / "results_reanalysis" / "cohort_v2.parquet",
                    columns=["hadm_id", "label_v2", "excl_index_death"])
C = C[~C.excl_index_death]
adm = pd.read_parquet(RAW / "admissions.parquet", columns=["hadm_id", "admittime", "dischtime"])
samp = np.random.default_rng(11).choice(C.hadm_id.to_numpy(), 20000, replace=False).tolist()
res = {"n_sample": 20000}
for tab, tcol in (("labevents", "charttime"), ("poe", "ordertime")):
    t = ds.dataset(RAW / f"{tab}.parquet").to_table(columns=["hadm_id", tcol],
                                                     filter=ds.field("hadm_id").isin(samp)).to_pandas()
    t = t.merge(adm, on="hadm_id")
    post = t[t[tcol] > t.dischtime]
    hrs = (post[tcol] - post.dischtime).dt.total_seconds() / 3600
    S = C[C.hadm_id.isin(samp)].copy()
    S["n_post"] = S.hadm_id.map(post.groupby("hadm_id").size()).fillna(0)
    res[tab] = {"row_share_after_discharge": round(float(len(post) / max(len(t), 1)), 4),
                "admissions_with_any_post_row": round(float((S.n_post > 0).mean()), 4),
                "median_hours_after_discharge": round(float(hrs.median()), 2) if len(hrs) else None,
                "p90_hours_after_discharge": round(float(hrs.quantile(.9)), 2) if len(hrs) else None,
                "readmit_rate_if_any_post": round(float(S.loc[S.n_post > 0, "label_v2"].mean()), 4),
                "readmit_rate_if_none": round(float(S.loc[S.n_post == 0, "label_v2"].mean()), 4),
                "auroc_any_post_for_outcome": round(float(roc_auc_score(S.label_v2, S.n_post > 0)), 4)}
print(json.dumps(res, indent=1))
(OUT / "postdischarge_rows_v8.json").write_text(json.dumps(res, indent=1))

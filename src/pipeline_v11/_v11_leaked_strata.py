# -*- coding: utf-8 -*-
"""Pre- vs post-correction test AUROC by prior-admission stratum (same rows)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
old = np.load(REPO / "results_v5" / "final_preds_v5.npz")
new = np.load(REPO / "results_v11" / "final_preds_v11.npz")
assert (old["y_te"] == new["y_te"]).all() and (old["s_te"] == new["s_te"]).all()
C = pd.read_parquet(REPO / "results_reanalysis" / "cohort_v2.parquet")
te = (~C.excl_index_death) & (C.partition == "test")
V = pd.read_parquet(PUB / "training_table_v11.parquet", columns=["hadm_id", "prior_admissions_all"])
pa = pd.DataFrame({"hadm_id": C.loc[te, "hadm_id"].to_numpy()}).merge(V, on="hadm_id", how="left").prior_admissions_all.to_numpy()
y = new["y_te"]
R = {"overall": {"pre": round(float(roc_auc_score(y, old["p_te"])), 4),
                 "post": round(float(roc_auc_score(y, new["p_te"])), 4)}}
for nm, m in (("0", pa == 0), ("1 to 2", (pa >= 1) & (pa <= 2)), ("3 to 9", (pa >= 3) & (pa <= 9)), ("10 or more", pa >= 10)):
    R[nm] = {"n": int(m.sum()), "pre": round(float(roc_auc_score(y[m], old["p_te"][m])), 4),
             "post": round(float(roc_auc_score(y[m], new["p_te"][m])), 4)}
print(json.dumps(R, indent=1))
(REPO / "results_v11" / "pre_post_strata_v11.json").write_text(json.dumps(R, indent=1))

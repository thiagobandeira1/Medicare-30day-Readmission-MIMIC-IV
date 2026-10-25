# -*- coding: utf-8 -*-
"""Try to reproduce the original los_trend_180d from cohort (Medicare) admissions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
PUB = Path(__file__).resolve().parent.parent
OUT = PUB / "medicare-30day-readmission-mimic-iv" / "results_v7"
V = pd.read_parquet(PUB / "training_table_v7.parquet", columns=["subject_id", "hadm_id", "los_trend_180d", "admittime_dt", "dischtime_dt"])
V["los"] = (V.dischtime_dt - V.admittime_dt).dt.total_seconds() / 86400
V["losd"] = np.floor(V.los)
V = V.sort_values(["subject_id", "admittime_dt"])
res = {}
def sl(v):
    return float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) > 1 else 0.0
def sl_t(t, v):
    return float(np.polyfit(t, v, 1)[0]) if len(v) > 1 and np.ptp(t) > 0 else 0.0
cands = {}
for sid, g in V.groupby("subject_id", sort=False):
    t = (g.admittime_dt - g.admittime_dt.iloc[0]).dt.total_seconds().to_numpy() / 86400
    lastw = g.admittime_dt >= g.admittime_dt.max() - pd.Timedelta(days=180)
    firstw = g.admittime_dt <= g.admittime_dt.min() + pd.Timedelta(days=180)
    for nm in ("los", "losd"):
        v = g[nm].to_numpy()
        cands.setdefault(f"slope_idx_{nm}", {})[sid] = sl(v)
        cands.setdefault(f"slope_days_{nm}", {})[sid] = sl_t(t, v)
        cands.setdefault(f"lastminusfirst_{nm}", {})[sid] = v[-1] - v[0] if len(v) > 1 else 0.0
        cands.setdefault(f"slope_idx_last180_{nm}", {})[sid] = sl(g.loc[lastw, nm].to_numpy())
        cands.setdefault(f"slope_idx_first180_{nm}", {})[sid] = sl(g.loc[firstw, nm].to_numpy())
        cands.setdefault(f"last_minus_mean_{nm}", {})[sid] = v[-1] - v.mean() if len(v) > 1 else 0.0
pt = V.drop_duplicates("subject_id").set_index("subject_id").los_trend_180d
for k, d in cands.items():
    s = pd.Series(d).reindex(pt.index)
    res[k] = round(float(np.isclose(s, pt, atol=1e-3).mean()), 4)
res["share_patients_single_admission_nonzero"] = round(float(
    (V.groupby("subject_id").size().reindex(pt.index).eq(1) & pt.ne(0)).sum() /
    V.groupby("subject_id").size().eq(1).sum()), 4)
print(json.dumps(dict(sorted(res.items(), key=lambda kv: -kv[1])), indent=1))
(OUT / "trend_formula_search_v8.json").write_text(json.dumps(res, indent=1))

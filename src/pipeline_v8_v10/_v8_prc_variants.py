# -*- coding: utf-8 -*-
"""prior_readmission_count: test label-based constructions (causal and leaky)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
PUB = Path(__file__).resolve().parent.parent
OUT = PUB / "medicare-30day-readmission-mimic-iv" / "results_v7"
V = pd.read_parquet(PUB / "training_table_v10.parquet",
                    columns=["subject_id", "hadm_id", "prior_readmission_count", "readmit_30d"])
a = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                    columns=["hadm_id", "admittime", "dischtime"])
V = V.merge(a, on="hadm_id").sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
g = V.groupby("subject_id")["readmit_30d"]
V["cum_excl"] = g.cumsum() - V.readmit_30d            # labels of earlier cohort admissions
V["cum_incl"] = g.cumsum()                            # includes the index label (leak)
V["tot"] = g.transform("sum")                         # whole record (leak)
V["cum_excl_lag2"] = V.groupby("subject_id")["cum_excl"].shift(1).fillna(0)  # drops most recent prior
# earlier admissions discharged before the index admission
res = {}
for c in ("cum_excl", "cum_incl", "tot", "cum_excl_lag2"):
    res[c] = round(float((V[c] == V.prior_readmission_count).mean()), 4)
d = V.prior_readmission_count - V.cum_excl
res["diff_dist_vs_cum_excl"] = d.value_counts().head(6).to_dict()
# does the residual relate to the index outcome? (leak signature)
res["mean_residual_if_index_readmit"] = round(float(d[V.readmit_30d == 1].mean()), 4)
res["mean_residual_if_not"] = round(float(d[V.readmit_30d == 0].mean()), 4)
V["first"] = V.groupby("subject_id").cumcount() == 0
res["first_cohort_adm_nonzero_share"] = round(float((V.loc[V["first"], "prior_readmission_count"] != 0).mean()), 4)
res["first_cohort_adm_readmit_if_nonzero"] = round(float(V.loc[V["first"] & (V.prior_readmission_count != 0), "readmit_30d"].mean()), 4) if (V["first"] & (V.prior_readmission_count != 0)).any() else None
res["first_cohort_adm_readmit_if_zero"] = round(float(V.loc[V["first"] & (V.prior_readmission_count == 0), "readmit_30d"].mean()), 4)
print(json.dumps(res, indent=1, default=str))
(OUT / "prior_readmission_variants_v8.json").write_text(json.dumps(res, indent=1, default=str))

# lookback-window variants: earlier cohort admissions with a 30-day readmission
# label, admitted (or discharged) within W days before the index admission
at = V.admittime.values.astype("datetime64[s]").astype(np.int64) / 86400
dtt = V.dischtime.values.astype("datetime64[s]").astype(np.int64) / 86400
lab = V.readmit_30d.values; sid = V.subject_id.values
out = {f"admit_{w}": np.zeros(len(V)) for w in (90, 180, 365, 730)}
out.update({f"disch_{w}": np.zeros(len(V)) for w in (180, 365)})
start = 0
for i in range(1, len(V) + 1):
    if i == len(V) or sid[i] != sid[start]:
        for k in range(start, i):
            j = np.arange(start, k)
            for w in (90, 180, 365, 730):
                out[f"admit_{w}"][k] = lab[j][at[j] >= at[k] - w].sum()
            for w in (180, 365):
                out[f"disch_{w}"][k] = lab[j][dtt[j] >= at[k] - w].sum()
        start = i
win = {k: round(float((v == V.prior_readmission_count.values).mean()), 4) for k, v in out.items()}
print(json.dumps(win, indent=1))
res["lookback_variants"] = win
(OUT / "prior_readmission_variants_v8.json").write_text(json.dumps(res, indent=1, default=str))

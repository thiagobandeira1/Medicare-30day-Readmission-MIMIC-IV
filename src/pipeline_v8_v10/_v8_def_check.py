# -*- coding: utf-8 -*-
"""Check orders_last_6h = late_order_rate * n_late_orders and n_labs_total =
count of numeric lab results in the hadm (sample)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
PUB = Path(__file__).resolve().parent.parent
OUT = PUB / "medicare-30day-readmission-mimic-iv" / "results_v7"
RAW = PUB / "Dataset" / "mimic-parquet"
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
DF = pd.concat([V10.reset_index(drop=True), V7[[c for c in V7.columns if c not in V10.columns]].reset_index(drop=True)], axis=1)
r = {}
prod = DF.late_order_rate.fillna(0) * DF.n_late_orders.fillna(0)
r["orders_last_6h_eq_rate_x_late_orders"] = round(float(np.isclose(DF.orders_last_6h.fillna(0), prod, atol=1e-4).mean()), 4)
samp = np.random.default_rng(7).choice(DF.hadm_id.to_numpy(), 3000, replace=False)
lab = ds.dataset(RAW / "labevents.parquet").to_table(columns=["hadm_id", "valuenum", "flag"],
      filter=ds.field("hadm_id").isin(samp.tolist())).to_pandas()
S = DF.loc[DF.hadm_id.isin(samp), ["hadm_id", "n_labs_total", "lab_abnormal_rate"]].copy()
cnt = lab.groupby("hadm_id").valuenum.count()
S["c"] = S.hadm_id.map(cnt).fillna(0)
nab = lab[lab.flag.astype(str).str.lower() == "abnormal"].groupby("hadm_id").size()
S["abn"] = S.hadm_id.map(nab).fillna(0) / S.n_labs_total.fillna(1).clip(lower=1)
r["n_labs_total_eq_numeric_results_in_hadm"] = round(float(np.isclose(S.n_labs_total.fillna(0), S.c).mean()), 4)
r["lab_abnormal_rate_eq_abnormal_rows_over_numeric_results"] = round(float(np.isclose(S.lab_abnormal_rate.fillna(0), S.abn, atol=1e-3).mean()), 4)
print(json.dumps(r, indent=1))
(OUT / "definition_checks_v8.json").write_text(json.dumps(r, indent=1))

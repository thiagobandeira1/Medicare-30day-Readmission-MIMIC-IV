# -*- coding: utf-8 -*-
"""v16 audit (2026-11-11): check that the recovered capstone code for the leaked
feature reproduces the stored los_trend_180d values.

The capstone notebooks ("Capstone Notebook v2.ipynb" and "Capstone Notebook v3.2
01-26-2026.ipynb", cell 16, block "Service intensity trend") built the feature
in four steps:
  1. sort admissions by patient and admission time;
  2. take each admission's length of stay minus the previous admission's;
  3. average those differences per patient over the whole record;
  4. attach that mean to every admission of the patient (0 for patients with
     a single admission). No 180-day window is applied.

This script re-implements those steps on the source feature table and compares
them with the stored column. It reads row-level data locally and writes only
aggregate shares to results_v13/trend_code_recovery_v16.json.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
SRC = PUB / "training_table_v10.parquet"
OUT = REPO / "results_v13" / "trend_code_recovery_v16.json"

df = pd.read_parquet(SRC, columns=["subject_id", "hadm_id", "admittime_dt",
                                   "los_days", "los_trend_180d"])
d = df.sort_values(["subject_id", "admittime_dt"], kind="mergesort").copy()
d["los_delta"] = d["los_days"] - d.groupby("subject_id")["los_days"].shift(1)
recon = d.groupby("subject_id")["los_delta"].mean().rename("recon")
d = d.merge(recon, left_on="subject_id", right_index=True, how="left")
d["recon"] = d["recon"].fillna(0.0)
d["exact"] = (d["recon"] - d["los_trend_180d"]).abs() < 1e-6

per_patient = d.groupby("subject_id")["exact"].all()
n_adm = d.groupby("subject_id").size()
multi = n_adm[n_adm > 1].index

res = {
    "description": ("Recovered capstone formula for the leaked feature "
                    "(per-patient mean of consecutive length-of-stay "
                    "differences over the whole record, attached to every "
                    "admission; 0 for single-admission patients) compared "
                    "with the stored los_trend_180d column of the source "
                    "feature table."),
    "source_notebooks": ["Capstone Notebook v2.ipynb",
                         "Capstone Notebook v3.2 01-26-2026.ipynb"],
    "source_cell": "cell 16, block 'Service intensity trend'",
    "tolerance_abs": 1e-6,
    "n_admissions": int(len(d)),
    "admissions_exact_share": round(float(d["exact"].mean()), 6),
    "n_patients": int(per_patient.size),
    "patients_all_exact_share": round(float(per_patient.mean()), 6),
    "n_multi_admission_patients": int(len(multi)),
    "multi_admission_patients_all_exact_share":
        round(float(per_patient.loc[multi].mean()), 6),
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("description",)}, indent=1))

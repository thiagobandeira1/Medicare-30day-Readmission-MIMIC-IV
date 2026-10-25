# -*- coding: utf-8 -*-
"""Cohort characteristics by partition (TRIPOD+AI item 20b). Aggregate only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v11"; OUT.mkdir(exist_ok=True)
C = pd.read_parquet(REPO / "results_reanalysis" / "cohort_v2.parquet")
V = pd.read_parquet(PUB / "training_table_v10.parquet",
                    columns=["hadm_id", "age_at_admit", "gender", "los_days",
                             "prior_admissions_all", "prior_admissions_6m", "icu_flag",
                             "admission_type", "n_labs_total"])
A = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                    columns=["hadm_id", "race", "anchor_year_group"] if False else ["hadm_id", "race"])
D = C[["hadm_id", "subject_id", "partition", "excl_index_death", "label_v2"]].merge(V, on="hadm_id").merge(A, on="hadm_id")
D = D[~D.excl_index_death]
r = D.race.astype(str)
D["race_g"] = np.select([r.str.contains("WHITE"), r.str.contains("BLACK"),
                         r.str.contains("HISPANIC|LATINO"), r.str.contains("ASIAN")],
                        ["White", "Black", "Hispanic or Latino", "Asian"], "Other or unknown")
D["partition"] = D.partition.map({"train": "Training", "val": "Validation", "test": "Test"})


def col(g):
    n = len(g)
    med = lambda s: f"{s.median():.0f} ({s.quantile(.25):.0f} to {s.quantile(.75):.0f})"
    med1 = lambda s: f"{s.median():.1f} ({s.quantile(.25):.1f} to {s.quantile(.75):.1f})"
    pct = lambda m: f"{m.sum():,} ({m.mean()*100:.1f}%)"
    return {
        "Admissions": f"{n:,}",
        "Patients": f"{g.subject_id.nunique():,}",
        "30-day readmission (events)": pct(g.label_v2 == 1),
        "Age, years, median (IQR)": med(g.age_at_admit),
        "Age under 65": pct(g.age_at_admit < 65),
        "Female": pct(g.gender.astype(str).str.upper() == "F"),
        "White": pct(g.race_g == "White"),
        "Black": pct(g.race_g == "Black"),
        "Hispanic or Latino": pct(g.race_g == "Hispanic or Latino"),
        "Asian": pct(g.race_g == "Asian"),
        "Other or unknown race": pct(g.race_g == "Other or unknown"),
        "Elective admission": pct(g.admission_type.astype(str).str.contains("ELECTIVE")),
        "Any ICU stay": pct(g.icu_flag == 1),
        "Length of stay, days, median (IQR)": med1(g.los_days),
        "Prior admissions, median (IQR)": med(g.prior_admissions_all),
        "No prior admission": pct(g.prior_admissions_all == 0),
        "Admission in prior 180 days": pct(g.prior_admissions_6m > 0),
        "No laboratory results recorded": pct(g.n_labs_total == 0),
    }


T = {p: col(D[D.partition == p]) for p in ("Training", "Validation", "Test")}
T["All"] = col(D)
(OUT / "cohort_characteristics_v11.json").write_text(json.dumps(T, indent=1))
print(pd.DataFrame(T).to_string())

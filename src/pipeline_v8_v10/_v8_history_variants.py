# -*- coding: utf-8 -*-
"""Identify the exact construction of history features that did not reproduce
under the first-pass definitions, by testing causal and future-using variants
against the stored values. Aggregate output: results_v7/history_variants_v8.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PUB = Path(__file__).resolve().parent.parent
OUT = PUB / "medicare-30day-readmission-mimic-iv" / "results_v7"
a = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                    columns=["subject_id", "hadm_id", "admittime", "dischtime", "insurance"])
a = a.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
a["los"] = (a.dischtime - a.admittime).dt.total_seconds() / 86400
V10 = pd.read_parquet(PUB / "training_table_v10.parquet",
                      columns=["hadm_id", "prior_admissions_6m", "prior_mean_los_6m",
                               "prior_readmission_count", "time_since_last_discharge",
                               "readmit_30d"])
cohort = set(V10.hadm_id)
D = 86400.0
rows = []
for sid, g in a.groupby("subject_id", sort=False):
    at = g.admittime.to_numpy().astype("datetime64[s]").astype(np.int64) / D
    dt = g.dischtime.to_numpy().astype("datetime64[s]").astype(np.int64) / D
    los = g.los.to_numpy(); losd = np.floor(los); h = g.hadm_id.to_numpy()
    inc = np.array([x in cohort for x in h])
    # gap from each admission's discharge to the next admission
    nxt = np.append(at[1:] - dt[:-1], np.nan)
    readm_next = (nxt > 0) & (nxt <= 30)                 # admission j followed by readmission
    is_readm = np.append(False, readm_next[:-1])          # admission j is itself a readmission
    for k in range(len(g)):
        if not inc[k]:
            continue
        pr = at < at[k]
        p6a = pr & (at >= at[k] - 180)
        p6b = pr & (at >= at[k] - 182.5)
        p6c = pr & (dt >= at[k] - 180)
        p6d = (np.arange(len(g)) < k) & (at >= at[k] - 180)
        r = {"hadm_id": h[k],
             # prior_admissions_6m variants
             "p6_admit180": p6a.sum(), "p6_admit182": p6b.sum(),
             "p6_disch180": p6c.sum(), "p6_order180": p6d.sum(),
             "p6_cohort180": (p6a & inc).sum(),
             # prior_mean_los_6m variants (causal)
             "mlos_floor_admit180": losd[p6a].mean() if p6a.any() else 0.0,
             "mlos_floor_disch180": losd[p6c].mean() if p6c.any() else 0.0,
             "mlos_floor_cohort": losd[p6a & inc].mean() if (p6a & inc).any() else 0.0,
             "mlos_round_admit180": np.round(los[p6a]).mean() if p6a.any() else 0.0,
             # prior_readmission_count variants
             "pr_is_readm_prior": is_readm[pr].sum(),
             "pr_is_readm_incl_index": is_readm[pr].sum() + is_readm[k],
             "pr_followed_prior": readm_next[pr].sum(),            # causal: next of j<k is <=k
             "pr_followed_cohort_prior": (readm_next & inc)[pr].sum(),
             # future-using variants (leaky by construction)
             "LEAK_pr_followed_incl_index": readm_next[pr].sum() + readm_next[k],
             "LEAK_pr_patient_total": readm_next.sum(),
             "LEAK_mlos_window_pm180": losd[(at >= at[k] - 180) & (at <= at[k] + 180) & (np.arange(len(g)) != k)].mean()
             if ((at >= at[k] - 180) & (at <= at[k] + 180) & (np.arange(len(g)) != k)).any() else 0.0,
             # time since last discharge variants
             "tsld_all": at[k] - dt[pr].max() if pr.any() else np.nan,
             "tsld_cohort": at[k] - dt[pr & inc].max() if (pr & inc).any() else np.nan}
        rows.append(r)
X = pd.DataFrame(rows).merge(V10, on="hadm_id")


def m(stored, cand, tol=1e-6):
    s = pd.to_numeric(X[stored], errors="coerce"); c = pd.to_numeric(X[cand], errors="coerce")
    ok = (s.isna() & c.isna()) | np.isclose(s.fillna(-9e9), c.fillna(-9e9), atol=tol)
    return round(float(ok.mean()), 4)


R = {"n_cohort_admissions": len(X)}
R["prior_admissions_6m"] = {c: m("prior_admissions_6m", c) for c in
                            ("p6_admit180", "p6_admit182", "p6_disch180", "p6_order180", "p6_cohort180")}
R["prior_mean_los_6m"] = {c: m("prior_mean_los_6m", c, 0.01) for c in
                          ("mlos_floor_admit180", "mlos_floor_disch180", "mlos_floor_cohort",
                           "mlos_round_admit180", "LEAK_mlos_window_pm180")}
R["prior_readmission_count"] = {c: m("prior_readmission_count", c) for c in
                                ("pr_is_readm_prior", "pr_is_readm_incl_index", "pr_followed_prior",
                                 "pr_followed_cohort_prior", "LEAK_pr_followed_incl_index",
                                 "LEAK_pr_patient_total")}
fe_fill = np.isclose(X.time_since_last_discharge, 80.22, atol=0.01)
R["time_since_last_discharge"] = {
    "match_all_admissions_nonfill": m("time_since_last_discharge", "tsld_all", 0.01),
    "share_fill_value": round(float(fe_fill.mean()), 4),
    "fill_when_no_prior_any": round(float((fe_fill == X.tsld_all.isna()).mean()), 4),
    "fill_when_no_prior_cohort": round(float((fe_fill == X.tsld_cohort.isna()).mean()), 4),
    "nonfill_match_cohort": round(float(np.isclose(X.loc[~fe_fill, "time_since_last_discharge"],
                                                   X.loc[~fe_fill, "tsld_cohort"], atol=0.01).mean()), 4),
    "nonfill_match_all": round(float(np.isclose(X.loc[~fe_fill, "time_since_last_discharge"],
                                                X.loc[~fe_fill, "tsld_all"], atol=0.01).mean()), 4)}
(OUT / "history_variants_v8.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))

# -*- coding: utf-8 -*-
"""Add two review-round sensitivity blocks to the canonical results file
(final_model_v6.json -> v6_extras): era-stratified test AUROC by
anchor_year_group, and the payer scope of readmission ascertainment.
Both are derived from stored predictions and raw MIMIC tables; no model
is retrained. Idempotent."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
FM_PATH = REPO / "results_v6" / "final_model_v6.json"
fm = json.loads(FM_PATH.read_text())
ex = fm["v6_extras"]

if "era_stratified_test" not in ex:
    P5 = np.load(REPO / "results_v5" / "final_preds_v5.npz")
    pat = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "patients.parquet",
                          columns=["subject_id", "anchor_year_group"])
    m = pd.DataFrame({"subject_id": P5["s_te"], "p": P5["p_te"],
                      "y": P5["y_te"]}).merge(pat, on="subject_id", how="left")
    assert m["anchor_year_group"].notna().all()
    bands = []
    for g, gdf in sorted(m.groupby("anchor_year_group"), key=lambda kv: kv[0]):
        bands.append({
            "band": g,
            "n": int(len(gdf)),
            "events": int(gdf["y"].sum()),
            "prevalence": round(float(gdf["y"].mean()), 4),
            "auroc": round(float(roc_auc_score(gdf["y"], gdf["p"])), 4),
        })
    ex["era_stratified_test"] = {
        "definition": "final-model test predictions stratified by the "
                      "patient-level MIMIC-IV anchor_year_group band; "
                      "descriptive, no CIs computed",
        "bands": bands,
    }
    print("era bands:", [(b["band"], b["auroc"]) for b in bands])

if "payer_scope" not in ex:
    ADM = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                          columns=["subject_id", "hadm_id", "admittime",
                                   "dischtime", "insurance"])
    EV = pd.read_parquet(REPO / "results_reanalysis" / "cohort_v3_events.parquet")
    adm = ADM.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    nxt = adm.groupby("subject_id").shift(-1)
    adm["delta_any"] = (nxt["admittime"] - adm["dischtime"]).dt.total_seconds() / 86400.0
    adm["next_ins"] = nxt["insurance"]
    med = adm[adm["insurance"] == "Medicare"].set_index("hadm_id")
    j = EV.set_index("hadm_id").join(med[["delta_any", "next_ins"]], how="inner")
    nonmed = j[(j["delta_any"] > 0) & (j["delta_any"] <= 30)
               & (j["next_ins"] != "Medicare")]
    ex["payer_scope"] = {
        "definition": "the outcome search covers ALL subsequent hospitalizations "
                      "of the patient regardless of the insurance recorded on "
                      "the subsequent admission (verified empirically against "
                      "raw admissions)",
        "n_30d_next_nonmedicare": int(len(nonmed)),
        "n_counted_as_readmission": int((nonmed["event_v3"] == 1).sum()),
    }
    print("payer_scope:", ex["payer_scope"]["n_30d_next_nonmedicare"],
          ex["payer_scope"]["n_counted_as_readmission"])

fm["v6_extras"] = ex
FM_PATH.write_text(json.dumps(fm, indent=1))
print("final_model_v6.json extended")

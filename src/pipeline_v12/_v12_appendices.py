# -*- coding: utf-8 -*-
"""Build Supplementary Appendices 1 to 4 and 6 for the v7 JAMIA submission
(leak-free rebuild) into 'JAMIA Submission Packet v7'. Appendix 5 (TRIPOD+AI)
is copied from the v6 packet and checked for stale section names."""
import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
O7 = REPO / "results_v12"
O7OLD = REPO / "results_v7"
R2 = REPO / "results_reanalysis"
FIG7 = REPO / "figures_v7"
PKT = PUB / "JAMIA Submission Packet v12"
PKT.mkdir(exist_ok=True)
FM = json.loads((O7 / "final_model_v12.json").read_text())
EX, RV = FM["extras"], FM["review"]
CMP = json.loads((O7 / "model_comparison.json").read_text())
POOL = json.loads((O7 / "eligible_pool_v12.json").read_text())
TRAJ = json.loads((O7 / "rfe_trajectories.json").read_text())
FOLDS = json.loads((O7 / "fold_selected_features.json").read_text())
SD = json.loads((O7 / "stage_differential_v12.json").read_text())
BD = json.loads((O7 / "boundary_v12.json").read_text())
LK = json.loads((O7 / "leak_test_v12.json").read_text())
BL = json.loads((R2 / "baselines_v2.json").read_text())
FD = json.loads((R2 / "fairness_dca_v2.json").read_text())
OLD = json.loads((REPO / "results_v6" / "final_model_v6.json").read_text())
NF = FM["n_features"]
TA = json.loads((O7OLD / "timestamp_audit_v8.json").read_text())
HV = json.loads((O7OLD / "history_variants_v8.json").read_text())
PRV = json.loads((O7OLD / "prior_readmission_variants_v8.json").read_text())
IA = json.loads((O7 / "index_stay_audit_v12.json").read_text())
DCK = json.loads((O7OLD / "definition_checks_v8.json").read_text())
SC = json.loads((O7 / "subgroup_calibration_v12.json").read_text())
PPS = json.loads((O7 / "pre_post_strata_v12.json").read_text())
EVL = json.loads((O7 / "evaluation_v12.json").read_text())
ER = json.loads((O7 / "episode_end_rule_v12.json").read_text())
TB = json.loads((O7 / "table_rebuild_v12.json").read_text())
BO = json.loads((O7 / "baselines_oof_v12.json").read_text())


def new_doc(title):
    d = Document()
    d.styles["Normal"].font.name = "Times New Roman"
    d.styles["Normal"].font.size = Pt(10)
    t = d.add_paragraph(); r = t.add_run(title); r.bold = True; r.font.size = Pt(13)
    return d


def add_table(d, rows, fs=8.5):
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            r = t.cell(i, j).paragraphs[0].add_run(str(cell))
            r.font.size = Pt(fs); r.font.name = "Times New Roman"
            if i == 0:
                r.bold = True
    return t


# ================================================================ Appendix 1
audit = pd.read_csv(REPO / "results_v4" / "feature_audit_207.csv")
reason = {e["feature"]: e["reason"] for e in POOL["excluded"]}
for f, rs in reason.items():
    m = audit["feature_name"] == f
    audit.loc[m, "reason_for_inclusion_or_exclusion"] = rs
for f in ("los_trend_180d", "los_trend_x_prior_6m", "los_trend_x_prior_admits"):
    m = audit["feature_name"] == f
    audit.loc[m, "available_by_discharge"] = "no (encodes later admissions)"
    audit.loc[m, "postdischarge_or_forward_looking"] = True
VDEF = {
 "prior_admissions_6m": ("Earlier admissions (any insurance) discharged in the 180 days before the index admission", "prior 180 days", HV["prior_admissions_6m"]["p6_disch180"]),
 "prior_mean_los_6m": ("Mean whole-day length of stay of earlier admissions discharged in the prior 180 days (0 if none)", "prior 180 days", HV["prior_mean_los_6m"]["mlos_floor_disch180"]),
 "prior_any_ed_6m": ("Any earlier admission in the prior 180 days with an emergency department registration", "prior 180 days", TA["history_recompute"]["prior_any_ed_6m"]["match_share"]),
 "prior_readmission_count": ("Earlier Medicare-insured admissions in the prior 365 days followed by a readmission within 30 days", "prior 365 days", PRV["lookback_variants"]["admit_365"]),
 "time_since_last_discharge": ("Days from the previous discharge of any insurance to the index admission; missing for first observed admissions (v13; v7 to v10 used the previous Medicare-insured discharge and the global median 80.22)", "all earlier admissions", 1.0),
 "prior_admissions_all": ("Earlier admissions of the patient (any insurance)", "all earlier admissions", TA["history_recompute"]["prior_admissions_all"]["match_share"]),
 "orders_last_6h": ("late_order_rate multiplied by n_late_orders (name does not reflect content)", "index stay", DCK["orders_last_6h_eq_rate_x_late_orders"]),
 "n_labs_total": ("Numeric laboratory results charted and stored on or before discharge (v12 cutoff rebuild)", "index stay, on or before discharge", 1.0),
 "lab_abnormal_rate": ("Abnormal laboratory rows divided by numeric results, both charted and stored on or before discharge (v12)", "index stay, on or before discharge", 1.0),
 "n_lab_item_types": ("Distinct laboratory test types charted and stored on or before discharge (v12)", "index stay, on or before discharge", 1.0),
 "n_meds_total": ("Prescriptions started on or before discharge (v12)", "index stay, on or before discharge", 1.0),
 "n_order_types": ("Distinct provider order types recorded under the hospital admission", "index admission (all rows)", IA["raw_spot_check"]["n_order_types"]["all_rows_of_hadm"]),
 "bmi_last": ("Last outpatient body-mass index dated on or before the admission time", "before admission", TA["omr_audit"]["bmi_last"]["on_or_before_admission_time"]["exact_match_any_value_on_last_date"]),
 "bp_diastolic_outpatient": ("Last outpatient diastolic pressure dated on or before the admission time", "before admission", TA["omr_audit"]["bp_diastolic_outpatient"]["on_or_before_admission_time"]["exact_match_any_value_on_last_date"])}
audit["missingness_overall_source_table"] = audit["missingness_overall"]
MS = json.loads((O7 / "missingness_v12.json").read_text())
for f_, v_ in MS.items():
    k_ = audit["feature_name"] == f_
    audit.loc[k_, "missingness_overall"] = v_["overall"]
    audit.loc[k_, "missingness_by_partition"] = json.dumps(v_["by_partition"])
    audit.loc[k_, "missingness_by_race_group"] = json.dumps(v_["by_race_group"])
audit["verified_definition_v8"] = ""
audit["verified_match_share_v8"] = None
for f, (dfn, win, m) in VDEF.items():
    k = audit["feature_name"] == f
    audit.loc[k, "transformation"] = dfn
    audit.loc[k, "measurement_window"] = win
    audit.loc[k, "verified_definition_v8"] = dfn
    audit.loc[k, "verified_match_share_v8"] = m
k = audit["feature_name"] == "los_trend_180d"
audit.loc[k, "verified_definition_v8"] = ("Per-patient summary over the whole record (constant within "
                                         "every multi-admission patient); removed")
elig = pd.DataFrame({"feature": POOL["eligible"]})
excl = pd.DataFrame(POOL["excluded"])
freq = CMP["_stability"]["selection_frequency"]
shap_map = {s["feature"]: s["mean_abs_shap"] for s in FM["shap"]}
cons = pd.DataFrame({"feature": FM["final_feature_set"]}).merge(
    audit[["feature_name", "clinical_domain", "source_table",
           "measurement_window", "missingness_overall"]],
    left_on="feature", right_on="feature_name", how="left").drop(columns=["feature_name"])
cons["selected_in_n_folds"] = cons["feature"].map(freq)
cons["mean_abs_shap_final_model"] = cons["feature"].map(shap_map)
cons = cons.sort_values("mean_abs_shap_final_model", ascending=False)
names = {"rfe": "RFE procedure (per-fold sets)",
         "all_eligible": f"All eligible ({POOL['n_eligible']})",
         "hist_clean": "Earlier development set, clean subset",
         "f66": "Earlier 66-feature set (billing and leaked variables; historical)",
         "logit": "Logistic regression (eligible)"}
comp_rows = []
for k, nm in names.items():
    e = CMP[k]
    comp_rows.append({"model": nm, "n_features": str(e["n_features"]),
                      "oof_auroc": e["oof_auroc"],
                      "ci_lo": e["auroc_ci95_cluster"][0],
                      "ci_hi": e["auroc_ci95_cluster"][1], "ap": e["ap"],
                      "brier": e["brier"], "ece": e["ece"],
                      "cal_slope": e["cal_slope"],
                      "top10_capture": e["top10pct_events_captured"],
                      "top10_ppv": e["top10pct_ppv"]})
paired_rows = [{"contrast": k, "point": v["point"], "ci_lo": v["ci95"][0],
                "ci_hi": v["ci95"][1], "note": "descriptive; no formal test"}
               for k, v in CMP["_paired_differences_vs_rfe"].items()]
sub_rows = []
for col in ("race", "sex", "age_band"):
    for g, e in FM["fairness"][col].items():
        row = {"axis": col, "group": g}
        for k in ("patients", "admissions", "events", "prevalence", "auroc", "ap",
                  "brier", "ece", "cal_slope", "cal_intercept", "sensitivity",
                  "specificity", "ppv", "npv", "fnr"):
            row[k] = e.get(k)
        for k in ("auroc", "sensitivity", "specificity", "ppv", "npv", "fnr",
                  "brier", "cal_slope"):
            c = e.get(f"{k}_ci95")
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = (c if c else (None, None))
        sub_rows.append(row)
for k, v in RV["prior_utilization"].items():
    sub_rows.append({"axis": "prior_admissions", "group": k,
                     "patients": v["patients"], "admissions": v["admissions"],
                     "prevalence": v["prevalence"], "auroc": v["auroc"],
                     "auroc_ci_lo": v["auroc_ci95"][0],
                     "auroc_ci_hi": v["auroc_ci95"][1]})
T = RV["timing"]
_ep = ER["timing_episode_rule"]
timing_rows = [{"day": d + 1, "auroc_primary": T["primary"]["auc"][d],
                "lo_primary": T["primary"]["lo"][d], "hi_primary": T["primary"]["hi"][d],
                "auroc_episode_rule": _ep["auc"][d],
                "lo_episode_rule": _ep["lo"][d], "hi_episode_rule": _ep["hi"][d]}
               for d in range(len(T["primary"]["auc"]))]
cb = RV["continuation_band"]
cb_rows = [{"interval": k, **v} for k, v in cb["rate_per_hour"].items()]
cmp_rows = [{"comparator": k, **{kk: vv for kk, vv in v.items() if kk != "uplift_ci95"},
             "uplift_ci_lo": v["uplift_ci95"][0], "uplift_ci_hi": v["uplift_ci95"][1]}
            for k, v in RV["comparators"].items() if k != "reconstruction_match"]
xlsx = PKT / "Supplementary Appendix 1 Predictor Audit and Selection.xlsx"
with pd.ExcelWriter(xlsx) as w:
    audit.to_excel(w, sheet_name="Audit_207", index=False)
    elig.to_excel(w, sheet_name=f"Eligible_{POOL['n_eligible']}", index=False)
    excl.to_excel(w, sheet_name=f"Excluded_{len(POOL['excluded'])}", index=False)
    cons.to_excel(w, sheet_name=f"Final_Consensus_{NF}", index=False)
    pd.DataFrame(LK["scan_top10"]).to_excel(w, sheet_name="Leak_Test_Scan", index=False)
    pd.DataFrame([LK["los_trend_180d"]]).to_excel(w, sheet_name="Leak_Test_Trend", index=False)
    pd.DataFrame([{"pipeline_choice": k, **v} for k, v in
                  RV["leakage_ablation"].items()]).to_excel(
        w, sheet_name="Leakage_Ablation", index=False)
    pd.DataFrame([{"outer_fold": k, "n_features": t["n"], "mean_inner_auroc":
                   t["mean_auc"], "sd": t["sd"]} for k, tr in TRAJ.items()
                  for t in tr]).to_excel(w, sheet_name="RFE_Trajectories", index=False)
    pd.DataFrame([{"outer_fold": k, "feature": f} for k, fs in FOLDS.items()
                  for f in fs]).to_excel(w, sheet_name="Selected_Per_Fold", index=False)
    pd.DataFrame(comp_rows).to_excel(w, sheet_name="Model_Comparison_OOF", index=False)
    pd.DataFrame(paired_rows).to_excel(w, sheet_name="Paired_Differences", index=False)
    pd.DataFrame([{"metric": k, "point": v["point"], "ci_lo": v["ci95"][0],
                   "ci_hi": v["ci95"][1]} for k, v in FM["metric_panel"].items()
                  if isinstance(v, dict)]).to_excel(w, sheet_name="Test_Metric_Panel",
                                                    index=False)
    pd.DataFrame(sub_rows).to_excel(w, sheet_name="Subgroup_Results", index=False)
    pd.DataFrame([RV["race_by_utilization"]["within_stratum_gap"]]).to_excel(
        w, sheet_name="Race_Within_Strata", index=False)
    pd.DataFrame(cmp_rows).to_excel(w, sheet_name="Comparators_Refit", index=False)
    pd.DataFrame(timing_rows).to_excel(w, sheet_name="Daily_AUROC_Bands", index=False)
    pd.DataFrame(FM["daily_event_counts_test"]).to_excel(
        w, sheet_name="Daily_Event_Counts_Test", index=False)
    pd.DataFrame(cb_rows).to_excel(w, sheet_name="Continuation_Band", index=False)
    pd.DataFrame(SD).to_excel(w, sheet_name="Stage_Differential", index=False)
    pd.DataFrame([{"analysis": k, **v} for k, v in FM["sensitivity"].items()] +
                 [{"analysis": "one_hour_episode_rule",
                   "definition": ER["rule"],
                   "test_events": ER["test_events_episode_rule"],
                   "test_prevalence": ER["test_prevalence_episode_rule"],
                   "test_auroc_refit": ER["test_auroc_refit"],
                   "test_auroc_final_model": ER["test_auroc_final_model"],
                   "test_auroc_final_model_ci95": str(ER["test_auroc_final_model_ci95"]),
                   "test_auroc_refit_ci95": str(ER["test_auroc_refit_ci95"])}]).to_excel(
        w, sheet_name="Outcome_Sensitivity", index=False)
    pd.DataFrame([{"step": k, "n": v} for k, v in ER["flow"].items()]).to_excel(
        w, sheet_name="Episode_Flow", index=False)
    pd.DataFrame([{"feature": k, "population": "all 244,576 Medicare admissions (before death exclusion)",
                   "version": "v11", **v} for k, v in TB.items() if isinstance(v, dict) and "new_missing" in v]
                 ).to_excel(w, sheet_name="Table_Rebuild", index=False)
    pd.DataFrame([{"feature": k, "version": "v12 (discharge cutoff on counts)", **v}
                  for k, v in TB.get("v12_cutoff_enforced", {}).items()] +
                 [{"feature": k, "version": "v13 (documentation-time cutoff)", **v}
                  for k, v in TB.get("v13_documentation_time_enforced", {}).items()]
                 ).to_excel(w, sheet_name="Cutoff_Rebuild", index=False)
    pd.DataFrame([{"metric": k, "value": (str(v) if isinstance(v, (dict, list)) else v)}
                  for k, v in BO.items()]).to_excel(w, sheet_name="Baselines_OOF", index=False)
    pd.DataFrame(RV["era"]).to_excel(w, sheet_name="Era_Bands", index=False)
    pd.DataFrame([{"contrast": k, "observed_point": (FM["fairness"]["white_minus_black"]["point"]
                   if k == "white_minus_black" else FM["baselines"]["uplift_lace" if k == "lace_uplift" else "uplift_hospital12"]),
                   "bootstrap_mean": v["point"], **{kk: vv for kk, vv in v.items() if kk not in ("ci95", "point")},
                   "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]}
                  for k, v in EX["highdraw_contrasts_5000"].items()]).to_excel(
        w, sheet_name="HighDraw_Contrasts_5000", index=False)
    pd.DataFrame([{"rule": k, **v} for k, v in EX["tie_rule_sensitivity"].items()
                  if isinstance(v, dict)]).to_excel(w, sheet_name="Tie_Rule", index=False)
    pd.DataFrame([{"feature": k, "share_constant_within_patient": v} for k, v in
                  IA["constant_within_patient_share"].items()]).to_excel(
        w, sheet_name="Constancy_Scan", index=False)
    pd.DataFrame([{"feature": f, "verified_definition": d_, "window": w_, "match_share": m_}
                  for f, (d_, w_, m_) in VDEF.items()]).to_excel(
        w, sheet_name="Timestamp_Audit", index=False)
    pd.DataFrame([{"definition_tested": k, "match_share": v} for k, v in
                  {**{f"prior_admissions_6m:{a}": b for a, b in HV["prior_admissions_6m"].items()},
                   **{f"prior_mean_los_6m:{a}": b for a, b in HV["prior_mean_los_6m"].items()},
                   **{f"prior_readmission_count:{a}": b for a, b in HV["prior_readmission_count"].items()},
                   **{f"prior_readmission_count:{a}": b for a, b in PRV["lookback_variants"].items()}}.items()
                  ]).to_excel(w, sheet_name="Definition_Variants", index=False)
    pd.DataFrame([{"group": k, **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}}
                  for k, v in SC.items() if ":" in k]).to_excel(
        w, sheet_name="Subgroup_Calibration", index=False)
    pd.DataFrame([{"prior_admissions": k, **v} for k, v in PPS.items()]).to_excel(
        w, sheet_name="Pre_Post_By_Stratum", index=False)
    pd.DataFrame([{"setting": k, "val_auroc": v} for k, v in EVL["hyperparam_val_auroc"].items()]
                 ).to_excel(w, sheet_name="Robustness_Hyperparams", index=False)
print("A1 written")

# ================================================================ Appendix 2
G = RV["comparators"]
d = new_doc("Supplementary Appendix 2. LACE and HOSPITAL Reconstruction")
d.add_paragraph(
    "Both scores were reconstructed on the test partition. They were derived "
    "for different outcomes (LACE: death or unplanned readmission; HOSPITAL: "
    "potentially avoidable readmission). Two components use coded data from "
    "the index stay (LACE comorbidity from ICD diagnoses; HOSPITAL procedure "
    "item from ICD procedures), information the study model was not allowed "
    "to use.")
RT = {"Adapted LACE": [
    ("L (length of stay)", "los_days: 0, 1 to 3, 4, 5, 7 points by the van Walraven bands"),
    ("A (acuity)", "3 points unless admission type is ELECTIVE or SURGICAL SAME DAY ADMISSION"),
    ("C (comorbidity)", "Charlson index from index ICD diagnoses, capped: below 4 gives min(score, 3); 4 or more gives 5"),
    ("E (ED visits)", "prior 180-day admissions with an emergency department registration (edregtime), capped at 4; ED visits without admission are not available (adapted)"),
    ("Outcome note", "original LACE outcome is death or unplanned readmission within 30 days; applied here to all-cause within-system readmission")],
  "HOSPITAL": [
    ("H (hemoglobin below 12 g/dL)", "last hemoglobin stored before discharge below 12 (missing counts as no)"),
    ("O (oncology service)", "any service record OMED during the stay, 2 points"),
    ("S (sodium below 135 mmol/L)", "last sodium stored before discharge below 135 (missing counts as no)"),
    ("P (procedure)", "any ICD-coded procedure during the index stay"),
    ("I (index type)", "nonelective admission"),
    ("T (admissions in prior year)", "admissions of any insurance in the 365 days before the index admission: 0 to 1 gives 0, 2 to 5 gives 2, more than 5 gives 5 (original 12-month window)"),
    ("A (length of stay 5 days or more)", "los_days of 5 or more, 2 points"),
    ("Outcome note", "original HOSPITAL outcome is potentially avoidable readmission; applied here to all-cause within-system readmission")]}
for score, rows_ in RT.items():
    p = d.add_paragraph(); r = p.add_run(score); r.bold = True
    add_table(d, [["Component", "MIMIC-IV mapping"]] + [list(x) for x in rows_])
d.add_paragraph(
    f"Recomputed scores matched the stored scores for "
    f"{G['reconstruction_match']['lace']*100:.1f}% (LACE) and "
    f"{G['reconstruction_match']['hospital']*100:.1f}% (HOSPITAL) of test "
    f"admissions. On the development partitions, out of fold, adapted LACE "
    f"scored {BO['oof_auroc_lace']:.4f} and HOSPITAL {BO['oof_auroc_hospital12']:.4f} "
    f"against {BO['oof_auroc_rfe']:.4f} for the selection procedure.")
add_table(d, [["Comparator", "AUROC", "Final-model uplift (95% CI)"]] +
          [[k.replace("_", " "), f"{v['auroc']:.4f}",
            f"{v['uplift']:.4f} ({v['uplift_ci95'][0]:.4f} to {v['uplift_ci95'][1]:.4f})"]
           for k, v in G.items() if k != "reconstruction_match"])
d.save(str(PKT / "Supplementary Appendix 2 LACE HOSPITAL Reconstruction.docx"))
print("A2 written")

# ================================================================ Appendix 3
d = new_doc("Supplementary Appendix 3. Race and Ethnicity Consolidation")
d.add_paragraph(
    "MIMIC-IV race and ethnicity strings were consolidated by substring "
    "mapping: WHITE to White; BLACK to Black; HISPANIC or LATINO to Hispanic "
    "or Latino; ASIAN to Asian; all remaining values to Other or unknown. "
    "Raw category counts on the test partition (category strings shown "
    "verbatim as stored in MIMIC-IV):")
add_table(d, [["Raw MIMIC category", "Test admissions"]] +
          [[k, f"{v:,}"] for k, v in sorted(FD["raw_race_categories_test"].items(),
                                            key=lambda kv: -kv[1])], fs=8)
d.save(str(PKT / "Supplementary Appendix 3 Race Consolidation.docx"))
print("A3 written")

# ================================================================ Appendix 4
d = new_doc("Supplementary Appendix 4. Decision Curves and Development History")
p = d.add_paragraph(); r = p.add_run("Decision curves, final model"); r.bold = True
d.add_paragraph().add_run().add_picture(str(FIG7 / "v7_dca.png"), width=Inches(5.5))
p = d.add_paragraph(); r = p.add_run("Development history"); r.bold = True
d.add_paragraph(
    "Earlier versions of this pipeline, including all versions released "
    "before October 2026, contained a 180-day length-of-stay trend feature "
    "(and two interaction terms built from it) later shown to encode the "
    "patient's later admissions. Their performance figures are therefore "
    "inflated and are listed only as history.")
hist = [["Version", "Result", "Status"],
        ["Capstone feature tables V1 to V7 (LightGBM)", "0.7051 to 0.7934",
         "Contained billing features and leaked variables"],
        ["Four boosting families, 10 seeds", "0.7935 to 0.7970",
         "Pre-correction cohort; leaked variables"],
        ["Corrected cohort, 66-feature model", "test 0.7800",
         "Billing features and leaked variables"],
        ["v5/v6 leakage-safe consensus model (31 features)",
         f"out of fold {OLD['validation_hierarchy']['primary_oof_procedure_dev_only']['auroc']:.4f}; "
         f"test {OLD['metric_panel']['auroc']['point']:.4f}",
         "Contained the leaked trend feature; superseded"],
        ["v7 to v10 leak-free consensus model (47 features)",
         "out of fold 0.7058; test 0.7064",
         "Used a source table with global median imputation; superseded"],
        [f"v11 and v12 consensus model ({NF} features) on the rebuilt table; v12 enforces the discharge cutoff on laboratory and medication counts and drops two duplicate pool features",
         f"out of fold {FM['validation_hierarchy']['primary_oof_procedure_dev_only']['auroc']:.4f}; "
         f"test {FM['metric_panel']['auroc']['point']:.4f}", "Reported model"],
        ["Race-encoding ablation (earlier model, validation partition)",
         "0.7954 with vs 0.7949 without", "Race encoding excluded throughout"]]
add_table(d, hist, fs=8.5)
d.save(str(PKT / "Supplementary Appendix 4 Decision Curves and Development History.docx"))
print("A4 written")

print("A5 and A6 are written by _v11_checklists.py (reused for v12)")

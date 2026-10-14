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
O7 = REPO / "results_v7"
R2 = REPO / "results_reanalysis"
FIG7 = REPO / "figures_v7"
PKT = PUB / "JAMIA Submission Packet v7"
PKT.mkdir(exist_ok=True)
FM = json.loads((O7 / "final_model_v7.json").read_text())
EX, RV = FM["extras"], FM["review"]
CMP = json.loads((O7 / "model_comparison.json").read_text())
POOL = json.loads((O7 / "eligible_pool_v7.json").read_text())
TRAJ = json.loads((O7 / "rfe_trajectories.json").read_text())
FOLDS = json.loads((O7 / "fold_selected_features.json").read_text())
SD = json.loads((O7 / "stage_differential_v7.json").read_text())
BD = json.loads((O7 / "boundary_v7.json").read_text())
LK = json.loads((O7 / "leak_test_v7.json").read_text())
BL = json.loads((R2 / "baselines_v2.json").read_text())
FD = json.loads((R2 / "fairness_dca_v2.json").read_text())
OLD = json.loads((REPO / "results_v6" / "final_model_v6.json").read_text())
NF = FM["n_features"]


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
                "ci_hi": v["ci95"][1], "p": v["p_bootstrap_2sided"]}
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
timing_rows = [{"day": d + 1, "auroc_primary": T["primary"]["auc"][d],
                "lo_primary": T["primary"]["lo"][d], "hi_primary": T["primary"]["hi"][d],
                "auroc_one_hour_rule": T["one_hour_rule"]["auc"][d],
                "lo_one_hour_rule": T["one_hour_rule"]["lo"][d],
                "hi_one_hour_rule": T["one_hour_rule"]["hi"][d]}
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
                 [{"analysis": "one_hour_rule", **RV["one_hour_rule"]}]).to_excel(
        w, sheet_name="Outcome_Sensitivity", index=False)
    pd.DataFrame(RV["era"]).to_excel(w, sheet_name="Era_Bands", index=False)
    pd.DataFrame([{"contrast": k, **{kk: vv for kk, vv in v.items() if kk != "ci95"},
                   "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]}
                  for k, v in EX["highdraw_contrasts_5000"].items()]).to_excel(
        w, sheet_name="HighDraw_Contrasts_5000", index=False)
    pd.DataFrame([{"rule": k, **v} for k, v in EX["tie_rule_sensitivity"].items()
                  if isinstance(v, dict)]).to_excel(w, sheet_name="Tie_Rule", index=False)
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
for score in ("LACE", "HOSPITAL"):
    p = d.add_paragraph(); r = p.add_run(score); r.bold = True
    add_table(d, [["Component", "MIMIC mapping"]] +
              [[k, v] for k, v in BL["reconstruction_table"][score].items()])
d.add_paragraph(
    f"Recomputed scores matched the stored scores for "
    f"{G['reconstruction_match']['lace']*100:.1f}% (LACE) and "
    f"{G['reconstruction_match']['hospital']*100:.1f}% (HOSPITAL) of test "
    f"admissions. HOSPITAL uses the original 12-month window (an earlier "
    f"6-month proxy is superseded).")
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
        [f"v7 leak-free consensus model ({NF} features)",
         f"out of fold {FM['validation_hierarchy']['primary_oof_procedure_dev_only']['auroc']:.4f}; "
         f"test {FM['metric_panel']['auroc']['point']:.4f}", "Reported model"],
        ["Race-encoding ablation (earlier model, validation partition)",
         "0.7954 with vs 0.7949 without", "Race encoding excluded throughout"]]
add_table(d, hist, fs=8.5)
d.save(str(PKT / "Supplementary Appendix 4 Decision Curves and Development History.docx"))
print("A4 written")

# ================================================================ Appendix 6
d = new_doc("Supplementary Appendix 6. PROBAST+AI Self-Assessment")
d.add_paragraph(
    "Self-assessment against PROBAST+AI (Moons et al, BMJ 2025;388:e082505). "
    "L = low concern, S = some concern, H = high concern.")
add_table(d, [
    ["Domain", "Judgment", "Rationale"],
    ["1. Participants", "S", "Single-center retrospective EHR cohort with clear "
     "eligibility (Medicare insurance, alive discharge); Medicare Advantage "
     "status unobservable; single center limits representativeness."],
    ["2. Predictors", "S", "All candidates audited for discharge-time "
     "availability; billing-derived, race-derived, and outcome-derived "
     "predictors excluded; an empirical first-admission test found and removed "
     "a history feature that encoded later admissions. Index-stay features were "
     "checked against documented timestamps only."],
    ["3. Outcome", "S", "Objective timestamp-based outcome with within-system "
     "capture; planned readmissions included; administrative continuations "
     "within 1 hour of discharge examined as a sensitivity outcome."],
    ["4. Analysis", "S", "Patient-grouped folds on development data, fold-local "
     "preprocessing, cluster-bootstrap inference, calibration and utility "
     "assessed; hyperparameters fixed rather than tuned within folds; "
     "validation partition used in selection and thresholding; test partition "
     "historically exposed, and the correction followed earlier evaluations."],
    ["Overall", "S", "Development with internal validation only; external "
     "validation required before any applicability claim."]], fs=9)
d.save(str(PKT / "Supplementary Appendix 6 PROBAST-AI Self-Assessment.docx"))
print("A6 written")

# ================================================================ Appendix 5
NE = POOL["n_eligible"]
M = "Materials and Methods/"
R = "Results/"
rows = [
 ["TRIPOD+AI item", "How addressed", "Location"],
 ["1 Title", "Identifies development and internal validation, population, outcome", "Title"],
 ["2 Abstract", "Structured abstract with TRIPOD+AI elements", "Abstract"],
 ["3a Background", "Health care context and rationale", "Background and Significance"],
 ["3b Objectives", "Development and validation objectives", "Objective"],
 ["4a Source of data", "MIMIC-IV v3.1; retrospective EHR cohort", M + "Study Design and Data Source"],
 ["4b Dates", "Admissions 2008 to 2022 (date-shifted)", M + "Study Design and Data Source"],
 ["5a Setting", "Single academic center, Boston, MA", M + "Study Design and Data Source"],
 ["5b Eligibility", "Medicare insurance; alive discharge; exclusions listed", M + "Cohort and Outcome Definition"],
 ["5c Treatments", "Not applicable (no treatment effects modeled)", "Not applicable"],
 ["6a Outcome", "30-day all-cause within-system readmission; transfers; planned readmissions; 1-hour continuation sensitivity", M + "Cohort and Outcome Definition; Supplementary Methods S1"],
 ["6b Outcome assessment", "Timestamps from admissions table; blinded by construction", M + "Cohort and Outcome Definition"],
 ["7a Predictors", f"207 candidates audited; {NE} eligible; final consensus set of {NF}; audit and definitions in Supplementary Appendix 1 and Supplementary Table S3", M + "Predictor Audit, Leakage Testing, and Feature Selection"],
 ["7b Predictor assessment", "Documented availability at discharge plus an empirical first-admission future-information test; billing-derived, race-derived, and outcome-derived predictors excluded", M + "Predictor Audit, Leakage Testing, and Feature Selection; " + R + "Future-Information Leakage"],
 ["8 Sample size", "All eligible admissions used; events per partition reported; no formal calculation; flat RFE profile (Supplementary Figure S1)", M + "Partitions and Validation Design"],
 ["9 Missing data", "Native handling by the tree learner; zero or fixed values for undefined counts; per-predictor missingness in Supplementary Appendix 1", M + "Predictor Audit, Leakage Testing, and Feature Selection; Supplementary Methods S4"],
 ["10a Model type", "XGBoost classifier; hyperparameters stated", M + "Model and Comparators"],
 ["10b Model building", "Staged RFE inside 5 outer by 3 inner patient-grouped folds on development data; fold-local target encodings; parsimony rule; consensus aggregation; development history in Supplementary Appendix 4", M + "Predictor Audit, Leakage Testing, and Feature Selection"],
 ["10c Ensembles", "Single final model; no ensembling", M + "Model and Comparators"],
 ["10d Performance measures", "AUROC, AP, Brier, ECE, calibration slope and calibration-in-the-large, decision curves, cluster-bootstrap CIs", M + "Statistical Analysis"],
 ["10e Model updating", "Not performed", "Not applicable"],
 ["11 Risk groups", "Validation-derived threshold; capacity (top decile) view", M + "Model and Comparators; " + R + "Clinical Utility"],
 ["12 Development vs validation", "Internal validation only; three-tier evidence hierarchy with out-of-fold procedure performance primary and the historically exposed test partition tertiary", M + "Partitions and Validation Design; Discussion/Limitations"],
 ["13a Participant flow", "Figure 1 cohort flow", R + "Cohort"],
 ["13b Characteristics", "Partition counts; subgroup sizes and events in Table 3", M + "Partitions and Validation Design; " + R + "Subgroup Performance"],
 ["14a Numbers in analyses", "Admissions, patients, events per partition", M + "Partitions and Validation Design"],
 ["14b Unadjusted associations", "Not applicable (prediction study)", "Not applicable"],
 ["15a Final model", "Model artifact archived with the repository release", "Data Availability Statement"],
 ["15b Model use", "Research only; earlier prototype disclosed as not the reported model", "Data Availability Statement; Discussion"],
 ["16 Performance", "Discrimination, calibration, operating metrics with CIs", R + "Discrimination and Calibration"],
 ["17 Subgroup performance (AI)", "Age, sex, race, and prior-admission audit; direct White vs Black test; race within utilization strata", R + "Subgroup Performance; Supplementary Table S2"],
 ["18 Limitations", "Dedicated section incl. within-system outcome, test-reuse history, single center", "Discussion/Limitations"],
 ["19 Interpretation", "Principal findings; clinical hypotheses framed as such", "Discussion"],
 ["20 Implications", "External validation and prospective evaluation required before use", "Conclusion"],
 ["21 Supplementary information", "Supplementary Material and Supplementary Appendices 1 to 6, incl. PROBAST+AI self-assessment", "Supplementary Material"],
 ["22 Funding", "No external funding", "Funding"],
 ["23 Ethics (AI)", "IRB approvals for MIMIC-IV; PhysioNet DUA; generative AI use disclosed", M + "Ethical Considerations; Acknowledgments"],
 ["24 Openness (AI)", "Code and results files public; data via PhysioNet DUA; Zenodo DOI", "Data Availability Statement"],
 ["25 Fairness (AI)", "Subgroup audit; representation discussed; no equity claim", R + "Subgroup Performance; Discussion"],
]
d = new_doc("Supplementary Appendix 5. TRIPOD+AI Checklist")
d.add_paragraph("Item numbering follows Collins et al, BMJ 2024;385:e078378. "
                "Location refers to sections of the manuscript.")
add_table(d, rows, fs=8.5)
d.save(str(PKT / "Supplementary Appendix 5 TRIPOD-AI Checklist.docx"))
print("A5 written")

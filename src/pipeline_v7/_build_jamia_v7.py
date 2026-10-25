# -*- coding: utf-8 -*-
"""Build the JAMIA manuscript, v7 (leak-free rebuild, 2026-10).

Every number is read from results_v7 (final_model_v7.json with "extras" and
"review", model_comparison.json, leak_test_v7.json) and, for the before
correction contrast only, results_v6/final_model_v6.json. Word caps are
asserted (body 3,850 for headroom under 4,000; abstract 250).
"""
import json
from _v7_core import *  # noqa
from _v7_core import _style  # underscore name: explicit
from _v7_labels import label

S = doc.styles
S["Normal"].font.size = Pt(11)
S["Normal"].paragraph_format.line_spacing = 2.0
S["Normal"].paragraph_format.space_after = Pt(0)
for h, sz in (("Heading 1", 13), ("Heading 2", 11.5), ("Heading 3", 11)):
    S[h].font.size = Pt(sz)
for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(doc.sections[0], attr, Inches(1.0))

OLD = json.loads((REPO / "results_v6" / "final_model_v6.json").read_text())
LK = json.loads((OUT7 / "leak_test_v7.json").read_text())
p = CT["partitions"]
wb = FM["fairness"]["white_minus_black"]
FR, AB = FM["fairness"]["race"], FM["fairness"]["age_band"]
st1, st2, st3 = FM["stages"]
THRV = FM["threshold_validation"]
VH = FM["validation_hierarchy"]
OOF = VH["primary_oof_procedure_dev_only"]
CCV = VH["secondary_consensus_cv_dev_only"]
BSE, CAP, MP = FM["baselines"], FM["capacity"], FM["metric_panel"]
N_ELIG = POOL5["n_eligible"]
NF = FM["n_features"]
CB, OH, TM = RV["continuation_band"], RV["one_hour_rule"], RV["timing"]
PU, RU, CMPR, ABL = (RV["prior_utilization"], RV["race_by_utilization"],
                     RV["comparators"], RV["leakage_ablation"])
OLD_OOF = OLD["validation_hierarchy"]["primary_oof_procedure_dev_only"]["auroc"]
OLD_TEST = OLD["metric_panel"]["auroc"]["point"]
OLD_WB = OLD["fairness"]["white_minus_black"]["point"]
LT = LK["los_trend_180d"]
STAB = CMP5["_stability"]
N_UNANIMOUS = sum(1 for v in STAB["selection_frequency"].values() if v == 5)
FCOUNTS = sorted(STAB["fold_counts"].values())
ERA = RV["era"]
FX = EX["fixed_consensus_vs_eligible_cv"]
TIE = EX["tie_rule_sensitivity"]
ABR, ABP = ABL["reference_leakage_safe"], ABL["plus_future_trend_features"]


def pct(x, d=1):
    return f"{x*100:.{d}f}%"


def f4(x):
    return f"{x:.4f}"


def c4(c):
    return f"{c[0]:.4f} to {c[1]:.4f}"


_COUNT = {"abstract": 0, "body": 0}


def PA(t, bold_prefix=None):
    _COUNT["abstract"] += len(((bold_prefix or "") + t).split())
    return P(t, bold_prefix=bold_prefix)


def PB(t, bold_prefix=None):
    _COUNT["body"] += len(((bold_prefix or "") + t).split())
    return P(t, bold_prefix=bold_prefix)


def HB1(t):
    _COUNT["body"] += len(t.split()); H1(t)


def HB2(t):
    _COUNT["body"] += len(t.split()); H2(t)


# ================================================================ TITLE PAGE
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Predicting 30-Day Hospital Readmission at Discharge: A "
              "Leakage-Safe, Calibrated Electronic Health Record Model "
              "in Medicare-Insured Adults")
r.bold = True; r.font.size = Pt(14)
for txt, sz, it, b in (
        ("Thiago Bandeira, MS; Armando Gonzalez, MS; Christian Poellabauer, "
         "PhD; Ananda Mohan Mondal, PhD", 11, False, True),
        ("tbati006@fiu.edu; agonz1689@fiu.edu; cpoellab@fiu.edu; "
         "amondal@cis.fiu.edu", 9.5, True, False),
        ("ORCID: 0009-0006-0204-5298 (TB); 0009-0007-6777-6072 (AG); "
         "0000-0002-0599-7941 (CP); 0000-0002-4005-9942 (AMM)", 8.5, False, False),
        ("All authors: Knight Foundation School of Computing and Information "
         "Sciences, Florida International University, Miami, FL, United "
         "States", 10.5, True, False),
        ("Corresponding author: Thiago Bandeira, MS (Thiago Batista Nunes "
         "Bandeira); [address withheld], United States; Phone: "
         "[phone withheld]; Email: tbati006@fiu.edu", 9.5, False, False)):
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(txt); r.font.size = Pt(sz); r.italic = it; r.bold = b
t = doc.add_paragraph()
r = t.add_run("Keywords: Patient Readmission; Medicare; Machine Learning; "
              "Electronic Health Records; Healthcare Disparities")
r.font.size = Pt(10)
t = doc.add_paragraph()
wc_run = t.add_run("WORDCOUNT_PLACEHOLDER"); wc_run.font.size = Pt(10)
doc.add_page_break()

# ================================================================== ABSTRACT
H1("Abstract")
PA("To develop and internally validate a calibrated model estimating "
   "30-day all-cause within-system readmission at discharge for "
   "Medicare-insured adults from structured electronic health record (EHR) "
   "fields, with empirical leakage testing and timing and subgroup "
   "analyses.", bold_prefix="Objective: ")
PA(f"Retrospective study of the Medicare-insured subset of MIMIC-IV v3.1 "
   f"({CT['cohort_v2_admissions']:,} admissions, {CT['cohort_v2_patients']:,} "
   f"patients; prevalence {CT['label_v2_prevalence']*100:.1f}%). Candidate "
   f"predictors were audited for discharge-time availability and tested "
   f"empirically for future information. Feature selection and "
   f"preprocessing ran inside patient-grouped folds; features selected in "
   f"at least 3 of 5 folds formed an extreme gradient boosting (XGBoost) "
   f"model. Uncertainty used patient-cluster bootstrap resampling.",
   bold_prefix="Materials and Methods: ")
PA(f"A documented prior-utilization trend feature encoded later "
   f"admissions; removing it and two derived terms lowered test AUROC "
   f"from {OLD_TEST:.3f} to {MP['auroc']['point']:.3f}. The leak-free "
   f"selection procedure achieved an out-of-fold area under the receiver "
   f"operating characteristic curve (AUROC) of {f4(OOF['auroc'])} (95% CI "
   f"{c4(OOF['ci95'])}; primary); the fixed {NF}-feature model reached "
   f"{f4(MP['auroc']['point'])} on the historically exposed test partition "
   f"(calibration slope {MP['slope']['point']:.2f}). It outperformed "
   f"reconstructed and refit LACE and HOSPITAL scores (all P<.001). An "
   f"apparent day-1 discrimination peak arose from "
   f"{CB['events_in_band']:,} records beginning within 1 hour of "
   f"discharge. Black and White patients showed similar discrimination "
   f"(difference {wb['point']:.3f}, 95% CI {wb['ci95'][0]:.3f} to "
   f"{wb['ci95'][1]:.3f}).", bold_prefix="Results: ")
PA("A documentation-based availability audit missed a future-information "
   "leak that inflated discrimination and produced spurious subgroup "
   "differences; an empirical test detected it.",
   bold_prefix="Discussion: ")
PA(f"A calibrated, leak-tested {NF}-feature model offers modest gains over "
   "bedside scores; external validation is required before clinical use.",
   bold_prefix="Conclusion: ")
doc.add_page_break()

# ================================================ BACKGROUND AND SIGNIFICANCE
HB1("Background and Significance")
PB("Unplanned 30-day readmissions burden the United States health system: "
   "the Centers for Medicare & Medicaid Services (CMS) puts the annual "
   "direct cost of Medicare readmissions above $26 billion [1], and the "
   "Hospital Readmissions Reduction Program (HRRP) has tied reimbursement to "
   "risk-adjusted readmission performance since 2013 [1,2]. Risk estimates "
   "available at discharge could target transitional care, but only if the "
   "model can be trusted at deployment time. Timing and equity sharpen the "
   "stakes: early and late readmissions differ in causes [3,4], and unequal "
   "model performance across groups can become unequal care [5,6].")
PB("Bedside indices remain the operational default. The LACE index [7] and "
   "the HOSPITAL score [8] reach AUROCs of roughly 0.60 to 0.72 in external "
   "validation [9,10,41], and systematic reviews report similar ceilings "
   "for regression models together with recurring weaknesses: "
   "leakage-prone splits, absent calibration assessment, and unexamined "
   "subgroup performance [9,11,12]. Machine learning on structured EHR "
   "data can exceed these ceilings [42], and gradient-boosted trees remain "
   "strong learners on tabular clinical data [13-16]. Clinical use also "
   "requires calibrated probabilities [17], interpretable explanations "
   "[18], same-cohort comparison with the scores hospitals use, fairness "
   "auditing [5,19-21], and reporting aligned with TRIPOD+AI [22] and "
   "PROBAST+AI [23].")
PB("Data leakage, the use of information during development that would not "
   "be available at prediction time, is a leading cause of overoptimistic "
   "and irreproducible results in machine-learning science [24]. EHR "
   "prediction is especially exposed: billing codes finalized after "
   "discharge, preprocessing fit on evaluation data, splits that ignore "
   "patient clustering, and engineered features whose time windows are not "
   "what their names suggest [23,25]. Roadmaps call for patient-level "
   "partitioning, fold-local preprocessing, and verification that every "
   "predictor is available at deployment time [22,23,26], but such "
   "verification is usually performed by reading documentation rather than "
   "by testing the data.")
PB("On the Medical Information Mart for Intensive Care (MIMIC-IV) [27], "
   "published 30-day readmission models report AUROCs of 0.791 with "
   "radiograph fusion on a selected subset [28], 0.727 with discharge notes "
   "[29], and 0.696 with 26 structured features [30]; ClinicalBERT reported "
   "approximately 0.714 on MIMIC-III notes [31]. None combines patient-grouped "
   "leakage-safe validation with an empirical test for future information, "
   "calibration, same-cohort reconstruction of LACE and HOSPITAL, timing "
   "analysis, and a subgroup audit. This study does so in the Medicare "
   "population, where the readmission penalty applies.")

# ================================================================== OBJECTIVE
HB1("Objective")
PB("Four research questions (RQs) organize the study. RQ1: Which "
   "discharge-available EHR features contribute most to prediction of "
   "30-day readmission in Medicare-insured adults? RQ2: Can a leakage-safe, "
   "calibrated machine-learning model improve prediction relative to "
   "established clinical scores? RQ3: Can global and patient-level Shapley "
   "additive explanations (SHAP) provide clinically interpretable insight "
   "into predicted risk? RQ4: How does performance vary across the "
   "post-discharge time window and across demographic subgroups?")

# ====================================================== MATERIALS AND METHODS
HB1("Materials and Methods")
HB2("Study Design and Data Source")
PB("We conducted a retrospective prediction-model development and "
   "internal-validation study reported in line with TRIPOD+AI [22] "
   "(checklist: Supplementary Appendix 5; PROBAST+AI self-assessment [23]: "
   "Supplementary Appendix 6). The data source is MIMIC-IV v3.1 [27,32], a "
   "deidentified EHR database of 546,028 hospitalizations between 2008 and "
   "2022 at Beth Israel Deaconess Medical Center (Boston, MA); dates are "
   "shifted per patient with within-patient intervals preserved. Clinical "
   "notes and imaging were not used.")
HB2("Cohort and Outcome Definition")
PB(f"The cohort comprised admissions with insurance recorded as Medicare "
   f"(n={CT['original_admissions']:,}). Because the model scores patients at "
   f"alive discharge, {CT['index_death_excluded']:,} admissions ending in "
   f"index in-hospital death were excluded, leaving "
   f"{CT['cohort_v2_admissions']:,} admissions from "
   f"{CT['cohort_v2_patients']:,} patients. Medicare insurance does not imply "
   f"age 65 or older, and fee-for-service cannot be distinguished from "
   f"Medicare Advantage, so this is not an HRRP regulatory cohort. The "
   f"outcome was the first same-system readmission of any cause beginning "
   f"more than 0 and up to 30 days after index discharge, identified from "
   f"all subsequent hospitalizations irrespective of insurance. Next "
   f"admissions overlapping the index discharge by up to 2 days "
   f"(n={CT['same_day_next_admit(delta<=0)']:,}) were treated as transfers. "
   f"Every index admission is eligible, so patients contribute multiple "
   f"admissions; all inference accounts for this clustering. Planned "
   f"readmissions could not be reliably identified; sensitivity analyses "
   f"vary the outcome definition (Supplementary Methods S1).")
HB2("Partitions and Validation Design")
PB(f"Admissions were split 80/20 at the patient level into development and "
   f"test data, with 10% of the development portion assigned, again "
   f"patient-grouped, to a validation partition used for the operating "
   f"threshold and for early stopping of the survival model. After "
   f"exclusions: training {p['train']['admissions']:,} admissions "
   f"({p['train']['patients']:,} patients, {p['train']['events']:,} events), "
   f"validation {p['val']['admissions']:,}, and test "
   f"{p['test']['admissions']:,} ({p['test']['patients']:,} patients, "
   f"{p['test']['events']:,} events).")
PB("Earlier development phases scored model variants on the original test "
   "partition repeatedly, so the test data are a historically exposed "
   "internal partition, not a pristine holdout. Three evidence tiers are "
   "reported in order of authority: (1) primary, the out-of-fold "
   "performance of the complete selection procedure under 5 outer "
   "patient-grouped folds over the development partitions; (2) secondary, "
   "patient-grouped cross-validation of the fixed consensus set on the same "
   "data (optimistic, because the set aggregates selection across folds); "
   "and (3) tertiary, a single evaluation of the final model on the test "
   "partition, used for detailed metrics, subgroups, baselines, and timing. "
   "No test outcome entered this version's selection, encoding, or "
   "threshold; the model family and fixed hyperparameters predate it "
   "(Limitations).")
HB2("Predictor Audit, Leakage Testing, and Feature Selection")
PB(f"All 207 candidate predictors were audited for provenance, measurement "
   f"window, missingness, and documented availability at discharge "
   f"(Supplementary Appendix 1). Sixty-three billing-derived features "
   f"(diagnosis-related groups, International Classification of Diseases "
   f"(ICD) codes, and their derivatives), one race-derived encoding, and "
   f"one precomputed encoding of discharge destination were excluded. "
   f"Because documentation can be wrong, we added an empirical "
   f"future-information test. Among first-ever admissions (no earlier "
   f"admission of any kind), a feature built only from prior stays carries "
   f"no information, so its values must not differ between patients who "
   f"will and will not be admitted again; a feature taking a fixed value "
   f"for every patient without a later admission, but not for others, "
   f"encodes the future. Outpatient-record features were additionally "
   f"compared with the outpatient table to confirm they used only records "
   f"dated on or before discharge (Supplementary Methods S3).")
PB(f"Feature selection used staged recursive feature elimination (RFE) "
   f"with an XGBoost estimator inside 5 outer patient-grouped folds on "
   f"development data only. Within each outer-training fold, 3 grouped "
   f"inner folds drove elimination along a fixed grid; categorical "
   f"variables were target-encoded from inner-training patients only; and "
   f"a parsimony rule, fixed before this analysis version, chose the "
   f"smallest feature count within 0.002 inner-validation AUROC of the "
   f"best. Features selected in at least 3 of 5 folds formed the consensus "
   f"set. Laboratory and vital-sign values were left missing for the tree "
   f"learner; counts and some derived prior-utilization features were set "
   f"to zero, or to a fixed value, when undefined (Supplementary Table S3).")
HB2("Model and Comparators")
PB(f"The final model is a single XGBoost classifier [14] on the consensus "
   f"set (600 trees without early stopping, learning rate 0.05, maximum "
   f"depth 5, subsample and column subsample 0.9, fixed seed), refit on the "
   f"training partition. Its operating threshold was the validation-partition "
   f"Youden point ({THRV:.3f}), development-derived and applied unchanged to "
   f"test. Comparators fitted per outer fold on identical data were the "
   f"full {N_ELIG}-feature eligible pool, the clean subset of an earlier "
   f"development feature set, the earlier 66-feature set (containing "
   f"billing features and leaked variables; historical reference), and "
   f"regularized logistic regression.")
HB2("Clinical-Score Baselines")
PB("LACE [7] and HOSPITAL [8] were reconstructed on the test partition, "
   "HOSPITAL with its original 12-month prior-admission window "
   "(Supplementary Appendix 2). Two components use coded data from the "
   "index stay (LACE comorbidity from ICD diagnoses; the HOSPITAL procedure "
   "item from ICD procedures), information our model was not allowed to "
   "use. Because the integer scores were derived for other outcomes, we also "
   "refit their components by logistic regression on the training "
   "partition, with and without the coded components.")
HB2("Leakage Ablation")
PB("To quantify how common pipeline choices inflate apparent performance, "
   "we compared the cross-validated (apparent) AUROC on development data "
   "with the held-out test AUROC of the model each choice would deploy, "
   "for admission-level instead of patient-level splitting, target "
   "encoding fit on all development data, and adding the future-information "
   "trend features, legacy precomputed encodings, or billing-derived "
   "features.")
HB2("Time-to-Event Analyses")
PB("Time to readmission used a competing-risks event process (readmission; "
   "death before readmission from hospital and Massachusetts registry "
   "records; administrative censoring at day 30) with Aalen-Johansen "
   "cumulative incidence and a cause-specific XGBoost accelerated failure "
   "time (AFT) model [34]. Evaluation used Harrell C [35] and cumulative/"
   "dynamic AUROC on days 1 to 29 [36,37], with patients who died before "
   "the evaluation day kept among the controls [37]. Because everyone not "
   "readmitted or dead is followed to day 30, no censoring occurs earlier "
   "and the estimates are empirical, with patient-cluster bootstrap bands. "
   "Three landmark models covered days 1 to 7, 8 to 14, and 15 to 30.")
HB2("Statistical Analysis")
PB("All confidence intervals (CIs) and P values use patient-level cluster "
   "bootstrap percentile resampling (1,000 resamples for the test panel; "
   "5,000 for contrasts with reported P values; two-sided empirical P values "
   "with plus-one correction). Model-to-model contrasts are paired; the "
   "White vs Black contrast resamples patients within each group. Formal "
   "testing was limited to three prespecified contrasts (vs LACE, vs "
   "HOSPITAL, and White vs Black); other results are descriptive. "
   "Calibration used the Brier score, expected calibration error (ECE) over "
   "10 equal-width bins, the calibration slope, and calibration-in-the-large "
   "(intercept with slope fixed at 1) [17]. Decision-curve analysis [38] "
   "covered thresholds 0.05 to 0.40. Analyses used Python 3.11.14 with "
   "XGBoost 3.2.0, scikit-learn 1.7.2, pandas 2.3.3, and NumPy 1.26.4 "
   "(seed 42).")
HB2("Subgroup Audit")
PB("Performance was audited on the test partition by age band, sex, race, "
   "and number of prior admissions (0, 1 to 2, 3 to 9, 10 or more), with "
   "race strings consolidated by substring mapping (Supplementary Appendix "
   "3). Race is not a model input; auditing by race is deliberate because "
   "omitting an attribute does not guarantee equal performance [5,19,20].")
HB2("Ethical Considerations")
PB("MIMIC-IV creation was approved by the institutional review boards of "
   "the Massachusetts Institute of Technology and Beth Israel Deaconess "
   "Medical Center with a waiver of informed consent [27,32]. The first "
   "author accessed the data under the PhysioNet Credentialed Health Data "
   "Use Agreement after required training; co-authors used only aggregate "
   "results, and no row-level data are redistributed. This secondary "
   "analysis of deidentified data did not require institutional review "
   "board review at Florida International University.")

# ==================================================================== RESULTS
HB1("Results")
HB2("Cohort")
PB(f"Figure 1 shows the cohort flow. Of {CT['original_admissions']:,} "
   f"Medicare admissions, {CT['index_death_excluded']:,} ended in index "
   f"in-hospital death, leaving {CT['cohort_v2_admissions']:,} admissions "
   f"from {CT['cohort_v2_patients']:,} patients; {CT['label_v2_events']:,} "
   f"({CT['label_v2_prevalence']*100:.1f}%) were followed by a within-system "
   f"readmission within 30 days.")
figure(FIG7 / "v7_fig1_flow.png",
       "Figure 1. Participant flow with patient-grouped partitions; feature "
       "selection ran inside 5 patient-grouped folds drawn from the "
       "development partitions.", width=4.6)
HB2("Future-Information Leakage")
PB(f"The empirical test flagged one documented prior-utilization feature, "
   f"a 180-day length-of-stay trend that had been the strongest predictor "
   f"of an earlier version of this model. Among "
   f"{LK['first_ever_admissions']:,} first-ever admissions, the trend was "
   f"nonzero for {pct(LT['nonzero_share_first_ever'])}; every such patient "
   f"had a later admission, and no patient without one had a nonzero "
   f"value (30-day readmission {pct(LT['readmit30_if_nonzero'])} vs "
   f"{pct(LT['readmit30_if_zero'])}). The feature, and two interaction "
   f"terms built from it, therefore used later stays. No other candidate "
   f"showed this pattern, and the outpatient features matched values dated "
   f"on or before discharge "
   f"({pct(LK['outpatient_features_exact_match']['bp_diastolic_outpatient']['on_or_before_discharge'])} "
   f"for diastolic pressure vs "
   f"{pct(LK['outpatient_features_exact_match']['bp_diastolic_outpatient']['overall_incl_future'])} "
   f"for any date). Removing the three features lowered the out-of-fold "
   f"AUROC from {OLD_OOF:.4f} to {OOF['auroc']:.4f} and test AUROC from "
   f"{OLD_TEST:.4f} to {MP['auroc']['point']:.4f}. All results below use "
   f"the leak-free pool of {N_ELIG} features.")
_ab_rows = [["Pipeline choice", "Features", "Apparent CV AUROC",
             "Held-out test AUROC", "Optimism"]]
_ab_names = {"reference_leakage_safe": "Leakage-safe reference",
             "admission_level_split": "Admission-level split",
             "global_target_encoding": "Encoding fit on all development data",
             "plus_future_trend_features": "Add future-information trend features",
             "plus_legacy_precomputed_encodings": "Add legacy precomputed encodings",
             "plus_billing_features": "Add billing-derived features",
             "all_combined": "All of the above combined"}
for k, nm in _ab_names.items():
    e = ABL[k]
    _ab_rows.append([nm, str(e["n_features"]), f4(e["apparent_cv_auroc"]),
                     f4(e["heldout_test_auroc"]), f"{e['optimism']:+.4f}"])
caption("Table 1. Leakage ablation. Apparent AUROC is the 5-fold "
        "cross-validated estimate on development data under each pipeline "
        "choice; held-out AUROC is that of the model the choice would "
        "deploy, trained on the training partition and scored on the test "
        "partition. Optimism is apparent minus held-out. CV: "
        "cross-validation.", keep_with_next=True)
table(_ab_rows, font_size=8.5, widths=[2.4, 0.8, 1.15, 1.15, 0.9])
_opt_trend = ABP["apparent_cv_auroc"] - ABR["apparent_cv_auroc"]
_ac = ABL["all_combined"]
PB(f"In the ablation (Table 1), the reference pipeline showed optimism of "
   f"{ABR['optimism']:+.4f}, and admission-level splitting or encoding on "
   f"all development data changed little on these features. Adding the "
   f"trend features raised apparent AUROC by {_opt_trend:.3f} and held-out "
   f"AUROC by a similar amount, because the future information was present "
   f"in the test data too: internal validation cannot detect this kind of "
   f"leak. Combining all choices produced apparent AUROC "
   f"{_ac['apparent_cv_auroc']:.3f} against {_ac['heldout_test_auroc']:.3f} "
   f"held out.")
HB2("Feature Selection and Model Comparison")
_cmp = {k: CMP5[k] for k in ("rfe", "all_eligible", "hist_clean", "f66", "logit")}
PAIRED = CMP5["_paired_differences_vs_rfe"]
PB(f"The parsimony rule selected {', '.join(str(v) for v in STAB['fold_counts'].values())} "
   f"features across the outer folds (mean pairwise Jaccard "
   f"{STAB['jaccard_mean']:.3f}; {N_UNANIMOUS} selected in all 5; Figure 2), "
   f"and the consensus rule yielded {NF} features. Out of fold, the "
   f"selection procedure achieved AUROC {f4(_cmp['rfe']['oof_auroc'])} "
   f"(95% CI {c4(_cmp['rfe']['auroc_ci95_cluster'])}), equal to the full "
   f"{N_ELIG}-feature pool ({f4(_cmp['all_eligible']['oof_auroc'])}; paired "
   f"difference {abs(PAIRED['all_eligible_minus_rfe']['point']):.4f}, 95% CI "
   f"{c4(PAIRED['all_eligible_minus_rfe']['ci95'])}) and above regularized "
   f"logistic regression ({f4(_cmp['logit']['oof_auroc'])}). The earlier "
   f"66-feature set scored {f4(_cmp['f66']['oof_auroc'])}, higher by "
   f"{PAIRED['f66_minus_rfe']['point']:.4f}, because it contains the leaked "
   f"trend and encodings. The consensus model contains no billing-derived "
   f"features.")
figure(FIG7 / "v7_stability.png",
       f"Figure 2. Selection frequency across the 5 outer folds for all "
       f"{len(STAB['selection_frequency'])} features selected at least once; "
       "teal bars form the consensus set, and the dashed line marks the "
       "consensus threshold (3 of 5 folds).", width=3.7)
HB2("Discrimination and Calibration")
PB(f"Development cross-validation of the consensus model gave AUROC "
   f"{f4(CCV['mean'])} (fold SD {CCV['sd']:.4f}). On the test partition "
   f"(tertiary evidence), the model reached AUROC {f4(MP['auroc']['point'])} "
   f"(95% CI {c4(MP['auroc']['ci95'])}) and average precision "
   f"{MP['ap']['point']:.3f} against a "
   f"{p['test']['events']/p['test']['admissions']*100:.1f}% prevalence. "
   f"Calibration was close to ideal: slope {MP['slope']['point']:.2f} (95% "
   f"CI {MP['slope']['ci95'][0]:.2f} to {MP['slope']['ci95'][1]:.2f}), "
   f"calibration-in-the-large {MP['citl']['point']:.2f} (95% CI "
   f"{MP['citl']['ci95'][0]:.2f} to {MP['citl']['ci95'][1]:.2f}; mean "
   f"predicted {pct(MP['mean_predicted'])} vs observed "
   f"{pct(MP['observed_rate'])}), ECE {MP['ece']['point']:.3f}, and Brier "
   f"{MP['brier']['point']:.4f} (Figure 3). At the threshold of {THRV:.3f}, "
   f"sensitivity was {MP['sensitivity']['point']:.3f}, specificity "
   f"{MP['specificity']['point']:.3f}, positive predictive value "
   f"{MP['ppv']['point']:.3f}, and negative predictive value "
   f"{MP['npv']['point']:.3f}.")
figure(FIG7 / "v7_roc_cal.png",
       "Figure 3. Receiver operating characteristic, precision and recall, "
       "calibration, and confusion-matrix panels for the final model on the "
       "test partition. Calibration points are decile bins of predicted risk; "
       "ECE uses 10 equal-width bins; CITL: calibration-in-the-large.",
       width=6.0)
HB2("Sensitivity Analyses")
_sa, _su, _sn = (SENS["sameday_as_readmission"], SENS["unplanned_only"],
                 SENS["nonelective_index_only"])
_eb = EX["era_stratified_test"]["bands"]
PB(f"Counting the transfer band as readmissions (AUROC "
   f"{f4(_sa['test_auroc_refit'])}), excluding next admissions typed "
   f"ELECTIVE ({f4(_su['test_auroc_refit'])}), restricting to nonelective "
   f"index admissions ({f4(_sn['test_auroc'])}), excluding admissions "
   f"reaching the 2022 end of data ({f4(BD5['auroc_safe'])}), and treating "
   f"returns within 1 hour of discharge as continuations "
   f"({f4(OH['test_auroc_refit'])}) all left discrimination essentially "
   f"unchanged. AUROC was stable across calendar-era bands "
   f"({min(b['auroc'] for b in _eb):.3f} to {max(b['auroc'] for b in _eb):.3f}; "
   f"Supplementary Table S1).")
HB2("Same-Cohort Clinical Baselines")
_g = CMPR
PB(f"The final model outperformed reconstructed LACE "
   f"({f4(BSE['lace_auroc'])}) by {BSE['uplift_lace']:.4f} (95% CI "
   f"{c4(BSE['uplift_lace_ci'])}; P<.001) and HOSPITAL "
   f"({f4(BSE['hospital12_auroc'])}) by {BSE['uplift_hospital12']:.4f} (95% "
   f"CI {c4(BSE['uplift_hospital12_ci'])}; P<.001). Refitting the score "
   f"components to this cohort raised LACE to "
   f"{f4(_g['lace_refit']['auroc'])} and HOSPITAL to "
   f"{f4(_g['hospital_refit']['auroc'])}; combining the components of both "
   f"without the coded items gave {f4(_g['lace_hospital_refit_no_coded']['auroc'])}, "
   f"still below the model by {_g['lace_hospital_refit_no_coded']['uplift']:.4f} "
   f"(95% CI {c4(_g['lace_hospital_refit_no_coded']['uplift_ci95'])}) "
   f"(Table 2).")
_t2 = [["Comparator", "Data", "AUROC", "Note"],
       ["LACE, integer score [7]", "This test partition", f4(BSE["lace_auroc"]),
        "Reconstructed; uses coded comorbidity"],
       ["LACE, components refit", "This test partition",
        f4(_g["lace_refit"]["auroc"]), "Logistic refit on training data"],
       ["HOSPITAL, integer score [8]", "This test partition",
        f4(BSE["hospital12_auroc"]), "12-month window; uses coded procedures"],
       ["HOSPITAL, components refit", "This test partition",
        f4(_g["hospital_refit"]["auroc"]), "Logistic refit on training data"],
       ["LACE and HOSPITAL, refit without coded items", "This test partition",
        f4(_g["lace_hospital_refit_no_coded"]["auroc"]),
        "Discharge-available components only"],
       ["Logistic regression, eligible pool", "Development, out of fold",
        f4(_cmp["logit"]["oof_auroc"]), f"{N_ELIG} features"],
       ["This study, selection procedure", "Development, out of fold",
        f4(OOF["auroc"]), "Primary evidence"],
       [f"This study, fixed {NF}-feature model", "Test partition",
        f4(MP["auroc"]["point"]), "Tertiary evidence"],
       ["Adisa 2026 [30]", "MIMIC-IV, all adults", "0.696",
        "Different cohort and outcome"],
       ["Almeida 2025 [29]", "MIMIC-IV, all adults", "0.727",
        "Discharge notes; different cohort"],
       ["Tang 2023 [28]", "MIMIC-IV, radiograph subset", "0.791",
        "Multimodal; imaging-selected"]]
caption("Table 2. Comparison with clinical scores, a linear model, and "
        "published MIMIC-IV readmission models. Published values come from "
        "different cohorts and outcome definitions and are reference points, "
        "not head-to-head results.", keep_with_next=True)
table(_t2, font_size=8.5, widths=[2.0, 1.5, 0.7, 2.2])
HB2("Timing of Readmission")
_cif = SV.get("aj_cif_30d")
_tp, _t1 = TM["primary"], TM["one_hour_rule"]
_l1 = TM["landmark_one_hour_rule"]
PB(f"The test partition contributed {SV['test_events']:,} readmissions and "
   f"{SV.get('test_competing_deaths'):,} deaths before readmission "
   f"(30-day cumulative incidence {_cif['readmission']*100:.1f}% and "
   f"{_cif['death']*100:.1f}%). The AFT model reached Harrell C "
   f"{FM['survival']['aft_harrell_c']:.4f}. Daily AUROC peaked on day 1 "
   f"({_tp['auc'][0]:.3f}, 95% CI {_tp['lo'][0]:.3f} to {_tp['hi'][0]:.3f}) "
   f"and fell to about {_tp['auc'][6]:.2f} by day 7 (Figure 4). The peak was "
   f"an artifact: across all partitions, {CB['events_in_band']:,} "
   f"readmissions ({pct(CB['share_of_events'])} of events) began within 1 "
   f"hour of discharge, a rate of {CB['rate_per_hour']['0_1h']['per_hour']:,.0f} "
   f"per hour against {CB['later_rate_range_per_hour'][0]:.0f} to "
   f"{CB['later_rate_range_per_hour'][1]:.0f} per hour at every later "
   f"interval. These were administrative continuations: "
   f"{pct(CB['next_internal_psych_transfer'], 0)} were internal psychiatric "
   f"transfers, {pct(CB['next_from_snf'], 0)} arrivals from skilled "
   f"nursing, and {pct(CB['next_ed_registration_before_index_discharge'], 0)} "
   f"carried an emergency department registration that began before the "
   f"index discharge. Reclassifying them as continuations removed the peak "
   f"(day 1 {_t1['auc'][0]:.3f}, 95% CI {_t1['lo'][0]:.3f} to "
   f"{_t1['hi'][0]:.3f}); discrimination was lowest in the first week "
   f"(minimum {min(_t1['auc']):.3f}) and rose to {_t1['auc'][-1]:.3f} by "
   f"day 29. Landmark AUROCs under this rule were "
   f"{_l1[0]['auroc']:.3f}, {_l1[1]['auroc']:.3f}, and {_l1[2]['auroc']:.3f} "
   f"for days 1 to 7, 8 to 14, and 15 to 30.")
figure(FIG7 / "v7_daily.png",
       "Figure 4. Cumulative/dynamic AUROC of the AFT risk score by day since "
       "discharge, with deaths before each day kept among the controls; "
       "shaded bands are 95% patient-cluster bootstrap intervals. The "
       "primary outcome definition counts returns beginning within 1 hour "
       "of discharge as readmissions; the second curve treats them as "
       "continuations.", width=5.4)
HB2("Subgroup Performance")
_s0, _s10 = PU["0"], PU["10 or more"]
PB(f"Discrimination was similar for Black and White patients (AUROC "
   f"{f4(FR['Black']['auroc'])} vs {f4(FR['White']['auroc'])}; difference "
   f"{wb['point']:.4f}, 95% CI {c4(wb['ci95'])}; {fmt_p(wb['p_bootstrap'])}). "
   f"Before the leak was removed, this difference had been {OLD_WB:.3f} "
   f"(P<.001): the leaked trend was informative mainly for patients with "
   f"few prior admissions, and White patients had fewer (median "
   f"{RU['median_prior_admissions']['White']:.0f} vs "
   f"{RU['median_prior_admissions']['Black']:.0f}). Discrimination was "
   f"lower in older patients ({f4(AB['75-84']['auroc'])} for ages 75 to 84 "
   f"and {f4(AB['85+']['auroc'])} for 85 or older, vs "
   f"{f4(AB['65-74']['auroc'])} for 65 to 74) and rose with prior "
   f"utilization, from {f4(_s0['auroc'])} (95% CI {c4(_s0['auroc_ci95'])}) "
   f"for first admissions, {pct(_s0['share'], 0)} of the test partition, "
   f"to {f4(_s10['auroc'])} for patients with 10 or more prior admissions, "
   f"who formed {pct(_s10['share_of_top10pct_flags'], 0)} of the "
   f"highest-risk decile (Figure 5, Supplementary Table S2). Within utilization strata "
   f"the White minus Black difference remained small (standardized "
   f"{RU['standardized_within_stratum_gap']:.3f}, 95% CI "
   f"{RU['standardized_ci95'][0]:.3f} to {RU['standardized_ci95'][1]:.3f}). At the shared threshold, specificity was "
   f"{FR['Black']['specificity']:.3f} for Black and "
   f"{FR['White']['specificity']:.3f} for White patients, so nonreadmitted "
   f"Black patients were flagged more often, an operational difference to "
   f"monitor. Calibration slopes by race ranged from "
   f"{min(e['cal_slope'] for e in FR.values()):.2f} to "
   f"{max(e['cal_slope'] for e in FR.values()):.2f}.")


def _ci3(e, key):
    c_ = e.get(f"{key}_ci95"); v = e.get(key)
    if key == "cal_slope":
        return f"{v:.2f} ({c_[0]:.2f} to {c_[1]:.2f})"
    if key == "auroc":
        return f"{v:.4f} ({c_[0]:.4f} to {c_[1]:.4f})"
    return f"{v:.3f} ({c_[0]:.3f} to {c_[1]:.3f})"


rows = [["Group", "Patients / admissions / events", "Prev.", "AUROC (95% CI)",
         "Sensitivity (95% CI)", "PPV", "Brier", "Cal. slope (95% CI)"]]
for axis, order_ in (("race", ["White", "Black", "Hispanic/Latino", "Asian",
                               "Other/Unknown"]),
                     ("sex", ["Female", "Male"]),
                     ("age_band", ["<65", "65-74", "75-84", "85+"])):
    for g_ in order_:
        e = FM["fairness"][axis][g_]
        rows.append([g_.replace("-", " to ").replace("Hispanic/Latino", "Hispanic")
                     .replace("Other/Unknown", "Other"),
                     f"{e['patients']:,} / {e['admissions']:,} / {e['events']:,}",
                     f"{e['events']/e['admissions']*100:.1f}%",
                     _ci3(e, "auroc"), _ci3(e, "sensitivity"),
                     f"{e['ppv']:.3f}", f"{e['brier']:.3f}", _ci3(e, "cal_slope")])
caption(f"Table 3. Subgroup performance on the test partition at the "
        f"development-derived threshold ({THRV:.3f}). CIs are 95% "
        "patient-cluster bootstrap intervals; complete metrics appear in "
        "Supplementary Appendix 1. Hispanic: Hispanic or Latino; Other: "
        "other or unknown race (heterogeneous, not interpreted). PPV: "
        "positive predictive value; Prev.: prevalence; Cal.: calibration. "
        "Patients whose recorded race or age band changes between admissions "
        "appear in more than one row of a block.", keep_with_next=True)
table(rows, font_size=8,
      widths=[0.75, 1.2, 0.5, 1.05, 1.0, 0.45, 0.5, 0.95])
figure(FIG7 / "v7_fairness.png",
       "Figure 5. Test AUROC by age band, sex, race, and number of prior "
       "admissions, with 95% patient-cluster bootstrap CIs; the dashed line "
       "marks the overall test AUROC.", width=5.6)
HB2("Interpretability via SHAP")
_t = [label(s["feature"]).replace(", 6 mo", " in the prior 6 months").lower() for s in SH[:5]]
PB(f"The leading predictors by mean absolute SHAP value were "
   f"{_t[0]} ({SH[0]['mean_abs_shap']:.3f} on the log-odds scale), {_t[1]}, "
   f"{_t[2]}, {_t[3]}, and {_t[4]} (Figure 6). No single feature dominated: "
   f"prior utilization, laboratory status at discharge, and the intensity of "
   f"medication and laboratory activity during the stay jointly carried the "
   f"signal. Additive SHAP decompositions give each admission a ranked list "
   f"of contributing factors.")
figure(FIG7 / "v7_shap.png",
       f"Figure 6. Top 10 predictors of the final {NF}-feature model by mean "
       "absolute SHAP value on the test partition (log-odds scale).",
       width=5.2)
HB2("Clinical Utility (Exploratory)")
PB(f"Decision curves showed positive net benefit over treat-all and "
   f"treat-none across thresholds 0.05 to 0.40, a range not yet grounded "
   f"in operational evidence (Supplementary Appendix 4). Flagging the top "
   f"10% of admissions captured {CAP['top_10']['captured']*100:.0f}% of "
   f"readmissions at a positive predictive value of "
   f"{CAP['top_10']['ppv']:.2f}. Because patients who die without "
   f"readmission are nonevents, low readmission risk must not be read as "
   f"clinical stability.")

# ================================================================= DISCUSSION
HB1("Discussion")
HB2("Principal Findings")
PB(f"In {CT['cohort_v2_admissions']:,} Medicare-insured admissions, a "
   f"{NF}-feature gradient-boosted model selected under patient-grouped, "
   f"fold-local preprocessing from empirically leak-tested, nonbilling "
   f"predictors estimated 30-day within-system readmission with an "
   f"out-of-fold AUROC of {OOF['auroc']:.3f} and near-ideal calibration. "
   f"Four findings answer the research questions. First (RQ1, RQ3), risk "
   f"was carried jointly by recency-weighted prior utilization, laboratory "
   f"status at discharge, and in-hospital care intensity. Second (RQ2), the "
   f"model outperformed LACE and HOSPITAL, including versions refit to this "
   f"cohort, by a modest but consistent margin. Third (RQ4), once "
   f"administrative continuations were separated from readmissions, "
   f"discrimination was lowest in the first week and rose later in the "
   f"window; it was similar across race and sex, but lower in older "
   f"patients and in first admissions. Fourth, and most "
   f"general, a feature that passed a documentation-based audit encoded "
   f"later admissions; it inflated AUROC by about "
   f"{OLD_TEST - MP['auroc']['point']:.2f} and created two findings, an "
   f"apparent advantage for first-time patients and a racial "
   f"discrimination gap, that disappeared once it was removed.")
HB2("Lessons for Leakage Auditing")
PB("The leaked feature carried a name and documentation describing a "
   "prior 180-day window and passed the original availability audit; "
   "patient-grouped validation could not expose it because the future "
   "information was present in every partition. Two inexpensive checks "
   "would have caught it: testing prior-history features among first-ever "
   "admissions, and confirming that time-windowed features reproduce from "
   f"raw timestamps. Similarly, {pct(CB['share_of_events'])} of outcome events were "
   "administrative continuations within an hour of discharge; inspecting the "
   "distribution of discharge-to-readmission gaps is a simple safeguard "
   "before any timing claim. We recommend both checks for EHR prediction "
   "studies that engineer history features from longitudinal records.")
HB2("Comparison With Prior Work")
PB(f"On MIMIC-IV, our out-of-fold AUROC of {OOF['auroc']:.3f} is comparable "
   f"to the structured-feature report (0.696) [30], below the notes-based "
   f"(0.727) [29] and multimodal (0.791) [28] reports, and below the 0.75 to "
   f"0.76 reported with full-EHR deep learning at two academic centers "
   f"[42]. Cohorts, outcomes, and validation designs differ, so these are "
   f"reference points; our controlled comparison is the same-cohort "
   f"baseline analysis. We cannot exclude similar leakage in published "
   f"models that engineer history features, and we suggest that "
   f"first-admission tests become routine.")
HB2("Clinical Interpretation (Hypotheses)")
PB("SHAP values describe the model's reliance on a feature, not causal "
   "effects. Recency-weighted prior admissions identify patients for whom "
   "standard discharge planning has already failed, but they also reflect "
   "outpatient-access barriers, so the score should be paired with "
   "additional resources, never reduced care [5]. Unresolved laboratory "
   "derangement at discharge suggests patients who may warrant contact "
   "within days. Discrimination is weakest for first admissions, where no "
   "history exists, and in the first post-discharge week, the window that "
   "transitional interventions target; early follow-up decisions should "
   "not rest on the score alone.")
HB2("Limitations")
PB(f"This is a single-center, retrospective, internally validated study "
   f"whose case mix ({FR['White']['admissions']/p['test']['admissions']*100:.1f}% "
   f"of test admissions White) differs from national Medicare; no "
   f"clinical-impact evidence exists. The outcome is within-system, so "
   f"out-of-system readmissions are missed, and planned readmissions are "
   f"included. Hyperparameters were fixed rather than tuned within folds. "
   f"The test partition was used repeatedly in earlier development, and "
   f"the leak itself was found after earlier versions had been evaluated; "
   f"the present estimates follow a principled correction rather than a "
   f"fresh holdout, and residual indirect influence cannot be excluded. "
   f"The validation partition participated in consensus selection and set "
   f"the threshold. Our leakage test covers features built from prior "
   f"admissions and outpatient records; features from within the index "
   f"stay were checked against documented timestamps only. Some prior-"
   f"utilization features use zero or a fixed value when undefined. Social "
   f"determinants are unavailable. Bootstrap intervals do not capture "
   f"retraining or reselection variability.")

# ================================================================= CONCLUSION
HB1("Conclusion")
PB(f"A calibrated, leak-tested {NF}-feature model estimates 30-day "
   f"within-system readmission at discharge in Medicare-insured adults with "
   f"an out-of-fold AUROC of {OOF['auroc']:.3f}, modestly above reconstructed "
   f"and refit bedside scores, with similar performance across race and sex. "
   f"An empirical first-admission test exposed a documented history feature "
   f"that encoded later admissions and had inflated earlier results; such "
   f"tests should accompany availability audits. External validation and "
   f"prospective evaluation are required before clinical use.")

# ================================================================ BACK MATTER
H1("Acknowledgments")
P("This work began as the capstone project for the Master of Science in "
  "Data Science and Artificial Intelligence at Florida International "
  "University. We thank the MIT Laboratory for Computational Physiology and "
  "the PhysioNet team for maintaining MIMIC-IV. Generative AI tools were "
  "used in this work: OpenAI ChatGPT for independent audits of draft "
  "manuscripts, and Anthropic Claude for analysis-code development, "
  "independent methodological review (including the review that identified "
  "the future-information leak), figure generation, and drafting and "
  "editing of the manuscript. The authors reviewed and verified all code, "
  "analyses, and text and take full responsibility for the content.")
H1("Conflict of Interest Statement")
P("None declared.")
H1("Funding")
P("This study received no external funding.")
H1("Data Availability Statement")
P("MIMIC-IV v3.1 is available from PhysioNet under a credentialed data use "
  "agreement [27,32] and is not redistributed. All analysis code, the "
  "leakage tests, and the machine-readable results files from which every "
  "value in this manuscript is generated (final_model_v7.json and companion "
  "files) are available at "
  "https://github.com/thiagobandeira1/Medicare-30day-Readmission-MIMIC-IV "
  "(tagged release v7.0-jamia) and archived at Zenodo (concept DOI: "
  "10.5281/zenodo.21987701, which resolves to the latest version). A "
  "demonstration prototype in the companion repositories "
  "(https://github.com/thiagobandeira1/readmission-risk-api and "
  "https://github.com/thiagobandeira1/riskpath-clinician-companion) "
  "illustrates an earlier development model and is not the model reported "
  "here.")
H1("Author Contributions")
P("Conceptualization: TB; methodology: TB, AG, CP, AMM; software, formal "
  "analysis, and visualization: TB, AG; data curation: TB, AG; "
  "investigation and validation: TB, AG; writing (original draft): TB; "
  "writing (review and editing): AG, CP, AMM; supervision: CP, AMM. All "
  "authors approved the final version. (CRediT taxonomy.)")
H1("Supplementary Material")
P("Supplementary Material (single file): Supplementary Methods (cohort and "
  "outcome definitions, continuation band, partitions, leakage test, "
  "missing-data handling, clinical-score reconstruction and refits, "
  "time-to-event and statistical detail), Supplementary Results "
  "(sensitivity and era analyses, payer scope, fixed-set comparison, tie "
  "rule, prior-utilization and race analyses, stage attribution), "
  "Supplementary Tables S1 to S3 (calendar era, prior-utilization strata, "
  "and definitions of the final features), and "
  "Supplementary Figures S1 to S3. Supplementary Appendix 1: predictor "
  "audit and selection workbook. Supplementary Appendix 2: LACE and "
  "HOSPITAL reconstruction. Supplementary Appendix 3: race consolidation. "
  "Supplementary Appendix 4: decision curves and development history. "
  "Supplementary Appendix 5: TRIPOD+AI checklist. Supplementary Appendix "
  "6: PROBAST+AI self-assessment.")

H1("References")
REFS = [
 "Centers for Medicare & Medicaid Services. Hospital Readmissions Reduction "
 "Program (HRRP). Baltimore, MD: CMS; 2024. URL: https://www.cms.gov/medicare/"
 "payment/prospective-payment-systems/acute-inpatient-pps/"
 "hospital-readmissions-reduction-program-hrrp [accessed 2026-08-16]",
 "McIlvennan CK, Eapen ZJ, Allen LA. Hospital readmissions reduction program. "
 "Circulation. 2015;131(20):1796-1803.",
 "van Walraven C, Dhalla IA, Bell C, et al. Derivation and validation of an index "
 "to predict early death or unplanned readmission after discharge from hospital to "
 "the community. CMAJ. 2010;182(6):551-557.",
 "Donze J, Aujesky D, Williams D, Schnipper JL. Potentially avoidable 30-day "
 "hospital readmissions in medical patients: derivation and validation of a "
 "prediction model. JAMA Intern Med. 2013;173(8):632-638.",
 "Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for "
 "hospital readmission: a systematic review. JAMA. 2011;306(15):1688-1698.",
 "Damery S, Combes G. Evaluating the predictive strength of the LACE index in "
 "identifying patients at high risk of hospital readmission: a retrospective "
 "cohort study. BMJ Open. 2017;7(7):e016921.",
 "Artetxe A, Beristain A, Grana M. Predictive models for hospital readmission "
 "risk: a systematic review of methods. Comput Methods Programs Biomed. "
 "2018;164:49-64.",
 "Mahmoudi E, Kamdar N, Kim N, et al. Use of electronic medical records in "
 "development and validation of risk prediction models of hospital readmission: "
 "systematic review. BMJ. 2020;369:m958.",
 "Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform "
 "deep learning on typical tabular data? Adv Neural Inf Process Syst (Datasets and "
 "Benchmarks). 2022.",
 "Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: Proceedings of "
 "KDD 2016. ACM; 2016:785-794.",
 "Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting "
 "decision tree. Adv Neural Inf Process Syst. 2017.",
 "Shwartz-Ziv R, Armon A. Tabular data: deep learning is not all you need. Inf "
 "Fusion. 2022;81:84-90.",
 "Van Calster B, Nieboer D, Vergouwe Y, et al. A calibration hierarchy for risk "
 "models was defined: from utopia to empirical data. J Clin Epidemiol. "
 "2016;74:167-176.",
 "Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Adv "
 "Neural Inf Process Syst. 2017.",
 "Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an "
 "algorithm used to manage the health of populations. Science. "
 "2019;366(6464):447-453.",
 "Chen IY, Pierson E, Rose S, et al. Ethical machine learning in healthcare. Annu "
 "Rev Biomed Data Sci. 2021;4:123-144.",
 "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance "
 "for reporting clinical prediction models that use regression or machine learning "
 "methods. BMJ. 2024;385:e078378.",
 "Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of "
 "bias, and applicability assessment tool for prediction models using regression "
 "or artificial intelligence methods. BMJ. 2025;388:e082505. "
 "doi:10.1136/bmj-2024-082505",
 "Vyas DA, Eisenstein LG, Jones DS. Hidden in plain sight: reconsidering the use "
 "of race correction in clinical algorithms. N Engl J Med. 2020;383(9):874-882.",
 "Inker LA, Eneanya ND, Coresh J, et al. New creatinine- and cystatin C-based "
 "equations to estimate GFR without race. N Engl J Med. 2021;385(19):1737-1749.",
 "Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible "
 "electronic health record dataset. Sci Data. 2023;10:1. (Database v3.1: "
 "PhysioNet. doi:10.13026/kpb9-mt58)",
 "Tang S, Tariq A, Dunnmon JA, et al. Predicting 30-day all-cause hospital "
 "readmission using multimodal spatiotemporal graph neural networks. IEEE J "
 "Biomed Health Inform. 2023;27(4):2071-2082.",
 "Almeida T, Moreno P, Barata C. Prediction of 30-day hospital readmission with "
 "clinical notes and EHR information. In: Pattern Recognition and Image "
 "Analysis (IbPRIA 2025). Cham: Springer; 2025:220-232. "
 "doi:10.1007/978-3-031-99568-2_18",
 "Adisa IT. An integrated framework for explainable, fair, and observable "
 "hospital readmission prediction: development and validation on MIMIC-IV. "
 "Preprint. arXiv:2604.22535. 2026. URL: https://arxiv.org/abs/2604.22535 "
 "[accessed 2026-08-16]",
 "Huang K, Altosaar J, Ranganath R. ClinicalBERT: modeling clinical notes and "
 "predicting hospital readmission. Preprint. arXiv:1904.05342. 2020. URL: "
 "https://arxiv.org/abs/1904.05342 [accessed 2026-09-15]",
 "Goldberger AL, Amaral LA, Glass L, et al. PhysioBank, PhysioToolkit, and "
 "PhysioNet: components of a new research resource for complex physiologic "
 "signals. Circulation. 2000;101(23):e215-e220.",
 "Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. Revisiting deep learning models "
 "for tabular data. Adv Neural Inf Process Syst. 2021.",
 "Barnwal A, Cho H, Hocking T. Survival regression with accelerated failure time "
 "model in XGBoost. J Comput Graph Stat. 2022;31(4):1292-1302. "
 "doi:10.1080/10618600.2022.2067548",
 "Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in "
 "developing models, evaluating assumptions and adequacy, and measuring and "
 "reducing errors. Stat Med. 1996;15(4):361-387.",
 "Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for "
 "evaluating overall adequacy of risk prediction procedures with censored survival "
 "data. Stat Med. 2011;30(10):1105-1117.",
 "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating "
 "prediction models. Med Decis Making. 2006;26(6):565-574.",
 "Johnson AEW, Pollard TJ, Horng S, et al. MIMIC-IV-Note: deidentified free-text "
 "clinical notes (version 2.2). PhysioNet. 2023. doi:10.13026/1n74-ne17",
 "Johnson AEW, Pollard TJ, Berkowitz SJ, et al. MIMIC-CXR, a de-identified "
 "publicly available database of chest radiographs with free-text reports. Sci "
 "Data. 2019;6:317.",
 "Kapoor S, Narayanan A. Leakage and the reproducibility crisis in "
 "machine-learning-based science. Patterns (N Y). 2023;4(9):100804. "
 "doi:10.1016/j.patter.2023.100804",
 "Futoma J, Simons M, Panch T, Doshi-Velez F, Celi LA. The myth of "
 "generalisability in clinical research and machine learning in health "
 "care. Lancet Digit Health. 2020;2(9):e489-e492.",
 "Wiens J, Saria S, Sendak M, et al. Do no harm: a roadmap for responsible "
 "machine learning for health care. Nat Med. 2019;25(9):1337-1340.",
 "Graham KL, Wilker EH, Howell MD, Davis RB, Marcantonio ER. Differences "
 "between early and late readmissions among patients: a cohort study. Ann "
 "Intern Med. 2015;162(11):741-749.",
 "Krumholz HM. Post-hospital syndrome: an acquired, transient condition "
 "of generalized risk. N Engl J Med. 2013;368(2):100-102.",
 "Seyyed-Kalantari L, Zhang H, McDermott MBA, Chen IY, Ghassemi M. "
 "Underdiagnosis bias of artificial intelligence algorithms applied to "
 "chest radiographs in under-served patient populations. Nat Med. "
 "2021;27(12):2176-2182.",
 "Blanche P, Dartigues JF, Jacqmin-Gadda H. Estimating and comparing "
 "time-dependent areas under receiver operating characteristic curves "
 "for censored event times with competing risks. Stat Med. "
 "2013;32(30):5381-5397.",
 "Donze JD, Williams MV, Robinson EJ, et al. International validity of the "
 "HOSPITAL score to predict 30-day potentially avoidable hospital "
 "readmissions. JAMA Intern Med. 2016;176(4):496-502. "
 "doi:10.1001/jamainternmed.2015.8462",
 "Rajkomar A, Oren E, Chen K, et al. Scalable and accurate deep learning "
 "with electronic health records. NPJ Digit Med. 2018;1:18. "
 "doi:10.1038/s41746-018-0029-1",
]
# Citation keys above use the builder-internal numbering of the JAMIA v6
# builder (1 to 40) plus [41] Donze 2016 and [42] Rajkomar 2018. The JAMIA v6
# text cited by its OWN renumbered order, so the in-text numbers written in
# this file follow that renumbered JAMIA v6 order; map them back here.
_V6_ORDER = json.loads((Path(__file__).resolve().parent /
                        "_jamia_ref_mapping.json").read_text())["mapping"]
_new_to_old = {v: int(k) for k, v in _V6_ORDER.items()}
REFS_BY_TEXTNUM = {n: REFS[_new_to_old[n] - 1] for n in range(1, 41)}
REFS_BY_TEXTNUM[41] = REFS[40]
REFS_BY_TEXTNUM[42] = REFS[41]
REFS_TEXTORDER = [REFS_BY_TEXTNUM[n] for n in range(1, 43)]

from _build_jamia_v7_renumber import renumber
FINAL_REFS = renumber(doc, REFS_TEXTORDER)
for i, ref in enumerate(FINAL_REFS, 1):
    P(f"{i}. {ref}")

BODY, ABS = _COUNT["body"], _COUNT["abstract"]
wc_run.text = _style(f"Word count: {BODY:,} (main text, excluding title page, "
                     f"abstract, references, figures, and tables); abstract: "
                     f"{ABS} words.")
print(f"BODY {BODY} | ABSTRACT {ABS} | REFS {len(FINAL_REFS)}")
assert ABS <= 250, ABS
assert BODY <= 3850, BODY
DST = PUB / "Paper JAMIA Submission v7.docx"
doc.save(str(DST))
print(f"saved {DST}")

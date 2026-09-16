# -*- coding: utf-8 -*-
"""Build the JAMIA submission manuscript (Research and Applications).

Derived from the JMIR v6.4 build; every number still comes from the canonical
artifacts (final_model_v6.json and companions) at build time. JAMIA format:
body up to 4,000 words, structured abstract up to 250 words (Objective,
Materials and Methods, Results, Discussion, Conclusion), up to 6 figures and
4 tables, up to 5 keywords, double-spaced. Word caps are asserted at build
time. Reference numbering is rewritten by first appearance and uncited
references are dropped; the old-to-new mapping is saved for the supplement.
"""
import json
from _build_jmir_finalsc_part1 import *  # noqa
from _build_jmir_finalsc_part1 import _style  # underscore name: explicit

# ------------------------------------------------------------- JAMIA restyle
S = doc.styles
S["Normal"].font.size = Pt(11)
S["Normal"].paragraph_format.line_spacing = 2.0
S["Normal"].paragraph_format.space_after = Pt(0)
for h, sz in (("Heading 1", 13), ("Heading 2", 11.5), ("Heading 3", 11)):
    S[h].font.size = Pt(sz)
sec0 = doc.sections[0]
for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec0, attr, Inches(1.0))

au = M["auroc"]; cv = B2["cv5_patient_grouped"]
p = CT["partitions"]
wb = FM["fairness"]["white_minus_black"]
FR = FM["fairness"]["race"]
AB = FM["fairness"]["age_band"]
st1, st2, st3 = FM["stages"]
THRV = FM["threshold_validation"]
VH = FM["validation_hierarchy"]
OOF = VH["primary_oof_procedure_dev_only"]
CCV = VH["secondary_consensus_cv_dev_only"]
BSE = FM["baselines"]
CAP = FM["capacity"]
N_ELIG = POOL5["n_eligible"]

# ------------------------------------------------- word-counted paragraph API
_COUNT = {"abstract": 0, "body": 0}
_SECTION = {"name": None, "words": {}}


def _bump(bucket, t):
    n = len(t.split())
    _COUNT[bucket] += n
    if bucket == "body" and _SECTION["name"]:
        _SECTION["words"][_SECTION["name"]] = (
            _SECTION["words"].get(_SECTION["name"], 0) + n)


def section(name):
    _SECTION["name"] = name


def PA(t, bold_prefix=None):
    """Abstract paragraph (counted against the 250-word cap)."""
    _bump("abstract", (bold_prefix or "") + t)
    return P(t, bold_prefix=bold_prefix)


def PB(t, bold_prefix=None):
    """Body paragraph (counted against the 4,000-word cap)."""
    _bump("body", (bold_prefix or "") + t)
    return P(t, bold_prefix=bold_prefix)


def HB1(t):
    _bump("body", t); H1(t)


def HB2(t):
    _bump("body", t); H2(t)


# ================================================================ TITLE PAGE
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Predicting 30-Day Hospital Readmission at Discharge: A "
              "Leakage-Safe, Calibrated Electronic Health Record Model "
              "in Medicare-Insured Adults")
r.bold = True; r.font.size = Pt(14)

# Byline as plain paragraphs (no layout table: avoids miscounting as a
# display table by portal checks or typesetters)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Thiago Bandeira, MS; Armando Gonzalez, MS; "
              "Christian Poellabauer, PhD; Ananda Mohan Mondal, PhD")
r.bold = True; r.font.size = Pt(11)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("tbati006@fiu.edu; agonz1689@fiu.edu; cpoellab@fiu.edu; "
              "amondal@cis.fiu.edu")
r.italic = True; r.font.size = Pt(9.5)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("ORCID: 0009-0006-0204-5298 (TB); 0009-0007-6777-6072 (AG); "
              "0000-0002-0599-7941 (CP); 0000-0002-4005-9942 (AMM)")
r.font.size = Pt(8.5)

for line in ("All authors: Knight Foundation School of Computing and "
             "Information Sciences, Florida International University, "
             "Miami, FL, United States",):
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(line); r.italic = True; r.font.size = Pt(10.5)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Corresponding author: Thiago Bandeira, MS (Thiago Batista "
              "Nunes Bandeira); 21324 NE 2nd Ct, Miami, FL 33179, United "
              "States; Phone: (815) 603-3286; Email: tbati006@fiu.edu")
r.font.size = Pt(9.5); r.font.name = "Times New Roman"
t = doc.add_paragraph()
r = t.add_run("Keywords: Patient Readmission; Medicare; Machine Learning; "
              "Electronic Health Records; Healthcare Disparities")
r.font.size = Pt(10)
t = doc.add_paragraph()
wc_run = t.add_run("WORDCOUNT_PLACEHOLDER")
wc_run.font.size = Pt(10)
doc.add_page_break()

# ================================================================== ABSTRACT
H1("Abstract")
PA("To develop and internally validate a leakage-safe, calibrated, "
   "interpretable model estimating the probability of 30-day all-cause "
   "within-system readmission at discharge for Medicare-insured adults "
   "from structured electronic health record (EHR) fields, and to "
   "characterize timing and subgroup performance.",
   bold_prefix="Objective: ")
PA(f"Retrospective study of the Medicare-insured subset of MIMIC-IV v3.1 "
   f"({CT['cohort_v2_admissions']:,} admissions from "
   f"{CT['cohort_v2_patients']:,} patients; readmission prevalence "
   f"{CT['label_v2_prevalence']*100:.1f}%). "
   f"All 207 candidate predictors were audited for discharge-time "
   f"availability, excluding billing-derived, race-derived, and "
   f"outcome-derived features ({N_ELIG} eligible). Feature selection and "
   f"all learned preprocessing ran inside patient-grouped folds; features "
   f"selected in at least 3 of 5 outer folds formed the consensus set of "
   f"an extreme gradient boosting (XGBoost) classifier. Uncertainty used "
   f"patient-cluster bootstrap resampling; timing used a competing-risks "
   f"reformulation.", bold_prefix="Materials and Methods: ")
PA(f"The selection procedure achieved a development-data "
   f"out-of-fold area under the receiver operating characteristic curve "
   f"(AUROC) of {OOF['auroc']:.4f} (95% CI {OOF['ci95'][0]:.4f} to "
   f"{OOF['ci95'][1]:.4f}; primary); the fixed {NF}-feature model "
   f"reached {MP['auroc']['point']:.4f} on the historically exposed test "
   f"partition with strong "
   f"calibration (slope {MP['slope']['point']:.2f}, expected calibration "
   f"error {MP['ece']['point']:.3f}). It outperformed reconstructed LACE "
   f"({BSE['lace_auroc']:.4f}) and HOSPITAL "
   f"({BSE['hospital12_auroc']:.4f}) (both "
   f"{fmt_p(max(BSE['uplift_lace_p'], BSE['uplift_hospital12_p']))}). "
   f"Time-dependent AUROC peaked on day 1 after discharge "
   f"({FM['survival']['daily_tdauc'][0]:.4f}). Discrimination was lower for "
   f"Black than for White patients ({FR['Black']['auroc']:.4f} vs "
   f"{FR['White']['auroc']:.4f}; {fmt_p(wb['p_bootstrap'])}) despite race "
   f"not being an input.", bold_prefix="Results: ")
PA("Leakage-safe validation, calibration, same-cohort baselines, timing, "
   "and subgroup auditing address gaps most readmission models leave "
   "open; the racial gap is an open limitation.",
   bold_prefix="Discussion: ")
PA(f"A parsimonious {NF}-feature model supports calibrated, interpretable "
   "readmission risk estimation at discharge; external validation and "
   "prospective evaluation are required before clinical use.",
   bold_prefix="Conclusion: ")
doc.add_page_break()

# ================================================ BACKGROUND AND SIGNIFICANCE
section("Background and Significance")
HB1("Background and Significance")
PB("Unplanned 30-day readmissions burden the United States health system: "
   "the Centers for Medicare & Medicaid Services (CMS) puts the annual "
   "direct cost of Medicare readmissions above $26 billion [1], and the "
   "Hospital Readmissions Reduction Program (HRRP) has tied reimbursement "
   "to risk-adjusted readmission performance since 2013 [1,2]. Accurate "
   "risk estimates available at the moment of discharge could target "
   "transitional-care resources, but only if the model can be trusted at "
   "deployment time. Timing and equity sharpen the stakes: transitional "
   "interventions act within days of discharge, so the timing of a return "
   "matters as much as its occurrence [37,38], and because risk scores "
   "steer resources, unequal model performance across groups becomes "
   "unequal care [15,39].")
PB("Bedside indices remain the operational default: the LACE index [3] and "
   "the HOSPITAL score [4] rarely exceed an area under the receiver "
   "operating characteristic curve (AUROC) of 0.70 in external validation "
   "[5,6], and systematic reviews report similar ceilings for "
   "regression-based models together with recurring weaknesses: "
   "leakage-prone splits, absent calibration assessment, and unexamined "
   "subgroup performance [5,7,8]. Machine learning on structured "
   "electronic health record (EHR) data can exceed these ceilings; "
   "gradient-boosted decision trees remain the strongest general-purpose "
   "learners on tabular clinical data [9-12]. Yet reported discrimination "
   "is not deployability: clinical use requires patient-level leakage "
   "prevention, calibrated absolute probabilities [13], interpretable "
   "per-patient explanations [14], same-cohort comparison with the "
   "scores hospitals actually use, fairness auditing "
   "[15,16,19,20], and reporting aligned with TRIPOD+AI [17] and "
   "PROBAST+AI [18].")
PB("Data leakage, the use of information during development that would not "
   "legitimately be available at prediction time, is increasingly "
   "recognized as a leading cause of overoptimistic and irreproducible "
   "results in machine-learning science [34]; clinical prediction is "
   "especially exposed through post-discharge billing artifacts, "
   "preprocessing fit on evaluation data, and splits that ignore patient "
   "clustering [18,35]. "
   "Roadmaps for responsible clinical machine learning call for "
   "patient-level partitioning, fold-local preprocessing, verification "
   "that every predictor is available at deployment time, and evaluation "
   "of the complete modeling procedure rather than a single exposed model "
   "[17,18,36]. Most readmission studies also collapse the 30-day horizon "
   "into one binary label, although early and late readmissions differ in "
   "causes and preventability [37,38], and rarely audit performance by "
   "subgroup [5,8,16,39].")
PB("On the Medical Information Mart for Intensive Care (MIMIC-IV) [21], "
   "the database used here, published 30-day readmission "
   "models report AUROCs of 0.791 with multimodal radiograph fusion on a "
   "selected subset [22], 0.727 with discharge notes [23], and 0.696 with "
   "26 structured features [24]; ClinicalBERT reported approximately "
   "0.714 on MIMIC-III notes [25]. None combines leakage-safe "
   "patient-grouped validation, calibration analysis, same-cohort "
   "reconstruction of LACE and HOSPITAL, readmission-timing analysis, and "
   "subgroup fairness auditing in a single evaluation. This study was "
   "designed to close that gap in the Medicare population, where the "
   "readmission penalty applies, using only predictors demonstrably "
   "available at discharge.")

# ================================================================== OBJECTIVE
section("Objective")
HB1("Objective")
PB("Four research questions (RQs) organize the study. RQ1: Which "
   "discharge-available EHR features contribute most to prediction of "
   "30-day readmission in Medicare-insured adults? RQ2: Can a "
   "leakage-safe, calibrated machine-learning model improve prediction "
   "relative to established clinical scores? RQ3: Can global and "
   "patient-level Shapley additive explanations (SHAP) provide clinically "
   "interpretable insight into predicted risk? RQ4: How does performance "
   "vary across the post-discharge time window and across demographic "
   "subgroups? Operationally, we developed and internally validated a "
   "calibrated model of 30-day all-cause within-system readmission (RQ1 "
   "to RQ3), compared it with LACE and HOSPITAL reconstructed on the "
   "identical test partition (RQ2), characterized readmission timing with "
   "competing-risks and landmark analyses (RQ4), audited subgroup "
   "performance (RQ4), and released the pipeline and a demonstration "
   "prototype.")

# ====================================================== MATERIALS AND METHODS
section("Materials and Methods")
HB1("Materials and Methods")
HB2("Study Design and Data Source")
PB("We conducted a retrospective prediction-model development and "
   "internal-validation study, reported in line with TRIPOD+AI [17] "
   "(completed checklist: Supplementary Appendix 5; PROBAST+AI "
   "self-assessment [18]: Supplementary Appendix 6). The data source is "
   "MIMIC-IV v3.1 [21,26], a deidentified EHR database of 546,028 "
   "hospitalizations between 2008 and 2022 at Beth Israel Deaconess "
   "Medical Center (Boston, MA); dates are shifted per patient with "
   "within-patient intervals preserved. "
   "Clinical notes and imaging exist as separately credentialed companion "
   "databases and were deliberately not used (Discussion).")
HB2("Cohort and Outcome Definition")
PB(f"The cohort comprised admissions with insurance recorded as Medicare "
   f"(n={CT['original_admissions']:,}). Because the model scores patients "
   f"at the moment of alive discharge, {CT['index_death_excluded']:,} "
   f"admissions (3.1%) ending in index in-hospital death were excluded, "
   f"leaving {CT['cohort_v2_admissions']:,} admissions from "
   f"{CT['cohort_v2_patients']:,} patients. Medicare insurance does not "
   f"imply age 65 or older, and fee-for-service cannot be distinguished "
   f"from Medicare Advantage, so this is not an HRRP regulatory cohort. "
   f"The outcome was the first same-system "
   f"readmission of any cause, beginning more than 0 and up to 30 days "
   f"after index discharge, identified from all subsequent "
   f"hospitalizations irrespective of their insurance or "
   f"how they ended. Next admissions beginning at, or up to 2 days "
   f"before, the index discharge timestamp "
   f"(n={CT['same_day_next_admit(delta<=0)']:,}) were treated as "
   f"transfers or administrative continuations, not "
   f"readmissions. Every index admission is eligible, so patients "
   f"contribute multiple admissions; all inference "
   f"accounts for this clustering. Planned readmissions could "
   f"not be reliably identified and were not excluded; sensitivity "
   f"analyses recount the transfer band as readmissions, exclude next "
   f"admissions typed ELECTIVE, and restrict evaluation to nonelective "
   f"index admissions (Results); full definitions: Supplementary "
   f"Methods.")
HB2("Partitions and Validation Design")
PB(f"Admissions were split 80/20 at the patient level into development "
   f"and test data, with 10% of the development portion assigned, again "
   f"patient-grouped, to a validation partition used for early stopping "
   f"and threshold derivation. After exclusions: training "
   f"{p['train']['admissions']:,} admissions ({p['train']['patients']:,} "
   f"patients, {p['train']['events']:,} events), validation "
   f"{p['val']['admissions']:,}, and test {p['test']['admissions']:,} "
   f"({p['test']['patients']:,} patients, {p['test']['events']:,} "
   f"events); no patient appears in more than one partition.")
PB("Earlier development phases scored model variants on the original test "
   "partition multiple times; the test data are therefore a historically "
   "exposed internal evaluation partition, not a pristine holdout. Three "
   "evidence tiers are reported in order of authority. (1) Primary: the "
   "out-of-fold performance of the complete selection procedure under 5 "
   "outer patient-grouped folds over the development partitions; every "
   "admission is scored by a model whose preprocessing, encodings, "
   "selection, and fitting never saw that patient. (2) Secondary: "
   "patient-grouped cross-validation of the fixed consensus set on the "
   "same development data, which aggregates selection information "
   "across folds (disclosed optimism). (3) Tertiary: a single evaluation on the "
   "historically exposed test partition, used for the detailed metric, "
   "subgroup, baseline, interpretability, and utility analyses. No "
   "test-partition outcome was used in this version's selection, "
   "encoding, consensus, refitting, or threshold steps; the model "
   "family, candidate pool, and fixed hyperparameters predate this rerun "
   "(Limitations). Hyperparameters were not tuned per fold, so the "
   "procedure is not fully nested. A temporal split was rejected because "
   "per-patient date shifting removes cross-patient ordering "
   "(era-stratified sensitivity analysis: Results). Throughout, "
   "leakage-safe denotes fold-local preprocessing, target encoding, and "
   "feature selection plus the discharge-availability predictor audit; "
   "residual design-level optimism is disclosed in Limitations.")
HB2("Predictor Audit and Leakage-Safe Feature Selection")
PB(f"All 207 candidate predictors from the engineered feature tables were "
   f"audited for provenance, measurement window, "
   f"missingness, and demonstrable availability at discharge "
   f"(Supplementary Appendix 1). Sixty-three billing-time features "
   f"(diagnosis-related groups, International Classification of Diseases "
   f"(ICD) diagnosis and procedure codes, and "
   f"their derivatives) were excluded "
   f"because their availability at the moment of discharge cannot be "
   f"established, together with one race-derived encoding and one "
   f"precomputed target encoding of discharge destination whose fold "
   f"provenance could not be aligned with the present validation design "
   f"(the raw column is retained and target-encoded strictly "
   f"fold-locally); no feature was forward-looking. The eligible pool "
   f"comprised {N_ELIG} predictors spanning demographics, admission "
   f"context, medications, laboratory values, orders, intensive care "
   f"unit (ICU) documentation, "
   f"and prior-utilization history. Feature selection used staged "
   f"recursive feature elimination with an XGBoost estimator inside 5 "
   f"outer patient-grouped folds on development data only. Within each "
   f"outer-training fold, 3 grouped inner folds drove "
   f"elimination along a prespecified grid (142 to 18 features); all "
   f"categorical target encodings were fitted from inner-training "
   f"patients only; the prespecified parsimony rule chose the smallest "
   f"feature count whose mean inner-validation AUROC lay within 0.002 of "
   f"the best. Each fold's selected model was refit on its complete "
   f"outer-training fold and scored once on its untouched "
   f"outer-evaluation fold. Features selected in at least 3 of 5 folds "
   f"formed the consensus set ({NF} features; {N_UNANIMOUS} selected in "
   f"all 5). Missing values were handled natively by the tree learners; "
   f"the logistic comparator received fold-local median imputation.")
HB2("Model and Comparators")
PB(f"The final model is a single XGBoost classifier [10] on the "
   f"{NF}-feature consensus set (600 trees, learning rate 0.05, maximum "
   f"depth 5, subsample and column subsample 0.9, "
   f"fixed seed), refit on the training partition with "
   f"training-partition encodings; its operating threshold was the "
   f"validation-partition Youden point ({THRV:.3f}), development-derived "
   f"and applied unchanged to test. "
   f"Comparators fitted per outer fold on identical data were "
   f"the full {N_ELIG}-feature eligible model, a prior 50-feature "
   f"subset, a prior 66-feature set containing billing predictors "
   f"(historical reference), and regularized logistic regression. "
   f"LightGBM [11], CatBoost, HistGradientBoosting, neural baselines "
   f"including FT-Transformer [27], and a convex blend were explored "
   f"during development; the blend improved nothing and "
   f"the boosted families spanned only 0.0035 AUROC, so a single model "
   f"was kept (Supplementary Methods).")
HB2("Clinical-Score Baselines")
PB("LACE [3] and HOSPITAL [4] were reconstructed on the identical test "
   "partition from strictly pre-discharge information, HOSPITAL with its "
   "original 12-month prior-admission window; full mappings appear in "
   "Supplementary Appendix 2. Both scores were derived for outcomes that "
   "differ from ours, so these comparisons quantify the scores' "
   "performance on this study's outcome.")
HB2("Time-to-Event and Landmark Analyses")
PB("Time to readmission was modeled with the same features and partitions "
   "under a competing-risks event process (readmission; death before "
   "readmission, observed through hospital and Massachusetts registry "
   "records; administrative censoring at day 30), with Aalen-Johansen "
   "cumulative incidence. The prognostic model is a cause-specific "
   "XGBoost accelerated failure time (AFT) model [28]. Evaluation "
   "used Harrell C [29] and cumulative/dynamic time-dependent AUROC on "
   "each of days 1 through 29, estimated with inverse probability of "
   "censoring weighting (IPCW) [30,40]; patients who died before the "
   "evaluation "
   "day remain in the comparison set as nonevents, a cause-specific "
   "competing-risks definition [40]. Three landmark models covered days "
   "1 to 7, 8 to 14, and 15 to 30 with death-aware risk sets "
   "(Supplementary Methods).")
HB2("Statistical Analysis")
PB("Because patients contribute multiple admissions, all confidence "
   "intervals (CIs) and P values use patient-level cluster bootstrap "
   "percentile resampling (1,000 resamples for the "
   "test metric panel; 5,000 for every contrast with a reported P value; "
   "two-sided empirical P values with plus-one correction, smallest "
   "reportable P<.001). Model-to-model AUROC contrasts "
   "are paired on identical admissions; the White vs Black contrast "
   "compares disjoint subgroups, resampling patients within each "
   "subgroup and testing the difference directly. Formal testing was "
   "limited to three prespecified contrasts (vs LACE, vs HOSPITAL, and "
   "the White vs Black difference); all other results are descriptive, "
   "with no multiplicity adjustment. "
   "Discrimination used AUROC and average precision; calibration "
   "used the Brier score, expected calibration error over 10 equal-width "
   "bins, and logistic recalibration slope and intercept [13]. "
   "Decision-curve analysis [31] compared net benefit against treat-all "
   "and treat-none across threshold probabilities 0.05 to 0.40, with a "
   "capacity view. Analyses used Python 3.11.14 with XGBoost 3.2.0, "
   "scikit-learn 1.7.2, pandas 2.3.3, and NumPy 1.26.4 (fixed seed 42).")
HB2("Fairness Audit")
PB("Subgroup performance was audited on the test partition across age "
   "bands, sex, and race, with race/ethnicity strings consolidated by "
   "substring mapping (raw counts: Supplementary Appendix 3). Race is "
   "not a model input; auditing by race is deliberate because not using "
   "an attribute does not guarantee equal performance across it "
   "[15,16,19]. A development-phase ablation showed that a precomputed "
   "race encoding contributed no measurable validation signal "
   "(Supplementary Appendix 4).")
HB2("Ethical Considerations")
PB("MIMIC-IV is a deidentified, publicly available database; its creation "
   "was approved by the institutional review boards of the Massachusetts "
   "Institute of Technology and Beth Israel Deaconess Medical Center "
   "with a waiver of informed consent [21,26]. The first author, the "
   "only author to access the raw data, did so under the PhysioNet "
   "Credentialed Health Data Use Agreement after required "
   "human-subjects training; co-authors worked only with aggregate "
   "results, and no row-level data are "
   "redistributed. This secondary analysis of deidentified data did not "
   "require institutional review board review at Florida International "
   "University. Prototype data safeguards appear in the Supplementary "
   "Methods.")

# ==================================================================== RESULTS
section("Results")
HB1("Results")
HB2("Cohort")
PB(f"Figure 1 shows the cohort flow. Of {CT['original_admissions']:,} "
   f"Medicare admissions, {CT['index_death_excluded']:,} ended in index "
   f"in-hospital death and were excluded, leaving "
   f"{CT['cohort_v2_admissions']:,} admissions from "
   f"{CT['cohort_v2_patients']:,} patients; {CT['label_v2_events']:,} "
   f"({CT['label_v2_prevalence']*100:.1f}%) were followed by a "
   f"within-system readmission within 30 days (mean time to readmission "
   f"among events {SV['mean_days_to_readmission']:.1f} days). This "
   f"all-cause, within-system prevalence is not comparable to HRRP "
   f"condition-specific rates.")
figure(FIG / "r2_fig1_flow_col.png",
       "Figure 1. Participant flow diagram with patient-grouped "
       "partitions; feature selection ran inside 5 patient-grouped folds "
       "drawn from the development partitions (Materials and Methods).",
       width=5.2)
HB2("Feature Selection, Stability, and Model Comparison")
_cmp = {k: CMP5[k] for k in ("rfe", "all_eligible", "f50", "f66", "logit")}
_pd_all = PAIRED["all_eligible_minus_rfe"]
_pd_f50 = PAIRED["f50_minus_rfe"]
_pd_f66 = PAIRED["f66_minus_rfe"]
_fx = EX["fixed31_vs_fixed142_paired_cv"]
PB(f"Inner-validation AUROC was nearly flat from {N_ELIG} features down "
   f"to roughly 32 in every outer fold and degraded at smaller counts "
   f"(Supplementary Figure S1); the prespecified parsimony rule selected "
   f"{', '.join(str(v) for v in STAB['fold_counts'].values())} features "
   f"across the 5 outer folds. Selection was stable (mean pairwise "
   f"Jaccard {STAB['jaccard_mean']:.3f}; {N_UNANIMOUS} features selected "
   f"in all 5 folds; Figure 2), and the consensus rule yielded the "
   f"{NF}-feature final set. Out-of-fold over the development data, the "
   f"selection procedure achieved AUROC {_cmp['rfe']['oof_auroc']:.4f} "
   f"(95% cluster CI {_cmp['rfe']['auroc_ci95_cluster'][0]:.4f} to "
   f"{_cmp['rfe']['auroc_ci95_cluster'][1]:.4f}). Paired differences on "
   f"identical observations (comparator minus procedure) were "
   f"{_pd_all['point']:+.4f} (95% CI {_pd_all['ci95'][0]:.4f} to "
   f"{_pd_all['ci95'][1]:.4f}) "
   f"for the full {N_ELIG}-feature eligible model "
   f"({_cmp['all_eligible']['oof_auroc']:.4f}) and {_pd_f50['point']:+.4f} "
   f"(95% CI {_pd_f50['ci95'][0]:.4f} to {_pd_f50['ci95'][1]:.4f}) for "
   f"the prior "
   f"50-feature subset ({_cmp['f50']['oof_auroc']:.4f}); both are smaller "
   f"than the prespecified 0.002 tolerance and indicate similar "
   f"discrimination, not statistical equivalence (fixed-set comparison: "
   f"Supplementary Results). The "
   f"prior 66-feature billing-containing set performed worst among the "
   f"boosted models under leakage-safe evaluation "
   f"({_cmp['f66']['oof_auroc']:.4f}; paired difference "
   f"{_pd_f66['point']:+.4f}, 95% CI {_pd_f66['ci95'][0]:.4f} to "
   f"{_pd_f66['ci95'][1]:.4f}), and regularized "
   f"logistic regression on the same inputs reached "
   f"{_cmp['logit']['oof_auroc']:.4f}. Because a negligible difference "
   f"does not outweigh a materially simpler set, the {NF}-feature "
   f"model was adopted; it contains no billing-derived features.")
figure(FIG5 / "v5_stability.png",
       f"Figure 2. Selection frequency across the 5 outer folds for all "
       f"{len(STAB['selection_frequency'])} features selected at least "
       "once; teal bars form the consensus set and the dashed line marks "
       "the consensus threshold (3 of 5 folds).", width=4.4)
HB2("Discrimination and Calibration of the Final Model")
PB(f"Development cross-validation of the consensus model gave AUROC "
   f"{CCV['mean']:.4f} (fold SD {CCV['sd']:.4f}). On the historically "
   f"exposed test partition ({p['test']['admissions']:,} admissions; "
   f"tertiary evidence), the refit model reached AUROC "
   f"{MP['auroc']['point']:.4f} (95% cluster CI "
   f"{MP['auroc']['ci95'][0]:.4f} to {MP['auroc']['ci95'][1]:.4f}) and "
   f"average precision {MP['ap']['point']:.3f} against a "
   f"{p['test']['events']/p['test']['admissions']*100:.1f}% prevalence. "
   f"Calibration was strong: Brier {MP['brier']['point']:.4f}, expected "
   f"calibration error {MP['ece']['point']:.3f}, slope "
   f"{MP['slope']['point']:.2f} (95% CI {MP['slope']['ci95'][0]:.2f} to "
   f"{MP['slope']['ci95'][1]:.2f}), intercept "
   f"{MP['intercept']['point']:.2f} (Figure 3). At the "
   f"development-derived threshold of {THRV:.3f}: sensitivity "
   f"{MP['sensitivity']['point']:.3f}, specificity "
   f"{MP['specificity']['point']:.3f}, positive predictive value "
   f"{MP['ppv']['point']:.3f}, and negative predictive value "
   f"{MP['npv']['point']:.3f}; the complete panel appears in "
   f"Supplementary Appendix 1.")
figure(FIG5 / "v5_roc_cal.png",
       f"Figure 3. Receiver operating characteristic, precision-recall, "
       f"calibration, and confusion-matrix panels for the final "
       f"{NF}-feature model (historically exposed test partition; "
       "tertiary evidence). Dashed lines mark chance performance and "
       "outcome prevalence.", width=6.0)
HB2("Sensitivity Analyses")
_era = EX["era_stratified_test"]["bands"]
_sa = SENS["sameday_as_readmission"]
_su = SENS["unplanned_only"]
_sn = SENS["nonelective_index_only"]
_svals = [_sa["test_auroc_refit"], _sa["test_auroc_final_model_vs_alt_label"],
          _su["test_auroc_refit"], _su["test_auroc_final_model_vs_alt_label"],
          _sn["test_auroc"], BD5["auroc_safe"]]
_smax_dev = max(abs(v - MP["auroc"]["point"]) for v in _svals)
PB(f"Five sensitivity analyses probed the outcome definition, the cohort "
   f"rules, and era composition (Supplementary Results). Recounting the "
   f"transfer/continuation band as readmissions, excluding next "
   f"admissions typed ELECTIVE, "
   f"restricting evaluation to nonelective index admissions, and "
   f"excluding admissions reaching the 2022 end of data collection all "
   f"left test AUROC within {_smax_dev:.3f} of the test estimate "
   f"(tertiary evidence; range {min(_svals):.4f} to {max(_svals):.4f}). "
   f"Stratified by the "
   f"patient-level era band, discrimination was maintained or higher in "
   f"recent eras, rising from {_era[0]['auroc']:.4f} (2008 to 2010) to "
   f"{_era[-1]['auroc']:.4f} (2020 to 2022) while observed prevalence "
   f"declined from {_era[0]['prevalence']*100:.1f}% to "
   f"{_era[-1]['prevalence']*100:.1f}%; era strata are descriptive. None "
   f"of these variations alters the study's conclusions.")
HB2("Same-Cohort Clinical Baselines")
PB(f"The final model (AUROC {MP['auroc']['point']:.4f}) outperformed "
   f"reconstructed LACE ({BSE['lace_auroc']:.4f}) by "
   f"{BSE['uplift_lace']:.4f} AUROC (95% CI "
   f"{BSE['uplift_lace_ci'][0]:.4f} to {BSE['uplift_lace_ci'][1]:.4f}; "
   f"{fmt_p(BSE['uplift_lace_p'])}; paired, 5,000 draws) and 12-month "
   f"HOSPITAL ({BSE['hospital12_auroc']:.4f}) by "
   f"{BSE['uplift_hospital12']:.4f} (95% CI "
   f"{BSE['uplift_hospital12_ci'][0]:.4f} to "
   f"{BSE['uplift_hospital12_ci'][1]:.4f}; "
   f"{fmt_p(BSE['uplift_hospital12_p'])}). Both reconstructed scores "
   f"fell below their published external validations, consistent with "
   f"within-system attenuation of their utilization inputs; because that "
   f"attenuation affects the scores but not the model's in-hospital "
   f"signals, the uplifts are best read as upper bounds for this "
   f"setting.")
HB2("Timing of Readmission and Competing Death")
_cif = SV.get("aj_cif_30d")
_cd = SV.get("test_competing_deaths")
PB(f"Under the competing-risks event process, the test partition "
   f"contributed {SV['test_events']:,} readmissions and {_cd:,} deaths "
   f"before readmission; Aalen-Johansen 30-day cumulative incidence was "
   f"{_cif['readmission']*100:.1f}% for readmission and "
   f"{_cif['death']*100:.1f}% for death without readmission. Same-date "
   f"ties were resolved death-first; a tie-rule "
   f"sensitivity analysis left every estimate essentially unchanged "
   f"(Supplementary Results). The cause-specific AFT model on the final "
   f"{NF}-feature set reached Harrell C "
   f"{FM['survival']['aft_harrell_c']:.4f}. Evaluated daily (Figure 4), "
   f"time-dependent AUROC was highest on the first day after discharge "
   f"({FM['survival']['daily_tdauc'][0]:.4f}), declined to "
   f"{FM['survival']['daily_tdauc'][6]:.4f} by day 7, and was "
   f"{FM['survival']['daily_tdauc'][-1]:.4f} at day 29. With death-aware "
   f"risk sets, landmark AUROCs were {st1['auroc']:.4f} for days 1 to 7, "
   f"{st2['auroc']:.4f} for days 8 to 14, and {st3['auroc']:.4f} for "
   f"days 15 to 30; stage-specific attribution analyses, including a "
   f"negative finding, appear in the "
   f"Supplementary Results and Supplementary Figure S2.")
figure(FIG5 / "v5_daily.png",
       "Figure 4. Time-dependent AUROC by day since discharge for the "
       f"final {NF}-feature AFT model (competing-risks labels; IPCW "
       "estimation).", width=5.4)
HB2("Fairness and Subgroup Performance")
_slopes = [(g, FM["fairness"]["race"][g]) for g in
           ["White", "Black", "Hispanic/Latino", "Asian", "Other/Unknown"]]
_smin = min(_slopes, key=lambda kv: kv[1]["cal_slope"])
_smax = max(_slopes, key=lambda kv: kv[1]["cal_slope"])
PB(f"The test partition is unevenly distributed across race categories "
   f"({FR['White']['admissions']:,} White "
   f"[{FR['White']['admissions']/p['test']['admissions']*100:.1f}%] vs "
   f"{FR['Black']['admissions']:,} Black "
   f"[{FR['Black']['admissions']/p['test']['admissions']*100:.1f}%] "
   f"admissions). Discrimination was "
   f"lower for Black than for White patients (AUROC "
   f"{FR['Black']['auroc']:.4f} vs {FR['White']['auroc']:.4f}; "
   f"difference {wb['point']:.4f}, 95% CI {wb['ci95'][0]:.4f} to "
   f"{wb['ci95'][1]:.4f}; {fmt_p(wb['p_bootstrap'])}, stratified "
   f"subgroup bootstrap) and varied across age bands, from "
   f"{AB['<65']['auroc']:.4f} among patients younger than 65 "
   f"years, largely disability-entitled Medicare, to "
   f"{AB['65-74']['auroc']:.4f} for ages 65 to 74, with similar "
   f"performance by sex (Figure 5, Table 1). Among the subgroup "
   f"comparisons, only the White vs Black contrast was prespecified and "
   f"formally tested; between-group differences partly reflect case mix "
   f"and outcome prevalence. At the shared threshold, specificity was "
   f"{FR['Black']['specificity']:.3f} for Black vs "
   f"{FR['White']['specificity']:.3f} for White patients, so a larger "
   f"share of nonreadmitted Black patients is flagged; under a "
   f"deployment posture that pairs flags with additional supportive "
   f"resources rather than reduced care (Discussion), this directs more "
   f"outreach, not less, but it is an operational disparity to "
   f"monitor. Race-group calibration slopes ranged from "
   f"{_smin[1]['cal_slope']:.2f} ({_smin[0]}) to "
   f"{_smax[1]['cal_slope']:.2f} ({_smax[0]}) (Table 1). Race is not a "
   f"model input, and removing a legacy race encoding did not close the "
   f"gap (Materials and Methods); because the disparity is one of "
   f"ranking, subgroup-specific recalibration cannot repair it, and "
   f"remedies such as representation-aware training are not claimed "
   f"here. Because the outcome is within-system, subgroup differences in "
   f"out-of-system readmission could contribute to, or mask, the "
   f"observed gap; external validation with claims-complete follow-up "
   f"is the appropriate check.")


def _ci3(e, key):
    c = e.get(f"{key}_ci95")
    v = e.get(key)
    if v is None:
        return "NA"
    if key == "cal_slope":
        return f"{v:.2f} ({c[0]:.2f} to {c[1]:.2f})" if c else f"{v:.2f}"
    if key == "auroc":
        return f"{v:.4f} ({c[0]:.4f} to {c[1]:.4f})" if c else f"{v:.4f}"
    return f"{v:.3f} ({c[0]:.3f} to {c[1]:.3f})" if c else f"{v:.3f}"


rows = [["Group", "Patients / admissions / events", "Prev.",
         "AUROC (95% CI)", "Sensitivity (95% CI)", "FNR (95% CI)",
         "PPV", "Brier", "Cal. slope (95% CI)"]]
for axis, order_ in (("race", ["White", "Black", "Hispanic/Latino", "Asian",
                               "Other/Unknown"]),
                     ("sex", ["Female", "Male"]),
                     ("age_band", ["<65", "65-74", "75-84", "85+"])):
    for g in order_:
        e = FM["fairness"][axis][g]
        rows.append([g.replace("-", " to "),
                     f"{e['patients']:,} / {e['admissions']:,} / "
                     f"{e['events']:,}",
                     f"{e['events']/e['admissions']*100:.1f}%",
                     _ci3(e, "auroc"), _ci3(e, "sensitivity"),
                     _ci3(e, "fnr"),
                     f"{e['ppv']:.3f}" if e.get("ppv") is not None else "NA",
                     f"{e['brier']:.4f}" if e.get("brier") is not None
                     else "NA",
                     _ci3(e, "cal_slope")])
caption("Table 1. Subgroup performance of the final model on the test "
        f"partition at the development-derived threshold ({THRV:.3f}). CIs "
        "are 95% patient-cluster bootstrap percentile intervals; PPV and "
        "Brier are point estimates here, and the complete metric set "
        "appears in Supplementary Appendix 1. AUROC: area under the "
        "receiver operating characteristic curve; FNR: false-negative "
        "rate; PPV: positive predictive value; Prev.: outcome prevalence; "
        "Cal.: calibration. Other/Unknown is heterogeneous and is not "
        "interpreted; patients whose recorded race varies across "
        "admissions appear in more than one race row.",
        keep_with_next=True)
table(rows, font_size=8,
      widths=[0.72, 1.12, 0.45, 0.95, 0.95, 0.9, 0.45, 0.45, 0.79])
figure(FIG5 / "v5_fairness.png",
       f"Figure 5. Subgroup AUROC of the final model with 95% "
       f"patient-cluster bootstrap CIs; the dashed line marks the overall "
       f"test AUROC ({MP['auroc']['point']:.4f}).", width=5.4)
HB2("Interpretability via SHAP")
_t3 = [f"{flabel(s['feature'])} ({s['feature']}; mean |SHAP| "
       f"{s['mean_abs_shap']:.3f})" for s in SH[:3]]
_top3 = f"{_t3[0]}, {_t3[1]}, and {_t3[2]}"
_n4 = [flabel(s["feature"]) for s in SH[3:7]]
_next4 = f"{_n4[0]}, {_n4[1]}, {_n4[2]}, and {_n4[3]}"
PB(f"Figure 6 ranks the final model's predictors by mean absolute SHAP "
   f"value. The strongest signals are {_top3}; "
   f"{_next4} complete the top seven. Readmission risk is multifactorial "
   f"(no single feature dominates), and the leading features map to "
   f"observables a discharge team can see: an escalating admission "
   f"pattern, where the patient is going next, and unresolved laboratory "
   f"abnormalities. Per-patient additive SHAP decompositions are "
   f"produced at serving time by the released prototype.")
figure(FIG5 / "v5_shap.png",
       f"Figure 6. Top 10 predictors of the final {NF}-feature model by "
       "mean absolute SHAP value (historically exposed test partition).",
       width=5.4)
HB2("Clinical Utility (Exploratory)")
PB(f"Decision-curve analysis showed positive net benefit over treat-all "
   f"and treat-none across the evaluated threshold range of 0.05 to "
   f"0.40, a range not yet grounded in operational evidence "
   f"(Supplementary Appendix 4). Flagging "
   f"the top 5% of test admissions captured "
   f"{CAP['top_5']['captured']*100:.0f}% of all readmissions at a "
   f"positive predictive value of {CAP['top_5']['ppv']:.2f}; the top 10% "
   f"captured {CAP['top_10']['captured']*100:.0f}% at "
   f"{CAP['top_10']['ppv']:.2f}; the top 20% captured "
   f"{CAP['top_20']['captured']*100:.1f}% at {CAP['top_20']['ppv']:.2f}. "
   f"Whether acting on these flags improves outcomes "
   f"requires prospective evaluation. Because patients who die without "
   f"readmission are nonevents, a low readmission risk must not be read "
   f"as clinical stability; deployment should surface competing "
   f"mortality risk alongside the score.")

# ================================================================= DISCUSSION
section("Discussion")
HB1("Discussion")
HB2("Principal Findings")
PB(f"Four contributions carry this study. In "
   f"{CT['cohort_v2_admissions']:,} Medicare-insured adult admissions, a "
   f"{NF}-feature gradient-boosted model, selected from "
   f"demonstrably discharge-available, nonbilling predictors "
   f"under patient-grouped, fold-local preprocessing and feature "
   f"selection, estimated 30-day within-system readmission with strong "
   f"calibration; the selection procedure achieved a development-data "
   f"out-of-fold AUROC of {OOF['auroc']:.4f} (primary estimate) and the "
   f"fixed model {MP['auroc']['point']:.4f} on the historically exposed "
   f"test partition. First, the parsimonious procedure showed no "
   f"material decrement against the full eligible pool under "
   f"leakage-safe comparison, while the previous billing-containing "
   f"model performed worse: most of the usable signal is captured by a "
   f"compact, operationally available predictor set. Second, "
   f"discrimination is highest for readmissions occurring during the "
   f"first day after discharge, the window in which a "
   f"discharge-initiated intervention must act (window-level landmark "
   f"discrimination, a different estimand, was lowest for days 1 to 7; "
   f"Results). Third, "
   f"the model performs less well for Black patients despite race not "
   f"being an input, a disparity quantified directly and reported as an "
   f"open limitation. Fourth, the model clearly outperforms the "
   f"reconstructed LACE and HOSPITAL scores on the identical cohort and "
   f"outcome, with calibrated absolute risks the bedside scores do not "
   f"provide. These findings answer RQ1 (leading features: the 180-day "
   f"length-of-stay trend and prior utilization; Figure 6), RQ2 "
   f"(improvement over clinical scores), RQ3 (per-patient SHAP output), "
   f"and RQ4 (timing and subgroup variation).")
HB2("Comparison With Prior Work")
PB(f"On the same database, published models report 0.791 using chest "
   f"radiographs on a selected 14,532-admission subset [22], 0.727 "
   f"using discharge notes [23], and 0.696 using 26 structured features "
   f"[24]. The selection procedure achieved {OOF['auroc']:.4f} out of "
   f"fold, and the fixed {NF}-feature model {MP['auroc']['point']:.4f} "
   f"on test, numerically above the notes-based and structured-feature "
   f"reports and near the multimodal result while scoring every "
   f"admission. Because "
   f"cohorts, outcome definitions, modalities, and validation designs "
   f"differ, these are reference points and no superiority claim is "
   f"made; the controlled comparison is the same-cohort baseline "
   f"analysis (Table 2). MIMIC-IV's clinical notes and radiographs "
   f"[32,33] were excluded by design: structured fields are available "
   f"in any EHR at discharge, discharge summaries are authored at or "
   f"after discharge and risk importing the outcome into the features, "
   f"and restricting to imaged admissions would shrink and bias the "
   f"cohort. Whether unstructured modalities can raise the "
   f"structured-data ceiling is a planned follow-up study.")
_t2 = [["Study / score", "Data", "AUROC", "Notes"],
       ["LACE (reconstructed) [3]",
        "Same cohort: this study's MIMIC-IV Medicare test partition",
        f"{BSE['lace_auroc']:.4f}", "Controlled: same patients, same outcome"],
       ["HOSPITAL (reconstructed, 12-mo) [4]",
        "Same cohort: this study's MIMIC-IV Medicare test partition",
        f"{BSE['hospital12_auroc']:.4f}",
        "Controlled: same patients, same outcome"],
       ["Adisa 2026 [24]", "Same database: MIMIC-IV (all-adult, 415,231)",
        "0.696", "26 structured features; different cohort/definition"],
       ["Almeida 2025 [23]", "Same database: MIMIC-IV (all-adult, 303,571)",
        "0.727", "Discharge notes + graph; different cohort/definition"],
       ["Tang 2023 [22]",
        "Same database: MIMIC-IV (radiograph-selected, 14,532)",
        "0.791", "Multimodal, chest X-rays; imaging-selected subset"],
       ["ClinicalBERT [25]", "Different database: MIMIC-III", "≈0.714",
        "BERT over discharge notes; different era"],
       ["This study: selection procedure",
        f"MIMIC-IV Medicare, development partitions "
        f"({CMP5['_design']['n_dev']:,} admissions)",
        f"{OOF['auroc']:.4f} (OOF)",
        "Primary; fold-specific sets of 27 to 38 features; leakage-safe, "
        "out of fold"],
       [f"This study: fixed consensus-{NF}, development CV",
        "MIMIC-IV Medicare, development partitions",
        f"{CCV['mean']:.4f}",
        "Secondary; fixed set aggregates selection information across "
        "folds (optimism disclosed)"],
       [f"This study: fixed consensus-{NF}, test",
        f"MIMIC-IV Medicare, test partition "
        f"({p['test']['admissions']:,} admissions)",
        f"{MP['auroc']['point']:.4f}",
        "Tertiary; historically exposed partition, not used in this "
        "version's RFE or consensus construction"]]
caption("Table 2. Comparison with clinical scores and published MIMIC "
        "readmission models. The Data column marks which comparators share "
        "this study's cohort (controlled) or database (closely related); "
        "published values from differing cohorts and outcome definitions "
        "are reference points, not head-to-head results. This study's "
        "three rows are different estimands and are labeled by evidence "
        "tier. OOF: out-of-fold.", keep_with_next=True)
table(_t2, font_size=8.5, widths=[1.5, 2.0, 0.9, 1.8])
HB2("Clinical Interpretation of the Leading Predictors (Hypotheses)")
assert SH[0]["feature"] == "los_trend_180d", \
    f"clinical text assumes los_trend_180d leads; got {SH[0]['feature']}"
PB("Throughout, driver means contribution to the model's prediction, not "
   "a demonstrated causal effect. The 180-day length-of-stay trend, the "
   "strongest signal, is best read as a compressed biomarker of disease "
   "trajectory: when recent admissions grow progressively longer, "
   "stabilization is taking longer and functional reserve at separation "
   "is falling; a rising trend could prompt pre-discharge geriatric or "
   "palliative-care consultation, though the feature is partly "
   "system-shaped by delayed "
   "discharges. The prior-utilization cluster captures the "
   "dose-response between recent hospital use and near-term return "
   "documented repeatedly in the readmission literature [5,7]; because "
   "prior utilization partly reflects outpatient-access barriers and "
   "social instability, pairing the score with additional resources, "
   "never reduced care, is the only defensible deployment posture [15]. "
   "The abnormal-laboratory rate is the clearest physiological signal: "
   "leaving the hospital with substantial unresolved laboratory "
   "derangement marks post-discharge vulnerability. Together these "
   "predictors indicate that readmission risk reflects both the "
   "patient's trajectory and the system's response; the intervention "
   "points the model identifies are system-level processes, so clinical "
   "utility depends on whether they can be acted on, not on "
   "discrimination alone.")
HB2("Limitations")
PB(f"This is a single-center, retrospective, internally validated study "
   f"whose Medicare case "
   f"mix and demographic composition "
   f"({FR['White']['admissions']/p['test']['admissions']*100:.1f}% of "
   f"test admissions White) differ from national Medicare; performance "
   f"may not transfer without recalibration, and no clinical-impact "
   f"evidence exists. The outcome is within-system readmission; "
   f"out-of-system readmissions are unobservable. Post-discharge death "
   f"is censored 1 year after each patient's last discharge, and "
   f"registry death dates carry day-level "
   f"granularity. Planned readmissions were not excluded from the "
   f"primary outcome, and end-of-record censoring cannot be verified "
   f"under date shifting. Hyperparameters were fixed and not tuned "
   f"inside the selection folds, so the procedure is not fully nested "
   f"and residual selection optimism cannot be excluded; the consensus "
   f"set aggregates across folds and inherits mild optimism "
   f"relative to the procedure estimate. The validation partition "
   f"participated in the consensus-selection folds and supplied the "
   f"operating threshold, so it is not independent of model selection. "
   f"The test partition carries historical exposure from earlier "
   f"development phases: the model family, candidate pool, fixed "
   f"hyperparameters, and researcher decisions taken after observing "
   f"earlier test results predate this version, so residual indirect "
   f"influence on the tertiary estimate cannot be excluded. Bootstrap "
   f"intervals quantify "
   f"evaluation-sample uncertainty conditional on the fitted models and "
   f"do not capture retraining or reselection variability. Secular "
   f"drift across 2008 to 2022 cannot be fully assessed under date "
   f"shifting; the era-stratified analysis is reassuring but coarse. "
   f"Social determinants of health are unavailable in MIMIC-IV. The "
   f"subgroup disparity for Black patients is unexplained and "
   f"unresolved and, like all subgroup analyses here, was estimated on "
   f"the historically exposed test partition. The released prototype is "
   f"a research demonstration, and no claim is made that the model "
   f"improves outcomes.")

# ================================================================= CONCLUSION
section("Conclusion")
HB1("Conclusion")
PB(f"A parsimonious {NF}-feature XGBoost model, selected from "
   f"demonstrably discharge-available, nonbilling "
   f"structured EHR predictors under patient-grouped, fold-local "
   f"preprocessing and feature selection, estimates 30-day "
   f"within-system readmission in Medicare-insured adults with strong "
   f"calibration and per-patient SHAP explanations; the selection "
   f"procedure achieved a development-data out-of-fold AUROC of "
   f"{OOF['auroc']:.4f} and the fixed model {MP['auroc']['point']:.4f} "
   f"on the historically exposed test partition, clearly exceeding the "
   f"reconstructed bedside scores on the identical cohort. "
   f"Discrimination was highest for events occurring during "
   f"the first day after discharge, and the subgroup audit surfaced "
   f"lower discrimination for Black patients, reported openly as a "
   f"limitation. External validation and prospective evaluation are the "
   f"necessary next steps before any clinical use.")

# ================================================================ BACK MATTER
H1("Acknowledgments")
P("This work began as the capstone project for the Master of Science in "
  "Data Science and Artificial Intelligence at Florida International "
  "University. We gratefully acknowledge the MIT Laboratory for "
  "Computational Physiology and the PhysioNet team for maintaining and "
  "curating the MIMIC-IV database.")
H1("Conflict of Interest Statement")
P("None declared.")
H1("Funding")
P("This study received no external funding.")
H1("Data Availability Statement")
P("MIMIC-IV v3.1 is available from PhysioNet under a credentialed data "
  "use agreement [21,26] and is not redistributed. All analysis code, "
  "the cohort-construction and reanalysis scripts, the machine-readable "
  "results bundle from which the values in this manuscript are generated "
  "(final_model_v6.json and companion files), the predictor dictionary, "
  "and the prototype source are available in the project repositories: "
  "https://github.com/thiagobandeira1/Medicare-30day-Readmission-MIMIC-IV, "
  "https://github.com/thiagobandeira1/readmission-risk-api, and "
  "https://github.com/thiagobandeira1/riskpath-clinician-companion "
  "(tagged release v6.4-submission). An archival copy of the tagged "
  "release of the primary repository is deposited at Zenodo "
  "(DOI: 10.5281/zenodo.21987702).")
H1("Author Contributions")
P("Conceptualization: TB; methodology: TB, AG, CP, AMM; software, formal "
  "analysis, and visualization: TB, AG; data curation: TB, AG; "
  "investigation and validation: TB, AG; writing (original draft): TB; "
  "writing (review and editing): AG, CP, AMM; supervision: CP, AMM. All "
  "authors approved the final version. (CRediT taxonomy.)")
H1("Supplementary Material")
P(f"Supplementary Material (single file): Supplementary Methods (full "
  f"cohort and outcome definitions, partition and data-flow detail, "
  f"selection, baseline, time-to-event, statistical, and prototype "
  f"detail), Supplementary Results (sensitivity, payer-scope, tie-rule, "
  f"and landmark analyses), and Supplementary Figures S1 and S2. "
  f"Supplementary Appendix 1: predictor audit and selection workbook "
  f"(207-candidate audit, eligible pool, final {NF}-feature dictionary, "
  f"RFE trajectories, out-of-fold model comparison, full subgroup "
  f"results with CIs, outcome-sensitivity analyses, daily "
  f"competing-risk event counts, stage-differential estimates). "
  f"Supplementary Appendix 2: LACE and HOSPITAL reconstruction tables. "
  f"Supplementary Appendix 3: race-category consolidation with raw "
  f"counts. Supplementary Appendix 4: supplementary figures and "
  f"development history. Supplementary Appendix 5: completed TRIPOD+AI "
  f"checklist. Supplementary Appendix 6: PROBAST+AI self-assessment.")

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
]


# ---- build-time citation renumbering by first appearance -------------------
def _renumber_citations(doc, refs):
    import re as _re3
    pat = _re3.compile(r"\[([0-9][0-9,\-]*)\]")

    def parse(group):
        out = []
        for part in group.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                if not (a.isdigit() and b.isdigit()):
                    return None
                out.extend(range(int(a), int(b) + 1))
            elif part.isdigit():
                out.append(int(part))
            else:
                return None
        if any(n < 1 or n > len(refs) for n in out):
            return None
        return out

    def paragraphs():
        for par in doc.paragraphs:
            yield par
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        yield par

    order, seen = [], set()
    for par in paragraphs():
        for mm in pat.finditer(par.text):
            nums = parse(mm.group(1))
            if nums is None:
                continue
            for n in nums:
                if n not in seen:
                    seen.add(n)
                    order.append(n)
    n_cited = len(order)
    for n in range(1, len(refs) + 1):
        if n not in seen:
            print(f"  UNCITED (dropped): [{n}] {refs[n-1][:70]}")
            order.append(n)
    mapping = {old: new for new, old in enumerate(order, 1)}

    def rewrite(group):
        nums = parse(group)
        if nums is None:
            return f"[{group}]"
        new = sorted(mapping[n] for n in nums)
        parts, i = [], 0
        while i < len(new):
            j = i
            while j + 1 < len(new) and new[j + 1] == new[j] + 1:
                j += 1
            parts.append(str(new[i]) if j == i else
                         (f"{new[i]},{new[j]}" if j == i + 1 else
                          f"{new[i]}-{new[j]}"))
            i = j + 1
        return "[" + ",".join(parts) + "]"

    for par in paragraphs():
        if not pat.search(par.text):
            continue
        for run in par.runs:
            if pat.search(run.text):
                run.text = pat.sub(lambda mm: rewrite(mm.group(1)), run.text)
    new_refs = [refs[old - 1] for old in order]
    seen2, mx = set(), 0
    for par in paragraphs():
        for mm in pat.finditer(par.text):
            nums = parse(mm.group(1))
            if nums is None:
                continue
            for n in nums:
                if n not in seen2:
                    seen2.add(n)
                    assert n >= mx, f"citation order broken at [{n}]"
                    mx = max(mx, n)
    return new_refs, mapping, n_cited


REFS, MAPPING, N_CITED = _renumber_citations(doc, REFS)
REFS = REFS[:N_CITED]
(Path(__file__).resolve().parent / "_jamia_ref_mapping.json").write_text(
    json.dumps({"note": "keys are the BUILDER-INTERNAL source numbering of "
                        "the REFS list in _build_jamia.py (identical to the "
                        "pre-renumber REFS order in the JMIR part2 builder), "
                        "NOT the renumbered JMIR dump",
                "mapping": MAPPING, "n_cited": N_CITED}, indent=1))

for i, ref in enumerate(REFS, 1):
    P(f"{i}. {ref}")

# ------------------------------------------------------- word counts and save
BODY = _COUNT["body"]
ABS = _COUNT["abstract"]
wc_run.text = _style(
    f"Word count: {BODY:,} (main text, excluding title page, abstract, "
    f"references, figures, and tables); abstract: {ABS} words.")
print("--- section word counts ---")
for k, v in _SECTION["words"].items():
    print(f"  {k:28s} {v:5d}")
print(f"BODY {BODY} (cap 4000) | ABSTRACT {ABS} (cap 250) | "
      f"REFS {len(REFS)} (cited {N_CITED})")
assert ABS <= 250, f"abstract over cap: {ABS}"
assert BODY <= 4000, f"body over cap: {BODY}"

DST = PUB / "Paper JAMIA Submission FINAL.docx"
doc.save(str(DST))
print(f"saved {DST} ({DST.stat().st_size:,} bytes)")

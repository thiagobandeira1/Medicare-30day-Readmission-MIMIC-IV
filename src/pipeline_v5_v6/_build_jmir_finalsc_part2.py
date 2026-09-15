# -*- coding: utf-8 -*-
"""Build the complete revised JMIR AI manuscript (v6.4: Armando's audit fixes
— repaired Figures 1-2, verified numeric/citation defects, AMA style pass,
back-matter reorder, Methods trim). Run this file."""
from _build_jmir_finalsc_part1 import *  # noqa

au = M["auroc"]; cv = B2["cv5_patient_grouped"]
p = CT["partitions"]
wb = FM["fairness"]["white_minus_black"]
FR = FM["fairness"]["race"]
AB = FM["fairness"]["age_band"]
lace, hosp = BL["lace"], BL["hospital"]
st1, st2, st3 = FM["stages"]
THRV = FM["threshold_validation"]
VH = FM["validation_hierarchy"]
OOF = VH["primary_oof_procedure_dev_only"]
CCV = VH["secondary_consensus_cv_dev_only"]
BSE = FM["baselines"]
CAP = FM["capacity"]
N_EXCL = len(POOL5["excluded"])          # 65: 63 billing + 1 race + 1 leaked TE
N_ELIG = POOL5["n_eligible"]             # 142

# ============================================================ TITLE PAGE
# Front matter reproduced in the authors' original layout: centered bold title,
# italic subtitle, four-column author/email row, school and date lines.
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Predicting 30-Day Hospital Readmission at Discharge: A "
              "Leakage-Safe, Calibrated Electronic Health Record Model "
              "in Medicare-Insured Adults")
r.bold = True; r.font.size = Pt(15)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Retrospective Development and Internal Validation of an "
              "Interpretable Gradient-Boosting Model on MIMIC-IV v3.1, "
              "With Readmission-Timing and Subgroup-Fairness Analyses")
r.italic = True; r.font.size = Pt(11.5)

# Confirmed author roster: degrees (2026-08-16) and registry-verified ORCIDs
AUTHORS = [("Thiago Bandeira, MS", "tbati006@fiu.edu", "0009-0006-0204-5298"),
           ("Armando Gonzalez, MS", "agonz1689@fiu.edu", "0009-0007-6777-6072"),
           ("Christian Poellabauer, PhD", "cpoellab@fiu.edu",
            "0000-0002-0599-7941"),
           ("Ananda Mohan Mondal, PhD", "amondal@cis.fiu.edu",
            "0000-0002-4005-9942")]
atab = doc.add_table(rows=1, cols=4)      # default style: no borders
for j, (name, mail, orcid) in enumerate(AUTHORS):
    cell = atab.cell(0, j)
    p1 = cell.paragraphs[0]; p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p1.add_run(name); r.bold = True; r.font.size = Pt(10.5)
    r.font.name = "Times New Roman"
    p2 = cell.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p2.add_run(mail); r.italic = True; r.font.size = Pt(9.5)
    r.font.name = "Times New Roman"
    p3 = cell.add_paragraph(); p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p3.add_run("ORCID " + orcid); r.font.size = Pt(8)
    r.font.name = "Times New Roman"

# manuscript date removed per JMIR submission requirements (audit item 3)
for line in ("Knight Foundation School of Computing and Information Sciences",
             "Florida International University, Miami, FL, United States"):
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(line); r.italic = True; r.font.size = Pt(10.5)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Corresponding author: Thiago Bandeira, MS (Thiago Batista "
              "Nunes Bandeira); 21324 NE 2nd Ct, Miami, FL 33179, United "
              "States; Phone: (815) 603-3286; Email: tbati006@fiu.edu")
r.font.size = Pt(9); r.font.name = "Times New Roman"
begin_two_columns()

# ============================================================ ABSTRACT
H1("Abstract")
P("Unplanned hospital readmission within 30 days is a quality measure tied to "
  "financial penalties under the Centers for Medicare & Medicaid Services Hospital "
  "Readmissions Reduction Program. Bedside scores such as LACE and "
  "HOSPITAL rarely exceed an area under the receiver operating characteristic curve "
  "(AUROC) of 0.70 in external validation. Although machine-learning "
  "approaches may improve discrimination, many studies have limited "
  "assessment of data leakage, calibration, readmission timing, "
  "comparison with established "
  "scores in the same cohort, and subgroup performance.",
  bold_prefix="Background: ")
P("To develop and internally validate an interpretable, calibrated model that "
  "estimates the probability of 30-day all-cause within-system readmission "
  "at hospital discharge for Medicare-insured adults, using only structured electronic "
  "health record (EHR) data, and to characterize when patients return.",
  bold_prefix="Objective: ")
P(f"We conducted a retrospective study of the Medicare-insured subset of MIMIC-IV "
  f"v3.1. After excluding {CT['index_death_excluded']:,} index in-hospital "
  f"deaths, the cohort comprised {CT['cohort_v2_admissions']:,} admissions "
  f"from {CT['cohort_v2_patients']:,} patients (readmission prevalence "
  f"{CT['label_v2_prevalence']*100:.1f}%). The outcome was first same-system "
  f"readmission 0<t≤30 days after discharge. Partitions were "
  f"patient-grouped. All 207 candidate predictors were "
  f"audited for discharge-time availability; billing-derived, race-derived, and "
  f"outcome-derived features were excluded ({N_ELIG} eligible). "
  f"Feature selection and all learned preprocessing were performed within "
  f"patient-grouped cross-validation folds to prevent information leakage: "
  f"staged recursive feature elimination ran inside 5 outer folds on the "
  f"development partitions only, and features selected in at least 3 of 5 folds formed "
  f"the consensus set of the final extreme gradient boosting (XGBoost) "
  f"classifier. The prespecified primary performance estimate was the "
  f"out-of-fold discrimination of the complete selection procedure. "
  f"Uncertainty used patient-cluster bootstrap resampling. Timing was "
  f"assessed with a competing-risks reformulation and landmark models, "
  f"and subgroup performance was audited by age, sex, and race.",
  bold_prefix="Methods: ")
_fx = EX["fixed31_vs_fixed142_paired_cv"]
P(f"The selection procedure, in which each outer fold selected its own "
  f"27 to 38 predictors, achieved a development-data out-of-fold AUROC of {OOF['auroc']:.4f} "
  f"(95% CI {OOF['ci95'][0]:.4f} to {OOF['ci95'][1]:.4f}; primary). The "
  f"fixed {NF}-feature consensus model reached {CCV['mean']:.4f} "
  f"in development cross-validation (secondary; SD {CCV['sd']:.4f}) and "
  f"{MP['auroc']['point']:.4f} (95% CI {MP['auroc']['ci95'][0]:.4f} to "
  f"{MP['auroc']['ci95'][1]:.4f}) on the historically exposed test "
  f"partition (tertiary). Calibration was "
  f"strong (slope {MP['slope']['point']:.2f}, expected calibration error "
  f"{MP['ece']['point']:.3f}). The model outperformed reconstructed LACE "
  f"({BSE['lace_auroc']:.4f}) by {BSE['uplift_lace']:.4f} and 12-month "
  f"HOSPITAL ({BSE['hospital12_auroc']:.4f}) by "
  f"{BSE['uplift_hospital12']:.4f} AUROC (both "
  f"{fmt_p(max(BSE['uplift_lace_p'], BSE['uplift_hospital12_p']))}). "
  f"The 180-day length-of-stay trend and prior utilization led SHAP "
  f"attributions. "
  f"Time-dependent AUROC was highest on the first day after discharge "
  f"({FM['survival']['daily_tdauc'][0]:.4f}). Discrimination was lower for "
  f"Black patients (AUROC {FR['Black']['auroc']:.4f}) than for White patients "
  f"({FR['White']['auroc']:.4f}; difference {wb['point']:.3f}, 95% CI "
  f"{wb['ci95'][0]:.3f} to {wb['ci95'][1]:.3f}; "
  f"{fmt_p(wb['p_bootstrap'])}) despite race not being a model input.",
  bold_prefix="Results: ")
P(f"A parsimonious {NF}-feature XGBoost model, selected from demonstrably "
  "discharge-available, nonbilling predictors under patient-grouped, fold-local "
  "preprocessing and feature selection, supports calibrated and interpretable "
  "estimation of 30-day within-system readmission risk in Medicare-insured "
  "adults. This is a research model; external validation and prospective "
  "evaluation are required before clinical use.",
  bold_prefix="Conclusions: ")
H2("Keywords")
P("hospital readmission; Medicare; MIMIC-IV; machine learning; gradient "
  "boosting; data leakage; calibration; algorithmic fairness; survival "
  "analysis; clinical prediction model; electronic health records")

# ============================================================ INTRODUCTION
H1("Introduction")
H2("Background")
P("Unplanned 30-day readmissions burden the United States health system; the Centers "
  "for Medicare & Medicaid Services (CMS) puts the annual direct cost of "
  "Medicare readmissions above $26 billion [1], and the Hospital Readmissions "
  "Reduction Program (HRRP) has tied reimbursement to risk-adjusted "
  "readmission performance since 2013 [1,2]. Readmissions are associated with in-hospital mortality, "
  "functional decline, and caregiver strain. Accurate risk estimates "
  "available at the moment of discharge could target transitional-care "
  "resources, but only if the underlying model can be trusted at "
  "deployment time. Timing and equity sharpen the stakes: transitional "
  "interventions act within days of discharge, so the timing of a return "
  "matters as much as its occurrence, and because risk scores steer "
  "resources, unequal model performance across groups becomes unequal "
  "care.")


def H3(t):
    doc.add_paragraph(t, style="Heading 3")


H2("Related Work")
H3("Clinical Scores and Regression-Based Models")
P("Bedside indices remain the operational default for readmission risk: the "
  "LACE index combines length of stay, acuity, comorbidity, and "
  "emergency-department use [3], and the HOSPITAL score targets potentially "
  "avoidable readmission [4]. External validations of such scores rarely "
  "exceed an area under the receiver operating characteristic curve "
  "(AUROC) of 0.70 [5,6], and systematic reviews covering hundreds "
  "of readmission models report similar ceilings for regression-based "
  "approaches together with recurring methodological weaknesses: "
  "leakage-prone splits, absent calibration assessment, and unexamined "
  "subgroup performance [5,7,8].")
H3("Machine Learning for Readmission Prediction")
P("Machine learning on structured electronic health record (EHR) data can "
  "exceed these ceilings, and "
  "gradient-boosted decision trees remain the strongest general-purpose "
  "learners on tabular clinical data [9-12]. Yet reported discrimination is "
  "not deployability: clinical use requires patient-level leakage "
  "prevention, calibrated absolute probabilities [13], interpretable "
  "per-patient explanations [14], comparison with clinical scores on the "
  "same cohort rather than against published values, fairness auditing "
  "[15,16], and reporting aligned with TRIPOD+AI [17] and PROBAST [18]. "
  "Race-aware modeling requires particular care following the removal of "
  "race coefficients from clinical equations such as the estimated "
  "glomerular filtration rate [19,20].")
H3("Data Leakage and Validation Pitfalls")
P("Data leakage, the use of information during model development that "
  "would not legitimately be available at prediction time, is increasingly "
  "recognized as a leading cause of overoptimistic and irreproducible "
  "results across machine-learning science [34]. Clinical prediction is "
  "especially exposed: features generated after the prediction moment (for "
  "example, billing codes assigned during claims processing), "
  "preprocessing fit on evaluation data, and splits that ignore patient "
  "clustering all inflate apparent performance, and single-site results "
  "often fail to generalize [18,35]. Roadmaps for responsible clinical "
  "machine learning therefore call for patient-level partitioning, "
  "fold-local preprocessing, verification that every predictor is "
  "available at deployment time, and evaluation of the complete modeling "
  "procedure rather than a single exposed model [17,18,36]. These "
  "recommendations directly shaped this study's validation design.")
H3("Readmission Timing and Subgroup Performance")
P("When patients return matters as much as whether they return: early and "
  "late readmissions differ in causes and preventability [37], the first "
  "days after discharge carry a generalized transient vulnerability [38], "
  "and transitional-care interventions act on a limited window, yet most "
  "prediction studies collapse the 30-day horizon into a single binary "
  "label and report no timing analysis. Subgroup performance is similarly "
  "underexamined: audits of deployed clinical algorithms have found racial "
  "bias in resource-allocation scores [15] and systematic underdiagnosis "
  "of underserved groups by imaging models [39], gaps can persist even "
  "when protected attributes are not model inputs [16], and readmission "
  "models are rarely audited by subgroup [5,8]. The timing and subgroup "
  "analyses in this study address both gaps.")
H3("Prior Work on MIMIC")
P("Three recent studies predict 30-day readmission on MIMIC-IV, the "
  "database used in this study "
  "[21]. A multimodal spatiotemporal graph neural network combining EHR time series "
  "with chest radiographs reported an AUROC of 0.791 on a radiograph-selected subset "
  "of 14,532 admissions [22]; a graph model over discharge summaries reported 0.727 "
  "on 303,571 all-adult admissions [23]; and a 26-feature XGBoost framework reported "
  "0.696 on 415,231 all-adult admissions [24]. ClinicalBERT, a clinical "
  "adaptation of a pretrained language model, reported approximately "
  "0.714 on a MIMIC-III cohort using discharge notes [25]. Because cohorts and "
  "outcome definitions differ, these are reference points rather than "
  "head-to-head comparisons.")
H2("Study Rationale")
P("The gap, therefore, is not another discrimination estimate. None of the "
  "MIMIC studies above combines leakage-safe patient-grouped validation, "
  "calibration analysis, same-cohort reconstruction of the clinical scores "
  "hospitals actually use, readmission-timing analysis, and subgroup "
  "fairness auditing in a single evaluation, and few evaluate the complete "
  "selection procedure rather than one exposed model. This study was "
  "designed to close that gap in the Medicare population, where the "
  "readmission penalty applies, using only predictors demonstrably "
  "available at the moment of discharge.")
H2("Research Questions and Objectives")
P("Four research questions organize the study. RQ1: Which "
  "discharge-available EHR features contribute most to prediction of "
  "30-day readmission in Medicare-insured adults? RQ2: Can a leakage-safe, "
  "calibrated machine-learning model improve prediction of 30-day "
  "readmission relative to established clinical scores? RQ3: Can global "
  "and patient-level Shapley additive explanations (SHAP) provide "
  "clinically interpretable insight into the factors driving predicted "
  "readmission risk? RQ4: How does predictive performance vary across the "
  "post-discharge time window and across demographic subgroups? "
  "Operationally, the study aimed to (1) develop and internally validate a "
  "calibrated, interpretable model of 30-day all-cause within-system readmission "
  "using only structured EHR fields available at discharge (RQ1, RQ2, RQ3); (2) "
  "compare it against LACE and HOSPITAL reconstructed on the identical test "
  "partition (RQ2); (3) characterize when patients return using a "
  "time-to-event reformulation and stage-specific landmark models (RQ4); "
  "(4) audit subgroup performance across age, sex, and race with "
  "cluster-level uncertainty (RQ4); and (5) release the pipeline and "
  "a demonstration prototype in support of reproducibility.")

# ============================================================ METHODS
H1("Methods")
H2("Study Design and Data Source")
P("We conducted a retrospective prediction-model development and internal-validation "
  "study, reported in line with TRIPOD+AI [17]; the completed checklist is "
  "Multimedia Appendix 5 and a PROBAST+AI self-assessment [18] is "
  "Multimedia Appendix 6. The data source is MIMIC-IV v3.1 "
  "[21,26], a deidentified EHR database of 546,028 hospitalizations "
  "occurring between 2008 and 2022 at Beth Israel Deaconess Medical "
  "Center (Boston, MA). Dates are "
  "shifted into the future by a patient-specific offset; within-patient intervals "
  "are preserved. The database provides administrative, diagnosis, procedure, "
  "medication, laboratory, order, and ICU tables. Clinical notes and imaging exist "
  "as separately credentialed companion databases and were deliberately not used "
  "(see Discussion).")
H2("Cohort and Outcome Definition")
P(f"The cohort comprised admissions with insurance recorded as Medicare "
  f"(n={CT['original_admissions']:,}). Because the model is designed to score "
  f"patients at the moment of alive discharge, we excluded "
  f"{CT['index_death_excluded']:,} admissions (3.1%) that ended in index in-hospital "
  f"death, leaving {CT['cohort_v2_admissions']:,} admissions from "
  f"{CT['cohort_v2_patients']:,} unique patients. Medicare insurance does not imply "
  f"age ≥65 years ({FD['fairness']['age_band']['<65']['n']:,} test admissions "
  f"were younger than 65 years, typically through disability entitlement) "
  f"and MIMIC-IV cannot distinguish "
  f"fee-for-service from Medicare Advantage, so the cohort is not a CMS HRRP "
  f"regulatory cohort.")
P("The outcome was the first subsequent hospitalization of the same patient in the "
  "same health system beginning more than 0 and up to 30 days (day 30 inclusive) "
  "after index discharge, of any cause, and irrespective of how that subsequent "
  "admission ended; a readmission followed by in-hospital death is a "
  "readmission. Readmissions were identified from all subsequent "
  f"hospitalizations of the patient in the database, irrespective of the "
  f"insurance recorded on the subsequent admission "
  f"({EX['payer_scope']['n_30d_next_nonmedicare']} 30-day next admissions "
  f"carried non-Medicare insurance). In this binary frame, admissions "
  "followed by death without readmission within 30 days are nonevents; "
  "the competing-risks reformulation below addresses the interpretation. "
  f"The interval Δ is the exact timestamp difference between the next "
  f"admission's admittime and the index dischtime. Next admissions beginning "
  f"at, or up to 2 days before, the index discharge timestamp (−2<Δ≤0 days; "
  f"n={CT['same_day_next_admit(delta<=0)']:,}; no record in this cohort "
  f"overlaps by more than 2 days) were treated as inter-unit transfers or "
  f"administrative continuations, not readmissions. Every index admission of a "
  f"patient is eligible, so one patient can contribute multiple index admissions; "
  f"all inference accounts for this clustering. Because MIMIC-IV is single-center, "
  f"readmissions to other hospitals are unobservable and the outcome is explicitly "
  f"within-system. Planned readmissions could not be reliably identified and were "
  f"not excluded; a sensitivity outcome excluding next admissions typed ELECTIVE "
  f"(cohort-wide prevalence {CT['label_sens_prevalence']*100:.1f}% vs "
  f"{CT['label_v2_prevalence']*100:.1f}%; test-partition values in Results) is "
  f"analyzed in the Results, together "
  f"with sensitivity analyses that recount the transfer band as readmissions and "
  f"that restrict evaluation to nonelective index admissions. Whether the "
  f"final admissions in each patient's record have 30 complete days of observation "
  f"cannot be verified under date shifting; this is a limitation.")
H2("Partitions and Validation Design")
_tot = sum(p[k]["admissions"] for k in ("train", "val", "test"))
P(f"Admissions were first split 80/20 at the patient level "
  f"(GroupShuffleSplit on subject_id) into development and test data; 10% of "
  f"the development portion was then assigned, again patient-grouped, to a "
  f"validation partition used for early stopping, threshold derivation, and "
  f"ablation decisions. The final proportions are therefore approximately "
  f"72% training, 8% validation, and 20% test. After exclusions the "
  f"partitions were: training {p['train']['admissions']:,} admissions "
  f"({p['train']['admissions']/_tot*100:.0f}%; {p['train']['patients']:,} "
  f"patients, {p['train']['events']:,} events); validation "
  f"{p['val']['admissions']:,} ({p['val']['admissions']/_tot*100:.0f}%; "
  f"{p['val']['patients']:,} patients, {p['val']['events']:,} events); test "
  f"{p['test']['admissions']:,} ({p['test']['admissions']/_tot*100:.0f}%; "
  f"{p['test']['patients']:,} patients, {p['test']['events']:,} events). No "
  f"patient appears in more than one partition. No formal sample-size "
  f"calculation was performed: the cohort was fixed by the available "
  f"data, and with {p['train']['events']:,} training-partition events "
  f"against {N_ELIG} candidate predictors the events-per-candidate ratio "
  f"exceeds 250, far above common minima for prediction-model "
  f"development.")
P("Validation hierarchy and test-set transparency: earlier development phases, "
  "retained in the project repository, scored model variants on the original test "
  "partition multiple times; we therefore describe the test data as a "
  "historically exposed internal evaluation partition rather than a pristine "
  "holdout. Three tiers of evidence are reported, in order of authority: "
  "(1) primary: the out-of-fold performance of the complete selection procedure "
  "under 5 outer patient-grouped folds over the development partitions "
  "(training plus validation), in which every admission is scored by a model "
  "whose preprocessing, target encodings, feature selection, and fitting never "
  "saw that patient; this estimates the procedure, with fold-specific feature "
  "sets, not the fixed consensus model; (2) secondary: 5-fold patient-grouped "
  "cross-validation of the fixed consensus feature set on the same development "
  "data with fold-local encodings; the fixed set aggregates selection "
  "information across folds, so this estimate carries disclosed optimism; "
  "(3) tertiary: a single evaluation on the historically exposed test "
  "partition, used for the detailed metric panel, subgroup, baseline, "
  "interpretability, and utility analyses, all of which inherit its historical "
  "exposure. In this version's pipeline, no test-partition outcome was used for "
  "feature selection, fold-local encoding, consensus construction, model "
  "refitting, or threshold derivation; the model family, candidate-predictor "
  "pool, and fixed hyperparameters, however, predated this rerun and may have "
  "been indirectly influenced by earlier test-partition evaluations "
  "(Limitations). Hyperparameters were fixed throughout and not tuned per fold "
  "(disclosed limitation); we do not describe any procedure here as fully "
  "nested cross-validation for that reason. A temporal split was considered and "
  "rejected: per-patient date shifting removes fine-grained cross-patient "
  "ordering, and the coarse patient-level anchor_year_group bands that "
  "remain would yield unequal, clinically heterogeneous partitions; an "
  "era-stratified sensitivity analysis appears in the Results, and "
  "residual secular drift is acknowledged in Limitations. Independent "
  "external validation remains necessary. Throughout, leakage-safe "
  "denotes this design: fold-local preprocessing, target encoding, and "
  "feature selection, plus the discharge-availability predictor audit; "
  "residual design-level optimism (the historically exposed test "
  "partition, the development-derived threshold, and fixed "
  "hyperparameters) is disclosed here and in Limitations.")
P("The complete data flow is: (1) patient-grouped 80/20 development-test "
  "split on subject_id, with 10% of the development portion assigned to "
  "validation (approximately 72/8/20); (2) the 5 outer selection folds are "
  "drawn from all development (training+validation) patients; (3) within each "
  "outer-training fold, 3 grouped inner folds drive staged elimination, with "
  "target encodings refit from inner-training patients at every step; "
  "(4) each fold's selected model is refit on its full outer-training fold "
  "(encodings refit) and scored once on its outer-evaluation fold, yielding "
  "the primary out-of-fold estimate; (5) the consensus set is defined from "
  "the 5 per-fold selections; (6) the final model is refit on the training "
  "partition with training-partition encodings, its operating threshold is "
  "derived on the validation partition, and it is evaluated once on the test "
  "partition. Because the selection folds span all development patients, the "
  "validation partition participated in consensus-feature selection; the "
  "threshold is therefore development-derived, not derived on data "
  "independent of model selection.")
H2("Predictor Audit, Eligible Pool, and Leakage-Safe Feature Selection")
P(f"All 207 candidate predictors from the engineered feature tables were audited "
  f"for source table, transformation, measurement window, missingness (overall, "
  f"by partition, and by race group), and demonstrable availability at discharge "
  f"(Multimedia Appendix 1). Sixty-three features derived from billing-time "
  f"artifacts, namely diagnosis-related groups, ICD diagnosis and procedure codes and "
  f"their derivatives (including comorbidity indices and procedure flags with "
  f"clinical names whose provenance was verified in the feature-build code), "
  f"were excluded because their availability at the moment of discharge cannot "
  f"be established, together with one race-derived encoding; no feature was "
  f"forward-looking. In addition, one precomputed target encoding of discharge "
  f"destination (discharge_location_te) that coexists in the legacy feature "
  f"tables with the raw column was excluded as outcome-derived. Its original "
  f"transformation code is no longer available; an empirical audit found "
  f"within-category value patterns inconsistent with a single training-only "
  f"category map (values vary within category on the development partitions "
  f"but are constant within category on test), so its fold provenance could "
  f"not be aligned with the present validation design and the variable was "
  f"excluded conservatively. The raw discharge_location column is retained "
  f"and, like every categorical, is target-encoded strictly fold-locally. The eligible pool "
  f"comprised {N_ELIG} predictors spanning demographics, admission context, "
  f"medications, laboratory values, orders and operations, ICU documentation, "
  f"and prior-utilization history.")
P(f"Feature selection used staged recursive feature elimination with an XGBoost "
  f"estimator inside 5 outer patient-grouped folds drawn from the development "
  f"partitions only. Within each outer-training fold, 3 grouped inner folds "
  f"were formed; all categorical target encodings were fitted from "
  f"inner-training patients only; elimination proceeded along a prespecified "
  f"grid ({N_ELIG} to 18 features), dropping the lowest mean-gain features "
  f"between stages and scoring each stage by mean inner-validation AUROC. The "
  f"parsimony rule was prespecified before results were inspected: the smallest "
  f"feature count whose mean inner-validation AUROC lay within 0.002 of the "
  f"best. The per-fold selected model was refit on the complete outer-training "
  f"fold (encodings refit there) and scored once on the untouched "
  f"outer-evaluation fold. The final consensus predictor set comprises the "
  f"features selected "
  f"in at least 3 of 5 outer folds ({NF} features; {N_UNANIMOUS} selected in all "
  f"5); no test-partition outcome was used in this construction. Missing "
  f"values were handled natively by the histogram-based learners, "
  f"preserving informative missingness; the regularized "
  f"logistic-regression comparator received fold-local median imputation "
  f"with standardization.")
H2("Model, Comparators, and Development History")
P(f"The final prediction model is a single XGBoost classifier [10] on the "
  f"{NF}-feature consensus set (600 trees, learning rate 0.05, maximum depth 5, "
  f"subsample and column subsample 0.9, histogram method, fixed seed), with "
  f"comparators fitted per outer fold on identical data: the full "
  f"{N_ELIG}-feature eligible pool, the prior model's 50-feature billing-free "
  f"and leakage-free subset, the prior 66-feature set containing billing "
  f"predictors and the excluded encoding (historical reference only), and "
  "a regularized logistic regression on the eligible inputs. During development, LightGBM [11], CatBoost, and "
  "scikit-learn HistGradientBoosting were trained as comparators and a "
  "Nelder-Mead-optimized convex blend of the four families was evaluated; the blend "
  "improved nothing over the best single family and the four families spanned only "
  "0.0035 AUROC, so a single model was retained. Earlier 10-seed averaged "
  "results constitute seed ensembles; the final model here is one seed and one "
  "artifact, and the blend-weight bounds are documented in the repository. "
  "CatBoost received categorical variables natively rather than "
  "target encoded. Neural baselines (multilayer perceptrons, LSTM/GRU hybrids, "
  "FT-Transformer [27], and a stacking ensemble) were explored on an earlier "
  "feature version with single configurations and are reported in the repository as "
  "development history; we therefore describe boosting as the better performer in "
  "our experiments, consistent with published tabular benchmarks [9,12], rather "
  "than claiming a definitive architecture comparison. XGBoost was selected over "
  "the marginally higher-scoring LightGBM for consistency with the accelerated "
  "failure time (AFT) survival objective and exact TreeSHAP tooling [14].")
H2("Clinical-Score Baselines")
P("LACE [3] and HOSPITAL [4] were reconstructed on the identical test partition. "
  "LACE: length-of-stay points per the original bands; 3 acuity points unless the "
  "admission was elective; Charlson-derived comorbidity points; and true prior "
  "180-day emergency-department visits (edregtime) capped at 4. HOSPITAL: "
  "hemoglobin <12 g/dL, oncology service (OMED), sodium <135 mmol/L, any procedure, "
  "nonelective index type, prior admissions in the original 12-month window "
  "(counted from each patient's admission history, which is strictly "
  "pre-discharge information), and length of stay ≥5 days. An earlier "
  "development-phase variant used a 6-month prior-admission proxy and is "
  "retained only as a sensitivity result in Multimedia Appendix 2. Full "
  "mappings appear in Multimedia Appendix 2. Both scores were derived for outcomes that "
  "differ from ours (LACE: death or unplanned readmission; HOSPITAL: potentially "
  "avoidable readmission), so these comparisons quantify performance of the scores "
  "on our outcome, not a refutation of the scores on theirs.")
H2("Time-to-Event and Landmark Analyses")
P("Time to readmission was modeled with the same features and partitions under a "
  "competing-risks event process built from patients.dod, which records death "
  "from hospital records and the Massachusetts State Registry with deaths beyond "
  "one year after last discharge censored by deidentification. Events were "
  "defined as: readmission (event 1) at the first subsequent admission within 30 "
  "days, irrespective of how that admission ended; death before readmission "
  "within 30 days (event 2, competing); administrative censoring at day 30 "
  "otherwise. Cumulative incidence used Aalen-Johansen estimators. The prognostic "
  "models are cause-specific for readmission: XGBoost with the accelerated "
  "failure time (AFT) objective [28] (a single-event survival model evaluated under "
  "censoring, not itself a competing-risk model) and penalized cause-specific "
  "Cox regression as the linear reference, with competing deaths censored at "
  "their death time in the cause-specific fits. Evaluation used Harrell C "
  "[29] and cumulative/dynamic time-dependent AUROC on each of days 1 "
  "through 29, estimated with inverse probability of censoring weighting "
  "[30,40]; patients who died before the evaluation day remain in the "
  "comparison set as nonevents, a cause-specific competing-risks "
  "definition [40], with patient-cluster bootstrap uncertainty bands and "
  "daily event counts reported. "
  "Three landmark models covered days 1 to 7, 8 to 14, and 15 to 30, each trained only on "
  "patients still at risk when the stage opens; patients already readmitted or "
  "already deceased are removed from the risk set, not counted as nonevents; "
  "stage windows differ in length and prevalence by design, so stage results are "
  "descriptive; landmark models were trained on training-partition "
  "patients, evaluated on the test partition, and reuse discharge-time "
  "features without post-discharge updating. Stage-specific TreeSHAP "
  "attribution shares are exploratory.")
H2("Statistical Analysis")
P(f"Because patients contribute multiple admissions, all confidence "
f"intervals (CIs) "
  f"and P values use cluster bootstrap resampling at the patient level with "
  f"percentile intervals (1,000 resamples for the test metric panel; 800 for "
  f"subgroup CIs; 500 for the primary out-of-fold estimate and model "
  f"comparisons; every contrast for "
  f"which a P value is reported was computed with 5,000 resamples). P values "
  f"are empirical two-sided bootstrap probabilities with a plus-one "
  f"correction, so the smallest reportable value at 5,000 resamples is "
  f"P<.001; no P value is reported at a precision the resample count cannot "
  f"support. Model-to-model AUROC contrasts (final model vs LACE, HOSPITAL, "
  f"and out-of-fold comparators) are paired: both scores are evaluated on the "
  f"identical admissions of each bootstrap draw and the difference is "
  f"resampled directly. The White vs Black AUROC contrast compares disjoint "
  f"subgroups and is therefore not paired; patients were resampled within "
  f"each subgroup independently and the difference of subgroup AUROCs was "
  f"computed in each draw and tested directly by bootstrap rather than by "
f"CI overlap. No formal equivalence or noninferiority test was "
  f"prespecified; near-ties are described against the 0.002 tolerance used in "
  f"the prespecified parsimony rule, not as statistical equivalence. Formal "
f"hypothesis testing was limited to three prespecified contrasts (final "
f"model vs LACE, vs HOSPITAL, and the White vs Black AUROC difference); "
f"all other subgroup, timing, era, and sensitivity results are "
f"descriptive, and no adjustment for multiple comparisons was applied. All "
  f"bootstrap intervals were computed by resampling patients and their "
  f"stored prediction-outcome pairs; feature selection and model fitting "
  f"were not repeated within bootstrap samples, so the intervals quantify "
  f"evaluation-sample uncertainty conditional on the fitted models and do "
  f"not capture the variability of retraining or reselection. "
  f"Discrimination used AUROC and average precision; calibration used "
  f"the Brier score, expected calibration error over 10 equal-width "
f"probability bins, and logistic "
  f"recalibration slope and intercept [13]. The operating threshold "
  f"({THRV:.3f}) was the Youden point on the validation partition (a "
  f"development-derived threshold, because validation patients also "
  f"participated in the consensus-selection folds) and was applied "
  f"unchanged to the test partition; sensitivity, specificity, and positive and "
  f"negative predictive values are reported with cluster-bootstrap CIs. "
  f"The Youden threshold is illustrative for the metric panel and is not "
  f"proposed for deployment; the capacity-constrained view is the "
  f"operational lens. The "
  f"Decision-curve analysis [31] compared net benefit against treat-all "
  f"and treat-none across threshold probabilities 0.05 to 0.40, with a capacity view "
  f"(top-k% flagged). Analyses used Python {FM['env']['python']} with "
f"XGBoost {FM['env']['xgboost']}, scikit-learn {FM['env']['sklearn']}, "
f"pandas {FM['env']['pandas']}, and NumPy {FM['env']['numpy']} (fixed "
f"seed {FM['env']['seed']}); LightGBM, lifelines, scikit-survival, and "
f"SHAP versions are pinned in the repository.")
H2("Fairness Audit")
P("Subgroup performance was audited on the test partition across age bands, sex, "
  "and race. MIMIC race/ethnicity strings were consolidated by substring mapping "
  "(WHITE* to White; BLACK* to Black; HISPANIC*/LATINO* to Hispanic/Latino; ASIAN* "
  "to Asian; all others, declined, or unknown to Other/Unknown); raw category "
  "counts appear in Multimedia Appendix 3. Race is not a model input; auditing by "
  "race is deliberate because not using an attribute does not guarantee equal "
  "performance across it [15,16,19]. During development, a race-encoding "
  "ablation was performed on the validation partition: the historical model "
  "was refit with and without the precomputed race encoding and validation "
  "AUROC compared (0.7954 with vs 0.7949 without); the encoding contributed "
  "no measurable signal and race-derived features were excluded from all "
  "final configurations (Multimedia Appendix 4). This result describes one "
  "encoding's marginal contribution and does not imply that excluding race "
  "guarantees equal subgroup performance.")
H2("Ethical Considerations")
P("MIMIC-IV is a deidentified, publicly available database; its creation was "
  "approved by the institutional review boards of the Massachusetts Institute of "
  "Technology and Beth Israel Deaconess Medical Center with a waiver of informed "
  "consent [21,26]. TB, the only author to access the raw data, did so under "
  "the PhysioNet Credentialed Health Data Use Agreement after completing the "
  "required human-subjects training (CITI Data or Specimens Only Research); "
  "co-authors worked only with aggregate, deidentified derived results. No "
  "attempt was made to reidentify individuals; no row-level data are "
  "redistributed. This secondary analysis of deidentified, publicly available "
  "data does not constitute human-subjects research and therefore did not "
  "require institutional review board review at Florida International "
  "University, consistent with the United States federal definition of "
  "human-subjects research for secondary analyses of deidentified data. "
  "The public demonstration prototype accepts only synthetic or "
  "manually entered values, displays a research-only, no-real-patient-data "
  "notice, and stores no submitted data.")

# ============================================================ RESULTS
H1("Results")
H2("Cohort")
P(f"Figure 1 shows the cohort flow. Of {CT['original_admissions']:,} Medicare "
  f"admissions, {CT['index_death_excluded']:,} ended in index in-hospital death and "
  f"were excluded, leaving {CT['cohort_v2_admissions']:,} admissions from "
  f"{CT['cohort_v2_patients']:,} patients; {CT['label_v2_events']:,} "
  f"({CT['label_v2_prevalence']*100:.1f}%) were followed by a within-system "
  f"readmission within 30 days. Mean time to readmission among events was "
  f"{SV['mean_days_to_readmission']:.1f} days. This prevalence reflects "
  f"an all-cause, within-system definition that includes planned returns "
  f"and all Medicare admission types, and is not comparable to HRRP "
  f"condition-specific national rates.")
figure(FIG / "r2_fig1_flow_col.png",
       "Figure 1. Participant flow diagram with patient-grouped "
       "partitions; feature selection ran inside 5 patient-grouped folds "
       "drawn from the development partitions (Methods).")
H2("Feature Selection: Trajectories, Stability, and Model Comparison")
_cmp = {k: CMP5[k] for k in ("rfe", "all_eligible", "f50", "f66", "logit")}
_pd_all = PAIRED["all_eligible_minus_rfe"]
_pd_f50 = PAIRED["f50_minus_rfe"]
_pd_f66 = PAIRED["f66_minus_rfe"]
P(f"Inner-validation AUROC was nearly flat from {N_ELIG} features down to "
  f"roughly 32 in every outer fold and degraded at smaller counts (Figure 2); "
  f"the prespecified parsimony rule selected "
  f"{', '.join(str(v) for v in STAB['fold_counts'].values())} "
  f"features across the 5 outer folds. Selection was stable (mean pairwise "
  f"Jaccard {STAB['jaccard_mean']:.3f}; {N_UNANIMOUS} features selected in "
  f"all 5 folds; Figure 3), and the consensus rule (at least 3 of 5 folds) "
  f"yielded the {NF}-feature final set. Out-of-fold over the development "
  f"data, the selection procedure achieved AUROC {_cmp['rfe']['oof_auroc']:.4f} "
  f"(95% cluster CI {_cmp['rfe']['auroc_ci95_cluster'][0]:.4f} to "
  f"{_cmp['rfe']['auroc_ci95_cluster'][1]:.4f}). Paired patient-cluster "
  f"bootstrap differences on identical observations (each computed as the "
  f"comparator minus the selection procedure, so a negative value favors the "
  f"procedure) were "
  f"{_pd_all['point']:+.4f} ({nbci(_pd_all['ci95'][0], _pd_all['ci95'][1])}) "
  f"for the full {N_ELIG}-feature eligible model "
  f"({_cmp['all_eligible']['oof_auroc']:.4f}) and {_pd_f50['point']:+.4f} "
  f"({nbci(_pd_f50['ci95'][0], _pd_f50['ci95'][1])}) for the "
  f"prior 50-feature subset ({_cmp['f50']['oof_auroc']:.4f}); these absolute "
  f"differences are smaller than the 0.002 tolerance used in the prespecified "
  f"parsimony rule and are described as similar discrimination rather than "
  f"statistical equivalence, which was not formally tested. In addition, "
  f"conditional on treating both predictor sets as fixed, grouped "
  f"cross-validation on identical development folds produced AUROCs of "
  f"{EX['fixed31_vs_fixed142_paired_cv']['auroc_fixed31_cv']:.4f} for the "
  f"{NF}-feature consensus model and "
  f"{EX['fixed31_vs_fixed142_paired_cv']['auroc_fixed142_cv']:.4f} for the "
  f"complete {N_ELIG}-feature model, a difference of "
  f"{EX['fixed31_vs_fixed142_paired_cv']['paired_diff_31_minus_142']:+.4f} "
  f"(95% CI {EX['fixed31_vs_fixed142_paired_cv']['ci95'][0]:.4f} to "
  f"{EX['fixed31_vs_fixed142_paired_cv']['ci95'][1]:.4f}). This secondary, "
  f"descriptive comparison is potentially optimistic for the {NF}-feature "
  f"model, whose consensus set was selected using these same development "
  f"data, whereas the eligible pool was defined by an a priori audit; the "
  f"complete selection-procedure estimate remains the primary evidence. "
  f"The prior 66-feature set containing billing predictors "
  f"and the excluded encoding performed worst among the boosted models under "
  f"leakage-safe evaluation ({_cmp['f66']['oof_auroc']:.4f}; paired difference "
  f"{_pd_f66['point']:+.4f}, {nbci(_pd_f66['ci95'][0], _pd_f66['ci95'][1])}), "
  f"and regularized logistic regression on the "
  f"same eligible inputs reached {_cmp['logit']['oof_auroc']:.4f}. Because a "
  f"negligible absolute difference does not outweigh a materially simpler set, "
  f"the {NF}-feature consensus model was adopted. It contains no "
  f"billing-derived features by construction; whether concurrent "
  f"diagnosis coding would add signal at sites where it is reliably "
  f"available at discharge is a site-specific question left to future "
  f"work, and the earlier development-phase sensitivity analyses are "
  f"retained in Multimedia Appendix 4.")
figure(FIG5 / "v5_rfe_curve.png",
       "Figure 2. Leakage-safe staged recursive feature elimination on "
       "development data: mean inner-validation AUROC versus retained "
       f"features, per outer fold; dashed line marks the {NF}-feature "
       "consensus.")
figure(FIG5 / "v5_stability.png",
       f"Figure 3. Selection frequency across the 5 outer folds for all "
       f"{len(STAB['selection_frequency'])} features selected at least "
       "once; teal bars form the consensus set and the dashed line marks "
       "the consensus threshold (3 of 5 folds).")
H2("Discrimination and Calibration of the Final Model")
P(f"The consensus model's grouped cross-validation AUROC on development data "
  f"was {CCV['mean']:.4f} (fold SD {CCV['sd']:.4f}; folds "
  f"{', '.join(f'{x:.4f}' for x in CCV['folds'])}; the fixed feature set "
  f"aggregates selection information across folds and so carries mild "
  f"optimism relative to the primary out-of-fold estimate). On "
  f"the historically exposed test partition ({p['test']['admissions']:,} "
  f"admissions, {p['test']['patients']:,} patients; tertiary evidence), the "
  f"refit model reached AUROC {MP['auroc']['point']:.4f} (95% cluster CI "
  f"{MP['auroc']['ci95'][0]:.4f} to {MP['auroc']['ci95'][1]:.4f}) and average "
  f"precision {MP['ap']['point']:.3f} (95% CI {MP['ap']['ci95'][0]:.3f} to "
  f"{MP['ap']['ci95'][1]:.3f}) against a "
  f"{p['test']['events']/p['test']['admissions']*100:.1f}% prevalence. "
  f"Calibration was strong: Brier {MP['brier']['point']:.4f}, expected "
  f"calibration error {MP['ece']['point']:.3f} (95% CI "
  f"{MP['ece']['ci95'][0]:.3f} to {MP['ece']['ci95'][1]:.3f}), slope "
  f"{MP['slope']['point']:.2f} (95% CI {MP['slope']['ci95'][0]:.2f} to "
  f"{MP['slope']['ci95'][1]:.2f}), intercept {MP['intercept']['point']:.2f} "
  f"(95% CI {MP['intercept']['ci95'][0]:.2f} to "
  f"{MP['intercept']['ci95'][1]:.2f}) "
  f"(Figure 4). At the development-derived threshold of {THRV:.3f}: sensitivity "
  f"{MP['sensitivity']['point']:.3f} (95% CI {MP['sensitivity']['ci95'][0]:.3f} to "
  f"{MP['sensitivity']['ci95'][1]:.3f}), specificity "
  f"{MP['specificity']['point']:.3f}, positive predictive value "
  f"{MP['ppv']['point']:.3f}, negative predictive value "
  f"{MP['npv']['point']:.3f}, and false-negative rate {MP['fnr']['point']:.3f}, "
  f"each with cluster-bootstrap CIs in the machine-readable results file and "
  f"Multimedia Appendix 1.")
figure(FIG5 / "v5_roc_cal.png",
       "Figure 4. Receiver operating characteristic, precision-recall, "
       f"calibration, and confusion-matrix panels for the final {NF}-feature "
       "model (historically exposed test partition; tertiary evidence). "
       "Dashed lines mark chance performance (receiver operating "
       "characteristic panel) and outcome prevalence (precision-recall "
       "panel).")
H2("Outcome-Definition and Cohort Sensitivity Analyses")
_era = EX["era_stratified_test"]["bands"]
_era_txt = "; ".join(
    f"{b['band'].replace(' - ', ' to ')}: {b['auroc']:.4f} "
    f"(n={b['n']:,})" for b in _era)
_sa = SENS["sameday_as_readmission"]
_su = SENS["unplanned_only"]
_sn = SENS["nonelective_index_only"]
P(f"Five sensitivity analyses probe the outcome definition, the cohort rules, and era composition. (1) Counting "
  f"the transfer/continuation band (next admission beginning at, or up to 2 "
  f"days before, the index discharge timestamp: −2<Δ≤0 days; "
  f"{EX['transfer_band']['n_band_by_partition']['test']} test admissions) as "
  f"readmissions raised test events from "
  f"{_sa['test_events_primary']:,} to {_sa['test_events_alt']:,} "
  f"({_sa['test_events_alt']/p['test']['admissions']*100:.1f}% prevalence); a model refit under "
  f"this label reached AUROC {_sa['test_auroc_refit']:.4f}, and the final "
  f"model ranked the alternative label at {_sa['test_auroc_final_model_vs_alt_label']:.4f}. "
  f"(2) Excluding next admissions typed ELECTIVE (a proxy for planned "
  f"readmissions; {_su['test_events_alt']:,} test events, "
  f"{_su['test_prevalence_alt']*100:.1f}% prevalence) gave a refit AUROC of "
  f"{_su['test_auroc_refit']:.4f}; the final model ranked this outcome at "
  f"{_su['test_auroc_final_model_vs_alt_label']:.4f}. (3) Restricting "
  f"evaluation to nonelective index admissions ({_sn['test_admissions']:,} "
  f"admissions, {_sn['test_prevalence']*100:.1f}% prevalence) gave AUROC "
  f"{_sn['test_auroc']:.4f}. (4) Excluding the {BD5['n_boundary']:,} test "
  f"admissions ({BD5['share']*100:.1f}%) whose latest possible calendar year "
  f"reached the 2022 end of data collection changed AUROC to "
  f"{BD5['auroc_safe']:.4f}. (5) Stratified by the patient-level "
f"anchor_year_group era band, test discrimination was maintained or "
f"higher in recent eras ({_era_txt}), while observed prevalence declined "
f"from {_era[0]['prevalence']*100:.1f}% to "
f"{_era[-1]['prevalence']*100:.1f}%; era strata are descriptive. None of "
f"these variations alters the study's conclusions.")
H2("Same-Cohort Clinical Baselines")
P(f"On the identical test partition, both reconstructed scores fell below "
f"their published external validations, reflecting within-system attenuation "
f"of their utilization inputs; because that attenuation affects the scores "
f"but not the model's in-hospital signals, the uplifts below are best read "
f"as upper bounds for this setting. The final model (AUROC "
  f"{MP['auroc']['point']:.4f}) outperformed reconstructed LACE "
  f"({BSE['lace_auroc']:.4f}) by {BSE['uplift_lace']:.4f} AUROC (95% CI "
  f"{BSE['uplift_lace_ci'][0]:.4f} to {BSE['uplift_lace_ci'][1]:.4f}; "
  f"{fmt_p(BSE['uplift_lace_p'])}; paired, 5,000 draws) and "
  f"HOSPITAL, rebuilt with its original 12-month prior-admission window "
  f"({BSE['hospital12_auroc']:.4f}), by {BSE['uplift_hospital12']:.4f} (95% CI "
  f"{BSE['uplift_hospital12_ci'][0]:.4f} to {BSE['uplift_hospital12_ci'][1]:.4f}; "
  f"{fmt_p(BSE['uplift_hospital12_p'])}) (Multimedia Appendix 4). Both reconstructed scores fell below their "
  f"published values; the "
  f"comparison applies both scores to this study's all-cause within-system "
  f"outcome, which differs from their derivation outcomes (Methods), and does "
  f"not invalidate the original instruments.")
H2("Timing of Readmission and Competing Death")
_cif = SV.get("aj_cif_30d")
_cd = SV.get("test_competing_deaths")
P((f"Under the competing-risks event process, the test partition contributed "
   f"{SV['test_events']:,} readmissions and {_cd:,} deaths before readmission; "
   f"Aalen-Johansen 30-day cumulative incidence was {_cif['readmission']*100:.1f}% "
   f"for readmission and {_cif['death']*100:.1f}% for death without readmission. "
   if _cif else "") +
  f"The competing-risks frame counts {p['test']['events']-SV['test_events']} "
  f"fewer readmissions than the binary frame ({SV['test_events']:,} vs "
  f"{p['test']['events']:,}) because registry death dates carry day-level "
  f"granularity: when a readmission and a death share the same recorded date, "
  f"the tie was resolved as death first, a documented rule adopted during "
  f"analysis, whose true ordering the day-level dates cannot establish. A "
  f"tie-rule sensitivity analysis resolving all "
  f"{EX['tie_rule_sensitivity']['n_ties']} same-date ties in the cohort "
  f"({EX['tie_rule_sensitivity']['n_ties_test']} in the test partition) "
  f"readmission-first left every "
  f"estimate essentially unchanged (Harrell C "
  f"{EX['tie_rule_sensitivity']['death_first_primary']['harrell_c']:.4f} vs "
  f"{EX['tie_rule_sensitivity']['readmission_first']['harrell_c']:.4f}; 30-day "
  f"readmission cumulative incidence "
  f"{EX['tie_rule_sensitivity']['death_first_primary']['aj_cif30_readmission']*100:.1f}% vs "
  f"{EX['tie_rule_sensitivity']['readmission_first']['aj_cif30_readmission']*100:.1f}%). "
  f"Daily event counts appear in Multimedia Appendix 1. "
  f"The cause-specific AFT model on the final {NF}-feature set reached Harrell C "
  f"{FM['survival']['aft_harrell_c']:.4f} (the fuller development sets scored "
  f"marginally higher; Multimedia Appendix 4). Evaluated daily (Figure 5), "
  f"time-dependent AUROC was highest on the first day after discharge "
  f"({FM['survival']['daily_tdauc'][0]:.3f}), declined to "
  f"{FM['survival']['daily_tdauc'][6]:.3f} by day 7, and was "
  f"{FM['survival']['daily_tdauc'][-1]:.3f} at day 29. Concordance and AUROC are "
  f"related but distinct quantities and are not directly equated.")
figure(FIG5 / "v5_daily.png",
       "Figure 5. Time-dependent AUROC by day since discharge for the final "
       f"{NF}-feature AFT model (competing-risks labels; IPCW estimation).")
H2("Stage-Specific Landmark Models (Exploratory)")
P(f"With death-aware risk sets, landmark AUROCs for the final feature set were "
  f"{st1['auroc']:.4f} for days 1 to 7 ({st1['events']:,} events among "
  f"{st1['at_risk']:,} at risk), {st2['auroc']:.4f} for days 8 to 14 "
  f"({st2['events']:,} events among {st2['at_risk']:,} at risk), and {st3['auroc']:.4f} for days "
  f"15 to 30 ({st3['events']:,} events among {st3['at_risk']:,} at risk); discrimination improved for "
  f"later windows. A negative finding accompanies this: the domain-composition "
  f"gradient observed in richer development feature sets (laboratory attribution "
  f"falling and utilization rising across the window) did not replicate in the "
  f"parsimonious final model, whose attribution is dominated by prior "
  f"utilization at every stage "
  f"({ST5[0]['shares']['Prior utilization']:.0f}% to "
  f"{ST5[2]['shares']['Prior utilization']:.0f}%) with no monotonic laboratory "
  f"trend (Figure 6, panel A). The earlier finding is therefore "
  f"feature-set-dependent and is reported as exploratory development history in "
  f"Multimedia Appendix 4.")
_sig = [r for r in SD5 if r["excludes_zero"]]
_early = [r for r in _sig if r["diff_pp"] > 0]
_late = [r for r in _sig if r["diff_pp"] < 0]
_elist = "; ".join(f"{flabel(r['feature'])} ({r['diff_pp']:+.1f} percentage "
                   f"points)" for r in _early)
_llist = "; ".join(f"{flabel(r['feature'])} ({r['diff_pp']:.1f} "
                   f"percentage points)" for r in _late)
P(f"At the individual-feature level, however, early and late returns do differ "
  f"in character (Figure 6, panel B; "
  f"{'all 10' if len(_sig) == 10 else f'{len(_sig)} of the 10'} leading "
  f"differentials have 95% cluster-bootstrap CIs excluding zero; the 10 "
  f"largest observed differentials were selected before interval "
  f"inspection, so these intervals are descriptive). Signals "
  f"disproportionately important for a return within days 1 to 7 are "
  f"predominantly states of the index discharge itself: {_elist}. Signals "
  f"disproportionately important for returns in days 15 to 30 are predominantly "
  f"the longer-horizon utilization aggregates: {_llist}. These are "
  f"exploratory, descriptive contrasts in model reliance, offered as "
  f"hypotheses about differently timed follow-up rather than causal claims.")
figure(FIG5 / "v5_stage_combined.png",
       f"Figure 6. Attribution across post-discharge windows, final "
       f"{NF}-feature model. (A) Share of total model attribution by clinical "
       "domain. (B) Features whose attribution share differs most between the "
       "earliest (days 1 to 7) and latest (days 15 to 30) windows; positive values "
       "indicate greater early importance. Error bars are 95% "
       "cluster-bootstrap CIs.", width=3.1)
H2("Fairness and Subgroup Performance")


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
                     f"{e['brier']:.4f}" if e.get("brier") is not None else "NA",
                     _ci3(e, "cal_slope")])
_slopes = [(g, FM["fairness"]["race"][g]) for g in
           ["White", "Black", "Hispanic/Latino", "Asian", "Other/Unknown"]]
_smin = min(_slopes, key=lambda kv: kv[1]["cal_slope"])
_smax = max(_slopes, key=lambda kv: kv[1]["cal_slope"])
P(f"The test partition is unevenly distributed across race categories "
  f"({FR['White']['admissions']:,} White "
  f"[{FR['White']['admissions']/p['test']['admissions']*100:.1f}%] vs "
  f"{FR['Black']['admissions']:,} Black "
  f"[{FR['Black']['admissions']/p['test']['admissions']*100:.1f}%] "
  f"admissions), and underrepresentation in training data is itself a plausible "
  f"contributor to subgroup differences. Discrimination was lower for "
  f"Black patients than for White patients: difference "
  f"{wb['point']:.4f} (95% CI {wb['ci95'][0]:.4f} to {wb['ci95'][1]:.4f}; "
  f"{fmt_p(wb['p_bootstrap'])}, stratified subgroup bootstrap with 5,000 "
  f"draws, patients resampled within each subgroup), and it also varied "
  f"across age bands, from {AB['<65']['auroc']:.4f} (95% CI "
  f"{AB['<65']['auroc_ci95'][0]:.4f} to "
  f"{AB['<65']['auroc_ci95'][1]:.4f}) among patients younger than 65 "
  f"years, largely disability-entitled Medicare, to "
  f"{AB['65-74']['auroc']:.4f} (95% CI "
  f"{AB['65-74']['auroc_ci95'][0]:.4f} to "
  f"{AB['65-74']['auroc_ci95'][1]:.4f}) for ages 65 to 74, with similar "
  f"performance by sex (Figure 7, Table 1). Among the subgroup "
  f"comparisons, only the White vs Black contrast was prespecified and "
  f"formally tested. Between-group AUROC "
  f"differences partly reflect case mix and outcome prevalence, which "
  f"differ across groups. At the shared threshold, specificity was "
  f"{FR['Black']['specificity']:.3f} for Black vs "
  f"{FR['White']['specificity']:.3f} for White patients, so a larger "
  f"share of nonreadmitted Black patients is flagged; under a "
  f"deployment posture that pairs flags with additional supportive "
  f"resources rather than reduced care (Discussion), this directs more "
  f"outreach rather than less, but it is an operational disparity to "
  f"monitor. "
  f"Calibration slopes across race groups ranged from "
  f"{_smin[1]['cal_slope']:.2f} ({_smin[0]}; 95% CI "
  f"{_smin[1]['cal_slope_ci95'][0]:.2f} to {_smin[1]['cal_slope_ci95'][1]:.2f}) to "
  f"{_smax[1]['cal_slope']:.2f} ({_smax[0]}; 95% CI "
  f"{_smax[1]['cal_slope_ci95'][0]:.2f} to {_smax[1]['cal_slope_ci95'][1]:.2f}); "
  f"per-group calibration summaries with uncertainty appear in Table 1; "
  f"across age bands the youngest and oldest bands deviated from 1 "
  f"({AB['<65']['cal_slope']:.2f}, 95% CI "
  f"{AB['<65']['cal_slope_ci95'][0]:.2f} to "
  f"{AB['<65']['cal_slope_ci95'][1]:.2f}, and "
  f"{AB['85+']['cal_slope']:.2f}, 95% CI "
  f"{AB['85+']['cal_slope_ci95'][0]:.2f} to "
  f"{AB['85+']['cal_slope_ci95'][1]:.2f}), indicating mild over- and "
  f"underdispersion respectively. Race is not a model input; a validation-partition ablation showed the "
  f"previously selected race encoding contributed nothing (Methods), and its "
  f"removal did not close the gap. Because the disparity is one of ranking, "
  f"subgroup-specific recalibration cannot repair it; candidate remedies are "
  f"representation-aware training, improved measurement, reweighting, or model "
  f"redevelopment, and none is claimed here. Because the outcome is "
  f"within-system, subgroup differences in out-of-system readmission "
  f"would appear as differential outcome misclassification and could "
  f"contribute to, or mask, the observed gap; MIMIC-IV cannot test this, "
  f"and external validation with claims-complete follow-up is the "
  f"appropriate check.")
begin_full_width()
caption("Table 1. Subgroup performance of the final model on the test "
        f"partition at the development-derived threshold ({THRV:.3f}). CIs "
        "are 95% patient-cluster bootstrap percentile intervals; PPV and "
        "Brier are shown as point estimates here, and the complete metric "
        "set (including specificity, NPV, and CIs for every metric) appears "
        "in Multimedia Appendix 1. AUROC: area under the receiver operating "
        "characteristic curve; FNR: false-negative rate; PPV: positive "
        "predictive value; Prev.: outcome prevalence; Cal.: calibration. "
        "Other/Unknown is heterogeneous (declined and unknown race "
        "included) and is not interpreted; patients whose recorded race "
        "varies across admissions appear in more than one race row, so "
        "race-row patient counts exceed unique test patients.",
        keep_with_next=True)
table(rows, font_size=8,
      widths=[0.72, 1.12, 0.45, 0.95, 0.95, 0.9, 0.45, 0.45, 0.79])
begin_two_columns()
figure(FIG5 / "v5_fairness.png",
       f"Figure 7. Subgroup AUROC of the final model with 95% "
       f"patient-cluster bootstrap CIs; the dashed line marks the overall "
       f"test AUROC ({MP['auroc']['point']:.4f}).")
H2("Interpretability via SHAP")
_t3 = [f"{flabel(s['feature'])} ({s['feature']}; mean |SHAP| "
       f"{s['mean_abs_shap']:.3f})" for s in SH[:3]]
_top3 = f"{_t3[0]}, {_t3[1]}, and {_t3[2]}"
_n4 = [flabel(s["feature"]) for s in SH[3:7]]
_next4 = f"{_n4[0]}, {_n4[1]}, {_n4[2]}, and {_n4[3]}"
P(f"Figure 8 ranks the final model's predictors by mean absolute SHAP value on "
  f"the test partition. The strongest signals are {_top3}; {_next4} complete "
  f"the top seven. Readmission risk is multifactorial (no single feature "
  f"dominates), and the leading features map to observables a discharge team "
  f"can see: an escalating admission pattern, where the patient is going next, "
  f"and unresolved laboratory abnormalities. Per-patient additive SHAP "
  f"decompositions are produced at serving time by the released prototype, "
  f"converting each score into a ranked list of that patient's contributing "
  f"factors.")
figure(FIG5 / "v5_shap.png",
       f"Figure 8. Top 10 predictors of the final {NF}-feature model by mean "
       "absolute SHAP value (historically exposed test partition).")
H2("Clinical Utility (Exploratory)")
P(f"Decision-curve analysis showed positive net benefit over treat-all and "
  f"treat-none across the evaluated threshold range of 0.05 to 0.40, a "
  f"range not yet grounded in operational evidence (Multimedia Appendix 4). "
  f"Under capacity constraints, flagging the top 5% of test admissions captured "
  f"{CAP['top_5']['captured']*100:.0f}% of all readmissions at a positive "
  f"predictive value of {CAP['top_5']['ppv']:.2f}; the top 10% captured "
  f"{CAP['top_10']['captured']*100:.0f}% at {CAP['top_10']['ppv']:.2f}; the top "
  f"20% captured {CAP['top_20']['captured']*100:.1f}% at "
  f"{CAP['top_20']['ppv']:.2f}. Whether acting on these flags improves outcomes "
  f"is untested and requires prospective evaluation. Patients who die out "
  f"of hospital within 30 days without readmission are nonevents under "
  f"the binary outcome; a low predicted readmission risk must not be "
  f"read as clinical stability, and any deployment should surface "
  f"competing mortality risk alongside the readmission score, "
  f"particularly for hospice and facility discharges.")

# ============================================================ DISCUSSION
H1("Discussion")
H2("Principal Findings")
P(f"Four contributions carry this study, examined in turn below. In "
f"{CT['cohort_v2_admissions']:,} Medicare-insured adult admissions, a "
  f"{NF}-feature gradient-boosted model, selected from the complete pool of "
  f"demonstrably discharge-available, nonbilling predictors under patient-"
  f"grouped, fold-local preprocessing and feature selection on development "
  f"data only, estimated 30-day within-system readmission with strong "
  f"calibration; the selection procedure achieved a development-data "
  f"out-of-fold AUROC of {OOF['auroc']:.4f} (primary estimate) and the fixed "
  f"model {MP['auroc']['point']:.4f} on the historically exposed test "
  f"partition. First, the parsimonious "
  f"procedure showed no material decrement in discrimination against the full "
  f"eligible pool under leakage-safe "
  f"out-of-fold comparison, and the descriptive fixed-set comparison on "
  f"identical development folds showed a negligible difference "
  f"({EX['fixed31_vs_fixed142_paired_cv']['paired_diff_31_minus_142']:+.4f}, "
  f"95% CI {EX['fixed31_vs_fixed142_paired_cv']['ci95'][0]:.4f} to "
  f"{EX['fixed31_vs_fixed142_paired_cv']['ci95'][1]:.4f}; potentially "
  f"optimistic for the consensus set, which was selected on the same data), "
  f"while the previous billing-containing model performed worse under "
  f"leakage-safe evaluation: most of the usable signal is captured by a "
  f"compact, operationally available predictor set. Second, discrimination "
  f"is highest for readmissions occurring during the first day after "
  f"discharge, the window in which an intervention initiated at discharge "
  f"would need to act (day-level and window-level estimands differ: "
  f"window-level landmark discrimination was lowest for days 1 to 7 and "
  f"highest for days 15 to 30; Results). Third, the model performs less well for Black "
  f"patients despite race not being an input, a disparity we quantify "
  f"directly and report as an open limitation. Fourth, the model clearly "
  f"outperforms the reconstructed LACE and HOSPITAL scores on the identical "
  f"cohort and outcome, with calibrated absolute risks the bedside scores "
  f"do not provide.")
H2("Answers to the Research Questions")
_rq1 = (", ".join(flabel(s["feature"]) for s in SH[:6])
        + ", and " + flabel(SH[6]["feature"]))
P(f"RQ1 (which features): {_rq1} lead the final model (Figure 8); "
  f"{N_UNANIMOUS} features were "
  f"selected in every outer fold, indicating that the predictive core is stable "
  f"rather than an artifact of one selection run. RQ2 (improvement over "
  f"clinical scores): yes; on the identical cohort and outcome the "
  f"leakage-safe final model outperformed reconstructed LACE by "
  f"{BSE['uplift_lace']:.4f} and 12-month HOSPITAL by "
  f"{BSE['uplift_hospital12']:.4f} AUROC (both "
  f"{fmt_p(max(BSE['uplift_lace_p'], BSE['uplift_hospital12_p']))}), while "
  f"adding calibrated absolute risks; among learners, gradient boosting "
  f"led ({OOF['auroc']:.4f} vs {CMP5['logit']['oof_auroc']:.4f} for the "
  f"regularized logistic-regression procedure on identical inputs, with "
  f"LightGBM, CatBoost, and HistGradientBoosting within 0.0035 and "
  f"blending adding nothing). RQ3 (interpretable output): global SHAP "
  f"identifies what the model relies on, and per-patient additive "
  f"decompositions convert each score into a ranked list of that "
  f"patient's contributing factors at serving time. RQ4 (timing and "
  f"subgroups): discrimination is highest on the first day after "
  f"discharge (time-dependent AUROC "
  f"{FM['survival']['daily_tdauc'][0]:.4f}) and declines over the "
  f"window, and performance differs across race groups (AUROC "
  f"{FR['Black']['auroc']:.4f} for Black patients vs "
  f"{FR['White']['auroc']:.4f} for White patients; "
  f"{fmt_p(wb['p_bootstrap'])}) and across age bands "
  f"({AB['<65']['auroc']:.4f} for patients younger than 65 years vs "
  f"{AB['65-74']['auroc']:.4f} for ages 65 to 74), disparities reported "
  f"openly as limitations and targets for remediation.")
H2("Comparison With Prior Work")
P(f"On the same database, published models report 0.791 using chest radiographs on "
  f"a selected 14,532-admission subset [22], 0.727 using discharge notes [23], and "
  f"0.696 using 26 structured features [24]. The selection procedure, retaining "
  f"27 to 38 nonbilling structured predictors across folds, achieved a "
  f"development-data out-of-fold AUROC of {OOF['auroc']:.4f} on a "
  f"Medicare-insured cohort, and the final fixed {NF}-feature consensus model "
  f"reached {MP['auroc']['point']:.4f} on the historically exposed test "
  f"partition, numerically above the notes-based and structured-feature "
  f"reports and near the multimodal result, while scoring every admission "
  f"rather than a radiograph-selected subset. Because cohorts, outcome "
  f"definitions, modalities, and validation designs differ across these "
  f"studies, they are reference points and no superiority claim is made; "
  f"our controlled comparison is the same-cohort "
  f"baseline analysis, where reconstructing HOSPITAL with its original 12-month "
  f"prior-admission window (AUROC "
  f"{RB['C_hospital_12m']['hospital_12m_auroc']:.4f} vs "
  f"{RB['C_hospital_12m']['hospital_6m_proxy_auroc']:.4f} under the 6-month "
  f"proxy) does not alter the conclusion. Table 2 assembles the comparison. "
  f"Against the broader literature, the result is consistent "
  f"with evidence that boosted trees remain highly competitive on structured "
  f"clinical data [9,12] and that careful validation design matters more than "
  f"architecture [5,7,17].")
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
        f"MIMIC-IV Medicare, development partitions",
        f"{CCV['mean']:.4f}",
        "Secondary; fixed set aggregates selection information across folds "
        "(optimism disclosed)"],
       [f"This study: fixed consensus-{NF}, test",
        f"MIMIC-IV Medicare, test partition "
        f"({p['test']['admissions']:,} admissions)",
        f"{MP['auroc']['point']:.4f}",
        "Tertiary; historically exposed partition, not used in this "
        "version's RFE or consensus construction"]]
begin_full_width()
caption("Table 2. Comparison with clinical scores and published MIMIC "
        "readmission models. The Data column marks which comparators share this "
        "study's cohort (controlled) or database (closely related); published "
        "values from differing cohorts and outcome definitions are reference "
        "points, not head-to-head results. This study's three rows are "
        "different estimands (selection procedure vs fixed consensus model) "
        "and are labeled by evidence tier. OOF: out-of-fold.",
        keep_with_next=True)
table(_t2, font_size=8.5, widths=[1.7, 2.2, 0.95, 1.95])
begin_two_columns()
H2("Exclusion of Clinical Notes and Imaging as a Design Decision")
P("MIMIC-IV offers deidentified discharge summaries and radiology reports "
  "(MIMIC-IV-Note) and linked radiographs (MIMIC-CXR) as separately credentialed "
  "companions [32,33]; their exclusion was a design decision. Structured fields are "
  "available in any EHR at discharge, whereas notes and imaging require "
  "natural language processing (NLP) or picture archiving and "
  "communication system (PACS) integration many sites cannot provide; discharge summaries are authored at "
  "or after discharge and routinely reference planned follow-up, so they risk "
  "importing the outcome into the features; and restricting to imaged admissions "
  "would shrink and bias the cohort, as the radiograph-selected subset of [22] "
  "illustrates. Whether unstructured modalities can raise the structured-data "
  "ceiling, and at what cost to leakage safety, is a planned follow-up study.")
H2("Clinical Interpretation of the Leading Predictors (Hypotheses)")
# guards: the following hand-written clinical prose assumes these SHAP facts;
# if the v5 ranking moved, the build must fail so the text is rewritten.
assert SH[0]["feature"] == "los_trend_180d", \
    f"clinical text assumes los_trend_180d leads; got {SH[0]['feature']}"
assert "lab_abnormal_rate" in [s["feature"] for s in SH[:8]], \
    "clinical text assumes lab_abnormal_rate in top signals"
P("Four predictors merit clinical reading because their SHAP magnitude and "
  "actionability are jointly greatest; throughout, “driver” means contribution to "
  "the model's prediction, not a demonstrated causal effect.")
P("The 180-day length-of-stay trend, the strongest signal, is best read as a "
  "compressed biomarker of disease trajectory: when recent admissions grow "
  "progressively longer, stabilization is taking longer and functional reserve at "
  "separation is falling. A rising trend could prompt pre-discharge geriatric or "
  "palliative-care consultation and earlier escalation to complex-care management, "
  "as a hypothesis for prospective testing. The caveat is that the feature is "
  "partly system-shaped: it runs high for patients whose discharges are delayed "
  "for logistical or social reasons, so treating it as purely clinical risks "
  "conflating social need with medical severity.")
P("The prior-utilization cluster (admission frequency by recency, prior 6-month "
  "admissions, prior readmission count) captures the dose-response between recent "
  "hospital use and near-term return that the readmission literature documents "
  "repeatedly [5,7]. High values identify patients for whom standard discharge "
  "planning has already demonstrably failed: candidates for intensive case "
  "management with early ambulatory follow-up. The equity caveat is consequential: "
  "prior utilization partly reflects outpatient-access barriers, unstable housing, "
  "and insurance gaps, so a score weighting it heavily assigns elevated risk to "
  "exactly the groups whose risk is socially driven; pairing the score with "
  "additional resources, never reduced care, is the only defensible deployment "
  "posture [15].")
P("The abnormal-laboratory rate (the fraction of the index stay's laboratory "
  "results flagged abnormal) is the clearest physiological signal in the final "
  "model: leaving the hospital with substantial unresolved laboratory "
  "derangement marks post-discharge vulnerability, suggesting, as a hypothesis, "
  "that such patients may warrant clinical contact within days rather than "
  "weeks. Notably, the final model carries this signal without any "
  "billing-derived diagnosis coding: the admission's clinical character is "
  "captured through discharge destination, admission type, medications, and "
  "laboratory values instead of diagnosis-related groups, which are excluded by "
  "construction because their discharge-time availability cannot be "
  "established.")
P("Together, these predictors indicate that readmission risk reflects both the "
  "patient's clinical trajectory and the system's response to it. Readmission "
  "reduction is therefore unlikely to be achieved by scoring alone: the "
  "intervention points identified by the model (the disposition decision, the "
  "post-acute hand-off, complex-care enrollment) are system-level processes, "
  "and clinical utility depends on whether they can be acted on, not on "
  "discrimination alone.")
H2("Limitations")
P(f"This is a single-center, retrospective, internally validated study "
  f"from one tertiary academic referral center in one region, whose "
  f"Medicare case mix, discharge resources, and demographic composition "
  f"({FR['White']['admissions']/p['test']['admissions']*100:.1f}% of test "
  f"admissions White) differ from national Medicare; performance "
  f"may not transfer without recalibration, and no clinical-impact evidence exists. "
  "The outcome is within-system readmission; out-of-system readmissions remain "
  "unobservable. Post-discharge death is observable through patients.dod (hospital "
  "and Massachusetts registry sources) but is censored 1 year after each "
  "patient's last discharge by deidentification, and registry dates carry "
  "day-level granularity. Planned readmissions were not excluded from the primary "
  "outcome. End-of-record censoring cannot be verified under date shifting, "
  "though the boundary sensitivity analysis bounds its influence. "
  "Hyperparameters were fixed and not tuned inside the selection folds, so the "
  "procedure is not fully nested and residual selection optimism cannot be "
  "excluded; the consensus set additionally aggregates across folds, so its "
  "specific refit inherits mild optimism relative to the procedure estimate. "
  "The validation partition participated in the consensus-selection folds and "
  "also supplied early stopping and the operating threshold, so it is not "
  "independent of model selection. "
  "The test partition carries historical exposure from earlier development "
  "phases (Methods): although no test-partition outcome entered this "
  "version's selection, encoding, consensus, or threshold steps, the model "
  "family, the engineered candidate-predictor pool, the fixed "
  "hyperparameters, and researcher decisions taken after observing earlier "
  "test results all predate this version, so residual indirect influence on "
  "the tertiary estimate cannot be excluded. Bootstrap intervals quantify "
  "evaluation-sample uncertainty conditional on the fitted models and do "
  "not capture retraining or reselection variability (Methods). The stage-specific domain-composition finding proved "
  "feature-set-dependent and is reported as exploratory only. Secular "
  "drift across 2008 to 2022, including HRRP maturation and the COVID-19 "
  "period, cannot be fully assessed under date shifting; the "
  "era-stratified sensitivity analysis is reassuring but coarse. Social "
  "determinants of health are unavailable in MIMIC-IV. The subgroup disparity "
  "for Black patients is unexplained and unresolved and, like all "
  "subgroup analyses here, was estimated on the historically exposed "
  "test partition without out-of-fold corroboration. Medicare Advantage versus "
  "fee-for-service status is unobservable. The released prototype is a research "
  "demonstration, not a deployed clinical system, and no claim is made that the "
  "model improves outcomes.")
H2("Conclusions")
P(f"A parsimonious {NF}-feature XGBoost model, selected from the complete pool "
  f"of demonstrably discharge-available, nonbilling structured EHR predictors "
  f"under patient-grouped, fold-local preprocessing and feature selection on "
  f"development data, estimates 30-day within-system readmission in "
  f"Medicare-insured adults with strong calibration and per-patient SHAP "
  f"explanations; the selection procedure "
  f"achieved a development-data out-of-fold AUROC of {OOF['auroc']:.4f} and "
  f"the fixed model {MP['auroc']['point']:.4f} on the historically exposed "
  f"test partition, clearly exceeding the reconstructed "
  f"bedside clinical scores on the identical cohort and outcome. "
  f"Discrimination was highest for events occurring during the first day "
  f"after discharge, and the subgroup audit surfaced lower discrimination "
  f"for Black patients, reported openly as a limitation. External validation "
  f"and prospective evaluation are the necessary next steps before any "
  f"clinical use.")

# ============================================================ BACK MATTER
# Order per JMIR AI: Acknowledgments, Funding, Conflicts of Interest,
# Data Availability, Authors' Contributions, Abbreviations, Appendices, Refs.
H1("Acknowledgments")
P("This work began as the capstone project for the Master of Science in Data "
  "Science and Artificial Intelligence at Florida International University. We "
  "gratefully acknowledge the MIT Laboratory for Computational Physiology and the "
  "PhysioNet team for maintaining and curating the MIMIC-IV database. "
  "This study received no external funding.")
H1("Conflicts of Interest")
P("None declared.")
H1("Data Availability")
P("MIMIC-IV v3.1 is available from PhysioNet under a credentialed data use "
  "agreement [21,26] and is not redistributed. All analysis code, the corrected "
  "cohort-construction and reanalysis scripts, the machine-readable "
  "results bundle from which the values in this manuscript are generated "
  "(final_model_v6.json and the companion files enumerated in Multimedia "
  "Appendix 1), the predictor dictionary, and the "
  "prototype source are available in the project repositories: "
  "https://github.com/thiagobandeira1/Medicare-30day-Readmission-MIMIC-IV, "
  "https://github.com/thiagobandeira1/readmission-risk-api, and "
  "https://github.com/thiagobandeira1/riskpath-clinician-companion (tagged "
  "release v6.4-submission). An archival copy of the tagged release of the "
  "primary repository is deposited at Zenodo "
  "(DOI: 10.5281/zenodo.21987702).")
H1("Authors' Contributions")
P("Conceptualization: TB; methodology: TB, AG, CP, AMM; software, formal analysis, "
  "and visualization: TB, AG; data curation: TB, AG; investigation and validation: "
  "TB, AG; writing (original draft): TB; writing (review and editing): AG, CP, AMM; "
  "supervision: CP, AMM. All authors approved the final version. (CRediT taxonomy.)")
H1("Abbreviations")
P("AFT: accelerated failure time; AUROC: area under the receiver operating "
  "characteristic curve; CI: confidence interval; CMS: Centers for Medicare & "
  "Medicaid Services; EHR: electronic health record; FNR: "
  "false-negative rate; HRRP: Hospital Readmissions Reduction Program; "
  "ICD: International Classification of Diseases; ICU: intensive care "
  "unit; IPCW: inverse probability of censoring weighting; "
  "MIMIC: Medical Information Mart for Intensive Care; NLP: natural language "
  "processing; NPV: negative predictive value; OOF: out-of-fold; PACS: picture "
  "archiving and communication system; PPV: positive predictive value; "
  "SD: standard deviation; SHAP: Shapley "
  "additive explanations; TRIPOD: Transparent Reporting of a multivariable "
  "prediction model for Individual Prognosis Or Diagnosis; XGBoost: extreme "
  "gradient boosting.")

H1("Multimedia Appendices")
P(f"Multimedia Appendix 1: predictor audit and selection workbook (207-candidate audit "
  f"with per-predictor missingness, eligible pool, exclusion reasons, final "
  f"{NF}-feature dictionary, RFE trajectories and per-fold selections, "
  f"out-of-fold model comparison with paired differences, full subgroup "
  f"results with CIs, outcome-sensitivity analyses, daily competing-risk event "
  f"counts, and stage-differential estimates). Multimedia Appendix 2: LACE "
  f"and HOSPITAL "
  f"reconstruction tables. Multimedia Appendix 3: race-category "
  f"consolidation with raw "
  f"counts. Multimedia Appendix 4: supplementary figures and development "
  f"history "
  f"(decision curves; same-cohort ROC curves; development-phase results "
  f"including earlier feature versions and model families). Multimedia "
  f"Appendix 5: "
  f"completed TRIPOD+AI checklist. Multimedia Appendix 6: PROBAST+AI "
  f"self-assessment.")

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
 "Biomed Health Inform. 2023;27(4):2071-2082. PMID: 37018684.",
 "Almeida T, Moreno P, Barata C. Prediction of 30-day hospital readmission with "
 "clinical notes and EHR information. In: Pattern Recognition and Image "
 "Analysis (IbPRIA 2025). Cham: Springer; 2025:220-232. "
 "doi:10.1007/978-3-031-99568-2_18",
 "Adisa IT. An integrated framework for explainable, fair, and observable "
 "hospital readmission prediction: development and validation on MIMIC-IV. "
 "Preprint. arXiv:2604.22535. 2026. URL: https://arxiv.org/abs/2604.22535 "
 "[accessed 2026-08-16]",
 "Huang K, Altosaar J, Ranganath R. ClinicalBERT: modeling clinical notes and "
 "predicting hospital readmission. arXiv preprint arXiv:1904.05342. 2020.",
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
 "clinical notes (PhysioNet). 2023.",
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


# ---- build-time citation renumbering by first appearance (review round) ----
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
    for n in range(1, len(refs) + 1):
        if n not in seen:
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
    # verify: first appearances are now strictly increasing
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
    return new_refs


REFS = _renumber_citations(doc, REFS)

for i, ref in enumerate(REFS, 1):
    P(f"{i}. {ref}")
DST = PUB / "Paper JMIR AI Submission FINAL SINGLE-COLUMN.docx"
doc.save(str(DST))
n_words = sum(len(par.text.split()) for par in doc.paragraphs)
# abstract word count (JMIR limit 450): paragraphs between 'Abstract' and
# 'Keywords' headings
_abs, _in = [], False
for par in doc.paragraphs:
    if par.style.name.startswith("Heading"):
        if par.text.strip() == "Abstract":
            _in = True
            continue
        if _in:
            break
    elif _in:
        _abs.append(par.text)
n_abs = sum(len(t.split()) for t in _abs)
print(f"saved {DST.name}  (~{n_words:,} words, {len(REFS)} references, "
      f"abstract {n_abs} words)")
assert n_abs <= 450, f"abstract over JMIR limit (450): {n_abs}"


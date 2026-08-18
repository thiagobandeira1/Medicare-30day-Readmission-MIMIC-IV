# -*- coding: utf-8 -*-
"""Build the complete revised JMIR AI manuscript (v6.4: Armando's audit fixes
— repaired Figures 1-2, verified numeric/citation defects, AMA style pass,
back-matter reorder, Methods trim). Run this file."""
from _build_jmir_v64sc_part1 import *  # noqa

au = M["auroc"]; cv = B2["cv5_patient_grouped"]
p = CT["partitions"]
wb = FM["fairness"]["white_minus_black"]
FR = FM["fairness"]["race"]
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
r = t.add_run("Predicting 30-Day Hospital Readmission in Medicare Patients")
r.bold = True; r.font.size = Pt(15)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("An Interpretable Gradient-Boosting Model on MIMIC-IV v3.1: "
              "Retrospective Development and Internal Validation")
r.italic = True; r.font.size = Pt(11.5)

# JMIR/AMA style: no honorifics; degrees to be appended per author once
# confirmed (AUTHOR input) — "Christian Poellabauer, PhD" etc.
AUTHORS = [("Thiago Bandeira", "tbati006@fiu.edu"),
           ("Armando Gonzalez", "agonz1689@fiu.edu"),
           ("Christian Poellabauer", "cpoellab@fiu.edu"),
           ("Ananda Mohan Mondal", "amondal@cis.fiu.edu")]
atab = doc.add_table(rows=1, cols=4)      # default style: no borders
for j, (name, mail) in enumerate(AUTHORS):
    cell = atab.cell(0, j)
    p1 = cell.paragraphs[0]; p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p1.add_run(name); r.bold = True; r.font.size = Pt(10.5)
    r.font.name = "Times New Roman"
    p2 = cell.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p2.add_run(mail); r.italic = True; r.font.size = Pt(9.5)
    r.font.name = "Times New Roman"

# manuscript date removed per JMIR submission requirements (audit item 3)
for line in ("Knight Foundation School of Computing and Information Sciences",
             "Florida International University, Miami, Florida"):
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(line); r.italic = True; r.font.size = Pt(10.5)
begin_two_columns()

# ============================================================ ABSTRACT
H1("Abstract")
P("Unplanned hospital readmission within 30 days is a quality measure tied to "
  "financial penalties under the Centers for Medicare & Medicaid Services Hospital "
  "Readmissions Reduction Program. Established bedside scores such as LACE and "
  "HOSPITAL rarely exceed an area under the receiver operating characteristic curve "
  "(AUROC) of 0.70 in external validation, and machine-learning reports often "
  "lack leakage-safe validation, calibration, same-cohort baselines, or "
  "fairness auditing.",
  bold_prefix="Background: ")
P("To develop and internally validate an interpretable, calibrated model that "
  "estimates the probability of 30-day all-cause within-system readmission at the "
  "moment of discharge for Medicare-insured adults, using only structured electronic "
  "health record (EHR) data, and to characterize when patients return.",
  bold_prefix="Objective: ")
P(f"We conducted a retrospective study of the Medicare-insured subset of MIMIC-IV "
  f"v3.1. After excluding {CT['index_death_excluded']:,} admissions ending in index "
  f"in-hospital death, the cohort comprised {CT['cohort_v2_admissions']:,} admissions "
  f"from {CT['cohort_v2_patients']:,} patients (readmission prevalence "
  f"{CT['label_v2_prevalence']*100:.1f}%). The outcome was first same-system "
  f"readmission 0<t≤30 days after discharge, all cause. Partitions were "
  f"patient-grouped on subject_id (test: {p['test']['admissions']:,} admissions, "
  f"{p['test']['patients']:,} patients). All 207 candidate predictors were "
  f"audited for discharge-time availability; billing-derived, race-derived, and "
  f"precomputed outcome-derived features were excluded ({N_ELIG} eligible). "
  f"Staged recursive feature elimination ran inside 5 outer patient-grouped "
  f"folds on the development partitions only (3 grouped inner folds each; all "
  f"learned preprocessing fold-local; prespecified parsimony rule); "
  f"features selected in at least 3 of 5 folds formed the consensus set of the "
  f"final extreme gradient boosting (XGBoost) classifier. The primary estimate "
  f"is the out-of-fold performance of the complete selection procedure. "
  f"Uncertainty used patient-cluster bootstrap resampling; the operating "
  f"threshold was development-derived. Timing was assessed with a "
  f"competing-risks reformulation and landmark models.",
  bold_prefix="Methods: ")
_fx = EX["fixed31_vs_fixed142_paired_cv"]
P(f"The selection procedure (fold-specific feature sets of 27-38 predictors) "
  f"achieved a development-data out-of-fold AUROC of {OOF['auroc']:.4f} "
  f"(95% cluster CI {OOF['ci95'][0]:.4f}-{OOF['ci95'][1]:.4f}; primary). The "
  f"fixed {NF}-feature consensus model reached {CCV['mean']:.4f} "
  f"(SD {CCV['sd']:.4f}) under grouped cross-validation (secondary; the "
  f"consensus set was selected on the same development data) and "
  f"{MP['auroc']['point']:.4f} (95% CI {MP['auroc']['ci95'][0]:.4f}-"
  f"{MP['auroc']['ci95'][1]:.4f}) on a historically exposed test partition "
  f"not used in this version's feature selection. Calibration was "
  f"strong (slope {MP['slope']['point']:.2f}, expected calibration error "
  f"{MP['ece']['point']:.3f}). The model outperformed reconstructed LACE "
  f"({BSE['lace_auroc']:.4f}) by {BSE['uplift_lace']:.4f} and 12-month "
  f"HOSPITAL ({BSE['hospital12_auroc']:.4f}) by "
  f"{BSE['uplift_hospital12']:.4f} AUROC (both "
  f"{fmt_p(max(BSE['uplift_lace_p'], BSE['uplift_hospital12_p']))}). "
  f"Time-dependent AUROC was highest on the first day after discharge "
  f"({FM['survival']['daily_tdauc'][0]:.3f}). Discrimination was lower for "
  f"Black patients (AUROC {FR['Black']['auroc']:.3f}) than White patients "
  f"({FR['White']['auroc']:.3f}; difference {wb['point']:.3f}, 95% CI "
  f"{wb['ci95'][0]:.3f}-{wb['ci95'][1]:.3f}; "
  f"{fmt_p(wb['p_bootstrap'])}) despite race not being a model input.",
  bold_prefix="Results: ")
P(f"A parsimonious {NF}-feature XGBoost model, selected from demonstrably "
  "discharge-available, nonbilling predictors under patient-grouped, fold-local "
  "preprocessing and feature selection, supports calibrated and interpretable "
  "estimation of 30-day within-system readmission risk in Medicare-insured "
  "adults. This is a research prediction model with a demonstration prototype; "
  "external validation and prospective evaluation are required before any "
  "clinical use.",
  bold_prefix="Conclusions: ")
H2("Keywords")
P("hospital readmission; Medicare; MIMIC-IV; machine learning; gradient boosting; "
  "calibration; algorithmic fairness; survival analysis; clinical prediction model; "
  "electronic health records")

# ============================================================ INTRODUCTION
H1("Introduction")
H2("Background")
P("Unplanned 30-day readmissions burden the United States health system; the Centers "
  "for Medicare & Medicaid Services (CMS) estimates the annual direct cost of "
  "Medicare readmissions above $26 billion [1], and the Hospital Readmissions "
  "Reduction Program has tied reimbursement to risk-adjusted readmission performance "
  "since 2013 [1,2]. Readmissions are associated with in-hospital mortality, "
  "functional decline, and caregiver strain. Bedside indices remain the operational "
  "default: the LACE index combines length of stay, acuity, comorbidity, and "
  "emergency-department use [3], and the HOSPITAL score targets potentially avoidable "
  "readmission [4]; external validations of such scores rarely exceed an AUROC of "
  "0.70 [5,6]. Systematic reviews of readmission models report similar ceilings for "
  "regression-based approaches and identify recurring methodological weaknesses: "
  "leakage-prone splits, absent calibration, and unexamined subgroup performance "
  "[5,7,8].")
P("Machine learning on structured EHR data can exceed these ceilings, and "
  "gradient-boosted decision trees remain the strongest general-purpose learners on "
  "tabular clinical data [9-12]. Yet reported discrimination is not deployability: "
  "clinical use requires patient-level leakage prevention, calibrated absolute "
  "probabilities [13], interpretable per-patient explanations [14], comparison with "
  "clinical scores on the same cohort rather than published values, fairness "
  "auditing [15,16], and reporting aligned with TRIPOD+AI [17] and PROBAST [18]. "
  "Race-aware modeling requires particular care following the removal of race "
  "coefficients from clinical equations such as the estimated glomerular filtration "
  "rate [19,20].")
H2("Prior Work on MIMIC-IV and Study Rationale")
P("Three recent studies predict 30-day readmission on the same MIMIC-IV database "
  "[21]. A multimodal spatiotemporal graph neural network combining EHR time series "
  "with chest radiographs reported an AUROC of 0.791 on a radiograph-selected subset "
  "of 14,532 admissions [22]; a graph model over discharge summaries reported 0.727 "
  "on 303,571 all-adult admissions [23]; and a 26-feature XGBoost framework reported "
  "0.696 on 415,231 all-adult admissions [24]. ClinicalBERT reported approximately "
  "0.714 on a MIMIC-III cohort using discharge notes [25]. Because cohorts and "
  "outcome definitions differ, these are reference points rather than head-to-head "
  "comparisons; none combines leakage-safe patient-grouped validation, calibration "
  "analysis, same-cohort clinical baselines, subgroup fairness auditing, and "
  "readmission-timing analysis in one evaluation.")
H2("Research Questions and Objectives")
P("Three research questions, carried forward from the project's inception, "
  "organize the study. RQ1: which features are most predictive of 30-day "
  "readmission in Medicare-insured adults? RQ2: which modeling approach achieves "
  "the best predictive performance under a leakage-safe protocol? RQ3: can "
  "interpretable machine-learning outputs - global and patient-level Shapley "
  "additive explanations (SHAP) "
  "- provide actionable insight for discharge teams? "
  "Operationally, the study aimed to (1) develop and internally validate a "
  "calibrated, interpretable model of 30-day all-cause within-system readmission "
  "using only structured EHR fields available at discharge; (2) compare it against "
  "LACE and HOSPITAL reconstructed on the identical test partition; (3) "
  "characterize when patients return using a time-to-event reformulation and "
  "stage-specific landmark models; (4) audit subgroup performance across age, sex, "
  "and race with cluster-level uncertainty; and (5) release the pipeline and a "
  "demonstration prototype for reproducibility.")

# ============================================================ METHODS
H1("Methods")
H2("Study Design and Data Source")
P("We conducted a retrospective prediction-model development and internal-validation "
  "study, reported in line with TRIPOD+AI [17]. The data source is MIMIC-IV v3.1 "
  "[21,26], a deidentified EHR database of 546,028 hospitalizations at Beth Israel "
  "Deaconess Medical Center (Boston, MA) admitted between 2008 and 2022. Dates are "
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
  f"were younger, eg, disability entitlement) and MIMIC-IV cannot distinguish "
  f"fee-for-service from Medicare Advantage, so the cohort is not a CMS HRRP "
  f"regulatory cohort.")
P("The outcome was the first subsequent hospitalization of the same patient in the "
  "same health system beginning more than 0 and up to 30 days (day 30 inclusive) "
  "after index discharge, all cause, and irrespective of how that subsequent "
  "admission ended; a readmission followed by in-hospital death is a readmission. "
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
  f"patient appears in more than one partition.")
P("Validation hierarchy and test-set transparency: earlier development phases, "
  "retained in the project repository, scored model variants on the original test "
  "partition multiple times; we therefore describe the test data as a "
  "historically exposed internal evaluation partition rather than a pristine "
  "holdout. Three tiers of evidence are reported, in order of authority: "
  "(1) primary - the out-of-fold performance of the complete selection procedure "
  "under 5 outer patient-grouped folds over the development partitions "
  "(training plus validation), in which every admission is scored by a model "
  "whose preprocessing, target encodings, feature selection, and fitting never "
  "saw that patient; this estimates the procedure, with fold-specific feature "
  "sets, not the fixed consensus model; (2) secondary - 5-fold patient-grouped "
  "cross-validation of the fixed consensus feature set on the same development "
  "data with fold-local encodings; the fixed set aggregates selection "
  "information across folds, so this estimate carries disclosed optimism; "
  "(3) tertiary - a single evaluation on the historically exposed test "
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
  "rejected because per-patient date shifting removes cross-patient temporal "
  "ordering. Independent external validation remains necessary.")
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
  f"artifacts - diagnosis-related groups, ICD diagnosis and procedure codes and "
  f"their derivatives (including comorbidity indices and procedure flags with "
  f"clinical names whose provenance was verified in the feature-build code) - "
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
  f"outer-evaluation fold. The final consensus predictor set was defined from "
  f"these development-data selections as the consensus of features selected "
  f"in at least 3 of 5 folds ({NF} features; {N_UNANIMOUS} selected in all "
  f"5); no test-partition outcome was used in this construction. Missing values were handled "
  f"natively by the histogram-based learner, preserving informative "
  f"missingness.")
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
P("LACE [3] and HOSPITAL [4] were re-implemented on the identical test partition. "
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
  "failure time (AFT) objective [28] - a single-event survival model evaluated under "
  "censoring, not itself a competing-risk model - and penalized cause-specific "
  "Cox regression as the linear reference, with competing deaths censored at "
  "their death time. Evaluation used Harrell C [29] and inverse-probability-"
  "of-censoring-weighted time-dependent AUROC at each day 1-29 [30], with "
  "patient-cluster bootstrap uncertainty bands and daily event counts reported. "
  "Three landmark models covered days 1-7, 8-14, and 15-30, each trained only on "
  "patients still at risk when the stage opens - patients already readmitted or "
  "already deceased are removed from the risk set, not counted as nonevents; "
  "stage windows differ in length and prevalence by design, so stage results are "
  "descriptive. Stage-specific TreeSHAP attribution shares are exploratory.")
H2("Statistical Analysis")
P(f"Because patients contribute multiple admissions, all confidence intervals "
  f"and P values use cluster bootstrap resampling at the patient level with "
  f"percentile intervals (1,000 resamples for the test metric panel; 800 for "
  f"subgroup CIs; 500 for out-of-fold model comparisons; every contrast for "
  f"which a P value is reported was computed with 5,000 resamples). P values "
  f"are empirical two-sided bootstrap probabilities with a plus-one "
  f"correction, so the smallest reportable value at 5,000 resamples is "
  f"P<.001; no P value is reported at a precision the resample count cannot "
  f"support. Model-to-model AUROC contrasts (final model vs LACE, HOSPITAL, "
  f"and out-of-fold comparators) are paired: both scores are evaluated on the "
  f"identical admissions of each bootstrap draw and the difference is "
  f"resampled directly. The White-Black AUROC contrast compares disjoint "
  f"subgroups and is therefore not paired; patients were resampled within "
  f"each subgroup independently and the difference of subgroup AUROCs was "
  f"computed in each draw. No formal equivalence or noninferiority test was "
  f"prespecified; near-ties are described against the 0.002 tolerance used in "
  f"the prespecified parsimony rule, not as statistical equivalence. All "
  f"bootstrap intervals were computed by resampling patients and their "
  f"stored prediction-outcome pairs; feature selection and model fitting "
  f"were not repeated within bootstrap samples, so the intervals quantify "
  f"evaluation-sample uncertainty conditional on the fitted models and do "
  f"not capture the variability of retraining or reselection. "
  f"Discrimination used AUROC and average precision; calibration used "
  f"the Brier score, expected calibration error over deciles, and logistic "
  f"recalibration slope and intercept [13]. The operating threshold "
  f"({THRV:.3f}) was the Youden point on the validation partition - a "
  f"development-derived threshold, because validation patients also "
  f"participated in the consensus-selection folds - and was applied "
  f"unchanged to the test partition; sensitivity, specificity, positive and "
  f"negative predictive values are reported with cluster-bootstrap CIs. The "
  f"White-Black AUROC difference was tested directly by bootstrap rather than by "
  f"CI overlap. Decision-curve analysis [31] compared net benefit against treat-all "
  f"and treat-none across threshold probabilities 0.05-0.40, with a capacity view "
  f"(top-k% flagged). Analyses used Python 3.11-3.12 (XGBoost, LightGBM, "
  f"scikit-learn, lifelines, scikit-survival, SHAP); exact versions are pinned in "
  f"the repository.")
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
P("[AUTHOR ACTION REQUIRED - the following describes the standard MIMIC-IV basis "
  "and must be confirmed, with credential and training records, before submission.] "
  "MIMIC-IV is a deidentified, publicly available database; its creation was "
  "approved by the institutional review boards of the Massachusetts Institute of "
  "Technology and Beth Israel Deaconess Medical Center with a waiver of informed "
  "consent [21,26]. The authors accessed the data under the PhysioNet Credentialed "
  "Health Data Use Agreement after completing required human-subjects training "
  "[CONFIRM: credential holder names and CITI completion records]. No attempt was "
  "made to reidentify individuals; no row-level data are redistributed. This "
  "work is a secondary analysis of deidentified data. [CONFIRM whether FIU IRB "
  "determination or exemption was obtained; insert protocol number or state "
  "that review was not required.] The "
  "public demonstration prototype accepts only synthetic or manually entered "
  "values, displays a research-only, no-real-patient-data notice, and stores no "
  "submitted data.")

# ============================================================ RESULTS
H1("Results")
H2("Cohort")
P(f"Figure 1 shows the cohort flow. Of {CT['original_admissions']:,} Medicare "
  f"admissions, {CT['index_death_excluded']:,} ended in index in-hospital death and "
  f"were excluded, leaving {CT['cohort_v2_admissions']:,} admissions from "
  f"{CT['cohort_v2_patients']:,} patients; {CT['label_v2_events']:,} "
  f"({CT['label_v2_prevalence']*100:.1f}%) were followed by a within-system "
  f"readmission within 30 days. Mean time to readmission among events was "
  f"{SV['mean_days_to_readmission']:.1f} days.")
figure(FIG / "r2_fig1_flow_col.png",
       "Figure 1. Cohort flow diagram (STROBE-style) with patient-grouped partitions.")
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
  f"Jaccard {STAB['jaccard_mean']:.2f}; {N_UNANIMOUS} features selected in "
  f"all 5 folds; Figure 3), and the consensus rule (at least 3 of 5 folds) "
  f"yielded the {NF}-feature final set. Out-of-fold over the development "
  f"data, the selection procedure achieved AUROC {_cmp['rfe']['oof_auroc']:.4f} "
  f"(95% cluster CI {_cmp['rfe']['auroc_ci95_cluster'][0]:.4f}-"
  f"{_cmp['rfe']['auroc_ci95_cluster'][1]:.4f}). Paired patient-cluster "
  f"bootstrap differences on identical observations (each computed as the "
  f"comparator minus the RFE procedure, so a negative value favors the "
  f"procedure) were "
  f"{_pd_all['point']:+.4f} ({nbci(_pd_all['ci95'][0], _pd_all['ci95'][1])}) "
  f"for the full {N_ELIG}-feature eligible model "
  f"({_cmp['all_eligible']['oof_auroc']:.4f}) and {_pd_f50['point']:+.4f} "
  f"({nbci(_pd_f50['ci95'][0], _pd_f50['ci95'][1])}) for the "
  f"prior 50-feature subset ({_cmp['f50']['oof_auroc']:.4f}) - absolute "
  f"differences smaller than the 0.002 tolerance used in the prespecified "
  f"parsimony rule, described as similar discrimination rather than "
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
  f"complete RFE-procedure estimate remains the primary evidence. "
  f"The prior 66-feature set containing billing predictors "
  f"and the excluded encoding performed worst among the boosted models under "
  f"leakage-safe evaluation ({_cmp['f66']['oof_auroc']:.4f}; paired difference "
  f"{_pd_f66['point']:+.4f}, {nbci(_pd_f66['ci95'][0], _pd_f66['ci95'][1])}), "
  f"and regularized logistic regression on the "
  f"same eligible inputs reached {_cmp['logit']['oof_auroc']:.4f}. Because a "
  f"negligible absolute difference does not outweigh a materially simpler set, "
  f"the {NF}-feature consensus model was adopted. It contains no "
  f"billing-derived features by construction, so no separate "
  f"coding-availability sensitivity model is required; the earlier "
  f"development-phase sensitivity analyses are retained in Multimedia "
  f"Appendix 4.")
figure(FIG5 / "v5_rfe_curve.png",
       "Figure 2. Leakage-safe staged recursive feature elimination on "
       "development data: mean inner-validation AUROC versus retained "
       f"features, per outer fold; dashed line marks the {NF}-feature "
       "consensus.")
figure(FIG5 / "v5_stability.png",
       "Figure 3. Feature-selection frequency across the 5 outer folds; teal "
       "bars form the consensus set (selected in at least 3 of 5 folds).")
H2("Discrimination and Calibration of the Final Model")
P(f"The consensus model's grouped cross-validation AUROC on development data "
  f"was {CCV['mean']:.4f} (SD {CCV['sd']:.4f}; folds "
  f"{', '.join(f'{x:.4f}' for x in CCV['folds'])}; the fixed feature set "
  f"aggregates selection information across folds and so carries mild "
  f"optimism relative to the primary out-of-fold estimate). On "
  f"the historically exposed test partition ({p['test']['admissions']:,} "
  f"admissions, {p['test']['patients']:,} patients; tertiary evidence), the "
  f"refit model reached AUROC {MP['auroc']['point']:.4f} (95% cluster CI "
  f"{MP['auroc']['ci95'][0]:.4f}-{MP['auroc']['ci95'][1]:.4f}) and average "
  f"precision {MP['ap']['point']:.3f} (95% CI {MP['ap']['ci95'][0]:.3f}-"
  f"{MP['ap']['ci95'][1]:.3f}) against a "
  f"{p['test']['events']/p['test']['admissions']*100:.1f}% prevalence. "
  f"Calibration was strong: Brier {MP['brier']['point']:.4f}, expected "
  f"calibration error {MP['ece']['point']:.3f} (95% CI "
  f"{MP['ece']['ci95'][0]:.3f}-{MP['ece']['ci95'][1]:.3f}), slope "
  f"{MP['slope']['point']:.2f} (95% CI {MP['slope']['ci95'][0]:.2f}-"
  f"{MP['slope']['ci95'][1]:.2f}), intercept {MP['intercept']['point']:.2f} "
  f"(Figure 4). At the development-derived threshold of {THRV:.3f}: sensitivity "
  f"{MP['sensitivity']['point']:.3f} (95% CI {MP['sensitivity']['ci95'][0]:.3f}-"
  f"{MP['sensitivity']['ci95'][1]:.3f}), specificity "
  f"{MP['specificity']['point']:.3f}, positive predictive value "
  f"{MP['ppv']['point']:.3f}, negative predictive value "
  f"{MP['npv']['point']:.3f}, and false-negative rate {MP['fnr']['point']:.3f}, "
  f"each with cluster-bootstrap CIs in the machine-readable results file and "
  f"Multimedia Appendix 1.")
figure(FIG5 / "v5_roc_cal.png",
       "Figure 4. Receiver operating characteristic, precision-recall, "
       f"calibration, and confusion-matrix panels for the final {NF}-feature "
       "model (historically exposed test partition; tertiary evidence).")
H2("Outcome-Definition and Cohort Sensitivity Analyses")
_sa = SENS["sameday_as_readmission"]
_su = SENS["unplanned_only"]
_sn = SENS["nonelective_index_only"]
P(f"Four sensitivity analyses probe the outcome and cohort rules. (1) Counting "
  f"the transfer/continuation band (next admission beginning at, or up to 2 "
  f"days before, the index discharge timestamp: −2<Δ≤0 days; "
  f"{EX['transfer_band']['n_band_by_partition']['test']} test admissions) as "
  f"readmissions raised test events from "
  f"{_sa['test_events_primary']:,} to {_sa['test_events_alt']:,} "
  f"({_sa['test_prevalence_alt']*100:.1f}% prevalence); a model refit under "
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
  f"{BD5['auroc_safe']:.4f}. None of these variations alters the study's "
  f"conclusions.")
H2("Same-Cohort Clinical Baselines")
P(f"On the identical test partition the final model (AUROC "
  f"{MP['auroc']['point']:.4f}) outperformed reconstructed LACE "
  f"({BSE['lace_auroc']:.4f}) by {BSE['uplift_lace']:.4f} AUROC (95% CI "
  f"{BSE['uplift_lace_ci'][0]:.4f}-{BSE['uplift_lace_ci'][1]:.4f}; "
  f"{fmt_p(BSE['uplift_lace_p'])}; paired, 5,000 draws) and "
  f"HOSPITAL, rebuilt with its original 12-month prior-admission window "
  f"({BSE['hospital12_auroc']:.4f}), by {BSE['uplift_hospital12']:.4f} (95% CI "
  f"{BSE['uplift_hospital12_ci'][0]:.4f}-{BSE['uplift_hospital12_ci'][1]:.4f}; "
  f"{fmt_p(BSE['uplift_hospital12_p'])}) (Multimedia Appendix 4). Both adapted scores fall below their "
  f"published values, consistent with MIMIC-IV capturing only within-system "
  f"prior utilization, which attenuates their utilization components; the "
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
  f"the tie was resolved as death first - a documented rule, adopted during "
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
  f"{st1['auroc']:.4f} for days 1-7 ({st1['events']:,} events among "
  f"{st1['at_risk']:,} at risk), {st2['auroc']:.4f} for days 8-14 "
  f"({st2['events']:,}/{st2['at_risk']:,}), and {st3['auroc']:.4f} for days "
  f"15-30 ({st3['events']:,}/{st3['at_risk']:,}) - discrimination improves for "
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
_llist = "; ".join(f"{flabel(r['feature'])} ({r['diff_pp']:.1f})"
                   for r in _late)
P(f"At the individual-feature level, however, early and late returns do differ "
  f"in character (Figure 6, panel B; "
  f"{'all 10' if len(_sig) == 10 else f'{len(_sig)} of the 10'} leading "
  f"differentials have 95% cluster-bootstrap CIs excluding zero). Signals "
  f"disproportionately important for a return within days 1-7 are "
  f"predominantly states of the index discharge itself: {_elist}. Signals "
  f"disproportionately important for returns in days 15-30 are predominantly "
  f"the longer-horizon utilization aggregates: {_llist}. These are "
  f"exploratory, descriptive contrasts in model reliance, offered as "
  f"hypotheses about differently timed follow-up rather than causal claims.")
figure(FIG5 / "v5_stage_combined.png",
       f"Figure 6. Attribution across post-discharge windows, final "
       f"{NF}-feature model. (A) Share of total model attribution by clinical "
       "domain. (B) Features whose attribution share differs most between the "
       "earliest (days 1-7) and latest (days 15-30) windows; positive values "
       "indicate greater early importance. Error bars are 95% patient-cluster "
       "bootstrap CIs.", width=3.1)
H2("Fairness and Subgroup Performance")


def _ci3(e, key):
    c = e.get(f"{key}_ci95")
    v = e.get(key)
    if v is None:
        return "-"
    if key == "cal_slope":
        return f"{v:.2f} ({c[0]:.2f}-{c[1]:.2f})" if c else f"{v:.2f}"
    return f"{v:.3f} ({c[0]:.3f}-{c[1]:.3f})" if c else f"{v:.3f}"


rows = [["Group", "Patients / admissions / events", "Prev.",
         "AUROC (95% CI)", "Sensitivity (95% CI)", "FNR (95% CI)",
         "PPV", "Brier", "Cal. slope (95% CI)"]]
for axis, order_ in (("race", ["White", "Black", "Hispanic/Latino", "Asian",
                               "Other/Unknown"]),
                     ("sex", ["Female", "Male"]),
                     ("age_band", ["<65", "65-74", "75-84", "85+"])):
    for g in order_:
        e = FM["fairness"][axis][g]
        rows.append([g,
                     f"{e['patients']:,} / {e['admissions']:,} / "
                     f"{e['events']:,}",
                     f"{e['prevalence']*100:.1f}%",
                     _ci3(e, "auroc"), _ci3(e, "sensitivity"),
                     _ci3(e, "fnr"),
                     f"{e['ppv']:.3f}" if e.get("ppv") is not None else "-",
                     f"{e['brier']:.3f}" if e.get("brier") is not None else "-",
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
  f"contributor to subgroup differences. Discrimination was stable across age bands "
  f"and sex but lower for Black patients than White patients: difference "
  f"{wb['point']:.4f} (95% CI {wb['ci95'][0]:.4f}-{wb['ci95'][1]:.4f}; "
  f"{fmt_p(wb['p_bootstrap'])}, stratified subgroup bootstrap with 5,000 "
  f"draws, patients resampled within each subgroup) (Figure 7, Table 1). "
  f"Calibration slopes across race groups ranged from "
  f"{_smin[1]['cal_slope']:.2f} ({_smin[0]}; 95% CI "
  f"{_smin[1]['cal_slope_ci95'][0]:.2f}-{_smin[1]['cal_slope_ci95'][1]:.2f}) to "
  f"{_smax[1]['cal_slope']:.2f} ({_smax[0]}; 95% CI "
  f"{_smax[1]['cal_slope_ci95'][0]:.2f}-{_smax[1]['cal_slope_ci95'][1]:.2f}); "
  f"per-group calibration summaries with uncertainty appear in Table 1. Race "
  f"is not a model input; a validation-partition ablation showed the "
  f"previously selected race encoding contributed nothing (Methods), and its "
  f"removal did not close the gap. Because the disparity is one of ranking, "
  f"subgroup-specific recalibration cannot repair it; candidate remedies are "
  f"representation-aware training, improved measurement, reweighting, or model "
  f"redevelopment, and none is claimed here.")
begin_full_width()
caption("Table 1. Subgroup performance of the final model on the test "
        f"partition at the development-derived threshold ({THRV:.3f}). CIs "
        "are 95% patient-cluster bootstrap percentile intervals; PPV and "
        "Brier are shown as point estimates here, and the complete metric "
        "set (including specificity, NPV, and CIs for every metric) appears "
        "in Multimedia Appendix 1. AUROC: area under the receiver operating "
        "characteristic curve; FNR: false-negative rate; PPV: positive "
        "predictive value; Prev.: outcome prevalence; Cal.: calibration.",
        keep_with_next=True)
table(rows, font_size=8,
      widths=[0.72, 1.12, 0.45, 0.95, 0.95, 0.9, 0.45, 0.45, 0.79])
begin_two_columns()
figure(FIG5 / "v5_fairness.png",
       "Figure 7. Subgroup AUROC of the final model with 95% patient-cluster "
       "bootstrap CIs.")
H2("Interpretability via SHAP")
_top3 = ", ".join(f"{flabel(s['feature'])} ({s['feature']}; mean |SHAP| "
                  f"{s['mean_abs_shap']:.3f})" for s in SH[:3])
_next4 = ", ".join(flabel(s["feature"]) for s in SH[3:7])
P(f"Figure 8 ranks the final model's predictors by mean absolute SHAP value on "
  f"the test partition. The strongest signals are {_top3}; {_next4} complete "
  f"the top seven. Readmission risk is multifactorial - no single feature "
  f"dominates - and the leading features map to observables a discharge team "
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
  f"treat-none across an exploratory threshold range of roughly 0.10-0.40, a "
  f"range not yet grounded in operational evidence (Multimedia Appendix 4). "
  f"Under capacity constraints, flagging the top 5% of test admissions captured "
  f"{CAP['top_5']['captured']*100:.0f}% of all readmissions at a positive "
  f"predictive value of {CAP['top_5']['ppv']:.2f}; the top 10% captured "
  f"{CAP['top_10']['captured']*100:.0f}% at {CAP['top_10']['ppv']:.2f}; the top "
  f"20% captured {CAP['top_20']['captured']*100:.1f}% at "
  f"{CAP['top_20']['ppv']:.2f}. Whether acting on these flags improves outcomes "
  f"is untested and requires prospective evaluation.")

# ============================================================ DISCUSSION
H1("Discussion")
H2("Principal Findings")
P(f"In {CT['cohort_v2_admissions']:,} Medicare-insured adult admissions, a "
  f"{NF}-feature gradient-boosted model - selected from the complete pool of "
  f"demonstrably discharge-available, nonbilling predictors under patient-"
  f"grouped, fold-local preprocessing and feature selection on development "
  f"data only - estimated 30-day within-system readmission with strong "
  f"calibration; the selection procedure achieved a development-data "
  f"out-of-fold AUROC of {OOF['auroc']:.4f} (primary estimate) and the fixed "
  f"model {MP['auroc']['point']:.4f} on the historically exposed test "
  f"partition. Four findings carry the study. First, the parsimonious "
  f"procedure showed no material decrement in discrimination against the full "
  f"eligible pool under leakage-safe "
  f"out-of-fold comparison, and the descriptive fixed-set comparison on "
  f"identical development folds showed a negligible difference "
  f"({EX['fixed31_vs_fixed142_paired_cv']['paired_diff_31_minus_142']:+.4f}, "
  f"95% CI {EX['fixed31_vs_fixed142_paired_cv']['ci95'][0]:.4f} to "
  f"{EX['fixed31_vs_fixed142_paired_cv']['ci95'][1]:.4f}; potentially "
  f"optimistic for the consensus set, which was selected on the same data), "
  f"while the previous billing-containing model performed worse under "
  f"leakage-safe evaluation - most of the usable signal is captured by a "
  f"compact, operationally available predictor set. Second, the model clearly outperforms the "
  f"adapted LACE and HOSPITAL scores on the identical cohort and outcome. "
  f"Third, discrimination is highest for readmissions occurring during the "
  f"first day after discharge - the window in which an intervention initiated "
  f"at discharge would need to act. Fourth, the model performs less well for "
  f"Black patients "
  f"despite race not being an input, a disparity we quantify directly and "
  f"report as an open limitation.")
H2("Comparison With Prior Work")
P(f"On the same database, published models report 0.791 using chest radiographs on "
  f"a selected 14,532-admission subset [22], 0.727 using discharge notes [23], and "
  f"0.696 using 26 structured features [24]. The RFE procedure, retaining "
  f"27-38 nonbilling structured predictors across folds, achieved a "
  f"development-data out-of-fold AUROC of {OOF['auroc']:.4f} on a "
  f"Medicare-insured cohort, and the final fixed {NF}-feature consensus model "
  f"reached {MP['auroc']['point']:.4f} on the historically exposed test "
  f"partition - numerically above the notes-based and structured-feature "
  f"reports and near the multimodal result, while scoring every admission "
  f"rather than an imaging-selected subset. But cohorts, outcome definitions, "
  f"modalities, and validation designs differ across these "
  f"studies, so no superiority claim is made. Cohorts and outcome definitions differ, so "
  f"these are reference points; our controlled comparison is the same-cohort "
  f"baseline analysis, where rebuilding HOSPITAL with its original 12-month "
  f"prior-admission window (AUROC "
  f"{RB['C_hospital_12m']['hospital_12m_auroc']:.4f} vs "
  f"{RB['C_hospital_12m']['hospital_6m_proxy_auroc']:.4f} under the 6-month "
  f"proxy) does not alter the conclusion. Table 2 assembles the comparison. "
  f"Against the broader literature, the result is consistent "
  f"with evidence that boosted trees remain highly competitive on structured "
  f"clinical data [9,12] and that careful validation design matters more than "
  f"architecture [5,7,17].")
_t2 = [["Study / score", "Data", "AUROC", "Notes"],
       ["LACE (adapted) [3]",
        "Same cohort - this study's MIMIC-IV Medicare test partition",
        f"{BSE['lace_auroc']:.4f}", "Controlled: same patients, same outcome"],
       ["HOSPITAL (adapted, 12-mo) [4]",
        "Same cohort - this study's MIMIC-IV Medicare test partition",
        f"{BSE['hospital12_auroc']:.4f}",
        "Controlled: same patients, same outcome"],
       ["Adisa 2026 [24]", "Same database - MIMIC-IV (all-adult, 415,231)",
        "0.696", "26 structured features; different cohort/definition"],
       ["Almeida 2025 [23]", "Same database - MIMIC-IV (all-adult, 303,571)",
        "0.727", "Discharge notes + graph; different cohort/definition"],
       ["Tang 2023 [22]",
        "Same database - MIMIC-IV (radiograph-selected, 14,532)",
        "0.791", "Multimodal, chest X-rays; imaging-selected subset"],
       ["ClinicalBERT [25]", "Different database - MIMIC-III", "≈0.714",
        "BERT over discharge notes; different era"],
       ["This study: RFE selection procedure",
        f"MIMIC-IV Medicare, development partitions "
        f"({CMP5['_design']['n_dev']:,} admissions)",
        f"{OOF['auroc']:.4f} (OOF)",
        "Primary; fold-specific sets of 27-38 features; leakage-safe "
        "out-of-fold"],
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
        "and are labeled by evidence tier. OOF: out-of-fold; RFE: recursive "
        "feature elimination.", keep_with_next=True)
table(_t2, font_size=8.5, widths=[1.7, 2.2, 0.95, 1.95])
begin_two_columns()
H2("Exclusion of Clinical Notes and Imaging as a Design Decision")
P("MIMIC-IV offers deidentified discharge summaries and radiology reports "
  "(MIMIC-IV-Note) and linked radiographs (MIMIC-CXR) as separately credentialed "
  "companions [32,33]; their exclusion was a design decision. Structured fields are "
  "available in any EHR at discharge, whereas notes and imaging require NLP or "
  "PACS integration many sites cannot provide; discharge summaries are authored at "
  "or after discharge and routinely reference planned follow-up, so they risk "
  "importing the outcome into the features; and restricting to imaged admissions "
  "would shrink and bias the cohort, as the radiograph-selected subset of [22] "
  "illustrates. Whether unstructured modalities can raise the structured-data "
  "ceiling, and at what cost to leakage safety, is a planned follow-up study.")
H2("Answers to the Research Questions")
_rq1 = ", ".join(flabel(s["feature"]) for s in SH[:7])
P(f"RQ1 (most predictive features): {_rq1} lead the final model (Figure 8); "
  f"{N_UNANIMOUS} features were "
  f"selected in every outer fold, indicating that the predictive core is stable "
  f"rather than an artifact of one selection run. RQ2 (best modeling approach): "
  f"gradient boosting - the XGBoost-based RFE procedure achieved an "
  f"out-of-fold AUROC of {OOF['auroc']:.4f} against "
  f"{CMP5['logit']['oof_auroc']:.4f} for the regularized logistic-regression "
  f"procedure on identical inputs - with LightGBM, CatBoost, and HistGradientBoosting "
  f"within 0.0035 during development and blending adding nothing over the best "
  f"single model. RQ3 (interpretable output): global SHAP identifies what the "
  f"model relies on, per-patient additive decompositions convert each score into "
  f"a ranked list of that patient's contributing factors at serving time, and "
  f"the stage-specific analysis adds when discrimination is strongest across the "
  f"post-discharge window - providing discharge teams with an explanation "
  f"alongside the risk estimate.")
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
  "palliative-care consultation and earlier escalation to complex-care management "
  "- as a hypothesis for prospective testing. The caveat is that the feature is "
  "partly system-shaped: it runs high for patients whose discharges are delayed "
  "for logistical or social reasons, so treating it as purely clinical risks "
  "conflating social need with medical severity.")
P("The prior-utilization cluster (admission frequency by recency, prior 6-month "
  "admissions, prior readmission count) captures the dose-response between recent "
  "hospital use and near-term return that the readmission literature documents "
  "repeatedly [5,7]. High values identify patients for whom standard discharge "
  "planning has already demonstrably failed - candidates for intensive case "
  "management with early ambulatory follow-up. The equity caveat is consequential: "
  "prior utilization partly reflects outpatient-access barriers, unstable housing, "
  "and insurance gaps, so a score weighting it heavily assigns elevated risk to "
  "exactly the groups whose risk is socially driven; pairing the score with "
  "additional resources, never reduced care, is the only defensible deployment "
  "posture [15].")
P("The abnormal-laboratory rate - the fraction of the index stay's laboratory "
  "results flagged abnormal - is the clearest physiological signal in the final "
  "model: leaving the hospital with substantial unresolved laboratory "
  "derangement marks post-discharge vulnerability, suggesting - as a hypothesis "
  "- that such patients may warrant clinical contact within days rather than "
  "weeks. Notably, the final model carries this signal without any "
  "billing-derived diagnosis coding: the admission's clinical character is "
  "captured through discharge destination, admission type, medications, and "
  "laboratory values instead of diagnosis-related groups, which are excluded by "
  "construction because their discharge-time availability cannot be "
  "established.")
P("Together, these predictors indicate that readmission risk reflects both the "
  "patient's clinical trajectory and the system's response to it. Readmission "
  "reduction is therefore unlikely to be achieved by scoring alone: the "
  "intervention points identified by the model - the disposition decision, the "
  "post-acute hand-off, complex-care enrollment - are system-level processes, "
  "and clinical utility depends on whether they can be acted on, not on "
  "discrimination alone.")
H2("Limitations")
P("This is a single-center, retrospective, internally validated study; performance "
  "may not transfer without recalibration, and no clinical-impact evidence exists. "
  "The outcome is within-system readmission; out-of-system readmissions remain "
  "unobservable. Post-discharge death is observable through patients.dod (hospital "
  "and Massachusetts registry sources) but is censored one year after each "
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
  "feature-set-dependent and is reported as exploratory only. Social "
  "determinants of health are unavailable in MIMIC-IV. The subgroup disparity "
  "for Black patients is unexplained and unresolved. Medicare-Advantage versus "
  "fee-for-service status is unobservable. The released prototype is a research "
  "demonstration, not a deployed clinical system, and no claim is made that the "
  "model improves outcomes.")
H1("Conclusions")
P(f"A parsimonious {NF}-feature XGBoost model, selected from the complete pool "
  f"of demonstrably discharge-available, nonbilling structured EHR predictors "
  f"under patient-grouped, fold-local preprocessing and feature selection on "
  f"development data, estimates 30-day within-system readmission in "
  f"Medicare-insured adults with strong calibration; the selection procedure "
  f"achieved a development-data out-of-fold AUROC of {OOF['auroc']:.4f} and "
  f"the fixed model {MP['auroc']['point']:.4f} on the historically exposed "
  f"test partition, clearly exceeding adapted "
  f"bedside clinical scores on the identical cohort and outcome. "
  f"Discrimination was highest for events occurring during the first day "
  f"after discharge. External validation and prospective "
  f"evaluation are the necessary next steps before any clinical use.")

# ============================================================ BACK MATTER
# Order per JMIR AI: Acknowledgments, Funding, Conflicts of Interest,
# Data Availability, Authors' Contributions, Abbreviations, Appendices, Refs.
H1("Acknowledgments")
P("This work began as the capstone project for the Master of Science in Data "
  "Science and Artificial Intelligence at Florida International University. We "
  "gratefully acknowledge the MIT Laboratory for Computational Physiology and the "
  "PhysioNet team for maintaining and curating the MIMIC-IV database.")
H1("Funding Statement")
P("[AUTHOR ACTION REQUIRED: confirm. Suggested if accurate:] This study received "
  "no external funding.")
H1("Conflicts of Interest")
P("None declared.")
H1("Data Availability")
P("MIMIC-IV v3.1 is available from PhysioNet under a credentialed data use "
  "agreement [21,26] and is not redistributed. All analysis code, the corrected "
  "cohort-construction and reanalysis scripts, the machine-readable results "
  "file from which every value in this manuscript is generated "
  "(final_model_v6.json and companions), the predictor dictionary, and the "
  "prototype source are available in the project repositories "
  "(Medicare-30day-Readmission-MIMIC-IV; readmission-risk-api; "
  "riskpath-clinician-companion). [AUTHOR ACTION: insert exact repository "
  "URLs, the tagged release, and an archival DOI (eg, Zenodo).]")
H1("Authors' Contributions")
P("Conceptualization: TB; methodology: TB, AG, CP, AMM; software, formal analysis, "
  "and visualization: TB, AG; data curation: TB, AG; investigation and validation: "
  "TB, AG; writing-original draft: TB; writing-review and editing: AG, CP, AMM; "
  "supervision: CP, AMM. All authors approved the final version. (CRediT taxonomy.)")
H1("Abbreviations")
P("AFT: accelerated failure time; AUROC: area under the receiver operating "
  "characteristic curve; CI: confidence interval; CMS: Centers for Medicare & "
  "Medicaid Services; DCA: decision-curve analysis; DRG: diagnosis-related group; "
  "ECE: expected calibration error; EHR: electronic health record; FNR: "
  "false-negative rate; HRRP: Hospital Readmissions Reduction Program; IPCW: "
  "inverse probability of censoring weighting; IRB: institutional review board; "
  "MIMIC: Medical Information Mart for Intensive Care; NLP: natural language "
  "processing; NPV: negative predictive value; OOF: out-of-fold; PACS: picture "
  "archiving and communication system; PPV: positive predictive value; RFE: "
  "recursive feature elimination; SD: standard deviation; SHAP: Shapley "
  "additive explanations; STROBE: Strengthening the Reporting of Observational "
  "Studies in Epidemiology; TRIPOD: Transparent Reporting of a multivariable "
  "prediction model for Individual Prognosis Or Diagnosis; XGBoost: extreme "
  "gradient boosting.")

H1("Multimedia Appendices")
P(f"Appendix 1: predictor audit and selection workbook (207-candidate audit "
  f"with per-predictor missingness, eligible pool, exclusion reasons, final "
  f"{NF}-feature dictionary, RFE trajectories and per-fold selections, "
  f"out-of-fold model comparison with paired differences, full subgroup "
  f"results with CIs, outcome-sensitivity analyses, daily competing-risk event "
  f"counts, and stage-differential estimates). Appendix 2: LACE and HOSPITAL "
  f"reconstruction tables. Appendix 3: race-category consolidation with raw "
  f"counts. Appendix 4: supplementary figures and development history "
  f"(decision curves; same-cohort ROC curves; development-phase results "
  f"including earlier feature versions and model families). Appendix 5: "
  f"completed TRIPOD+AI checklist. Appendix 6: PROBAST+AI self-assessment.")

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
 "Vyas DA, Eisenstein LG, Jones DS. Hidden in plain sight - reconsidering the use "
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
]
for i, ref in enumerate(REFS, 1):
    P(f"{i}. {ref}")
P("[SUBMISSION NOTE: references must be reformatted to the JMIR reference "
  "style during portal entry; the portal will also require author degrees, "
  "ORCID iDs, and corresponding-author contact details.]",
  italic=True)

DST = PUB / "Paper JMIR AI Submission v6.4 SINGLE-COLUMN.docx"
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
assert n_abs <= 440, f"abstract over target (440, JMIR limit 450): {n_abs}"

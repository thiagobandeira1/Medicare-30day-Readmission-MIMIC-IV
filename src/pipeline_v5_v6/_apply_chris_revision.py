# -*- coding: utf-8 -*-
"""Mentor revision round 1 (Dr. Poellabauer, 2026-09-14). Applies to BOTH
FINAL builders:

  * new contribution-led title (his tracked edit) + JMIR-convention design
    subtitle retained (answers his subtitle comment: JMIR expects it)
  * abstract: his Background reframing sentence, "at hospital discharge",
    his two Methods leakage/primary-estimate sentences merged in, Results
    procedure sentence clarified, compensating trims to stay within 450
  * keywords: add "data leakage"
  * Introduction restructured: Background -> Related Work (4 subsections,
    incl. a new Data Leakage and Validation Pitfalls subsection citing
    refs 34 to 36) -> explicit Study Rationale -> 4 research questions
    (his RQ1 to RQ3 wording, HER typo fixed to EHR, new RQ4)
  * Discussion: Principal Findings reordered (leakage, timing, fairness,
    clinical scores), Answers section rewritten for 4 RQs
  * references: Kapoor 2023, Futoma 2020, Wiens 2019 appended (34 to 36)
  * abstract assert relaxed 440 -> 450 (Chris's framing takes the room)

Every replacement is verified; the script fails loudly on any miss."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPS = [
    # ---------------------------------------------------------------- title
    ('r = t.add_run("Predicting 30-Day Hospital Readmission in Medicare Patients")',
     'r = t.add_run("Predicting 30-Day Hospital Readmission at Discharge: A "\n'
     '              "Leakage-Safe, Calibrated EHR Model in Medicare-Insured Adults")'),
    ('r = t.add_run("An Interpretable Gradient-Boosting Model on MIMIC-IV v3.1: "\n'
     '              "Retrospective Development and Internal Validation")',
     'r = t.add_run("Retrospective Development and Internal Validation of an "\n'
     '              "Interpretable Gradient-Boosting Model on MIMIC-IV v3.1")'),
    # ---------------------------------------------------- abstract Background
    ('"(AUROC) of 0.70 in external validation, and machine-learning reports often "\n'
     '  "lack leakage-safe validation, calibration, same-cohort baselines, or "\n'
     '  "fairness auditing.",',
     '"(AUROC) of 0.70 in external validation. Although machine-learning "\n'
     '  "approaches may improve discrimination, many studies have limited "\n'
     '  "assessment of data leakage, calibration, comparison with established "\n'
     '  "scores in the same cohort, and performance across patient subgroups.",'),
    # ----------------------------------------------------- abstract Objective
    ('"estimates the probability of 30-day all-cause within-system readmission at the "\n'
     '  "moment of discharge for Medicare-insured adults, using only structured electronic "',
     '"estimates the probability of 30-day all-cause within-system readmission "\n'
     '  "at hospital discharge for Medicare-insured adults, using only structured electronic "'),
    # ------------------------------------------------------- abstract Methods
    ('f"Staged recursive feature elimination ran inside 5 outer patient-grouped "\n'
     '  f"folds on the development partitions only (3 grouped inner folds each; all "\n'
     '  f"learned preprocessing fold-local; prespecified parsimony rule); "\n'
     '  f"features selected in at least 3 of 5 folds formed the consensus set of the "\n'
     '  f"final extreme gradient boosting (XGBoost) classifier. The primary estimate "\n'
     '  f"is the out-of-fold performance of the complete selection procedure. "',
     'f"Feature selection and all learned preprocessing were performed within "\n'
     '  f"patient-grouped cross-validation folds to prevent information leakage: "\n'
     '  f"staged recursive feature elimination ran inside 5 outer folds on the "\n'
     '  f"development partitions only (3 grouped inner folds each; prespecified "\n'
     '  f"parsimony rule), and features selected in at least 3 of 5 folds formed "\n'
     '  f"the consensus set of the final extreme gradient boosting (XGBoost) "\n'
     '  f"classifier. The prespecified primary performance estimate was the "\n'
     '  f"out-of-fold discrimination of the complete feature-selection and "\n'
     '  f"modeling procedure. "'),
    ('f"Uncertainty used patient-cluster bootstrap resampling; the operating "\n'
     '  f"threshold was development-derived. Timing was assessed with a "\n'
     '  f"competing-risks reformulation and landmark models.",',
     'f"Uncertainty used patient-cluster bootstrap resampling. Timing was "\n'
     '  f"assessed with a competing-risks reformulation and landmark models.",'),
    # ------------------------------------------------------- abstract Results
    ('f"The selection procedure (fold-specific sets of 27 to 38 predictors) "\n'
     '  f"achieved a development-data out-of-fold AUROC of {OOF[\'auroc\']:.4f} ',
     'f"This procedure, in which each outer fold selected its own 27 to 38 "\n'
     '  f"predictors, achieved a development-data out-of-fold AUROC of {OOF[\'auroc\']:.4f} '),
    ('f"(SD {CCV[\'sd\']:.4f}) under grouped cross-validation (secondary; "\n'
     '  f"consensus selected on the same development data) and "',
     'f"(SD {CCV[\'sd\']:.4f}) under grouped cross-validation (secondary) and "'),
    ('f"{MP[\'auroc\'][\'ci95\'][1]:.4f}) on a historically exposed test partition "\n'
     '  f"not used in this version\'s feature selection. Calibration was "',
     'f"{MP[\'auroc\'][\'ci95\'][1]:.4f}) on the historically exposed test "\n'
     '  f"partition (tertiary). Calibration was "'),
    # -------------------------------------------------------------- keywords
    ('P("hospital readmission; Medicare; MIMIC-IV; machine learning; gradient boosting; "\n'
     '  "calibration; algorithmic fairness; survival analysis; clinical prediction model; "\n'
     '  "electronic health records")',
     'P("hospital readmission; Medicare; MIMIC-IV; machine learning; gradient "\n'
     '  "boosting; data leakage; calibration; algorithmic fairness; survival "\n'
     '  "analysis; clinical prediction model; electronic health records")'),
    # ------------------------------------- Introduction: Related Work + RQs
    ('  "functional decline, and caregiver strain. Bedside indices remain the operational "\n'
     '  "default: the LACE index combines length of stay, acuity, comorbidity, and "\n'
     '  "emergency-department use [3], and the HOSPITAL score targets potentially avoidable "\n'
     '  "readmission [4]; external validations of such scores rarely exceed an AUROC of "\n'
     '  "0.70 [5,6]. Systematic reviews of readmission models report similar ceilings for "\n'
     '  "regression-based approaches and identify recurring methodological weaknesses: "\n'
     '  "leakage-prone splits, absent calibration, and unexamined subgroup performance "\n'
     '  "[5,7,8].")\n'
     'P("Machine learning on structured EHR data can exceed these ceilings, and "\n'
     '  "gradient-boosted decision trees remain the strongest general-purpose learners on "\n'
     '  "tabular clinical data [9-12]. Yet reported discrimination is not deployability: "\n'
     '  "clinical use requires patient-level leakage prevention, calibrated absolute "\n'
     '  "probabilities [13], interpretable per-patient explanations [14], comparison with "\n'
     '  "clinical scores on the same cohort rather than published values, fairness "\n'
     '  "auditing [15,16], and reporting aligned with TRIPOD+AI [17] and PROBAST [18]. "\n'
     '  "Race-aware modeling requires particular care following the removal of race "\n'
     '  "coefficients from clinical equations such as the estimated glomerular filtration "\n'
     '  "rate [19,20].")\n'
     'H2("Prior Work on MIMIC-IV and Study Rationale")\n'
     'P("Three recent studies predict 30-day readmission on the same MIMIC-IV database "\n'
     '  "[21]. A multimodal spatiotemporal graph neural network combining EHR time series "\n'
     '  "with chest radiographs reported an AUROC of 0.791 on a radiograph-selected subset "\n'
     '  "of 14,532 admissions [22]; a graph model over discharge summaries reported 0.727 "\n'
     '  "on 303,571 all-adult admissions [23]; and a 26-feature XGBoost framework reported "\n'
     '  "0.696 on 415,231 all-adult admissions [24]. ClinicalBERT reported approximately "\n'
     '  "0.714 on a MIMIC-III cohort using discharge notes [25]. Because cohorts and "\n'
     '  "outcome definitions differ, these are reference points rather than head-to-head "\n'
     '  "comparisons; none combines leakage-safe patient-grouped validation, calibration "\n'
     '  "analysis, same-cohort clinical baselines, subgroup fairness auditing, and "\n'
     '  "readmission-timing analysis in one evaluation.")\n'
     'H2("Research Questions and Objectives")\n'
     'P("Three research questions, carried forward from the project\'s inception, "\n'
     '  "organize the study. RQ1: which features are most predictive of 30-day "\n'
     '  "readmission in Medicare-insured adults? RQ2: which modeling approach achieves "\n'
     '  "the best predictive performance under a leakage-safe protocol? RQ3: can "\n'
     '  "interpretable machine-learning outputs based on global and patient-level "\n'
     '  "Shapley additive explanations (SHAP) "\n'
     '  "provide actionable insight for discharge teams? "',
     '  "functional decline, and caregiver strain. Accurate risk estimates "\n'
     '  "available at the moment of discharge could target transitional-care "\n'
     '  "resources, but only if the underlying model can be trusted at "\n'
     '  "deployment time.")\n'
     '\n'
     '\n'
     'def H3(t):\n'
     '    doc.add_paragraph(_style(t), style="Heading 3")\n'
     '\n'
     '\n'
     'H2("Related Work")\n'
     'H3("Clinical Scores and Regression-Based Models")\n'
     'P("Bedside indices remain the operational default for readmission risk: the "\n'
     '  "LACE index combines length of stay, acuity, comorbidity, and "\n'
     '  "emergency-department use [3], and the HOSPITAL score targets potentially "\n'
     '  "avoidable readmission [4]. External validations of such scores rarely "\n'
     '  "exceed an AUROC of 0.70 [5,6], and systematic reviews covering hundreds "\n'
     '  "of readmission models report similar ceilings for regression-based "\n'
     '  "approaches together with recurring methodological weaknesses: "\n'
     '  "leakage-prone splits, absent calibration assessment, and unexamined "\n'
     '  "subgroup performance [5,7,8].")\n'
     'H3("Machine Learning for Readmission Prediction")\n'
     'P("Machine learning on structured EHR data can exceed these ceilings, and "\n'
     '  "gradient-boosted decision trees remain the strongest general-purpose "\n'
     '  "learners on tabular clinical data [9-12]. Yet reported discrimination is "\n'
     '  "not deployability: clinical use requires patient-level leakage "\n'
     '  "prevention, calibrated absolute probabilities [13], interpretable "\n'
     '  "per-patient explanations [14], comparison with clinical scores on the "\n'
     '  "same cohort rather than against published values, fairness auditing "\n'
     '  "[15,16], and reporting aligned with TRIPOD+AI [17] and PROBAST [18]. "\n'
     '  "Race-aware modeling requires particular care following the removal of "\n'
     '  "race coefficients from clinical equations such as the estimated "\n'
     '  "glomerular filtration rate [19,20].")\n'
     'H3("Data Leakage and Validation Pitfalls")\n'
     'P("Data leakage, the use of information during model development that "\n'
     '  "would not legitimately be available at prediction time, is increasingly "\n'
     '  "recognized as a leading cause of overoptimistic and irreproducible "\n'
     '  "results across machine-learning science [34]. Clinical prediction is "\n'
     '  "especially exposed: features generated after the prediction moment (for "\n'
     '  "example, billing codes assigned during claims processing), "\n'
     '  "preprocessing fit on evaluation data, and splits that ignore patient "\n'
     '  "clustering all inflate apparent performance, and single-site results "\n'
     '  "often fail to generalize [18,35]. Roadmaps for responsible clinical "\n'
     '  "machine learning therefore call for patient-level partitioning, "\n'
     '  "fold-local preprocessing, verification that every predictor is "\n'
     '  "available at deployment time, and evaluation of the complete modeling "\n'
     '  "procedure rather than a single exposed model [17,18,36]. These "\n'
     '  "recommendations directly shaped this study\'s validation design.")\n'
     'H3("Prior Work on MIMIC")\n'
     'P("Three recent studies predict 30-day readmission on the same MIMIC-IV database "\n'
     '  "[21]. A multimodal spatiotemporal graph neural network combining EHR time series "\n'
     '  "with chest radiographs reported an AUROC of 0.791 on a radiograph-selected subset "\n'
     '  "of 14,532 admissions [22]; a graph model over discharge summaries reported 0.727 "\n'
     '  "on 303,571 all-adult admissions [23]; and a 26-feature XGBoost framework reported "\n'
     '  "0.696 on 415,231 all-adult admissions [24]. ClinicalBERT reported approximately "\n'
     '  "0.714 on a MIMIC-III cohort using discharge notes [25]. Because cohorts and "\n'
     '  "outcome definitions differ, these are reference points rather than "\n'
     '  "head-to-head comparisons.")\n'
     'H2("Study Rationale")\n'
     'P("The gap, therefore, is not another discrimination estimate. None of the "\n'
     '  "MIMIC studies above combines leakage-safe patient-grouped validation, "\n'
     '  "calibration analysis, same-cohort reconstruction of the clinical scores "\n'
     '  "hospitals actually use, readmission-timing analysis, and subgroup "\n'
     '  "fairness auditing in a single evaluation, and few evaluate the complete "\n'
     '  "selection procedure rather than one exposed model. This study was "\n'
     '  "designed to close that gap in the Medicare population, where the "\n'
     '  "readmission penalty applies, using only predictors demonstrably "\n'
     '  "available at the moment of discharge.")\n'
     'H2("Research Questions and Objectives")\n'
     'P("Four research questions organize the study. RQ1: Which "\n'
     '  "discharge-available EHR features contribute most to prediction of "\n'
     '  "30-day readmission in Medicare-insured adults? RQ2: Can a leakage-safe, "\n'
     '  "calibrated machine-learning model improve prediction of 30-day "\n'
     '  "readmission relative to established clinical scores? RQ3: Can global "\n'
     '  "and patient-level Shapley additive explanations (SHAP) provide "\n'
     '  "clinically interpretable insight into the factors driving predicted "\n'
     '  "readmission risk? RQ4: How does predictive performance vary across the "\n'
     '  "post-discharge time window and across demographic subgroups? "'),
    # --------------------------------------- Principal Findings reorder
    ('f"compact, operationally available predictor set. Second, the model clearly outperforms the "\n'
     '  f"adapted LACE and HOSPITAL scores on the identical cohort and outcome. "\n'
     '  f"Third, discrimination is highest for readmissions occurring during the "\n'
     '  f"first day after discharge, the window in which an intervention initiated "\n'
     '  f"at discharge would need to act. Fourth, the model performs less well for "\n'
     '  f"Black patients "\n'
     '  f"despite race not being an input, a disparity we quantify directly and "\n'
     '  f"report as an open limitation.")',
     'f"compact, operationally available predictor set. Second, discrimination "\n'
     '  f"is highest for readmissions occurring during the first day after "\n'
     '  f"discharge, the window in which an intervention initiated at discharge "\n'
     '  f"would need to act. Third, the model performs less well for Black "\n'
     '  f"patients despite race not being an input, a disparity we quantify "\n'
     '  f"directly and report as an open limitation. Fourth, the model clearly "\n'
     '  f"outperforms the adapted LACE and HOSPITAL scores on the identical "\n'
     '  f"cohort and outcome, with calibrated absolute risks the bedside scores "\n'
     '  f"do not provide.")'),
    # --------------------------------------- Answers to the Research Questions
    ('P(f"RQ1 (most predictive features): {_rq1} lead the final model (Figure 8); "\n'
     '  f"{N_UNANIMOUS} features were "\n'
     '  f"selected in every outer fold, indicating that the predictive core is stable "\n'
     '  f"rather than an artifact of one selection run. RQ2 (best modeling approach): "\n'
     '  f"gradient boosting; the XGBoost-based RFE procedure achieved an "\n'
     '  f"out-of-fold AUROC of {OOF[\'auroc\']:.4f} against "\n'
     '  f"{CMP5[\'logit\'][\'oof_auroc\']:.4f} for the regularized logistic-regression "\n'
     '  f"procedure on identical inputs, with LightGBM, CatBoost, and HistGradientBoosting "\n'
     '  f"within 0.0035 during development and blending adding nothing over the best "\n'
     '  f"single model. RQ3 (interpretable output): global SHAP identifies what the "\n'
     '  f"model relies on, per-patient additive decompositions convert each score into "\n'
     '  f"a ranked list of that patient\'s contributing factors at serving time, and "\n'
     '  f"the stage-specific analysis adds when discrimination is strongest across the "\n'
     '  f"post-discharge window, providing discharge teams with an explanation "\n'
     '  f"alongside the risk estimate.")',
     'P(f"RQ1 (which features): {_rq1} lead the final model (Figure 8); "\n'
     '  f"{N_UNANIMOUS} features were "\n'
     '  f"selected in every outer fold, indicating that the predictive core is stable "\n'
     '  f"rather than an artifact of one selection run. RQ2 (improvement over "\n'
     '  f"clinical scores): yes; on the identical cohort and outcome the "\n'
     '  f"leakage-safe procedure outperformed reconstructed LACE by "\n'
     '  f"{BSE[\'uplift_lace\']:.4f} and 12-month HOSPITAL by "\n'
     '  f"{BSE[\'uplift_hospital12\']:.4f} AUROC (both "\n'
     '  f"{fmt_p(max(BSE[\'uplift_lace_p\'], BSE[\'uplift_hospital12_p\']))}), while "\n'
     '  f"adding calibrated absolute risks; among learners, gradient boosting "\n'
     '  f"led ({OOF[\'auroc\']:.4f} vs {CMP5[\'logit\'][\'oof_auroc\']:.4f} for the "\n'
     '  f"regularized logistic-regression procedure on identical inputs, with "\n'
     '  f"LightGBM, CatBoost, and HistGradientBoosting within 0.0035 and "\n'
     '  f"blending adding nothing). RQ3 (interpretable output): global SHAP "\n'
     '  f"identifies what the model relies on, and per-patient additive "\n'
     '  f"decompositions convert each score into a ranked list of that "\n'
     '  f"patient\'s contributing factors at serving time. RQ4 (timing and "\n'
     '  f"subgroups): discrimination is highest on the first day after "\n'
     '  f"discharge (time-dependent AUROC "\n'
     '  f"{FM[\'survival\'][\'daily_tdauc\'][0]:.3f}) and declines over the "\n'
     '  f"window, and performance differs across race groups (AUROC "\n'
     '  f"{FR[\'Black\'][\'auroc\']:.3f} for Black patients vs "\n'
     '  f"{FR[\'White\'][\'auroc\']:.3f} for White patients; "\n'
     '  f"{fmt_p(wb[\'p_bootstrap\'])}) while remaining stable across age bands "\n'
     '  f"and sex, a disparity reported openly as a limitation and a target "\n'
     '  f"for remediation.")'),
    # ------------------------------------------------------------ references
    ('"Johnson AEW, Pollard TJ, Berkowitz SJ, et al. MIMIC-CXR, a de-identified "\n'
     ' "publicly available database of chest radiographs with free-text reports. Sci "\n'
     ' "Data. 2019;6:317.",\n'
     ']',
     '"Johnson AEW, Pollard TJ, Berkowitz SJ, et al. MIMIC-CXR, a de-identified "\n'
     ' "publicly available database of chest radiographs with free-text reports. Sci "\n'
     ' "Data. 2019;6:317.",\n'
     ' "Kapoor S, Narayanan A. Leakage and the reproducibility crisis in "\n'
     ' "machine-learning-based science. Patterns (N Y). 2023;4(9):100804. "\n'
     ' "doi:10.1016/j.patter.2023.100804",\n'
     ' "Futoma J, Simons M, Panch T, Doshi-Velez F, Celi LA. The myth of "\n'
     ' "generalisability in clinical research and machine learning in health "\n'
     ' "care. Lancet Digit Health. 2020;2(9):e489-e492.",\n'
     ' "Wiens J, Saria S, Sendak M, et al. Do no harm: a roadmap for responsible "\n'
     ' "machine learning for health care. Nat Med. 2019;25(9):1337-1340.",\n'
     ']'),
    # ------------------------------------------------------- abstract cap
    ('assert n_abs <= 440, f"abstract over target (440, JMIR limit 450): {n_abs}"',
     'assert n_abs <= 450, f"abstract over JMIR limit (450): {n_abs}"'),
]


def patch(path):
    t = path.read_text(encoding="utf-8")
    miss = []
    for old, new in REPS:
        n = t.count(old)
        if n == 0 and t.count(new):
            continue
        if n != 1:
            miss.append((old[:90], n))
            continue
        t = t.replace(old, new)
    if miss:
        for m, n in miss:
            print(f"  MISS ({n} hits) in {path.name}: {m!r}")
        raise SystemExit(f"{path.name}: {len(miss)} replacement(s) failed")
    path.write_text(t, encoding="utf-8")
    print(f"patched {path.name}")


for name in ("_build_jmir_final_part2.py", "_build_jmir_finalsc_part2.py"):
    patch(HERE / name)
print("CHRIS REVISION APPLIED")

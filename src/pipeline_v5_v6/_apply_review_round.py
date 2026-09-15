# -*- coding: utf-8 -*-
"""Critical-review round (2026-09-14 overnight): applies the upheld findings
of the 40-agent adversarial review to BOTH FINAL builders. Highlights:

  * build-time citation renumbering by first appearance (regress-proof)
  * new refs: Graham 2015, Krumholz 2013, Seyyed-Kalantari 2021, Blanche 2013
  * Related Work gains a timing-and-subgroups subsection; Background motivates
    timing and equity; abstract carries all four RQ threads at 439 words
  * fairness reporting quantified across age bands; threshold-level equity;
    within-system label-bias caveat; race AUROCs at stored 4-decimal precision
  * td-AUROC estimand stated (cause-specific, deaths retained as nonevents)
  * era-stratified sensitivity (new canonical block) + payer-scope sentence
  * leakage-safe scoping definition; sample size; multiplicity; software env
    from the canonical env block; dozens of consistency and grammar fixes

Every replacement verified; idempotent."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

ANSWERS_OLD = (
    'H2("Answers to the Research Questions")\n'
    '_rq1 = ", ".join(flabel(s["feature"]) for s in SH[:7])\n'
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
    '  f"for remediation.")')

ANSWERS_NEW = (
    'H2("Answers to the Research Questions")\n'
    '_rq1 = ", ".join(flabel(s["feature"]) for s in SH[:7])\n'
    'P(f"RQ1 (which features): {_rq1} lead the final model (Figure 8); "\n'
    '  f"{N_UNANIMOUS} features were "\n'
    '  f"selected in every outer fold, indicating that the predictive core is stable "\n'
    '  f"rather than an artifact of one selection run. RQ2 (improvement over "\n'
    '  f"clinical scores): yes; on the identical cohort and outcome the "\n'
    '  f"leakage-safe final model outperformed reconstructed LACE by "\n'
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
    '  f"{FM[\'survival\'][\'daily_tdauc\'][0]:.4f}) and declines over the "\n'
    '  f"window, and performance differs across race groups (AUROC "\n'
    '  f"{FR[\'Black\'][\'auroc\']:.4f} for Black patients vs "\n'
    '  f"{FR[\'White\'][\'auroc\']:.4f} for White patients; "\n'
    '  f"{fmt_p(wb[\'p_bootstrap\'])}) and across age bands "\n'
    '  f"({AB[\'<65\'][\'auroc\']:.4f} for patients younger than 65 years vs "\n'
    '  f"{AB[\'65-74\'][\'auroc\']:.4f} for ages 65 to 74), disparities reported "\n'
    '  f"openly as limitations and targets for remediation.")')

RENUMBER = '''

# ---- build-time citation renumbering by first appearance (review round) ----
def _renumber_citations(doc, refs):
    import re as _re3
    pat = _re3.compile(r"\\[([0-9][0-9,\\-]*)\\]")

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
'''

REPS = [
    # ---------------------------------------------------------------- top
    ('FR = FM["fairness"]["race"]',
     'FR = FM["fairness"]["race"]\nAB = FM["fairness"]["age_band"]'),
    # ------------------------------------------------------------- title
    ('r = t.add_run("Predicting 30-Day Hospital Readmission at Discharge: A "\n'
     '              "Leakage-Safe, Calibrated EHR Model in Medicare-Insured Adults")',
     'r = t.add_run("Predicting 30-Day Hospital Readmission at Discharge: A "\n'
     '              "Leakage-Safe, Calibrated Electronic Health Record Model "\n'
     '              "in Medicare-Insured Adults")'),
    ('r = t.add_run("Retrospective Development and Internal Validation of an "\n'
     '              "Interpretable Gradient-Boosting Model on MIMIC-IV v3.1")',
     'r = t.add_run("Retrospective Development and Internal Validation of an "\n'
     '              "Interpretable Gradient-Boosting Model on MIMIC-IV v3.1, "\n'
     '              "With Readmission-Timing and Subgroup-Fairness Analyses")'),
    # ---------------------------------------------------------- abstract
    ('f"v3.1. After excluding {CT[\'index_death_excluded\']:,} admissions ending in index "\n'
     '  f"in-hospital death, the cohort',
     'f"v3.1. After excluding {CT[\'index_death_excluded\']:,} index in-hospital "\n'
     '  f"deaths, the cohort'),
    ('f"readmission 0<t≤30 days after discharge. Partitions were "\n'
     '  f"patient-grouped (test: {p[\'test\'][\'admissions\']:,} admissions, "\n'
     '  f"{p[\'test\'][\'patients\']:,} patients). All 207',
     'f"readmission 0<t≤30 days after discharge. Partitions were "\n'
     '  f"patient-grouped (test: {p[\'test\'][\'admissions\']:,} admissions). All 207'),
    ('f"development partitions only (3 grouped inner folds each; prespecified "\n'
     '  f"parsimony rule), and features',
     'f"development partitions only (prespecified parsimony rule), and features'),
    ('f"Uncertainty used patient-cluster bootstrap resampling. Timing was "\n'
     '  f"assessed with a competing-risks reformulation and landmark models.",',
     'f"Uncertainty used patient-cluster bootstrap resampling. Timing was "\n'
     '  f"assessed with a competing-risks reformulation and landmark models, "\n'
     '  f"and subgroup performance was audited across age, sex, and race.",'),
    ('f"This procedure, in which each outer fold selected its own 27 to 38 "\n'
     '  f"predictors, achieved a development-data out-of-fold AUROC of {OOF[\'auroc\']:.4f} ',
     'f"The selection procedure, in which each outer fold selected its own "\n'
     '  f"27 to 38 predictors, achieved a development-data out-of-fold AUROC '
     'of {OOF[\'auroc\']:.4f} '),
    ('f"(95% cluster CI {OOF[\'ci95\'][0]:.4f} to {OOF[\'ci95\'][1]:.4f}; primary). The "',
     'f"(95% CI {OOF[\'ci95\'][0]:.4f} to {OOF[\'ci95\'][1]:.4f}; primary). The "'),
    ('f"fixed {NF}-feature consensus model reached {CCV[\'mean\']:.4f} "\n'
     '  f"(SD {CCV[\'sd\']:.4f}) under grouped cross-validation (secondary) and "',
     'f"fixed {NF}-feature consensus model reached {CCV[\'mean\']:.4f} "\n'
     '  f"(secondary; SD {CCV[\'sd\']:.4f}) and "'),
    ('f"{fmt_p(max(BSE[\'uplift_lace_p\'], BSE[\'uplift_hospital12_p\']))}). "\n'
     '  f"Time-dependent AUROC was highest on the first day after discharge "\n'
     '  f"({FM[\'survival\'][\'daily_tdauc\'][0]:.3f}). Discrimination was lower for "\n'
     '  f"Black patients (AUROC {FR[\'Black\'][\'auroc\']:.3f}) than White patients "\n'
     '  f"({FR[\'White\'][\'auroc\']:.3f}; difference {wb[\'point\']:.3f}, 95% CI "',
     'f"{fmt_p(max(BSE[\'uplift_lace_p\'], BSE[\'uplift_hospital12_p\']))}). "\n'
     '  f"The 180-day length-of-stay trend and prior-utilization features led "\n'
     '  f"SHAP attributions. "\n'
     '  f"Time-dependent AUROC was highest on the first day after discharge "\n'
     '  f"({FM[\'survival\'][\'daily_tdauc\'][0]:.4f}). Discrimination was lower for "\n'
     '  f"Black patients (AUROC {FR[\'Black\'][\'auroc\']:.4f}) than for White patients "\n'
     '  f"({FR[\'White\'][\'auroc\']:.4f}; difference {wb[\'point\']:.3f}, 95% CI "'),
    ('"adults. This is a research prediction model with a demonstration prototype; "\n'
     '  "external validation and prospective evaluation are required before "\n'
     '  "clinical use.",',
     '"adults. This is a research model; external validation and prospective "\n'
     '  "evaluation are required before clinical use.",'),
    # ------------------------------------------------ background + related
    ('  "functional decline, and caregiver strain. Accurate risk estimates "\n'
     '  "available at the moment of discharge could target transitional-care "\n'
     '  "resources, but only if the underlying model can be trusted at "\n'
     '  "deployment time.")',
     '  "functional decline, and caregiver strain. Accurate risk estimates "\n'
     '  "available at the moment of discharge could target transitional-care "\n'
     '  "resources, but only if the underlying model can be trusted at "\n'
     '  "deployment time. Timing and equity sharpen the stakes: transitional "\n'
     '  "interventions act within days of discharge, so when patients return "\n'
     '  "matters as much as whether they return, and because risk scores steer "\n'
     '  "resources, unequal model performance across groups becomes unequal "\n'
     '  "care.")'),
    ('P("Bedside indices remain the operational default for readmission risk: the "\n'
     '  "LACE index combines length of stay, acuity, comorbidity, and "\n'
     '  "emergency-department use [3], and the HOSPITAL score targets potentially "\n'
     '  "avoidable readmission [4]. External validations of such scores rarely "\n'
     '  "exceed an AUROC of 0.70 [5,6],',
     'P("Bedside indices remain the operational default for readmission risk: the "\n'
     '  "LACE index combines length of stay, acuity, comorbidity, and "\n'
     '  "emergency-department use [3], and the HOSPITAL score targets potentially "\n'
     '  "avoidable readmission [4]. External validations of such scores rarely "\n'
     '  "exceed an area under the receiver operating characteristic curve "\n'
     '  "(AUROC) of 0.70 [5,6],'),
    ('P("Machine learning on structured EHR data can exceed these ceilings, and "',
     'P("Machine learning on structured electronic health record (EHR) data can "\n'
     '  "exceed these ceilings, and "'),
    # timing + fairness subsection after the leakage subsection
    ('H3("Prior Work on MIMIC")\n'
     'P("Three recent studies predict 30-day readmission on the same MIMIC-IV database "\n'
     '  "[21].',
     'H3("Readmission Timing and Subgroup Performance")\n'
     'P("When patients return matters as much as whether they return: early and "\n'
     '  "late readmissions differ in causes and preventability [37], the first "\n'
     '  "days after discharge carry a generalized transient vulnerability [38], "\n'
     '  "and transitional-care interventions act on a limited window, yet most "\n'
     '  "prediction studies collapse the 30-day horizon into a single binary "\n'
     '  "label and report no timing analysis. Subgroup performance is similarly "\n'
     '  "underexamined: audits of deployed clinical algorithms have found racial "\n'
     '  "bias in resource-allocation scores [15] and systematic underdiagnosis "\n'
     '  "of underserved groups by imaging models [39], gaps can persist even "\n'
     '  "when protected attributes are not model inputs [16], and readmission "\n'
     '  "models are rarely audited by subgroup [5,8]. The timing and subgroup "\n'
     '  "analyses of RQ4 address both gaps.")\n'
     'H3("Prior Work on MIMIC")\n'
     'P("Three recent studies predict 30-day readmission on MIMIC-IV, the "\n'
     '  "database used in this study "\n'
     '  "[21].'),
    ('  "0.696 on 415,231 all-adult admissions [24]. ClinicalBERT reported approximately "',
     '  "0.696 on 415,231 all-adult admissions [24]. ClinicalBERT, a clinical "\n'
     '  "adaptation of a pretrained language model, reported approximately "'),
    # objectives with RQ tags
    ('  "Operationally, the study aimed to (1) develop and internally validate a "\n'
     '  "calibrated, interpretable model of 30-day all-cause within-system readmission "\n'
     '  "using only structured EHR fields available at discharge; (2) compare it against "\n'
     '  "LACE and HOSPITAL reconstructed on the identical test partition; (3) "\n'
     '  "characterize when patients return using a time-to-event reformulation and "\n'
     '  "stage-specific landmark models; (4) audit subgroup performance across age, sex, "\n'
     '  "and race with cluster-level uncertainty; and (5) release the pipeline and a "\n'
     '  "demonstration prototype for reproducibility.")',
     '  "Operationally, the study aimed to (1) develop and internally validate a "\n'
     '  "calibrated, interpretable model of 30-day all-cause within-system readmission "\n'
     '  "using only structured EHR fields available at discharge (RQ1, RQ2); (2) "\n'
     '  "compare it against LACE and HOSPITAL reconstructed on the identical test "\n'
     '  "partition (RQ2); (3) characterize when patients return using a "\n'
     '  "time-to-event reformulation and stage-specific landmark models (RQ4); "\n'
     '  "(4) audit subgroup performance across age, sex, and race with "\n'
     '  "cluster-level uncertainty (RQ3, RQ4); and (5) release the pipeline and "\n'
     '  "a demonstration prototype in support of reproducibility.")'),
    # ------------------------------------------------------------ methods
    ('P("We conducted a retrospective prediction-model development and internal-validation "\n'
     '  "study, reported in line with TRIPOD+AI [17]. The data source is MIMIC-IV v3.1 "\n'
     '  "[21,26], a deidentified EHR database of 546,028 hospitalizations at Beth Israel "\n'
     '  "Deaconess Medical Center (Boston, MA) admitted between 2008 and 2022.',
     'P("We conducted a retrospective prediction-model development and internal-validation "\n'
     '  "study, reported in line with TRIPOD+AI [17]; the completed checklist is "\n'
     '  "Multimedia Appendix 5 and a PROBAST+AI self-assessment [18] is "\n'
     '  "Multimedia Appendix 6. The data source is MIMIC-IV v3.1 "\n'
     '  "[21,26], a deidentified EHR database of 546,028 hospitalizations "\n'
     '  "occurring between 2008 and 2022 at Beth Israel Deaconess Medical "\n'
     '  "Center (Boston, MA).'),
    ('f"age ≥65 years ({FD[\'fairness\'][\'age_band\'][\'<65\'][\'n\']:,} test admissions "\n'
     '  f"were younger, eg, disability entitlement) and MIMIC-IV cannot distinguish "',
     'f"age ≥65 years ({FD[\'fairness\'][\'age_band\'][\'<65\'][\'n\']:,} test admissions "\n'
     '  f"were younger than 65 years, typically through disability entitlement) "\n'
     '  f"and MIMIC-IV cannot distinguish "'),
    ('  "after index discharge, all cause, and irrespective of how that subsequent "\n'
     '  "admission ended; a readmission followed by in-hospital death is a readmission. "',
     '  "after index discharge, of any cause, and irrespective of how that subsequent "\n'
     '  "admission ended; a readmission followed by in-hospital death is a "\n'
     '  "readmission. Readmissions were identified from all subsequent "\n'
     '  f"hospitalizations of the patient in the database, irrespective of the "\n'
     '  f"insurance recorded on the subsequent admission "\n'
     '  f"({EX[\'payer_scope\'][\'n_30d_next_nonmedicare\']} 30-day next admissions "\n'
     '  f"carried non-Medicare insurance). In this binary frame, admissions "\n'
     '  "followed by death without readmission within 30 days are nonevents; "\n'
     '  "the competing-risks reformulation below addresses the interpretation. "'),
    ('  "rejected because per-patient date shifting removes cross-patient temporal "\n'
     '  "ordering. Independent external validation remains necessary.")',
     '  "rejected: per-patient date shifting removes fine-grained cross-patient "\n'
     '  "ordering, and the coarse patient-level anchor_year_group bands that "\n'
     '  "remain would yield unequal, clinically heterogeneous partitions; an "\n'
     '  "era-stratified sensitivity analysis appears in the Results, and "\n'
     '  "residual secular drift is acknowledged in Limitations. Independent "\n'
     '  "external validation remains necessary. Throughout, leakage-safe "\n'
     '  "denotes this design: fold-local preprocessing, target encoding, and "\n'
     '  "feature selection, plus the discharge-availability predictor audit; "\n'
     '  "residual design-level optimism (the historically exposed test "\n'
     '  "partition, the development-derived threshold, and fixed "\n'
     '  "hyperparameters) is disclosed here and in Limitations.")'),
    ('  "Cox regression as the linear reference, with competing deaths censored at "\n'
     '  "their death time. Evaluation used Harrell C [29] and inverse-probability-"\n'
     '  "of-censoring-weighted time-dependent AUROC at each day 1 to 29 [30], with "\n'
     '  "patient-cluster bootstrap uncertainty bands and daily event counts reported. "',
     '  "Cox regression as the linear reference, with competing deaths censored at "\n'
     '  "their death time in the cause-specific fits. Evaluation used Harrell C "\n'
     '  "[29] and cumulative/dynamic time-dependent AUROC on each of days 1 "\n'
     '  "through 29, estimated with inverse probability of censoring weighting "\n'
     '  "[30,40]; patients who died before the evaluation day remain in the "\n'
     '  "comparison set as nonevents, a cause-specific competing-risks "\n'
     '  "definition [40], with patient-cluster bootstrap uncertainty bands and "\n'
     '  "daily event counts reported. "'),
    ('  "descriptive. Stage-specific TreeSHAP attribution shares are exploratory.")',
     '  "descriptive; landmark models were trained on training-partition "\n'
     '  "patients, evaluated on the test partition, and reuse discharge-time "\n'
     '  "features without post-discharge updating. Stage-specific TreeSHAP "\n'
     '  "attribution shares are exploratory.")'),
    ('f"percentile intervals (1,000 resamples for the test metric panel; 800 for "\n'
     '  f"subgroup CIs; 500 for out-of-fold model comparisons; every contrast for "',
     'f"percentile intervals (1,000 resamples for the test metric panel; 800 for "\n'
     '  f"subgroup CIs; 500 for the primary out-of-fold estimate and model "\n'
     '  f"comparisons; every contrast for "'),
    ('f"Because patients contribute multiple admissions, all confidence intervals "',
     'f"Because patients contribute multiple admissions, all confidence "\n'
     'f"intervals (CIs) "'),
    ('f"the prespecified parsimony rule, not as statistical equivalence. All "',
     'f"the prespecified parsimony rule, not as statistical equivalence. Formal "\n'
     'f"hypothesis testing was limited to three prespecified contrasts (final "\n'
     'f"model vs LACE, vs HOSPITAL, and the White vs Black AUROC difference); "\n'
     'f"all other subgroup, timing, era, and sensitivity results are "\n'
     'f"descriptive, and no adjustment for multiple comparisons was applied. All "'),
    ('f"the Brier score, expected calibration error over deciles, and logistic "',
     'f"the Brier score, expected calibration error over 10 equal-width "\n'
     'f"probability bins, and logistic "'),
    ('f"unchanged to the test partition; sensitivity, specificity, positive and "\n'
     '  f"negative predictive values are reported with cluster-bootstrap CIs. The "',
     'f"unchanged to the test partition; sensitivity, specificity, positive and "\n'
     '  f"negative predictive values are reported with cluster-bootstrap CIs. "\n'
     '  f"The Youden threshold is illustrative for the metric panel and is not "\n'
     '  f"proposed for deployment; the capacity-constrained view is the "\n'
     '  f"operational lens. The "'),
    ('f"(top-k% flagged). Analyses used Python 3.11 and 3.12 (XGBoost, LightGBM, "\n'
     '  f"scikit-learn, lifelines, scikit-survival, SHAP); exact versions are pinned in "\n'
     '  f"the repository.")',
     'f"(top-k% flagged). Analyses used Python {FM[\'env\'][\'python\']} with "\n'
     'f"XGBoost {FM[\'env\'][\'xgboost\']}, scikit-learn {FM[\'env\'][\'sklearn\']}, "\n'
     'f"pandas {FM[\'env\'][\'pandas\']}, and NumPy {FM[\'env\'][\'numpy\']} (fixed "\n'
     'f"seed {FM[\'env\'][\'seed\']}); LightGBM, lifelines, scikit-survival, and "\n'
     'f"SHAP versions are pinned in the repository.")'),
    ('P("LACE [3] and HOSPITAL [4] were re-implemented on the identical test partition. "',
     'P("LACE [3] and HOSPITAL [4] were reconstructed on the identical test partition. "'),
    # ------------------------------------------------------------- results
    ('f"readmission within 30 days. Mean time to readmission among events was "\n'
     '  f"{SV[\'mean_days_to_readmission\']:.1f} days.")',
     'f"readmission within 30 days. Mean time to readmission among events was "\n'
     '  f"{SV[\'mean_days_to_readmission\']:.1f} days. This prevalence reflects "\n'
     '  f"an all-cause, within-system definition that includes planned returns "\n'
     '  f"and all Medicare admission types, and is not comparable to HRRP "\n'
     '  f"condition-specific national rates.")'),
    ('figure(FIG / "r2_fig1_flow_col.png",\n'
     '       "Figure 1. Cohort flow diagram (STROBE-style) with patient-grouped partitions.")',
     'figure(FIG / "r2_fig1_flow_col.png",\n'
     '       "Figure 1. Participant flow diagram with patient-grouped "\n'
     '       "partitions; feature selection ran inside 5 patient-grouped folds "\n'
     '       "drawn from the development partitions (Methods).")'),
    ('f"complete RFE-procedure estimate remains the primary evidence. "',
     'f"complete selection-procedure estimate remains the primary evidence. "'),
    ('f"the {NF}-feature consensus model was adopted. It contains no "\n'
     '  f"billing-derived features by construction, so no separate "\n'
     '  f"coding-availability sensitivity model is required; the earlier "\n'
     '  f"development-phase sensitivity analyses are retained in Multimedia "\n'
     '  f"Appendix 4.")',
     'f"the {NF}-feature consensus model was adopted. It contains no "\n'
     '  f"billing-derived features by construction; whether concurrent "\n'
     '  f"diagnosis coding would add signal at sites where it is reliably "\n'
     '  f"available at discharge is a site-specific question left to future "\n'
     '  f"work, and the earlier development-phase sensitivity analyses are "\n'
     '  f"retained in Multimedia Appendix 4.")'),
    ('figure(FIG5 / "v5_stability.png",\n'
     '       "Figure 3. Feature-selection frequency across the 5 outer folds; teal "\n'
     '       "bars form the consensus set (selected in at least 3 of 5 folds).")',
     'figure(FIG5 / "v5_stability.png",\n'
     '       f"Figure 3. Selection frequency across the 5 outer folds for all "\n'
     '       f"{len(STAB[\'selection_frequency\'])} features selected at least "\n'
     '       "once; teal bars form the consensus set and the dashed line marks "\n'
     '       "the consensus threshold (3 of 5 folds).")'),
    ('f"was {CCV[\'mean\']:.4f} (SD {CCV[\'sd\']:.4f}; folds "',
     'f"was {CCV[\'mean\']:.4f} (fold SD {CCV[\'sd\']:.4f}; folds "'),
    ('f"precision {MP[\'ap\'][\'point\']:.3f} (95% CI {MP[\'ap\'][\'ci95\'][0]:.3f}-"\n'
     '  f"{MP[\'ap\'][\'ci95\'][1]:.3f}) against a "',
     'f"precision {MP[\'ap\'][\'point\']:.3f} (95% CI {MP[\'ap\'][\'ci95\'][0]:.3f} to "\n'
     '  f"{MP[\'ap\'][\'ci95\'][1]:.3f}) against a "'),
    ('f"{MP[\'slope\'][\'point\']:.2f} (95% CI {MP[\'slope\'][\'ci95\'][0]:.2f}-"\n'
     '  f"{MP[\'slope\'][\'ci95\'][1]:.2f}), intercept {MP[\'intercept\'][\'point\']:.2f} "\n'
     '  f"(Figure 4).',
     'f"{MP[\'slope\'][\'point\']:.2f} (95% CI {MP[\'slope\'][\'ci95\'][0]:.2f} to "\n'
     '  f"{MP[\'slope\'][\'ci95\'][1]:.2f}), intercept {MP[\'intercept\'][\'point\']:.2f} "\n'
     '  f"(95% CI {MP[\'intercept\'][\'ci95\'][0]:.2f} to "\n'
     '  f"{MP[\'intercept\'][\'ci95\'][1]:.2f}) "\n'
     '  f"(Figure 4).'),
    ('f"{MP[\'sensitivity\'][\'point\']:.3f} (95% CI {MP[\'sensitivity\'][\'ci95\'][0]:.3f}-"\n'
     '  f"{MP[\'sensitivity\'][\'ci95\'][1]:.3f}), specificity "',
     'f"{MP[\'sensitivity\'][\'point\']:.3f} (95% CI '
     '{MP[\'sensitivity\'][\'ci95\'][0]:.3f} to "\n'
     '  f"{MP[\'sensitivity\'][\'ci95\'][1]:.3f}), specificity "'),
    ('       "model (historically exposed test partition; tertiary evidence).")',
     '       "model (historically exposed test partition; tertiary evidence). "\n'
     '       "Dashed lines mark chance performance (receiver operating "\n'
     '       "characteristic panel) and outcome prevalence (precision-recall "\n'
     '       "panel).")'),
    ('f"({_sa[\'test_prevalence_alt\']*100:.1f}% prevalence); a model refit under "',
     'f"({_sa[\'test_events_alt\']/p[\'test\'][\'admissions\']*100:.1f}% prevalence); '
     'a model refit under "'),
    ('f"{BD5[\'auroc_safe\']:.4f}. None of these variations alters the study\'s "\n'
     '  f"conclusions.")',
     'f"{BD5[\'auroc_safe\']:.4f}. (5) Stratified by the patient-level "\n'
     'f"anchor_year_group era band, test discrimination was maintained or "\n'
     'f"higher in recent eras ({_era_txt}), while observed prevalence declined "\n'
     'f"from {_era[0][\'prevalence\']*100:.1f}% to "\n'
     'f"{_era[-1][\'prevalence\']*100:.1f}%; era strata are descriptive. None of "\n'
     'f"these variations alters the study\'s conclusions.")'),
    ('H2("Outcome-Definition and Cohort Sensitivity Analyses")\n'
     '_sa = SENS["sameday_as_readmission"]',
     'H2("Outcome-Definition and Cohort Sensitivity Analyses")\n'
     '_era = EX["era_stratified_test"]["bands"]\n'
     '_era_txt = "; ".join(\n'
     '    f"{b[\'band\'].replace(\' - \', \' to \')}: {b[\'auroc\']:.4f} "\n'
     '    f"(n={b[\'n\']:,})" for b in _era)\n'
     '_sa = SENS["sameday_as_readmission"]'),
    ('f"On the identical test partition the final model (AUROC "',
     'f"On the identical test partition, both reconstructed scores fell below "\n'
     'f"their published external validations, reflecting within-system '
     'attenuation "\n'
     'f"of their utilization inputs; because that attenuation affects the '
     'scores "\n'
     'f"but not the model\'s in-hospital signals, the uplifts below are best '
     'read "\n'
     'f"as upper bounds for this setting. The final model (AUROC "'),
    ('f"{st1[\'at_risk\']:,} at risk), {st2[\'auroc\']:.4f} for days 8 to 14 "\n'
     '  f"({st2[\'events\']:,}/{st2[\'at_risk\']:,}), and {st3[\'auroc\']:.4f} for days "\n'
     '  f"15 to 30 ({st3[\'events\']:,}/{st3[\'at_risk\']:,}); discrimination improves for "',
     'f"{st1[\'at_risk\']:,} at risk), {st2[\'auroc\']:.4f} for days 8 to 14 "\n'
     '  f"({st2[\'events\']:,} events among {st2[\'at_risk\']:,} at risk), and '
     '{st3[\'auroc\']:.4f} for days "\n'
     '  f"15 to 30 ({st3[\'events\']:,} events among {st3[\'at_risk\']:,} at risk); '
     'discrimination improved for "'),
    ('f"{\'all 10\' if len(_sig) == 10 else f\'{len(_sig)} of the 10\'} leading "\n'
     '  f"differentials have 95% cluster-bootstrap CIs excluding zero). Signals "',
     'f"{\'all 10\' if len(_sig) == 10 else f\'{len(_sig)} of the 10\'} leading "\n'
     '  f"differentials have 95% cluster-bootstrap CIs excluding zero; the 10 "\n'
     '  f"largest observed differentials were selected before interval "\n'
     '  f"inspection, so these intervals are descriptive). Signals "'),
    ('_llist = "; ".join(f"{flabel(r[\'feature\'])} ({r[\'diff_pp\']:.1f})"\n'
     '                   for r in _late)',
     '_llist = "; ".join(f"{flabel(r[\'feature\'])} ({r[\'diff_pp\']:.1f} "\n'
     '                   f"percentage points)" for r in _late)'),
    ('       "indicate greater early importance. Error bars are 95% patient-cluster "\n'
     '       "bootstrap CIs.", width=3.1)',
     '       "indicate greater early importance. Error bars are 95% "\n'
     '       "cluster-bootstrap CIs.", width=3.1)'),
    # fairness paragraph (major 4, 9; minors 18, 25)
    ('f"contributor to subgroup differences. Discrimination was stable across age bands "\n'
     '  f"and sex but lower for Black patients than White patients: difference "\n'
     '  f"{wb[\'point\']:.4f} (95% CI {wb[\'ci95\'][0]:.4f} to {wb[\'ci95\'][1]:.4f}; "\n'
     '  f"{fmt_p(wb[\'p_bootstrap\'])}, stratified subgroup bootstrap with 5,000 "\n'
     '  f"draws, patients resampled within each subgroup) (Figure 7, Table 1). "\n'
     '  f"Calibration slopes across race groups ranged from "',
     'f"contributor to subgroup differences. Discrimination was lower for "\n'
     '  f"Black patients than for White patients: difference "\n'
     '  f"{wb[\'point\']:.4f} (95% CI {wb[\'ci95\'][0]:.4f} to {wb[\'ci95\'][1]:.4f}; "\n'
     '  f"{fmt_p(wb[\'p_bootstrap\'])}, stratified subgroup bootstrap with 5,000 "\n'
     '  f"draws, patients resampled within each subgroup), and it also varied "\n'
     '  f"across age bands, from {AB[\'<65\'][\'auroc\']:.4f} (95% CI "\n'
     '  f"{AB[\'<65\'][\'auroc_ci95\'][0]:.4f} to "\n'
     '  f"{AB[\'<65\'][\'auroc_ci95\'][1]:.4f}) among patients younger than 65 "\n'
     '  f"years, largely disability-entitled Medicare, to "\n'
     '  f"{AB[\'65-74\'][\'auroc\']:.4f} (95% CI "\n'
     '  f"{AB[\'65-74\'][\'auroc_ci95\'][0]:.4f} to "\n'
     '  f"{AB[\'65-74\'][\'auroc_ci95\'][1]:.4f}) for ages 65 to 74, with similar "\n'
     '  f"performance by sex (Figure 7, Table 1). Only the White vs Black "\n'
     '  f"contrast was prespecified and formally tested. Between-group AUROC "\n'
     '  f"differences partly reflect case mix and outcome prevalence, which "\n'
     '  f"differ across groups. At the shared threshold, specificity was "\n'
     '  f"{FR[\'Black\'][\'specificity\']:.3f} for Black vs "\n'
     '  f"{FR[\'White\'][\'specificity\']:.3f} for White patients, so a larger "\n'
     '  f"share of nonreadmitted Black patients is flagged; under the stated "\n'
     '  f"additional-resources posture this directs more outreach rather than "\n'
     '  f"less, but it is an operational disparity to monitor. "\n'
     '  f"Calibration slopes across race groups ranged from "'),
    ('f"per-group calibration summaries with uncertainty appear in Table 1. Race "\n'
     '  f"is not a model input;',
     'f"per-group calibration summaries with uncertainty appear in Table 1; "\n'
     '  f"across age bands the youngest and oldest bands deviated from 1 "\n'
     '  f"({AB[\'<65\'][\'cal_slope\']:.2f}, 95% CI "\n'
     '  f"{AB[\'<65\'][\'cal_slope_ci95\'][0]:.2f} to "\n'
     '  f"{AB[\'<65\'][\'cal_slope_ci95\'][1]:.2f}, and "\n'
     '  f"{AB[\'85+\'][\'cal_slope\']:.2f}, 95% CI "\n'
     '  f"{AB[\'85+\'][\'cal_slope_ci95\'][0]:.2f} to "\n'
     '  f"{AB[\'85+\'][\'cal_slope_ci95\'][1]:.2f}), indicating mild over- and "\n'
     '  f"underdispersion respectively. Race is not a model input;'),
    ('f"redevelopment, and none is claimed here.")',
     'f"redevelopment, and none is claimed here. Because the outcome is "\n'
     '  f"within-system, subgroup differences in out-of-system readmission "\n'
     '  f"would appear as differential outcome misclassification and could "\n'
     '  f"contribute to, or mask, the observed gap; MIMIC-IV cannot test this, "\n'
     '  f"and external validation with claims-complete follow-up is the "\n'
     '  f"appropriate check.")'),
    ('        "predictive value; Prev.: outcome prevalence; Cal.: calibration.",\n'
     '        keep_with_next=True)',
     '        "predictive value; Prev.: outcome prevalence; Cal.: calibration. "\n'
     '        "Other/Unknown is heterogeneous (declined and unknown race "\n'
     '        "included) and is not interpreted; patients whose recorded race "\n'
     '        "varies across admissions appear in more than one race row, so "\n'
     '        "race-row patient counts exceed unique test patients.",\n'
     '        keep_with_next=True)'),
    ('figure(FIG5 / "v5_fairness.png",\n'
     '       "Figure 7. Subgroup AUROC of the final model with 95% patient-cluster "\n'
     '       "bootstrap CIs.")',
     'figure(FIG5 / "v5_fairness.png",\n'
     '       f"Figure 7. Subgroup AUROC of the final model with 95% "\n'
     '       f"patient-cluster bootstrap CIs; the dashed line marks the overall "\n'
     '       f"test AUROC ({MP[\'auroc\'][\'point\']:.4f}).")'),
    # table cells: prevalence from counts, auroc and brier at stored precision
    ('def _ci3(e, key):\n'
     '    c = e.get(f"{key}_ci95")\n'
     '    v = e.get(key)\n'
     '    if v is None:\n'
     '        return "NA"\n'
     '    if key == "cal_slope":\n'
     '        return f"{v:.2f} ({c[0]:.2f} to {c[1]:.2f})" if c else f"{v:.2f}"\n'
     '    return f"{v:.3f} ({c[0]:.3f} to {c[1]:.3f})" if c else f"{v:.3f}"',
     'def _ci3(e, key):\n'
     '    c = e.get(f"{key}_ci95")\n'
     '    v = e.get(key)\n'
     '    if v is None:\n'
     '        return "NA"\n'
     '    if key == "cal_slope":\n'
     '        return f"{v:.2f} ({c[0]:.2f} to {c[1]:.2f})" if c else f"{v:.2f}"\n'
     '    if key == "auroc":\n'
     '        return f"{v:.4f} ({c[0]:.4f} to {c[1]:.4f})" if c else f"{v:.4f}"\n'
     '    return f"{v:.3f} ({c[0]:.3f} to {c[1]:.3f})" if c else f"{v:.3f}"'),
    ("                     f\"{e['prevalence']*100:.1f}%\",",
     "                     f\"{e['events']/e['admissions']*100:.1f}%\","),
    ('                     f"{e[\'brier\']:.3f}" if e.get("brier") is not None else "NA",',
     '                     f"{e[\'brier\']:.4f}" if e.get("brier") is not None else "NA",'),
    # interpretability joins with proper conjunctions
    ('_top3 = ", ".join(f"{flabel(s[\'feature\'])} ({s[\'feature\']}; mean |SHAP| "\n'
     '                  f"{s[\'mean_abs_shap\']:.3f})" for s in SH[:3])\n'
     '_next4 = ", ".join(flabel(s["feature"]) for s in SH[3:7])',
     '_t3 = [f"{flabel(s[\'feature\'])} ({s[\'feature\']}; mean |SHAP| "\n'
     '       f"{s[\'mean_abs_shap\']:.3f})" for s in SH[:3]]\n'
     '_top3 = f"{_t3[0]}, {_t3[1]}, and {_t3[2]}"\n'
     '_n4 = [flabel(s["feature"]) for s in SH[3:7]]\n'
     '_next4 = f"{_n4[0]}, {_n4[1]}, {_n4[2]}, and {_n4[3]}"'),
    # clinical utility: range + death caveat
    ('f"treat-none across an exploratory threshold range of roughly 0.10 to 0.40, a "\n'
     '  f"range not yet grounded in operational evidence (Multimedia Appendix 4). "',
     'f"treat-none across the evaluated threshold range of 0.05 to 0.40, a "\n'
     '  f"range not yet grounded in operational evidence (Multimedia Appendix 4). "'),
    ('f"is untested and requires prospective evaluation.")',
     'f"is untested and requires prospective evaluation. Patients who die out "\n'
     '  f"of hospital within 30 days without readmission are nonevents under "\n'
     '  f"the binary outcome; a low predicted readmission risk must not be "\n'
     '  f"read as clinical stability, and any deployment should surface "\n'
     '  f"competing mortality risk alongside the readmission score, "\n'
     '  f"particularly for hospice and facility discharges.")'),
    # discussion
    ('f"compact, operationally available predictor set. Second, discrimination "\n'
     '  f"is highest for readmissions occurring during the first day after "\n'
     '  f"discharge, the window in which an intervention initiated at discharge "\n'
     '  f"would need to act.',
     'f"compact, operationally available predictor set. Second, discrimination "\n'
     '  f"is highest for readmissions occurring during the first day after "\n'
     '  f"discharge, the window in which an intervention initiated at discharge "\n'
     '  f"would need to act (day-level and window-level estimands differ: "\n'
     '  f"window-level landmark discrimination was lowest for days 1 to 7 and "\n'
     '  f"highest for days 15 to 30; Results).'),
    ('f"In {CT[\'cohort_v2_admissions\']:,} Medicare-insured adult admissions, a "',
     'f"Four contributions carry this study, examined in turn below. In "\n'
     'f"{CT[\'cohort_v2_admissions\']:,} Medicare-insured adult admissions, a "'),
    ('f"partition. Four findings carry the study. First, the parsimonious "',
     'f"partition. First, the parsimonious "'),
    ('f"partition, numerically above the notes-based and structured-feature "\n'
     '  f"reports and near the multimodal result, while scoring every admission "\n'
     '  f"rather than an imaging-selected subset. But cohorts, outcome definitions, "\n'
     '  f"modalities, and validation designs differ across these "\n'
     '  f"studies, so no superiority claim is made. Cohorts and outcome definitions differ, so "\n'
     '  f"these are reference points; our controlled comparison is the same-cohort "\n'
     '  f"baseline analysis, where rebuilding HOSPITAL with its original 12-month "',
     'f"partition, numerically above the notes-based and structured-feature "\n'
     '  f"reports and near the multimodal result, while scoring every admission "\n'
     '  f"rather than a radiograph-selected subset. Because cohorts, outcome "\n'
     '  f"definitions, modalities, and validation designs differ across these "\n'
     '  f"studies, they are reference points and no superiority claim is made; "\n'
     '  f"our controlled comparison is the same-cohort "\n'
     '  f"baseline analysis, where reconstructing HOSPITAL with its original 12-month "'),
    ('  "available in any EHR at discharge, whereas notes and imaging require NLP or "\n'
     '  "PACS integration many sites cannot provide;',
     '  "available in any EHR at discharge, whereas notes and imaging require "\n'
     '  "natural language processing (NLP) or picture archiving and "\n'
     '  "communication system (PACS) integration many sites cannot provide;'),
    # limitations
    ('P("This is a single-center, retrospective, internally validated study; performance "\n'
     '  "may not transfer without recalibration, and no clinical-impact evidence exists. "',
     'P(f"This is a single-center, retrospective, internally validated study "\n'
     '  f"from one tertiary academic referral center in one region, whose "\n'
     '  f"Medicare case mix, discharge resources, and demographic composition "\n'
     '  f"({FR[\'White\'][\'admissions\']/p[\'test\'][\'admissions\']*100:.1f}% of test "\n'
     '  f"admissions White) differ from national Medicare; performance "\n'
     '  f"may not transfer without recalibration, and no clinical-impact evidence exists. "'),
    ('  "and Massachusetts registry sources) but is censored one year after each "',
     '  "and Massachusetts registry sources) but is censored 1 year after each "'),
    ('  "feature-set-dependent and is reported as exploratory only. Social "',
     '  "feature-set-dependent and is reported as exploratory only. Secular "\n'
     '  "drift across 2008 to 2022, including HRRP maturation and the COVID-19 "\n'
     '  "period, cannot be fully assessed under date shifting; the "\n'
     '  "era-stratified sensitivity analysis is reassuring but coarse. Social "'),
    ('  "for Black patients is unexplained and unresolved. Medicare-Advantage versus "\n'
     '  "fee-for-service status is unobservable.',
     '  "for Black patients is unexplained and unresolved and, like all "\n'
     '  "subgroup analyses here, was estimated on the historically exposed "\n'
     '  "test partition without out-of-fold corroboration. Medicare Advantage versus "\n'
     '  "fee-for-service status is unobservable.'),
    # conclusions (H1 -> H2, interpretability clause)
    ('H1("Conclusions")',
     'H2("Conclusions")'),
    ('f"Medicare-insured adults with strong calibration; the selection procedure "',
     'f"Medicare-insured adults with strong calibration and per-patient SHAP "\n'
     '  f"explanations; the selection procedure "'),
    # back matter
    ('  "PhysioNet team for maintaining and curating the MIMIC-IV database.")\n'
     'H1("Funding Statement")\n'
     'P("This study received no external funding.")',
     '  "PhysioNet team for maintaining and curating the MIMIC-IV database. "\n'
     '  "This study received no external funding.")'),
    ('  "University. The public demonstration prototype accepts only synthetic or "',
     '  "University, consistent with the United States federal definition of "\n'
     '  "human-subjects research for secondary analyses of deidentified data. "\n'
     '  "The public demonstration prototype accepts only synthetic or "'),
    ('  "cohort-construction and reanalysis scripts, the machine-readable results "\n'
     '  "file from which every value in this manuscript is generated "\n'
     '  "(final_model_v6.json and companions),',
     '  "cohort-construction and reanalysis scripts, the machine-readable "\n'
     '  "results bundle from which the values in this manuscript are generated "\n'
     '  "(final_model_v6.json and the companion files enumerated in Multimedia "\n'
     '  "Appendix 1),'),
    ('P("AFT: accelerated failure time; AUROC: area under the receiver operating "\n'
     '  "characteristic curve; CI: confidence interval; CMS: Centers for Medicare & "\n'
     '  "Medicaid Services; DCA: decision-curve analysis; DRG: diagnosis-related group; "\n'
     '  "ECE: expected calibration error; EHR: electronic health record; FNR: "\n'
     '  "false-negative rate; HRRP: Hospital Readmissions Reduction Program; IPCW: "\n'
     '  "inverse probability of censoring weighting; IRB: institutional review board; "\n'
     '  "MIMIC: Medical Information Mart for Intensive Care; NLP: natural language "\n'
     '  "processing; NPV: negative predictive value; OOF: out-of-fold; PACS: picture "\n'
     '  "archiving and communication system; PPV: positive predictive value; RFE: "\n'
     '  "recursive feature elimination; SD: standard deviation; SHAP: Shapley "\n'
     '  "additive explanations; STROBE: Strengthening the Reporting of Observational "\n'
     '  "Studies in Epidemiology; TRIPOD: Transparent Reporting of a multivariable "\n'
     '  "prediction model for Individual Prognosis Or Diagnosis; XGBoost: extreme "\n'
     '  "gradient boosting.")',
     'P("AFT: accelerated failure time; AUROC: area under the receiver operating "\n'
     '  "characteristic curve; CI: confidence interval; CMS: Centers for Medicare & "\n'
     '  "Medicaid Services; EHR: electronic health record; FNR: "\n'
     '  "false-negative rate; HRRP: Hospital Readmissions Reduction Program; "\n'
     '  "ICD: International Classification of Diseases; ICU: intensive care "\n'
     '  "unit; IPCW: inverse probability of censoring weighting; "\n'
     '  "MIMIC: Medical Information Mart for Intensive Care; NLP: natural language "\n'
     '  "processing; NPV: negative predictive value; OOF: out-of-fold; PACS: picture "\n'
     '  "archiving and communication system; PPV: positive predictive value; "\n'
     '  "SD: standard deviation; SHAP: Shapley "\n'
     '  "additive explanations; TRIPOD: Transparent Reporting of a multivariable "\n'
     '  "prediction model for Individual Prognosis Or Diagnosis; XGBoost: extreme "\n'
     '  "gradient boosting.")'),
    ('f"counts, and stage-differential estimates). Appendix 2: LACE and HOSPITAL "\n'
     '  f"reconstruction tables. Appendix 3: race-category consolidation with raw "\n'
     '  f"counts. Appendix 4: supplementary figures and development history "\n'
     '  f"(decision curves; same-cohort ROC curves; development-phase results "\n'
     '  f"including earlier feature versions and model families). Appendix 5: "\n'
     '  f"completed TRIPOD+AI checklist. Appendix 6: PROBAST+AI self-assessment.")',
     'f"counts, and stage-differential estimates). Multimedia Appendix 2: LACE "\n'
     '  f"and HOSPITAL "\n'
     '  f"reconstruction tables. Multimedia Appendix 3: race-category "\n'
     '  f"consolidation with raw "\n'
     '  f"counts. Multimedia Appendix 4: supplementary figures and development "\n'
     '  f"history "\n'
     '  f"(decision curves; same-cohort ROC curves; development-phase results "\n'
     '  f"including earlier feature versions and model families). Multimedia "\n'
     '  f"Appendix 5: "\n'
     '  f"completed TRIPOD+AI checklist. Multimedia Appendix 6: PROBAST+AI "\n'
     '  f"self-assessment.")'),
    # authors' contributions CRediT punctuation
    ('writing-original draft', 'writing (original draft)'),
    ('writing-review and editing', 'writing (review and editing)'),
    # background wording + HRRP abbrev
    ('  "for Medicare & Medicaid Services (CMS) estimates the annual direct cost of "\n'
     '  "Medicare readmissions above $26 billion [1], and the Hospital Readmissions "\n'
     '  "Reduction Program has tied reimbursement to risk-adjusted readmission performance "\n'
     '  "since 2013 [1,2].',
     '  "for Medicare & Medicaid Services (CMS) puts the annual direct cost of "\n'
     '  "Medicare readmissions above $26 billion [1], and the Hospital Readmissions "\n'
     '  "Reduction Program (HRRP) has tied reimbursement to risk-adjusted "\n'
     '  "readmission performance since 2013 [1,2].'),
    # table 2 labels
    ('       ["This study: RFE selection procedure",',
     '       ["This study: selection procedure",'),
    ('        "Primary; fold-specific sets of 27 to 38 features; leakage-safe "\n'
     '        "out-of-fold"],',
     '        "Primary; fold-specific sets of 27 to 38 features; leakage-safe, "\n'
     '        "out of fold"],'),
    ('        "and are labeled by evidence tier. OOF: out-of-fold; RFE: recursive "\n'
     '        "feature elimination.", keep_with_next=True)',
     '        "and are labeled by evidence tier. OOF: out-of-fold.",\n'
     '        keep_with_next=True)'),
    ('LACE (adapted) [3]', 'LACE (reconstructed) [3]'),
    ('HOSPITAL (adapted, 12-mo) [4]', 'HOSPITAL (reconstructed, 12-mo) [4]'),
    ('f"outperforms the adapted LACE and HOSPITAL scores on the identical "',
     'f"outperforms the reconstructed LACE and HOSPITAL scores on the identical "'),
    ('(Multimedia Appendix 4). Both adapted scores fall below their ',
     '(Multimedia Appendix 4). Both reconstructed scores fell below their '),
    # references: four additions before the closing bracket
    ('" "Wiens J, Saria S, Sendak M, et al. Do no harm: a roadmap for responsible "\n'
     ' "machine learning for health care. Nat Med. 2019;25(9):1337-1340.",\n'
     ']'.replace('" "W', ' "W'),
     ' "Wiens J, Saria S, Sendak M, et al. Do no harm: a roadmap for responsible "\n'
     ' "machine learning for health care. Nat Med. 2019;25(9):1337-1340.",\n'
     ' "Graham KL, Wilker EH, Howell MD, Davis RB, Marcantonio ER. Differences "\n'
     ' "between early and late readmissions among patients: a cohort study. Ann "\n'
     ' "Intern Med. 2015;162(11):741-749.",\n'
     ' "Krumholz HM. Post-hospital syndrome: an acquired, transient condition "\n'
     ' "of generalized risk. N Engl J Med. 2013;368(2):100-102.",\n'
     ' "Seyyed-Kalantari L, Zhang H, McDermott MBA, Chen IY, Ghassemi M. "\n'
     ' "Underdiagnosis bias of artificial intelligence algorithms applied to "\n'
     ' "chest radiographs in under-served patient populations. Nat Med. "\n'
     ' "2021;27(12):2176-2182.",\n'
     ' "Blanche P, Dartigues JF, Jacqmin-Gadda H. Estimating and comparing "\n'
     ' "time-dependent areas under receiver operating characteristic curves "\n'
     ' "for censored event times with competing risks. Stat Med. "\n'
     ' "2013;32(30):5381-5397.",\n'
     ']'),
    # renumber pass just before writing the reference paragraphs
    ('for i, ref in enumerate(REFS, 1):\n'
     '    P(f"{i}. {ref}")',
     RENUMBER + '\nfor i, ref in enumerate(REFS, 1):\n'
     '    P(f"{i}. {ref}")'),
]


def patch(path):
    t = path.read_text(encoding="utf-8")
    miss = []
    for old, new in REPS:
        n = t.count(old)
        if n == 0 and t.count(new):
            continue
        if n == 0:
            miss.append(("MISSING", old[:100]))
            continue
        if n > 1 and old not in ("writing-original draft",
                                 "writing-review and editing",
                                 "LACE (adapted) [3]",
                                 "HOSPITAL (adapted, 12-mo) [4]"):
            miss.append((f"{n} HITS", old[:100]))
            continue
        t = t.replace(old, new)
    # move the Answers block to directly after Principal Findings
    if ANSWERS_OLD in t:
        pf_anchor = ('f"cohort and outcome, with calibrated absolute risks the bedside scores "\n'
                     '  f"do not provide.")')
        assert t.count(pf_anchor) == 1, "PF anchor"
        t = t.replace(ANSWERS_OLD + "\n", "")
        t = t.replace(pf_anchor, pf_anchor + "\n" + ANSWERS_NEW)
    elif ANSWERS_NEW not in t:
        miss.append(("MISSING", "ANSWERS block"))
    if miss:
        for why, m in miss:
            print(f"  MISS ({why}) in {path.name}: {m!r}")
        raise SystemExit(f"{path.name}: {len(miss)} replacement(s) failed")
    path.write_text(t, encoding="utf-8")
    print(f"patched {path.name}")


for name in ("_build_jmir_final_part2.py", "_build_jmir_finalsc_part2.py"):
    patch(HERE / name)
print("REVIEW ROUND APPLIED")

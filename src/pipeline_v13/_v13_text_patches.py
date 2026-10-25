# -*- coding: utf-8 -*-
"""Documentation fixes for v13, applied to the v12 builders before they are
copied to v13 (idempotent)."""
from pathlib import Path

H = Path(__file__).resolve().parent


def patch(fn, pairs):
    p = H / fn; s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        if s.count(a) == 0 and s.count(b) >= 1:
            continue
        assert s.count(a) == 1, (fn, s.count(a), a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8"); print("patched", fn)


patch("_build_jamia_v12.py", [
 # constancy claim limited to the relevant signature
 ('''   f"the whole record. It and two interaction terms were removed. No other "
   f"candidate showed either signature, and every history feature of the "''',
  '''   f"the whole record. It and two interaction terms were removed. No other "
   f"candidate showed the first-admission signature (sex, recorded once per "
   f"patient, is constant for a legitimate reason), and every history feature of the "'''),
 # joint resampling for the race contrast
 ('''   "with plus-one correction). Model-to-model contrasts are paired; the "
   "White vs Black contrast resamples patients within each group. Formal "''',
  '''   "with plus-one correction). Model-to-model contrasts are paired; the "
   "White vs Black contrast resamples patients jointly, so a patient "
   "recorded in both groups receives the same multiplicity in both. Formal "'''),
 ('wb = FM["fairness"]["white_minus_black"]\n',
  'wb = {"point": SC["white_minus_black_auroc_joint"]["point_observed"],\n'
  '      "ci95": SC["white_minus_black_auroc_joint"]["ci95"],\n'
  '      "p_bootstrap": SC["white_minus_black_auroc_joint"]["p_two_sided_plus1"]}\n'),
 ('WBS = SC["white_minus_black_specificity"]\n',
  'WBS = {"point": SC["white_minus_black_specificity_joint"]["point_observed"],\n'
  '       "ci95": SC["white_minus_black_specificity_joint"]["ci95"]}\n'),
])

patch("_build_jamia_supp_v12.py", [
 ('''P(f"Within-patient constancy. Among the {IA['n_multi_admission_patients']:,} "
  f"patients with more than one admission, no final feature was constant "
  f"within every patient; the highest share was "
  f"{pct(max(v for k, v in CONST.items() if k != 'los_trend_180d'))} for a "
  f"binary unit flag, against {pct(CONST['los_trend_180d'])} for the "
  f"removed trend.")''',
  '''P(f"Within-patient constancy. Among the {IA['n_multi_admission_patients']:,} "
  f"patients with more than one admission, the removed trend was constant "
  f"within {pct(CONST['los_trend_180d'])} of patients. Sex is constant by "
  f"construction (recorded once per patient) and is a legitimate predictor; "
  f"among the other final features the highest share was "
  f"{pct(max(v for k, v in CONST.items() if k not in ('los_trend_180d', 'gender')))} "
  f"for a binary unit flag. Constancy is a screen, not evidence of leakage; "
  f"it is combined with the first-admission test and recomputation from raw "
  f"timestamps.")'''),
 # S4 denominators and the discharge interval definition
 ('''  f"when a component is missing. The tree learner handles missing values "
  f"natively.''',
  '''  f"when a component is missing. Percentages in this paragraph that come "
  f"from the rebuild report use all {CT['original_admissions']:,} Medicare "
  f"admissions as denominator; the audit workbook reports missingness on the "
  f"analysis cohort of {CT['cohort_v2_admissions']:,} admissions, which is "
  f"why its values differ slightly. The tree learner handles missing values "
  f"natively.'''),
 ('''  f"last discharge is missing when no earlier admission exists "
  f"({pct(TB['time_since_last_discharge']['set_missing_share'])} of "
  f"admissions; previously the global median, 80.22 days); recency-weighted "
  f"frequency is zero for those admissions.''',
  '''  f"last discharge is the interval from the previous discharge of any "
  f"insurance and is missing only for first observed admissions "
  f"({pct(TB['v13_documentation_time_enforced']['time_since_last_discharge']['share_missing_new'])} "
  f"of admissions; the source table had used the previous Medicare-insured "
  f"discharge and the global median, 80.22 days, when none existed); "
  f"recency-weighted frequency is zero for first observed admissions.'''),
 # S7 bootstrap description
 ('''  "fitting were not repeated within draws. Race strings were consolidated by "''',
  '''  "fitting were not repeated within draws. The White vs Black contrasts "
  "resample the union of the two patient sets jointly, so a patient recorded "
  "in both groups receives the same multiplicity in both. Race strings were consolidated by "'''),
 # S13 overlap sentence now describes the joint design
 ('''  f"descriptive). The White minus Black contrast resamples patients within "
  f"each group independently; {SC['patients_in_both_white_and_black']} of "
  f"{SC['patients_white'] + SC['patients_black']:,} patients carry both race "
  f"labels across admissions and appear in both groups, a dependence the "
  f"resampling ignores. The point estimate is the observed difference; the "
  f"bootstrap mean appears in Supplementary Appendix 1. "''',
  '''  f"descriptive). {SC['patients_in_both_white_and_black']} of "
  f"{SC['patients_white'] + SC['patients_black'] - SC['patients_in_both_white_and_black']:,} "
  f"patients carry both race labels across admissions and appear in both "
  f"groups; the contrasts therefore resample the union of patients jointly "
  f"(point estimates are observed differences; bootstrap means appear in "
  f"Supplementary Appendix 1). "'''),
 (' "time_since_last_discharge": "Days from the previous Medicare-insured discharge to the index admission; missing if none",',
  ' "time_since_last_discharge": "Days from the previous discharge of any insurance to the index admission; missing for first observed admissions",'),
 (' "prior_readmission_count": "Earlier Medicare-insured admissions in the prior 365 days followed by a readmission within 30 days",',
  ' "prior_readmission_count": "Earlier Medicare-insured admissions in the prior 365 days followed by a readmission within 30 days (cohort-table definition, verified)",'),
])

patch("_v12_appendices.py", [
 (' "time_since_last_discharge": ("Days from the previous Medicare-insured discharge to the index admission; missing if none (v7 to v10 used the global median, 80.22)", "all earlier admissions", HV["time_since_last_discharge"]["nonfill_match_cohort"]),',
  ' "time_since_last_discharge": ("Days from the previous discharge of any insurance to the index admission; missing for first observed admissions (v13; v7 to v10 used the previous Medicare-insured discharge and the global median 80.22)", "all earlier admissions", 1.0),'),
 ('''    pd.DataFrame([{"feature": k, **v} for k, v in TB.items() if isinstance(v, dict) and "new_missing" in v]
                 ).to_excel(w, sheet_name="Table_Rebuild", index=False)''',
  '''    pd.DataFrame([{"feature": k, "population": "all 244,576 Medicare admissions (before death exclusion)",
                   "version": "v11", **v} for k, v in TB.items() if isinstance(v, dict) and "new_missing" in v]
                 ).to_excel(w, sheet_name="Table_Rebuild", index=False)
    pd.DataFrame([{"feature": k, "version": "v12 (discharge cutoff on counts)", **v}
                  for k, v in TB.get("v12_cutoff_enforced", {}).items()] +
                 [{"feature": k, "version": "v13 (documentation-time cutoff)", **v}
                  for k, v in TB.get("v13_documentation_time_enforced", {}).items()]
                 ).to_excel(w, sheet_name="Cutoff_Rebuild", index=False)'''),
])

patch("_v11_checklists.py", [
 ('"Public repository with tagged release and Zenodo archive"', '"Public repository; the submitted version will be tagged and archived at Zenodo before publication"'),
 ('"Model artifact, encodings, and feature order released; freely available under MIT license"',
  '"Model artifact, encodings, and feature order are in the public repository (MIT license) and will be in the tagged release"'),
 ('"Documented availability audit, first-admission and constancy tests, recomputation of history features from timestamps, laboratory results required to be stored before discharge; some index-stay features rely on documented timestamps only"',
  '"Documented availability audit, first-admission and constancy tests, recomputation of history features from timestamps, laboratory, chart, eMAR, and prescription features rebuilt with documentation-time cutoffs; order and transfer features rely on the source extraction filter and spot checks"'),
 ('"Quality concern: low. Rationale: availability was audited by documentation and enforced by recomputation from raw tables with discharge cutoffs for history, laboratory, order, and medication features; the residual uncertainty concerns a minority of index-stay features that rely on documented extraction filters only.",',
  '"Quality concern: low. Rationale: availability was audited by documentation and enforced by recomputation from raw tables with discharge and documentation-time cutoffs for history, laboratory, chart, eMAR, and prescription features; order and transfer features were extracted with a discharge-time filter and spot-checked rather than fully recomputed.",'),
 ('"Risk of bias: high for the historically exposed test partition (signalling question 4.5) and low for the out-of-fold procedure estimate, which replicates selection and encoding inside each outer fold on data never used for earlier model choices. Under the PROBAST+AI reclassification rules, the large development data and internal validation do not remove the concern created by prior test-partition exposure, so the test-partition results are presented as tertiary evidence.",',
  '"Risk of bias: high for the historically exposed test partition (signalling question 4.5) and unclear for the out-of-fold procedure estimate: selection and encoding are replicated inside each outer fold, but the model family and hyperparameters were chosen during earlier development that used the same development data and the test partition, and that choice is not replicated within the resampling (4.6). Under the PROBAST+AI reclassification rules, the large development data and internal validation do not remove the concern created by prior test-partition exposure, so the test-partition results are presented as tertiary evidence.",'),
 ('("4.6 If resampling methods were used to evaluate model performance, were all model development steps replicated in the resampling process?", PY, "Feature selection and encodings replicated inside each outer fold; hyperparameters fixed; consensus aggregation itself not replicated"),',
  '("4.6 If resampling methods were used to evaluate model performance, were all model development steps replicated in the resampling process?", PN, "Feature selection and encodings replicated inside each outer fold; the choice of model family and hyperparameters, made in earlier development, and the consensus aggregation are not replicated"),'),
 ('"center; low for that setting.")', '"center; low for that setting. The out-of-fold rating is unclear rather than low because earlier development choices are not replicated within the resampling.")'),
])

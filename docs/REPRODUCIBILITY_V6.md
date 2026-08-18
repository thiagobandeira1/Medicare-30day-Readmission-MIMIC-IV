# Reproducibility report: v6 pipeline (2026-08-12)

v6 = v5 final model (unchanged: consensus-31 XGBoost, artifacts_v5/) plus
corrected inference, new supplementary analyses, and estimand-clean reporting.
No model, feature set, threshold, or prediction changed between v5 and v6;
what changed is how contrasts are computed/described and the manuscript text.

## Environment
- Same as v5 (see REPRODUCIBILITY_V5.md): conda env `capstone`, Python 3.12,
  XGBoost 3.2.0, scikit-learn 1.7.2, pandas 2.3.3, lifelines, scikit-survival.
- Seed 42; threads pinned to 8.

## Commands, in order (from presentation-update/)
1. `python _v6_extras.py`  (~20 min)
   -> results_v6/v6_extras.json, results_v6/final_model_v6.json (canonical:
   deep copy of final_model_v5.json with W-B contrast replaced by the
   stratified 5,000-draw version, baseline-uplift CIs/P recomputed at 5,000
   draws, and a v6_extras section appended)
   Contents:
   a. Fixed-31 vs fixed-142 paired grouped CV on the development partitions
      (same 5 grouped folds, fold-local encodings, 600 trees), paired
      patient-cluster bootstrap 5,000 draws, plus-one-corrected P.
   b. LACE and HOSPITAL-12m uplifts re-bootstrapped: paired same-draw
      patient-cluster resampling, 5,000 draws, plus-one correction.
   c. White-Black AUROC gap re-bootstrapped: stratified design (patients
      resampled independently WITHIN each subgroup), 5,000 draws; the
      subgroups are disjoint, so this is not a paired comparison and is no
      longer described as one.
   d. Death/readmission same-date tie-rule sensitivity: primary death-first
      vs readmission-first (ties identified as event=death & binary
      readmission label=1); AFT refit under each rule; Harrell C, day-1/29
      td-AUROC, Aalen-Johansen 30-day CIFs.
   e. Transfer-band documentation: delta = exact timestamp difference;
      band -2<delta<=0 days; 650 admissions (458 train / 52 val / 140 test);
      zero records overlap by more than 2 days.
2. `python _build_jmir_v6_part2.py` -> "Paper JMIR AI Submission v6.docx"
   (asserts abstract <= 440 words and the SHAP facts the clinical prose uses)
3. Word COM export -> "Paper JMIR AI Submission v6.pdf"
4. `python _v6_appendices.py`; `python _build_tripod_checklist.py`
5. `python _v6_consistency.py` -> CONSISTENCY_REPORT_V6.md (value-level
   agreement between canonical JSON and manuscript; stale-string scan;
   P-value precision scan)

## Key v6 corrections of record
- Partition description corrected: patient-grouped 80/20 development-test
  split, then 10% of development to validation => ~72/8/20 (train 170,677;
  val 18,557; test 47,672). The "80/10/10" phrase in v5 was wrong.
- HOSPITAL Methods corrected: the primary reconstruction uses the original
  12-month prior-admission window (AUROC 0.6493); the 6-month variant
  (0.6377) is a development-phase sensitivity only (Appendix 2).
- P values: every reported P now comes from a 5,000-draw plus-one-corrected
  empirical bootstrap; smallest reportable value is P<.001. No P is reported
  beyond the precision its resample count supports.
- White-Black contrast redescribed (and recomputed) as a stratified
  disjoint-subgroup bootstrap, not "paired".
- Threshold renamed development-derived; validation partition's role in
  selection folds disclosed in Methods and Limitations.
- discharge_location_te provenance language softened to observation +
  conservative exclusion (original code unavailable; pattern inconsistent
  with a single training-only map).
- Tie rule no longer called "prespecified"; both orderings reported.
- Calibration values of record: slope 1.04 (1.0363), ECE 0.007 (0.0068);
  earlier summary documents that carried 1.03/0.006 were stale (the v5
  manuscript itself was already correct).

## Headline numbers (canonical: results_v6/final_model_v6.json)
- PRIMARY dev-only OOF of the selection procedure: 0.7748 [0.7715-0.7782]
- SECONDARY fixed consensus-31 dev CV: 0.7756 +/- 0.0020
- TERTIARY fixed consensus-31, historically exposed test: 0.7738
  [0.7671-0.7804]
- NEW fixed-31 vs fixed-142, identical dev folds: 0.7756 vs 0.7757,
  paired diff -0.0001 [-0.0009, 0.0007], P=.758 (5,000 draws)
- (uplift and W-B P values: see v6_extras.json / manuscript, 5,000 draws)

## v6.1 addendum (2026-08-12): surgical corrections, no model rerun

Builders: _build_jmir_v61_part1.py / _build_jmir_v61_part2.py ->
"Paper JMIR AI Submission v6.1.docx" + PDF (14 pages).
Consistency: _v61_consistency.py -> CONSISTENCY_REPORT_V6.1.md (PASS).

Canonical point-estimate rounding rule (documented per item 4):
  point estimate = observed difference computed from UNROUNDED canonical
  point AUROCs, rounded once at print time. The 5,000-draw patient-cluster
  bootstrap distribution supplies the 95% CI and the plus-one-corrected
  two-sided P ONLY; its mean is NOT used as the point estimate.
  Fields corrected in results_v6/final_model_v6.json and v6_extras.json:
    v6_extras.highdraw_contrasts_5000.{lace_uplift,hospital12_uplift,
    white_minus_black}.point  (were bootstrap means 0.1688/0.1245/0.0346;
    now observed 0.1688/0.1244/0.0344). Manuscript-facing fields
    (baselines.uplift_*, fairness.white_minus_black.point) were already
    correct and unchanged.

Bootstrap scope (documented per item 6): all intervals resample patients and
their stored prediction-outcome pairs; feature selection, encoding, and model
fitting are NOT repeated inside resamples. Intervals are conditional on the
fitted CV/test models and do not capture retraining or reselection variability.

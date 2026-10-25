# Leakage correction, v7 (October 2026)

## What was wrong

`los_trend_180d` was documented as a 180-day length-of-stay trend computed
from the patient's earlier stays. An empirical test on first-ever admissions
(patients with no earlier admission of any kind, so a history feature can
carry no information) showed that it encodes later admissions instead:

| Check (78,419 first-ever cohort admissions) | Value |
|---|---|
| Share with a nonzero value | 44.2% |
| Later admission among nonzero | 100% |
| Nonzero among patients with no later admission | 0% |
| 30-day readmission, nonzero vs zero | 29.8% vs 3.1% |
| AUROC of "nonzero" for any later admission | 0.93 |

Two interactions built from it, `los_trend_x_prior_6m` and
`los_trend_x_prior_admits`, were also removed. A scan of every other
candidate predictor flagged no further feature
(`results_v7/leak_test_v7.json`); outpatient-record features were confirmed
to use only records dated on or before discharge.

## Impact

| Result | Before (v6) | After (v7) |
|---|---|---|
| Out-of-fold AUROC, complete selection procedure | 0.7748 | 0.7058 (95% CI 0.7018 to 0.7104) |
| Test AUROC, final model | 0.7738 | 0.7064 (95% CI 0.6973 to 0.7150) |
| White minus Black test AUROC | 0.034 (P<.001) | 0.006 (P=.57) |
| AUROC trend across calendar eras | 0.744 to 0.856 | 0.692 to 0.706 (flat) |
| Consensus feature set | 31 features | 47 features |

The leak also explained the apparent advantage for first-time admissions, the
rising AUROC across calendar eras, and the apparent racial gap in
discrimination; all disappeared after the correction. A separate review found
that the apparent day-1 discrimination peak reflects 2,368 administrative
continuations beginning within 1 hour of discharge; the manuscript now
reports the timing curve both with and without them.

## Other corrections in v7

* Daily time-dependent AUROC now uses the estimand described in the text
  (deaths before the evaluation day kept among controls); the earlier
  censored variant is retained in `final_model_v7.json` for reference.
* Calibration-in-the-large is reported alongside the calibration slope.
* The final classifier uses 600 trees without early stopping, as stated.
* LACE and HOSPITAL comparisons disclose their coded (billing-derived)
  components and add logistic refits with and without them.
* Figure 1 shows 301,452 non-Medicare admissions excluded.

## Where things are

* Pipeline: `src/pipeline_v7/` (scripts were run from the sibling
  `presentation-update` working folder; paths are relative to it).
* Results: `results_v7/` (aggregate JSON only; row-level predictions are
  not committed, per the PhysioNet data use agreement).
* Figures: `figures_v7/`.
* Superseded: `results_v5/`, `results_v6/`, `src/pipeline_v5_v6/`, tag
  `v6.4-submission` and Zenodo version 10.5281/zenodo.21987702.

# Versions 8 to 11 (October 2026)

* **v8 to v10** added audits without changing the model: recomputation of
  every history feature from raw admission timestamps, a within-patient
  constancy scan, subgroup calibration, and a corrected continuation
  sensitivity rule (chained 1-hour continuations merged into one stay with
  the 30-day window starting at the end of that stay). Scripts are in
  `src/pipeline_v8_v10/`; aggregate outputs in `results_v7/*_v8.json`,
  `*_v9.json`, `*_v10.json`.
* **v11** found a second defect by comparing stored columns with raw tables:
  the inherited capstone feature table had imputed 16 continuous
  measurements with medians computed over all partitions (including test)
  and had zero-filled 5 others, so no final predictor showed any
  missingness. The table was rebuilt from raw MIMIC-IV tables with missing
  values preserved (laboratory results kept only when stored on or before
  discharge) and the complete pipeline was rerun (`src/pipeline_v11/`,
  `results_v11/`, `figures_v11/`). The consensus model has 48 features;
  out-of-fold AUROC 0.7058 (unchanged), test AUROC 0.7060, calibration
  slope 0.99. `results_v11/table_rebuild_v11.json` records, per feature,
  the true missingness and the agreement with the old values where both
  exist.

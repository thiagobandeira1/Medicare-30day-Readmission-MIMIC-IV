# Phase 1 Reproduction Report

| Manuscript claim | Artifact value | Match | Producing script | Artifact |
|---|---|---|---|---|
| Cohort 236,906 admissions | 236906 | EXACT | _reanalysis_step1.py | cohort_v2_counts.json |
| Test AUROC 0.7800 | 0.78 | EXACT | _reanalysis_step2.py | binary_v2.json |
| 5-fold grouped CV 0.7828 | 0.7828 | EXACT | _reanalysis_step2.py | binary_v2.json |
| Fold-local TE CV 0.7771 | 0.7771 | EXACT | _reanalysis_step10_robustness.py | robustness_v3.json |
| 51-feature billing-free 0.7762 | 0.7762 | EXACT | _reanalysis_step10_robustness.py | robustness_v3.json |
| AFT Harrell C 0.7422 | 0.7422 | EXACT | _reanalysis_step9b_model.py | survival_v3.json |
| Day-1 td-AUROC 0.862 | 0.8616 | EXACT | _reanalysis_step9b_model.py | survival_v3.json |
| LACE 0.6050 | 0.605 | EXACT | _reanalysis_step6.py | baselines_v2.json |
| HOSPITAL 12m 0.6493 | 0.6493 | EXACT | _reanalysis_step10_robustness.py | robustness_v3.json |
| White-Black delta 0.0348 | 0.0348 | EXACT | _reanalysis_step4.py | fairness_dca_v2.json |

Provenance notes: the 207-pool originates in the RFE notebooks (Feature_Selection_RFE_v2 / results/rfe_selection_results.json, 67 selected); race_te removal in _run_race_ablation.py (test-based, superseded by the validation-based ablation in _reanalysis_step4.py); the 66-feature artifact is readmission-api/artifacts/model.json. All v2/v3 artifacts above were produced in this session with seed 42 and match the manuscript exactly; leakage audit (feature_provenance_audit.csv + results_v4/feature_audit_207.csv) finds no forward-looking predictor in the eligible pool.
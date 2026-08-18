# TRIPOD+AI Reporting Checklist

**Manuscript:** Predicting 30-Day Hospital Readmission in Medicare Patients: An Interpretable Gradient-Boosting Model on MIMIC-IV v3.1
**Standard:** TRIPOD+AI (Collins et al., *BMJ* 2024): Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis: AI extension.

Status legend: ✅ addressed · ◐ partially addressed · ☐ to add before submission

| # | TRIPOD+AI item | Where addressed | Status |
|---|---|---|---|
| **Title & Abstract** | | | |
| 1 | Identify as a study developing/validating a prediction model; specify it is ML | Title; Abstract | ✅ |
| 2 | Structured abstract: objectives, data, methods, results, conclusions | Abstract | ✅ |
| **Introduction** | | | |
| 3a | Background and rationale, including intended clinical use | §Introduction, §4 Motivation | ✅ |
| 3b | Study objectives / research questions | §2 Goals, §3 Objectives | ✅ |
| **Methods: Data** | | | |
| 4a | Source of data (MIMIC-IV v3.1), study design | §6.1 | ✅ |
| 4b | Study dates / data span (2008 to 2022) | §6.1 | ✅ |
| 5a | Eligibility criteria (insurance == Medicare) | §6.2 | ✅ |
| 5b | **Cohort flow diagram** (inclusions/exclusions) | Figure: `cohort_flow.png` | ✅ (Tier 1) |
| 6a | Outcome definition (`readmit_30d`, pre-discharge timestamps) | §6.2 | ✅ |
| 6b | Blinding of outcome assessment / leakage prevention | §7.2 | ✅ |
| 7a | Predictors and how/when measured | §7.3, feature tables | ✅ |
| 7b | Predictor selection (RFE from 207-feature pool) | Feature_Selection_RFE notebook → new §7.x | ◐ add to manuscript |
| 8 | Sample size justification | §6.2, §7.2 (244,576; 80/20) | ◐ add explicit power note |
| 9 | Missing-data handling (NaN-as-signal; median impute for survival) | §5.3, §7.2 | ✅ |
| **Methods: Analysis** | | | |
| 10a | How predictors were handled (encoding, target encoding OOF) | §7.2 | ✅ |
| 10b | Model type and building procedure (4 GBM families, 10 seeds) | §7.1, §7.4 | ✅ |
| 10c | Hyperparameter tuning protocol (Optuna; tuned on inner-val only) | §7.1 | ☐ state #trials, objective, no-test-leakage |
| 10d | Train/validation/test split (patient-grouped GroupShuffleSplit) | §7.2 | ✅ |
| 11 | Performance measures (AUROC, AUPRC, Brier, **ECE, calibration slope/intercept**) | §8.3, Statistical_Rigor | ✅ (Tier 1) |
| 12 | Model comparison & **uncertainty (bootstrap 95% CIs, paired tests)** | Statistical_Rigor → §8.x | ✅ (Tier 1) |
| **Methods: AI specifics** | | | |
| 13a | Software, packages, versions | §7.1, Reproducibility | ✅ |
| 13b | Code / model availability | §Reproducibility (GitHub repo) | ✅ |
| 13c | Fairness considerations across subgroups | §9.1 (narrative) → **fairness analysis pending** | ☐ Tier 2 (subgroup AUROC) |
| **Results** | | | |
| 14a | Participant flow and characteristics | **Table 1** (`table1_cohort.csv`), Figure cohort_flow | ✅ (Tier 1) |
| 14b | Comparison of development vs validation data | §7.2 (same cohort, patient split) | ✅ |
| 15 | Model specification (final features, deployed XGBoost) | §7.4, SHAP §8.4 | ✅ |
| 16 | Model performance with **confidence intervals** | Statistical_Rigor → §8.x | ✅ (Tier 1) |
| 17 | Model updating / recalibration | n/a (single development cohort) | ✅ n/a |
| **Discussion** | | | |
| 18 | Limitations (single-center, SDOH, external readmissions, LACE caveats) | §9.3 | ✅ |
| 19a | Interpretation vs prior work (same-cohort LACE/HOSPITAL + literature) | §8.5, Baselines notebook | ◐ expand related work |
| 19b | Clinical implications / actionability | §9.1, time-resolved survival | ✅ |
| 20 | Generalizability (**temporal validation**) | **pending** | ☐ Tier 2 |
| **Other** | | | |
| 21 | Funding / conflicts | §Acknowledgments | ◐ confirm |
| 22 | Data/code/protocol availability | §Reproducibility | ✅ |

## Summary of gaps to close before submission
- **§7.x feature selection**: add the RFE method + result (67 features, Jaccard 0.44 vs hand-crafted V7) to the manuscript text. *(notebook done)*
- **§10c hyperparameter protocol**: state the number of Optuna trials, the objective, and that tuning used the inner-validation split only (no test leakage).
- **§8.x uncertainty**: fold the bootstrap CIs, paired comparisons, and calibration (ECE/slope/intercept) into the Results. *(notebook done)*
- **§13c / §20 fairness + temporal validation**: Tier 2 analyses (subgroup AUROC; train ≤2018 / test ≥2019).
- **Related work**: expand §5 / §8.5 with recent MIMIC-based readmission studies.

Most Methods/Results items are satisfied once the Tier-1 notebooks' outputs (CIs, calibration, Table 1, cohort flow) are written into the manuscript. The remaining ☐ items are the Tier-2 differentiators (fairness, temporal validation) and two short Methods additions.

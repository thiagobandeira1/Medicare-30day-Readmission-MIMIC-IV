# Missing-Data Handling — paste-in section for Capstone_Final_Report.docx

> **Where to insert in the report:** as a new subsection under your Methods / Data section, immediately after the cohort-construction description and before the feature-engineering description. Suggested heading: **"3.x Missing-Data Handling"** (renumber to match your current outline). The same content appears as **§5.3** of the publication notebook (`notebook/Capstone_Final_Notebook.ipynb`), so the report and notebook stay aligned.

---

## Missing-Data Handling

Raw MIMIC-IV lab, vital-signs, and medication tables contain extensive missingness because clinicians order tests selectively rather than measuring every variable on every admission. For specialty labs such as B-type natriuretic peptide (BNP), serum lactate, and albumin, the value is undefined for the majority of admissions. Naïve listwise deletion would discard most of the cohort, and a single global imputation rule (e.g. mean-fill every column) would erase clinically meaningful structure. We therefore handled missingness in **two stages**.

### Stage 1 — Upstream imputation in the DuckDB feature-engineering pipeline

Each `training_table_v_n.parquet` is constructed so that every column is fully populated, with the imputation rule chosen by *feature class* so the imputed value remains semantically meaningful rather than just numerically convenient.

| Feature class | Examples | Source missingness | Strategy in V_n |
|---|---|---|---|
| **Count / event features** | `n_critical_labs`, `n_meds_total`, `prior_admissions_6m`, `n_lab_item_types` | Undefined when nothing measured | **Zero** — semantically valid ("no event observed → count is 0") |
| **Aggregate measurements** | `creatinine_last`, `bnp_last`, `hr_last`, `last_bmi` | Undefined when the lab/vital was never ordered | **Population median** — a clinically defensible proxy for "approximately normal" |
| **Derived deltas / trends** | `creatinine_delta`, `los_trend_180d`, `med_entropy_90d` | Undefined when there is no comparison point | **Zero** — represents "no change observed" |
| **Binary clinical flags** | `ventilator_flag`, `vasopressor_flag`, `hypertension_flag`, `surgical_service_flag` | Absent unless documented in the EHR | **Zero** — consistent with the EHR convention of "absent unless explicitly recorded" |

A consequence of this scheme is that **the published training tables contain zero literal `NaN` values** — the modelling code does not need to impute at run time, and the same training tables can be loaded directly by any downstream library (scikit-learn, LightGBM, XGBoost, CatBoost) without preprocessing.

### Preserving the missingness signal

A well-known limitation of value-imputation is that the *fact* of missingness is partially erased — for clinical risk, "no creatinine drawn" can itself be informative ("the clinician did not consider renal function a concern"). We mitigate this risk by **retaining lab-count features** (`n_labs_total`, `n_lab_item_types`, `n_critical_labs`) in every V_n. These columns are themselves fully observed (their imputed value is the semantically correct zero), and they allow the model to recover the missingness signal indirectly: an admission with a low `n_lab_item_types` value tells the model that few measurements were taken, even though no individual measurement column carries an explicit `NaN`.

### Stage 2 — Defensive missing-value support in the modelling pipelines

Although the published training tables are already imputed, our model pipelines retain defensive missing-value handling so that the same code runs without modification on any future re-engineered table that may contain `NaN`.

- **Tree-based models** — LightGBM, XGBoost, CatBoost, and scikit-learn's HistGradientBoosting — handle `NaN` natively. Each split learns whether `NaN` routes to the left or right child during fitting, treating missingness as an informative branch rather than a value that must be filled.
- **Logistic regression and MLP baselines** (used for the V1 → V6 progression in §9 of the notebook) wrap inputs in a `sklearn.pipeline.Pipeline` that begins with `SimpleImputer(strategy="median")` followed by `StandardScaler`. The imputer's median is fit only on the training split, so no value statistics leak across the train / validation / test boundary.

### Empirical confirmation

§5.3 of the publication notebook profiles the V7 training table at run time and confirms (a) zero literal `NaN` values across all 96 columns and (b) the distribution of zero-share values, which serves as a proxy for the implicit-imputation footprint. Roughly two-thirds of the numeric features are continuous quantities with no exact-zero entries (consistent with median imputation of source-missing values), while the remaining one-third are count-type features whose zero entries denote genuine absence of events.

### Limitations

1. **Median imputation can attenuate clinical extremes.** For labs that are missing-not-at-random (e.g. albumin is preferentially ordered for clinically suspicious patients), the median fill biases predicted risk toward the cohort average for patients with truly unmeasured values.
2. **The count-feature workaround is a partial fix.** It preserves the *cardinality* of missingness but not the *identity* of which specific labs were missing — distinguishing "BNP unmeasured" from "lactate unmeasured" requires per-feature missingness flags, which we did not include.
3. **No multiple imputation.** Single-value imputation does not propagate uncertainty about the imputed values into the downstream model's predictions. For risk-scoring applications where calibration matters as much as discrimination, multiple imputation (e.g. via `IterativeImputer` or MICE) is a reasonable extension.

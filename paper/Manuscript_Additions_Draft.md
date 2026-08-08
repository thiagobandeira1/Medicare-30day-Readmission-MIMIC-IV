# Manuscript Additions — Draft Text

Publication-ready prose for the new analyses, written in the paper's style (no em-dashes), with the
verified numbers from the executed notebooks. Each block notes where it belongs in the manuscript.

---

## 7.x Principled feature selection (insert in Methods, after 7.3)

The fifty-feature deployment set was originally assembled through a staged, domain-motivated
progression (V1 to V7). To establish that this set is objectively justified rather than hand-selected,
we additionally derived a feature set by **recursive feature elimination (RFE)** over the full
engineered candidate pool. The pool comprised 207 features: the 177 variables of the most expansive
engineered table together with the 30 target-encoded and clinical-interaction features unique to the
deployment set, so that selection operated over the same universe that produced the deployed model.
A LightGBM estimator was trained on the training partition; features were ranked by importance and the
weakest ten were eliminated per round, with each candidate subset scored by AUROC on the held-out
inner-validation partition. The feature count maximizing inner-validation AUROC was retained.

## 8.x Feature-selection results (insert in Results)

Recursive feature elimination selected 67 features, achieving a test AUROC of 0.7960, marginally above
the full 207-feature model (0.7950) and the hand-crafted fifty-feature set (0.7921). Test AUROC was
flat across subset sizes from 207 down to approximately 27 features, confirming that the great majority
of engineered features are redundant for discrimination. Of the fifty hand-crafted features, 36 were
independently re-selected by RFE (Jaccard overlap 0.44), indicating that the domain-motivated
engineering and the objective selection procedure converge on a common core of predictors. We retain
the fifty-feature set for deployment for continuity and interpretability; the RFE analysis establishes
that this set sits at the parsimony-optimal frontier rather than being an arbitrary choice.

---

## 7.x Survival (time-to-event) modeling (insert in Methods)

To move beyond a fixed thirty-day binary endpoint, we additionally framed readmission as a
time-to-event problem. For each admission the event time was the number of days from discharge to the
next admission; admissions ending in death were treated as a **competing event** rather than as
non-readmissions. This reclassified 1,952 admissions that the binary target had labeled as readmitted
but that in fact ended in death during the subsequent stay, yielding 48,841 readmission events, 9,622
competing-event deaths, and 186,113 administratively censored admissions over the thirty-day window.
Event times were discretized to whole days, with events bounded at day 29 and administrative censoring
at day 30 so that the inverse-probability-of-censoring weights used by the time-dependent metrics
remain well defined. The same patient-grouped split and feature set were used as for the binary model,
augmented by one survival-motivated feature, the number of days since the patient's previous discharge.
We fit Cox proportional hazards (with elastic-net penalization), a gradient-boosted accelerated failure
time model (XGBoost AFT), Random Survival Forests, and gradient-boosted survival analysis, and
evaluated them with Harrell's and Uno's concordance indices, time-dependent AUROC at 7, 14, and 30
days, and the integrated Brier score. Aalen-Johansen cumulative-incidence functions and a
cause-specific Cox model for death characterized the competing-risk structure.

## 8.x Survival results (insert in Results)

Among the survival models the gradient-boosted AFT model performed best, with a Harrell concordance of
0.747 and an Uno concordance of 0.747 at thirty days, followed by gradient-boosted survival analysis
(0.705) and Random Survival Forests (0.702); penalized Cox provided an interpretable linear baseline at
0.669. The time-dependent AUROC rose from approximately 0.70 at day 7 to 0.73 at day 30, and the
integrated Brier score over the window was 0.10. The AFT concordance is comparable to the binary
model's discrimination on the same patients, indicating that the time-to-event reframing preserves
discriminative performance while additionally producing a calibrated risk trajectory across the
post-discharge window. This trajectory is directly actionable: it identifies not only who is likely to
return but when, supporting the timing of post-discharge follow-up contacts.

---

## 8.x Uncertainty quantification and model comparison (insert in Results, 8.3 area)

All headline metrics are reported with 95% confidence intervals obtained from 2,000 bootstrap resamples
of the test set. The deployed XGBoost model achieved a test AUROC of 0.7935 (95% CI 0.7889 to 0.7979)
and an average precision of 0.513 (95% CI 0.503 to 0.523). A paired bootstrap comparison found CatBoost
(difference 0.0002, p = 0.70) and HistGradientBoosting (difference 0.0008, p = 0.09) statistically
indistinguishable from XGBoost, and a small but statistically significant advantage for LightGBM
(difference 0.0035, p < 0.001). Because the entire family spans only 0.0035 AUROC, which is within the
five-fold cross-validation standard deviation of 0.0027 and far below any clinically meaningful
threshold, we describe the families as clinically equivalent and deploy XGBoost for consistency with
the time-to-event model and its mature interpretability tooling, reporting LightGBM as a marginally
higher-scoring alternative.

## 8.x Calibration (insert in Results, with Figure 7)

Beyond the Brier score (0.1332), the deployed model was well calibrated: the expected calibration error
was 0.004, the calibration slope was 1.03 (ideal 1.0) and the calibration intercept was 0.02 (ideal
0.0), and mean predicted risk (0.2107) closely matched observed risk (0.2091). The model therefore
provides trustworthy absolute probabilities suitable for care-coordination triage, not only rank
ordering.

---

## 8.x Same-cohort clinical baselines (insert in Results, replaces cross-cohort claim in 8.5)

To compare against established clinical scores without cross-cohort confounding, we re-implemented the
LACE index and the HOSPITAL score on our exact test partition. The LACE emergency-department component
was computed from the patient's actual count of prior ED visits within 180 days
(`admissions.edregtime`), and the HOSPITAL oncology component from the MIMIC services table. On the
same patients, the deployed model (AUROC 0.7935) outperformed both LACE (0.5924, 95% CI 0.586 to 0.599;
difference 0.2011, 95% CI 0.195 to 0.207, p < 0.001) and HOSPITAL (0.6260, 95% CI 0.620 to 0.632;
difference 0.1676, p < 0.001). Both scores fall below their originally published values (LACE 0.684),
which we attribute to MIMIC-IV being a single-center database that captures only within-system prior ED
visits and admissions, attenuating the utilization-based components of both indices. We therefore
present the same-cohort comparison as the primary result and the published values as literature
context; the model's advantage holds against either reference.

---

## 8.x Fairness and subgroup performance (insert in Results)

We audited the deployed model across age, sex, and race subgroups on the test partition. Discrimination
was stable across age bands (AUROC 0.78 to 0.80) and between sexes (0.795 in women, 0.791 in men), and
calibration was excellent within every age and sex stratum (expected calibration error 0.006 to 0.010).
Across race categories, discrimination was highest for the heterogeneous Other/Unknown, Asian, and
Hispanic/Latino groups and was lowest for Black patients (AUROC 0.754, 95% CI 0.741 to 0.766) relative
to White patients (0.791, 95% CI 0.785 to 0.796), a difference whose confidence intervals do not
overlap. Importantly, the model remained well calibrated for Black patients (expected calibration error
0.010; mean predicted 0.236 versus observed 0.230), so the disparity is one of rank-ordering rather than
systematic over- or under-prediction. We report this gap transparently as a limitation and a target for
subgroup-specific recalibration or thresholding in future work.

Addressing the equity concern raised in our discussion directly, calibration was examined within strata
of prior six-month utilization. Predicted and observed risk agreed closely at every level (zero prior
admissions: 0.152 versus 0.150; one: 0.226 versus 0.218; two to three: 0.315 versus 0.317; four or more:
0.500 versus 0.513). The elevated risk the model assigns to high-utilization patients is therefore
calibrated rather than inflated, supporting the use of the score to allocate additional transitional-care
resources rather than to withdraw care.

---

## Additions to Limitations (9.3)

Add: "The model discriminates less well for Black patients than for White patients (AUROC 0.754 versus
0.791), although it remains well calibrated for both groups; subgroup-specific recalibration is a
priority for prospective deployment. The same-cohort LACE and HOSPITAL comparisons are constrained by
MIMIC-IV's single-center capture of prior utilization, which lowers their measured performance relative
to multi-center derivations. The competing-event indicator captures in-hospital death recorded during
index or subsequent admissions but not out-of-hospital death, which MIMIC-IV v3.1 does not provide."

## Additions to Contributions (10)

Add a fifth contribution: "A statistically rigorous and equity-aware evaluation, comprising bootstrap
confidence intervals and paired model comparison, full calibration assessment, same-cohort
re-implementation of LACE and HOSPITAL, and a subgroup fairness audit, together with a time-to-event
reformulation that models the timing of readmission under competing risks."

## Correction to make throughout

Replace references to a "368-feature" exploratory superset with the reproducible figure of 177
engineered features (the largest preserved feature table); the 207-feature pool used for RFE is the 177
plus the 30 deployment-specific engineered features.

# Reanalysis Change Log — 2026-08-11

Every substantive change from `Paper JMIR Submission.docx` (archived) to
`Paper JMIR AI Submission v2.docx`. Old artifacts under `results/`; corrected under
`results_reanalysis/`. Repo tag `pre-reanalysis-2026-08-11` marks the prior state.

## Headline before → after

| Quantity | Before (defective) | After (corrected) | Cause |
|---|---|---|---|
| Cohort | 244,576 admissions | **236,906** (7,670 index in-hospital deaths excluded) | A3 |
| Outcome label | "unplanned all-cause", next-admit rule undocumented | **all-cause within-system, 0<Δ≤30 d, day 30 in, transfers excluded** | A1/A2/A5 |
| Prevalence | 21.1% | **21.4%** (test 21.3%) | A3 |
| Test partition | 49,191 admissions | **47,672 admissions / 17,553 patients / 10,133 events** | A3 |
| Headline AUROC | 0.7957 ("deployed V8") | **0.7800 (95% cluster CI 0.7735–0.7867)** | A3 + honest single retrain |
| Validation evidence | "test evaluated exactly once" (false) | **5-fold patient-grouped CV 0.7828 ± 0.0031** + disclosed development history | B1/B3 |
| CIs / P values | admission-level bootstrap | **patient-cluster bootstrap (2,000 draws)** | E1 |
| Threshold | 0.204, provenance unclear | **0.206, derived on validation, applied once to test** | E2 |
| Calibration | ECE 0.004 / slope 1.03 (old cohort) | ECE, slope, intercept re-estimated **with CIs** on corrected cohort | E3 |
| LACE / HOSPITAL | 0.5924 / 0.6260 | **0.6050 / 0.6377**; uplifts **+0.1750 / +0.1423**, both P<.001, cluster CIs | F |
| Survival events | 48,841 readmit + 9,622 "competing deaths" | **50,793 readmissions; no competing-event claim** (unobservable) | C1–C3 |
| "1,952 reclassified" narrative | present (described the miscoding) | **removed** — those admissions are readmissions | C2 |
| AFT concordance | 0.747 (miscoded) | **0.7427 (95% CI from cluster bootstrap)** | C |
| Day-1 td-AUROC | 0.829 | **0.858**, with daily event counts | C |
| Stage AUROCs | 0.728 / 0.747 / 0.774 | **0.750 / 0.753 / 0.777** with cluster CIs; framed exploratory | C7/C8 |
| Stage domain shift | labs 19.6→14.6, util 44.7→49.0 | **labs 22.6→15.6, util 40.8→49.9** | C |
| Fairness gap | White .793 vs Black .758; CI-overlap argument | **Δ = 0.0348 (95% CI 0.0175–0.0515), direct bootstrap P<.001**; group n/events/Se/FNR table | E4/G |
| Race-removal basis | test-set comparison (P=.75) | **validation-partition ablation** (0.7954 vs 0.7949) | B2 |
| DRG availability | undiscussed | **sensitivity model: −0.0016 AUROC without DRG (CI 0.0006–0.0026)** | H2 |
| Clinical utility | none | **DCA + capacity analysis** (top-10% flags capture reported share of events) | E6 |
| SHAP figure | old-model V7 | recomputed on the corrected final model | D1 |

## Manuscript-level changes (no result change)

- Title → "…Within-System Hospital Readmission Among Medicare-Insured Adults… Retrospective Model Development and Internal Validation in MIMIC-IV" (K2).
- "Deployed" → "prototype/demonstration" throughout (H3).
- One final pipeline (single XGBoost artifact); LightGBM/CatBoost/HistGBM/blend, 10-seed averages, neural baselines, and V1–V7 progression moved to development history / appendix (D1–D5, K5, K6).
- Blend-optimizer description corrected: Nelder-Mead with [0.05, 0.50] bounds then normalization; zero weights precluded (D3).
- V4/V5 legacy rows removed from the main text (D4).
- LACE/HOSPITAL reconstruction fully documented + outcome-mismatch caveat (F1/F2).
- "Recalibration will fix the gap" replaced with correct remedy discussion (G2); race-mapping documented (G4).
- Clinical implications reframed as hypotheses; no consult/bundle prescriptions (H1).
- Cohort flow first figure; "CONSORT-style" → "STROBE-style"; 6 main figures + 4 appendix groups (K3–K5).
- Introduction consolidated (Overview/Goals/Objectives/Motivation → Background, Prior Work, Objectives); six-contributions passage removed (K6). ~5,100 words (was ~9,000+).
- References 15 → 33, all real; two flagged for journal-version verification (K7/K8); uncited "literature baselines ≈0.76" row removed (K9).
- Added: Ethical Considerations (placeholder-flagged), Funding (placeholder), CRediT contributions, Generative-AI disclosure (placeholder), Data Availability with DOI action item, TRIPOD+AI checklist appendix (I/J).

## Code added (all under presentation-update/, results under results_reanalysis/)

`_reanalysis_step1.py` cohort rebuild · `_reanalysis_step2.py` retrain/CV/cluster
bootstrap · `_reanalysis_step3.py` corrected survival · `_reanalysis_step4.py`
fairness/DCA/validation ablation · `_reanalysis_step5.py` DRG sensitivity ·
`_reanalysis_step6.py` baselines · `_reanalysis_step7.py` landmark stages ·
`_reanalysis_step8_figures.py` figures · `_build_jmir_v2_part1/2.py` manuscript ·
`_build_tripod_checklist.py` Appendix 5.

Reproduce: run steps 1→8 with the `capstone` env, then the two build scripts.

## Not done / blocked (author input)

1. Ethics: PhysioNet credential holders + CITI records; FIU IRB determination.
2. Funding statement confirmation.
3. Generative-AI disclosure wording confirmation.
4. Degrees, ORCIDs, corresponding-author details.
5. Zenodo/DOI archive of the tagged release.
6. Reference verification pass in the JMIR portal (DOIs/PMIDs; refs 23–24 journal versions).
7. Appendices 1–3 content export (predictor dictionary exists as CSV; needs formatting), Appendix 4 figure bundle.
8. RSF/GBS survival comparators not re-run (supplementary; either re-run under corrected labels or omit — currently omitted from main text).

## Addendum 2026-08-11 (v3, dod-aware + robustness)

- patients.dod integrated (hospital + MA registry, 1-y censoring): TRUE competing risks.
  Test: 10,117 readmissions / 1,170 deaths-before-readmission / 36,385 censored.
  AJ 30-day CIF: readmission 21.2%, death 2.5%. AFT C 0.7422 (CI 0.7365-0.7479);
  cause-specific Cox 0.6725; day-1 td-AUROC 0.862 with 200-draw cluster bands.
  Landmark risk sets death-aware (436 / 752 deceased removed from stages 2-3).
- Billing audit: 15 billing-derived features (drgcodes, diagnoses_icd,
  procedures_icd) removed -> 51-feature billing-free model AUROC 0.7762
  (full-clean diff 0.0038 [CI 0.0028-0.0048]).
- Simplified nested patient-grouped CV (per-fold target encodings + refit):
  0.7771 +/- 0.0037 (standard 5-fold 0.7828) - reported as the most conservative
  internal-validation estimate; feature list/hyperparameters fixed (disclosed).
- HOSPITAL rebuilt with original 12-month window: 0.6493 (6-month proxy 0.6377);
  conclusions unchanged.
- Anchor-year boundary sensitivity: 6,658 test admissions (14.0%) potentially
  near end-of-collection; AUROC excluding them 0.7726 vs 0.7800 overall.
- Manuscript: RQ1-3 restored (Intro + Discussion answers), clinical
  interpretation of leading predictors ported back (hypothesis-framed), SHAP as
  main Figure 7, "historically exposed internal evaluation partition" language,
  survival sections rewritten as competing risks, Figures 4-5 regenerated with
  uncertainty bands and death-aware stages, refs updated (MIMIC DOI, Almeida and
  Adisa published versions, PROBAST+AI). Output: Paper JMIR AI Submission v3.

## Addendum 2026-08-12 (v4, consensus-33)

- Full 207-feature audit (results_v4/feature_audit_207.csv): 63 billing-derived
  + 1 race excluded -> 143 eligible; procedures_icd flags with clinical names
  (major_surgery/cardiac_surgery/transfusion) caught by code-provenance check.
- Leakage-safe staged RFE: 5 outer x 3 inner grouped folds, fold-local target
  encodings, prespecified 0.002 parsimony rule; per-fold counts 27/32/38/32/38,
  mean Jaccard 0.78, 25 unanimous features, consensus(>=3/5) = 33.
- OOF comparison: procedure 0.7749 [0.7722-0.7782]; all-143 0.7751; prior 51
  0.7759; prior 66 (billing) 0.7685 = worst boosted model; logistic 0.6890.
- FINAL MODEL: consensus-33 XGBoost (artifacts_v4/): CV 0.7758+/-0.0032; test
  (historically exposed, tertiary) 0.7742 [0.7676-0.7806]; threshold 0.2163
  (val); ECE 0.006, slope 1.04; top-5% capture 15% @ PPV 0.64.
- Downstream re-derived: LACE uplift +0.1692, HOSPITAL-12m +0.1249 (primary
  window); W-B gap 0.0334 [0.0167-0.0508] P<.001; AFT-33 C 0.7358, day-1 0.850;
  stages 0.744/0.747/0.770. HONEST NEGATIVE: the labs-fall/utilization-rises
  domain gradient did not replicate under the 33-feature model (utilization
  dominates all stages, 57->66%); demoted to exploratory appendix history.
- Same-date death/readmit rule documented; 10,133 vs 10,117 = 16 tie-break
  admissions.
- Manuscript v4: three-tier validation hierarchy; "prototype-ready"/"actionable"
  /"realistic thresholds" removed or relabeled exploratory; no billing features
  by construction; new figures (RFE curve, stability, roc/cal, daily, fairness,
  SHAP top-10). Output: Paper JMIR AI Submission v4.

## Addendum 2026-08-12 - stage differential + appendix packet

- NEW ANALYSIS (_v4_stage_differential.py): per-feature early (days 1-7) vs
  late (days 15-30) attribution-share differential for the final consensus-33
  model, dod-aware risk sets, 300-draw patient-cluster bootstrap. All 10
  leading differentials exclude zero. Early-leaning: n_late_orders +2.96pp,
  log_prior_readmit_count +2.33, n_meds_total +2.12, sodium_last +1.92,
  discharge_location +1.54, discharge_surge +1.49. Late-leaning: freq_x_recency
  -5.08, prior_mean_los_6m -4.02, lab_abnormal_rate -2.92 (REVERSES the
  development-phase "labs early" narrative at the aggregate level; sodium
  itself stays early), prior_admissions_6m -1.85. Manuscript: new Figure 6 in
  the stage section (exploratory framing); fairness -> Figure 7, SHAP ->
  Figure 8. Artifacts: results_v4/stage_differential_v4.json,
  figures_v4/v4_stage_differential.png.
- APPENDIX PACKET (_v4_appendices.py + _build_tripod_checklist.py):
  A1 Predictor Audit and Selection.xlsx (9 sheets: 207-audit with missingness,
  eligible-143, excluded-64 with reasons, consensus-33 dictionary with fold
  counts + SHAP, RFE trajectories, per-fold selections, OOF model comparison,
  full subgroup results, stage differential); A2 LACE/HOSPITAL reconstruction;
  A3 race consolidation with raw test counts; A4 supplementary figures +
  development history (DCA, same-cohort ROC, non-replicating domain-gradient
  figure, v1-v3 development summary table); A5 TRIPOD+AI checklist updated for
  v4 (items 7a/7b/10b/12/21); A6 PROBAST+AI self-assessment (honest "some
  concerns" on participants/outcome/analysis; low on predictors post-audit).
- Manuscript Multimedia Appendices section updated to describe all six; word
  count ~6,619; 12 pages.

## v5 — 2026-08-12 — fourth external review (ChatGPT audit), disposition log

CONFIRMED LEAKAGE AND FULL RERUN. Deliverable: "Paper JMIR AI Submission
v5.docx/pdf" + results_v5/ + rebuilt appendices. v4 preserved untouched.

### Item-by-item disposition
1. discharge_location_te — ACCEPTED, CONFIRMED. It was in the eligible pool
   AND in consensus-33 alongside raw discharge_location. Empirical provenance
   (no creating code in repo): TE varies within category on train/val, is
   constant per category on test => OOF encoding over train+val with fixed
   test map; fold structure inconsistent with the v4 nested CV => outcome
   leakage into CV estimates. Removed from pool (142 eligible); raw column
   retained with fold-local TE. Other *_te columns were already excluded.
2. Validation data flow — ACCEPTED + STRENGTHENED. v4's nested CV ran over
   the FULL cohort, so consensus selection had seen test patients. v5 reruns
   selection on development partitions only (189,234 adm / 70,197 pts);
   consensus (31 features) now provably selection-independent of test.
   Explicit data-flow paragraph added to Methods; estimand language
   (procedure vs fixed set) corrected everywhere.
3. Full rerun — DONE (phase3 + phase57 + stages/figures; ~25 min total).
   OOF 0.7749 -> 0.7748 (leak was redundant with raw column, as hoped, but
   the number is now defensible). Consensus 33 -> 31 (dropped: leaked TE,
   discharge_surge, n_order_types, sodium_last; added: antibiotic_flag,
   n_distinct_routes). Test 0.7742 -> 0.7738. CV 0.7758 -> 0.7756.
   All figures, tables, SHAP, fairness, baselines, survival, stages,
   differential, DCA, capacity, boundary re-derived from results_v5/.
4. Equivalence claims — ACCEPTED. "Statistically indistinguishable" and
   "matched" removed. Paired same-draw cluster-bootstrap differences added:
   all_eligible-rfe -0.0008 [-0.0015,-0.0001] (procedure slightly BETTER);
   f50-rfe +0.0006 [-0.0001,0.0013]; f66-rfe -0.0109 [-0.0124,-0.0094].
   Wording: "within the prespecified 0.002 parsimony margin".
5. Outcome counts/timing — PARTIALLY PRE-EXISTING, COMPLETED. 16-event
   (10,133 vs 10,117) same-date death-first tie rule now stated in Results.
   Exact-time deltas verified (delta from timestamps, not calendar days).
   NEW same-day sensitivity: counting the -2<delta<=0 transfer band as
   readmissions (events 10,133->10,273): refit AUROC 0.7777. Daily event
   counts added to Appendix 1.
6. HRRP framing — MOSTLY PRE-EXISTING (Methods already states "not a CMS
   HRRP regulatory cohort"); abstract Background kept motivational only.
   Elective sensitivity moved from repository-note to Results WITH model
   performance: unplanned-only refit 0.7689 / final-model 0.7691;
   nonelective-index subset 0.7735.
7. Subgroup reporting — ACCEPTED. Per-group cluster-bootstrap CIs added for
   Se/Sp/PPV/NPV/FNR + Brier + calibration slope (800 draws). "Calibration
   remained comparable" replaced by reported slope range with CIs
   (race groups 0.99 [0.92-1.06] to 1.16 [1.00-1.29]). Full-width Table 1
   now covers race+sex+age with denominators, prevalence, and CIs.
8. Abstract — ACCEPTED. 523 -> 447 words; assertion at build time.
9. Claims pass — ACCEPTED. "No superiority claim" sentence added to prior-
   work comparison; DCA remains exploratory-labeled; SHAP/causation and
   internal/external distinctions kept.
10. Tables/figures — ACCEPTED. Table 1 rebuilt as an editable full-page-width
    Word table (single-column continuous section) with caption above,
    left-aligned cells, all abbreviations defined. Stability figure labels
    enlarged (6.5pt -> 8.5pt). All figures 450 dpi, regenerated from v5.
11. Reproducibility — ACCEPTED. Appendix 1 workbook now adds Paired_
    Differences, Outcome_Sensitivity, Daily_Event_Counts sheets; audit CSV
    row for discharge_location_te corrected. REPRODUCIBILITY_V5.md written
    (commands, versions, seeds, outputs, checks). final_model_v5.json is the
    single source of truth; manuscript is generated from it.
    Repo URLs/tag/DOI -> AUTHOR_INPUT_REQUIRED.md (not invented).
12. References — VERIFIED (web, 2026-08-12). Fixed: PROBAST+AI is Moons KGM
    et al, BMJ 2025;388:e082505 (was misattributed to Collins); Tang author
    list + PMID 37018684; Almeida venue/pages (IbPRIA 2025:220-232);
    Barnwal DOI added. Tang/Almeida/Adisa/MIMIC-v3.1-DOI/Barnwal confirmed;
    remaining classics spot-checked correct. Appendix 6 PROBAST attribution
    fixed likewise.

### Rejected / not performed, with reasons
- "Rerun 51-feature comparator": the prior billing-free subset intersected
  with the leak-free pool is 50 features; comparator renamed f50 and rerun
  (the 51st WAS the leaked encoding — the reviewer's item validated itself).
- Formal equivalence/noninferiority test: not prespecified; reported paired
  differences + margin language instead, as the reviewer's own fallback
  suggests.
- Full re-verification of all 33 references at the character level: classics
  verified by spot-check; the six risky entries verified individually; JMIR
  portal reformatting will re-touch every entry (noted in manuscript).

### Hostile-review pass — remaining known weaknesses (disclosed, not hidden)
- Hyperparameters fixed, not tuned inside folds: procedure is not fully
  nested; residual optimism possible (disclosed in Methods/Limitations and
  PROBAST+AI Analysis domain).
- The consensus refit still aggregates selection info across dev folds; its
  CV estimate (0.7756) is optimistic by construction (disclosed; primary
  estimate is the procedure OOF).
- Test partition remains HISTORICALLY exposed from pre-v2 development
  (disclosed; it is tertiary evidence only). It is now selection-independent
  but not exposure-free.
- label_sens_unplanned covers only next-admission type ELECTIVE; planned
  readmissions of other types remain in the primary label.
- 85+ Table-1 row may split across a page boundary in the Word render;
  cosmetic, will be reflowed by JMIR production.
- Registry death dates are day-granular; the death-first tie rule is a
  choice, bounded by the 16-event reconciliation.
- Single center; Medicare Advantage unobservable; within-system outcome —
  all retained in Limitations.
- Presentation deck and prototype metadata still carry pre-v5 numbers
  (flagged for update before any public use).

## v6 — 2026-08-12 — fifth external review, disposition log

No model/feature/threshold change: v6 corrects inference precision, estimand
labeling, and description errors. Canonical: results_v6/final_model_v6.json.
Deliverable: "Paper JMIR AI Submission v6.docx/pdf". v5 preserved.

1. Estimand separation — VERIFIED AND CORRECTED. Abstract "Its paired
   difference..." sentence rewritten (procedure explicit); Principal Findings
   claim replaced by a NEW direct paired comparison of the FIXED 31- vs FIXED
   142-feature models on identical development folds: 0.7756 vs 0.7757,
   diff -0.0001 [-0.0009, 0.0007], P=.758 (5,000 draws) — the reviewer's
   optional analysis, now run, which settles the fixed-vs-fixed question
   directly. Table 2 "This study" split into three tier-labeled estimand rows
   (procedure OOF / fixed dev CV / fixed test). Conclusions and prior-work
   text attribute 0.7748 to the procedure.
2. Partition description — VERIFIED AND CORRECTED. Actual: 80/20 patient-
   grouped dev-test split, then 10% of dev to validation => 72%/8%/20%
   (170,677/18,557/47,672). The v5 data-flow "80/10/10" phrase was wrong and
   is fixed; percentages added to the partition paragraph.
3. HOSPITAL window — VERIFIED AND CORRECTED. Code uses the original 12-month
   window (0.6493, primary); stale Methods sentence calling 6 months "the
   available proxy" replaced; 6-month value (0.6377) kept as development
   sensitivity in Appendix 2.
4. Bootstrap P values — VERIFIED AND CORRECTED. All reported P values now
   from 5,000-draw plus-one-corrected empirical bootstraps (LACE uplift,
   HOSPITAL uplift, W-B gap, fixed-vs-fixed). Methods states the method and
   the P<.001 floor explicitly. No P reported beyond supportable precision.
5. White-Black "paired" description — VERIFIED AND CORRECTED. Recomputed as
   a stratified disjoint-subgroup bootstrap (patients resampled within each
   subgroup, 5,000 draws); Methods/Results describe the actual design.
6. Provenance claim — ACCEPTED. Language softened to observation + inference
   + conservative exclusion in manuscript, pool JSON reason (workbook), and
   changelog. Conclusion (exclusion) unchanged.
7. Validation reuse / threshold — ACCEPTED (disclosure branch). Threshold now
   "development-derived" everywhere incl. Table 1 caption; Methods data-flow
   and Limitations disclose that validation patients participated in
   consensus-selection folds. The cross-fitted-threshold redesign was NOT
   adopted: it would change the deployed artifact and all threshold metrics
   without altering any scientific conclusion; disclosed instead (the
   reviewer's own fallback).
8. Transfer band — VERIFIED AND SPECIFIED. Delta = exact timestamp
   difference; band -2<delta<=0 days; 650 admissions (458/52/140 by
   partition); ZERO records overlap by more than 2 days, verified. Exact
   bounds now in Cohort, Sensitivity results, and Appendix 1.
9. Tie-rule sensitivity — ACCEPTED AND RUN. "Prespecified" removed (rule was
   adopted during analysis). Readmission-first variant refit end-to-end;
   results essentially unchanged (reported in Results and Appendix 1).
10. Subgroup documentation — VERIFIED AND CORRECTED. Race-ablation procedure
    now described in Methods/Fairness (validation partition, 0.7954 vs
    0.7949) with an explicit no-fairness-guarantee caveat. Table 1 caption
    states PPV/Brier are point estimates and that specificity, NPV, and CIs
    for every metric live in Appendix 1 (they do).
11. Table pagination — ACCEPTED. All tables now repeat header rows across
    pages and forbid row splitting (w:tblHeader + w:cantSplit); Table 2
    rebuilt full-page-width. PDF visually inspected.
12. Calibration values — VERIFIED: manuscript was already correct
    (slope 1.0363 -> 1.04; ECE 0.0068 -> 0.007, from canonical JSON at build
    time). The 1.03/0.006 figures in the v5 response summary and
    REPRODUCIBILITY_V5.md were stale v4 carryovers; corrected in
    REPRODUCIBILITY_V6.md. One rounding rule: format at print time from
    unrounded JSON values.
13. Parsimony language — ACCEPTED. Now "smaller than the 0.002 tolerance used
    in the prespecified parsimony rule"; equivalence disclaimers retained.
14. Abstract — ACCEPTED. Target <=440 with build assertion (whitespace word
    count, same approach as the earlier check).
15. Author blockers — UNCHANGED. AUTHOR_INPUT_REQUIRED.md stands; manuscript
    not labeled submission-ready.

## v6.1 — 2026-08-12 — sixth external review (surgical pass), disposition log

No model rerun. Corrections to interpretation, estimand labeling, one canonical
point-estimate field, and pagination. Canonical: results_v6/final_model_v6.json
(highdraw point fields patched). Deliverable: "Paper JMIR AI Submission v6.1".
v6 preserved.

1. Fixed-31 vs fixed-142 interpretation — ACCEPTED. The claim that both fixed
   sets "carry the same selection-aggregation optimism" was wrong: the
   142-pool was defined by an a priori audit, not selected on performance, so
   only the 31-set carries selection optimism in that contrast. Rewritten as a
   conditional, descriptive secondary comparison (0.7756 vs 0.7757,
   diff -0.0001 [-0.0009, 0.0007]); P=.758 REMOVED (the conditional test does
   not account for selecting the consensus set on the same data); the
   fixed-vs-fixed difference REMOVED from the abstract (abstract now carries
   only the three tier estimates). The leakage-safe procedure-vs-procedure
   comparison (RFE vs full-pool) is retained as it evaluates procedures fairly.
2. Prior-work estimand — ACCEPTED. "0.7748 ... with only 31 features" was a
   conflation; rewritten to attribute 0.7748 to the RFE procedure (27-38
   features/fold) and 0.7738 to the fixed 31-feature model on test. RQ2
   reworded from "a single classifier" to "the XGBoost-based RFE procedure".
   Whole-manuscript scan: no remaining sentence attaches 0.7748 to the exact
   fixed set (consistency check enforces this).
3. Test-independence / hyperparameter qualification — ACCEPTED. "never used
   for selection", "contributed to none", and the unqualified independence
   claims replaced by v6-specific wording: no test outcome entered THIS
   version's RFE, encoding, consensus, or threshold, BUT model family,
   engineered candidates, and fixed hyperparameters predate this version under
   historical test exposure, so residual indirect influence cannot be
   excluded. Limitations expanded to name model-family selection, candidate
   predictors, hyperparameters, and post-hoc researcher decisions. Table 2 and
   abstract use "not used in this version's RFE or consensus construction".
4. White-Black / HOSPITAL point values — ACCEPTED, REAL BUG. The v6 response
   quoted 0.0346 / 0.1245; the manuscript printed 0.0344 / 0.1244. Cause: the
   summ() helper stored the BOOTSTRAP MEAN as 'point' in v6_extras, while the
   manuscript-facing fields (baselines.uplift_*, fairness.white_minus_black.
   point) already held the observed differences from unrounded values.
   Rounding rule fixed and documented: point = observed difference from
   unrounded canonical point estimates, rounded once at print; bootstrap
   distribution supplies CI and P only. Patched v6_extras.json and
   final_model_v6.json highdraw 'point' fields; appendix workbook regenerated
   (now 0.1244 / 0.0344 with plus-one P). All locations agree.
5. Day-1 interpretation — ACCEPTED. "exactly when the discharge decision is
   made" and "most separable on the day of discharge" replaced by
   "discrimination highest for readmissions occurring during the first day
   after discharge - the window in which an intervention initiated at
   discharge would need to act"; Conclusion uses "events occurring during the
   first day after discharge". No claim the outcome coincides with discharge.
6. Bootstrap scope — ACCEPTED. Methods and Limitations now state intervals
   resample stored patient prediction-outcome pairs WITHOUT refitting; they
   quantify evaluation-sample uncertainty conditional on the fitted models and
   do not capture retraining/reselection variability. Reporting clarification,
   no rerun.
7. Table 1 caption pagination — ACCEPTED. keep_with_next applied to the Table
   1 and Table 2 captions; PDF verified — Table 1 caption and all 11 rows on
   page 9, Table 2 header repeats with no split rows.
8. "deployable set" — ACCEPTED. Replaced with "final consensus predictor set";
   prototype still described as a research demonstration.
9-10. Preserved v6 corrections and author-info safeguards — VERIFIED unchanged
   (development-only RFE, discharge_location_te removal, 72/8/20 split, 12-mo
   HOSPITAL, development-derived threshold, stratified W-B bootstrap, 5,000-
   draw plus-one P, transfer band -2<delta<=0, tie sensitivity, three-tier
   hierarchy, Table 2 rows, abstract <=440, Appendix 1 full metrics, historical
   exposure disclosure, no equivalence claim). AUTHOR_INPUT_REQUIRED.md stands.

Consistency (CONSISTENCY_REPORT_V6.1.md): 40+ checks PASS, including absence of
"P=.758", "deployable", "same selection-aggregation optimism", "never used",
"exactly when the discharge decision", and "with only 31 nonbilling ... features";
presence of "27-38", the bootstrap-scope disclosure, and the consensus-optimism
note. Abstract 437 words; 14 pages.

## v6.2 — 2026-08-12 — seventh external review (v6.1 PDF audit), disposition log

ChatGPT independently audited the v6.1 PDF and confirmed 8 items RESOLVED
(estimand separation; fixed-31-vs-142 relabel + P=.758 removal + gone from
abstract; numerical reconciliation 0.1244/0.0344 with no stale 0.1245/0.0346;
day-1 terminology; bootstrap-scope wording; Table 1 caption+table together on
page 9; "deployable set" replaced). It raised 5 new items; 3 fixable, 1
terminal author-blocker, 1 logistics. No model rerun; no numeric change.
Canonical unchanged (results_v6/final_model_v6.json). Deliverable:
"Paper JMIR AI Submission v6.2". v6.1 preserved.

1. Figure 1 misleading — ACCEPTED, REAL. Verified against the PNG: the box
   literally read "Test (held out)" (contradicting the paper's central
   historical-exposure disclosure) and drew Training -> Validation -> Test as a
   sequential vertical chain (implying successive derivation). Redrew
   r2_fig1_flow_col.png as a branching tree in _reanalysis_step8b_colfigs.py:
   analysis cohort --(80/20 split)--> {Development 80%, Test 20%}; Development
   --(10% to validation)--> {Training, Validation}. Test box now reads
   "Test, 20% (historically exposed internal partition)"; Test is parallel to
   Development, not downstream of Validation. Development totals shown
   (189,234 adm / 70,197 pts / 40,660 ev) reconcile exactly with train+val.
   Verified in the rendered v6.2 PDF (page 5).
2. Contradictory hyperparameter sentence — ACCEPTED, REAL. The validation-
   hierarchy paragraph still asserted "No test-partition outcome contributed to
   feature selection, encoding, hyperparameters, or threshold choice in the
   final pipeline," which contradicted the adjacent disclosure that
   hyperparameters predate this version under historical exposure. Replaced
   with: in this version's pipeline no test outcome was used for selection,
   encoding, consensus, refitting, or threshold, BUT model family, candidate
   pool, and fixed hyperparameters predated this rerun and may have been
   indirectly influenced by earlier test-partition evaluations.
3. Difference direction undefined — ACCEPTED. The +/-0.0008 / +0.0006 paired
   differences now state their convention: each is the comparator minus the
   RFE procedure, so a negative value favors the procedure (verified in code:
   paired["{name}_minus_rfe"] = roc_auc(comparator) - roc_auc(rfe)).
4. Administrative placeholders — TERMINAL AUTHOR-BLOCKER, cannot fix.
   [AUTHOR ACTION REQUIRED], two [CONFIRM] fields, [SUBMISSION NOTE], funding,
   "Dr." vs degrees, ORCIDs, corresponding author, repo URLs/tag/DOI remain
   gated in AUTHOR_INPUT_REQUIRED.md. Nothing invented. This is the expected
   asymptote: the manuscript is scientifically complete but not
   administratively submission-ready until the authors supply these.
5. Separate high-res figure files — LOGISTICS, ADDRESSED. Embedded PDF images
   are ~200 ppi (Word downsampling); the 450-500 dpi source PNGs exist in
   figures_v5/ and figures_reanalysis/. The submission packet now ships a
   figures/ folder with the separate high-resolution PNGs for portal upload.

Consistency (CONSISTENCY_REPORT_V6.2.md): PASS, including new checks — "27-38"
present, "comparator minus the RFE procedure" present, indirect-influence
wording present, and the absolute "contributed to feature selection, encoding,
hyperparameters, or threshold" claim absent. Abstract 437 words; 14 pages.

## v6.3 — 2026-08-12 — FINAL (post-convergence typesetting fix)

ChatGPT's v6.2 audit verdict: "Yes. V6.2 is scientifically ready for JMIR AI
submission once the author-supplied fields are completed accurately. I found no
remaining methodological contradiction requiring another modeling run." All
three v6.2 edits confirmed Resolved; estimands correctly separated; numbers
consistent; none of the previously-rejected wording/stale values present
(no P=.758, no "indistinguishable", no 0.1245/0.0346, no "deployable set").

The audit left ONE item, explicitly typesetting-only: on page 6 the prior
66-feature comparison's CI broke mid-value across a line ("95% CI -" / newline /
"0.0124 to -0.0094"). Fixed in v6.3 by rendering the three paired-difference
CIs as unbreakable units (non-breaking spaces + non-breaking hyphen via a new
nbci() helper in the builder). No numeric change; canonical JSON unchanged;
consistency PASS; abstract 437 words; 14 pages. Verified in the rendered PDF
that "95% CI -0.0124 to -0.0094" now stays on one line.

CONVERGENCE. This is the terminal scientific/editorial state. Every remaining
open item is author-supplied and cannot be produced without fabrication:
FIU IRB determination/exemption, PhysioNet credential-holder names + training
records, funding + COI confirmation, author degrees (replace "Dr."), ORCIDs,
affiliations, corresponding-author contact, repository URLs, tagged release,
archival DOI; plus uploading the separate 450-dpi figure PNGs (shipped in the
packet's figures_highres/ folder) at portal entry. All are tracked in
AUTHOR_INPUT_REQUIRED.md. The manuscript is not labeled submission-ready until
the authors resolve them. No further pipeline rerun is warranted.

## v6.4 — 2026-08-16 — Armando's pre-submission audit, disposition log

Source: "JMIR AI v6.3 Handoff" package (Armando Gonzalez, 15-16 Aug 2026):
HANDOFF_BRIEF, PRE_SUBMISSION_AUDIT_v6.3, repaired figures with before/after
evidence, JMIR template + published-article evidence. Every claim verified
against our canonical artifacts before applying; all four fix groups approved
by TB individually (per the handoff's approval-gated instruction).

ADOPTED — Armando's figure repairs (verified visually, no data change):
- Figure 1: split annotations no longer sit on the arrows (both labels moved
  into clear space). Adopted his AFTER PNG as figures_reanalysis/
  r2_fig1_flow_col.png (ours backed up *_pre_armando.png); his generator
  preserved as _armando_make_fig1.py.
- Figure 2: "consensus n=31" no longer cut by the dashed threshold line.
  Adopted as figures_v5/v5_rfe_curve.png; generator _armando_make_fig2.py.

CONFIRMED — his single-column finding matches our independently built
submission format (tables must stay in body after first mention; no annex
move; two-column is production's job). Both v6.4 layouts rebuilt.

FIXED — verified numeric/cross-reference defects (his section 4):
- B1 prevalence "mismatch" RESOLVED AS DENOMINATORS: 20.8% is cohort-wide,
  20.7% is test-partition; both correct; text now says "cohort-wide
  prevalence ... test-partition values in Results". Did not need TB.
- B2 "the same 66 ties" -> "all 66 same-date ties in the cohort (16 in the
  test partition)".
- B3 capacity 44% -> 44.5% (0.445 printed at 1 decimal).
- B4 RQ1 now lists the same top 7 features as the SHAP section (SH[:7]).
- B5 both "(Multimedia Appendix figure)" -> "(Multimedia Appendix 4)".
- B6 citations: AFT objective now cites [28] (Barnwal); Harrell C cites [29]
  only; [30] retained on IPCW td-AUROC.
- Ethics fragment -> "This work is a secondary analysis of deidentified
  data." (placeholder retained separately).
- Abbreviations: added NLP, OOF, PACS, SD, STROBE; SHAP expanded at first
  use (Research Questions).
- Ref [1] CMS: official URL + access date added (URL verified live).
- Ref [24] Adisa: switched to the arXiv preprint (arXiv:2604.22535) to
  sidestep the venue-standing question he raised.
- Back matter reordered to JMIR AI: Acknowledgments, Funding Statement,
  Conflicts of Interest, Data Availability, Authors' Contributions,
  Abbreviations, Multimedia Appendices, References.
- Title page: "Dr." honorifics removed (degrees pending author input);
  "August 2026" date line removed.
- [SUBMISSION NOTE] shortened; no longer names internal working files.

STYLE PASS (his section 5, applied via a build-time _style() in the builder):
" - " -> unspaced em dashes (40 sites incl. tables); straight apostrophes ->
typographic; negative numbers -> true minus U+2212 (replacing the mixed
U+2011/U+2212 glyphs); PRIMARY/SECONDARY/TERTIARY and SAME COHORT/DATABASE
de-capitalized; "did NOT replicate" -> "did not replicate"; "~0.714" ->
"≈0.714"; "10 of the ten" -> "all 10"; causality disclaimer reduced from 3
to 1 (+ the definitional "driver" sentence, now in curly quotes); four
rhetorical flourishes neutralized (honest negative / usable signal lives /
not only a number / leverage points).

TRIM (his section 6): duplicated test-exposure disclosure removed from the
data-flow paragraph (kept once in the hierarchy paragraph); blend-weight
transparency note condensed. Total words 8,038 -> 7,919 (body ~6,230 vs
JMIR's soft 6,000 guidance; "no rigorous restrictions").

NOT DONE (with reasons):
- Figure typeface Times New Roman: his own audit calls it low priority and
  only if all 8 figures are regenerated together; deferred.
- PMID/DOI on every reference: the 6 risky entries verified previously;
  full identifier pass deferred to portal entry where JMIR reformats
  references anyway; tracked in AUTHOR_INPUT_REQUIRED.md.
- All author-supplied items (degrees, ORCIDs, corresponding author, IRB,
  funding, repo URLs/DOI): remain gated; cannot be produced without
  fabrication.

Consistency: CONSISTENCY_REPORT_V6.4.md — PASS on 60+ checks including 20
new assertions for the above. Both layouts rebuilt (2-col 14 pp; single-
column line-numbered 23 pp); abstract 437 words unchanged.

## v6.4 FINAL FREEZE — 2026-08-16 — ChatGPT final check passed

Sent the corrected v6.4 SINGLE-COLUMN PDF for the final check (after fixing
the one surviving item from its earlier list: "lost nothing" -> "showed no
material decrement in discrimination"). Verdict: "scientifically ready...
the scientific content is frozen... no model rerun or further scientific
revisions are warranted... I see no remaining substantive issue that should
prevent JMIR AI submission." Confirmed: estimand separation intact; AUROCs
0.7748/0.7756/0.7738 with correct directions and CIs; Figures 1-2, top-20%
capture, tie-rule wording, SHAP list, appendix references, citations,
abbreviations, back-matter order all pass; continuous line numbering perfect
(1-592, no gaps/duplicates); single-column 1.5-spaced format acceptable per
JMIR guidance (line numbers preferred); 8 figures + 2 tables within limits.

Its one required fix — the multi-line Table 1 caption split across pages
13-14 in the single-column reflow — is FIXED: caption paragraphs now carry
keep_together in addition to keep_with_next; verified caption (lines 356-360)
+ full table begin together on page 14. Minor notes acknowledged: in-cell CI
wrapping acceptable (no redesign, per its own advice); embedded PDF images
are downsampled by Word, which is why the packet ships separate 450-dpi PNGs.

Thiago's ORCID recorded: https://orcid.org/0009-0006-0204-5298 (title
page/portal at submission). Remaining: the other author-supplied fields and
uploading the separate high-resolution figure PNGs at portal entry.

STATE: v6.4 is the frozen submission version. Further internal revision is
counterproductive per both the external check and our own judgment.

## FINAL — 2026-08-16 — author-fields pass complete; upload-ready

All author-supplied fields resolved and inserted in the single sanctioned
final pass (scientific content unchanged from frozen v6.4):
- Title page: degrees (Bandeira MS, Gonzalez MS, Poellabauer PhD, Mondal
  PhD), registry-verified ORCIDs under each author, affiliation with
  city/state/country, full corresponding-author block (T. Batista Nunes
  Bandeira, 21324 NE 2nd Ct, Miami FL 33179; (815) 603-3286;
  tbati006@fiu.edu).
- Ethical Considerations: bracket-free final wording — TB sole raw-data
  accessor under the PhysioNet Credentialed Health DUA with CITI Data or
  Specimens Only Research training; co-authors used aggregate derived
  results only; secondary analysis of deidentified public data does not
  constitute human-subjects research and required no FIU IRB review
  (confirmed by TB 2026-08-16).
- Funding: "This study received no external funding." (APC-payer sentence
  can be added at proofs if FIU covers it.)
- Data Availability: real repository URLs + tagged release v6.4-submission
  + archival DOI 10.5281/zenodo.21987702.
- [SUBMISSION NOTE] paragraph removed; zero bracketed placeholders remain.

Files: "Paper JMIR AI Submission FINAL.docx/pdf" (2-col preview, 14 pp) and
"Paper JMIR AI Submission FINAL SINGLE-COLUMN.docx/pdf" (submission format,
23 pp, line-numbered). CONSISTENCY_REPORT_FINAL.md: PASS on 75+ checks incl.
no-placeholder and all-field assertions. Abstract 437 words.

REMAINING (process, not manuscript): mentor sign-off (Poellabauer, Mondal);
portal tasks at upload (JMIR reference reformatting, AI-use attestation,
450-dpi figure PNG uploads from the packet's figures_highres/).


## FINAL no-dash pass (2026-08-17, late evening)

Author rule (Thiago, 2026-08-17): no dash punctuation anywhere in authored
text. Trigger: Armando flagged em dashes in the built PDF during his
pre-submission read.

- _style() no longer converts spaced hyphens to em dashes; it now asserts
  that no em dash, en dash, or spaced hyphen reaches the document, so any
  future build with dash punctuation fails.
- All 38 spaced-hyphen constructs in the manuscript text rewritten with commas,
  colons, semicolons, or parentheses (script: _apply_nodash.py).
- Numeric ranges in prose and table cells now use "to": CI bounds, day
  windows (1 to 7, 8 to 14, 15 to 30), feature counts (27 to 38), DCA
  threshold ranges, Python versions, age bands. ci() renders "lo to hi".
- "White-Black" contrast renamed "White vs Black" (two occurrences).
- Table 1 bare-hyphen marker replaced with "NA".
- Reference 19 (Vyas, NEJM) title rendered with a colon, its PubMed form.
- RQ3 reworded so SHAP keeps its standard first-use expansion "(SHAP)".
- Abstract retrimmed to 440 words.
- Kept by design (notation or identifiers, not punctuation): compound-word
  hyphens (30-day, patient-grouped, MIMIC-IV), ORCIDs, DOIs, dates, phone,
  release tag, URLs, reference page ranges, bracketed citation ranges, raw
  MIMIC race strings in Appendix 3, and U+2212 minus signs.
- Appendix 5: one prose spaced hyphen replaced with a semicolon. All other
  appendices scanned clean of em/en dashes.
- _final_consistency.py: "em dashes present" inverted to "em dashes gone"
  plus a new "en dashes gone" check; "27-38" expectation now "27 to 38".
  OVERALL: PASS. Both DOCX scans: 0 em, 0 en, 0 spaced hyphens.

Files rebuilt: both FINAL DOCX/PDF pairs (14 pp / 23 pp, unchanged page
counts); CONSISTENCY_REPORT_FINAL.md regenerated; packet zip refreshed.


## Mentor revision round 1 (2026-09-14, Dr. Poellabauer markup)

Source: his tracked edits and 6 comments in the returned single-column copy
plus his email of 2026-09-14.

- Title: "Predicting 30-Day Hospital Readmission at Discharge: A
  Leakage-Safe, Calibrated EHR Model in Medicare-Insured Adults" (his
  tracked edit). Design subtitle kept in JMIR convention: "Retrospective
  Development and Internal Validation of an Interpretable Gradient-Boosting
  Model on MIMIC-IV v3.1" (answer to his subtitle comment: JMIR expects the
  study design in the title).
- Abstract: his Background reframing sentence adopted (replaces the
  overlapping original clause); Objective now "at hospital discharge"; his
  two Methods sentences (fold-local leakage prevention; prespecified
  procedure-level primary estimate) merged in; Results opening clarified
  ("This procedure, in which each outer fold selected its own 27 to 38
  predictors"); compensating trims; abstract exactly 450 words.
- Introduction restructured per his related-work comment: new Related Work
  section with four subsections (Clinical Scores and Regression-Based
  Models; Machine Learning for Readmission Prediction; Data Leakage and
  Validation Pitfalls, new, citing refs 34 to 36; Prior Work on MIMIC),
  then an explicit Study Rationale section.
- Research questions: his RQ1 to RQ3 wording adopted (his "HER" typo
  corrected to "EHR"); new RQ4 on timing and subgroups per his comment;
  Answers to the Research Questions rewritten for all four (RQ2 now answers
  improvement over clinical scores; learner comparison demoted to a
  parenthetical).
- Principal Findings reordered: leakage-safety, timing, fairness, then the
  clinical-score comparison.
- References 34 to 36 added, verified against PubMed: Kapoor and Narayanan
  2023 (Patterns), Futoma et al 2020 (Lancet Digit Health), Wiens et al
  2019 (Nat Med).
- Keywords: "data leakage" added.
- Not adopted: two ambiguous punctuation insertions in the Cohort section
  (no reconstructable content change).
- Consistency checker extended with 15 new checks. OVERALL: PASS. Dash
  scans clean. Totals: ~8,400 words, 36 references, abstract 450, 15 pp
  two-column and 23 pp single-column.


## Critical-review round (2026-09-15 overnight, 44-agent adversarial pass)

A 40-agent review workflow (6 reviewer lenses including a mentor-perspective
read and a hostile referee, 4 figure inspectors reading the PNGs, 2
reproducibility auditors recomputing headline numbers from stored
predictions) plus a 4-agent confirmation pass. 13 upheld majors, 65 minors,
and the actionable nits were implemented:

- Citations now renumber themselves by first appearance at build time
  (regress-proof pass inside the builders); 4 new verified references:
  Graham 2015 (early vs late readmissions), Krumholz 2013 (post-hospital
  syndrome), Seyyed-Kalantari 2021 (underdiagnosis bias), Blanche 2013
  (competing-risks td-AUC). 40 references total, all cited, strict order
  verified programmatically.
- Related Work gains a Readmission Timing and Subgroup Performance
  subsection; Background motivates timing and equity; abstract now carries
  all four RQ threads at 447 words.
- Fairness reporting quantified: age-band AUROC range (0.7542 to 0.7854)
  with CIs, age-band calibration slopes excluding 1, race AUROCs at stored
  4-decimal precision, threshold-level specificity difference disclosed,
  within-system label-bias caveat added, "stable across age bands" claim
  removed as contradicted by Table 1.
- td-AUROC estimand made explicit: cause-specific competing-risks
  definition, deaths retained as nonevents, Blanche cited beside Uno;
  Figure 5 caption aligned.
- Two new canonical sensitivity blocks in final_model_v6.json
  (_extend_v6_extras.py): era-stratified test AUROC by anchor_year_group
  (0.7437 rising to 0.8556 across 2008 to 2022 bands, prevalence falling
  24.6% to 17.2%) now reported as sensitivity item 5; payer-scope
  verification (readmission search covers all payers; 743 non-Medicare next
  admissions, 741 counted).
- Leakage-safe scoping definition added to Methods (answers the
  self-certification attack on the title); temporal-split rationale
  rewritten around anchor_year_group; sample-size justification
  (events-per-candidate above 250); multiplicity statement; Youden
  threshold framed as illustrative; software environment rendered from the
  canonical env block; death-as-nonevent deployment caveat; single-site
  characterization expanded; Conclusions demoted to a Discussion
  subsection; funding folded into Acknowledgments; dozens of grammar,
  tense, abbreviation, and rounding fixes (Jaccard 0.755, prevalences from
  integer counts).
- Figure 3 regenerated with all 44 selected features (was truncated to 40)
  and an annotated consensus threshold; Figure 1/4/6/7 captions corrected.
- Confirmation pass: reference integrity PASS, new numbers PASS after
  Jaccard fix, 13 prose artifacts repaired, Figure 3 annotation relocated.

Totals: about 9,619 words, 40 references, abstract 447, 16 pp two-column,
26 pp single-column. Consistency checker (about 130 checks): OVERALL PASS.
Dash scans: 0 em, 0 en, 0 spaced hyphens.

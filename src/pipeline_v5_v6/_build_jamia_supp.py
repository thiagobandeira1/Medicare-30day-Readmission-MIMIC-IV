# -*- coding: utf-8 -*-
"""Build the JAMIA Supplementary Material DOCX.

Carries the methodological and results detail moved out of the 4,000-word
main text. Bracket citations use the ORIGINAL (pre-renumber) numbering of
the build system and are rewritten at build time with the same old-to-new
mapping the main JAMIA manuscript produced (_jamia_ref_mapping.json), so
supplement citations match the main manuscript's reference list exactly.
"""
import json
import re as _re2
from _build_jmir_finalsc_part1 import *  # noqa
from _build_jmir_finalsc_part1 import _style  # underscore name: explicit

HERE = Path(__file__).resolve().parent
_MAP = json.loads((HERE / "_jamia_ref_mapping.json").read_text())
MAPPING = {int(k): v for k, v in _MAP["mapping"].items()}

_PAT = _re2.compile(r"\[([0-9][0-9,\-]*)\]")


def _remap(text):
    """Rewrite bracket citations old->new per the main manuscript mapping."""
    def parse(group):
        out = []
        for part in group.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out

    def rewrite(mm):
        new = sorted(MAPPING[n] for n in parse(mm.group(1)))
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

    return _PAT.sub(rewrite, text)


def SP(t, bold_prefix=None):
    return P(_remap(t), bold_prefix=bold_prefix)


# restyle: supplement, 11 pt, 1.5 spacing
S = doc.styles
S["Normal"].font.size = Pt(11)

au = M["auroc"]
p = CT["partitions"]
st1, st2, st3 = FM["stages"]
THRV = FM["threshold_validation"]
VH = FM["validation_hierarchy"]
OOF = VH["primary_oof_procedure_dev_only"]
CCV = VH["secondary_consensus_cv_dev_only"]
BSE = FM["baselines"]
N_ELIG = POOL5["n_eligible"]
SENSD = FM["sensitivity"]
EXX = FM["v6_extras"]

t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Supplementary Material")
r.bold = True; r.font.size = Pt(14)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Predicting 30-Day Hospital Readmission at Discharge: A "
              "Leakage-Safe, Calibrated Electronic Health Record Model in "
              "Medicare-Insured Adults")
r.italic = True; r.font.size = Pt(11.5)
P("Reference numbers in this document refer to the reference list of the "
  "main manuscript. Supplementary Appendices 1 to 6 are separate files "
  "enumerated in the main manuscript's Supplementary Material section.")

# ====================================================== SUPPLEMENTARY METHODS
H1("Supplementary Methods")
H2("S1. Cohort and Outcome Definition (Full Detail)")
SP(f"The outcome was the first subsequent hospitalization of the same "
   f"patient in the same health system beginning more than 0 and up to "
   f"30 days (day 30 inclusive) after index discharge, of any cause, "
   f"and irrespective of how that subsequent admission ended; a "
   f"readmission followed by in-hospital death is a readmission. The "
   f"interval Δ is the exact timestamp difference between the next "
   f"admission's admittime and the index dischtime. Next admissions "
   f"beginning at, or up to 2 days before, the index discharge "
   f"timestamp (−2<Δ≤0 days; "
   f"n={CT['same_day_next_admit(delta<=0)']:,}; no record in this "
   f"cohort overlaps by more than 2 days) were treated as inter-unit "
   f"transfers or administrative continuations, not readmissions. In "
   f"the binary frame, admissions followed by death without "
   f"readmission within 30 days are nonevents; the competing-risks "
   f"reformulation (S6) addresses the interpretation. Because MIMIC-IV "
   f"is single-center, readmissions to other hospitals are "
   f"unobservable and the outcome is explicitly within-system. Whether "
   f"the final admissions in each patient's record have 30 complete "
   f"days of observation cannot be verified under date shifting; the "
   f"boundary sensitivity analysis (S9, item 4) bounds its influence.")
H2("S2. Partition Construction and Complete Data Flow")
SP(f"Admissions were split 80/20 at the patient level (GroupShuffleSplit "
   f"on subject_id) into development and test data; 10% of the "
   f"development portion was then assigned, again patient-grouped, to a "
   f"validation partition used for early stopping, threshold derivation, "
   f"and ablation decisions. The final proportions are approximately 72% "
   f"training, 8% validation, and 20% test; the validation partition "
   f"comprised {p['val']['admissions']:,} admissions from "
   f"{p['val']['patients']:,} patients with {p['val']['events']:,} "
   f"events. No formal sample-size "
   f"calculation was performed: the cohort was fixed by the available "
   f"data, and with {p['train']['events']:,} training-partition events "
   f"against {N_ELIG} candidate predictors the events-per-candidate "
   f"ratio exceeds 250, far above common minima for prediction-model "
   f"development.")
SP("The complete data flow is: (1) patient-grouped 80/20 "
   "development-test split on subject_id, with 10% of the development "
   "portion assigned to validation; (2) the 5 outer selection folds are "
   "drawn from all development (training plus validation) patients; (3) "
   "within each outer-training fold, 3 grouped inner folds drive staged "
   "elimination, with target encodings refit from inner-training "
   "patients at every step; (4) each fold's selected model is refit on "
   "its full outer-training fold (encodings refit) and scored once on "
   "its outer-evaluation fold, yielding the primary out-of-fold "
   "estimate; (5) the consensus set is defined from the 5 per-fold "
   "selections; (6) the final model is refit on the training partition "
   "with training-partition encodings, its operating threshold is "
   "derived on the validation partition, and it is evaluated once on "
   "the test partition. Because the selection folds span all "
   "development patients, the validation partition participated in "
   "consensus-feature selection; the threshold is therefore "
   "development-derived, not derived on data independent of model "
   "selection.")
H2("S3. Excluded Target Encoding: Audit Detail")
SP("One precomputed target encoding of discharge destination "
   "(discharge_location_te) coexists in the legacy feature tables with "
   "the raw column and was excluded as outcome-derived. Its original "
   "transformation code is no longer available; an empirical audit "
   "found within-category value patterns inconsistent with a single "
   "training-only category map (values vary within category on the "
   "development partitions but are constant within category on test), "
   "so its fold provenance could not be aligned with the present "
   "validation design and the variable was excluded conservatively. "
   "The raw discharge_location column is retained and, like every "
   "categorical, is target-encoded strictly fold-locally.")
H2("S4. Model Family Exploration and Development History")
SP("During development, LightGBM [11], CatBoost, and scikit-learn "
   "HistGradientBoosting were trained as comparators and a "
   "Nelder-Mead-optimized convex blend of the four families was "
   "evaluated; the blend improved nothing over the best single family "
   "and the four families spanned only 0.0035 AUROC, so a single model "
   "was retained. CatBoost received categorical variables natively "
   "rather than target encoded. Neural baselines (multilayer "
   "perceptrons, LSTM/GRU hybrids, FT-Transformer [27], and a stacking "
   "ensemble) were explored on an earlier feature version with single "
   "configurations and are reported in the repository as development "
   "history; boosting is therefore described as the better performer "
   "in our experiments, consistent with published tabular benchmarks "
   "[9,12], rather than as a definitive architecture comparison. "
   "XGBoost was selected over the marginally higher-scoring LightGBM "
   "for consistency with the accelerated failure time survival "
   "objective and exact TreeSHAP tooling [14]. Earlier 10-seed "
   "averaged results constitute seed ensembles; the final model here "
   "is one seed and one artifact.")
H2("S5. Clinical-Score Reconstruction")
SP("LACE [3]: length-of-stay points per the original bands; 3 acuity "
   "points unless the admission was elective; Charlson-derived "
   "comorbidity points; and true prior 180-day emergency-department "
   "visits (edregtime) capped at 4. HOSPITAL [4]: hemoglobin <12 g/dL, "
   "oncology service (OMED), sodium <135 mmol/L, any procedure, "
   "nonelective index type, prior admissions in the original 12-month "
   "window (counted from each patient's admission history, strictly "
   "pre-discharge information), and length of stay of at least 5 days. "
   f"An earlier development-phase variant used a 6-month prior-admission "
   f"proxy (AUROC {RB['C_hospital_12m']['hospital_6m_proxy_auroc']:.4f} "
   f"vs {RB['C_hospital_12m']['hospital_12m_auroc']:.4f} for the "
   f"original 12-month window) and is retained as a sensitivity result "
   f"in Supplementary Appendix 2. Full mappings "
   "appear in Supplementary Appendix 2. LACE was derived for death or "
   "unplanned readmission and HOSPITAL for potentially avoidable "
   "readmission, so the same-cohort comparisons quantify performance "
   "of the scores on this study's all-cause within-system outcome.")
H2("S6. Time-to-Event Analysis Detail")
SP("The competing-risks event process was built from patients.dod, "
   "which records death from hospital records and the Massachusetts "
   "State Registry with deaths beyond 1 year after last discharge "
   "censored by deidentification. Events: readmission (event 1) at the "
   "first subsequent admission within 30 days, irrespective of how "
   "that admission ended; death before readmission within 30 days "
   "(event 2, competing); administrative censoring at day 30 "
   "otherwise. Cumulative incidence used Aalen-Johansen estimators. "
   "The prognostic models are cause-specific for readmission: XGBoost "
   "with the accelerated failure time objective [28] (a single-event "
   "survival model evaluated under censoring, not itself a "
   "competing-risk model) and penalized cause-specific Cox regression "
   "as the linear reference (reference-model results are retained as "
   "development history in the project repository), with competing "
   "deaths censored at their "
   "death time in the cause-specific fits. Time-dependent AUROC used "
   "cumulative/dynamic estimation with inverse probability of "
   "censoring weighting [30,40]; patients who died before the "
   "evaluation day remain in the comparison set as nonevents, a "
   "cause-specific competing-risks definition [40], with "
   "patient-cluster bootstrap uncertainty bands and daily event counts "
   "reported in Supplementary Appendix 1. The three landmark models "
   "(days 1 to 7, 8 to 14, 15 to 30) were each trained only on "
   "patients still at risk when the stage opens; patients already "
   "readmitted or deceased are removed from the risk set, not counted "
   f"as nonevents (risk sets: {st1['events']:,} events among "
   f"{st1['at_risk']:,} at risk for days 1 to 7; {st2['events']:,} "
   f"among {st2['at_risk']:,} for days 8 to 14; {st3['events']:,} "
   f"among {st3['at_risk']:,} for days 15 to 30); stage windows differ "
   "in length and prevalence by "
   "design, so stage results are descriptive. Landmark models were "
   "trained on training-partition patients, evaluated on the test "
   "partition, and reuse discharge-time features without "
   "post-discharge updating. Stage-specific TreeSHAP attribution "
   "shares are exploratory.")
H2("S7. Bootstrap Detail and Subgroup Consolidation")
SP("Resample counts by analysis: 1,000 for the test metric panel; 800 "
   "for subgroup CIs; 500 for the primary out-of-fold estimate and "
   "model comparisons; 5,000 for every contrast with a reported P "
   "value. P values are empirical two-sided bootstrap probabilities "
   "with a plus-one correction, so the smallest reportable value at "
   "5,000 resamples is P<.001. All intervals were computed by "
   "resampling patients and their stored prediction-outcome pairs; "
   "feature selection and model fitting were not repeated within "
   "bootstrap samples, so the intervals quantify evaluation-sample "
   "uncertainty conditional on the fitted models and do not capture "
   "the variability of retraining or reselection. The Youden threshold "
   "is illustrative for the metric panel and is not proposed for "
   "deployment; the capacity-constrained view is the operational "
   "lens. MIMIC race/ethnicity strings were consolidated by substring "
   "mapping (WHITE* to White; BLACK* to Black; HISPANIC*/LATINO* to "
   "Hispanic/Latino; ASIAN* to Asian; all others, declined, or unknown "
   "to Other/Unknown); raw category counts appear in Supplementary "
   "Appendix 3. During development, a race-encoding ablation was "
   "performed on the validation partition: the historical model was "
   "refit with and without the precomputed race encoding and "
   "validation AUROC compared (0.7954 with vs 0.7949 without); the "
   "encoding contributed no measurable signal and race-derived "
   "features were excluded from all final configurations "
   "(Supplementary Appendix 4). This result describes one encoding's "
   "marginal contribution and does not imply that excluding race "
   "guarantees equal subgroup performance.")
H2("S8. Demonstration Prototype Safeguards")
SP("The public demonstration prototype accepts only synthetic or "
   "manually entered values, displays a research-only, "
   "no-real-patient-data notice, and stores no submitted data. It is a "
   "research demonstration, not a deployed clinical system.")

# ====================================================== SUPPLEMENTARY RESULTS
H1("Supplementary Results")
H2("S9. Outcome-Definition and Cohort Sensitivity Analyses (Full Detail)")
_sa = SENSD["sameday_as_readmission"]
_su = SENSD["unplanned_only"]
_sn = SENSD["nonelective_index_only"]
SP(f"(1) Counting the transfer/continuation band (next admission "
   f"beginning at, or up to 2 days before, the index discharge "
   f"timestamp: {EXX['transfer_band']['n_band_by_partition']['test']} "
   f"test admissions) as readmissions raised test events from "
   f"{_sa['test_events_primary']:,} to {_sa['test_events_alt']:,} "
   f"({_sa['test_events_alt']/p['test']['admissions']*100:.1f}% "
   f"prevalence); a model refit under this label reached AUROC "
   f"{_sa['test_auroc_refit']:.4f}, and the final model ranked the "
   f"alternative label at "
   f"{_sa['test_auroc_final_model_vs_alt_label']:.4f}. (2) Excluding "
   f"next admissions typed ELECTIVE (a proxy for planned readmissions; "
   f"{_su['test_events_alt']:,} test events, "
   f"{_su['test_prevalence_alt']*100:.1f}% prevalence) gave a refit "
   f"AUROC of {_su['test_auroc_refit']:.4f}; the final model ranked "
   f"this outcome at "
   f"{_su['test_auroc_final_model_vs_alt_label']:.4f}. (3) Restricting "
   f"evaluation to nonelective index admissions "
   f"({_sn['test_admissions']:,} admissions, "
   f"{_sn['test_prevalence']*100:.1f}% prevalence) gave AUROC "
   f"{_sn['test_auroc']:.4f}. (4) Excluding the {BD5['n_boundary']:,} "
   f"test admissions ({BD5['share']*100:.1f}%) whose latest possible "
   f"calendar year reached the 2022 end of data collection changed "
   f"AUROC to {BD5['auroc_safe']:.4f}. (5) The era-stratified analysis "
   f"appears in Table S1. The cohort-wide prevalence under the "
   f"ELECTIVE-excluded outcome is "
   f"{CT['label_sens_prevalence']*100:.1f}% vs "
   f"{CT['label_v2_prevalence']*100:.1f}% for the primary outcome.")
_era = EXX["era_stratified_test"]["bands"]
_rows = [["Era band (anchor_year_group)", "Test admissions", "Prevalence",
          "AUROC"]]
for b in _era:
    _rows.append([b["band"].replace(" - ", " to "), f"{b['n']:,}",
                  f"{100*b['events']/b['n']:.1f}%", f"{b['auroc']:.4f}"])
caption("Table S1. Era-stratified test discrimination of the final model "
        "by the patient-level anchor_year_group band. Bands are "
        "descriptive: patients are assigned by deidentified anchor era, "
        "band sizes differ, and case mix shifts across eras.",
        keep_with_next=True)
table(_rows, widths=[2.2, 1.2, 1.0, 1.0])
H2("S10. Payer-Scope Check on Outcome Ascertainment")
_ps = EXX["payer_scope"]
SP(f"Readmissions were ascertained from all subsequent hospitalizations "
   f"irrespective of the insurance recorded on the subsequent "
   f"admission: {_ps['n_30d_next_nonmedicare']:,} 30-day next "
   f"admissions carried non-Medicare insurance, of which "
   f"{_ps['n_counted_as_readmission']:,} were counted as readmissions "
   f"under the primary outcome (the remainder fall in the "
   f"transfer/continuation band). Restricting the outcome to "
   f"Medicare-insured next admissions would have misclassified these "
   f"as nonevents.")
H2("S11. Fixed-Set Paired Cross-Validation Comparison")
_fx = EXX["fixed31_vs_fixed142_paired_cv"]
SP(f"Conditional on treating both predictor sets as fixed, grouped "
   f"cross-validation on identical development folds produced AUROCs "
   f"of {_fx['auroc_fixed31_cv']:.4f} for the {NF}-feature consensus "
   f"model and {_fx['auroc_fixed142_cv']:.4f} for the complete "
   f"{N_ELIG}-feature model, a paired difference of "
   f"{_fx['paired_diff_31_minus_142']:+.4f} (95% CI "
   f"{_fx['ci95'][0]:.4f} to {_fx['ci95'][1]:.4f}). This secondary, "
   f"descriptive comparison is potentially optimistic for the "
   f"{NF}-feature model, whose consensus set was selected using these "
   f"same development data, whereas the eligible pool was defined by "
   f"an a priori audit; the complete selection-procedure estimate "
   f"remains the primary evidence. The consensus model's development "
   f"cross-validation folds were "
   f"{', '.join(f'{x:.4f}' for x in CCV['folds'])}.")
H2("S12. Tie-Rule Sensitivity for Same-Date Readmission and Death")
_tr = EXX["tie_rule_sensitivity"]
SP(f"Registry death dates carry day-level granularity: when a "
   f"readmission and a death share the same recorded date, the primary "
   f"analysis resolved the tie as death first, a documented rule "
   f"adopted during analysis; the true ordering cannot be established "
   f"from day-level dates. The competing-risks frame therefore counts "
   f"{p['test']['events']-SV['test_events']} fewer readmissions than "
   f"the binary frame ({SV['test_events']:,} vs "
   f"{p['test']['events']:,}). Resolving all {_tr['n_ties']} same-date "
   f"ties in the cohort ({_tr['n_ties_test']} in the test partition) "
   f"readmission-first left every estimate essentially unchanged "
   f"(Harrell C {_tr['death_first_primary']['harrell_c']:.4f} vs "
   f"{_tr['readmission_first']['harrell_c']:.4f}; 30-day readmission "
   f"cumulative incidence "
   f"{_tr['death_first_primary']['aj_cif30_readmission']*100:.1f}% vs "
   f"{_tr['readmission_first']['aj_cif30_readmission']*100:.1f}%).")
H2("S13. Stage-Specific Attribution (Exploratory)")
_sig = [r_ for r_ in SD5 if r_["excludes_zero"]]
_early = [r_ for r_ in _sig if r_["diff_pp"] > 0]
_late = [r_ for r_ in _sig if r_["diff_pp"] < 0]
_elist = "; ".join(f"{flabel(r_['feature'])} ({r_['diff_pp']:+.1f} "
                   f"percentage points)" for r_ in _early)
_llist = "; ".join(f"{flabel(r_['feature'])} ({r_['diff_pp']:.1f} "
                   f"percentage points)" for r_ in _late)
SP(f"A negative finding: the domain-composition gradient observed in "
   f"richer development feature sets (laboratory attribution falling "
   f"and utilization rising across the window) did not replicate in "
   f"the parsimonious final model, whose attribution is dominated by "
   f"prior utilization at every stage "
   f"({ST5[0]['shares']['Prior utilization']:.0f}% to "
   f"{ST5[2]['shares']['Prior utilization']:.0f}%) with no monotonic "
   f"laboratory trend (Supplementary Figure S2, panel A). The earlier "
   f"finding is therefore feature-set-dependent and is reported as "
   f"exploratory development history in Supplementary Appendix 4.")
SP(f"At the individual-feature level, early and late returns do differ "
   f"in character (Supplementary Figure S2, panel B; "
   f"{'all 10' if len(_sig) == 10 else f'{len(_sig)} of the 10'} "
   f"leading differentials have 95% cluster-bootstrap CIs excluding "
   f"zero; the 10 largest observed differentials were selected before "
   f"interval inspection, so these intervals are descriptive). Signals "
   f"disproportionately important for a return within days 1 to 7 are "
   f"predominantly states of the index discharge itself: {_elist}. "
   f"Signals disproportionately important for returns in days 15 to "
   f"30 are predominantly the longer-horizon utilization aggregates: "
   f"{_llist}. These are exploratory, descriptive contrasts in model "
   f"reliance, offered as hypotheses about differently timed "
   f"follow-up rather than causal claims.")

# ====================================================== SUPPLEMENTARY FIGURES
H1("Supplementary Figures")
figure(FIG5 / "v5_rfe_curve.png",
       "Supplementary Figure S1. Leakage-safe staged recursive feature "
       "elimination on development data: mean inner-validation AUROC "
       f"versus retained features, per outer fold; dashed line marks the "
       f"{NF}-feature consensus.", width=5.6)
figure(FIG5 / "v5_stage_combined.png",
       f"Supplementary Figure S2. Attribution across post-discharge "
       f"windows, final {NF}-feature model. (A) Share of total model "
       "attribution by clinical domain. (B) Features whose attribution "
       "share differs most between the earliest (days 1 to 7) and latest "
       "(days 15 to 30) windows; positive values indicate greater early "
       "importance. Error bars are 95% cluster-bootstrap CIs.", width=5.2)

DST = PUB / "Paper JAMIA Supplementary Material.docx"
doc.save(str(DST))
print(f"saved {DST} ({DST.stat().st_size:,} bytes)")

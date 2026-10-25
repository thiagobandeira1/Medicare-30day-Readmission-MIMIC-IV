# -*- coding: utf-8 -*-
"""Build the JAMIA Supplementary Material, v9 (episode rule for continuations).
All values are read from results_v7. References are named in text rather
than numbered, so the supplement does not depend on the main reference list."""
import json
from _v13_core import *  # noqa
from _v13_core import _style  # noqa
from _v7_labels import label, SHORT as LBL

S = doc.styles
S["Normal"].font.size = Pt(11)
OUT7OLD = REPO / "results_v7"
LK = json.loads((OUT7 / "leak_test_v13.json").read_text())
TC = json.loads((OUT7 / "trend_causal_check_v13.json").read_text())
TA = json.loads((OUT7OLD / "timestamp_audit_v8.json").read_text())
HV = json.loads((OUT7OLD / "history_variants_v8.json").read_text())
PRV = json.loads((OUT7OLD / "prior_readmission_variants_v8.json").read_text())
IA = json.loads((OUT7 / "index_stay_audit_v13.json").read_text())
PDR = json.loads((OUT7OLD / "postdischarge_rows_v8.json").read_text())
DCK = json.loads((OUT7OLD / "definition_checks_v8.json").read_text())
TFS = json.loads((OUT7OLD / "trend_formula_search_v8.json").read_text())
SC = json.loads((OUT7 / "subgroup_calibration_v13.json").read_text())
EVL = json.loads((OUT7 / "evaluation_v13.json").read_text())
TB = json.loads((OUT7 / "table_rebuild_v13.json").read_text())
BO = json.loads((OUT7 / "baselines_oof_v13.json").read_text())
LABS = [k for k, v in TB.items() if isinstance(v, dict) and "new_missing" in v and v["new_missing"] > 0 and k not in ("hr_std", "sbp_std", "gcs_min")]
MEDF = [k for k, v in TB.items() if isinstance(v, dict) and v.get("old_value_where_new_missing_mode") not in (None, 0.0)]
ZEROF = [k for k, v in TB.items() if isinstance(v, dict) and v.get("old_value_where_new_missing_mode") == 0.0]
CONST = IA["constant_within_patient_share"]
OLD = json.loads((REPO / "results_v6" / "final_model_v6.json").read_text())
p = CT["partitions"]
NF = FM["n_features"]
CB, OH, TM = RV["continuation_band"], RV["one_hour_rule"], RV["timing"]
ER = json.loads((OUT7 / "episode_end_rule_v13.json").read_text())
TO = json.loads((OUT7OLD / "continuation_time_origin_v10.json").read_text())
FLOW = ER["flow"]
OH_OLD = OH
TM = dict(TM, landmark_one_hour_rule=ER["landmark_episode_rule"])
PU, RU, G, ABL = (RV["prior_utilization"], RV["race_by_utilization"],
                  RV["comparators"], RV["leakage_ablation"])
LT = LK["los_trend_180d"]
FC = FM["final_feature_set"]
SENSD = FM["sensitivity"]


def pct(x, d=1):
    return f"{x*100:.{d}f}%"


t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Supplementary Material"); r.bold = True; r.font.size = Pt(14)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Future-Information Leakage in 30-Day Readmission "
              "Prediction: An Empirical Audit and a Calibrated Electronic "
              "Health Record Model in Medicare-Insured Adults"); r.italic = True
P("Supplementary Appendices 1 to 6 are separate files listed in the main "
  "manuscript. All values below are generated from the released results "
  "files (results_v13 and the retained v8 to v10 audit files in results_v7).")

H1("Supplementary Methods")
H2("S1. Cohort, Outcome, and Continuation Band")
P(f"The outcome was the first subsequent hospitalization of the same patient "
  f"in the same health system beginning more than 0 and up to 30 days (day 30 "
  f"inclusive) after index discharge, of any cause and irrespective of how it "
  f"ended or of the insurance recorded on it. The interval Δ is the exact "
  f"difference between the next admission time and the index discharge time. "
  f"Next admissions with −2<Δ≤0 days "
  f"(n={CT['same_day_next_admit(delta<=0)']:,}; none overlaps by more than 2 "
  f"days) were treated as transfers. Admissions followed by death without "
  f"readmission are nonevents in the binary frame. Because MIMIC-IV is "
  f"single-center, the outcome is within-system.")
_r = CB["rate_per_hour"]
P(f"Continuation band. Among all {CB['events_all']:,} outcome events, "
  f"{CB['events_in_band']:,} ({pct(CB['share_of_events'])}; "
  f"{CB['events_in_band_test']:,} in the test partition) began within 1 hour "
  f"of discharge, a rate of {_r['0_1h']['per_hour']:,.0f} per hour, against "
  f"{_r['1_2h']['per_hour']:.0f} per hour at 1 to 2 hours, "
  f"{_r['2_4h']['per_hour']:.0f} at 2 to 4 hours, {_r['6_12h']['per_hour']:.0f} "
  f"at 6 to 12 hours, and {_r['24_72h']['per_hour']:.0f} at 1 to 3 days. "
  f"Of these early records, {pct(CB['next_internal_psych_transfer'])} were "
  f"internal transfers to or from psychiatry, {pct(CB['next_from_snf'])} "
  f"arrivals from skilled nursing, and "
  f"{pct(CB['next_ed_registration_before_index_discharge'])} carried an "
  f"emergency department registration that began before the index discharge; "
  f"the index discharge destination was missing for "
  f"{pct(CB['index_discharge_location_missing'])}, hospice for "
  f"{pct(CB['index_discharge_hospice'])}, and a psychiatric facility for "
  f"{pct(CB['index_discharge_psych_facility'])}. By contrast, records "
  f"beginning 1 to 12 hours after discharge arrived predominantly through "
  f"the emergency department after a typical emergency stay, consistent with "
  f"genuine early returns. These features make most early records probable "
  f"administrative continuations, but they do not establish that each one "
  f"is; and the 1-hour window was chosen after inspecting the data. The "
  f"primary definition was therefore retained. The 1-hour sensitivity rule "
  f"treats each continuation as part of the index stay. In the raw "
  f"timestamps the median gap between the index discharge and the "
  f"continuation was {TO['gap_minutes_quantiles'][2]:.0f} minute (90th "
  f"percentile {TO['gap_minutes_quantiles'][4]:.0f} minutes) and the "
  f"continuation lasted a median of {TO['continuation_los_days_quantiles'][2]:.1f} "
  f"days ({pct(TO['continuation_extends_past_day30_share'])} lasted beyond day "
  f"30), so timing the window from the first discharge would count hospital "
  f"days as time at risk. The rule therefore merges every consecutive "
  f"admission beginning within 1 hour of the previous discharge into one "
  f"stay ({ER['episodes_with_more_than_one_continuation']} stays had more "
  f"than one link), starts the 30-day window at the discharge of the last "
  f"link, excludes stays ending in death "
  f"({ER['episodes_ending_in_death_excluded_all']}; "
  f"{ER['episodes_ending_in_death_excluded_test']} in test), as index "
  f"in-hospital deaths are excluded from the cohort, and counts the next "
  f"admission if it begins within that window. Features and predictions "
  f"remain those of the first discharge. Of the "
  f"{ER['continuations_all']:,} remaining stays, "
  f"{ER['continuations_with_later_return_all']} ({ER['continuations_with_later_return_test']} "
  f"in test) were followed by a genuine return and remain events; simply "
  f"relabeling every continuation as a nonevent would misclassify these "
  f"returns (frozen-model AUROC {OH_OLD['test_auroc_final_model']:.4f} under "
  f"that cruder rule). Under the episode rule ({ER['test_events_episode_rule']:,} "
  f"test events, {pct(ER['test_prevalence_episode_rule'])}), the frozen final "
  f"model scores {ER['test_auroc_final_model']:.4f} (95% CI "
  f"{ER['test_auroc_final_model_ci95'][0]:.4f} to "
  f"{ER['test_auroc_final_model_ci95'][1]:.4f}) and a model refit to this "
  f"outcome {ER['test_auroc_refit']:.4f} (95% CI "
  f"{ER['test_auroc_refit_ci95'][0]:.4f} to {ER['test_auroc_refit_ci95'][1]:.4f}); "
  f"part of the frozen model's primary AUROC therefore reflects the "
  f"continuations.")
_fl = [["Step", "All partitions", "Test partition"],
       ["Index admissions, primary outcome", f"{FLOW['all_admissions_primary']:,}", f"{FLOW['test_admissions_primary']:,}"],
       ["Events, primary outcome", f"{CT['label_v2_events']:,}", f"{FLOW['test_events_primary']:,}"],
       ["Rows that are continuation links (begin within 1 h of the previous discharge), merged into the preceding episode", f"{FLOW['all_rows_that_are_continuation_links_merged']:,}", f"{FLOW['test_rows_that_are_continuation_links_merged']:,}"],
       ["Merged episodes ending in in-hospital death, excluded", f"{FLOW['all_episodes_ending_in_death_excluded']:,}", f"{FLOW['test_episodes_ending_in_death_excluded']:,}"],
       ["Rows in both categories above (a link whose own chain ends in death), counted once", f"{FLOW['all_rows_in_both_categories']:,}", f"{FLOW['test_rows_in_both_categories']:,}"],
       ["Episodes retained (each contributes once, scored at its first discharge)", f"{FLOW['all_episodes_retained']:,}", f"{FLOW['test_episodes_retained']:,}"],
       ["Events, episode rule", f"{FLOW['all_events_episode_rule']:,}", f"{FLOW['test_events_episode_rule']:,}"]]
caption("Table S4. Episode flow under the 1-hour continuation sensitivity rule. Retained episodes equal index admissions minus merged links minus death exclusions plus the rows counted in both categories.",
        keep_with_next=True)
table(_fl, widths=[3.6, 1.3, 1.3])
H2("S2. Partitions and Data Flow")
P(f"Admissions were split 80/20 at the patient level (GroupShuffleSplit on "
  f"subject_id); 10% of the development portion formed the validation "
  f"partition ({p['val']['admissions']:,} admissions, "
  f"{p['val']['patients']:,} patients, {p['val']['events']:,} events). The 5 "
  f"outer selection folds span all development patients; within each "
  f"outer-training fold, 3 grouped inner folds drive staged elimination with "
  f"encodings refit from inner-training patients at every step; each fold's "
  f"selected model is refit on its outer-training fold and scored once on its "
  f"outer-evaluation fold (primary estimate); the consensus set is formed "
  f"from the 5 selections; and the final model is refit on the training "
  f"partition, thresholded on the validation partition, and scored once on "
  f"test. The validation partition therefore participated in consensus "
  f"selection.")
H2("S3. Empirical Future-Information Tests and Timestamp Audit")
P(f"Among {LK['first_ever_admissions']:,} first observed index admissions (no "
  f"earlier admission in MIMIC-IV), "
  f"{pct(LK['share_with_later_admission'])} had a later admission. For each "
  f"of the 207 candidates we compared the share of rows at the feature's "
  f"modal value between patients without and with a later admission. For a "
  f"feature built only from earlier encounters this test has a sharp "
  f"prediction (a single fixed value for first observed admissions); for other "
  f"features, such as index-stay measurements or age, differences are "
  f"expected and the test is not informative. A "
  f"prior-only feature should show no systematic difference; a feature "
  f"fixed at one value for every patient without a later admission, but not "
  f"for others, encodes the future. Only the 180-day length-of-stay trend "
  f"met this criterion: it was nonzero for {pct(LT['nonzero_share_first_ever'])} "
  f"of first observed admissions, every one of which had a later admission "
  f"({pct(LT['nonzero_share_without_any_later_admission'])} of patients "
  f"without one had a nonzero value; AUROC "
  f"{LT['auroc_nonzero_for_later_admission']:.3f} for a later admission). "
  f"Thirty-day readmission was {pct(LT['readmit30_if_nonzero'])} when the "
  f"trend was nonzero and {pct(LT['readmit30_if_zero'])} when it was zero. "
  f"The trend was also identical across all admissions of every one of the "
  f"{TA['trend_patients_multi_admission']:,} cohort patients with more than one "
  f"admission ({pct(TA['trend_constant_within_patient_share'])}) and zero for "
  f"every patient with a single admission, so it was a per-patient summary "
  f"of the whole longitudinal record. The original extraction code was not "
  f"retained, and no candidate reconstruction reproduced it exactly (the "
  f"closest, the slope of whole-day length of stay across all of a "
  f"patient's Medicare admissions, matched {pct(TFS['slope_idx_losd'])} of "
  f"patients, including single-admission patients for whom every candidate "
  f"is zero). "
  f"The trend and the two interaction terms built from it were removed. "
  f"To test whether the trend concept itself carried signal, we rebuilt it "
  f"causally: the least-squares slope of length of stay across the "
  f"patient's stays admitted within 180 days before, and discharged "
  f"before, the index admission, followed by the index stay (0 when no "
  f"such stay exists), with the same two interactions. The rebuilt trend "
  f"was zero for all first observed admissions and, among the "
  f"{pct(TC['share_with_prior_stay_180d'])} of admissions with an earlier "
  f"stay in the window, was unrelated to later admission (AUROC "
  f"{TC['los_trend_180d_prior_auroc_later_admission_given_prior']:.3f}). "
  f"Added to the final features (both sets refit in the same run) it "
  f"changed development cross-validation "
  f"AUROC from {TC['devcv_auroc_v7_47'][0]:.4f} to "
  f"{TC['devcv_auroc_v7_47_plus_trend3'][0]:.4f} and validation-partition "
  f"AUROC from {TC['val_auroc_v7_47']:.4f} to "
  f"{TC['val_auroc_v7_47_plus_trend3']:.4f}, so it was not added. The "
  f"largest gap among the {NF} final features was "
  f"{LK['scan_max_gap_final_features']:.3f} (age).")
_hr = TA["history_recompute"]
P(f"Within-patient constancy. Among the {IA['n_multi_admission_patients']:,} "
  f"patients with more than one admission, the removed trend was constant "
  f"within {pct(CONST['los_trend_180d'])} of patients. Sex is constant by "
  f"construction (recorded once per patient) and is a legitimate predictor; "
  f"among the other final features the highest share was "
  f"{pct(max(v for k, v in CONST.items() if k not in ('los_trend_180d', 'gender')))} "
  f"for a binary unit flag. Constancy is a screen, not evidence of leakage; "
  f"it is combined with the first-admission test and recomputation from raw "
  f"timestamps.")
P(f"Recomputation of history features from raw admission timestamps (all "
  f"{HV['n_cohort_admissions']:,} cohort admissions, using only admissions "
  f"before the index admission): prior admissions of any insurance "
  f"{pct(_hr['prior_admissions_all']['match_share'])}; admissions discharged "
  f"in the prior 180 days {pct(HV['prior_admissions_6m']['p6_disch180'])}; "
  f"mean whole-day length of stay of those admissions "
  f"{pct(HV['prior_mean_los_6m']['mlos_floor_disch180'])}; any prior 180-day "
  f"admission with an emergency department registration "
  f"{pct(_hr['prior_any_ed_6m']['match_share'])}; days since the previous "
  f"Medicare-insured discharge "
  f"{pct(HV['time_since_last_discharge']['nonfill_match_cohort'], 2)} (a fixed "
  f"value whenever no earlier Medicare-insured admission exists, "
  f"{pct(HV['time_since_last_discharge']['fill_when_no_prior_cohort'])} "
  f"agreement); and earlier Medicare-insured admissions in the prior 365 "
  f"days followed by a 30-day readmission "
  f"{pct(PRV['lookback_variants']['admit_365'])}. Definitions that would "
  f"use later admissions matched worse (for prior readmissions, counting "
  f"the index admission's own outcome {pct(PRV['cum_incl'])}, the whole "
  f"record {pct(PRV['tot'])}). Several first-pass definitions did not "
  f"match; each was resolved by a causal variant, and Supplementary Table S3 "
  f"now gives the verified definitions.")
_om = TA["omr_audit"]
P(f"Outpatient-record features (candidates, not in the final model) "
  f"reproduced exactly as the last value dated on or before the admission "
  f"time: body-mass index "
  f"{pct(_om['bmi_last']['on_or_before_admission_time']['exact_match_any_value_on_last_date'])}, "
  f"diastolic pressure "
  f"{pct(_om['bp_diastolic_outpatient']['on_or_before_admission_time']['exact_match_any_value_on_last_date'])}. "
  f"Agreement with the last value dated on or before discharge was lower "
  f"({pct(LK['outpatient_features_exact_match']['bmi_last']['on_or_before_discharge'])} "
  f"and {pct(LK['outpatient_features_exact_match']['bp_diastolic_outpatient']['on_or_before_discharge'])}) "
  f"because {pct(_om['bmi_last']['admissions_with_value_charted_during_stay'])} "
  f"and {pct(_om['bp_diastolic_outpatient']['admissions_with_value_charted_during_stay'])} "
  f"of admissions had a value charted during the stay, which the features "
  f"correctly exclude.")
_rs = IA["raw_spot_check"]
P(f"Index-stay features (random sample of {IA['raw_spot_check_n']:,} "
  f"admissions): numeric laboratory results per admission reproduced "
  f"{pct(DCK['n_labs_total_eq_numeric_results_in_hadm'])}, the abnormal "
  f"laboratory rate {pct(DCK['lab_abnormal_rate_eq_abnormal_rows_over_numeric_results'])}, "
  f"distinct laboratory test types "
  f"{pct(_rs['n_lab_item_types']['all_rows_of_hadm'])}, and distinct order "
  f"types {pct(_rs['n_order_types']['all_rows_of_hadm'])} (spot check made "
  f"before the cutoff rebuild described below). The feature stored as "
  f"orders_last_6h equals late order rate multiplied by late orders "
  f"({pct(DCK['orders_last_6h_eq_rate_x_late_orders'])}), not orders in the "
  f"final 6 hours as its name suggests. Rows recorded under an admission "
  f"but timestamped after discharge were rare "
  f"({pct(PDR['labevents']['row_share_after_discharge'], 2)} of laboratory "
  f"and {pct(PDR['poe']['row_share_after_discharge'], 2)} of order rows in a "
  f"sample of {PDR['n_sample']:,} admissions; median "
  f"{PDR['poe']['median_hours_after_discharge']:.1f} hours after discharge "
  f"for orders) and unrelated to the outcome (AUROC "
  f"{PDR['labevents']['auroc_any_post_for_outcome']:.2f} and "
  f"{PDR['poe']['auroc_any_post_for_outcome']:.2f}). Laboratory value "
  f"features (last hemoglobin, albumin, sodium, and white cell count; "
  f"maximum bilirubin) were rebuilt for every admission from the raw "
  f"laboratory table, keeping a result only if both its collection time and "
  f"its storage time (storetime, when the result became available) fell on "
  f"or before discharge; {TB['lab_rows_stored_after_discharge']['n_stored_after']:,} "
  f"of {TB['lab_rows_stored_after_discharge']['n_rows']:,} results "
  f"({pct(TB['lab_rows_stored_after_discharge']['share'], 2)}) collected "
  f"before discharge were stored afterwards and are excluded. The "
  f"laboratory counts (results, abnormal rate, test types, critical results) "
  f"and the prescription count, which the source pipeline had taken from "
  f"every row of the admission, were rebuilt for every admission under the "
  f"same cutoffs; "
  f"{pct(TB['v12_cutoff_enforced']['n_labs_total']['share_changed'])} of "
  f"laboratory result counts and "
  f"{pct(TB['v12_cutoff_enforced']['n_meds_total']['share_changed'])} of "
  f"medication counts changed (mean results "
  f"{TB['v12_cutoff_enforced']['n_labs_total']['mean_old']:.1f} to "
  f"{TB['v12_cutoff_enforced']['n_labs_total']['mean_new']:.1f}; Supplementary "
  f"Appendix 1, Table_Rebuild). Two features that were exact copies of "
  f"others under different names (orders_first_24h, identical to orders "
  f"per day; nonenglish_x_prior_admits, identical to prior admissions in 6 "
  f"months) were removed from the eligible pool.")
_d = TB["v13_documentation_time_enforced"]
_ch = [k for k in ("hr_std", "sbp_std", "gcs_min", "ventilator_flag") if k in _d]
_em = [k for k in ("n_emar_details", "n_distinct_routes", "iv_admin_rate", "antibiotic_flag", "n_med_administrations") if k in _d]
_rx = [k for k in ("n_meds_total", "distinct_drugs", "n_discharge_drugs") if k in _d]
P(f"Documentation time. MIMIC-IV records when an event occurred (charttime, "
  f"starttime) and when it was documented (storetime for chart and "
  f"medication-administration records; pharmacy entertime for prescriptions). "
  f"The source pipeline filtered these tables on event time only. Every "
  f"chart, medication-administration, and prescription feature was therefore "
  f"rebuilt from the raw tables requiring both times to fall on or before "
  f"discharge. Of records already filtered by event time, "
  f"{_d['_emar_rows']['stored_after_discharge']:,} of "
  f"{_d['_emar_rows']['charted_before_discharge']:,} medication-administration "
  f"records ({pct(_d['_emar_rows']['stored_after_discharge'] / _d['_emar_rows']['charted_before_discharge'], 3)}), "
  f"{_d['_chart_rows']['stored_after_discharge']:,} of "
  f"{_d['_chart_rows']['charted_before_discharge']:,} chart records for the "
  f"selected items ({pct(_d['_chart_rows']['stored_after_discharge'] / _d['_chart_rows']['charted_before_discharge'], 3)}), "
  f"and {_d['_prescription_rows']['entered_after_discharge']:,} of "
  f"{_d['_prescription_rows']['started_before_discharge']:,} prescriptions "
  f"({pct(_d['_prescription_rows']['entered_after_discharge'] / _d['_prescription_rows']['started_before_discharge'], 3)}) "
  f"were documented after discharge and are now excluded. Before the cutoff "
  f"the recomputed features reproduced the stored values for "
  f"{pct(min(_d[k]['definition_match_without_documentation_cutoff'] for k in _ch + _em + _rx))} to "
  f"{pct(max(_d[k]['definition_match_without_documentation_cutoff'] for k in _ch + _em + _rx), 0)} of admissions, "
  f"confirming the definitions; the cutoff changed "
  f"{pct(max(_d[k]['share_changed_by_documentation_cutoff'] for k in _ch + _em + _rx), 2)} of values at most "
  f"(per feature in Supplementary Appendix 1, Cutoff_Rebuild). The discharge "
  f"interval was also recomputed from any earlier admission "
  f"({pct(_d['time_since_last_discharge']['share_changed'])} of values changed). "
  f"Order and transfer features were extracted with a discharge-time filter "
  f"in the source pipeline (ordertime, transfertime, or intime on or before "
  f"discharge) and were spot-checked, not fully recomputed, against raw "
  f"tables; the provider order entry table carries no separate documentation "
  f"time.")
H2("S4. Missing-Data Handling")
P(f"The source feature table, inherited from the capstone pipeline, had no "
  f"missing values: continuous measurements that were never taken had been "
  f"imputed with the median computed over all {CT['original_admissions']:,} "
  f"admissions, including the test partition, and some had been zero-filled. "
  f"Comparing stored columns with raw tables identified {len(MEDF)} "
  f"median-imputed features ("
  + ", ".join(label(k).lower() if k in LBL else k for k in MEDF)
  + f") and {len(ZEROF)} zero-filled features ("
  + ", ".join(label(k).lower() if k in LBL else k for k in ZEROF)
  + f"). All were rebuilt from raw MIMIC-IV tables with missing values "
  f"preserved; true missingness ranged from "
  f"{pct(min(TB[k]['new_missing'] for k in MEDF + ZEROF))} to "
  f"{pct(max(TB[k]['new_missing'] for k in MEDF + ZEROF))}, and where a "
  f"measurement existed the old and new values agreed for "
  f"{pct(min(TB[k]['agree_where_both_present'] for k in LABS))} or more of "
  f"admissions (the remainder reflect the storage-time rule). Intensive "
  f"care hours and the longest unit stay are zero when no intensive care "
  f"unit stay occurred. Counts are zero when no event occurred. Time since "
  f"last discharge is the interval from the previous discharge of any "
  f"insurance and is missing only for first observed admissions "
  f"({pct(TB['v13_documentation_time_enforced']['time_since_last_discharge']['share_missing_new'])} "
  f"of admissions; the source table had used the previous Medicare-insured "
  f"discharge and the global median, 80.22 days, when none existed); "
  f"recency-weighted frequency is zero for first observed admissions. Interaction terms are missing "
  f"when a component is missing. Percentages in this paragraph that come "
  f"from the rebuild report use all {CT['original_admissions']:,} Medicare "
  f"admissions as denominator; the audit workbook reports missingness on the "
  f"analysis cohort of {CT['cohort_v2_admissions']:,} admissions, which is "
  f"why its values differ slightly. The tree learner handles missing values "
  f"natively. The logistic comparator used fold-local median imputation "
  f"with standardization. The global imputation leaked only the "
  f"distribution of predictors across partitions, not outcomes; the "
  f"complete pipeline was rerun on the rebuilt table.")
H2("S5. Clinical-Score Reconstruction and Refits")
P(f"Adapted LACE (van Walraven et al.): length-of-stay points by the original "
  f"bands; 3 acuity points unless elective; Charlson comorbidity points from "
  f"index ICD diagnoses; and prior 180-day emergency department visits capped "
  f"at 4, counting only visits that led to admission (emergency department "
  f"visits without admission are not in the MIMIC-IV core tables, so the "
  f"score is adapted and may understate the E component). HOSPITAL (Donze et al.): hemoglobin below 12 g/dL, oncology service, "
  f"sodium below 135 mmol/L, any ICD-coded procedure, nonelective admission, "
  f"prior admissions in 12 months, and length of stay of 5 days or more. "
  f"Recomputed scores matched the stored test scores for "
  f"{pct(G['reconstruction_match']['lace'])} (LACE) and "
  f"{pct(G['reconstruction_match']['hospital'])} (HOSPITAL) of admissions. "
  f"Refits used logistic regression on indicator-coded components, fitted on "
  f"the training partition: LACE all components AUROC "
  f"{G['lace_refit']['auroc']:.4f}, without the coded comorbidity item "
  f"{G['lace_refit_no_coded']['auroc']:.4f}; HOSPITAL all components "
  f"{G['hospital_refit']['auroc']:.4f}, without the coded procedure item "
  f"{G['hospital_refit_no_coded']['auroc']:.4f}; both scores' "
  f"discharge-available components together "
  f"{G['lace_hospital_refit_no_coded']['auroc']:.4f}. Uplifts of the final "
  f"model over these refits were {G['lace_refit']['uplift']:.4f}, "
  f"{G['lace_refit_no_coded']['uplift']:.4f}, {G['hospital_refit']['uplift']:.4f}, "
  f"{G['hospital_refit_no_coded']['uplift']:.4f}, and "
  f"{G['lace_hospital_refit_no_coded']['uplift']:.4f} (all 95% CIs excluding "
  f"zero; descriptive, no P values computed; Supplementary Appendix 1). On "
  f"the development partitions, out of fold, adapted LACE scored "
  f"{BO['oof_auroc_lace']:.4f} and HOSPITAL {BO['oof_auroc_hospital12']:.4f} "
  f"against {BO['oof_auroc_rfe']:.4f} for the selection procedure (paired "
  f"uplifts {BO['oof_uplift_lace']:.4f}, 95% CI "
  f"{BO['oof_uplift_lace_ci95'][0]:.4f} to {BO['oof_uplift_lace_ci95'][1]:.4f}, "
  f"and {BO['oof_uplift_hospital12']:.4f}, 95% CI "
  f"{BO['oof_uplift_hospital12_ci95'][0]:.4f} to "
  f"{BO['oof_uplift_hospital12_ci95'][1]:.4f}). Top-decile event capture out "
  f"of fold was {pct(BO['top10_capture_oof']['model']['captured'], 0)} for the "
  f"procedure, {pct(BO['top10_capture_oof']['lace']['captured'], 0)} for LACE, "
  f"and {pct(BO['top10_capture_oof']['hospital12']['captured'], 0)} for "
  f"HOSPITAL.")
H2("S6. Time-to-Event Detail")
_sv = FM["survival"]
P(f"Events were readmission (event 1), death before readmission within 30 "
  f"days (event 2; registry deaths beyond 1 year after last discharge are "
  f"censored by deidentification), and administrative censoring at day 30. "
  f"The AFT model used early stopping on the validation partition (best "
  f"iteration {_sv['aft_best_iteration']}). Daily cumulative/dynamic AUROC "
  f"keeps deaths before day t among the controls (Blanche definition 2); "
  f"with no censoring before day 30, the estimate is the empirical AUROC. "
  f"Treating deaths as censored at death with inverse probability weighting "
  f"gave nearly identical values (day 1 "
  f"{_sv['daily_tdauc_deaths_censored_ipcw'][0]:.4f} vs "
  f"{_sv['daily_tdauc'][0]:.4f}). Landmark models were trained on "
  f"training-partition patients still at risk when each stage opens and "
  f"evaluated on the test partition: primary definition "
  + ", ".join(f"{s['auroc']:.4f} ({s['events']:,} events among "
              f"{s['at_risk']:,})" for s in FM["stages"])
  + "; 1-hour episode rule "
  + ", ".join(f"{s['auroc']:.4f}" for s in TM["landmark_one_hour_rule"])
  + " for days 1 to 7, 8 to 14, and 15 to 30.")
H2("S7. Bootstrap and Subgroup Detail")
P("Resample counts: 1,000 for the test panel; 800 for subgroup and "
  "prior-utilization CIs; 500 for the primary out-of-fold estimate, model "
  "comparisons, and daily AUROC bands; 5,000 for contrasts with P values. "
  "Intervals resample patients and their stored predictions; selection and "
  "fitting were not repeated within draws. The White vs Black contrasts "
  "resample the union of the two patient sets jointly, so a patient recorded "
  "in both groups receives the same multiplicity in both. Race strings were consolidated by "
  "substring (WHITE to White; BLACK to Black; HISPANIC or LATINO to Hispanic "
  "or Latino; ASIAN to Asian; all others, declined, or unknown to Other or "
  "unknown). Prior admissions count every earlier admission of the patient "
  "in MIMIC-IV regardless of insurance.")
H2("S8. Demonstration Prototype")
P("The public demonstration prototype accepts only synthetic or manually "
  "entered values, displays a research-only notice, and stores no data. It "
  "runs an earlier development model, not the model reported here.")

H1("Supplementary Results")
H2("S9. Sensitivity and Era Analyses")
_sa, _su, _sn = (SENSD["sameday_as_readmission"], SENSD["unplanned_only"],
                 SENSD["nonelective_index_only"])
P(f"(1) Counting the transfer band as readmissions raised test events from "
  f"{_sa['test_events_primary']:,} to {_sa['test_events_alt']:,}; refit AUROC "
  f"{_sa['test_auroc_refit']:.4f}. (2) Excluding next admissions typed "
  f"ELECTIVE ({_su['test_events_alt']:,} events, "
  f"{pct(_su['test_prevalence_alt'])}) gave {_su['test_auroc_refit']:.4f}. "
  f"(3) Nonelective index admissions only ({_sn['test_admissions']:,}): "
  f"{_sn['test_auroc']:.4f}. (4) Excluding {BD5['n_boundary']:,} test "
  f"admissions ({pct(BD5['share'])}) whose latest possible year reached "
  f"2022: {BD5['auroc_safe']:.4f}. (1) and (2) are models refit to the "
  f"revised outcome; (3) and (4) evaluate the final model on subsets. (5) "
  f"Treating returns within 1 hour as part of the index stay (episode rule, "
  f"Table S4; {ER['test_events_episode_rule']:,} test events, "
  f"{pct(ER['test_prevalence_episode_rule'])}): refit "
  f"{ER['test_auroc_refit']:.4f}, final model {ER['test_auroc_final_model']:.4f} "
  f"(S1). (6) Era bands appear in Table S1.")
_rows = [["Era band", "Test admissions", "AUROC", "First admissions",
          "AUROC, first admissions", "eMAR coverage"]]
for e in RV["era"]:
    _rows.append([e["band"], f"{e['n']:,}", f"{e['auroc']:.4f}",
                  pct(e["first_admission_share"]),
                  f"{e['auroc_first_admissions']:.4f}", pct(e["emar_coverage"])])
caption("Table S1. Test discrimination by patient-level anchor_year_group "
        "band. eMAR coverage is the share of admissions with at least one "
        "medication-administration record. Bands are descriptive.",
        keep_with_next=True)
table(_rows, widths=[1.1, 1.0, 0.7, 1.0, 1.2, 0.9])
P("Discrimination was stable across eras despite large differences in "
  "medication-administration data coverage. Before the leak was removed, "
  "test AUROC had risen from "
  f"{OLD['v6_extras']['era_stratified_test']['bands'][0]['auroc']:.3f} to "
  f"{OLD['v6_extras']['era_stratified_test']['bands'][-1]['auroc']:.3f} "
  "across these bands, a gradient that disappeared with the leaked features.")
H2("S10. Payer Scope of Outcome Ascertainment")
P(f"{EX['payer_scope']['n_30d_next_nonmedicare']:,} next admissions within "
  f"30 days carried non-Medicare insurance; "
  f"{EX['payer_scope']['n_counted_as_readmission']:,} were counted as "
  f"readmissions, the remainder falling in the transfer band.")
H2("S11. Fixed-Set Comparison and Robustness Checks")
_fx = EX["fixed_consensus_vs_eligible_cv"]
VH_FOLDS = FM["validation_hierarchy"]["secondary_consensus_cv_dev_only"]["folds"]
P(f"With both predictor sets fixed, grouped cross-validation on identical "
  f"development folds gave AUROC {_fx['auroc_consensus_cv']:.4f} for the "
  f"{_fx['n_consensus']}-feature consensus set and "
  f"{_fx['auroc_eligible_cv']:.4f} for the {_fx['n_eligible']}-feature pool "
  f"(difference {_fx['paired_diff_consensus_minus_eligible']:+.4f}, 95% CI "
  f"{_fx['ci95'][0]:.4f} to {_fx['ci95'][1]:.4f}). Consensus cross-validation "
  f"folds were {', '.join(f'{x:.4f}' for x in VH_FOLDS)}.")
_hp = EVL["hyperparam_val_auroc"]
P(f"Post hoc robustness checks on the validation partition (not used to "
  f"change the model): with the final features fitted on the training "
  f"partition, the fixed hyperparameters gave AUROC "
  f"{EVL['val_auroc_full']:.4f}; {len(_hp)} alternative settings (maximum "
  f"depth 3 to 8, learning rate 0.03 to 0.1 with matched tree counts) gave "
  f"{min(_hp.values()):.4f} to {max(_hp.values()):.4f}; five further seeds "
  f"gave {min(EVL['seed_val_auroc']):.4f} to {max(EVL['seed_val_auroc']):.4f}; "
  f"and training-set AUROC was {EVL['train_auroc']:.4f}. Using 10%, 50%, and "
  f"100% of training patients gave {EVL['learning_curve_val_auroc']['0.1']:.4f}, "
  f"{EVL['learning_curve_val_auroc']['0.5']:.4f}, and "
  f"{EVL['learning_curve_val_auroc']['1.0']:.4f}. A model restricted to the "
  f"{len(EVL['util_features'])} prior-utilization features of the final set "
  f"reached {EVL['util_xgb_val_auroc']:.4f} on validation and "
  f"{EVL['test_auroc_util_only']:.4f} on test (one additional test "
  f"evaluation; the full model's advantage on test was "
  f"{EVL['test_auroc_full'] - EVL['test_auroc_util_only']:.4f}, 95% CI "
  f"{EVL['test_full_minus_util_ci95'][0]:.4f} to "
  f"{EVL['test_full_minus_util_ci95'][1]:.4f}; descriptive).")
H2("S12. Same-Date Readmission and Death")
_tr = EX["tie_rule_sensitivity"]
P(f"Registry death dates are day-granular; same-date readmission and death "
  f"were resolved death-first in the primary analysis. Resolving all "
  f"{_tr['n_ties']} ties ({_tr['n_ties_test']} in test) readmission-first "
  f"changed Harrell C from {_tr['death_first_primary']['harrell_c']:.4f} to "
  f"{_tr['readmission_first']['harrell_c']:.4f} and the 30-day readmission "
  f"cumulative incidence from "
  f"{pct(_tr['death_first_primary']['aj_cif30_readmission'])} to "
  f"{pct(_tr['readmission_first']['aj_cif30_readmission'])}.")
H2("S13. Prior Utilization, Race, and Subgroup Calibration")
_rows = [["Prior admissions", "Admissions (share)", "Prevalence",
          "AUROC (95% CI)", "Share of top-decile flags"]]
for k, v in PU.items():
    _rows.append([k, f"{v['admissions']:,} ({pct(v['share'])})",
                  pct(v["prevalence"]),
                  f"{v['auroc']:.4f} ({v['auroc_ci95'][0]:.4f} to "
                  f"{v['auroc_ci95'][1]:.4f})",
                  pct(v["share_of_top10pct_flags"])])
caption("Table S2. Test performance by number of prior admissions of any "
        "insurance.", keep_with_next=True)
table(_rows, widths=[1.1, 1.4, 0.9, 1.7, 1.2])
_w = RU["within_stratum_gap"]
P(f"White minus Black AUROC differences within strata were "
  + ", ".join(f"{v:+.3f} ({k})" for k, v in _w.items())
  + f". Standardized to the White utilization mix, the difference was "
  f"{RU['standardized_within_stratum_gap']:.3f} (95% CI "
  f"{RU['standardized_ci95'][0]:.3f} to {RU['standardized_ci95'][1]:.3f}); "
  f"crude {RU['crude_gap']:.4f}. Median prior admissions were "
  f"{RU['median_prior_admissions']['White']:.0f} for White and "
  f"{RU['median_prior_admissions']['Black']:.0f} for Black patients. Every "
  f"stratum has an AUROC below the overall {FM['metric_panel']['auroc']['point']:.4f} "
  f"because part of the model's discrimination comes from separating "
  f"patients between utilization levels, whose readmission rates differ "
  f"sharply.")
_g8 = {k.split(":", 1)[1]: v for k, v in SC.items() if k.startswith("race:")}
_a8 = {k: SC[f"age_band:{k}"] for k in ("<65", "65-74", "75-84", "85+")}
_nm = {"Hispanic/Latino": "Hispanic or Latino", "Other/Unknown": "other or unknown"}
P("Calibration-in-the-large by subgroup (observed to expected ratio, 95% "
  "patient-cluster bootstrap CI, 800 resamples): "
  + "; ".join(f"{_nm.get(k, k)} {v['o_e']:.2f} ({v['o_e_ci95'][0]:.2f} to "
              f"{v['o_e_ci95'][1]:.2f})" for k, v in _g8.items())
  + "; by age band "
  + "; ".join(f"{k.replace('-', ' to ')} {v['o_e']:.2f}" for k, v in _a8.items())
  + f". On average calibration only the heterogeneous other or unknown group "
  f"was miscalibrated (risk overestimated); the Black calibration-slope "
  f"interval excludes 1 (Table 4). Specificity at the shared threshold was "
  f"{_g8['White']['spec']:.3f} for White and {_g8['Black']['spec']:.3f} for "
  f"Black patients (difference "
  f"{SC['white_minus_black_specificity']['point']:.3f}, 95% CI "
  f"{SC['white_minus_black_specificity']['ci95'][0]:.3f} to "
  f"{SC['white_minus_black_specificity']['ci95'][1]:.3f}; 2,000 resamples; "
  f"descriptive). {SC['patients_in_both_white_and_black']} of "
  f"{SC['patients_white'] + SC['patients_black'] - SC['patients_in_both_white_and_black']:,} "
  f"patients carry both race labels across admissions and appear in both "
  f"groups; the contrasts therefore resample the union of patients jointly "
  f"(point estimates are observed differences; bootstrap means appear in "
  f"Supplementary Appendix 1). "
  f"A nonsignificant AUROC difference does not establish "
  f"equivalent performance; the White minus Black AUROC interval in the main "
  f"text spans differences from {FM['fairness']['white_minus_black']['ci95'][0]:.3f} "
  f"to {FM['fairness']['white_minus_black']['ci95'][1]:.3f}.")
H2("S14. Stage-Specific Attribution (Exploratory)")
_sig = [r_ for r_ in SD5 if r_["excludes_zero"]]
_e = "; ".join(f"{label(r_['feature']).lower()} ({r_['diff_pp']:+.1f} "
               f"percentage points)" for r_ in _sig if r_["diff_pp"] > 0)
_l = "; ".join(f"{label(r_['feature']).lower().replace(' ed ', ' ED ')} ({r_['diff_pp']:.1f})"
               for r_ in _sig if r_["diff_pp"] < 0)
_sh = {s["stage"]: s["shares"] for s in ST5}
P(f"Across days 1 to 7, 8 to 14, and 15 to 30, the prior-utilization share "
  f"of attribution rose from {_sh['1-7']['Prior utilization']:.1f}% to "
  f"{_sh['15-30']['Prior utilization']:.1f}%, while medication and order "
  f"activity fell from {_sh['1-7']['Medications and orders']:.1f}% to "
  f"{_sh['15-30']['Medications and orders']:.1f}% (Supplementary Figure S2, "
  f"panel A). Features weighted more for early returns: {_e}. Features "
  f"weighted more for late returns: {_l}. The 10 largest differentials were "
  f"selected before interval inspection, so the intervals are descriptive. "
  f"Stage models use the primary outcome definition.")

H1("Supplementary Table S3. Definitions of the Final Features")
DEFS = {
 "admission_type": "MIMIC admission type; target-encoded fold-locally",
 "admission_location": "MIMIC admission source; target-encoded fold-locally",
 "discharge_location": "Recorded discharge destination; target-encoded fold-locally",
 "age_at_admit": "Age in years at admission",
 "los_days": "Length of stay in whole days (discharge minus admission, floored)",
 "log_los_days": "log(1 + length of stay)",
 "albumin_last": "Last albumin stored on or before discharge (g/dL); missing if none",
 "albumin_x_los": "Last albumin multiplied by length of stay",
 "hemoglobin_last": "Last hemoglobin stored on or before discharge (g/dL); missing if none",
 "sodium_last": "Last sodium stored on or before discharge (mmol/L); missing if none",
 "wbc_last": "Last white cell count stored on or before discharge; missing if none",
 "bilirubin_max": "Maximum total bilirubin stored on or before discharge; missing if none",
 "lab_abnormal_rate": "Laboratory rows flagged abnormal divided by numeric results, both charted and stored on or before discharge",
 "n_labs_total": "Numeric laboratory results charted and stored on or before discharge",
 "log_n_labs": "log(1 + laboratory results)",
 "n_lab_item_types": "Distinct laboratory test types charted and stored on or before discharge",
 "n_lab_orders": "Laboratory orders placed",
 "gcs_min": "Minimum Glasgow Coma Scale charted during the stay; missing if never charted",
 "icu_flag": "Any intensive care unit stay",
 "stepdown_flag": "Any step-down unit stay",
 "ventilator_flag": "Documented mechanical ventilation",
 "antibiotic_flag": "Any antibiotic administered",
 "distinct_drugs": "Distinct drugs administered",
 "n_meds_total": "Prescriptions started on or before discharge",
 "n_discharge_drugs": "Medications on the discharge list",
 "n_distinct_routes": "Distinct medication administration routes",
 "n_emar_details": "Medication administration (eMAR) records",
 "n_order_types": "Distinct provider order types recorded under the admission",
 "orders_per_day": "Provider orders per day of stay",
 "orders_last_6h": "Late order rate multiplied by late orders (source column orders_last_6h)",
 "n_late_orders": "Orders placed late in the stay",
 "late_order_rate": "Late orders as a share of all orders",
 "late_order_burden": "Composite late-order burden",
 "consult_per_day": "Consult orders per day of stay",
 "discharge_surge": "Discharge between 10:00 and 14:59 (source column discharge_surge)",
 "n_datetime_events": "Charted date-time events",
 "provider_fragmentation_x_los": "Length of stay multiplied by distinct providers",
 "prior_admissions_all": "Earlier admissions of the patient (any insurance)",
 "prior_admissions_6m": "Earlier admissions (any insurance) discharged in the 180 days before the index admission",
 "prior_admits_6m_sq": "Square of prior admissions in 6 months",
 "prior_any_ed_6m": "Any earlier admission in the prior 180 days with an emergency department registration",
 "prior_mean_los_6m": "Mean whole-day length of stay of earlier admissions discharged in the prior 180 days (0 if none)",
 "prior_readmission_count": "Earlier Medicare-insured admissions in the prior 365 days followed by a readmission within 30 days (cohort-table definition, verified)",
 "log_prior_readmit_count": "log(1 + prior readmissions)",
 "time_since_last_discharge": "Days from the previous discharge of any insurance to the index admission; missing for first observed admissions",
 "freq_x_recency": "Prior admissions divided by (days since last discharge + 1)",
 "los_per_prior_admit": "Length of stay divided by (prior admissions + 1)",
}
_hr = TA["history_recompute"]
VER = {f: "Components, 100%" for f in (
    "freq_x_recency", "los_per_prior_admit", "albumin_x_los",
    "provider_fragmentation_x_los", "log_los_days", "log_n_labs",
    "log_prior_readmit_count", "prior_admits_6m_sq")}
VER["orders_last_6h"] = f"Components, {pct(DCK['orders_last_6h_eq_rate_x_late_orders'], 0)}"
VER.update({
    "prior_admissions_all": f"Raw timestamps, {pct(_hr['prior_admissions_all']['match_share'], 0)}",
    "prior_admissions_6m": f"Raw timestamps, {pct(HV['prior_admissions_6m']['p6_disch180'])}",
    "prior_mean_los_6m": f"Raw timestamps, {pct(HV['prior_mean_los_6m']['mlos_floor_disch180'])}",
    "prior_any_ed_6m": f"Raw timestamps, {pct(_hr['prior_any_ed_6m']['match_share'])}",
    "prior_readmission_count": f"Raw timestamps, {pct(PRV['lookback_variants']['admit_365'], 0)}",
    "time_since_last_discharge": f"Raw timestamps, {pct(HV['time_since_last_discharge']['nonfill_match_cohort'], 2)}",
    "n_labs_total": f"Raw table, sample, {pct(DCK['n_labs_total_eq_numeric_results_in_hadm'], 0)}",
    "lab_abnormal_rate": f"Raw table, sample, {pct(DCK['lab_abnormal_rate_eq_abnormal_rows_over_numeric_results'], 0)}",
    "n_lab_item_types": f"Raw table, sample, {pct(IA['raw_spot_check']['n_lab_item_types']['all_rows_of_hadm'], 0)}",
    "n_order_types": f"Raw table, sample, {pct(IA['raw_spot_check']['n_order_types']['all_rows_of_hadm'])}"})
for _f in ("hemoglobin_last", "albumin_last", "sodium_last", "wbc_last", "bilirubin_max"):
    if _f in TB:
        VER[_f] = f"Raw table, all admissions, stored before discharge ({pct(TB[_f]['agree_where_both_present'])} agreement)"
for _f in ("n_labs_total", "lab_abnormal_rate", "n_lab_item_types", "log_n_labs", "n_meds_total"):
    VER[_f] = "Raw table, all admissions, rebuilt with discharge cutoff (v12)"
if "gcs_min" in TB:
    VER["gcs_min"] = f"Raw table, all admissions ({pct(TB['gcs_min']['agree_where_both_present'], 0)} agreement)"
_rows = [["Feature", "Definition", "Verification"]]
for f in sorted(FC, key=lambda x: label(x)):
    _rows.append([label(f), DEFS.get(f, label(f)), VER.get(f, "Documented")])
caption("Table S3. The final features. Verification: share of admissions for "
        "which the stored value reproduces from its component features, from "
        "raw MIMIC-IV admission timestamps (all cohort admissions, earlier "
        "admissions only), or from raw event tables (random sample of 3,000 "
        "admissions); Documented: definition taken from the feature "
        "documentation and not recomputed.", keep_with_next=True)
table(_rows, font_size=8.5, widths=[1.8, 3.6, 1.1])

H1("Supplementary Figures")
figure(FIG7 / "v13_rfe_curve.png",
       "Supplementary Figure S1. Staged recursive feature elimination on "
       "development data: mean inner-validation AUROC versus features "
       f"retained, per outer fold; the dashed line marks the {NF}-feature "
       "consensus.", width=5.4)
figure(FIG7 / "v13_stage_combined.png",
       "Supplementary Figure S2. Attribution across post-discharge windows. "
       "(A) Share of total attribution by clinical domain. (B) Features whose "
       "attribution share differs most between days 1 to 7 and days 15 to 30; "
       "positive values indicate greater early importance; bars are 95% "
       "cluster-bootstrap CIs.", width=5.0)
figure(FIG7 / "v13_dca.png",
       "Supplementary Figure S3. Decision curves for the final model on the "
       "test partition (exploratory threshold range).", width=5.0)

DST = PUB / "Paper JAMIA Supplementary Material v13.docx"
doc.save(str(DST))
print(f"saved {DST}")

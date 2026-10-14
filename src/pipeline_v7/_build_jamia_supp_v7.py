# -*- coding: utf-8 -*-
"""Build the JAMIA Supplementary Material, v7 (leak-free rebuild).
All values are read from results_v7. References are named in text rather
than numbered, so the supplement does not depend on the main reference list."""
import json
from _v7_core import *  # noqa
from _v7_core import _style  # noqa
from _v7_labels import label

S = doc.styles
S["Normal"].font.size = Pt(11)
LK = json.loads((OUT7 / "leak_test_v7.json").read_text())
OLD = json.loads((REPO / "results_v6" / "final_model_v6.json").read_text())
p = CT["partitions"]
NF = FM["n_features"]
CB, OH, TM = RV["continuation_band"], RV["one_hour_rule"], RV["timing"]
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
r = t.add_run("Predicting 30-Day Hospital Readmission at Discharge: A "
              "Leakage-Safe, Calibrated Electronic Health Record Model in "
              "Medicare-Insured Adults"); r.italic = True
P("Supplementary Appendices 1 to 6 are separate files listed in the main "
  "manuscript. All values below are generated from the released results "
  "files (results_v7).")

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
  f"genuine early returns. The 1-hour window was therefore chosen from the "
  f"data, and the primary definition is retained with the 1-hour rule "
  f"reported as a sensitivity outcome (S9) and in the timing analysis.")
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
H2("S3. Empirical Future-Information Test")
P(f"Among {LK['first_ever_admissions']:,} first-ever index admissions (no "
  f"earlier admission of any kind in MIMIC-IV), "
  f"{pct(LK['share_with_later_admission'])} had a later admission. For each "
  f"of the 207 candidates we compared the share of rows at the feature's "
  f"modal value between patients without and with a later admission. A "
  f"prior-only feature should show no systematic difference; a feature "
  f"fixed at one value for every patient without a later admission, but not "
  f"for others, encodes the future. Only the 180-day length-of-stay trend "
  f"met this criterion: it was nonzero for {pct(LT['nonzero_share_first_ever'])} "
  f"of first-ever admissions, every one of which had a later admission "
  f"({pct(LT['nonzero_share_without_any_later_admission'])} of patients "
  f"without one had a nonzero value; AUROC "
  f"{LT['auroc_nonzero_for_later_admission']:.3f} for a later admission). "
  f"Thirty-day readmission was {pct(LT['readmit30_if_nonzero'])} when the "
  f"trend was nonzero and {pct(LT['readmit30_if_zero'])} when it was zero. "
  f"The trend and the two interaction terms built from it were removed. The "
  f"largest gap among the {NF} final features was "
  f"{LK['scan_max_gap_final_features']:.3f}.")
_o = LK["outpatient_features_exact_match"]
P(f"Outpatient-record features were compared with the outpatient table: "
  f"stored body-mass index matched the last value dated on or before "
  f"discharge for {pct(_o['bmi_last']['on_or_before_discharge'])} of "
  f"admissions versus {pct(_o['bmi_last']['overall_incl_future'])} for the "
  f"last value on any date; diastolic pressure matched for "
  f"{pct(_o['bp_diastolic_outpatient']['on_or_before_discharge'])} versus "
  f"{pct(_o['bp_diastolic_outpatient']['overall_incl_future'])}, confirming "
  f"time-bounded construction. Features built within the index stay were "
  f"checked against documented timestamps only.")
H2("S4. Missing-Data Handling")
P(f"Laboratory, vital-sign, and outpatient values that were never measured "
  f"were left missing and handled by the tree learner. Counts were set to "
  f"zero when no event occurred. Time since last discharge is set to a single "
  f"fixed value ({LK['time_since_last_discharge_first_ever_value']} days) for "
  f"first-ever admissions, so recency-weighted frequency is zero for them. "
  f"Interaction terms are missing when a component is missing. The logistic "
  f"comparator used fold-local median imputation with standardization.")
H2("S5. Clinical-Score Reconstruction and Refits")
P(f"LACE (van Walraven et al.): length-of-stay points by the original bands; "
  f"3 acuity points unless elective; Charlson comorbidity points from index "
  f"ICD diagnoses; and prior 180-day emergency department visits capped at "
  f"4. HOSPITAL (Donze et al.): hemoglobin below 12 g/dL, oncology service, "
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
  f"zero; Supplementary Appendix 1).")
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
  + "; 1-hour rule "
  + ", ".join(f"{s['auroc']:.4f}" for s in TM["landmark_one_hour_rule"])
  + " for days 1 to 7, 8 to 14, and 15 to 30.")
H2("S7. Bootstrap and Subgroup Detail")
P("Resample counts: 1,000 for the test panel; 800 for subgroup and "
  "prior-utilization CIs; 500 for the primary out-of-fold estimate, model "
  "comparisons, and daily AUROC bands; 5,000 for contrasts with P values. "
  "Intervals resample patients and their stored predictions; selection and "
  "fitting were not repeated within draws. Race strings were consolidated by "
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
  f"2022: {BD5['auroc_safe']:.4f}. (5) Treating returns within 1 hour as "
  f"continuations ({OH['test_events']:,} test events, "
  f"{pct(OH['test_prevalence'])}): refit {OH['test_auroc_refit']:.4f}, final "
  f"model {OH['test_auroc_final_model']:.4f}. (6) Era bands appear in Table S1.")
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
H2("S11. Fixed-Set Comparison")
_fx = EX["fixed_consensus_vs_eligible_cv"]
VH_FOLDS = FM["validation_hierarchy"]["secondary_consensus_cv_dev_only"]["folds"]
P(f"With both predictor sets fixed, grouped cross-validation on identical "
  f"development folds gave AUROC {_fx['auroc_consensus_cv']:.4f} for the "
  f"{_fx['n_consensus']}-feature consensus set and "
  f"{_fx['auroc_eligible_cv']:.4f} for the {_fx['n_eligible']}-feature pool "
  f"(difference {_fx['paired_diff_consensus_minus_eligible']:+.4f}, 95% CI "
  f"{_fx['ci95'][0]:.4f} to {_fx['ci95'][1]:.4f}). Consensus cross-validation "
  f"folds were {', '.join(f'{x:.4f}' for x in VH_FOLDS)}.")
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
H2("S13. Prior Utilization and Race")
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
H2("S14. Stage-Specific Attribution (Exploratory)")
_sig = [r_ for r_ in SD5 if r_["excludes_zero"]]
_e = "; ".join(f"{label(r_['feature']).lower()} ({r_['diff_pp']:+.1f} "
               f"percentage points)" for r_ in _sig if r_["diff_pp"] > 0)
_l = "; ".join(f"{label(r_['feature']).lower()} ({r_['diff_pp']:.1f})"
               for r_ in _sig if r_["diff_pp"] < 0)
_sh = {s["stage"]: s["shares"] for s in ST5}
P(f"Across days 1 to 7, 8 to 14, and 15 to 30, the prior-utilization share "
  f"of attribution rose from {_sh['1-7']['Prior utilization']:.1f}% to "
  f"{_sh['15-30']['Prior utilization']:.1f}%, while medication and order "
  f"activity fell from {_sh['1-7']['Medications and orders']:.1f}% to "
  f"{_sh['15-30']['Medications and orders']:.1f}% (Supplementary Figure S2, "
  f"panel A). Features weighted more for early returns: {_e}. Features "
  f"weighted more for late returns: {_l}. The 10 largest differentials were "
  f"selected before interval inspection, so the intervals are descriptive.")

H1("Supplementary Table S3. Definitions of the Final Features")
DEFS = {
 "admission_type": "MIMIC admission type; target-encoded fold-locally",
 "admission_location": "MIMIC admission source; target-encoded fold-locally",
 "discharge_location": "Recorded discharge destination; target-encoded fold-locally",
 "age_at_admit": "Age in years at admission",
 "los_days": "Length of stay in whole days (discharge minus admission, floored)",
 "log_los_days": "log(1 + length of stay)",
 "albumin_last": "Last albumin of the stay (g/dL)",
 "albumin_x_los": "Last albumin multiplied by length of stay",
 "hemoglobin_last": "Last hemoglobin of the stay (g/dL)",
 "sodium_last": "Last sodium of the stay (mmol/L)",
 "wbc_last": "Last white cell count of the stay",
 "bilirubin_max": "Maximum bilirubin during the stay",
 "lab_abnormal_rate": "Share of laboratory results flagged abnormal during the stay",
 "n_labs_total": "Laboratory results during the stay",
 "log_n_labs": "log(1 + laboratory results)",
 "n_lab_item_types": "Distinct laboratory test types",
 "n_lab_orders": "Laboratory orders placed",
 "gcs_min": "Minimum Glasgow Coma Scale charted",
 "icu_flag": "Any intensive care unit stay",
 "stepdown_flag": "Any step-down unit stay",
 "ventilator_flag": "Documented mechanical ventilation",
 "antibiotic_flag": "Any antibiotic administered",
 "distinct_drugs": "Distinct drugs administered",
 "n_meds_total": "Medication orders during the stay",
 "n_discharge_drugs": "Medications on the discharge list",
 "n_distinct_routes": "Distinct medication administration routes",
 "n_emar_details": "Medication administration (eMAR) records",
 "n_order_types": "Distinct order types",
 "orders_per_day": "Provider orders per day of stay",
 "orders_last_6h": "Orders in the final 6 hours before discharge",
 "n_late_orders": "Orders placed late in the stay",
 "late_order_rate": "Late orders as a share of all orders",
 "late_order_burden": "Composite late-order burden",
 "consult_per_day": "Consult orders per day of stay",
 "discharge_surge": "Indicator of increased order activity near discharge",
 "n_datetime_events": "Charted date-time events",
 "provider_fragmentation_x_los": "Length of stay multiplied by distinct providers",
 "prior_admissions_all": "Earlier admissions of the patient (any insurance)",
 "prior_admissions_6m": "Admissions beginning in the 180 days before the index admission",
 "prior_admits_6m_sq": "Square of prior admissions in 6 months",
 "prior_any_ed_6m": "Any emergency department visit in the prior 6 months",
 "prior_mean_los_6m": "Mean length of stay of prior 6-month admissions (0 if none)",
 "prior_readmission_count": "Prior 30-day readmissions",
 "log_prior_readmit_count": "log(1 + prior readmissions)",
 "time_since_last_discharge": "Days from the previous discharge to the index admission (fixed value if none)",
 "freq_x_recency": "Prior admissions divided by (days since last discharge + 1)",
 "los_per_prior_admit": "Length of stay divided by (prior admissions + 1)",
}
VER = {"freq_x_recency", "los_per_prior_admit", "albumin_x_los",
       "provider_fragmentation_x_los", "log_los_days", "log_n_labs",
       "log_prior_readmit_count", "prior_admits_6m_sq", "prior_admissions_all"}
_rows = [["Feature", "Definition", "Formula verified from data"]]
for f in sorted(FC, key=lambda x: label(x)):
    assert f in DEFS, f
    _rows.append([label(f), DEFS[f], "Yes" if f in VER else "Documented"])
caption("Table S3. The final features. Formula verified from data: the stored "
        "column reproduces exactly from its components or from raw MIMIC-IV "
        "timestamps for every admission.", keep_with_next=True)
table(_rows, font_size=8.5, widths=[1.8, 3.6, 1.1])

H1("Supplementary Figures")
figure(FIG7 / "v7_rfe_curve.png",
       "Supplementary Figure S1. Staged recursive feature elimination on "
       "development data: mean inner-validation AUROC versus features "
       f"retained, per outer fold; the dashed line marks the {NF}-feature "
       "consensus.", width=5.4)
figure(FIG7 / "v7_stage_combined.png",
       "Supplementary Figure S2. Attribution across post-discharge windows. "
       "(A) Share of total attribution by clinical domain. (B) Features whose "
       "attribution share differs most between days 1 to 7 and days 15 to 30; "
       "positive values indicate greater early importance; bars are 95% "
       "cluster-bootstrap CIs.", width=5.0)
figure(FIG7 / "v7_dca.png",
       "Supplementary Figure S3. Decision curves for the final model on the "
       "test partition (exploratory threshold range).", width=5.0)

DST = PUB / "Paper JAMIA Supplementary Material v7.docx"
doc.save(str(DST))
print(f"saved {DST}")

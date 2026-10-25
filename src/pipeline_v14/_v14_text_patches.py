# -*- coding: utf-8 -*-
"""v14 documentation-consistency fixes (no model change; results_v13)."""
from pathlib import Path

H = Path(__file__).resolve().parent


def patch(fn, pairs):
    p = H / fn; s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        if s.count(a) == 0 and s.count(b) >= 1:
            continue
        assert s.count(a) == 1, (fn, s.count(a), a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8"); print("patched", fn)


patch("_build_jamia_v14.py", [
 # Table 4 calibration slope to 3 decimals so the Black interval shows 0.999
 ('''    if key == "cal_slope":
        return f"{v:.2f} ({c_[0]:.2f} to {c_[1]:.2f})"''',
  '''    if key == "cal_slope":
        return f"{v:.3f} ({c_[0]:.3f} to {c_[1]:.3f})"'''),
 ('''   f"calibration slope was below 1 ({FR['Black']['cal_slope']:.2f}, 95% CI "
   f"{FR['Black']['cal_slope_ci95'][0]:.2f} to "
   f"{FR['Black']['cal_slope_ci95'][1]:.2f}), indicating slightly too "''',
  '''   f"calibration slope was below 1 ({FR['Black']['cal_slope']:.3f}, 95% CI "
   f"{FR['Black']['cal_slope_ci95'][0]:.3f} to "
   f"{FR['Black']['cal_slope_ci95'][1]:.3f}), indicating slightly too "'''),
 # limitations: independence of the primary estimate from earlier choices
 ('''   f"included. Hyperparameters were fixed rather than tuned within folds. "''',
  '''   f"included. The model family and hyperparameters were chosen during "
   f"earlier development, which had evaluated models on the test partition, "
   f"and were not re-selected within the outer folds, so the out-of-fold "
   f"estimate is not fully independent of those choices. "'''),
 # word budget
 ('''   f"signal. Additive SHAP decompositions give each admission a ranked list "
   f"of contributing factors.")''', '''   f"signal.")'''),
 ('''   f"window; no difference by race or sex was detected, with wide "
   f"intervals for smaller groups and lower specificity for Black "
   f"patients, and discrimination was lower in older patients and in "''',
  '''   f"window; no difference by race or sex was detected, but specificity "
   f"was lower for Black patients, and discrimination was lower in older patients and in "'''),
 ('''PB("The leaked feature carried a name and documentation describing a "
   "prior 180-day window and passed the original availability audit; "''',
  '''PB("The leaked feature was documented as a prior 180-day window and "
   "passed the availability audit; "'''),
])

patch("_build_jamia_supp_v14.py", [
 # S13 from the joint bootstrap
 ('''  f"{SC['white_minus_black_specificity']['point']:.3f}, 95% CI "
  f"{SC['white_minus_black_specificity']['ci95'][0]:.3f} to "
  f"{SC['white_minus_black_specificity']['ci95'][1]:.3f}; 2,000 resamples; "
  f"descriptive). {SC['patients_in_both_white_and_black']} of "''',
  '''  f"{SC['white_minus_black_specificity_joint']['point_observed']:.3f}, 95% CI "
  f"{SC['white_minus_black_specificity_joint']['ci95'][0]:.3f} to "
  f"{SC['white_minus_black_specificity_joint']['ci95'][1]:.3f}; "
  f"{SC['white_minus_black_specificity_joint']['n_boot']:,} joint draws; "
  f"descriptive). {SC['patients_in_both_white_and_black']} of "'''),
 ('''  f"text spans differences from {FM['fairness']['white_minus_black']['ci95'][0]:.3f} "
  f"to {FM['fairness']['white_minus_black']['ci95'][1]:.3f}.")''',
  '''  f"text spans differences from {SC['white_minus_black_auroc_joint']['ci95'][0]:.3f} "
  f"to {SC['white_minus_black_auroc_joint']['ci95'][1]:.3f} (observed "
  f"{SC['white_minus_black_auroc_joint']['point_observed']:.4f}, bootstrap mean "
  f"{SC['white_minus_black_auroc_joint']['bootstrap_mean']:.4f}, "
  f"{SC['white_minus_black_auroc_joint']['n_boot']:,} joint draws).")'''),
 # S3: label the historical audit, restrict the agreement range to final features
 ('''P(f"Recomputation of history features from raw admission timestamps (all "
  f"{HV['n_cohort_admissions']:,} cohort admissions, using only admissions "
  f"before the index admission): prior admissions of any insurance "''',
  '''P(f"Historical audit of the source table (v8; the discharge interval was "
  f"later redefined in v13, below). Recomputation of history features from "
  f"raw admission timestamps (all "
  f"{HV['n_cohort_admissions']:,} cohort admissions, using only admissions "
  f"before the index admission): prior admissions of any insurance "'''),
 ('''  f"were documented after discharge and are now excluded. Before the cutoff "
  f"the recomputed features reproduced the stored values for "
  f"{pct(min(_d[k]['definition_match_without_documentation_cutoff'] for k in _ch + _em + _rx))} to "
  f"{pct(max(_d[k]['definition_match_without_documentation_cutoff'] for k in _ch + _em + _rx), 0)} of admissions, "
  f"confirming the definitions; the cutoff changed "''',
  '''  f"were documented after discharge and are now excluded. Before the cutoff "
  f"the recomputed final-model features reproduced the stored values for "
  f"{pct(min(_d[k]['definition_match_without_documentation_cutoff'] for k in _ch + _em + _rx if k in FC))} to "
  f"{pct(max(_d[k]['definition_match_without_documentation_cutoff'] for k in _ch + _em + _rx if k in FC), 0)} of admissions, "
  f"confirming the definitions (two eligible candidates outside the final "
  f"set, incomplete-dose count and rate, reproduced "
  f"{pct(_d['n_incomplete_doses']['definition_match_without_documentation_cutoff'])} "
  f"because the source parsed the incomplete-dose field differently); the cutoff changed "'''),
 ('''  f"(per feature in Supplementary Appendix 1, Cutoff_Rebuild). The discharge "''',
  '''  f"(per feature in Supplementary Appendix 1, Cutoff_Rebuild, which also "
  f"records the v12 count rebuild). The discharge "'''),
 (' "n_meds_total": "Prescriptions started on or before discharge",',
  ' "n_meds_total": "Prescriptions started on or before discharge and entered into the pharmacy system on or before discharge",'),
 (' "n_order_types": "Distinct provider order types recorded under the admission",',
  ' "n_order_types": "Distinct provider order types with ordertime on or before discharge (source extraction filter)",'),
 ('''    "time_since_last_discharge": f"Raw timestamps, {pct(HV['time_since_last_discharge']['nonfill_match_cohort'], 2)}",''',
  '''    "time_since_last_discharge": "Raw timestamps, 100% (v13 definition recomputed for every admission)",'''),
 ('''for _f in ("n_labs_total", "lab_abnormal_rate", "n_lab_item_types", "log_n_labs", "n_meds_total"):
    VER[_f] = "Raw table, all admissions, rebuilt with discharge cutoff (v12)"''',
  '''for _f in ("n_labs_total", "lab_abnormal_rate", "n_lab_item_types", "log_n_labs"):
    VER[_f] = "Raw table, all admissions, rebuilt with discharge cutoff (v12)"
_d13 = TB.get("v13_documentation_time_enforced", {})
for _f, _v in _d13.items():
    if _f.startswith("_") or not isinstance(_v, dict) or "definition_match_without_documentation_cutoff" not in _v:
        continue
    VER[_f] = (f"Raw table, all admissions, documentation-time cutoff (v13; "
               f"{pct(_v['definition_match_without_documentation_cutoff'])} agreement before cutoff)")
VER["freq_x_recency"] = "Components, 100% (v13 interval)"
VER["log_time_since_discharge"] = "Components, 100% (v13 interval)"'''),
])

patch("_v14_appendices.py", [
 (' "n_order_types": ("Distinct provider order types recorded under the hospital admission", "index admission (all rows)", IA["raw_spot_check"]["n_order_types"]["all_rows_of_hadm"]),',
  ' "n_order_types": ("Distinct provider order types with ordertime on or before discharge (source extraction filter; spot-checked)", "index stay, ordertime on or before discharge", IA["raw_spot_check"]["n_order_types"]["all_rows_of_hadm"]),'),
 # HighDraw sheet: joint bootstrap rows for the race contrasts
 ('''    pd.DataFrame([{"contrast": k, "observed_point": (FM["fairness"]["white_minus_black"]["point"]
                   if k == "white_minus_black" else FM["baselines"]["uplift_lace" if k == "lace_uplift" else "uplift_hospital12"]),
                   "bootstrap_mean": v["point"], **{kk: vv for kk, vv in v.items() if kk not in ("ci95", "point")},
                   "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]}
                  for k, v in EX["highdraw_contrasts_5000"].items()]).to_excel(''',
  '''    _hd = [{"contrast": k, "observed_point": FM["baselines"]["uplift_lace" if k == "lace_uplift" else "uplift_hospital12"],
            "bootstrap_mean": v["point"], **{kk: vv for kk, vv in v.items() if kk not in ("ci95", "point")},
            "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]}
           for k, v in EX["highdraw_contrasts_5000"].items() if k != "white_minus_black"]
    for k in ("white_minus_black_auroc_joint", "white_minus_black_specificity_joint"):
        v = SC[k]
        _hd.append({"contrast": k, "observed_point": v["point_observed"], "bootstrap_mean": v["bootstrap_mean"],
                    "p_two_sided_plus1": v["p_two_sided_plus1"], "n_boot": v["n_boot"], "design": v["design"],
                    "patients_union": SC["patients_white"] + SC["patients_black"] - SC["patients_in_both_white_and_black"],
                    "patients_in_both_groups": SC["patients_in_both_white_and_black"],
                    "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]})
    pd.DataFrame(_hd).to_excel('''),
 # Cutoff_Rebuild: label the stage of the 80.22 count
 ('''                 [{"feature": k, "version": "v13 (documentation-time cutoff)", **v}
                  for k, v in TB.get("v13_documentation_time_enforced", {}).items()]''',
  '''                 [{"feature": k, "version": "v13 (documentation-time cutoff)",
                   **({**v, "share_old_fill_80_22_before_v13_rebuild": v.pop("share_old_fill_80_22_remaining"),
                       "share_fill_80_22_after_v13_rebuild": 0.0} if "share_old_fill_80_22_remaining" in v else v)}
                  for k, v in TB.get("v13_documentation_time_enforced", {}).items()]'''),
 # Appendix 2: label bootstrap draws and use the 5,000-draw intervals for the two tested contrasts
 ('''add_table(d, [["Comparator", "AUROC", "Final-model uplift (95% CI)"]] +
          [[k.replace("_", " "), f"{v['auroc']:.4f}",
            f"{v['uplift']:.4f} ({v['uplift_ci95'][0]:.4f} to {v['uplift_ci95'][1]:.4f})"]
           for k, v in G.items() if k != "reconstruction_match"])''',
  '''_ci5 = {"lace_original": FM["baselines"]["uplift_lace_ci"], "hospital_original": FM["baselines"]["uplift_hospital12_ci"]}
add_table(d, [["Comparator", "AUROC", "Final-model uplift (95% CI)", "Bootstrap draws"]] +
          [[k.replace("_", " "), f"{v['auroc']:.4f}",
            (f"{v['uplift']:.4f} ({_ci5[k][0]:.4f} to {_ci5[k][1]:.4f})" if k in _ci5 else
             f"{v['uplift']:.4f} ({v['uplift_ci95'][0]:.4f} to {v['uplift_ci95'][1]:.4f})"),
            "5,000 (prespecified test; same as main text)" if k in _ci5 else "1,000 (descriptive)"]
           for k, v in G.items() if k != "reconstruction_match"])'''),
 # Appendix 4: separate the 48-feature and 44-feature rows
 ('''        [f"v11 to v13 consensus model ({NF} features) on the rebuilt table; v12 enforces the discharge cutoff on laboratory and medication counts and drops two duplicate pool features; v13 enforces documentation time on chart, medication-administration, and prescription features",
         f"out of fold {FM['validation_hierarchy']['primary_oof_procedure_dev_only']['auroc']:.4f}; "
         f"test {FM['metric_panel']['auroc']['point']:.4f}", "Reported model"],''',
  '''        ["v11 consensus model (48 features) on the table rebuilt without global imputation",
         "out of fold 0.7058; test 0.7060", "Superseded"],
        ["v12 consensus model (48 features); discharge cutoff on laboratory and medication counts; two duplicate pool features dropped",
         "out of fold 0.7057; test 0.7055", "Superseded"],
        [f"v13 consensus model ({NF} features); documentation-time cutoff on chart, medication-administration, and prescription features; discharge interval from any earlier admission",
         f"out of fold {FM['validation_hierarchy']['primary_oof_procedure_dev_only']['auroc']:.4f}; "
         f"test {FM['metric_panel']['auroc']['point']:.4f}", "Reported model"],'''),
])

patch("_v11_checklists.py", [
 ('"bias: high for the historically exposed test partition and low for the out-of-fold "',
  '"bias: high for the historically exposed test partition and unclear for the out-of-fold "'),
])

# -*- coding: utf-8 -*-
"""v12 text edits on the generated v12 builders (run once after the sed copy)."""
from pathlib import Path

H = Path(__file__).resolve().parent


def patch(fn, pairs):
    p = H / fn; s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        if s.count(a) == 0 and s.count(b) == 1:
            continue                      # already applied
        if s.count(a) == 0 and a.startswith('"""'):
            continue                      # docstring differs; cosmetic
        assert s.count(a) == 1, (fn, s.count(a), a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8"); print("patched", fn)


patch("_build_jamia_v12.py", [
 ('"""Build the JAMIA manuscript, v11 (imputation-free rebuild, 2026-10).',
  '"""Build the JAMIA manuscript, v12 (cutoff-enforced counts, duplicate removal, 2026-10).'),
 ('''f"leak-free procedure reached out-of-fold AUROC {f4(OOF['auroc'])} "''',
  '''f"corrected procedure reached out-of-fold AUROC {f4(OOF['auroc'])} "'''),
 ('''   f"recomputed from raw admission timestamps, laboratory features were "
   f"rebuilt from raw results stored on or before discharge, and outpatient "
   f"and selected index-stay features were checked against raw tables "
   f"(Supplementary Methods S3). These checks target one class of leakage "
   f"and cannot certify that every predictor is leak-free.")''',
  '''   f"recomputed from raw admission timestamps; laboratory values and counts "
   f"were rebuilt from raw results stored on or before discharge and "
   f"medication counts from prescriptions started on or before discharge; "
   f"two exact duplicate features were dropped; and outpatient and other "
   f"index-stay features were checked against raw tables (Supplementary "
   f"Methods S3). These checks target one class of leakage and cannot "
   f"certify that every predictor is leak-free.")'''),
 ('''   f"admissions with no measurement (Supplementary Methods S4); the table "
   f"was rebuilt with missing values preserved, and all results below use "
   f"it and the leak-free pool of {N_ELIG} features.")''',
  '''   f"admissions with no measurement (Supplementary Methods S4); the table "
   f"was rebuilt with missing values preserved and a discharge cutoff "
   f"enforced on every laboratory and medication count, and all results "
   f"below use it and the corrected pool of {N_ELIG} features.")'''),
 ('''   "analysis of deidentified data did not require institutional review "
   "board review at Florida International University.")''',
  '''   "analysis of deidentified data did not require institutional review "
   "board review at Florida International University. No protocol was "
   "prepared, the study was not registered, and patients and the public "
   "were not involved.")'''),
 ('''   f"Index-stay aggregates use all rows of the hospital admission, including "
   f"rare rows timestamped after discharge "
   f"({pct(PDR['labevents']['row_share_after_discharge'], 2)} of laboratory "
   f"and {pct(PDR['poe']['row_share_after_discharge'], 2)} of order rows, "
   f"unrelated to the outcome), and only some index-stay features were "
   f"checked against raw tables. The primary outcome retains probable "''',
  '''   f"Order, medication-administration, chart, and transfer features were "
   f"extracted with a discharge-time filter in the source pipeline and "
   f"spot-checked, not fully recomputed, against raw tables. The primary "
   f"outcome retains probable "'''),
 ('''   "patient clustering, and engineered features whose time windows are not "
   "what their names suggest [23,25]. Roadmaps call for patient-level "
   "partitioning, fold-local preprocessing, and verification that every "
   "predictor is available at deployment time [22,23,26], but such "
   "verification is usually performed by reading documentation rather than "
   "by testing the data.")''',
  '''   "patient clustering, and engineered features whose time windows are not "
   "what their names suggest [23,25]. Roadmaps call for patient-level "
   "partitioning, fold-local preprocessing, and verified predictor "
   "availability [22,23,26], but verification usually means reading "
   "documentation rather than testing the data.")'''),
 ('''   f"clinical stability. Net benefit in decision curves is not evidence "
   f"that using the model would improve care or reduce readmissions.")''',
  '''   f"clinical stability. Net benefit is not evidence that use would "
   f"improve care.")'''),
])

patch("_build_jamia_supp_v12.py", [
 ('"""Build the JAMIA Supplementary Material, v11 (independent review revisions).',
  '"""Build the JAMIA Supplementary Material, v12.'),
 ('''  f"types {pct(_rs['n_order_types']['all_rows_of_hadm'])}, all counted over "
  f"every row recorded under the hospital admission. The feature stored as "''',
  '''  f"types {pct(_rs['n_order_types']['all_rows_of_hadm'])} (spot check made "
  f"before the cutoff rebuild described below). The feature stored as "'''),
 ('''  f"before discharge were stored afterwards and are excluded. Other "
  f"index-stay features were not recomputed and rely on documented "
  f"timestamps.")''',
  '''  f"before discharge were stored afterwards and are excluded. The "
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
  f"months) were removed from the eligible pool. Order, "
  f"medication-administration, chart, date-time, service, and transfer "
  f"features were extracted with a discharge-time filter in the source "
  f"pipeline (ordertime, charttime, transfertime, or intime on or before "
  f"discharge) and were spot-checked, not fully recomputed, against raw "
  f"tables.")'''),
 ('''       ["Merged episodes ending in in-hospital death, excluded", f"{FLOW['all_episodes_ending_in_death_excluded']:,}", f"{FLOW['test_episodes_ending_in_death_excluded']:,}"],''',
  '''       ["Merged episodes ending in in-hospital death, excluded", f"{FLOW['all_episodes_ending_in_death_excluded']:,}", f"{FLOW['test_episodes_ending_in_death_excluded']:,}"],
       ["Rows in both categories above (a link whose own chain ends in death), counted once", f"{FLOW['all_rows_in_both_categories']:,}", f"{FLOW['test_rows_in_both_categories']:,}"],'''),
 ('caption("Table S4. Episode flow under the 1-hour continuation sensitivity rule.",',
  'caption("Table S4. Episode flow under the 1-hour continuation sensitivity rule. Retained episodes equal index admissions minus merged links minus death exclusions plus the rows counted in both categories.",'),
 (' "n_labs_total": "Numeric laboratory results recorded under the admission",',
  ' "n_labs_total": "Numeric laboratory results charted and stored on or before discharge",'),
 (' "lab_abnormal_rate": "Laboratory rows flagged abnormal divided by numeric laboratory results of the admission",',
  ' "lab_abnormal_rate": "Laboratory rows flagged abnormal divided by numeric results, both charted and stored on or before discharge",'),
 (' "n_lab_item_types": "Distinct laboratory test types recorded under the admission",',
  ' "n_lab_item_types": "Distinct laboratory test types charted and stored on or before discharge",'),
 (' "n_meds_total": "Medication orders during the stay",',
  ' "n_meds_total": "Prescriptions started on or before discharge",'),
 ('''for _f in ("hemoglobin_last", "albumin_last", "sodium_last", "wbc_last", "bilirubin_max"):
    if _f in TB:
        VER[_f] = f"Raw table, all admissions, stored before discharge ({pct(TB[_f]['agree_where_both_present'])} agreement)"''',
  '''for _f in ("hemoglobin_last", "albumin_last", "sodium_last", "wbc_last", "bilirubin_max"):
    if _f in TB:
        VER[_f] = f"Raw table, all admissions, stored before discharge ({pct(TB[_f]['agree_where_both_present'])} agreement)"
for _f in ("n_labs_total", "lab_abnormal_rate", "n_lab_item_types", "log_n_labs", "n_meds_total"):
    VER[_f] = "Raw table, all admissions, rebuilt with discharge cutoff (v12)"'''),
])

patch("_v12_appendices.py", [
 ('        [f"v11 consensus model ({NF} features) on the rebuilt table",',
  '        [f"v11 and v12 consensus model ({NF} features) on the rebuilt table; v12 enforces the discharge cutoff on laboratory and medication counts and drops two duplicate pool features",'),
 ('print("A5 and A6 are written by _v11_checklists.py")', 'print("A5 and A6 are written by _v11_checklists.py (reused for v12)")'),
])

patch("_build_jamia_cover_v12.py", [
 ('"missing values preserved, required laboratory results to be stored "\n  "before discharge, and reran the complete pipeline. On the test "',
  '"missing values preserved, required laboratory results to be stored "\n  "before discharge, enforced the discharge cutoff on every laboratory and "\n  "medication count, and reran the complete pipeline. On the test "'),
])

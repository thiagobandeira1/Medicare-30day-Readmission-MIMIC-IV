# -*- coding: utf-8 -*-
"""v16: co-author review round (2026-10 to 2026-11). No model or result
change; every number still comes from results_v13.

Part 1 (this file, VERSION): the v16 builders are copies of the v15 builders;
repoint every output name from v15 to v16 so no v16 script can touch a v15 file.
Part 2 (TRACKED): wording revisions from the co-author review.
Part 3 (COMMENTS): clarifications requested in the co-author review.

Each (old, new) pair must match exactly once; a pair already applied is skipped,
so the script can be rerun safely."""
from pathlib import Path

H = Path(__file__).resolve().parent

# Results of earlier pairs that a later pair (audit corrections, 2026-11-11)
# rewrites on purpose. When neither side of such an earlier pair is present,
# the pair is skipped as superseded; every other pair must still match.
SUPERSEDED = (
    "The submitted version will be tagged v16.0-jamia",
    "(CMS) estimates the annual",
    "for regression models, along with recurring weaknesses: ",
    "value in this manuscript is generated (final_model_v13.json, the v8 ",
    "data can improve on these scores [42]",
    "and is not intended for real patient ",
    "author's Master of Science capstone report (May 2026)",
)


def patch(fn, pairs):
    p = H / fn; s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        if s.count(b) >= 1:          # already applied (also when b contains a)
            continue
        if s.count(a) == 0 and any(m in b for m in SUPERSEDED):
            print("  superseded by a later pair:", fn, "|", b.strip()[:60])
            continue
        assert s.count(a) == 1, (fn, s.count(a), a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8"); print("patched", fn)


# ------------------------------------------------------------------ VERSION
patch("_build_jamia_v16.py", [
 ('"""Build the JAMIA manuscript, v12 (cutoff-enforced counts, duplicate removal, 2026-10).',
  '"""Build the JAMIA manuscript, v16 (co-author review round, 2026-11; same model as v13).'),
 ('  "The submitted version will be tagged v15.0-jamia and archived at Zenodo "',
  '  "The submitted version will be tagged v16.0-jamia and archived at Zenodo "'),
 ('DST = PUB / "Paper JAMIA Submission v15.docx"',
  'DST = PUB / "Paper JAMIA Submission v16.docx"'),
])
patch("_build_jamia_supp_v16.py", [
 ('"""Build the JAMIA Supplementary Material, v9 (episode rule for continuations).',
  '"""Build the JAMIA Supplementary Material, v16 (co-author review round, 2026-11).'),
 ('DST = PUB / "Paper JAMIA Supplementary Material v15.docx"',
  'DST = PUB / "Paper JAMIA Supplementary Material v16.docx"'),
])
patch("_build_jamia_cover_v16.py", [
 ('DST = PUB / "JAMIA Cover Letter v15.docx"',
  'DST = PUB / "JAMIA Cover Letter v16.docx"'),
])
patch("_v16_appendices.py", [
 ('PKT = PUB / "JAMIA Submission Packet v15"',
  'PKT = PUB / "JAMIA Submission Packet v16"'),
])
patch("_export_jamia_v16_pdf.py", [
 ('DOCS = ["Paper JAMIA Submission v15.docx",\n        "Paper JAMIA Supplementary Material v15.docx"]',
  'DOCS = ["Paper JAMIA Submission v16.docx",\n        "Paper JAMIA Supplementary Material v16.docx"]'),
])
patch("_v16_repro_check.py", [
 ('Document(PUB / "Paper JAMIA Submission v15.docx")',
  'Document(PUB / "Paper JAMIA Submission v16.docx")'),
 ('Document(PUB / "Paper JAMIA Supplementary Material v15.docx")',
  'Document(PUB / "Paper JAMIA Supplementary Material v16.docx")'),
])

# ------------------------------------------------------------------ TRACKED
# Wording revisions from the co-author review.
patch("_build_jamia_v16.py", [
 # ins 1 "an", ins 2 "of" (Abstract, Results)
 ('''f"corrected procedure reached out-of-fold AUROC {f4(OOF['auroc'])} "''',
  '''f"corrected procedure reached an out-of-fold AUROC of {f4(OOF['auroc'])} "'''),
 # ins 5 "estimates", del 6 "puts"
 ('''"the Centers for Medicare & Medicaid Services (CMS) puts the annual "''',
  '''"the Centers for Medicare & Medicaid Services (CMS) estimates the annual "'''),
 # ins 7 "lead to", del 8 "become"
 ('''"model performance across groups can become unequal care [5,6].")''',
  '''"model performance across groups can lead to unequal care [5,6].")'''),
 # ins 9 ", along", del 10 " together"
 ('''"for regression models together with recurring weaknesses: "''',
  '''"for regression models, along with recurring weaknesses: "'''),
 # del 11 "absent", ins 12 "absence of"
 ('''"leakage-prone splits, absent calibration assessment, and unexamined "''',
  '''"leakage-prone splits, absence of calibration assessment, and unexamined "'''),
 # del 13 "insurance recorded as", ins 14 "insurance" (his edit leaves a double space; one space kept)
 ('''f"The cohort comprised admissions with insurance recorded as Medicare "''',
  '''f"The cohort comprised admissions with Medicare insurance "'''),
 # ins 15 "for", del 16 "of"; ins 17 "within 0 to 30 days", del 18 "beginning more than 0 and up to 30 days"
 ('''f"outcome was the first same-system readmission of any cause beginning "
   f"more than 0 and up to 30 days after index discharge, identified from "''',
  '''f"outcome was the first same-system readmission for any cause within "
   f"0 to 30 days after index discharge, identified from "'''),
 # ins 19 "s", del 20 "s"
 ('''f"admissions; all inference accounts for this clustering. Planned "''',
  '''f"admissions; all inferences account for this clustering. Planned "'''),
 # ins 21 ","
 ('''f"admissions (no earlier admission in MIMIC-IV) a feature built only "''',
  '''f"admissions (no earlier admission in MIMIC-IV), a feature built only "'''),
 # ins 23 "for", del 24 "of"
 ('''f"admissions of each of the {TA['trend_patients_multi_admission']:,} "''',
  '''f"admissions for each of the {TA['trend_patients_multi_admission']:,} "'''),
 # ins 25 "we", del 26 "it had been", ins 27 "it"
 ('''f"multi-admission patients: it had been computed once per patient over "''',
  '''f"multi-admission patients: we computed it once per patient over "'''),
 # ins 28 "was"
 ('''f"final model reproduced from earlier admissions only "''',
  '''f"final model was reproduced from earlier admissions only "'''),
])

# ----------------------------------------------------------------- COMMENTS
# Clarifications from the co-author review. Abstract kept at 250 words
# (labels counted); main text +4 words (3,994).
patch("_build_jamia_v16.py", [
 # Abstract cap: the cohort size moves before "admissions", so the repeated word goes (1 word)
 ('''PA(f"Retrospective study of Medicare-insured MIMIC-IV v3.1 admissions "
   f"({CT['cohort_v2_admissions']:,} admissions, {CT['cohort_v2_patients']:,} "
   f"patients; prevalence''',
  '''PA(f"Retrospective study of {CT['cohort_v2_admissions']:,} Medicare-insured "
   f"MIMIC-IV v3.1 admissions ({CT['cohort_v2_patients']:,} "
   f"patients; prevalence'''),
 # Abstract cap: XGBoost is a model name; the family stays as "gradient-boosted" (2 words)
 ('''f"A consensus extreme gradient boosting (XGBoost) model was selected "''',
  '''f"A consensus gradient-boosted (XGBoost) model was selected "'''),
 # Clarity: say what the documentation claimed about the feature (+5);
 # "was computed over" becomes "used" (2 words) and one article goes (1 word)
 ('''PA(f"A documented trend feature was computed over each patient's whole "
   f"record, including later admissions; removing it and two derived terms "
   f"lowered the test area under the receiver operating characteristic "''',
  '''PA(f"A feature documented as a prior 180-day length-of-stay trend used "
   f"each patient's whole record, including later admissions; removing it "
   f"and two derived terms lowered test area under the receiver operating characteristic "'''),
 # Clarity: the main text's own verb and no undefined "index";
 # "frozen" is internal shorthand (1 word)
 ('''f"within 1 hour of discharge; folding these into the index stay "
   f"gave test AUROC {OH['test_auroc_final_model']:.3f} (frozen) and "
   f"{OH['test_auroc_refit']:.3f} (refit). The White minus Black AUROC "''',
  '''f"within 1 hour of discharge; merging these into the preceding stay "
   f"gave test AUROC {OH['test_auroc_final_model']:.3f} "
   f"({OH['test_auroc_refit']:.3f} after refitting). The White minus Black AUROC "'''),
 # Terminology: the bands are 3-year calendar periods
 ('''"that inflated discrimination and created spurious subgroup and era "
   "patterns; simple empirical checks found it."''',
  '''"that inflated discrimination and created spurious subgroup and "
   "calendar-period differences; simple empirical checks found it."'''),
 # Name the class of leakage the checks target (+4)
 ('''f"Methods S3). These checks target one class of leakage and cannot "''',
  '''f"Methods S3). These checks target one class of leakage (information "
   f"from later encounters) and cannot "'''),
 # One term in the whole paper: "calendar period" replaces "era" (0 words net)
 ('''f"AUROC was stable across calendar-era bands "''',
  '''f"AUROC was stable across calendar-period bands "'''),
 ('''a rising calendar-era gradient, and a racial "''',
  '''a rising calendar-period gradient, and a racial "'''),
 ('''f"discrimination, and produced spurious subgroup and era findings; "''',
  '''f"discrimination, and produced spurious subgroup and calendar-period findings; "'''),
 ('''"(sensitivity and era analyses, payer scope, fixed-set comparison, "''',
  '''"(sensitivity and calendar-period analyses, payer scope, fixed-set comparison, "'''),
 ('''"Supplementary Tables S1 to S4 (calendar era, prior-utilization strata, "''',
  '''"Supplementary Tables S1 to S4 (calendar period, prior-utilization strata, "'''),
 # Table 3: rows for this study in bold, said in the
 # caption (bold survives typesetting; shading does not; bold must not read as "best")
 ('''        "not head-to-head results.", keep_with_next=True)
table(_t2, font_size=8.5, widths=[2.0, 1.5, 0.7, 2.2])''',
  '''        "not head-to-head results. Rows for this study are in bold.",
        keep_with_next=True)
_t3 = table(_t2, font_size=8.5, widths=[2.0, 1.5, 0.7, 2.2])
for _row in _t3.rows:
    if _row.cells[0].text.startswith("This study"):
        for _c in _row.cells:
            for _par in _c.paragraphs:
                for _r in _par.runs:
                    _r.bold = True'''),
])
patch("_build_jamia_supp_v16.py", [
 ('H2("S9. Sensitivity and Era Analyses")', 'H2("S9. Sensitivity and Calendar-Period Analyses")'),
 ('f"(S1). (6) Era bands appear in Table S1.")', 'f"(S1). (6) Calendar-period bands appear in Table S1.")'),
 ('_rows = [["Era band", "Test admissions",', '_rows = [["Calendar period", "Test admissions",'),
 ('P("Discrimination was stable across eras despite large differences in "',
  'P("Discrimination was stable across calendar periods despite large differences in "'),
])
patch("_v16_checklists.py", [
 ('"Single center; performance by calendar era reported"', '"Single center; performance by calendar period reported"'),
 ('"Calendar-era bands only (single center)"', '"Calendar-period bands only (single center)"'),
])
patch("_build_jamia_cover_v16.py", [
 ('"discrimination, an advantage for first observed admissions, and a calendar-era "',
  '"discrimination, an advantage for first observed admissions, and a calendar-period "'),
])
patch("_v16_appendices.py", [
 ('sheet_name="Era_Bands"', 'sheet_name="Calendar_Period_Bands"'),
])

# -------------------------------------------------------------------- EXTRA
# From the independent check of v16 (2026-11-09); present since v15, not
# raised by the reviewers.
# 1. Data Availability named final_model_v7.json (the earlier 47-feature
#    model); every value is generated from results_v13/final_model_v13.json.
# 2. Page numbers in the footer (main: 1, 2, ...; supplement: S1, S2, ...),
#    so the page references in the TRIPOD+AI checklist can be followed.
_FOOTER = '''
def _page_footer(prefix=""):
    """Centered page number in the footer; sits inside the bottom margin, so
    the text layout does not change."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    for sec in doc.sections:
        par = sec.footer.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if prefix:
            par.add_run(prefix).font.size = Pt(10)
        fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
        run = OxmlElement("w:r"); rpr = OxmlElement("w:rPr")
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "20"); rpr.append(sz); run.append(rpr)
        t = OxmlElement("w:t"); t.text = "1"; run.append(t); fld.append(run)
        par._p.append(fld)


'''
patch("_build_jamia_v16.py", [
 ('''  "value in this manuscript is generated (final_model_v7.json, the v8 "''',
  '''  "value in this manuscript is generated (final_model_v13.json, the v8 "'''),
 ('DST = PUB / "Paper JAMIA Submission v16.docx"',
  _FOOTER + '_page_footer()\nDST = PUB / "Paper JAMIA Submission v16.docx"'),
])
patch("_build_jamia_supp_v16.py", [
 ('DST = PUB / "Paper JAMIA Supplementary Material v16.docx"',
  _FOOTER + '_page_footer("S")\nDST = PUB / "Paper JAMIA Supplementary Material v16.docx"'),
])

# ----------------------------------------------------------- AI DISCLOSURE
# Authors' decision (2026-11-10): one short disclosure sentence, the same in
# the Acknowledgments and in the cover letter.
patch("_build_jamia_v16.py", [
 ('''  "the PhysioNet team for maintaining MIMIC-IV. Generative AI tools were "
  "used in this work: OpenAI ChatGPT for independent audits of draft "
  "manuscripts, and Anthropic Claude for analysis-code development, "
  "independent methodological review (including the review that identified "
  "the future-information leak), figure generation, and drafting and "
  "editing of the manuscript. The authors reviewed and verified all code, "
  "analyses, and text and take full responsibility for the content.")''',
  '''  "the PhysioNet team for maintaining MIMIC-IV. Generative AI tools "
  "(Anthropic Claude and OpenAI ChatGPT) were used for coding, writing, and "
  "review assistance; the authors take full responsibility for the content.")'''),
])
patch("_build_jamia_cover_v16.py", [
 ('''P("Generative AI tools were used in this work: OpenAI ChatGPT for "
  "independent audits of draft manuscripts, and Anthropic Claude for "
  "analysis-code development, independent methodological review (including "
  "the review that identified the leak), figure generation, and drafting "
  "and editing of the manuscript. The authors reviewed and verified all "
  "code, analyses, and text and take full responsibility for the content.; "
  "this use is also disclosed in the Acknowledgments.")''',
  '''P("Generative AI tools (Anthropic Claude and OpenAI ChatGPT) were used for "
  "coding, writing, and review assistance; the authors take full "
  "responsibility for the content.")'''),
])

# ------------------------------------------------------ DISCOVERY WORDING
# Authors' decision (2026-11-10): say in the authors' voice that the leak was
# found when they re-examined an earlier version's features (main text net 0).
patch("_build_jamia_v16.py", [
 ('''   "would have caught it: testing history features among first observed "''',
  '''   "caught it: testing history features among first observed "'''),
 ('''   f"the leak itself was found after earlier versions had been evaluated; "''',
  '''   f"the leak itself was found only when we re-examined an earlier version's features; "'''),
])
patch("_build_jamia_cover_v16.py", [
 ('''  "versions of this work. We detected it with simple empirical checks "''',
  '''  "versions of this work. We detected it when we re-examined the features of "
  "an earlier version with simple empirical checks "'''),
])

# -------------------------------------------------------------- CONTACT
# Authors' decision (2026-11-10): the corresponding author's postal address is
# the school's address, not a home address (JAMIA prints it in the article).
# The phone stays on the title page for the editorial office.
patch("_build_jamia_v16.py", [
 ('''         "Bandeira); [address withheld], United States; Phone: "''',
  '''         "Bandeira); Knight Foundation School of Computing and Information "
         "Sciences, Florida International University, 11200 SW 8th St, Miami, "
         "FL 33199, United States; Phone: "'''),
])
patch("_build_jamia_cover_v16.py", [
 ('''P("[address withheld], United States; Phone: [phone withheld]; "''',
  '''P("11200 SW 8th St, Miami, FL 33199, United States; Phone: [phone withheld]; "'''),
])

# ------------------------------------------- AUDIT CORRECTIONS (2026-11-11)
# Pre-submission audit, 2026-11-11 (summary in LEAKAGE_CORRECTION_v7.md,
# v16.1 entry). Factual corrections and JAMIA
# compliance only; no analysis or result changes.
patch("_build_jamia_v16.py", [
 # The cited CMS page (ref 1) does not report a $26 billion cost; keep only
 # what refs 1 and 2 support (HRRP payment adjustments since fiscal year 2013).
 ('''PB("Unplanned 30-day readmissions burden the United States health system: "
   "the Centers for Medicare & Medicaid Services (CMS) estimates the annual "
   "direct cost of Medicare readmissions above $26 billion [1], and the "
   "Hospital Readmissions Reduction Program (HRRP) has tied reimbursement to "
   "risk-adjusted readmission performance since 2013 [1,2]. Risk estimates "''',
  '''PB("Unplanned 30-day readmissions burden the United States health system, "
   "and the Hospital Readmissions Reduction Program (HRRP) of the Centers "
   "for Medicare & Medicaid Services has tied reimbursement to "
   "risk-adjusted readmission performance since 2013 [1,2]. Risk estimates "'''),
 # Cited studies give C statistics of about 0.68 (LACE derivation) to 0.77
 # (LACE, UK validation); HOSPITAL 0.71 (derivation) and 0.72 (validation).
 ('''   "the HOSPITAL score [8] reach AUROCs of roughly 0.60 to 0.72 in external "
   "validation [9,10,41], and systematic reviews report similar ceilings "
   "for regression models, along with recurring weaknesses: "''',
  '''   "the HOSPITAL score [8] reach AUROCs of about 0.68 to 0.77 in derivation "
   "and validation studies [7,8,10,41]; a systematic review found that most "
   "readmission risk models perform poorly [9], and reviews note recurring "
   "weaknesses: "'''),
 ('''   "data can exceed these ceilings [42], and gradient-boosted trees remain "''',
  '''   "data can improve on these scores [42], and gradient-boosted trees remain "'''),
 # RQ3: only global and stage-level SHAP results are reported.
 ('''   "established clinical scores? RQ3: Can global and patient-level Shapley "''',
  '''   "established clinical scores? RQ3: Can global Shapley "'''),
 # JAMIA: tables cited in numerical order; Table 4 cited in the text.
 ('''   "with the held-out test AUROC of the model each choice would deploy "
   "(Table 2).")''',
  '''   "with the held-out test AUROC of the model each choice would deploy.")'''),
 ('''   f"detected (AUROC {f4(FR['Black']['auroc'])} vs {f4(FR['White']['auroc'])}; "''',
  '''   f"detected (Table 4; AUROC {f4(FR['Black']['auroc'])} vs {f4(FR['White']['auroc'])}; "'''),
 # Table 2: apparent AUROC rose by 0.071 and held-out AUROC by 0.069.
 ('''   f"and held-out AUROC alike (by {_opt_trend:.3f}), because the future "''',
  '''   f"and held-out AUROC alike (by {_opt_trend:.3f} and "
   f"{ABP['heldout_test_auroc'] - ABR['heldout_test_auroc']:.3f}), because the future "'''),
 # prior_any_ed_6m is not a final-model feature; the range is 99.7% to 100%.
 ('''              TA["history_recompute"]["prior_any_ed_6m"]["match_share"],
''',
  '''              # prior_any_ed_6m dropped: not a final-model feature (audit 2026-11-11)
'''),
 # The original capstone code was recovered (results_v13/trend_code_recovery_v16.json).
 ('''   f"feature table; it was rebuilt before this version. The original "
   f"code of the leaked feature was not "
   f"retained, so its exact formula is inferred from its signatures. "''',
  '''   f"feature table; it was rebuilt before this version. The original "
   f"code of the leaked feature, recovered in a final audit, confirms "
   f"the mechanism the checks inferred. "'''),
 # MIMIC-IV (ref 27): the BIDMC IRB granted the waiver and approved sharing.
 ('''PB("MIMIC-IV creation was approved by the institutional review boards of "
   "the Massachusetts Institute of Technology and Beth Israel Deaconess "
   "Medical Center with a waiver of informed consent [27,32]. The first "''',
  '''PB("The institutional review board of Beth Israel Deaconess Medical "
   "Center granted a waiver of informed consent for MIMIC-IV and approved "
   "its sharing [27]. The first "'''),
 # Data Availability: the upstream feature extraction is not in the
 # repository; the v16 archive already exists.
 ('''  "agreement [27,32] and is not redistributed. All analysis code, the "
  "leakage tests, and the machine-readable results files from which every "
  "value in this manuscript is generated (final_model_v13.json, the v8 "
  "to v13 audit files, and companions) are available at "
  "https://github.com/thiagobandeira1/Medicare-30day-Readmission-MIMIC-IV. "
  "The submitted version will be tagged v16.0-jamia and archived at Zenodo "
  "under concept DOI 10.5281/zenodo.21987701 before publication; Zenodo "''',
  '''  "agreement [27,32] and is not redistributed. The analysis code from the "
  "source feature table onward, the leakage tests, and the machine-readable "
  "results files from which every value in this manuscript is generated "
  "(final_model_v13.json, the v8 to v13 audit files, and companions) are "
  "available at "
  "https://github.com/thiagobandeira1/Medicare-30day-Readmission-MIMIC-IV "
  "and archived at Zenodo "
  "under concept DOI 10.5281/zenodo.21987701; Zenodo "'''),
])
patch("_v16_checklists.py", [
 ('''"Public repository; the submitted version will be tagged and archived at Zenodo before publication"''',
  '''"Public repository; archived at Zenodo under concept DOI 10.5281/zenodo.21987701"'''),
])
patch("_build_jamia_supp_v16.py", [
 ('''SC = json.loads((OUT7 / "subgroup_calibration_v13.json").read_text())''',
  '''SC = json.loads((OUT7 / "subgroup_calibration_v13.json").read_text())
TCR = json.loads((OUT7 / "trend_code_recovery_v16.json").read_text())
_FMR = json.loads((OUT7 / "final_model_v13.json").read_text())["fairness"]["race"]'''),
 # S3: the recovered capstone code reproduces the stored feature exactly.
 ('''  f"of the whole longitudinal record. The original extraction code was not "
  f"retained, and no candidate reconstruction reproduced it exactly (the "
  f"closest, the slope of whole-day length of stay across all of a "
  f"patient's Medicare admissions, matched {pct(TFS['slope_idx_losd'])} of "
  f"patients, including single-admission patients for whom every candidate "
  f"is zero). "''',
  '''  f"of the whole longitudinal record. A final audit recovered the original "
  f"capstone code, which confirms this: for each admission it took the "
  f"length of stay minus that of the patient's previous admission, averaged "
  f"these differences over the patient's whole record, and attached the "
  f"mean to every admission (zero for patients with one admission), with "
  f"no 180-day window. Applied to the source table, it reproduces the "
  f"stored values for {pct(TCR['admissions_exact_share'], 0)} of its "
  f"{TCR['n_admissions']:,} admissions. "'''),
 # S8: describe the demonstration prototype as its repositories document it.
 ('''P("The public demonstration prototype accepts only synthetic or manually "
  "entered values, displays a research-only notice, and stores no data. It "
  "runs an earlier development model, not the model reported here.")''',
  '''P("The public demonstration prototype is documented as a research "
  "prototype, not a medical device, and is not intended for real patient "
  "data. It runs an earlier development model, not the model reported here.")'''),
 # S11: the 15-entry grid includes the fixed setting, so 14 alternatives.
 ('''  f"{EVL['val_auroc_full']:.4f}; {len(_hp)} alternative settings (maximum "''',
  '''  f"{EVL['val_auroc_full']:.4f}; {len(_hp) - 1} alternative settings (maximum "'''),
 # S13: specificity from the same source as the main text (0.610, not 0.611).
 ('''  f"{_g8['White']['spec']:.3f} for White and {_g8['Black']['spec']:.3f} for "''',
  '''  f"{_FMR['White']['specificity']:.3f} for White and {_FMR['Black']['specificity']:.3f} for "'''),
])
patch("_v16_appendices.py", [
 # Appendix 4: the reported (v13) decision curve, as in Supplementary Figure S3.
 ('''d.add_paragraph().add_run().add_picture(str(FIG7 / "v7_dca.png"), width=Inches(5.5))''',
  '''d.add_paragraph().add_run().add_picture(str(REPO / "figures_v13" / "v13_dca.png"), width=Inches(4.75))'''),
 ('''        ["v5/v6 leakage-safe consensus model (31 features)",''',
  '''        ["v5/v6 consensus model (31 features)",'''),
 ('''        ["v7 to v10 leak-free consensus model (47 features)",''',
  '''        ["v7 to v10 consensus model without the trend features (47 features)",'''),
])
patch("_build_jamia_cover_v16.py", [
 ('''  "yet most published models are evaluated in ways that overstate what a "''',
  '''  "yet many published models are evaluated in ways that overstate what a "'''),
 ('''  "development, gives secondary detail only (AUROC 0.706, good calibration). "''',
  '''  "development, provides supporting detail only (AUROC 0.706, good calibration). "'''),
 ('''  "gradient, all of which disappear once it is removed. The same comparison "''',
  '''  "gradient, all of which disappear once it is removed. A later comparison "'''),
 ('''  "and the complete pipeline is public.")''',
  '''  "and the analysis code from the source feature table onward is public.")'''),
 # Related material disclosed (JAMIA cover letter rules); preprint plan.
 ('''  "any journal; this is its first submission. Analysis code and results "
  "files from earlier versions of this work are archived in a public "
  "repository (Zenodo); the corrected version will be archived there on "
  "submission. The authors have no related papers published or under "
  "consideration. The study is a "
  "secondary analysis of the deidentified, publicly available MIMIC-IV "''',
  '''  "any journal; this is its first submission. Analysis code and results "
  "files from earlier versions of this work, together with the first "
  "author's Master of Science capstone report (May 2026), which reports "
  "pre-correction results, are archived in a public repository (Zenodo, "
  "doi:10.5281/zenodo.21987702); the corrected version is archived there "
  "as well. We intend to post this submitted version "
  "as a preprint on arXiv (cs.LG) under the arXiv non-exclusive license and "
  "will send the identifier to the editorial office once it is announced. "
  "The study is a "
  "secondary analysis of the deidentified, credentialed-access MIMIC-IV "'''),
])

# ------------------------------------ AUDIT CORRECTIONS, ROUND 2 (2026-11-11)
# A second verification pass found these.
patch("_build_jamia_v16.py", [
 # Ref 14 (Rajkomar 2018) used full EHR data (including notes) and compared
 # against a model refit on the HOSPITAL score variables.
 ('''   "subgroup performance [9,11,12]. Machine learning on structured EHR "
   "data can improve on these scores [42], and gradient-boosted trees remain "''',
  '''   "subgroup performance [9,11,12]. Machine learning on EHR "
   "data can improve on HOSPITAL-type scores [42], and gradient-boosted trees remain "'''),
 # Results and Supplement S4: the imputation was found by comparing stored
 # columns with raw tables.
 ('''   "decisive step; it also exposed median imputation fitted on all "''',
  '''   "decisive step; comparing stored columns with raw tables also exposed "
   "median imputation fitted on all "'''),
 # final_model_v13.json dca: near-equal to treat-all at 0.05 to 0.07
 # (differences 0.0000 to -0.0002), above it from 0.08.
 ('''PB(f"Decision curves showed positive net benefit over treat-all and "
   f"treat-none across thresholds 0.05 to 0.40 (Supplementary Appendix "''',
  '''PB(f"Decision curves showed net benefit above treat-none across "
   f"thresholds 0.05 to 0.40 and above treat-all from 0.08 to 0.40 (Supplementary Appendix "'''),
])
patch("_build_jamia_supp_v16.py", [
 # S8: keep only what the public repositories document.
 ('''P("The public demonstration prototype is documented as a research "
  "prototype, not a medical device, and is not intended for real patient "
  "data. It runs an earlier development model, not the model reported here.")''',
  '''P("The public demonstration prototype is documented as a research "
  "prototype, not a medical device. It runs an earlier development model, "
  "not the model reported here.")'''),
])
patch("_v16_checklists.py", [
 ('''"MIT and BIDMC IRBs with waiver of consent; PhysioNet DUA; no FIU IRB review required"''',
  '''"BIDMC IRB granted a waiver of informed consent and approved sharing of MIMIC-IV; PhysioNet DUA; no FIU IRB review required"'''),
 ('''"Model artifact, encodings, and feature order are in the public repository (MIT license) and will be in the tagged release"''',
  '''"Model artifact, encodings, and feature order are in the public repository (MIT license) and its Zenodo archive (concept DOI 10.5281/zenodo.21987701)"'''),
])
patch("_build_jamia_cover_v16.py", [
 # Name the capstone report's authors and cite the concept DOI.
 ('''  "files from earlier versions of this work, together with the first "
  "author's Master of Science capstone report (May 2026), which reports "
  "pre-correction results, are archived in a public repository (Zenodo, "
  "doi:10.5281/zenodo.21987702); the corrected version is archived there "
  "as well. We intend to post this submitted version "''',
  '''  "files from earlier versions of this work, together with the capstone "
  "report of the first two authors with Dr. Poellabauer as senior author "
  "(April 2026, revised May 2026), which reports pre-correction results, "
  "are archived on Zenodo (doi:10.5281/zenodo.21987702, an August 2026 "
  "version under concept doi:10.5281/zenodo.21987701); the corrected "
  "version is archived under the same concept DOI. We intend to post this "
  "submitted version "'''),
])

# ------------------------------------ AUDIT CORRECTIONS, ROUND 3 (2026-11-11)
# Ref 14 (Rajkomar 2018) used two US health systems' EHR data, not MIMIC-IV.
patch("_build_jamia_v16.py", [
 ('''   f"(0.727) [29], multimodal (0.791) [28], and full-EHR deep-learning "
   f"(0.75 to 0.76) [42] reports, which differ in cohort, outcome, and "''',
  '''   f"(0.727) [29] and multimodal (0.791) [28] reports and full-EHR deep "
   f"learning elsewhere (0.75 to 0.76) [42], which differ in cohort, outcome, and "'''),
])
patch("_v16_checklists.py", [
 # Appendix 5 page anchors now come from the journal's reviewer PDF, whose
 # rendering paginates the manuscript in 29 pages (Word: 30).
 ('''            "evaluation. Page numbers refer to the manuscript PDF (p) and the Supplementary "
            "Material PDF (p S).")''',
  '''            "evaluation. Page numbers refer to the manuscript as paginated in the journal's "
            "reviewer PDF (p) and to the Supplementary Material PDF (p S).")'''),
])

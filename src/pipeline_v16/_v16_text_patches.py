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


def patch(fn, pairs):
    p = H / fn; s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        if s.count(b) >= 1:          # already applied (also when b contains a)
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

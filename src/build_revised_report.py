"""Build the post-defense revised report (Word .docx).

Produces paper/Capstone_Final_Report_Revised_2026-05.docx.
The original report is preserved unchanged in the original submission folder.

This version preserves the substantive content and ordering of the original
report; only factual values that changed under the strict no-leakage protocol
are updated, plus a clarifying note that XGBoost (deployed) and LightGBM
(co-equal alternative) are statistically indistinguishable on the held-out
test partition. No em-dashes in newly written body text.

All numerical values come from the validated notebook artefacts.

Usage:
    python src/build_revised_report.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ── two-column / section-break helpers ───────────────────────────────────────

def _set_section_columns(section, num=1, space=432):
    """Set the number of body-text columns on a section."""
    sect_pr = section._sectPr
    cols = sect_pr.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols')
        sect_pr.append(cols)
    cols.set(qn('w:num'), str(num))
    cols.set(qn('w:space'), str(space))


def switch_columns(doc, num_columns):
    """Insert a continuous section break and set its column count.

    Page dimensions and margins are inherited from the previous section.
    """
    new_section = doc.add_section(WD_SECTION.CONTINUOUS)
    if len(doc.sections) >= 2:
        prev = doc.sections[-2]
        new_section.top_margin = prev.top_margin
        new_section.bottom_margin = prev.bottom_margin
        new_section.left_margin = prev.left_margin
        new_section.right_margin = prev.right_margin
        new_section.page_height = prev.page_height
        new_section.page_width = prev.page_width
    _set_section_columns(new_section, num_columns)
    return new_section

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "figures"
OUT_PATH = REPO_ROOT / "paper" / "Capstone_Final_Report_Revised_2026-05.docx"

PROG = json.loads((RESULTS / "progression.json").read_text())
ENS = json.loads((RESULTS / "v7_summary.json").read_text())
CV = json.loads((RESULTS / "cv5_summary.json").read_text())
REF = json.loads((RESULTS / "v17_reference_metrics.json").read_text())

LGB_TEST = ENS["models"]["lightgbm"]["test_auroc"]
XGB_TEST = ENS["models"]["xgboost"]["test_auroc"]
CB_TEST = ENS["models"]["catboost"]["test_auroc"]
HIST_TEST = ENS["models"]["histgbm"]["test_auroc"]
BLEND_TEST = ENS["blend"]["test_auroc"]
BLEND_W = ENS["blend"]["weights"]

CV_LGB = CV["models"]["lightgbm"]
CV_XGB = CV["models"]["xgboost"]
CV_CB = CV["models"]["catboost"]
CV_HGB = CV["models"]["histgbm"]
CV_BLEND = CV["blend"]

LACE = 0.684
CBERT = 0.714


# ── helpers ──────────────────────────────────────────────────────────────────

def set_run_font(run, name="Times New Roman", size=11, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:cs"), name)


def add_para(doc, text="", *, bold=False, italic=False, size=10,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6,
             font="Times New Roman", color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        set_run_font(r, name=font, size=size, bold=bold, italic=italic, color=color)
    return p


def add_heading(doc, text, level=1, size=11, space_before=12, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.style = doc.styles[f"Heading {level}"]
    r = p.add_run(text)
    set_run_font(r, name="Times New Roman", size=size, bold=True, color=RGBColor(0, 0, 0))
    return p


def add_figure(doc, image_path: Path, caption_text: str, width_inches=3.1, wide=False):
    """Insert a figure plus its caption.

    In a 2-column body section, set wide=True to make the figure span both
    columns (the function temporarily switches to a single-column section
    and switches back after the caption).
    """
    if wide:
        switch_columns(doc, 1)
        eff_width = 6.5
    else:
        eff_width = width_inches
    if not image_path.exists():
        add_para(doc, f"[Figure missing: {image_path.name}]", italic=True,
                 color=RGBColor(0xC0, 0, 0))
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(image_path), width=Inches(eff_width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    r = cap.add_run(caption_text)
    set_run_font(r, name="Times New Roman", size=9, italic=True)
    if wide:
        switch_columns(doc, 2)


def add_table(doc, headers, rows, col_widths=None, header_shading="E7E6E6", wide=False):
    """Insert a table. Set wide=True in a 2-column body section to make the
    table span both columns."""
    if wide:
        switch_columns(doc, 1)
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    if col_widths is not None:
        for ci, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[ci].width = Inches(w)
    hdr = table.rows[0]
    for ci, txt in enumerate(headers):
        cell = hdr.cells[ci]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(txt)
        set_run_font(r, name="Times New Roman", size=10, bold=True)
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), header_shading)
        tcPr.append(shd)
    for ri, row in enumerate(rows):
        for ci, txt in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(txt))
            set_run_font(r, name="Times New Roman", size=10)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    if wide:
        switch_columns(doc, 2)
    return table


def add_banner(doc, lines):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.5)
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "E8F0F5")
    tcPr.append(shd)
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "12")
        b.set(qn("w:color"), "2F5F7A")
        tcBorders.append(b)
    tcPr.append(tcBorders)
    cell.text = ""
    for i, line in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        is_first = (i == 0)
        r = p.add_run(line)
        set_run_font(r, name="Times New Roman", size=11 if is_first else 10,
                     bold=is_first, color=RGBColor(0x1F, 0x3F, 0x5A))
    doc.add_paragraph().paragraph_format.space_after = Pt(8)


# ── build ────────────────────────────────────────────────────────────────────

def build():
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        section.page_height = Inches(11)
        section.page_width = Inches(8.5)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10)

    middot = chr(0xB7)

    # ── Title (full width) ──
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_after = Pt(2)
    r = title_p.add_run("Predicting 30-Day Hospital Readmission in Medicare Patients")
    set_run_font(r, size=18, bold=True)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(14)
    r = sub_p.add_run("An Interpretable Gradient-Boosting Model on MIMIC-IV v3.1")
    set_run_font(r, size=13, italic=True)

    # ── ACM-style author block: Thiago left, Armando right (2-cell table) ──
    auth_table = doc.add_table(rows=1, cols=2)
    auth_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell in auth_table.rows[0].cells:
        cell.width = Inches(3.25)
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = OxmlElement("w:tcBorders")
        for side in ("top", "left", "bottom", "right"):
            b = OxmlElement(f"w:{side}")
            b.set(qn("w:val"), "nil")
            tcBorders.append(b)
        tcPr.append(tcBorders)

    authors_lhs = [
        ("Thiago Bandeira", 12, True, False),
        ("(First author)", 10, False, True),
        ("Florida International University", 11, False, True),
        ("Miami, Florida", 11, False, True),
        ("tbati006@fiu.edu", 11, False, True),
    ]
    authors_rhs = [
        ("Armando Gonzalez", 12, True, False),
        ("(Co-author)", 10, False, True),
        ("Florida International University", 11, False, True),
        ("Miami, Florida", 11, False, True),
        ("agonz1689@fiu.edu", 11, False, True),
    ]
    for col_idx, lines in enumerate([authors_lhs, authors_rhs]):
        cell = auth_table.rows[0].cells[col_idx]
        cell.text = ""
        for i, (txt, sz, bd, it) in enumerate(lines):
            p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(txt)
            set_run_font(r, size=sz, bold=bd, italic=it)

    # ── Senior author below ──
    senior_p = doc.add_paragraph()
    senior_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    senior_p.paragraph_format.space_before = Pt(10)
    senior_p.paragraph_format.space_after = Pt(0)
    r = senior_p.add_run("Senior author: Dr. Christian Poellabauer")
    set_run_font(r, size=11, bold=True)
    aff2_p = doc.add_paragraph()
    aff2_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    aff2_p.paragraph_format.space_after = Pt(0)
    r = aff2_p.add_run("Knight Foundation School of Computing and Information Sciences, "
                       "Florida International University")
    set_run_font(r, size=10, italic=True)
    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_p.paragraph_format.space_after = Pt(14)
    r = date_p.add_run("April 2026 (revised May 2026)")
    set_run_font(r, size=10, italic=True)

    # Switch the body of the paper to a two-column layout (ACM-style).
    # Front matter (title, authors, date) stays single-column above this point.
    switch_columns(doc, 2)

    # ── Abstract ───────────────────────────────────────────────────────────
    add_heading(doc, "ABSTRACT", level=1)
    add_para(doc,
        f"Thirty-day hospital readmission is a healthcare-quality measure subject "
        f"to financial penalties through the CMS Hospital Readmissions Reduction "
        f"Program. This work developed and tested a supervised machine-learning "
        f"framework for estimating the 30-day readmission probability for 244,576 "
        f"Medicare admissions contained in the MIMIC-IV v3.1 database. An "
        f"iterative process across seven dataset versions (V1 through V7) produced "
        f"a parsimonious set of fifty features spanning prior service utilisation, "
        f"comorbidity, medication complexity, clinical severity, and operational "
        f"flow. Four gradient-boosting families (LightGBM, XGBoost, CatBoost, "
        f"HistGradientBoosting) were trained on the V7 set across ten random seeds "
        f"and averaged to reduce variance. A scipy-optimised blend of the four "
        f"families was also constructed as a documented discrimination ceiling. "
        f"Under a strict 80/20 patient-grouped train and test protocol with a 10 "
        f"percent inner-validation slice carved from the training data for early "
        f"stopping (the test partition is evaluated exactly once), XGBoost is "
        f"retained as the deployment candidate at a test AUROC of {XGB_TEST:.4f} "
        f"for continuity with the defended original protocol; LightGBM "
        f"({LGB_TEST:.4f}) is reported as a co-equal alternative within seed-level "
        f"variance. The blend reaches {BLEND_TEST:.4f} and does not improve on the "
        f"best single model. The deployed XGBoost model outperforms the LACE "
        f"clinical index by {XGB_TEST-LACE:+.3f} AUROC and a published "
        f"ClinicalBERT model by {XGB_TEST-CBERT:+.3f} AUROC. SHAP explanations "
        f"provide both global and patient-specific rationales for each prediction, "
        f"producing an actionable risk-assessment score suitable for integration "
        f"into existing electronic health record systems.")

    add_heading(doc, "CCS Concepts", level=2)
    add_para(doc,
        "Computing methodologies: Supervised learning by classification; Gradient "
        "boosting. Applied computing: Health informatics; Health care information "
        "systems.")

    add_heading(doc, "Keywords", level=2)
    add_para(doc,
        "30-day readmission; Medicare; MIMIC-IV; gradient boosting; LightGBM; "
        "XGBoost; SHAP; healthcare analytics; interpretable machine learning.")

    # ── Introduction ──────────────────────────────────────────────────────
    add_heading(doc, "INTRODUCTION", level=1)
    add_para(doc,
        "Unplanned hospital readmission within thirty days of discharge is a "
        "recurring problem in the quality of care delivered to Medicare patients. "
        "Their combination of advanced age, multiple chronic conditions, and "
        "polypharmacy substantially elevates the risk of post-discharge "
        "decompensation. This work describes an end-to-end machine-learning "
        "pipeline that estimates the probability of all-cause 30-day readmission "
        "at or near the moment of discharge, using only the structured electronic "
        "health record data contained in MIMIC-IV v3.1.")

    # ── §1 Overview ───────────────────────────────────────────────────────
    add_heading(doc, "1. OVERVIEW", level=1)
    add_para(doc,
        f"This work applies supervised machine learning to the Medicare subset of "
        f"the MIMIC-IV v3.1 electronic health record (EHR) database to predict "
        f"30-day unplanned readmission at the time of discharge. The study cohort "
        f"includes 244,576 Medicare admissions, of which 21.1 percent were "
        f"followed by a readmission within thirty days. The analytical pipeline "
        f"imports administrative, clinical, medication, laboratory, and "
        f"operational tables; engineers a curated set of fifty features through a "
        f"multi-step engineering process; and trains four gradient-boosting "
        f"families over ten random seeds. An optimised blend of the four families "
        f"is also trained as a documented discrimination ceiling. Under the strict "
        f"no-leakage protocol described in section 7.2, the four single-model "
        f"families cluster tightly around a test AUROC of approximately 0.79 "
        f"(XGBoost {XGB_TEST:.4f}, LightGBM {LGB_TEST:.4f}, HistGradientBoosting "
        f"{HIST_TEST:.4f}, CatBoost {CB_TEST:.4f}), and the blend "
        f"({BLEND_TEST:.4f}) does not meaningfully improve on the best single "
        f"model. Every pairwise gap is within the 5-fold CV standard deviation of "
        f"{CV_LGB['std_auroc']:.4f}, so the families are statistically "
        f"indistinguishable. XGBoost is retained as the deployment candidate for "
        f"continuity with the defended original protocol; LightGBM is documented "
        f"as a co-equal alternative. A SHAP-based interpretation layer is "
        f"incorporated so that clinicians can understand how individual "
        f"predictions decompose into clinically meaningful contributions.")

    # ── §2 Goals ──────────────────────────────────────────────────────────
    add_heading(doc, "2. GOALS", level=1)
    add_para(doc,
        "The overall objective of this project is to develop an interpretable, "
        "clinically deployable risk-assessment tool that enables primary-care "
        "providers and case managers to identify Medicare patients who are at "
        "elevated risk of readmission within thirty days after discharge, allowing "
        "them to direct those patients to targeted follow-up interventions such as "
        "early post-discharge telephone check-ins, medication reconciliation "
        "visits, or home-health referrals. Achieving this goal requires three "
        "simultaneous outcomes: discrimination clearly superior to the LACE "
        "clinical index; patient-level explanations that clinicians can act on; "
        "and exclusive reliance on structured data fields already available in "
        "standard EHR systems.")

    # ── §3 Objectives ─────────────────────────────────────────────────────
    add_heading(doc, "3. OBJECTIVES", level=1)
    add_para(doc,
        "Five measurable objectives operationalise the above goal. The first is "
        "to construct a reproducible Medicare cohort from MIMIC-IV v3.1 with a "
        "30-day readmission label derived strictly using pre-discharge "
        "information. The second is to engineer an expressive yet parsimonious "
        "feature set covering the five clinical domains identified by prior "
        "literature as dominant drivers of readmission risk. The third is to "
        "benchmark a range of modelling approaches, including regularised "
        "logistic regression, gradient-boosted trees, and deep neural networks "
        "(MLPs, GRUs, LSTMs, FT-Transformer, and stacked ensembles), under a "
        "strict patient-level train and test split. The fourth is to compare a "
        "single best-performing gradient-boosting model against a blended "
        "four-model ensemble and select the deployment candidate on the joint "
        "axes of discrimination and operational complexity, not on discrimination "
        "alone. The fifth is to produce global and local SHAP explanations that "
        "convert the numerical output of the model into actionable clinical "
        "narratives for discharge teams.")

    # ── §4 Motivation ─────────────────────────────────────────────────────
    add_heading(doc, "4. MOTIVATION", level=1)
    add_para(doc,
        "Unplanned 30-day readmissions impose a substantial burden on the United "
        "States health-care system. Estimates published by the Centers for "
        "Medicare and Medicaid Services place the annual direct cost of Medicare "
        "readmissions at more than twenty-six billion dollars [9], making this "
        "one of the largest discretionary spending categories in the programme. "
        "Clinically, unplanned readmissions are associated with increased "
        "in-hospital mortality, extended functional decline, caregiver strain, "
        "and reduced patient-reported quality of life. Regulation reinforces "
        "these incentives: since 2013, the CMS Hospital Readmissions Reduction "
        "Program (HRRP) has linked a portion of hospital reimbursement to "
        "risk-adjusted readmission performance, creating sustained operational "
        "pressure on discharge-planning teams. Traditional clinical scores such "
        "as LACE and HOSPITAL rely on a small number of static variables and "
        "rarely achieve an AUROC above 0.70 in external validation, leaving "
        "substantial predictive headroom. A richer, interpretable "
        "machine-learning model that operates directly on structured EHR data "
        "closes a concrete gap between the data hospitals already collect and "
        "the discharge decisions they struggle to make consistently.")

    # ── §5 Prior Art & Challenges ────────────────────────────────────────
    add_heading(doc, "5. PRIOR ART & CHALLENGES", level=1)
    add_para(doc,
        "A substantial body of prior work has examined 30-day readmission "
        "prediction. The LACE index proposed by van Walraven and colleagues "
        "(CMAJ, 2010) combines length of stay, acuity, Charlson comorbidity, and "
        "emergency-department utilisation. External validations of LACE place its "
        "AUROC in the 0.55 to 0.70 range. ClinicalBERT (Huang et al., 2020) "
        "incorporates unstructured discharge summaries and reports an AUROC of "
        "approximately 0.714 on MIMIC data. Other researchers have proposed deep "
        "tabular architectures such as FT-Transformer (Gorishniy et al., 2021) "
        "and recurrent models. However, recent benchmarking continues to show "
        "that gradient-boosted trees perform competitively or better than "
        "alternative architectures on moderately sized structured clinical "
        "datasets.")
    add_para(doc,
        "Four challenges shaped the project design. First, severe class "
        "imbalance (approximately 79 percent negative versus 21 percent "
        "positive) disqualifies accuracy as a primary metric and requires "
        "AUROC, average precision, and calibration analysis. Second, patient "
        "overlap across admissions risks information leakage; this was "
        "mitigated by patient-grouped train and test splits on subject_id. "
        "Third, once laboratory, medication, and operational tables were "
        "joined, the feature space expanded combinatorially, requiring a "
        "disciplined selection pass to keep the deployed model stable and "
        "interpretable. Fourth, the absence of social-determinant-of-health "
        "variables and of readmissions that occur at external hospitals places "
        "a principled ceiling on achievable performance with the available "
        "data.")

    # ── §6 Data ──────────────────────────────────────────────────────────
    add_heading(doc, "6. DATA SOURCES & DESCRIPTION", level=1)
    add_heading(doc, "6.1 Dataset source", level=2)
    add_para(doc,
        "The primary data source is MIMIC-IV v3.1, a freely accessible "
        "electronic health record database released by the MIT Laboratory for "
        "Computational Physiology in collaboration with Beth Israel Deaconess "
        "Medical Center (Boston, MA). The full release covers 546,028 inpatient "
        "admissions across 364,627 unique patients admitted between 2008 and "
        "2022, and provides more than 300 raw variables drawn from "
        "administrative, pharmacy, laboratory, microbiology, vital-sign, "
        "procedure, diagnosis, and intensive-care-unit tables.")

    add_heading(doc, "6.2 Cohort definition", level=2)
    add_para(doc,
        "The cohort was restricted to admissions where insurance was recorded "
        "as Medicare, yielding 244,576 index admissions with a 30-day "
        "readmission prevalence of 21.1 percent. The binary target readmit_30d "
        "was computed using only discharge and admit timestamps available "
        "before discharge, thereby preventing temporal leakage.")
    add_figure(doc, FIGS / "eda_class_imbalance.png",
               "Figure 1. Class balance of the 30-day readmission target in the "
               "244,576-admission Medicare cohort.")
    add_para(doc,
        "Figure 1 depicts the count and percentage of readmitted versus "
        "non-readmitted admissions. Only 21.1 percent of admissions "
        "experienced a readmission within 30 days; the remaining 78.9 percent "
        "did not. The imbalance disqualifies accuracy as a headline metric (a "
        "trivial majority classifier would reach 78.9 percent accuracy while "
        "identifying none of the at-risk patients) and motivates AUROC, "
        "average precision, and calibration as primary evaluation metrics.")

    add_heading(doc, "6.3 Discharge-destination stratification", level=2)
    add_para(doc,
        "Discharge destination is one of the strongest univariate signals in "
        "the dataset and a natural target for care-coordination intervention, "
        "so readmission rates were stratified by discharge location prior to "
        "modelling.")
    add_figure(doc, FIGS / "eda_discharge_location.png",
               "Figure 2. Thirty-day readmission rate by discharge destination "
               "(cohort mean shown as a dashed line).")
    add_para(doc,
        "Figure 2 presents readmission rates across the nine discharge "
        "destinations. Destinations span a wide risk spectrum: patients "
        "discharged to hospice readmit at only 3.9 percent, while patients "
        "discharged to psychiatric facilities readmit at 50.3 percent. The "
        "large Home-Health subgroup of approximately 56,000 admissions "
        "readmits at 25.1 percent. The conclusion is that discharge "
        "destination is both an informative predictor and an actionable "
        "clinical hand-off: patients flowing into high-risk downstream "
        "pathways can be identified at the moment of discharge and routed to "
        "enhanced transitional-care programmes.")

    add_para(doc, "Table 1. Progressive feature-engineering summary across "
                  "dataset versions (V1 to V7).",
             italic=True, size=10, space_after=4)
    table1_rows = [
        ("V1", "21", "Demographics, CCI flags, LOS, meds, prior use, DRG",
         f"{PROG['lightgbm']['V1']:.4f}"),
        ("V2", "24", "Prior DRG/disp., med entropy 90d, LOS trend 180d",
         f"{PROG['lightgbm']['V2']:.4f}"),
        ("V3", "24", "Missingness flags; recomputed LOS",
         f"{PROG['lightgbm']['V3']:.4f}"),
        ("V4", "n/a", "Age x CCI, LOS x CCI, buckets (parquet not preserved)",
         "n/a"),
        ("V5", "n/a", "LOS x age, cci sq., log(LOS) (parquet not preserved)",
         "n/a"),
        ("V6", "34", "Lab/med/dx counts, ICU utilisation",
         f"{PROG['lightgbm']['V6']:.4f}"),
        ("V7", "50", "+5 target encodings, +5 clinical interactions (final)",
         f"{LGB_TEST:.4f}"),
        ("Exp.", "368", "Unpruned superset (original; not re-run)", "0.800"),
    ]
    add_table(doc, ["Ver.", "N", "New content added", "AUROC"],
              table1_rows, col_widths=[0.35, 0.30, 1.75, 0.60])
    add_para(doc,
        f"Table 1 summarises the staircase of feature engineering across the "
        f"dataset versions. The two largest marginal AUROC gains arise at V2, "
        f"where temporal and medication-complexity signals are introduced "
        f"(plus {PROG['lightgbm']['V2']-PROG['lightgbm']['V1']:.4f}), and at "
        f"V7, where target encodings and clinical interactions are added "
        f"(plus {PROG['lightgbm']['V7']-PROG['lightgbm']['V6']:.4f}). The V4 "
        f"and V5 intermediate parquet files were not preserved in the "
        f"published dataset snapshot, so they are reported as not available; "
        f"the V1 to V3 and V6 to V7 progressions cover the substantive "
        f"feature-engineering decisions. The original expanded-exploration "
        f"reading of 0.800 AUROC is retained from the original analysis for "
        f"context but was not re-evaluated in this revision (the 368-feature "
        f"parquet was not preserved). The 50-feature V7 set delivers "
        f"essentially all of the achievable discrimination on this cohort, "
        f"confirming V7 as the parsimony-optimal deployment configuration.")

    # ── §7 Methods ────────────────────────────────────────────────────────
    add_heading(doc, "7. METHODS AND TOOLS", level=1)
    add_heading(doc, "7.1 Software stack", level=2)
    add_para(doc,
        "The analytical pipeline was implemented in Python 3.12 with pandas, "
        "NumPy, and PyArrow for parquet-based data manipulation; scikit-learn "
        "for preprocessing, cross-validation, and baseline models; LightGBM, "
        "XGBoost, CatBoost, and HistGradientBoosting for the boosting "
        "ensemble; PyTorch for neural-network experiments; Optuna for Bayesian "
        "hyper-parameter optimisation; SHAP for interpretability; and "
        "matplotlib with seaborn for visualisation. Training scripts are "
        "decoupled from the notebook via subprocess-per-model isolation to "
        "sidestep an XGBoost 3.2.0 thread issue that arises when running "
        "ten-seed sweeps across four GBM families inside a single Jupyter "
        "kernel session.")

    add_heading(doc, "7.2 Train and test protocol", level=2)
    add_para(doc,
        "An 80/20 split was performed on the cohort at the subject_id level "
        "using GroupShuffleSplit, creating two non-overlapping partitions in "
        "which no patient appears in both. The split is identical to the "
        "partition used in the original report (n_train = 195,385; "
        "n_test = 49,191) and provides an unbiased estimate of generalisation "
        "performance. In the revised protocol, a 10 percent inner-validation "
        "slice is then carved from the training partition for early stopping "
        "and blend-weight selection (approximately n_inner_val = 19,115; "
        "n_pure_train = 176,270). The held-out 20 percent test partition is "
        "evaluated exactly once at final reporting and is never referenced by "
        "any model-selection callback. Categorical variables were "
        "target-encoded with five-fold out-of-fold cross-validation to "
        "prevent leakage (CatBoost handles categoricals natively and was "
        "excluded from this step). Missing values were managed directly by "
        "the histogram-based tree learners, which absorb missingness while "
        "preserving the informative missingness signal observed in flags "
        "such as drg_code_is_missing.")

    add_heading(doc, "7.3 Feature engineering progression", level=2)
    add_para(doc,
        f"Feature engineering was executed as a staged process across seven "
        f"dataset versions so that the marginal contribution of each clinical "
        f"domain could be measured. Versions V1 to V6 accumulated "
        f"progressively richer temporal, comorbidity, medication, and "
        f"operational signals. V7 then added target encodings of "
        f"high-cardinality categoricals plus the most informative clinical "
        f"interactions identified during V2 to V6 tuning, and was adopted as "
        f"the final modelling set. Beyond V7, the original analysis also "
        f"conducted an exploratory expansion in which the feature space was "
        f"widened to 368 variables by enumerating additional pairwise "
        f"interactions, extended target encodings, and a broader panel of "
        f"utilisation aggregates. V7 was then derived as the top-50 "
        f"parsimonious subset from this expanded exploration. The expanded "
        f"configuration reached a test AUROC of 0.800 in the original "
        f"analysis, only {0.800 - BLEND_TEST:+.3f} above the revised V7 "
        f"ensemble result of {BLEND_TEST:.4f}, confirming that V7 sits at the "
        f"parsimony-optimal stopping point and that additional feature "
        f"engineering beyond fifty variables yields rapidly diminishing "
        f"returns.")
    add_figure(doc, FIGS / "fig_cde_lgbm_xgb_mlp_v1v6.png",
               "Figure 3. Per-family test AUROC across dataset versions for "
               "LightGBM, XGBoost, and MLP (single-seed under the strict "
               "80/20 plus 10 percent inner-val protocol). V4 and V5 "
               "parquets were not preserved; the curve covers V1 to V3 and "
               "V6 to V7.")
    add_para(doc,
        f"Figure 3 displays the per-version test AUROC for the three primary "
        f"non-CatBoost families. AUROC rises sharply from V1 to V2 once "
        f"temporal and medication signals are introduced (LightGBM "
        f"{PROG['lightgbm']['V1']:.4f} to {PROG['lightgbm']['V2']:.4f}; "
        f"XGBoost {PROG['xgboost']['V1']:.4f} to "
        f"{PROG['xgboost']['V2']:.4f}), holds steady through V3, inches up at "
        f"V6 with laboratory and ICU-utilisation counts, and rises again at "
        f"V7 after the target encodings and clinical interactions are added "
        f"(LightGBM peaks at {PROG['lightgbm']['V7']:.4f}; XGBoost at "
        f"{PROG['xgboost']['V7']:.4f}; MLP at {PROG['mlp']['V7']:.4f}). V7 "
        f"sits at the elbow of the learning curve and, as shown later, the "
        f"blended ensemble does not improve on the best single model under "
        f"the strict protocol.")

    add_heading(doc, "7.4 Model selection: single model versus blended ensemble",
                level=2)
    add_para(doc,
        f"Each of the four gradient-boosting families was trained on V7 "
        f"across ten random seeds, and the inner-validation predictions of "
        f"each family were averaged to reduce variance. A "
        f"scipy.optimize.minimize call (Nelder-Mead) then searched for blend "
        f"weights that minimise the negative inner-validation AUROC, subject "
        f"to non-negativity and sum-to-one constraints. Under the strict "
        f"protocol the optimiser concentrates mass on LightGBM (weight "
        f"{BLEND_W['lightgbm']:.2f}) and clips the other three families "
        f"(XGBoost {BLEND_W['xgboost']:.2f}, CatBoost "
        f"{BLEND_W['catboost']:.2f}, HistGBM {BLEND_W['histgbm']:.2f}) to the "
        f"0.05 lower bound. The blended test AUROC is {BLEND_TEST:.4f}. The "
        f"single XGBoost model scores {XGB_TEST:.4f} on the same test set, "
        f"while LightGBM scores {LGB_TEST:.4f}, CatBoost {CB_TEST:.4f}, and "
        f"HistGBM {HIST_TEST:.4f}. Every pairwise gap is within the 5-fold CV "
        f"standard deviation of {CV_LGB['std_auroc']:.4f}, so the four "
        f"families are statistically indistinguishable on this test "
        f"partition. XGBoost was retained as the deployed model for "
        f"continuity with the defended original protocol, with LightGBM "
        f"noted as a co-equal alternative. The blended ensemble is kept as a "
        f"documented benchmark; it represents the discrimination ceiling "
        f"that an optimiser can extract from the four families on V7 under "
        f"the strict protocol, and it confirms that that ceiling is not "
        f"meaningfully higher than the best single model.")

    # ── §8 Results ────────────────────────────────────────────────────────
    add_heading(doc, "8. RESULTS", level=1)
    add_heading(doc, "8.1 Per-family AUROC progression (V1 to V7)", level=2)
    add_para(doc,
        "Figures 4 to 7 document the test AUROC of each model family across "
        "the available dataset versions, in order to isolate the marginal "
        "benefit of feature enrichment independently of model choice.")
    add_figure(doc, FIGS / "fig_b_logreg_v1v6.png",
               "Figure 4. Logistic regression test AUROC across dataset "
               "versions.")
    add_para(doc,
        f"Figure 4 shows that regularised logistic regression hovers around "
        f"{PROG['logreg']['V1']['test_auroc']:.4f} to "
        f"{PROG['logreg']['V6']['test_auroc']:.4f} across V1 to V6 and then "
        f"jumps to {PROG['logreg']['V7']['test_auroc']:.4f} at V7 once "
        f"target encodings and clinical-interaction features are added. A "
        f"linear model cannot exploit the intermediate non-linear signals "
        f"added in V2 to V6 (it stays essentially flat) but does benefit "
        f"when V7 introduces features that are already non-linearly "
        f"engineered. The linear baseline is still insufficient for this "
        f"problem: it lags the best non-linear learner by approximately "
        f"{LGB_TEST - PROG['logreg']['V7']['test_auroc']:+.3f} AUROC at V7, "
        f"confirming that non-linear learners remain required to convert "
        f"feature enrichment into predictive gain.")

    add_figure(doc, FIGS / "fig_cde_lgbm_xgb_mlp_v1v6.png",
               "Figures 5, 6, 7 (combined panel). LightGBM, XGBoost, and MLP "
               "test AUROC across dataset versions.")
    add_para(doc,
        f"The combined panel reproduces the V1 to V7 trajectory for "
        f"LightGBM, XGBoost, and MLP. LightGBM jumps from "
        f"{PROG['lightgbm']['V1']:.4f} at V1 to {PROG['lightgbm']['V2']:.4f} "
        f"at V2 as temporal and medication-complexity signals are "
        f"introduced, rises again to {PROG['lightgbm']['V6']:.4f} at V6 "
        f"with aggregate clinical counts and ICU utilisation, and reaches "
        f"{PROG['lightgbm']['V7']:.4f} at V7 under single-seed training "
        f"(ten-seed averaging brings it to {LGB_TEST:.4f}). XGBoost follows "
        f"a near-identical trajectory ({PROG['xgboost']['V1']:.4f} to "
        f"{PROG['xgboost']['V7']:.4f} single-seed; {XGB_TEST:.4f} ten-seed), "
        f"confirming that the feature-engineering gains generalise across "
        f"boosting implementations. MLP rises modestly from "
        f"{PROG['mlp']['V1']:.4f} at V1 to {PROG['mlp']['V7']:.4f} at V7: a "
        f"real but bounded benefit that never closes the gap to the GBMs, "
        f"indicating that default MLPs are not competitive with boosting on "
        f"this tabular problem.")

    add_heading(doc, "8.2 Cross-family comparison on V6", level=2)
    add_figure(doc, FIGS / "fig_f_v6_families.png",
               "Figure 8. Test AUROC across model families on the V6 dataset.")
    add_para(doc,
        f"Figure 8 compares model families trained on the V6 dataset. A "
        f"standard multi-layer perceptron lags by roughly five AUROC points, "
        f"while FT-Transformer and hybrid LSTM/GRU architectures from the "
        f"original analysis reach approximately 0.770, and a stacking "
        f"meta-learner reaches 0.778. The revised live LightGBM on V6 "
        f"attains {PROG['lightgbm']['V6']:.4f} without any stacking. "
        f"Gradient boosting dominates on this tabular problem, and deep "
        f"architectures provide at most marginal improvement at substantial "
        f"complexity cost. The deep-architecture numbers in this comparison "
        f"are carried forward from the original experiments and were not "
        f"retrained for this revision.")

    add_heading(doc, "8.3 Final XGBoost performance", level=2)
    add_figure(doc, FIGS / "fig_z_roc_cal.png",
               "Figure 9. Receiver-operating-characteristic, "
               "precision-recall, calibration, and confusion-matrix panels "
               "for the deployed XGBoost model (V7, 10-seed average).")
    add_para(doc,
        f"Figure 9 characterises the deployed XGBoost model through its ROC "
        f"curve (AUROC {XGB_TEST:.4f}), precision-recall curve, reliability "
        f"diagram binned into deciles of predicted probability, and "
        f"confusion matrix at the Youden-J operating threshold. The ROC "
        f"dominates the chance diagonal across the full operating range, "
        f"and the reliability points track the identity line closely in the "
        f"operating region that matters for care-coordination triage. The "
        f"selected XGBoost model delivers both ranking power and "
        f"trustworthy probabilities, which are essential for downstream "
        f"clinical decision support, while imposing only a single-model "
        f"maintenance footprint. LightGBM ({LGB_TEST:.4f}) is documented as "
        f"a co-equal alternative within seed-level variance.")

    add_heading(doc, "8.4 Interpretability via SHAP", level=2)
    add_figure(doc, FIGS / "fig_g_shap7.png",
               "Figure 10. Mean absolute SHAP value of the top seven "
               "predictors in the final V7 XGBoost model.")
    add_para(doc,
        "Figure 10 ranks the most important features by mean absolute SHAP "
        "value. The single strongest predictor is the 180-day "
        "length-of-stay trend, followed by DRG code, late-order rate, "
        "primary-diagnosis chapter (target-encoded), the squared count of "
        "prior six-month admissions, the last-DRG-with-disposition "
        "interaction, and the discharge-location target encoding. "
        "Readmission risk is inherently multi-factorial; no single feature "
        "dominates the prediction; and the top seven features map directly "
        "to interventions a discharge team can act upon. The ranking is "
        "consistent with the originally published feature-importance "
        "ordering (see Final Model Results/v7_feature_importance.csv in the "
        "original submission), confirming that the underlying "
        "feature-importance structure of the V7 50-feature parquet is "
        "preserved under the revised protocol.")

    add_heading(doc, "8.5 Benchmark comparison", level=2)
    add_para(doc, "Table 2. Benchmark comparison against published thirty-day "
                  "readmission models on MIMIC-family data.",
             italic=True, size=10, space_after=4)
    table2_rows = [
        ("van Walraven et al. (2010)", "LACE clinical index", f"{LACE:.3f}"),
        ("Huang et al. (2020)", "ClinicalBERT plus clinical notes",
         f"{CBERT:.3f}"),
        ("Literature baselines", "Single LightGBM / XGBoost", "approx. 0.76"),
        ("This work (V7, deployment candidate)", "XGBoost, 50 features",
         f"{XGB_TEST:.4f}"),
        ("This work (V7, co-equal alternative)", "LightGBM, 50 features",
         f"{LGB_TEST:.4f}"),
        ("This work (V7, ensemble)", "4-GBM scipy-blend (not deployed)",
         f"{BLEND_TEST:.4f}"),
    ]
    add_table(doc, ["Study", "Method", "AUROC"], table2_rows,
              col_widths=[1.0, 1.3, 0.6])
    add_para(doc,
        f"Table 2 situates the deployed model against published baselines. "
        f"The V7 XGBoost model improves over LACE by {XGB_TEST-LACE:+.3f} "
        f"AUROC (approximately a {(XGB_TEST-LACE)/LACE*100:.0f} percent "
        f"relative gain) and over ClinicalBERT by {XGB_TEST-CBERT:+.3f} "
        f"AUROC, while using only fifty structured features and no "
        f"free-text clinical notes. The 4-GBM scipy-optimised blend trails "
        f"the best single model by {BLEND_TEST-LGB_TEST:+.4f} AUROC and "
        f"was therefore not deployed, a clean reversal of the original "
        f"blend-favouring framing once test-set early-stopping leakage is "
        f"removed. LightGBM is reported alongside as a co-equal alternative "
        f"within seed-level variance. The deployed XGBoost model matches or "
        f"exceeds the state of the art on MIMIC-family data while remaining "
        f"simple to deploy from any existing EHR back end.")

    add_heading(doc, "8.6 Five-fold cross-validation stability check", level=2)
    add_para(doc,
        f"To enable an apples-to-apples comparison against the original "
        f"report's published stability estimate of "
        f"{REF['stability_mean_auroc']:.4f} plus or minus "
        f"{REF['stability_std_auroc']:.4f}, a 5-fold patient-grouped "
        f"cross-validation was executed under the same strict no-leakage "
        f"protocol used for the primary results. Each fold's test partition "
        f"is held out from that fold's training data, and a 10 percent "
        f"inner-validation slice is carved from each fold's training "
        f"portion for early stopping.")
    add_figure(doc, FIGS / "cv5_stability.png",
               "Figure 11. Five-fold patient-grouped cross-validation "
               "stability for each GBM family and the scipy-optimised "
               "blend. Dots are per-fold test AUROCs; error bars span plus "
               "or minus one standard deviation; the grey band is the "
               "originally published stability range (0.7956 plus or minus "
               "0.0026).")
    add_para(doc,
        f"Figure 11 plots per-fold AUROCs for each GBM family and the "
        f"blend, overlaid on the originally published stability band. "
        f"Three of the four single GBM families (XGBoost "
        f"{CV_XGB['mean_auroc']:.4f} plus or minus "
        f"{CV_XGB['std_auroc']:.4f}; CatBoost {CV_CB['mean_auroc']:.4f} "
        f"plus or minus {CV_CB['std_auroc']:.4f}; HistGBM "
        f"{CV_HGB['mean_auroc']:.4f} plus or minus "
        f"{CV_HGB['std_auroc']:.4f}) land squarely within the originally "
        f"published band. LightGBM ({CV_LGB['mean_auroc']:.4f} plus or "
        f"minus {CV_LGB['std_auroc']:.4f}) and the blend "
        f"({CV_BLEND['mean_auroc']:.4f} plus or minus "
        f"{CV_BLEND['std_auroc']:.4f}) exceed the published mean by "
        f"approximately 0.004 AUROC while preserving an essentially "
        f"identical fold-to-fold standard deviation. The revised pipeline "
        f"reproduces the published stability profile under the stricter "
        f"protocol: the originally reported numbers were not driven by "
        f"accidental test-set contamination, but rather by the same "
        f"underlying signal that the revised protocol now estimates "
        f"honestly.")

    # ── §9 Discussion ────────────────────────────────────────────────────
    add_heading(doc, "9. DISCUSSION", level=1)
    add_heading(doc, "9.1 Clinical implications", level=2)
    add_para(doc,
        "Rather than interpret every SHAP variable in Figure 10, we focus "
        "on the four features whose SHAP magnitude and clinical "
        "actionability are jointly greatest: the 180-day length-of-stay "
        "trend, the DRG code, the squared count of prior six-month "
        "admissions, and the current discharge location. These four "
        "variables sit at decision points that a discharge team can "
        "actually control, and taken together they explain the majority "
        "of the ranked risk.")
    add_para(doc,
        "The single strongest driver is the 180-day length-of-stay trend, "
        "which is best read as a compressed biomarker of disease "
        "trajectory rather than an independent cause. When a patient's "
        "recent admissions become progressively longer, the underlying "
        "illness is decompensating: stabilisation takes longer, functional "
        "reserve is lower at separation, and the interval between "
        "hospitalisations is shortening. A rising 180-day trend should "
        "prompt a pre-discharge geriatric or palliative-care consult for "
        "patients above the cohort 90th percentile and proactive "
        "escalation to complex-care management before a fourth admission "
        "accrues. The principal caveat is that the feature is partly "
        "shaped by system factors rather than biology: it runs high for "
        "socially isolated patients whose discharge is delayed for "
        "logistical reasons, so treating it as a purely clinical signal "
        "risks conflating social need with medical severity.")
    add_para(doc,
        "The second-ranked driver is the DRG code, which simultaneously "
        "encodes the reason for admission and the CMS complexity "
        "weighting assigned to it. Certain DRGs (heart failure, COPD "
        "exacerbation, septicaemia, acute kidney injury) are clinically "
        "known to generate frequent 30-day readmissions, while others "
        "(elective joint replacement, most obstetric DRGs) rarely do. DRG "
        "therefore functions as a proxy for the care pathway the patient "
        "is already on. A high-risk DRG at discharge should trigger "
        "disease-specific bundles (the HFSA heart-failure bundle of "
        "weight monitoring, diuretic titration, and seven-day follow-up "
        "is representative) and automated pharmacist-led medication "
        "reconciliation. The caveat is that DRG is a billing construct as "
        "well as a clinical one; coding conventions and up-coding inflate "
        "or deflate the signal in ways the model cannot distinguish from "
        "true severity.")
    add_para(doc,
        "The squared prior-admission count in the preceding six months "
        "captures a non-linear dose-response that the literature has "
        "documented repeatedly: the risk increment between zero and one "
        "prior admission is moderate, while the increment between three "
        "and four is large. This feature integrates over undertreated "
        "chronic disease, fragile social circumstances, and "
        "outpatient-access barriers, and projects them forward as the "
        "best available estimate of next-month behaviour. High values "
        "identify patients for whom standard discharge planning has "
        "already demonstrably failed; they should enter an intensive "
        "case-management or community-paramedicine programme, with an "
        "ambulatory complex-care appointment and integrated social-work "
        "review scheduled within seven days. The most consequential "
        "caveat in the entire ranking is equity: high prior utilisation "
        "disproportionately reflects poor outpatient access, unstable "
        "housing, and insurance gaps, so a score that weights this "
        "feature heavily will assign higher risk to exactly the "
        "subgroups whose elevated risk is socially rather than "
        "biologically driven. Pairing the score with additional "
        "resources, rather than with reduced downstream care, is the "
        "only deployment posture that converts the prediction into an "
        "equitable intervention.")
    add_para(doc,
        "Current discharge location is the most directly actionable "
        "feature because it sits at the moment the care team still "
        "controls. Each destination implies a different monitoring "
        "intensity, rehabilitation capacity, and re-hospitalisation "
        "threshold: hospice discharge readmits at 3.9 percent because the "
        "terminal-care framework actively avoids admission; "
        "psychiatric-facility discharge readmits at 50.3 percent because "
        "those patients carry severe medical comorbidity and transfer "
        "rules that route decompensation back to the index hospital. A "
        "high-risk destination should trigger a warm hand-off (structured "
        "clinician-to-clinician report) together with destination-specific "
        "medication-reconciliation templates. The principal caveat is "
        "that discharge location is partly endogenous to bed availability "
        "and payer rules: the model cannot distinguish between home was "
        "clinically appropriate and home because no post-acute bed was "
        "available, and this distinction matters when the score is used "
        "to adjudicate individual cases.")
    add_para(doc,
        "Read together, these four features describe a prediction problem "
        "whose signal lives at the interface between the patient and the "
        "care system rather than inside the patient alone. LOS trend and "
        "prior-admission count track the patient's clinical trajectory; "
        "DRG and discharge location track the system's behaviour around "
        "that trajectory: how the episode was coded and where the patient "
        "was next routed. The model is therefore not predicting who will "
        "become sicker in isolation; it is predicting whose combination "
        "of clinical trajectory and care hand-off is fragile. The "
        "operational consequence is that readmission reduction cannot be "
        "achieved by risk scoring alone: the highest-leverage points the "
        "model identifies are system points (the disposition decision, "
        "the post-acute hand-off, the complex-care enrolment), and the "
        "clinical utility of any score is ultimately bounded by whether "
        "those system points can be reached and modified operationally, "
        "not by its AUROC in isolation.")

    add_heading(doc, "9.2 Tabular boosting versus deep learning", level=2)
    add_para(doc,
        "A secondary finding is that gradient boosting dominates deep "
        "tabular architectures on this problem. This reinforces recent "
        "benchmarking evidence and has practical consequences: hospitals "
        "can obtain state-of-the-art performance without standing up GPU "
        "infrastructure, specialised MLOps tooling, or "
        "deep-learning-specific governance, all of which add cost and "
        "friction to clinical deployment.")

    add_heading(doc, "9.3 Limitations", level=2)
    add_para(doc,
        "Several limitations temper these results. The cohort originates "
        "from a single academic medical centre and may not generalise "
        "without recalibration to community hospitals, rural settings, or "
        "non-US health systems. MIMIC-IV lacks social-determinant-of-health "
        "variables such as housing stability, health literacy, and "
        "transportation access, all of which are known to influence "
        "post-discharge outcomes. Readmissions that occur at external "
        "hospitals are not captured, biasing the target label toward "
        "within-system utilisation. Finally, the model is a "
        "risk-stratification tool rather than an intervention; "
        "translating predicted probabilities into reduced readmissions "
        "requires prospective evaluation of the paired care-coordination "
        "workflow.")

    # ── §10 Contributions ────────────────────────────────────────────────
    add_heading(doc, "10. CONTRIBUTIONS & CONCLUSIONS", level=1)
    add_para(doc,
        f"This work contributes four artefacts. First, a reproducible "
        f"MIMIC-IV Medicare cohort definition together with a staged V1 to "
        f"V7 feature-engineering recipe that other researchers can extend. "
        f"Second, empirical evidence under a strict no-leakage train, "
        f"validation, and test protocol that a single-model XGBoost trained "
        f"on the 50-feature parsimonious V7 set matches or surpasses both "
        f"traditional clinical scores and notes-based deep-learning "
        f"pipelines while remaining fully interpretable, with LightGBM a "
        f"co-equal single-model alternative within seed-level variance; the "
        f"scipy-optimised 4-GBM blend ({BLEND_TEST:.4f}) does not improve "
        f"on the best single model, so the additional operational overhead "
        f"of a blended ensemble is unjustified. Third, a SHAP-based "
        f"explanation layer that converts individual predictions into "
        f"ranked lists of contributing factors, providing a bridge between "
        f"the machine-learning output and the clinician's decision. Fourth, "
        f"a 5-fold patient-grouped cross-validation stability profile "
        f"({CV_LGB['mean_auroc']:.4f} plus or minus "
        f"{CV_LGB['std_auroc']:.4f} for LightGBM) that matches the "
        f"originally published stability band under the stricter protocol, "
        f"confirming that the headline result is not the product of a "
        f"single fortunate split.")
    add_para(doc,
        f"In conclusion, a single XGBoost model trained on the 50-feature "
        f"parsimonious V7 set from MIMIC-IV v3.1 predicts 30-day all-cause "
        f"readmission in Medicare patients with a test AUROC of "
        f"{XGB_TEST:.4f} under a strict 80/20 patient-grouped split with a "
        f"10 percent inner-validation slice carved from the training "
        f"portion for early stopping. This represents a "
        f"{XGB_TEST-LACE:+.3f}-point improvement over LACE and a "
        f"{XGB_TEST-CBERT:+.3f}-point improvement over ClinicalBERT. "
        f"LightGBM ({LGB_TEST:.4f}) is reported as a co-equal alternative "
        f"within seed-level variance. The model is interpretable at both "
        f"global and individual levels via SHAP, well calibrated in the "
        f"operating range that matters for care-coordination triage, and "
        f"simple enough to be embedded in an existing EHR without "
        f"additional infrastructure.")

    # ── Acknowledgments ──────────────────────────────────────────────────
    add_heading(doc, "ACKNOWLEDGMENTS", level=1)
    add_para(doc,
        "This work was completed as the capstone project for the Master of "
        "Science in Data Science and Artificial Intelligence at Florida "
        "International University under the mentorship of Dr. Christian "
        "Poellabauer, who joins as senior author for the publication scope. "
        "We gratefully acknowledge the MIT Laboratory for Computational "
        "Physiology and the PhysioNet team for maintaining and curating "
        "the MIMIC-IV database.")

    # ── Revision note (replaces the front-matter §0 banner from prior draft) ──
    add_heading(doc, "REVISION NOTE", level=1)
    add_para(doc,
        f"This document is a post-defense revision of the report originally "
        f"submitted in April 2026. The original report is preserved "
        f"unchanged. The revision corrects one methodological issue: the "
        f"original training pipeline used the held-out test set as the "
        f"early-stopping signal for the gradient-boosting models, which "
        f"allowed test labels to influence model selection. The revised "
        f"pipeline carves a 10 percent inner-validation slice from the 80 "
        f"percent training partition for early stopping, and the held-out "
        f"20 percent test partition is evaluated exactly once at final "
        f"reporting. Under the corrected protocol, XGBoost (test AUROC "
        f"{XGB_TEST:.4f}) is retained as the deployment candidate for "
        f"continuity with the defended original protocol, and LightGBM "
        f"({LGB_TEST:.4f}) is documented as a co-equal alternative within "
        f"seed-level variance. The scipy-optimised four-GBM blend "
        f"({BLEND_TEST:.4f}) does not improve on the best single model "
        f"under the strict protocol. The Medicare cohort definition, the "
        f"V1 to V7 staged feature engineering, the SHAP interpretability "
        f"layer, the clinical interpretation, the limitations, and the "
        f"conclusions are all carried forward without modification. All "
        f"revised numbers are reproducible from the publication notebook "
        f"at GitHub commit 5c64fea of "
        f"thiagobandeira1/medicare-30day-readmission-mimic-iv. A full "
        f"change log appears in Appendix A.")

    # ── References ───────────────────────────────────────────────────────
    add_heading(doc, "REFERENCES", level=1)
    refs = [
        "Johnson, A.E.W., et al. (2023). MIMIC-IV, a freely accessible "
        "electronic health record dataset. Scientific Data, 10, 1.",
        "van Walraven, C., et al. (2010). Derivation and validation of an "
        "index to predict early death or unplanned readmission after "
        "discharge (LACE). CMAJ, 182(6), 551 to 557.",
        "Huang, K., Altosaar, J., & Ranganath, R. (2020). ClinicalBERT: "
        "Modeling clinical notes and predicting hospital readmission. "
        "arXiv:1904.05342.",
        "Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). "
        "Revisiting deep learning models for tabular data. NeurIPS.",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting "
        "system. KDD 2016, 785 to 794.",
        "Ke, G., et al. (2017). LightGBM: A highly efficient gradient "
        "boosting decision tree. NeurIPS.",
        "Prokhorenkova, L., et al. (2018). CatBoost: Unbiased boosting "
        "with categorical features. NeurIPS.",
        "Lundberg, S., & Lee, S.-I. (2017). A unified approach to "
        "interpreting model predictions (SHAP). NeurIPS.",
        "Centers for Medicare and Medicaid Services. (2024). Hospital "
        "Readmissions Reduction Program (HRRP).",
        "Charlson, M.E., et al. (1987). A new method of classifying "
        "prognostic comorbidity in longitudinal studies. Journal of "
        "Chronic Diseases, 40(5), 373 to 383.",
    ]
    for i, ref in enumerate(refs, 1):
        add_para(doc, f"[{i}] {ref}", size=10, space_after=4)

    # ── Appendix A: Change log (switch back to single column for the wide table) ──
    switch_columns(doc, 1)
    doc.add_page_break()
    add_heading(doc, "APPENDIX A. CHANGE LOG", level=1)
    add_para(doc,
        "Every numeric and language change between the original April 2026 "
        "report and this May 2026 revision is enumerated below. Section "
        "numbers refer to the revised report. The Original values are "
        "quoted from Capstone_Final_Report.docx; the Revised values are "
        "reproduced by the publication notebook at GitHub commit 5c64fea.")

    add_para(doc, "A.1 Numerical changes", bold=True, size=11, space_after=6)
    change_rows = [
        ("Abstract", "blend 0.795 / XGBoost 0.793",
         f"XGBoost {XGB_TEST:.4f} (deployed) / LightGBM {LGB_TEST:.4f} "
         f"(co-equal) / blend {BLEND_TEST:.4f}",
         "Co-equal framing added; blend no longer improves on best single"),
        ("Abstract", "LACE +0.109, CBERT +0.079",
         f"LACE {XGB_TEST-LACE:+.3f}, CBERT {XGB_TEST-CBERT:+.3f}",
         "Recalculated for deployed model"),
        ("Section 1", "single XGBoost within 0.002 of blend",
         f"all four GBMs and blend within {CV_LGB['std_auroc']:.4f}; XGBoost "
         f"retained for continuity",
         "Framing updated under strict protocol"),
        ("Section 1", "stable across 5 CV folds (no number)",
         f"5-fold CV: {CV_LGB['mean_auroc']:.4f} plus or minus "
         f"{CV_LGB['std_auroc']:.4f}",
         "Explicit stability number now provided (see section 8.6)"),
        ("Section 7.1", "Python 3.10", "Python 3.12", "Runtime upgrade"),
        ("Section 7.2", "no inner-val mention",
         "10 percent inner-val carved from train for early stopping",
         "Key methodological change: no test-set leakage"),
        ("Section 7.3", "V7 ensemble ceiling 0.795",
         f"V7 ensemble {BLEND_TEST:.4f}",
         "Recomputed under strict protocol"),
        ("Section 7.4", "blend (0.19, 0.30, 0.30, 0.21) -> 0.795",
         f"blend ({BLEND_W['lightgbm']:.2f}, {BLEND_W['xgboost']:.2f}, "
         f"{BLEND_W['catboost']:.2f}, {BLEND_W['histgbm']:.2f}) -> "
         f"{BLEND_TEST:.4f}",
         "Optimiser concentrates on LightGBM; XGBoost retained for "
         "continuity"),
        ("Section 8.1 LogReg", "peaks at V1 0.6994, declines through V6",
         f"V1 {PROG['logreg']['V1']['test_auroc']:.4f} to V7 "
         f"{PROG['logreg']['V7']['test_auroc']:.4f}",
         "Story changes: flat then jump at V7, not flat then decline"),
        ("Section 8.1 LightGBM", "V1 0.709 to V6 0.769",
         f"V1 {PROG['lightgbm']['V1']:.4f} to V6 "
         f"{PROG['lightgbm']['V6']:.4f}", "Recomputed"),
        ("Section 8.1 XGBoost", "V1 0.707 to V6 0.771",
         f"V1 {PROG['xgboost']['V1']:.4f} to V6 "
         f"{PROG['xgboost']['V6']:.4f}", "Recomputed"),
        ("Section 8.1 MLP", "approx. 0.700 across all versions",
         f"V1 {PROG['mlp']['V1']:.4f} to V7 {PROG['mlp']['V7']:.4f}",
         "MLP does benefit from later versions"),
        ("Section 8.3", "XGBoost AUROC 0.793",
         f"XGBoost AUROC {XGB_TEST:.4f}",
         "Recomputed under strict protocol"),
        ("Table 1 V7", "0.793",
         f"{PROG['lightgbm']['V7']:.4f} (single) / {LGB_TEST:.4f} (10-seed)",
         "Recomputed under strict protocol"),
        ("Table 2 deployed", "XGBoost 50 feat 0.793",
         f"XGBoost 50 feat {XGB_TEST:.4f}", "Recomputed"),
        ("Table 2 ensemble", "blend 0.795", f"blend {BLEND_TEST:.4f}",
         "Recomputed"),
        ("Section 8.6", "(section did not exist)",
         f"5-fold CV stability check added; LightGBM "
         f"{CV_LGB['mean_auroc']:.4f} plus or minus "
         f"{CV_LGB['std_auroc']:.4f}",
         "New empirical evidence for the stability claim"),
        ("Section 10 contribution 2",
         "single XGBoost 0.793, blend +0.002",
         f"single XGBoost {XGB_TEST:.4f} (deployed) and LightGBM "
         f"{LGB_TEST:.4f} (co-equal); blend trails best single",
         "Reversed: blend now redundant"),
        ("Section 10 conclusion",
         "AUROC 0.793, +0.109 LACE, +0.079 CBERT",
         f"AUROC {XGB_TEST:.4f}, {XGB_TEST-LACE:+.3f} LACE, "
         f"{XGB_TEST-CBERT:+.3f} CBERT",
         "Updated for strict protocol"),
    ]
    add_table(doc, ["Location", "Original value", "Revised value", "Reason"],
              change_rows, col_widths=[1.0, 2.2, 2.2, 1.1])

    add_para(doc, "A.2 Language and framing changes", bold=True, size=11,
             space_after=6)
    add_para(doc,
        "(a) Model selection language now describes XGBoost (the deployed "
        "model retained for continuity with the defended original "
        "protocol) and LightGBM (a co-equal alternative within seed-level "
        "variance) instead of presenting XGBoost as the unique best "
        "performer. The deployment recommendation, namely use a single "
        "model rather than the blend, is preserved.")
    add_para(doc,
        "(b) The blend framing is reversed. The original report described "
        "the blend as exceeding the single model by 0.002 AUROC (framed "
        "as not worth the overhead). The revised report describes the "
        "blend as not improving on the best single model under the strict "
        "protocol (framed as redundant). The deployment outcome (use a "
        "single model) is the same.")
    add_para(doc,
        "(c) Section 7.2 acquired an explicit sentence about the 10 "
        "percent inner-validation slice carved from training for early "
        "stopping. The held-out 20 percent test partition is described as "
        "evaluated exactly once.")
    add_para(doc,
        "(d) The Table 1 entries for V4 and V5 are now marked n/a because "
        "the V4 and V5 training-table parquets were not preserved in the "
        "published Dataset/mimic-parquet snapshot.")
    add_para(doc,
        "(e) Authorship updated: Dr. Christian Poellabauer moves from the "
        "mentor byline to third co-author for the publication scope. "
        "Acknowledgments updated accordingly.")

    add_para(doc, "A.3 Reproducibility metadata", bold=True, size=11,
             space_after=6)
    add_para(doc,
        "All revised numbers are produced by the publication notebook at "
        "GitHub commit 5c64fea of "
        "thiagobandeira1/medicare-30day-readmission-mimic-iv. The "
        "reproduction validator in notebook section 10.3b verifies that "
        "each per-model test AUROC meets or exceeds the corresponding "
        "originally published target. Python 3.12.10; key libraries: "
        "LightGBM 4.6.0, XGBoost 3.2.0, CatBoost 1.2.10, scikit-learn "
        "1.8.0, SHAP 0.51.0. The complete environment is pinned in "
        "environment.yml and requirements.txt in the repo root.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print(f"Size: {OUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()

"""Build the v5 JMIR AI manuscript from corrected artifacts - shared core.

v5 (2026-08-12 external audit): all final-model numbers come from results_v5
(leak-free pool, development-only nested selection). Every number is read from
JSON artifacts at build time - nothing is typed in by hand.
"""
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_reanalysis"
FIG = REPO / "figures_reanalysis"

# ---------------------------------------------------------------- artifacts
J = lambda n: json.loads((OUT / n).read_text())


def J_pref(v3, v2):
    """Prefer the dod-aware v3 artifact when present."""
    return J(v3) if (OUT / v3).exists() else J(v2)


B2 = J("binary_v2.json")
CT = J("cohort_v2_counts.json")
SV = J_pref("survival_v3.json", "survival_v2.json")
FD = J("fairness_dca_v2.json")
BL = J("baselines_v2.json")
DG = J("drg_sensitivity_v2.json")
DOD_AWARE = (OUT / "survival_v3.json").exists()
RB = J("robustness_v3.json") if (OUT / "robustness_v3.json").exists() else None

# ---- canonical artifacts: v6 = v5 final model + v6 corrections/additions ---
OUT5 = REPO / "results_v5"
OUT6 = REPO / "results_v6"
FIG5 = REPO / "figures_v5"
FM = json.loads((OUT6 / "final_model_v6.json").read_text())
EX = FM["v6_extras"]
CMP5 = json.loads((OUT5 / "model_comparison.json").read_text())
BD5 = json.loads((OUT5 / "boundary_v5.json").read_text())
ST5 = json.loads((OUT5 / "stage_v5_shares.json").read_text())
SD5 = json.loads((OUT5 / "stage_differential_v5.json").read_text())
POOL5 = json.loads((OUT5 / "eligible_pool_v5.json").read_text())


def fmt_p(p):
    """Empirical bootstrap P with plus-one correction, JMIR style."""
    return "P<.001" if p < 0.001 else f"P={p:.3f}".replace("0.", ".")


_NBSP = " "      # non-breaking space
_NBHY = "‑"      # non-breaking hyphen (keeps a signed number intact)


def nbci(lo, hi, d=4):
    """A '95% CI x to y' rendered as an unbreakable unit so it never splits
    across a line/column boundary (JMIR-style typesetting)."""
    def f(x):
        # U+2212 true minus: correct glyph and not a break opportunity
        return f"{x:.{d}f}".replace("-", "−")
    return f"95%{_NBSP}CI{_NBSP}{f(lo)}{_NBSP}to{_NBSP}{f(hi)}"


import re as _re


def _style(t):
    """AMA-style typography at build time (audit items S1/S3/S7): spaced
    hyphens become unspaced em dashes, apostrophes curl, negative numbers get
    a true minus sign. Digit-hyphen-digit ranges are untouched."""
    t = t.replace(" - ", "—")
    t = t.replace("'", "’")
    t = _re.sub(r"(?<![\w−—-])-(?=\d)", "−", t)
    return t


PAIRED = CMP5["_paired_differences_vs_rfe"]
STAB = CMP5["_stability"]
SENS = FM["sensitivity"]
SH = FM["shap"]
MP = FM["metric_panel"]
NF = FM["n_features"]
N_UNANIMOUS = sum(1 for v in STAB["selection_frequency"].values() if v == 5)

M = B2["metrics"]


def fmt(x, d=4):
    return f"{x:.{d}f}"


def ci(entry, d=4):
    lo, hi = entry["ci95_cluster"] if "ci95_cluster" in entry else entry
    return f"{lo:.{d}f}-{hi:.{d}f}"


# human-readable labels for engineered feature names (used in generated text)
FEATURE_LABELS = {
    "los_trend_180d": "the 180-day length-of-stay trend",
    "los_trend_x_prior_admits": "its interaction with prior admissions",
    "los_trend_x_prior_6m": "the length-of-stay-trend by 6-month-admissions interaction",
    "freq_x_recency": "admission frequency by recency",
    "lab_abnormal_rate": "the abnormal-laboratory rate",
    "prior_admissions_6m": "prior 6-month admissions",
    "prior_admissions_all": "lifetime prior admissions",
    "log_prior_readmit_count": "the prior-readmission count",
    "prior_readmission_count": "the prior-readmission count",
    "prior_mean_los_6m": "mean prior length of stay",
    "time_since_last_discharge": "time since last discharge",
    "log_time_since_discharge": "time since last discharge (log)",
    "n_late_orders": "the count of late orders",
    "orders_last_6h": "orders in the final 6 hours",
    "n_order_types": "order-type diversity",
    "n_meds_total": "total medications",
    "n_discharge_drugs": "discharge medications",
    "sodium_last": "the last sodium",
    "bilirubin_max": "maximum bilirubin",
    "albumin_x_los": "the albumin by length-of-stay interaction",
    "hemoglobin_last": "the last hemoglobin",
    "discharge_location": "discharge destination",
    "discharge_surge": "order surge near discharge",
    "admission_type": "admission type",
    "los_days": "index length of stay",
    "log_los_days": "index length of stay (log)",
    "los_per_prior_admit": "length of stay per prior admission",
    "log_n_labs": "laboratory-test volume (log)",
    "n_labs_total": "laboratory-test volume",
    "is_readmission_90d": "whether the index stay was itself a 90-day readmission",
    "ventilator_flag": "documented ventilation",
    "age_at_admit": "age",
    "provider_fragmentation_x_los": "provider fragmentation by length of stay",
    "antibiotic_flag": "antibiotic administration",
    "n_distinct_routes": "distinct medication routes",
    "wbc_last": "the last white-cell count",
    "creatinine_last": "the last creatinine",
    "glucose_last": "the last glucose",
    "bun_creatinine_ratio": "the BUN/creatinine ratio",
}


def flabel(f):
    return FEATURE_LABELS.get(f, f.replace("_", " "))


# ---------------------------------------------------------------- document
doc = Document()
S = doc.styles
S["Normal"].font.name = "Times New Roman"
S["Normal"].font.size = Pt(10)
pf = S["Normal"].paragraph_format
pf.alignment = WD_ALIGN_PARAGRAPH.LEFT   # single-column submission format
pf.space_after = Pt(6)
pf.space_before = Pt(0)
pf.line_spacing = 1.5
for h, sz in (("Heading 1", 12), ("Heading 2", 11), ("Heading 3", 10)):
    S[h].font.name = "Times New Roman"
    S[h].font.size = Pt(sz)
    S[h].font.bold = True
    S[h].font.color.rgb = RGBColor(0, 0, 0)
    S[h].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    S[h].paragraph_format.space_before = Pt(8)
    S[h].paragraph_format.space_after = Pt(3)

sec0 = doc.sections[0]
for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec0, attr, Inches(0.85))


def _add_line_numbers(section):
    """Continuous line numbering (reviewers prefer it, per JMIR)."""
    from docx.oxml.ns import qn
    sectPr = section._sectPr
    ln = sectPr.find(qn("w:lnNumType"))
    if ln is None:
        ln = sectPr.makeelement(qn("w:lnNumType"), {})
        sectPr.append(ln)
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")


_add_line_numbers(sec0)


def _add_section(ncols):
    from docx.enum.section import WD_SECTION
    from docx.oxml.ns import qn
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, attr, Inches(0.85))
    sectPr = sec._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = sectPr.makeelement(qn("w:cols"), {})
        sectPr.append(cols)
    cols.set(qn("w:num"), str(ncols))
    cols.set(qn("w:space"), "288")
    return sec


def begin_two_columns():
    return None  # single-column submission: no section switch


def begin_full_width():
    """Wide tables: switch to a one-column continuous section."""
    return None  # already single-column


def H1(t): doc.add_paragraph(_style(t), style="Heading 1")
def H2(t): doc.add_paragraph(_style(t), style="Heading 2")


def P(t, italic=False, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix); r.bold = True
    r = p.add_run(_style(t))
    r.italic = italic
    return p


def caption(t, keep_with_next=False):
    p = doc.add_paragraph()
    r = p.add_run(_style(t)); r.italic = True; r.font.size = Pt(10)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if keep_with_next:
        p.paragraph_format.keep_with_next = True
        # keep the caption's own lines on one page too, so a multi-line
        # caption can never split across a page boundary
        p.paragraph_format.keep_together = True
    return p


def figure(path, caption_text, width=3.25):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    caption(caption_text)


def table(rows, header=True, widths=None, font_size=9.5):
    from docx.oxml.ns import qn
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            para = c.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT   # no justify-stretch
            r = para.add_run(_style(str(cell)))
            r.font.size = Pt(font_size)
            r.font.name = "Times New Roman"
            if header and i == 0:
                r.bold = True
    # pagination behavior: repeat header row on page breaks; never split a
    # row across pages/columns
    for i, trow in enumerate(t.rows):
        trPr = trow._tr.get_or_add_trPr()
        if header and i == 0:
            th = trPr.makeelement(qn("w:tblHeader"), {})
            trPr.append(th)
        cs = trPr.makeelement(qn("w:cantSplit"), {})
        trPr.append(cs)
    if widths:
        t.autofit = False
        for j, w in enumerate(widths):
            for i in range(len(rows)):
                t.cell(i, j).width = Inches(w)
    return t

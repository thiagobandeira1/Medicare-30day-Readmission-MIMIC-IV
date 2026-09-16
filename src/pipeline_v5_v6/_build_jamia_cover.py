# -*- coding: utf-8 -*-
"""Build the JAMIA cover letter draft (no dash punctuation)."""
from pathlib import Path
from docx import Document
from docx.shared import Pt

PUB = Path(__file__).resolve().parent.parent
doc = Document()
S = doc.styles
S["Normal"].font.name = "Times New Roman"
S["Normal"].font.size = Pt(11)
S["Normal"].paragraph_format.space_after = Pt(8)

FORBID = ("—", "–", " - ")


def P(t):
    for f in FORBID:
        assert f not in t, f"dash punctuation in cover letter: {t[:80]!r}"
    doc.add_paragraph(t.replace("'", "’"))


P("Dear Editors,")
P("We are pleased to submit our manuscript, “Predicting 30-Day Hospital "
  "Readmission at Discharge: A Leakage-Safe, Calibrated Electronic Health "
  "Record Model in Medicare-Insured Adults,” for consideration as a "
  "Research and Applications article in the Journal of the American Medical "
  "Informatics Association.")
P("Thirty-day readmission scores are consulted at the moment of discharge, "
  "yet most published models are evaluated in ways that overstate what a "
  "deployed model would deliver. In 236,906 Medicare-insured admissions from "
  "MIMIC-IV v3.1, we develop and internally validate a 31-feature gradient "
  "boosting model under a validation design built for deployability: every "
  "one of 207 candidate predictors was audited for availability at the "
  "moment of discharge (excluding billing-derived features), all "
  "preprocessing and feature selection ran fold-locally inside "
  "patient-grouped cross-validation, and the prespecified primary estimate "
  "is the out-of-fold performance of the complete selection procedure "
  "(AUROC 0.7748) rather than a single exposed model. Four aspects may "
  "interest JAMIA readers: the leakage-safe design itself, including an "
  "open account of residual optimism; calibrated absolute risks compared "
  "against LACE and HOSPITAL reconstructed on the identical cohort; a "
  "readmission-timing analysis showing discrimination is highest on the "
  "first day after discharge, exactly when transitional interventions act; "
  "and a subgroup fairness audit that quantifies, and reports as an open "
  "limitation, lower discrimination for Black patients despite race not "
  "being a model input. Every number in the manuscript is generated at "
  "build time from a released machine-readable results bundle, and the "
  "complete pipeline is public.")
P("This manuscript is not under consideration by any other journal, has "
  "not been published previously in any form, and has not been reviewed "
  "previously by any journal; this is its first submission. The authors "
  "have no related papers published or under consideration. The study is a "
  "secondary analysis of the deidentified, publicly available MIMIC-IV "
  "v3.1 database, accessed under the PhysioNet Credentialed Health Data "
  "Use Agreement; it did not require institutional review board review at "
  "Florida International University. All authors have approved the "
  "manuscript, declare no competing interests, and report no external "
  "funding. The work began as the capstone project for the Master of "
  "Science in Data Science and Artificial Intelligence at Florida "
  "International University.")
P("Thank you for your consideration.")
P("Sincerely,")
P("Thiago Bandeira, MS (corresponding author)")
P("On behalf of: Armando Gonzalez, MS; Christian Poellabauer, PhD; Ananda "
  "Mohan Mondal, PhD")
P("Knight Foundation School of Computing and Information Sciences, Florida "
  "International University, Miami, FL, United States")
P("21324 NE 2nd Ct, Miami, FL 33179, United States; Phone: (815) 603-3286; "
  "Email: tbati006@fiu.edu")

DST = PUB / "JAMIA Cover Letter DRAFT.docx"
doc.save(str(DST))
print(f"saved {DST} ({DST.stat().st_size:,} bytes)")

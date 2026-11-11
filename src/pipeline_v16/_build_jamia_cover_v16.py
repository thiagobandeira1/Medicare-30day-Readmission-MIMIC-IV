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
P("We are pleased to submit our manuscript, “Future-Information Leakage in "
  "30-Day Readmission Prediction: An Empirical Audit and a Calibrated "
  "Electronic Health Record Model in Medicare-Insured Adults,” for "
  "consideration as a "
  "Research and Applications article in the Journal of the American Medical "
  "Informatics Association.")
P("Thirty-day readmission scores are consulted at the moment of discharge, "
  "yet many published models are evaluated in ways that overstate what a "
  "deployed model would deliver. In 236,906 Medicare-insured admissions from "
  "MIMIC-IV v3.1, we develop and internally validate a 44-feature gradient "
  "boosting model in which every one of 207 candidate predictors was audited "
  "for availability at discharge, all preprocessing and feature selection ran "
  "fold-locally inside patient-grouped cross-validation, and the primary "
  "estimate is the out-of-fold performance of the complete selection "
  "procedure (AUROC 0.706); the test partition, reused during earlier "
  "development, provides supporting detail only (AUROC 0.706, good calibration). "
  "Discrimination is moderate, and we do not present the model as ready "
  "for deployment.")
P("We believe the most useful contribution for JAMIA readers is a lesson in "
  "leakage auditing. A prior-utilization trend feature, documented as using "
  "only earlier stays, had in fact been computed over each patient's whole "
  "record, including later admissions; it "
  "survived a documentation-based availability audit and patient-grouped "
  "validation and inflated discrimination by about 0.07 AUROC in earlier "
  "versions of this work. We detected it when we re-examined the features of "
  "an earlier version with simple empirical checks "
  "(a test on first observed admissions, a within-patient constancy scan, and "
  "recomputation of history features from raw timestamps), state what these "
  "checks can and cannot rule out, report the correction openly, and quantify how "
  "common pipeline choices inflate apparent performance. The leak had also "
  "produced spurious findings, including an apparent racial gap in "
  "discrimination, an advantage for first observed admissions, and a calendar-period "
  "gradient, all of which disappear once it is removed. A later comparison "
  "of stored columns with raw tables revealed that the inherited feature "
  "table had been median-imputed over all partitions; we rebuilt it with "
  "missing values preserved, required laboratory results to be stored "
  "before discharge, enforced the discharge cutoff on every laboratory and "
  "medication count, and reran the complete pipeline. On the test "
  "partition the corrected model modestly exceeds LACE and HOSPITAL "
  "reconstructed and refit on the identical cohort, a comparison we "
  "describe with its limits (the scores target other outcomes), and a "
  "readmission-timing analysis shows that most of an apparent day-1 "
  "discrimination peak reflects probable administrative continuations. Every number in the "
  "manuscript is generated from released machine-readable results files, "
  "and the analysis code from the source feature table onward is public.")
P("Generative AI tools (Anthropic Claude and OpenAI ChatGPT) were used for "
  "coding, writing, and review assistance; the authors take full "
  "responsibility for the content.")
P("This manuscript is not under consideration by any other journal, has "
  "not been published previously, and has not been reviewed previously by "
  "any journal; this is its first submission. Analysis code and results "
  "files from earlier versions of this work, together with the capstone "
  "report of the first two authors with Dr. Poellabauer as senior author "
  "(April 2026, revised May 2026), which reports pre-correction results, "
  "are archived on Zenodo (doi:10.5281/zenodo.21987702, an August 2026 "
  "version under concept doi:10.5281/zenodo.21987701); the corrected "
  "version is archived under the same concept DOI. We intend to post this "
  "submitted version "
  "as a preprint on arXiv (cs.LG) under the arXiv non-exclusive license and "
  "will send the identifier to the editorial office once it is announced. "
  "The study is a "
  "secondary analysis of the deidentified, credentialed-access MIMIC-IV "
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
P("11200 SW 8th St, Miami, FL 33199, United States; Phone: [phone withheld]; "
  "Email: tbati006@fiu.edu")

DST = PUB / "JAMIA Cover Letter v16.docx"
doc.save(str(DST))
print(f"saved {DST} ({DST.stat().st_size:,} bytes)")

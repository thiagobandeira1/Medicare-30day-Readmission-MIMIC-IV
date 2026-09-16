# -*- coding: utf-8 -*-
"""Assemble the JAMIA submission packet folder and zip.

Collects the main document, supplement, cover letter, the six renamed
Supplementary Appendix files (already produced by _make_jamia_appendices.py),
separate figure files named by their manuscript numbers, and a README
manifest. Zips everything to Downloads.
"""
import shutil
import zipfile
from pathlib import Path

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
PKT = PUB / "JAMIA Submission Packet"
FIGD = PKT / "Figures"
PKT.mkdir(exist_ok=True)
FIGD.mkdir(exist_ok=True)

MAIN = [
    "Paper JAMIA Submission FINAL.docx",
    "Paper JAMIA Submission FINAL.pdf",
    "Paper JAMIA Supplementary Material.docx",
    "Paper JAMIA Supplementary Material.pdf",
    "JAMIA Cover Letter DRAFT.docx",
]
for name in MAIN:
    shutil.copy2(PUB / name, PKT / name)
    print(f"copied {name}")

FIGS = {
    "Figure 1.png": REPO / "figures_reanalysis" / "r2_fig1_flow_col.png",
    "Figure 2.png": REPO / "figures_v5" / "v5_stability.png",
    "Figure 3.png": REPO / "figures_v5" / "v5_roc_cal.png",
    "Figure 4.png": REPO / "figures_v5" / "v5_daily.png",
    "Figure 5.png": REPO / "figures_v5" / "v5_fairness.png",
    "Figure 6.png": REPO / "figures_v5" / "v5_shap.png",
    "Supplementary Figure S1.png": REPO / "figures_v5" / "v5_rfe_curve.png",
    "Supplementary Figure S2.png":
        REPO / "figures_v5" / "v5_stage_combined.png",
}
for dst, src in FIGS.items():
    shutil.copy2(src, FIGD / dst)
    print(f"copied Figures/{dst}")

README = """JAMIA SUBMISSION PACKET
Predicting 30-Day Hospital Readmission at Discharge: A Leakage-Safe,
Calibrated Electronic Health Record Model in Medicare-Insured Adults
Prepared 2026-09-15. Target: JAMIA, Research and Applications,
subscription route (no article processing charge).

FILES AND UPLOAD ROLES
1. Paper JAMIA Submission FINAL.docx: MAIN DOCUMENT (upload as the
   manuscript file; JAMIA requires Word). The PDF of the same name is a
   local reference copy only.
2. Paper JAMIA Supplementary Material.docx and .pdf: Supplementary
   Material, online only (upload the PDF if a choice is offered; it is
   posted as received). Contains Supplementary Methods S1 to S8,
   Supplementary Results S9 to S13, Table S1, Figures S1 and S2.
3. Supplementary Appendix 1 to 6 (files in this folder): online only
   supplementary files. Appendix 5 (TRIPOD+AI checklist) can go under a
   dedicated reporting-checklist file type if ScholarOne offers one.
4. JAMIA Cover Letter DRAFT.docx: finalize, then paste into the
   ScholarOne cover letter field. Confirm before sending: the manuscript
   was never submitted to any journal (the JMIR AI packet was prepared
   but not submitted), so the first-submission sentence is accurate.
5. Figures folder: separate image files named by manuscript number, 450
   to 500 dpi, ready if the portal requests individual figure uploads.
   The same images are embedded in the main document.

METADATA CHEAT SHEET (ScholarOne)
Type: Research and Applications. Main text 3,993 words (cap 4,000);
abstract 246 words (cap 250); 2 tables (cap 4); 6 figures (cap 6); 40
references; 7 supplementary files.
Keywords (all five are exact MeSH descriptors): Patient Readmission;
Medicare; Machine Learning; Electronic Health Records; Healthcare
Disparities.
Suggested running head: Leakage-Safe Readmission Prediction.
Authors in order, one affiliation for all (Knight Foundation School of
Computing and Information Sciences, Florida International University,
Miami, FL, United States):
  Thiago Bandeira, MS, tbati006@fiu.edu, ORCID 0009-0006-0204-5298
  (corresponding author; 21324 NE 2nd Ct, Miami, FL 33179, United
  States; phone (815) 603-3286)
  Armando Gonzalez, MS, agonz1689@fiu.edu, ORCID 0009-0007-6777-6072
  Christian Poellabauer, PhD, cpoellab@fiu.edu, ORCID 0000-0002-0599-7941
  Ananda Mohan Mondal, PhD, amondal@cis.fiu.edu, ORCID 0000-0002-4005-9942
Funding: none (statement in manuscript). Conflicts: none declared.
Data availability: PhysioNet credentialed DUA; GitHub repositories
(tagged v6.4-submission); Zenodo DOI 10.5281/zenodo.21987702.

BEFORE PORTAL ENTRY (author actions)
1. Mentor sign-off on THIS JAMIA version (Dr. Poellabauer, Dr. Mondal).
2. Confirm the Zenodo DOI and the three GitHub release tags resolve
   publicly.
3. Validate the corresponding author ORCID in the ScholarOne account.
4. Decide the generative AI disclosure answer for the submission form.
5. Optional: keep written documentation of the FIU IRB rationale
   (secondary analysis of deidentified public data) in case the
   editorial office asks.
"""
(PKT / "README FIRST.txt").write_text(README, encoding="utf-8")
for bad in ("—", "–", " - "):
    assert bad not in README, "dash punctuation in README"
print("wrote README FIRST.txt")

ZIP = Path.home() / "Downloads" / "JAMIA Submission Packet FINAL.zip"
if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(PKT))
print(f"zipped {ZIP} ({ZIP.stat().st_size:,} bytes)")

# -*- coding: utf-8 -*-
"""Create the JAMIA-named Supplementary Appendix files from the JMIR
Multimedia Appendix files.

For each DOCX: copy, then rewrite every occurrence of "Multimedia Appendix"
to "Supplementary Appendix" in paragraph and table runs. For the TRIPOD+AI
checklist (Appendix 5) and PROBAST+AI (Appendix 6), also remap Location
entries from the JMIR section names to the JAMIA ones:
Introduction -> Background and Significance; Methods -> Materials and
Methods (with the Objectives row pointing at Objective). The XLSX is copied
under the new name (verified to contain no "Multimedia" strings).
Outputs land in a dedicated folder so the JMIR originals stay untouched.
"""
import shutil
from pathlib import Path
from docx import Document

PUB = Path(__file__).resolve().parent.parent
DST_DIR = PUB / "JAMIA Submission Packet"
DST_DIR.mkdir(exist_ok=True)

FILES = {
    "Multimedia Appendix 1 Predictor Audit and Selection.xlsx":
        "Supplementary Appendix 1 Predictor Audit and Selection.xlsx",
    "Multimedia Appendix 2 LACE HOSPITAL Reconstruction.docx":
        "Supplementary Appendix 2 LACE HOSPITAL Reconstruction.docx",
    "Multimedia Appendix 3 Race Consolidation.docx":
        "Supplementary Appendix 3 Race Consolidation.docx",
    "Multimedia Appendix 4 Supplementary and Development History.docx":
        "Supplementary Appendix 4 Supplementary and Development History.docx",
    "Multimedia Appendix 5 TRIPOD-AI Checklist.docx":
        "Supplementary Appendix 5 TRIPOD-AI Checklist.docx",
    "Multimedia Appendix 6 PROBAST-AI Self-Assessment.docx":
        "Supplementary Appendix 6 PROBAST-AI Self-Assessment.docx",
}

# Location-column remaps for the reporting checklists (5 and 6).
# Order matters: longer strings first so substrings do not double-fire.
SECTION_MAP = [
    ("Materials and Methods", "Materials and Methods"),  # idempotent guard
    ("Methods", "Materials and Methods"),
    ("Introduction", "Background and Significance"),
    ("Conclusions", "Conclusion"),
]


def _iter_paragraphs(doc):
    for par in doc.paragraphs:
        yield par
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    yield par


def _replace_in_runs(par, old, new):
    if old not in par.text:
        return 0
    n = 0
    for run in par.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            n += 1
    return n


def fix_docx(src, dst, remap_sections):
    doc = Document(str(src))
    n_app = 0
    n_sec = 0
    for par in _iter_paragraphs(doc):
        n_app += _replace_in_runs(par, "Multimedia Appendices",
                                  "Supplementary Appendices")
        n_app += _replace_in_runs(par, "Multimedia Appendix",
                                  "Supplementary Appendix")
        # fallback for strings split across runs
        if "Multimedia" in par.text and par.runs:
            full = (par.text
                    .replace("Multimedia Appendices",
                             "Supplementary Appendices")
                    .replace("Multimedia Appendix",
                             "Supplementary Appendix"))
            par.runs[0].text = full
            for run in par.runs[1:]:
                run.text = ""
            n_app += 1
        if remap_sections:
            t = par.text
            # Only remap short Location-style cells, not prose sentences.
            if len(t) <= 80:
                for old, new in SECTION_MAP:
                    if old == new:
                        continue
                    if old in t and new not in t:
                        n_sec += _replace_in_runs(par, old, new)
    doc.save(str(dst))
    return n_app, n_sec


for src_name, dst_name in FILES.items():
    src = PUB / src_name
    dst = DST_DIR / dst_name
    if src.suffix == ".xlsx":
        shutil.copy2(src, dst)
        print(f"copied  {dst.name}")
    else:
        remap = src_name.startswith(("Multimedia Appendix 5",
                                     "Multimedia Appendix 6"))
        n_app, n_sec = fix_docx(src, dst, remap)
        print(f"rewrote {dst.name}: {n_app} appendix labels"
              + (f", {n_sec} section locations" if remap else ""))

# verify: no "Multimedia" left in any produced DOCX
bad = 0
for dst_name in FILES.values():
    p = DST_DIR / dst_name
    if p.suffix != ".docx":
        continue
    doc = Document(str(p))
    for par in _iter_paragraphs(doc):
        if "Multimedia" in par.text:
            print(f"  LEFTOVER in {p.name}: {par.text[:90]!r}")
            bad += 1
print("VERIFY:", "FAIL" if bad else "PASS (no Multimedia strings remain)")

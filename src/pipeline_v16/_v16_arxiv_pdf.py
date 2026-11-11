# -*- coding: utf-8 -*-
"""Build the arXiv preprint PDF: the v16 manuscript without the corresponding
author's phone number, followed by the Supplementary Material PDF.

Word COM runs in its own instance on a private copy (the user may have the
manuscript open). Output: To Send/arXiv/arxiv_preprint_manuscript_supplement.pdf
(arXiv accepts only simple file names)."""
import shutil
import tempfile
from pathlib import Path

import fitz
import win32com.client as win32
from docx import Document

PUB = Path(r"C:\Users\Thiago\Documents\Publication Hospital Research")
MAIN = PUB / "Paper JAMIA Submission v16.docx"
SUPP = PUB / "Paper JAMIA Supplementary Material v16.pdf"
OUTDIR = PUB / "To Send" / "arXiv"
OUT = OUTDIR / "arxiv_preprint_manuscript_supplement.pdf"
PHONE = "Phone: [phone withheld]; "

tmp = Path(tempfile.mkdtemp(prefix="jamia_arxiv_"))
work = tmp / "arxiv_main.docx"
shutil.copy2(MAIN, work)

d = Document(str(work))
hits = 0
for p in d.paragraphs:
    if PHONE not in p.text:
        continue
    runs = p.runs
    full = "".join(r.text for r in runs)
    start = full.index(PHONE)
    end = start + len(PHONE)
    pos = 0
    for r in runs:
        a, b = pos, pos + len(r.text)
        pos = b
        lo, hi = max(a, start), min(b, end)
        if lo < hi:
            r.text = r.text[:lo - a] + r.text[hi - a:]
    hits += 1
assert hits == 1, hits
d.save(str(work))
assert all(PHONE.strip() not in p.text for p in Document(str(work)).paragraphs)

pdf_main = tmp / "arxiv_main.pdf"
app = win32.DispatchEx("Word.Application")
app.Visible = False
app.DisplayAlerts = 0
try:
    doc = app.Documents.Open(str(work), ReadOnly=True, AddToRecentFiles=False)
    doc.ExportAsFixedFormat(str(pdf_main), 17)
    doc.Close(False)
finally:
    app.Quit()

OUTDIR.mkdir(parents=True, exist_ok=True)
m = fitz.open(str(pdf_main))
s = fitz.open(str(SUPP))
n_main, n_supp = m.page_count, s.page_count
m.insert_pdf(s)
m.set_metadata({
    "title": ("Future-Information Leakage in 30-Day Readmission Prediction: "
              "An Empirical Audit and a Calibrated Electronic Health Record "
              "Model in Medicare-Insured Adults"),
    "author": "Thiago Bandeira; Armando Gonzalez; Christian Poellabauer; Ananda Mohan Mondal",
    "subject": "Preprint (manuscript and supplementary material)",
    "keywords": ("Patient Readmission; Medicare; Machine Learning; "
                 "Electronic Health Records; Healthcare Disparities"),
    "creator": "Microsoft Word", "producer": "PyMuPDF",
})
m.del_xml_metadata()
m.save(str(OUT), garbage=3, deflate=True)
m.close(); s.close()
shutil.rmtree(tmp, ignore_errors=True)

chk = fitz.open(str(OUT))
text = "".join(pg.get_text() for pg in chk)
assert PHONE.strip() not in text
print(f"{OUT.name}: {chk.page_count} pages ({n_main} manuscript + {n_supp} supplement), "
      f"{OUT.stat().st_size:,} bytes; phone absent")

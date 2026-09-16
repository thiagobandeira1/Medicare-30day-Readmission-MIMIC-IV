# -*- coding: utf-8 -*-
"""Export the JAMIA manuscript and supplement to PDF via Word COM."""
from pathlib import Path
import win32com.client as win32

PUB = Path(r"C:\Users\Thiago\Documents\Publication Hospital Research")
DOCS = ["Paper JAMIA Submission FINAL.docx",
        "Paper JAMIA Supplementary Material.docx"]

app = win32.Dispatch("Word.Application")
app.Visible = False
app.DisplayAlerts = 0
try:
    for name in DOCS:
        src = (PUB / name).resolve()
        dst = src.with_suffix(".pdf")
        d = app.Documents.Open(str(src), ReadOnly=True)
        d.ExportAsFixedFormat(str(dst), 17)
        n = d.ComputeStatistics(2)
        d.Close(False)
        print(f"{dst.name}: {n} pages, {dst.stat().st_size:,} bytes")
finally:
    app.Quit()

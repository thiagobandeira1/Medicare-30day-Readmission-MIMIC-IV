# -*- coding: utf-8 -*-
"""Export the JAMIA manuscript and supplement to PDF via Word COM.

Uses its own Word instance and a private copy of each file, so a copy that is
open in the user's Word window can neither block the export nor be touched."""
import shutil
import tempfile
from pathlib import Path
import win32com.client as win32

PUB = Path(r"C:\Users\Thiago\Documents\Publication Hospital Research")
DOCS = ["Paper JAMIA Submission v16.docx",
        "Paper JAMIA Supplementary Material v16.docx"]

tmp = Path(tempfile.mkdtemp(prefix="jamia_v16_export_"))
app = win32.DispatchEx("Word.Application")
app.Visible = False
app.DisplayAlerts = 0
try:
    for name in DOCS:
        src = (PUB / name).resolve()
        dst = src.with_suffix(".pdf")
        cp = tmp / name
        shutil.copy2(src, cp)
        d = app.Documents.Open(str(cp), ReadOnly=True, AddToRecentFiles=False)
        d.ExportAsFixedFormat(str(dst), 17)
        n = d.ComputeStatistics(2)
        d.Close(False)
        print(f"{dst.name}: {n} pages, {dst.stat().st_size:,} bytes")
finally:
    app.Quit()
    shutil.rmtree(tmp, ignore_errors=True)

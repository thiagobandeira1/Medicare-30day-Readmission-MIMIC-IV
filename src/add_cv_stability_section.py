"""Insert §10.3c "5-fold CV stability check" section into the publication notebook.

Loads results/cv5_summary.json (produced by src/run_5fold_cv.py) and renders
a per-model + blend table plus the stability figure (figures/cv5_stability.png).

Idempotent — re-running refreshes the inserted cells in place.

Usage:
 python src/add_cv_stability_section.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

try:
 sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
 sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebooks" / "Capstone_Final_Notebook.ipynb"


MD_CELL = """\
### 10.3c 5-fold patient-grouped CV stability check

The single-split point estimate above (§10.1 / §10.3b) gives a clear, peer-review-defensible test AUROC, but it is one realisation. To compare directly against the original capstone report's stability estimate of **0.7956 ± 0.0026 (5-fold CV)**, we ran a matched 5-fold patient-grouped cross-validation on the same V7 (50-feature) parquet, with a 10% inner-validation slice carved from each fold's training portion for early stopping (so no fold ever leaks its own test partition into model selection).

Loaded from `results/cv5_summary.json` (produced by `src/run_5fold_cv.py`). The grey band on the figure marks the originally published stability range; per-fold AUROCs (dots) and the ± std interval are plotted for every GBM family plus the scipy-optimised blend.
"""


CODE_CELL = """\
# ── 10.3c 5-fold CV stability check ────────────────────────────────────────
import json
from IPython.display import Image, display

cv_path = ART_DIR / "cv5_summary.json"
if not cv_path.exists():
 print(f"[info] {cv_path} not found — generate with `python src/run_5fold_cv.py` to enable this section.")
else:
 cv = json.loads(cv_path.read_text())

 print("=" * 78)
 print(f"5-FOLD CV STABILITY (original capstone report reference: 0.7956 ± 0.0026)")
 print("=" * 78)
 print(f"{'Model':<10} {'Mean':>8} {'Std':>8} per-fold AUROCs")
 for m, s in cv["models"].items():
 folds_str = " ".join(f"{a:.4f}" for a in s["fold_aucs"])
 in_band = "[in original capstone stability band]" if 0.7956 - 0.0026 <= s["mean_auroc"] <= 0.7956 + 0.0026 else ""
 print(f"{m:<10} {s['mean_auroc']:>8.4f} {s['std_auroc']:>8.4f} {folds_str} {in_band}")
 b = cv["blend"]
 folds_str = " ".join(f"{a:.4f}" for a in b["fold_aucs"])
 print(f"{'blend':<10} {b['mean_auroc']:>8.4f} {b['std_auroc']:>8.4f} {folds_str}")
 print(f"\\nBlend weights (optimised on concatenated fold-test predictions):")
 print(" " + " ".join(f"{m}={w:.3f}" for m, w in b["weights"].items()))

 fig_path = FIG_DIR / "cv5_stability.png"
 if fig_path.exists():
 display(Image(filename=str(fig_path)))
 else:
 print(f"\\n[info] figure not found at {fig_path}")
"""


def find_cell(nb, needle):
 for i, c in enumerate(nb.cells):
 src = "".join(c.source) if isinstance(c.source, list) else c.source
 if needle in src:
 return i
 raise LookupError(needle)


def main():
 nb = nbf.read(NB_PATH, as_version=4)
 print(f"Loaded {NB_PATH.name}: {len(nb.cells)} cells")

 try:
 md_idx = find_cell(nb, "### 10.3c 5-fold patient-grouped CV stability")
 code_idx = md_idx + 1
 nb.cells[md_idx].source = MD_CELL
 nb.cells[code_idx].source = CODE_CELL
 nb.cells[code_idx].outputs = []
 nb.cells[code_idx].execution_count = None
 print(f" [{md_idx:3d},{code_idx:3d}] §10.3c refreshed in place")
 except LookupError:
 # Insert right after §10.3b REPRODUCTION VALIDATOR code cell
 anchor = find_cell(nb, "# ── 10.3b REPRODUCTION VALIDATOR")
 md_cell = nbf.v4.new_markdown_cell(source=MD_CELL)
 code_cell = nbf.v4.new_code_cell(source=CODE_CELL)
 nb.cells.insert(anchor + 1, md_cell)
 nb.cells.insert(anchor + 2, code_cell)
 print(f" inserted §10.3c (md at {anchor + 1}, code at {anchor + 2})")

 nbf.write(nb, NB_PATH)
 print(f"\nSaved to {NB_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
 main()

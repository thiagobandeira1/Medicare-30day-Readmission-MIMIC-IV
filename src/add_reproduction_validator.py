"""Insert REPRODUCTION VALIDATOR section right after §10.3 donut chart.

Loads results/v17_reference_metrics.json (V17 report's published numbers) and
compares against the current run's v7_summary.json. Emits the same validator
table format the user had in the original capstone notebook.

Idempotent.

Usage:
    python scripts/add_reproduction_validator.py
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
NB_PATH = REPO_ROOT / "notebook" / "Capstone_Final_Notebook.ipynb"


MD_CELL = """\
### 10.3b Reproduction validator — this run vs. V17 report

A targeted reproducibility check: compare the current run's per-model test AUROCs (loaded from `results/v7_summary.json`) against the **published V17 report** targets (stored in `results/v17_reference_metrics.json`, copied verbatim from `Final Model Results/model_v7_final_metrics.json` in the upstream archive).

Verdict legend: **✓** within ±0.003 of target  |  **~** within ±0.008 (one sigma)  |  **✗** outside ±0.008.

The original V17 report's stability estimate was **0.7956 ± 0.0026** (5-fold patient-grouped CV). Our current run is a single train/val/test split, so we should land in or just below this range; verdicts of ~ are expected for a single-split point estimate.
"""


CODE_CELL = """\
# ── 10.3b REPRODUCTION VALIDATOR — this run vs. V17 report ─────────────────
import json

ref_path = ART_DIR / "v17_reference_metrics.json"
if not ref_path.exists():
    print(f"[info] reference metrics not found at {ref_path} — skipping validator")
else:
    ref = json.loads(ref_path.read_text())
    targets = {
        "LightGBM": ref["models"]["LightGBM"]["AUROC"],
        "XGBoost":  ref["models"]["XGBoost"]["AUROC"],
        "CatBoost": ref["models"]["CatBoost"]["AUROC"],
        "HistGBM":  ref["models"]["HistGBM"]["AUROC"],
        "Blend":    ref["test_auroc"],
    }
    achieved = {
        "LightGBM": lgb_test_auroc,
        "XGBoost":  xgb_test_auroc,
        "CatBoost": cb_test_auroc,
        "HistGBM":  hist_test_auroc,
        "Blend":    blend_test_auroc,
    }

    print("=" * 64)
    print("REPRODUCTION VALIDATOR — this run vs. V17 report")
    print("=" * 64)
    print(f"  {'Model':<10} {'Target':>8} {'Achieved':>10} {'Δ':>8}  Verdict")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*8}  {'-'*7}")
    for k in ["LightGBM", "XGBoost", "CatBoost", "HistGBM", "Blend"]:
        t, a = targets[k], achieved[k]
        delta = a - t
        ok = "✓" if abs(delta) < 0.003 else ("~" if abs(delta) < 0.008 else "✗")
        print(f"  {k:<10} {t:>8.4f} {a:>10.4f} {delta:>+8.4f}  {ok}")
    print(f"\\n  Report stability: {ref['stability_mean_auroc']:.4f} ± "
          f"{ref['stability_std_auroc']:.4f}  (5-fold CV)")
    print(f"  Report n_train / n_test: {ref['n_train']:,} / {ref['n_test']:,}")
    print(f"  This   n_train / n_test: {len(X_train):,} / {len(X_test):,}")
    print(f"  This   n_val (inner, for early stopping): {len(X_val):,}")
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
        md_idx = find_cell(nb, "### 10.3b Reproduction validator")
        code_idx = find_cell(nb, "# ── 10.3b REPRODUCTION VALIDATOR")
        nb.cells[md_idx].source = MD_CELL
        nb.cells[code_idx].source = CODE_CELL
        nb.cells[code_idx].outputs = []
        nb.cells[code_idx].execution_count = None
        print(f"  [{md_idx:3d},{code_idx:3d}] §10.3b validator refreshed (already present)")
    except LookupError:
        # Insert right after §10.3 donut-chart code cell
        anchor = find_cell(nb, "# ── 10.3 Ensemble blend weights")
        md_cell = nbf.v4.new_markdown_cell(source=MD_CELL)
        code_cell = nbf.v4.new_code_cell(source=CODE_CELL)
        nb.cells.insert(anchor + 1, md_cell)
        nb.cells.insert(anchor + 2, code_cell)
        print(f"  inserted §10.3b (md at {anchor + 1}, code at {anchor + 2})")

    nbf.write(nb, NB_PATH)
    print(f"\nSaved to {NB_PATH}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()

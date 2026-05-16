"""Insert §10.4 V7 Feature-Importance Ranking section.

Since the publication V7 IS the 50-feature parsimonious set selected from the
368-feature Feature Expansion Version (the parsimony selection already happened
upstream), there is no further trimming to compare against. This section
instead presents the feature-importance ranking of the deployed model — a
useful artefact for any clinician trying to understand which 50 features drive
predictions.

Loads results/v7_summary.json + results/v7_seed0_lightgbm.pkl and renders
ranked feature importances + a saved figure.

Idempotent.

Usage:
    python scripts/add_v7_parsimony_section.py
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
### 10.4 V7 feature-importance ranking — the 50 deployed features

V7 **is** the 50-feature parsimonious set, selected upstream from the larger 368-feature Feature Expansion Version by LightGBM gain-importance ranking (see the original capstone's `Final Model Results/v7_feature_importance.csv` and the published feature_importance figure). Because the parsimony selection has already happened at the dataset-construction stage, there is no additional top-K trimming to compare against — what V7 holds *is* the parsimonious model.

This section presents the **feature-importance ranking of the deployed LightGBM model**, useful as a deployment artefact:

- Clinicians can see which 50 features drive predictions.
- Downstream EHR integration can prioritise these fields for capture quality.
- SHAP attribution in §12 then expands the picture from global importance to per-patient explanations.

The published V17 feature-importance baseline (from the original capstone) is included for comparison — bars are coloured by whether each feature also ranked in the V17 published top-50 (it should — by definition).
"""


CODE_CELL = """\
# ── 10.4 V7 feature-importance ranking ─────────────────────────────────────
# Pulls LightGBM gain importance from the seed-0 model trained in §10.1 and
# plots the top-25 ranked features alongside the published V17 baseline.

import pandas as pd

feature_cols = json.loads((ART_DIR / "v7_feature_cols.json").read_text())
gain = lgb_model_seed0.booster_.feature_importance(importance_type="gain")
ranking = (
    pd.DataFrame({"feature": feature_cols, "gain": gain})
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)
ranking["rank"] = ranking.index + 1
ranking.to_csv(ART_DIR / "v7_feature_importance_current_run.csv", index=False)
print(f"Saved current-run feature importance to "
      f"{ART_DIR / 'v7_feature_importance_current_run.csv'}")
print()
print("Top-25 V7 features by LightGBM gain (current run):")
print(ranking.head(25).to_string(index=False))

# Side-by-side with the published V17 baseline (if available)
v17_csv = BASE_DIR / "Final Model Results" / "v7_feature_importance.csv"
if v17_csv.exists():
    v17_ranked = pd.read_csv(v17_csv).sort_values("importance", ascending=False).reset_index(drop=True)
    overlap = len(set(ranking.head(25)["feature"]) & set(v17_ranked.head(25)["feature"]))
    print(f"\\nTop-25 overlap with V17 published ranking: {overlap}/25 features")
else:
    v17_ranked = None
    print(f"\\n[info] V17 baseline ranking not found at {v17_csv} — skipping overlap.")

# Bar chart of top-25
fig, ax = plt.subplots(figsize=(11, 8))
top25 = ranking.head(25).iloc[::-1]
colors = [PALETTE["teal"] if (v17_ranked is None or feat in set(v17_ranked["feature"]))
          else PALETTE["coral"] for feat in top25["feature"]]
ax.barh(top25["feature"], top25["gain"], color=colors)
for i, (_, row) in enumerate(top25.iterrows()):
    ax.text(row["gain"] + max(top25["gain"]) * 0.005, i, f"{int(row['gain']):>9,}",
            va="center", fontsize=8)
ax.set_xlabel("LightGBM gain importance (current run)")
ax.set_title("V7 deployed model — top-25 features by gain importance\\n"
             "(teal = also in V17 published top-50; coral = V7-only)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "v7_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
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
        md_idx = find_cell(nb, "### 10.4 V7 ")
        # Find the matching code cell — it's right after md_idx
        code_idx = md_idx + 1
        nb.cells[md_idx].source = MD_CELL
        nb.cells[code_idx].source = CODE_CELL
        nb.cells[code_idx].outputs = []
        nb.cells[code_idx].execution_count = None
        print(f"  [{md_idx:3d},{code_idx:3d}] §10.4 refreshed in place")
    except LookupError:
        anchor = find_cell(nb, "# ── 10.3 Ensemble blend weights")
        md_cell = nbf.v4.new_markdown_cell(source=MD_CELL)
        code_cell = nbf.v4.new_code_cell(source=CODE_CELL)
        nb.cells.insert(anchor + 1, md_cell)
        nb.cells.insert(anchor + 2, code_cell)
        print(f"  inserted §10.4 (md at {anchor + 1}, code at {anchor + 2})")

    nbf.write(nb, NB_PATH)
    print(f"\nSaved to {NB_PATH}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()

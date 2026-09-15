# -*- coding: utf-8 -*-
"""Regenerate Figure 3 (selection stability) showing ALL features selected in
at least one outer fold (44, previously truncated to 40) and an annotated
consensus threshold line. Drawing style matches _v5_stages_and_figures.py."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
FIG = REPO / "figures_v5"
TEAL, CORAL = "#1F7A8C", "#E76F51"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 450,
                     "axes.spines.top": False, "axes.spines.right": False})

CMP = json.loads((REPO / "results_v5" / "model_comparison.json").read_text())
freq = CMP["_stability"]["selection_frequency"]
NF = sum(1 for v in freq.values() if v >= 3)
items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
names = [k for k, _ in items][::-1]
vals = [v for _, v in items][::-1]
cols = [TEAL if v >= 3 else "#B9C4CF" for v in vals]

fig, ax = plt.subplots(figsize=(5.4, 8.2))
ax.barh(names, vals, color=cols, height=0.72)
ax.axvline(3, color=CORAL, ls="--", lw=1.2)
ax.annotate("consensus threshold\n(3 of 5 folds)", xy=(3, 5.4),
            xytext=(3.45, 2.0), color=CORAL, fontsize=8, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=1))
ax.set_xlabel("Outer folds selecting the feature (of 5)", fontsize=10)
ax.set_title(f"Feature-selection stability\n(teal = consensus set, n={NF}; "
             f"all {len(items)} features selected at least once)",
             fontsize=10.5)
ax.tick_params(axis="y", labelsize=7.6)
ax.tick_params(axis="x", labelsize=9)
ax.set_xticks([0, 1, 2, 3, 4, 5])
fig.tight_layout()
fig.savefig(FIG / "v5_stability.png", bbox_inches="tight")
plt.close(fig)
print(f"v5_stability.png regenerated: {len(items)} features, consensus n={NF}")

# -*- coding: utf-8 -*-
"""Redraw Supplementary Figure S1 only (label moved left of the consensus line)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v13"
FIG = REPO / "figures_v13"
CORAL = "#E76F51"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 450,
                     "axes.spines.top": False, "axes.spines.right": False})
TR = json.loads((OUT / "rfe_trajectories.json").read_text())
NF = len(json.loads((OUT / "final_model_v13.json").read_text())["final_feature_set"])

fig, ax = plt.subplots(figsize=(4.8, 3.6))
for k, traj in TR.items():
    ns = [t["n"] for t in traj]; a = [t["mean_auc"] for t in traj]
    ax.plot(ns, a, "o-", ms=3, lw=1.2, alpha=0.8, label=f"outer fold {k}")
ax.axvline(NF, color=CORAL, ls="--", lw=1.4)
ax.annotate(f"consensus\nn={NF}", xy=(NF, ax.get_ylim()[0]),
            xytext=(NF + 2, ax.get_ylim()[0] + 0.0008), ha="right",
            color=CORAL, fontweight="bold", fontsize=9)
ax.set_xlabel("Features retained"); ax.set_ylabel("Inner-validation AUROC")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax.invert_xaxis()
fig.tight_layout(); fig.savefig(FIG / "v13_rfe_curve.png", bbox_inches="tight")
print("saved", FIG / "v13_rfe_curve.png")

# -*- coding: utf-8 -*-
"""Regenerate ONLY v5_roc_cal (Figure 4), v5_shap (Figure 8), and v5_dca
(Appendix 4 S1) with dash-free titles. Draws from saved artifacts
(final_model_v5.json + final_preds_v5.npz); no training, and no touch of
Armando's hand-repaired Figure 2 or the separately fixed Figure 5.
Drawing code is copied verbatim from _v5_stages_and_figures.py except the
three title strings."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v5"
FIG = REPO / "figures_v5"
NAVY, TEAL, CORAL = "#081E3F", "#1F7A8C", "#E76F51"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 450,
                     "axes.spines.top": False, "axes.spines.right": False})

R = json.loads((OUT / "final_model_v5.json").read_text())
NF = len(R["final_feature_set"])
P5 = np.load(OUT / "final_preds_v5.npz")
p_te, y_te = P5["p_te"], P5["y_te"]
THR = R["threshold_validation"]

# ------------------------------------------------------- ROC/PR/cal panel
fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0))
fpr, tpr, _ = roc_curve(y_te, p_te)
axes[0, 0].plot(fpr, tpr, color=TEAL, lw=2,
                label=f"consensus-{NF} (AUROC {R['metric_panel']['auroc']['point']:.4f})")
axes[0, 0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
axes[0, 0].set_xlabel("False positive rate")
axes[0, 0].set_ylabel("True positive rate")
axes[0, 0].set_title("ROC"); axes[0, 0].legend(fontsize=8)
prec, rec, _ = precision_recall_curve(y_te, p_te)
axes[0, 1].plot(rec, prec, color=CORAL, lw=2,
                label=f"AP {R['metric_panel']['ap']['point']:.3f}")
axes[0, 1].axhline(y_te.mean(), ls="--", color="gray", lw=1)
axes[0, 1].set_xlabel("Recall"); axes[0, 1].set_ylabel("Precision")
axes[0, 1].set_title("Precision-recall"); axes[0, 1].legend(fontsize=8)
qs = np.quantile(p_te, np.linspace(0, 1, 11))
mids, obs = [], []
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (p_te >= lo) & (p_te <= hi)
    if m.sum():
        mids.append(p_te[m].mean()); obs.append(y_te[m].mean())
axes[1, 0].plot([0, 0.8], [0, 0.8], "--", color="gray", lw=1)
axes[1, 0].plot(mids, obs, "o-", color=TEAL, lw=2,
                label=f"slope {R['metric_panel']['slope']['point']:.2f}, "
                      f"ECE {R['metric_panel']['ece']['point']:.3f}")
axes[1, 0].set_xlabel("Mean predicted"); axes[1, 0].set_ylabel("Observed")
axes[1, 0].set_title("Calibration (deciles)"); axes[1, 0].legend(fontsize=8)
cm = confusion_matrix(y_te, (p_te >= THR).astype(int))
axes[1, 1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[1, 1].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else NAVY,
                        fontweight="bold")
axes[1, 1].set_xticks([0, 1]); axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(["Pred no", "Pred yes"])
axes[1, 1].set_yticklabels(["No readmit", "Readmit"])
axes[1, 1].set_title(f"Confusion (val threshold {THR:.3f})")
fig.suptitle(f"Final consensus-{NF} model: historically exposed test "
             "partition (tertiary evidence)", fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "v5_roc_cal.png", bbox_inches="tight")
plt.close(fig)
print("v5_roc_cal.png regenerated")

# ------------------------------------------------------------------- SHAP
sh = R["shap"][:10]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
names = [s["feature"] for s in sh][::-1]
vals = [s["mean_abs_shap"] for s in sh][::-1]
ax.barh(names, vals, color=TEAL, height=0.7)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(4, 0),
                va="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"Top 10 predictors: final consensus-{NF} model")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(FIG / "v5_shap.png", bbox_inches="tight")
plt.close(fig)
print("v5_shap.png regenerated")

# ------------------------------------------------------------------ DCA
dca = R["dca"]
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.plot(dca["thresholds"], dca["model"], color=TEAL, lw=2, label="Model")
ax.plot(dca["thresholds"], dca["treat_all"], color=CORAL, ls="--", lw=1.6,
        label="Treat all")
ax.axhline(0, color="gray", lw=1.2, label="Treat none")
ax.set_xlabel("Threshold probability (exploratory range)")
ax.set_ylabel("Net benefit")
ax.set_ylim(-0.02, max(dca["model"]) + 0.02)
ax.set_title("Decision curves: final model")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "v5_dca.png", bbox_inches="tight")
plt.close(fig)
print("v5_dca.png regenerated")

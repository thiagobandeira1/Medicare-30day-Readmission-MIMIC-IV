"""Paper-variant of the daily td-AUROC figure: single panel, bold large fonts,
sized to stay legible at the manuscript's 3.0-inch column width."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
d = json.loads((REPO / "results" / "survival_daily_tdauc.json").read_text())
DAYS = np.array(d["days"])
curves = d["curves"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "font.weight": "bold",
                     "axes.labelsize": 16, "axes.labelweight": "bold",
                     "axes.titlesize": 16, "axes.titleweight": "bold", "savefig.dpi": 300})

fig, ax = plt.subplots(figsize=(7.2, 5.4))
style = {
    "Binary XGBoost (deployed)": ("#8D99AE", "--", "o", "Binary XGBoost"),
    "XGBoost-AFT":               ("#E76F51", "-",  "o", "XGBoost-AFT"),
    "Gradient Boosting Survival": ("#1F7A8C", "-",  "s", "GB Survival"),
    "Random Survival Forest":    ("#2EC4B6", "-",  "^", "RSF"),
}
for name in ["Binary XGBoost (deployed)", "XGBoost-AFT",
             "Gradient Boosting Survival", "Random Survival Forest"]:
    c, ls, mk, lab = style[name]
    ax.plot(DAYS, curves[name], ls, color=c, marker=mk, markersize=6, markevery=2,
            lw=2.8, label=lab, zorder=4 if "AFT" in name else 3)

ax.annotate("day-1 peak: 0.83", xy=(1, curves["XGBoost-AFT"][0]), xytext=(6.5, 0.822),
            fontsize=13, color="#E76F51", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E76F51", lw=1.8))
ax.set_xlabel("Days since discharge")
ax.set_ylabel("Time-dependent AUROC")
ax.set_ylim(0.66, 0.85)
ax.set_xticks([1, 7, 14, 21, 29])
ax.grid(True, axis="y", ls=":", alpha=0.45)
ax.legend(loc="lower right", fontsize=12, framealpha=0.95)
fig.tight_layout()
fig.savefig(REPO / "figures" / "survival_daily_tdauc_paper.png", bbox_inches="tight")
print("saved figures/survival_daily_tdauc_paper.png")

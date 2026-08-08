"""Re-render the daily td-AUROC figure from saved results (no model refit).
Fixes: y-limit clipped the day-1 values (up to 0.83); day-29 label hid under the legend."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
d = json.loads((REPO / "results" / "survival_daily_tdauc.json").read_text())
DAYS = np.array(d["days"]); counts = np.array(d["events_per_day"]); curves = d["curves"]

plt.rcParams.update({"font.size": 12, "font.weight": "bold", "axes.labelsize": 13,
                     "axes.titlesize": 14, "axes.titleweight": "bold", "savefig.dpi": 300})
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
style = {
    "Binary XGBoost (deployed)": ("#8D99AE", "--", "o"),
    "XGBoost-AFT":               ("#E76F51", "-",  "o"),
    "Gradient Boosting Survival": ("#1F7A8C", "-",  "s"),
    "Random Survival Forest":    ("#2EC4B6", "-",  "^"),
}
for name in ["Binary XGBoost (deployed)", "XGBoost-AFT",
             "Gradient Boosting Survival", "Random Survival Forest"]:
    c, ls, mk = style[name]
    ax.plot(DAYS, curves[name], ls, color=c, marker=mk, markersize=5,
            lw=2.4, label=name, zorder=4 if "AFT" in name else 3)

for dd in (7, 14, 29):
    ax.axvline(dd, color="#B0BEC5", ls=":", lw=1.4, zorder=1)
    ax.annotate(f"day {dd}", xy=(dd, 0.848), fontsize=10, color="#5B7280",
                ha="center", va="top", fontweight="bold")
ax.set_ylabel("Time-dependent AUROC")
ax.set_title("Day-by-day discrimination across the 30-day post-discharge window")
ax.set_ylim(0.665, 0.85)
ax.grid(True, axis="y", ls=":", alpha=0.45)
ax.legend(loc="lower center", fontsize=10.5, framealpha=0.95, ncol=2)
ax.annotate("highest discrimination on day 1:\nvery early returns are the easiest to identify",
            xy=(1, float(curves["XGBoost-AFT"][0])), xytext=(5.2, 0.806),
            fontsize=9.5, color="#E76F51", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E76F51", lw=1.4))

ax2.bar(DAYS, counts, color="#1F7A8C", alpha=0.75, edgecolor="white", linewidth=0.6)
ax2.set_xlabel("Days since discharge")
ax2.set_ylabel("Readmissions\non that day")
ax2.grid(True, axis="y", ls=":", alpha=0.45)
ax2.set_xticks(np.arange(1, 30, 2))
fig.savefig(REPO / "figures" / "survival_daily_tdauc.png", bbox_inches="tight")
print("re-rendered figures/survival_daily_tdauc.png")

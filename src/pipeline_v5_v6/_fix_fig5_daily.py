# -*- coding: utf-8 -*-
"""Re-render Figure 5 (daily time-dependent AUROC) with the day-1 annotation
moved out of the title/legend and into empty plot space. Reads the canonical
daily_tdauc from final_model_v6.json; no model rerun."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
FIG = REPO / "figures_v5"
R = json.loads((REPO / "results_v6" / "final_model_v6.json").read_text())
CORAL = "#E76F51"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 450,
                     "axes.spines.top": False, "axes.spines.right": False})

NF = R["n_features"]
aft = R["survival"]["daily_tdauc"]
days = np.arange(1, 30, 1.0)

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.plot(days, aft, "o-", color=CORAL, lw=2, ms=4,
        label=f"XGBoost-AFT ({NF} features)")
# headroom above the peak so nothing collides with the title
ax.set_ylim(min(aft) - 0.008, aft[0] + 0.020)
# annotation placed in the empty middle band (well below the title/legend,
# well above the curve), arrow pointing back to the day-1 peak
ax.annotate(f"day-1 peak: {aft[0]:.3f}", xy=(1.3, aft[0]),
            xytext=(8.5, aft[0] - 0.022), color=CORAL, fontweight="bold",
            fontsize=9, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.3))
ax.set_xlabel("Days since discharge")
ax.set_ylabel("Time-dependent AUROC")
ax.set_title("Discrimination by day since discharge\n"
             "(competing-risks labels; IPCW)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "v5_daily.png", bbox_inches="tight")
print(f"v5_daily.png re-rendered; peak {aft[0]:.3f}, day7 {aft[6]:.3f}, "
      f"day29 {aft[-1]:.3f}")

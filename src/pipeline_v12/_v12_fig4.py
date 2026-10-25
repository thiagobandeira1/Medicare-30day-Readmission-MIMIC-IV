# -*- coding: utf-8 -*-
"""Figure 4 for v8: same data and style as v7 (review_v12.json timing), legend
wording aligned with the text (1-hour returns relabeled as nonevents)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
FIG = REPO / "figures_v12"; FIG.mkdir(exist_ok=True)
TEAL, CORAL = "#1F7A8C", "#E76F51"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 450,
                     "axes.spines.top": False, "axes.spines.right": False})
R7 = json.loads((REPO / "results_v12" / "review_v12.json").read_text())["timing"]
T = json.loads((REPO / "results_v12" / "episode_end_rule_v12.json").read_text())
days = np.arange(1, len(R7["primary"]["auc"]) + 1)
fig, ax = plt.subplots(figsize=(5.6, 3.7))
for key, col, lab in (("primary", CORAL, "Primary outcome definition"),
                      ("timing_episode_rule", TEAL,
                       "1 h returns merged into index stay, window from stay end (sensitivity)")):
    r = R7[key] if key == "primary" else T[key]
    ax.fill_between(days, r["lo"], r["hi"], color=col, alpha=0.18, lw=0)
    ax.plot(days, r["auc"], "o-", color=col, lw=1.8, ms=3.2, label=lab)
ax.set_xlabel("Days since discharge")
ax.set_ylabel("Cumulative/dynamic AUROC")
ax.legend(fontsize=7.6, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "v12_daily_episode.png", bbox_inches="tight")
print("saved")

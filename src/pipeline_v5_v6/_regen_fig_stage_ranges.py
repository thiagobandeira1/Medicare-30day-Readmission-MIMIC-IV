# -*- coding: utf-8 -*-
"""Redraw v5_stage_combined.png from the stored JSONs only (no retraining),
with range wording "1 to 7" etc. in tick labels and the panel B axis label,
per the house no-dash rule. Drawing code otherwise mirrors
_v5_stages_and_figures.py."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v5"
FIG = REPO / "figures_v5"
TEAL, CORAL, SLATE, GRAY = "#1F7A8C", "#E76F51", "#8A93A6", "#C2C8D0"
BTEAL = "#2EC4B6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 450,
                     "axes.spines.top": False, "axes.spines.right": False})

shares_out = json.loads((OUT / "stage_v5_shares.json").read_text())
SD = json.loads((OUT / "stage_differential_v5.json").read_text())

fig, (axA, axB) = plt.subplots(2, 1, figsize=(4.7, 8.2),
                               gridspec_kw={"height_ratios": [0.85, 1.15]})
stages_lbl = [s["stage"].replace("-", " to ") for s in shares_out]
xs = range(len(stages_lbl))
lab_y = {"Prior utilization": 66.5, "Laboratory / physiology": 19.5,
         "Index-stay context": 11.0, "Other": 3.5}
lab_txt = {"Prior utilization": "Prior utilization",
           "Laboratory / physiology": "Laboratory /\nphysiology",
           "Index-stay context": "Index-stay\ncontext", "Other": "Other"}
cols_d = {"Prior utilization": TEAL, "Laboratory / physiology": CORAL,
          "Index-stay context": SLATE, "Other": GRAY}
finals = {k: shares_out[-1]["shares"].get(k, 0) for k in lab_y}
slot_order = sorted([k for k in lab_y if k != "Prior utilization"],
                    key=lambda k: finals[k])
slots = [3.5, 11.0, 19.5]
for k, sl_y in zip(slot_order, slots):
    lab_y[k] = sl_y
lab_y["Prior utilization"] = min(72, finals["Prior utilization"] + 2)
for name in ("Prior utilization", "Laboratory / physiology",
             "Index-stay context", "Other"):
    ys = [s["shares"].get(name, 0) for s in shares_out]
    axA.plot(xs, ys, "o-", color=cols_d[name], lw=2.2, ms=5)
    axA.annotate(f"{lab_txt[name]}\n{ys[0]:.1f}% → {ys[-1]:.1f}%",
                 xy=(2.12, lab_y[name]), xycoords=("data", "data"),
                 color=cols_d[name], fontsize=6.8, fontweight="bold",
                 ha="left", va="center", linespacing=1.15)
axA.set_xticks(list(xs)); axA.set_xticklabels(stages_lbl, fontsize=8.5)
axA.set_xlim(-0.2, 3.05); axA.set_ylim(0, 74)
axA.set_xlabel("Days since discharge", fontsize=8.5, fontweight="bold")
axA.set_ylabel("Share of total attribution (%)", fontsize=8.5,
               fontweight="bold")
axA.grid(axis="y", alpha=0.3, ls=":")
axA.tick_params(axis="y", labelsize=8)
axA.set_title("A. Attribution by clinical domain", fontsize=9.5,
              fontweight="bold", loc="left")
sel = SD[::-1]
names = [r["feature"] for r in sel]
vals = [r["diff_pp"] for r in sel]
errs = [[r["diff_pp"] - r["ci95"][0] for r in sel],
        [r["ci95"][1] - r["diff_pp"] for r in sel]]
colsb = [CORAL if v > 0 else BTEAL for v in vals]
axB.barh(names, vals, xerr=errs, color=colsb, height=0.68,
         error_kw=dict(lw=1, capsize=2, ecolor="#333333"))
axB.axvline(0, color="#333333", lw=1)
axB.set_xlabel("Attribution share, days 1 to 7 minus days 15 to 30\n"
               "(percentage points, 95% cluster-bootstrap CI)", fontsize=8.5)
axB.set_title("B. Features leaning EARLY (coral) / LATE (teal)",
              fontsize=9.5, fontweight="bold", loc="left")
axB.tick_params(axis="y", labelsize=7)
axB.tick_params(axis="x", labelsize=8)
fig.tight_layout(h_pad=2.2)
fig.savefig(FIG / "v5_stage_combined.png", bbox_inches="tight")
plt.close(fig)
print(f"v5_stage_combined.png regenerated "
      f"({(FIG / 'v5_stage_combined.png').stat().st_size:,} bytes)")

"""Column-friendly variants of the two wide figures for the two-column layout."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_reanalysis"
FIG = REPO / "figures_reanalysis"
NAVY, TEAL, CORAL = "#081E3F", "#1F7A8C", "#E76F51"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 500,
                     "axes.spines.top": False, "axes.spines.right": False})

CT = json.loads((OUT / "cohort_v2_counts.json").read_text())
B2 = json.loads((OUT / "binary_v2.json").read_text())
FA = json.loads((OUT / "fairness_dca_v2.json").read_text())["fairness"]

# ------------------------------------------------ fig1 portrait flow
# Branching tree (not a linear chain): the analysis cohort is split 80/20 into
# development and test; development is then split 90/10 into training and
# validation. Test is a historically exposed internal partition, drawn parallel
# to development, NOT downstream of validation.
fig, ax = plt.subplots(figsize=(3.9, 5.5))
ax.axis("off")


def box(x, y, w, h, text, fc="#F7F8FA", fs=7.2):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=NAVY, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2))


p = CT["partitions"]
dev_adm = p["train"]["admissions"] + p["val"]["admissions"]
dev_pts = p["train"]["patients"] + p["val"]["patients"]
dev_ev = p["train"]["events"] + p["val"]["events"]

# ---- trunk (centered) : source -> cohort
cx0, cw = 0.17, 0.66
box(cx0, 0.905, cw, 0.068, "MIMIC-IV v3.1\n546,028 admissions")
arr(0.5, 0.905, 0.5, 0.877)
box(cx0, 0.792, cw, 0.083, "Medicare-insured\n244,576 admissions\n"
    "(161,452 others excluded)")
arr(0.5, 0.792, 0.5, 0.762)
box(cx0, 0.688, cw, 0.072, f"Excluded: {CT['index_death_excluded']:,} index\n"
    "in-hospital deaths", fc="#FBEAE6")
arr(0.5, 0.688, 0.5, 0.658)
box(cx0, 0.548, cw, 0.098, f"Analysis cohort\n{CT['cohort_v2_admissions']:,} "
    f"admissions\n{CT['cohort_v2_patients']:,} patients\n"
    f"{CT['label_v2_events']:,} events ({CT['label_v2_prevalence']*100:.1f}%)")

# ---- level 1 split: development (left) and test (right), parallel
ax.text(0.5, 0.516, "patient-grouped 80/20 split on subject_id", ha="center",
        fontsize=6.6, style="italic", color=NAVY)
box(0.02, 0.372, 0.47, 0.104, f"Development, 80%\n{dev_adm:,} adm | "
    f"{dev_pts:,} pts\n{dev_ev:,} events")
box(0.51, 0.372, 0.47, 0.104, "Test, 20% (historically\nexposed internal "
    f"partition)\n{p['test']['admissions']:,} adm | {p['test']['patients']:,} "
    f"pts\n{p['test']['events']:,} events", fc="#EEF2F7", fs=6.8)
arr(0.40, 0.548, 0.255, 0.476)   # cohort -> development
arr(0.60, 0.548, 0.745, 0.476)   # cohort -> test (parallel branch)

# ---- level 2 split: training and validation, both under development
ax.text(0.255, 0.35, "10% of development to validation", ha="center",
        fontsize=6.2, style="italic", color=NAVY)
box(0.02, 0.135, 0.225, 0.115, f"Training\n{p['train']['admissions']:,} adm\n"
    f"{p['train']['patients']:,} pts\n{p['train']['events']:,} events", fs=6.6)
box(0.265, 0.135, 0.225, 0.115, f"Validation\n{p['val']['admissions']:,} adm\n"
    f"{p['val']['patients']:,} pts\n{p['val']['events']:,} events", fs=6.6)
arr(0.20, 0.372, 0.13, 0.25)     # development -> training
arr(0.31, 0.372, 0.38, 0.25)     # development -> validation

ax.text(0.5, 0.055, "outcome: all-cause within-system readmission,\n"
        "0 < Δ ≤ 30 days", ha="center", fontsize=6.8, style="italic")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(FIG / "r2_fig1_flow_col.png", bbox_inches="tight")
plt.close(fig)
print("fig1 column variant done (branching tree; test = historically exposed)")

# ------------------------------------------------ fig6 stacked fairness
overall = B2["metrics"]["auroc"]["point"]
PANELS = [("age_band", ["<65", "65-74", "75-84", "85+"], "Age band"),
          ("sex", ["Female", "Male"], "Sex"),
          ("race", ["White", "Black", "Hispanic/Latino", "Asian", "Other/Unknown"],
           "Race")]
fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.6))
for ax, (col, order, title) in zip(axes, PANELS):
    xs, pts, lo, hi = [], [], [], []
    for i, g in enumerate(order):
        e = FA[col].get(g)
        if not e or e["auroc"] is None:
            continue
        xs.append(i); pts.append(e["auroc"])
        aci = e.get("auroc_ci95", [e["auroc"]] * 2)
        lo.append(e["auroc"] - aci[0]); hi.append(aci[1] - e["auroc"])
    ax.errorbar(xs, pts, yerr=[lo, hi], fmt="o", color=TEAL, capsize=2.5,
                markersize=4.5, lw=1.3, elinewidth=1.1)
    ax.axhline(overall, color=CORAL, ls="--", lw=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([g.replace("Hispanic/Latino", "Hisp./\nLatino")
                        .replace("Other/Unknown", "Other/\nUnk.") for g in order],
                       fontsize=6.8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0.66, 0.90)
    ax.set_ylabel("AUROC", fontsize=7.5, fontweight="bold")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=2)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.grid(axis="y", alpha=0.3, ls=":")
    for x, v in zip(xs, pts):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.2, fontweight="bold")
axes[-1].annotate(f"overall {overall:.3f}", xy=(0.98, 0.05),
                  xycoords="axes fraction", ha="right", fontsize=6.5,
                  color=CORAL, fontweight="bold")
fig.tight_layout(h_pad=1.0)
fig.savefig(FIG / "r2_fig6_fairness_col.png", bbox_inches="tight")
plt.close(fig)
print("fig6 column variant done")

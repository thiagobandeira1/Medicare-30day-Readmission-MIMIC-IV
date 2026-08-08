"""Paper-quality figures for the stage-wise warning-signs analysis.

Two main figures, drawn at their true print size so the type is genuinely ~7 pt
when placed in a 3.0-3.3 inch manuscript column (no downscaling):

  fig_stage_differential.png  which features matter more early vs late,
                              coloured by clinical domain
  fig_stage_domains.png       how each domain's share of the attribution moves
                              across the window

Both use one colour language: orange = laboratory / physiology,
grey = index-stay severity / disposition, teal = prior utilization. The residual
"other" group (medications, order patterns, demographics) is not shown in either
figure; it is flat across the window and is reported in the text instead.

  fig_stage_heatmap.png       optional supplement: feature x stage attribution

Reads results/stage_warning_signs.json; no model refitting.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
RES, FIGS = REPO / "results", REPO / "figures"
d = json.loads((RES / "stage_warning_signs.json").read_text())
stages = d["stages"]
SPANS = [s["span"].replace("-", "–") for s in stages]

# ---- one colour language, shared by both figures ---------------------------
C_LAB, C_SEV, C_UTIL = "#E76F51", "#8D99AE", "#1F7A8C"
# Figure B encodes direction, not domain; teal there is the brighter late colour.
C_UTIL_LATE = "#2EC4B6"

# JSON key -> (display name, colour). "Other" is deliberately omitted.
DOMAINS = {
    "Laboratory / physiology": ("Laboratory / physiology", C_LAB),
    "Index-stay severity / disposition": ("Index-stay severity", C_SEV),
    "Prior utilisation": ("Prior utilization", C_UTIL),
}
MEMBERS = {
    "Laboratory / physiology": ["hemoglobin_last", "sodium_last", "wbc_last", "glucose_last",
                                "bicarbonate_last", "albumin_last", "bun_last", "creatinine_last",
                                "bilirubin_max", "lab_abnormal_rate", "creatinine_x_bun",
                                "bun_creatinine_ratio", "n_lab_item_types", "n_labs_total",
                                "n_lab_orders", "albumin_x_los", "severity_x_lab_abnormal"],
    "Prior utilisation": ["prior_admissions_6m", "prior_admissions_all", "prior_readmission_count",
                          "prior_mean_los_6m", "time_since_last_discharge", "freq_x_recency",
                          "los_trend_180d", "los_trend_x_prior_admits", "log_time_since_discharge",
                          "prior_admits_6m_sq", "log_prior_admits_6m", "prior_admits_x_age",
                          "los_trend_x_prior_6m", "log_prior_readmit_count", "los_per_prior_admit"],
    "Index-stay severity / disposition": ["drg_code", "drg_code_te", "discharge_location",
                                          "discharge_location_te", "last_drg_dispo",
                                          "last_drg_dispo_te", "primary_dx_chapter",
                                          "primary_dx_chapter_te", "severity_composite",
                                          "los_days", "los_x_n_diagnoses", "n_diagnoses",
                                          "n_procedures", "comorbidity_pc4", "elix_mets",
                                          "elix_solid_tumor", "elix_psychoses", "clinical_complexity"],
}
FEAT2DOM = {f: k for k, fs in MEMBERS.items() for f in fs}

LAB = {
    "los_trend_180d": "LOS trend (180 d)",
    "freq_x_recency": "Admit frequency × recency",
    "los_trend_x_prior_admits": "LOS trend × prior admits",
    "prior_readmission_count": "Prior readmissions",
    "prior_admissions_6m": "Prior admissions (6 mo)",
    "prior_admissions_all": "Prior admissions (all)",
    "prior_mean_los_6m": "Mean prior LOS (6 mo)",
    "time_since_last_discharge": "Days since last discharge",
    "los_per_prior_admit": "LOS per prior admission",
    "hemoglobin_last": "Hemoglobin",
    "sodium_last": "Sodium",
    "wbc_last": "White blood cells",
    "glucose_last": "Glucose",
    "bicarbonate_last": "Bicarbonate",
    "albumin_last": "Albumin",
    "bilirubin_max": "Bilirubin (max)",
    "lab_abnormal_rate": "Abnormal lab rate",
    "n_lab_item_types": "Distinct lab types",
    "n_lab_orders": "Lab orders",
    "n_labs_total": "Total lab results",
    "creatinine_x_bun": "Creatinine × BUN",
    "albumin_x_los": "Albumin × LOS",
    "severity_x_lab_abnormal": "Severity × abnormal labs",
    "drg_code": "DRG code",
    "drg_code_te": "DRG (target-encoded)",
    "discharge_location_te": "Discharge location (enc.)",
    "last_drg_dispo_te": "Prior disposition (enc.)",
    "primary_dx_chapter_te": "Diagnosis chapter (enc.)",
    "severity_composite": "Severity composite",
    "clinical_complexity": "Clinical complexity",
    "comorbidity_pc4": "Comorbidity PC4",
    "los_days": "Length of stay",
    "los_x_n_diagnoses": "LOS × diagnoses",
    "n_diagnoses": "Number of diagnoses",
    "n_procedures": "Number of procedures",
    "race_te": "Race (target-encoded)",
    "age_at_admit": "Age at admission",
    "n_discharge_drugs": "Discharge medications",
}
def lab(f):
    return LAB.get(f, f.replace("_", " "))

shares = []
for s in stages:
    tot = sum(s["full_importance"].values()) or 1.0
    shares.append({k: 100.0 * v / tot for k, v in s["full_importance"].items()})

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 400, "axes.linewidth": 0.8})

# ================================================= FIGURE B  (differential)
# All 67 features are eligible; colour encodes DIRECTION (orange = leans early,
# teal = leans late). Because the early-leaning features are largely laboratory
# values and the late-leaning ones largely prior utilization, this reads
# consistently against Figure C even though the colour there encodes domain.
diff = {f: shares[0][f] - shares[2][f] for f in shares[0]}
ranked = sorted(diff, key=lambda f: abs(diff[f]), reverse=True)[:12]
ranked = sorted(ranked, key=lambda f: diff[f])
vals = [diff[f] for f in ranked]
cols = [C_UTIL_LATE if v < 0 else C_LAB for v in vals]

fig, ax = plt.subplots(figsize=(3.30, 0.305 * len(ranked) + 1.30))
ax.barh(range(len(ranked)), vals, color=cols, edgecolor="#16303A", linewidth=0.5, height=0.72)
ax.axvline(0, color="#16303A", linewidth=0.9)
ax.set_yticks(range(len(ranked)))
ax.set_yticklabels([lab(f) for f in ranked], fontsize=7.2)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=6.8)
ax.set_xlabel("Attribution share, days 1–7 minus days 15–30\n(percentage points)",
              fontsize=7.2, fontweight="bold")

span = max(abs(min(vals)), abs(max(vals)))
ax.set_xlim(-span * 1.45, span * 1.45)
for i, v in enumerate(vals):
    off = span * 0.055
    ax.text(v + (off if v >= 0 else -off), i, f"{v:+.1f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=6.5, fontweight="bold", color="#16303A")
ax.text(0.985, 1.022, "more important EARLY", transform=ax.transAxes, ha="right",
        fontsize=6.9, fontweight="bold", color=C_LAB)
ax.text(0.015, 1.022, "more important LATE", transform=ax.transAxes, ha="left",
        fontsize=6.9, fontweight="bold", color=C_UTIL_LATE)
ax.grid(True, axis="x", ls=":", alpha=0.35, linewidth=0.6)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.savefig(FIGS / "fig_stage_differential.png", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved fig_stage_differential.png")

# ================================================= FIGURE C  (domain slope)
fig, ax = plt.subplots(figsize=(3.30, 2.45))
xs = [0, 1, 2]
series = []
for key, (show, c) in DOMAINS.items():
    ys = [s["domain_share_pct"][key] for s in stages]
    ax.plot(xs, ys, "-o", color=c, linewidth=2.0, markersize=4.4,
            markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    series.append({"show": show, "c": c, "ys": ys})

YLO, YHI = 10, 52
MIN_GAP = (YHI - YLO) * 0.20          # the two lower series sit ~2 pp apart
series.sort(key=lambda s: s["ys"][2])
placed = []
for s in series:
    y = s["ys"][2]
    if placed and y - placed[-1] < MIN_GAP:
        y = placed[-1] + MIN_GAP
    placed.append(y)

for s, ylab in zip(series, placed):
    ax.annotate(f"{s['show']}\n{s['ys'][0]:.1f}% → {s['ys'][2]:.1f}%",
                xy=(2, s["ys"][2]), xytext=(2.13, ylab), textcoords="data",
                ha="left", va="center", linespacing=1.35,
                fontsize=6.7, fontweight="bold", color=s["c"], annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=s["c"], linewidth=0.7,
                                shrinkA=0, shrinkB=2, alpha=0.55)
                if abs(ylab - s["ys"][2]) > 0.35 else None)

ax.set_xticks(xs)
ax.set_xticklabels(["1–7", "8–14", "15–30"], fontsize=7.0, fontweight="bold")
ax.set_xlabel("Days since discharge", fontsize=7.2, fontweight="bold", labelpad=3)
ax.set_ylabel("Share of total attribution (%)", fontsize=7.2, fontweight="bold")
ax.tick_params(axis="y", labelsize=6.8)
ax.tick_params(length=2)
ax.set_xlim(-0.28, 2.06)
ax.set_ylim(YLO, YHI)
ax.grid(True, axis="y", ls=":", alpha=0.35, linewidth=0.6)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.19, right=0.50, top=0.96, bottom=0.19)
fig.savefig(FIGS / "fig_stage_domains.png", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print("saved fig_stage_domains.png")

# ================================================= supplement heatmap
top_union = []
for sh in shares:
    top_union += sorted(sh, key=sh.get, reverse=True)[:7]
feats = sorted(set(top_union), key=lambda f: -np.mean([sh[f] for sh in shares]))
M = np.array([[sh[f] for sh in shares] for f in feats])
cmap = LinearSegmentedColormap.from_list("teal", ["#FFFFFF", "#9BD3DD", "#1F7A8C", "#0C3D47"])
fig, ax = plt.subplots(figsize=(3.30, 0.34 * len(feats) + 1.55))
im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=M.max())
ax.set_xticks(range(3))
ax.set_xticklabels([f"Stage {i+1}\n{SPANS[i]}" for i in range(3)], fontsize=7.4, fontweight="bold")
ax.xaxis.set_ticks_position("top")
ax.set_yticks(range(len(feats)))
ax.set_yticklabels([lab(f) for f in feats], fontsize=7.2)
ax.tick_params(length=0)
for i in range(len(feats)):
    for j in range(3):
        v = M[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.6,
                fontweight="bold", color="white" if v > M.max() * 0.55 else "#16303A")
ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(feats), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.1)
ax.tick_params(which="minor", length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.035, fraction=0.045, aspect=34)
cb.set_label("Share of that stage's total attribution (%)", fontsize=7.0)
cb.ax.tick_params(labelsize=6.6, length=2)
cb.outline.set_visible(False)
fig.savefig(FIGS / "fig_stage_heatmap.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("saved fig_stage_heatmap.png")

print("\ndifferential features (early -> late):",
      [(lab(f), round(diff[f], 2)) for f in ranked[::-1]])

"""Rebuild Figure 1 (cohort flow) so the two split labels no longer sit on the arrows.

Geometry, palette, and typography are matched to the original 1900x2700 @ 500 dpi
render; the only substantive change is that the split annotations are moved out of
the arrow paths and set right-aligned in the clear space to the right of them.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

W, H, DPI = 1900, 2700, 500
NAVY = "#081e3f"
FILL = "#f7f8fa"
EXCL = "#fbeae6"
TEST = "#eef2f7"

FS_BOX = 7.0          # box body text
FS_LABEL = 6.9        # italic split annotations
FS_CAPTION = 7.2      # italic outcome note at the bottom
LW = 1.15             # box border / arrow line width (8 px at 500 dpi)
HEAD = 10             # arrowhead mutation scale

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)          # y grows downward, matching pixel coordinates
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")


def box(x0, y0, x1, y1, lines, fill=FILL):
    # coordinates are the fill bounds; the stroke sits centred 5.5 px outside them
    ax.add_patch(
        Rectangle(
            (x0 - 5.5, y0 - 5.5), (x1 - x0) + 11, (y1 - y0) + 11,
            facecolor=fill, edgecolor=NAVY, linewidth=LW, zorder=2,
        )
    )
    ax.text(
        (x0 + x1) / 2, (y0 + y1) / 2, "\n".join(lines),
        ha="center", va="center", fontsize=FS_BOX, color="black",
        linespacing=1.18, zorder=3,
    )


def arrow(x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=HEAD,
            linewidth=LW, color=NAVY,
            shrinkA=0, shrinkB=0, zorder=2,
        )
    )


# --- boxes (pixel coordinates taken from the original render) -----------------
box(361, 125, 1538, 291, ["MIMIC-IV v3.1", "546,028 admissions"])
box(361, 380, 1538, 585,
    ["Medicare-insured", "244,576 admissions", "(161,452 others excluded)"])
box(361, 679, 1538, 855,
    ["Excluded: 7,670 index", "in-hospital deaths"], fill=EXCL)
box(361, 975, 1538, 1219,
    ["Analysis cohort", "236,906 admissions", "87,750 patients",
     "50,793 events (21.4%)"])

box(91, 1417, 926, 1677,
    ["Development, 80%", "189,234 adm | 70,197 pts", "40,660 events"])
box(973, 1417, 1808, 1677,
    ["Test, 20% (historically", "exposed internal partition)",
     "47,672 adm | 17,553 pts", "10,133 events"], fill=TEST)

box(91, 2005, 485, 2293,
    ["Training", "170,677 adm", "63,155 pts", "36,665 events"])
box(532, 2005, 926, 2293,
    ["Validation", "18,557 adm", "7,042 pts", "3,995 events"])

# --- straight chain arrows ----------------------------------------------------
arrow(949.5, 310, 949.5, 351)
arrow(949.5, 601, 949.5, 651)
arrow(949.5, 871, 949.5, 921)

# --- split 1: analysis cohort -> development / test ---------------------------
arrow(773, 1226, 514, 1412)
arrow(1126, 1226, 1385, 1412)

# --- split 2: development -> training / validation ----------------------------
arrow(410, 1686, 288, 1998)
arrow(607, 1686, 729, 1998)

# --- split annotations, moved clear of the arrow paths ------------------------
# Right-aligned in the open wedge to the right of the right-hand arrow.
ax.text(
    1808, 1246, "patient-grouped 80/20\nsplit on subject_id",
    ha="right", va="top", fontsize=FS_LABEL, style="italic",
    color=NAVY, linespacing=1.18, zorder=4,
)
# Clear white space to the right of the development split arrows.
ax.text(
    810, 1845, "10% of development\nto validation",
    ha="left", va="center", fontsize=FS_LABEL, style="italic",
    color=NAVY, linespacing=1.18, zorder=4,
)

# --- outcome note -------------------------------------------------------------
ax.text(
    949.5, 2466,
    "outcome: all-cause within-system readmission,\n0 < Δ ≤ 30 days",
    ha="center", va="center", fontsize=FS_CAPTION, style="italic",
    color=NAVY, linespacing=1.05, zorder=4,
)

out = "/private/tmp/claude-501/-Users-armandogonzalez-Downloads-FIU-CAPSTONE-CODING/a273b0b0-7020-407d-a1b6-10300e9d2743/scratchpad/Figure1_cohort_flow.png"
fig.savefig(out, dpi=DPI, facecolor="white")
print("wrote", out)

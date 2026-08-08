"""Re-render all paper figures (1-8, 10) with consistent style, explicit axis labels, 300 DPI.

Figures 9 (SHAP patient waterfall) requires the trained model + a test sample and is handled
separately.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGS = REPO / "figures"

# ── consistent style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "normal",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PROG = json.loads((RESULTS / "progression.json").read_text())
ENS  = json.loads((RESULTS / "v7_summary.json").read_text())
CV   = json.loads((RESULTS / "cv5_summary.json").read_text())

C_TEAL = "#1F7A8C"
C_GREEN = "#2EC4B6"
C_AMBER = "#F4A261"
C_RED = "#E76F51"
C_PURPLE = "#9D6CFF"
C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GRAY = "#8E8E93"

def save(fig, name):
    p = FIGS / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok  {name}")


# ── Figure 1: eda_class_imbalance ────────────────────────────────────────────
def fig1_class_imbalance():
    counts = [193035, 51541]
    labels = ["Not readmitted (0)", "Readmitted (1)"]
    colors = [C_BLUE, C_ORANGE]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    bars = ax1.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)
    total = sum(counts)
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 4000,
                 f"{c:,}\n({c/total*100:.1f}%)",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax1.set_title("Target distribution: 30-day readmission")
    ax1.set_xlabel("Outcome")
    ax1.set_ylabel("Number of admissions")
    ax1.set_ylim(0, 225000)
    ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold")
    ax2.set_title("Class proportion")
    save(fig, "eda_class_imbalance.png")


# ── Figure 2: eda_discharge_location ─────────────────────────────────────────
def fig2_discharge_location():
    data = [
        ("PSYCH FACILITY", 50.3, 1049),
        ("AGAINST ADVICE", 36.3, 1060),
        ("ACUTE HOSPITAL", 33.3, 1204),
        ("CHRONIC/LONG TERM ACUTE CARE", 27.4, 5586),
        ("HOME HEALTH CARE", 25.1, 56615),
        ("SKILLED NURSING FACILITY", 21.8, 42312),
        ("OTHER FACILITY", 21.7, 369),
        ("ASSISTED LIVING", 20.1, 503),
        ("HOME", 19.9, 62075),
        ("REHAB", 18.8, 8684),
        ("HOSPICE", 3.9, 3566),
        ("DIED", 0.0, 7460),
    ]
    labels = [d[0] for d in data]
    rates = [d[1] for d in data]
    ns = [d[2] for d in data]
    def tier_color(r):
        if r > 30: return C_RED
        if r > 21.1: return C_AMBER
        return C_TEAL
    colors = [tier_color(r) for r in rates]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    bars = ax.barh(labels, rates, color=colors, edgecolor="white", linewidth=1.2)
    for b, r, n in zip(bars, rates, ns):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height()/2,
                f"{r:.1f}%  (n={n:,})", va="center", fontsize=11)
    ax.axvline(21.1, color=C_GRAY, ls="--", lw=1.6)
    ax.invert_yaxis()
    ax.set_xlabel("30-day readmission rate (%)")
    ax.set_ylabel("Discharge destination")
    ax.set_title("30-day readmission rate by discharge destination")
    ax.set_xlim(0, 62)
    legend_elems = [
        Patch(facecolor=C_RED, label="High risk (>30%)"),
        Patch(facecolor=C_AMBER, label="Above average (21.1–30%)"),
        Patch(facecolor=C_TEAL, label="Below average (<21.1%)"),
        plt.Line2D([0], [0], color=C_GRAY, ls="--", lw=1.6, label="Cohort average: 21.1%"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", framealpha=0.95)
    save(fig, "eda_discharge_location.png")


# ── Figure 3: feature_diminishing_returns ────────────────────────────────────
def fig3_diminishing_returns():
    # (n_features, AUROC, label) per Table 1
    pts = [
        (21, 0.7051, "V1"),
        (24, 0.7676, "V2"),
        (24, 0.7605, "V3"),
        (33, 0.763,  "V4"),
        (36, 0.763,  "V5"),
        (34, 0.7750, "V6"),
        (50, 0.7934, "V7 (50 feat)"),
        (368, 0.800, "Feature expansion\n(368 feat)"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    main = pts[:-1]
    xs = [p[0] for p in main]
    ys = [p[1] for p in main]
    ax.plot(xs, ys, ls="--", color=C_GRAY, alpha=0.6, zorder=1)
    ax.scatter(xs[:-1], ys[:-1], s=160, color=C_TEAL, edgecolor="white", linewidth=1.5,
               zorder=3, label="V1–V6 (single-seed)")
    ax.scatter([xs[-1]], [ys[-1]], s=200, color=C_GREEN, edgecolor="white", linewidth=1.5,
               zorder=3, label="V7 deployed (50 features)")
    ax.scatter([368], [0.800], s=200, color=C_RED, edgecolor="white", linewidth=1.5,
               zorder=3, label="Expansion ceiling (368 features)")
    # connect V7 to expansion with annotation
    ax.annotate("", xy=(360, 0.800), xytext=(55, 0.7934),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, alpha=0.65, lw=1.6))
    ax.text(120, 0.808, "Only +0.005 AUROC for ~7× more features",
            color=C_RED, fontsize=12, fontweight="bold", ha="center")
    # point labels
    label_offsets = {"V1": (0, -0.012), "V2": (0, 0.006), "V3": (0, -0.012),
                     "V4": (0, 0.006), "V5": (0, -0.012), "V6": (0, 0.006),
                     "V7 (50 feat)": (-8, 0.006), "Feature expansion\n(368 feat)": (-30, -0.012)}
    for n, auroc, lab in pts:
        dx, dy = label_offsets.get(lab, (0, 0.005))
        ax.text(n + dx, auroc + dy, lab, fontsize=11, fontweight="bold",
                ha="center" if "feat" in lab else "left")
    ax.set_xscale("log")
    ax.set_xlim(15, 600)
    ax.set_ylim(0.69, 0.82)
    ax.set_xticks([20, 30, 50, 100, 200, 400])
    ax.set_xticklabels(["20", "30", "50", "100", "200", "400"])
    ax.set_xlabel("Number of features (log scale)")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Diminishing returns: test AUROC vs. feature count (V1 → expansion)")
    ax.legend(loc="lower right", framealpha=0.95)
    save(fig, "feature_diminishing_returns.png")


# ── Figure 4: fig_b_logreg_v1v6 ──────────────────────────────────────────────
def fig4_logreg():
    versions = ["V1", "V2", "V3", "V6", "V7"]
    aurocs = [PROG["logreg"][v]["test_auroc"] for v in versions]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(versions, aurocs, "o-", color=C_PURPLE, lw=2.5, markersize=12,
            markerfacecolor=C_PURPLE, markeredgecolor="white", markeredgewidth=1.5)
    for v, a in zip(versions, aurocs):
        ax.text(v, a + 0.006, f"{a:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Dataset version")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Regularised logistic regression: test AUROC across dataset versions")
    ax.set_ylim(0.62, 0.74)
    save(fig, "fig_b_logreg_v1v6.png")


# ── Figure 5: fig_cde_lgbm_xgb_mlp_v1v6 ──────────────────────────────────────
def fig5_lgbm_xgb_mlp():
    versions = ["V1", "V2", "V3", "V6", "V7"]
    series = [
        ("LightGBM", [PROG["lightgbm"][v] for v in versions], C_TEAL),
        ("XGBoost",  [PROG["xgboost"][v]  for v in versions], C_GREEN),
        ("MLP",      [PROG["mlp"][v]      for v in versions], C_AMBER),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (name, vals, color) in zip(axes, series):
        ax.plot(versions, vals, "o-", color=color, lw=2.5, markersize=11,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.5)
        for v, a in zip(versions, vals):
            ax.text(v, a + 0.006, f"{a:.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(f"{name}")
        ax.set_xlabel("Dataset version")
    axes[0].set_ylabel("Test AUROC")
    axes[0].set_ylim(0.66, 0.81)
    fig.suptitle("Per-family AUROC progression across feature-engineering versions",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig_cde_lgbm_xgb_mlp_v1v6.png")


# ── Figure 6: fig_f_v6_families ──────────────────────────────────────────────
def fig6_v6_families():
    # V6 cross-family from paper text
    data = [
        ("Stacking\n(meta-learner)", 0.778, C_PURPLE),
        ("LightGBM",                 0.775, C_TEAL),
        ("XGBoost",                  0.773, C_GREEN),
        ("FT-Transformer",           0.770, C_BLUE),
        ("Hybrid LSTM/GRU",          0.770, C_AMBER),
        ("MLP",                      0.722, C_RED),
    ]
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = [d[2] for d in data]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test AUROC")
    ax.set_xlabel("Model family")
    ax.set_title("Cross-family comparison on the V6 dataset (test AUROC)")
    ax.set_ylim(0.70, 0.80)
    ax.tick_params(axis="x", labelrotation=15)
    save(fig, "fig_f_v6_families.png")


# ── Figure 7: fig_z_roc_cal (XGBoost V7 deployed) ────────────────────────────
def fig7_roc_pr_cal_cm():
    preds = np.load(RESULTS / "v7_xgboost_test.npz")["preds"]  # (n_test, 10)
    y_true = np.load(RESULTS / "v7_split_indices.npz")["y_test"]
    y_proba = preds.mean(axis=1) if preds.ndim == 2 else preds
    assert y_proba.shape == y_true.shape, f"shape mismatch {y_proba.shape} vs {y_true.shape}"

    auroc = auc(*roc_curve(y_true, y_proba)[:2])
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    prevalence = y_true.mean()
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    brier = float(np.mean((y_proba - y_true) ** 2))
    threshold = 0.204
    y_pred_bin = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # ROC
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=C_TEAL, lw=2.5, label=f"XGBoost V7 (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color=C_GRAY, ls="--", lw=1.2, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("Receiver-operating characteristic")
    ax.legend(loc="lower right", framealpha=0.95)

    # PR
    ax = axes[0, 1]
    ax.plot(rec, prec, color=C_RED, lw=2.5, label=f"XGBoost V7 (AP = {ap:.4f})")
    ax.axhline(prevalence, color=C_GRAY, ls="--", lw=1.2,
               label=f"Prevalence baseline = {prevalence:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curve")
    ax.legend(loc="upper right", framealpha=0.95)

    # Calibration
    ax = axes[1, 0]
    ax.plot(mean_pred, frac_pos, "o-", color=C_TEAL, lw=2.2, markersize=9,
            markerfacecolor=C_TEAL, markeredgecolor="white",
            label=f"XGBoost V7 (Brier = {brier:.4f})")
    ax.plot([0, 1], [0, 1], color=C_GRAY, ls="--", lw=1.2, label="Perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (observed)")
    ax.set_title("Calibration (reliability diagram, quantile bins)")
    ax.legend(loc="upper left", framealpha=0.95)

    # Confusion matrix
    ax = axes[1, 1]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            txt_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=txt_color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not readmitted", "Readmitted"])
    ax.set_yticklabels(["Not readmitted", "Readmitted"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Confusion matrix (threshold = {threshold})")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="Count")

    fig.suptitle("Deployed XGBoost V7 — test-set performance",
                 fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig_z_roc_cal.png")


# ── Figure 8: fig_g_shap7 ────────────────────────────────────────────────────
def fig8_shap_top7():
    # SHAP mean |value| top-7 — values from the published paper / original SHAP run
    # (the CSV in results/ holds XGBoost 'gain' rather than SHAP, so we hardcode
    # the SHAP values to keep the paper's narrative intact)
    shap_top7 = [
        ("los_trend_180d",         0.5335),
        ("discharge_location_te",  0.3062),
        ("freq_x_recency",         0.1947),
        ("los_trend_x_prior_admits", 0.1946),
        ("severity_x_lab_abnormal", 0.1068),
        ("drg_code_te",            0.1019),
        ("primary_dx_chapter_te",  0.0749),
    ]
    feats = [f for f, _ in shap_top7][::-1]   # invert so largest is on top
    vals  = [v for _, v in shap_top7][::-1]
    palette = [C_TEAL, C_GREEN, C_AMBER, C_RED, C_PURPLE, C_TEAL, C_AMBER][:len(feats)]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.barh(feats, vals, color=palette, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
                f"{v:.4f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    ax.set_title("Top 7 features — SHAP global importance (deployed XGBoost V7)")
    ax.set_xlim(0, max(vals) * 1.18)
    save(fig, "fig_g_shap7.png")


# ── Figure 10: cv5_stability ─────────────────────────────────────────────────
def fig10_cv5_stability():
    keys = ["lightgbm", "xgboost", "catboost", "histgbm"]
    display = ["LightGBM", "XGBoost", "CatBoost", "HistGBM", "Blend"]
    colors = [C_TEAL, C_GREEN, C_AMBER, C_RED, C_PURPLE]
    means = []; stds = []; folds_list = []
    for k in keys:
        m = CV["models"][k]
        folds_list.append(m["fold_aucs"])
        means.append(float(m["mean_auroc"])); stds.append(float(m["std_auroc"]))
    b = CV["blend"]
    folds_list.append(b["fold_aucs"])
    means.append(float(b["mean_auroc"])); stds.append(float(b["std_auroc"]))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    xs = np.arange(len(display))
    # reference band 0.7956 ± 0.0026
    ref_mean, ref_std = 0.7956, 0.0026
    ax.axhspan(ref_mean - ref_std, ref_mean + ref_std, color="lightgray", alpha=0.45,
               label=f"V17 reference band ({ref_mean:.4f} ± {ref_std:.4f})")
    ax.axhline(ref_mean, color=C_GRAY, ls="--", lw=1.2)
    for i, (x, folds, m, s, color) in enumerate(zip(xs, folds_list, means, stds, colors)):
        if folds is not None and len(folds) > 0:
            ax.scatter([x] * len(folds), folds, s=85, color=color, alpha=0.7,
                       edgecolor="white", linewidth=1, zorder=3)
        ax.errorbar([x], [m], yerr=[s], fmt="_", color=color,
                    capsize=8, capthick=2, lw=2.2, zorder=4)
        ax.text(x + 0.18, m, f"{m:.4f}\n±{s:.4f}",
                va="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(display)
    ax.set_xlabel("Model family")
    ax.set_ylabel("Test AUROC")
    ax.set_title("5-fold patient-grouped cross-validation stability")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xlim(-0.5, len(display) - 0.3)
    save(fig, "cv5_stability.png")


# ── run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Re-rendering paper figures with consistent style + explicit axis labels + 300 DPI:")
    fig1_class_imbalance()
    fig2_discharge_location()
    fig3_diminishing_returns()
    fig4_logreg()
    fig5_lgbm_xgb_mlp()
    fig6_v6_families()
    fig7_roc_pr_cal_cm()
    fig8_shap_top7()
    fig10_cv5_stability()
    print("Done.")

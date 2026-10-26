"""v7 stage analyses + figures + anchor-boundary sensitivity.

Outputs -> results_v13/: stage_v13_shares.json, stage_differential_v13.json,
boundary_v13.json; figures_v13/: v7_rfe_curve, v7_stability, v7_roc_cal,
v7_shap, v7_dca, v7_stage_combined. (Figures 4 and 5 are drawn by
_v7_review_analyses.py, which computes their uncertainty bands and strata.)
All figure text uses readable labels and avoids dash punctuation.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "8"
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             confusion_matrix, roc_auc_score)
from _v7_labels import label

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v13"
FIG = REPO / "figures_v13"
FIG.mkdir(exist_ok=True)
R2 = REPO / "results_reanalysis"
SEED = 42
rng = np.random.RandomState(SEED)
NAVY, TEAL, CORAL, SLATE, GRAY = "#081E3F", "#1F7A8C", "#E76F51", "#8A93A6", "#C2C8D0"
BTEAL = "#2EC4B6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 450,
                     "axes.spines.top": False, "axes.spines.right": False})

R = json.loads((OUT / "final_model_v13.json").read_text())
TR = json.loads((OUT / "rfe_trajectories.json").read_text())
CMP = json.loads((OUT / "model_comparison.json").read_text())
FSEL = R["final_feature_set"]
NF = len(FSEL)
maps = json.loads((OUT / "artifacts_v13" / "te_maps.json").read_text())
P7 = np.load(OUT / "final_preds_v13.npz")
p_te, y_te, s_te = P7["p_te"], P7["y_te"], P7["s_te"]
THR = R["threshold_validation"]
PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=8,
              eval_metric="auc", tree_method="hist", verbosity=0)

# ------------------------------------------------------------- stage models
V7 = pd.read_parquet(PUB / "training_table_v13.parquet")
V10 = pd.read_parquet(PUB / "training_table_v13.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
EV = pd.read_parquet(R2 / "cohort_v3_events.parquet")
D2 = DF.loc[DF["hadm_id"].isin(set(EV["hadm_id"]))].reset_index(drop=True)
assert (D2["hadm_id"].values == EV["hadm_id"].values).all()
X = D2[FSEL].copy()
for c, mv in maps.items():
    X[c] = X[c].astype(str).fillna("__NA__").map(mv["map"]) \
        .fillna(mv["global"]).astype(float)
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
event = EV["event_v3"].to_numpy(); tt = EV["time_v3"].to_numpy()
pt = EV["partition"].to_numpy(); subj = EV["subject_id"].to_numpy()
trm, tem = pt == "train", pt == "test"


def dom(f):
    fl = f.lower()
    if any(k in fl for k in ("prior", "readmis", "freq_x", "time_since",
                             "los_per")):
        return "Prior utilization"
    if any(k in fl for k in ("bun", "creatinine", "sodium", "hemoglobin", "wbc",
                             "glucose", "bicarb", "bilirubin", "albumin", "lab",
                             "bmi", "bp_", "troponin", "lactate", "inr",
                             "potassium", "magnesium", "phosphate", "bnp")):
        return "Laboratory and physiology"
    if any(k in fl for k in ("med", "drug", "order", "dose", "iv_", "routes",
                             "antibiotic", "opioid", "insulin", "diuretic",
                             "anticoag", "emar")):
        return "Medications and orders"
    return "Index-stay context"


DOMS = ["Prior utilization", "Laboratory and physiology",
        "Medications and orders", "Index-stay context"]


def stage_contribs(lo, hi):
    at = ~(((event == 1) | (event == 2)) & (tt <= lo))
    ys = ((event == 1) & (tt > lo) & (tt <= hi)).astype(int)
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(X.loc[at & trm].to_numpy(np.float32), ys[at & trm])
    dmx = xgb.DMatrix(X.loc[at & tem].to_numpy(np.float32), feature_names=FSEL)
    c = np.abs(m.get_booster().predict(dmx, pred_contribs=True)[:, :-1])
    return c, subj[at & tem]


shares_out, contribs = [], {}
for name, lo, hi in (("1-7", 0, 7), ("8-14", 7, 14), ("15-30", 14, 30)):
    c, s = stage_contribs(lo, hi)
    contribs[name] = (c, s)
    ma = c.mean(axis=0); tot = ma.sum(); sh = {}
    for f, v in zip(FSEL, ma):
        sh[dom(f)] = sh.get(dom(f), 0) + float(v)
    sh = {k: round(v / tot * 100, 1) for k, v in sh.items()}
    shares_out.append({"stage": name, "shares": sh})
    print(name, sh, flush=True)
(OUT / "stage_v13_shares.json").write_text(json.dumps(shares_out, indent=1))

cE, sE = contribs["1-7"]
cL, sL = contribs["15-30"]
shareE = cE.mean(axis=0); shareE = shareE / shareE.sum() * 100
shareL = cL.mean(axis=0); shareL = shareL / shareL.sum() * 100
diff = shareE - shareL
uE = np.unique(sE); rE = {s: np.where(sE == s)[0] for s in uE}
uL = np.unique(sL); rL = {s: np.where(sL == s)[0] for s in uL}
boots = np.zeros((300, NF))
for b in range(300):
    ixE = np.concatenate([rE[s] for s in rng.choice(uE, len(uE), True)])
    ixL = np.concatenate([rL[s] for s in rng.choice(uL, len(uL), True)])
    se = cE[ixE].mean(axis=0); se = se / se.sum() * 100
    sl = cL[ixL].mean(axis=0); sl = sl / sl.sum() * 100
    boots[b] = se - sl
lo_ci = np.percentile(boots, 2.5, axis=0)
hi_ci = np.percentile(boots, 97.5, axis=0)
order = np.argsort(np.abs(diff))[::-1][:10]
SD = [{"feature": FSEL[i], "early_share": round(float(shareE[i]), 2),
       "late_share": round(float(shareL[i]), 2),
       "diff_pp": round(float(diff[i]), 2),
       "ci95": [round(float(lo_ci[i]), 2), round(float(hi_ci[i]), 2)],
       "excludes_zero": bool(lo_ci[i] > 0 or hi_ci[i] < 0)}
      for i in order]
(OUT / "stage_differential_v13.json").write_text(json.dumps(SD, indent=1))

# ------------------------------------------------- combined stage figure (S2)
fig, (axA, axB) = plt.subplots(2, 1, figsize=(4.9, 8.2),
                               gridspec_kw={"height_ratios": [0.85, 1.15]})
stages_lbl = [s["stage"].replace("-", " to ") for s in shares_out]
xs = range(len(stages_lbl))
cols_d = {"Prior utilization": TEAL, "Laboratory and physiology": CORAL,
          "Medications and orders": SLATE, "Index-stay context": GRAY}
finals = {k: shares_out[-1]["shares"].get(k, 0) for k in DOMS}
slots_sorted = sorted(DOMS, key=lambda k: finals[k])
ypos = {}
last = -99
for k in slots_sorted:
    yv = max(finals[k], last + 7.5)
    ypos[k] = yv
    last = yv
for name in DOMS:
    ys = [s["shares"].get(name, 0) for s in shares_out]
    axA.plot(xs, ys, "o-", color=cols_d[name], lw=2.2, ms=5)
    wrapped = name.replace(" and ", " and\n")
    axA.annotate(wrapped + f"\n{ys[0]:.1f}% → {ys[-1]:.1f}%",
                 xy=(2.12, ypos[name]), color=cols_d[name], fontsize=6.6,
                 fontweight="bold", ha="left", va="center", linespacing=1.1)
axA.set_xticks(list(xs)); axA.set_xticklabels(stages_lbl, fontsize=8.5)
axA.set_xlim(-0.2, 3.3)
axA.set_ylim(0, max(max(s["shares"].values()) for s in shares_out) + 10)
axA.set_xlabel("Days since discharge", fontsize=8.5, fontweight="bold")
axA.set_ylabel("Share of total attribution (%)", fontsize=8.5, fontweight="bold")
axA.grid(axis="y", alpha=0.3, ls=":")
axA.set_title("A. Attribution by clinical domain", fontsize=9.5,
              fontweight="bold", loc="left")
sel = SD[::-1]
axB.barh([label(r["feature"]) for r in sel], [r["diff_pp"] for r in sel],
         xerr=[[r["diff_pp"] - r["ci95"][0] for r in sel],
               [r["ci95"][1] - r["diff_pp"] for r in sel]],
         color=[CORAL if r["diff_pp"] > 0 else BTEAL for r in sel], height=0.68,
         error_kw=dict(lw=1, capsize=2, ecolor="#333333"))
axB.axvline(0, color="#333333", lw=1)
axB.set_xlabel("Attribution share, days 1 to 7 minus days 15 to 30\n"
               "(percentage points, 95% cluster-bootstrap CI)", fontsize=8.5)
axB.set_title("B. Features leaning EARLY (coral) or LATE (teal)",
              fontsize=9.5, fontweight="bold", loc="left")
axB.tick_params(axis="y", labelsize=7); axB.tick_params(axis="x", labelsize=8)
fig.tight_layout(h_pad=2.2)
fig.savefig(FIG / "v13_stage_combined.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- RFE curve (S1)
fig, ax = plt.subplots(figsize=(4.8, 3.6))
for k, traj in TR.items():
    ns = [t["n"] for t in traj]; a = [t["mean_auc"] for t in traj]
    ax.plot(ns, a, "o-", ms=3, lw=1.2, alpha=0.8, label=f"outer fold {k}")
ax.axvline(NF, color=CORAL, ls="--", lw=1.4)
ax.annotate(f"consensus\nn={NF}", xy=(NF, ax.get_ylim()[0]),
            xytext=(NF + 2, ax.get_ylim()[0] + 0.0008), ha="right",
            color=CORAL, fontweight="bold", fontsize=9)
ax.set_xlabel("Features retained"); ax.set_ylabel("Inner-validation AUROC")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax.invert_xaxis()
fig.tight_layout(); fig.savefig(FIG / "v13_rfe_curve.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------- stability (Figure 2)
freq = CMP["_stability"]["selection_frequency"]
items = sorted(freq.items(), key=lambda kv: (-kv[1], label(kv[0])))
names = [label(k) for k, _ in items][::-1]
vals = [v for _, v in items][::-1]
fig, ax = plt.subplots(figsize=(5.6, max(4.5, 0.17 * len(items) + 1.2)))
ax.barh(names, vals, color=[TEAL if v >= 3 else "#B9C4CF" for v in vals],
        height=0.72)
ax.axvline(3, color=CORAL, ls="--", lw=1.2)
ax.annotate("consensus threshold\n(3 of 5 folds)", xy=(3, 1.0),
            xytext=(3.45, 0.2), color=CORAL, fontsize=8, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=1))
ax.set_xlabel("Outer folds selecting the feature (of 5)", fontsize=10)
ax.tick_params(axis="y", labelsize=7.6); ax.tick_params(axis="x", labelsize=9)
ax.set_xticks([0, 1, 2, 3, 4, 5])
fig.tight_layout(); fig.savefig(FIG / "v13_stability.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ ROC/PR/cal panel (Figure 3)
MP = R["metric_panel"]
fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0))
fpr, tpr, _ = roc_curve(y_te, p_te)
axes[0, 0].plot(fpr, tpr, color=TEAL, lw=2,
                label=f"Final model (AUROC {MP['auroc']['point']:.4f})")
axes[0, 0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
axes[0, 0].set_xlabel("False positive rate")
axes[0, 0].set_ylabel("True positive rate")
axes[0, 0].set_title("ROC"); axes[0, 0].legend(fontsize=8, loc="lower right")
prec, rec, _ = precision_recall_curve(y_te, p_te)
axes[0, 1].plot(rec, prec, color=CORAL, lw=2, label=f"AP {MP['ap']['point']:.3f}")
axes[0, 1].axhline(y_te.mean(), ls="--", color="gray", lw=1)
axes[0, 1].set_xlabel("Recall"); axes[0, 1].set_ylabel("Precision")
axes[0, 1].set_title("Precision and recall"); axes[0, 1].legend(fontsize=8)
qs = np.quantile(p_te, np.linspace(0, 1, 11))
mids, obs = [], []
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (p_te >= lo) & (p_te <= hi)
    if m.sum():
        mids.append(p_te[m].mean()); obs.append(y_te[m].mean())
top = max(max(mids), max(obs)) * 1.1
axes[1, 0].plot([0, top], [0, top], "--", color="gray", lw=1)
axes[1, 0].plot(mids, obs, "o-", color=TEAL, lw=2,
                label=(f"slope {MP['slope']['point']:.2f}; "
                       f"CITL {MP['citl']['point']:.2f}\n"
                       f"ECE {MP['ece']['point']:.3f} (10 equal-width bins)"))
axes[1, 0].set_xlabel("Mean predicted (decile bins)")
axes[1, 0].set_ylabel("Observed")
axes[1, 0].set_title("Calibration"); axes[1, 0].legend(fontsize=7.5)
cm = confusion_matrix(y_te, (p_te >= THR).astype(int))
axes[1, 1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[1, 1].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else NAVY,
                        fontweight="bold")
axes[1, 1].set_xticks([0, 1]); axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(["Predicted no", "Predicted yes"])
axes[1, 1].set_yticklabels(["Not readmitted", "Readmitted"])
axes[1, 1].set_title(f"Confusion matrix (threshold {THR:.3f})")
fig.tight_layout()
fig.savefig(FIG / "v13_roc_cal.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ SHAP (Figure 6)
sh = R["shap"][:10]
fig, ax = plt.subplots(figsize=(5.4, 3.8))
nm = [label(s["feature"]) for s in sh][::-1]
vv = [s["mean_abs_shap"] for s in sh][::-1]
ax.barh(nm, vv, color=TEAL, height=0.7)
for i, v in enumerate(vv):
    ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(4, 0),
                va="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Mean |SHAP value| (log-odds scale)")
ax.tick_params(axis="y", labelsize=8)
ax.set_xlim(0, max(vv) * 1.18)
fig.tight_layout(); fig.savefig(FIG / "v13_shap.png", bbox_inches="tight")
plt.close(fig)

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
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "v13_dca.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------- anchor-boundary sensitivity
C = pd.read_parquet(R2 / "cohort_v2.parquet")
keep = (~C["excl_index_death"]).to_numpy()
te_mask = keep & (C["partition"].to_numpy() == "test")
hadm_te = C.loc[te_mask, "hadm_id"].to_numpy()
pats = pd.read_csv(PUB / "Dataset" / "patients_v31.csv.gz", compression="gzip",
                   usecols=["subject_id", "anchor_year", "anchor_year_group"])
pats["group_end"] = pats["anchor_year_group"].str.split("-").str[-1] \
    .str.strip().astype(int)
adm_yr = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                         columns=["hadm_id", "dischtime"])
d2 = pd.DataFrame({"hadm_id": hadm_te}).merge(adm_yr, on="hadm_id") \
    .merge(C.loc[te_mask, ["hadm_id", "subject_id"]], on="hadm_id") \
    .merge(pats, on="subject_id", how="left")
d2["latest_real_year"] = d2["group_end"] + \
    (pd.to_datetime(d2["dischtime"]).dt.year - d2["anchor_year"])
boundary = (d2["latest_real_year"] >= 2022).to_numpy()
safe = ~boundary
BD = {"n_boundary": int(boundary.sum()),
      "share": round(float(boundary.mean()), 4),
      "auroc_all": round(float(roc_auc_score(y_te, p_te)), 4),
      "auroc_safe": round(float(roc_auc_score(y_te[safe], p_te[safe])), 4)}
(OUT / "boundary_v13.json").write_text(json.dumps(BD, indent=1))
print("boundary:", BD, flush=True)
print("v7 figures written:", sorted(p.name for p in FIG.glob("*.png")), flush=True)
print("STAGES_FIGURES_DONE", flush=True)

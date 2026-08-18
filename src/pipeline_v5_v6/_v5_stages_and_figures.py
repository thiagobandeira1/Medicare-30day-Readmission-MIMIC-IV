"""v5 stage analyses + all figures + anchor-boundary sensitivity.

Outputs -> results_v5/: stage_v5_shares.json, stage_differential_v5.json,
boundary_v5.json; figures_v5/: v5_rfe_curve, v5_stability (larger labels per
external audit item 10), v5_roc_cal, v5_shap, v5_fairness, v5_daily, v5_dca,
v5_stage_combined.
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
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import roc_auc_score

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v5"
FIG = REPO / "figures_v5"
FIG.mkdir(exist_ok=True)
R2 = REPO / "results_reanalysis"
SEED = 42
rng = np.random.RandomState(SEED)
NAVY, TEAL, CORAL, SLATE, GRAY = "#081E3F", "#1F7A8C", "#E76F51", "#8A93A6", "#C2C8D0"
BTEAL = "#2EC4B6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 450,
                     "axes.spines.top": False, "axes.spines.right": False})

R = json.loads((OUT / "final_model_v5.json").read_text())
TR = json.loads((OUT / "rfe_trajectories.json").read_text())
CMP = json.loads((OUT / "model_comparison.json").read_text())
FSEL = R["final_feature_set"]
NF = len(FSEL)
maps = json.loads((OUT / "artifacts_v5" / "te_maps.json").read_text())
P5 = np.load(OUT / "final_preds_v5.npz")
p_te, y_te, s_te = P5["p_te"], P5["y_te"], P5["s_te"]
THR = R["threshold_validation"]
PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=8,
              eval_metric="auc", tree_method="hist", verbosity=0)

# ------------------------------------------------------------- stage models
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
EV = pd.read_parquet(R2 / "cohort_v3_events.parquet")
D2 = DF.loc[DF["hadm_id"].isin(set(EV["hadm_id"]))].reset_index(drop=True)
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
                             "los_trend", "los_per")):
        return "Prior utilization"
    if any(k in fl for k in ("bun", "creatinine", "sodium", "hemoglobin", "wbc",
                             "glucose", "bicarb", "bilirubin", "albumin", "lab",
                             "bmi", "bp_")):
        return "Laboratory / physiology"
    if any(k in fl for k in ("los", "discharge", "admission", "icu")):
        return "Index-stay context"
    return "Other"


def stage_contribs(lo, hi):
    at = ~(((event == 1) | (event == 2)) & (tt <= lo))
    ys = ((event == 1) & (tt > lo) & (tt <= hi)).astype(int)
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(X.loc[at & trm].to_numpy(np.float32), ys[at & trm])
    dmx = xgb.DMatrix(X.loc[at & tem].to_numpy(np.float32), feature_names=FSEL)
    c = np.abs(m.get_booster().predict(dmx, pred_contribs=True)[:, :-1])
    return c, subj[at & tem]


shares_out = []
contribs = {}
for name, lo, hi in (("1-7", 0, 7), ("8-14", 7, 14), ("15-30", 14, 30)):
    c, s = stage_contribs(lo, hi)
    contribs[name] = (c, s)
    ma = c.mean(axis=0); tot = ma.sum(); sh = {}
    for f, v in zip(FSEL, ma):
        sh[dom(f)] = sh.get(dom(f), 0) + float(v)
    sh = {k: round(v / tot * 100, 1) for k, v in sh.items()}
    shares_out.append({"stage": name, "shares": sh})
    print(name, sh, flush=True)
(OUT / "stage_v5_shares.json").write_text(json.dumps(shares_out, indent=1))

# ------------------------------------------------- per-feature differential
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
(OUT / "stage_differential_v5.json").write_text(json.dumps(SD, indent=1))
for r in SD:
    print(f"{r['feature']:32s} {r['diff_pp']:+6.2f} pp  CI {r['ci95']}  "
          f"{'*' if r['excludes_zero'] else ''}", flush=True)

# ------------------------------------------------- combined stage figure
fig, (axA, axB) = plt.subplots(2, 1, figsize=(4.7, 8.2),
                               gridspec_kw={"height_ratios": [0.85, 1.15]})
stages_lbl = [s["stage"] for s in shares_out]
xs = range(len(stages_lbl))
lab_y = {"Prior utilization": 66.5, "Laboratory / physiology": 19.5,
         "Index-stay context": 11.0, "Other": 3.5}
lab_txt = {"Prior utilization": "Prior utilization",
           "Laboratory / physiology": "Laboratory /\nphysiology",
           "Index-stay context": "Index-stay\ncontext", "Other": "Other"}
cols_d = {"Prior utilization": TEAL, "Laboratory / physiology": CORAL,
          "Index-stay context": SLATE, "Other": GRAY}
# keep labels clear of each other: order by final share, assign slots bottom-up
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
axA.set_ylabel("Share of total attribution (%)", fontsize=8.5, fontweight="bold")
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
axB.set_xlabel("Attribution share, days 1-7 minus days 15-30\n"
               "(percentage points, 95% cluster-bootstrap CI)", fontsize=8.5)
axB.set_title("B. Features leaning EARLY (coral) / LATE (teal)",
              fontsize=9.5, fontweight="bold", loc="left")
axB.tick_params(axis="y", labelsize=7)
axB.tick_params(axis="x", labelsize=8)
fig.tight_layout(h_pad=2.2)
fig.savefig(FIG / "v5_stage_combined.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- RFE curve
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for k, traj in TR.items():
    ns = [t["n"] for t in traj]; a = [t["mean_auc"] for t in traj]
    ax.plot(ns, a, "o-", ms=3, lw=1.2, alpha=0.8, label=f"outer fold {k}")
ax.axvline(NF, color=CORAL, ls="--", lw=1.4)
ax.annotate(f"consensus\nn={NF}", xy=(NF, ax.get_ylim()[0]),
            xytext=(NF + 8, ax.get_ylim()[0] + 0.001),
            color=CORAL, fontweight="bold", fontsize=9)
ax.set_xlabel("Features retained"); ax.set_ylabel("Inner-validation AUROC")
ax.set_title("Leakage-safe staged RFE, development data\n"
             "(5 outer folds, 3 grouped inner folds each)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax.invert_xaxis()
fig.tight_layout(); fig.savefig(FIG / "v5_rfe_curve.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------- stability (LARGER labels)
freq = CMP["_stability"]["selection_frequency"]
items = sorted(freq.items(), key=lambda kv: -kv[1])[:40]
fig, ax = plt.subplots(figsize=(5.4, 7.6))
names = [k for k, _ in items][::-1]; vals = [v for _, v in items][::-1]
cols = [TEAL if v >= 3 else "#B9C4CF" for v in vals]
ax.barh(names, vals, color=cols, height=0.72)
ax.axvline(3, color=CORAL, ls="--", lw=1.2)
ax.set_xlabel("Outer folds selecting the feature (of 5)", fontsize=10)
ax.set_title(f"Feature-selection stability\n(teal = consensus set, n={NF})")
ax.tick_params(axis="y", labelsize=8.5)
ax.tick_params(axis="x", labelsize=9)
ax.set_xticks([0, 1, 2, 3, 4, 5])
fig.tight_layout(); fig.savefig(FIG / "v5_stability.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- ROC/PR/cal panel
fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0))
fpr, tpr, _ = roc_curve(y_te, p_te)
axes[0, 0].plot(fpr, tpr, color=TEAL, lw=2,
                label=f"consensus-{NF} (AUROC {R['metric_panel']['auroc']['point']:.4f})")
axes[0, 0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
axes[0, 0].set_xlabel("False positive rate")
axes[0, 0].set_ylabel("True positive rate")
axes[0, 0].set_title("ROC"); axes[0, 0].legend(fontsize=8)
prec, rec, _ = precision_recall_curve(y_te, p_te)
axes[0, 1].plot(rec, prec, color=CORAL, lw=2,
                label=f"AP {R['metric_panel']['ap']['point']:.3f}")
axes[0, 1].axhline(y_te.mean(), ls="--", color="gray", lw=1)
axes[0, 1].set_xlabel("Recall"); axes[0, 1].set_ylabel("Precision")
axes[0, 1].set_title("Precision-recall"); axes[0, 1].legend(fontsize=8)
qs = np.quantile(p_te, np.linspace(0, 1, 11))
mids, obs = [], []
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (p_te >= lo) & (p_te <= hi)
    if m.sum():
        mids.append(p_te[m].mean()); obs.append(y_te[m].mean())
axes[1, 0].plot([0, 0.8], [0, 0.8], "--", color="gray", lw=1)
axes[1, 0].plot(mids, obs, "o-", color=TEAL, lw=2,
                label=f"slope {R['metric_panel']['slope']['point']:.2f}, "
                      f"ECE {R['metric_panel']['ece']['point']:.3f}")
axes[1, 0].set_xlabel("Mean predicted"); axes[1, 0].set_ylabel("Observed")
axes[1, 0].set_title("Calibration (deciles)"); axes[1, 0].legend(fontsize=8)
cm = confusion_matrix(y_te, (p_te >= THR).astype(int))
axes[1, 1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[1, 1].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else NAVY,
                        fontweight="bold")
axes[1, 1].set_xticks([0, 1]); axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(["Pred no", "Pred yes"])
axes[1, 1].set_yticklabels(["No readmit", "Readmit"])
axes[1, 1].set_title(f"Confusion (val threshold {THR:.3f})")
fig.suptitle(f"Final consensus-{NF} model - historically exposed test "
             "partition (tertiary evidence)", fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "v5_roc_cal.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------- SHAP
sh = R["shap"][:10]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
names = [s["feature"] for s in sh][::-1]
vals = [s["mean_abs_shap"] for s in sh][::-1]
ax.barh(names, vals, color=TEAL, height=0.7)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(4, 0),
                va="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"Top 10 predictors - final consensus-{NF} model")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(FIG / "v5_shap.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------- fairness
FA = R["fairness"]
PANELS = [("age_band", ["<65", "65-74", "75-84", "85+"], "Age band"),
          ("sex", ["Female", "Male"], "Sex"),
          ("race", ["White", "Black", "Hispanic/Latino", "Asian",
                    "Other/Unknown"], "Race")]
overall = R["metric_panel"]["auroc"]["point"]
fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.8))
for ax, (col, order_, title) in zip(axes, PANELS):
    xs, pts, lo, hi = [], [], [], []
    for i, g in enumerate(order_):
        e = FA[col].get(g)
        if not e or e["auroc"] is None:
            continue
        xs.append(i); pts.append(e["auroc"])
        aci = e.get("auroc_ci95", [e["auroc"]] * 2)
        lo.append(e["auroc"] - aci[0]); hi.append(aci[1] - e["auroc"])
    ax.errorbar(xs, pts, yerr=[lo, hi], fmt="o", color=TEAL, capsize=2.5,
                ms=4.5, lw=1.3, elinewidth=1.1)
    ax.axhline(overall, color=CORAL, ls="--", lw=1)
    ax.set_xticks(range(len(order_)))
    ax.set_xticklabels([g.replace("Hispanic/Latino", "Hisp./\nLatino")
                        .replace("Other/Unknown", "Other/\nUnk.")
                        for g in order_], fontsize=6.6)
    ax.set_xlim(-0.6, len(order_) - 0.4); ax.set_ylim(0.64, 0.90)
    ax.set_ylabel("AUROC", fontsize=7.5, fontweight="bold")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=2)
    ax.tick_params(axis="y", labelsize=6.6); ax.grid(axis="y", alpha=0.3, ls=":")
    for x, v in zip(xs, pts):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6, fontweight="bold")
fig.tight_layout(h_pad=1.0)
fig.savefig(FIG / "v5_fairness.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- daily
days = np.arange(1, 30, 1.0)
aft = R["survival"]["daily_tdauc"]
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.plot(days, aft, "o-", color=CORAL, lw=2, ms=4,
        label=f"XGBoost-AFT ({NF} features)")
ax.annotate(f"day-1 peak: {aft[0]:.3f}", xy=(1, aft[0]),
            xytext=(4, aft[0] + 0.004), color=CORAL, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CORAL))
ax.set_xlabel("Days since discharge"); ax.set_ylabel("Time-dependent AUROC")
ax.set_title("Discrimination by day since discharge\n"
             "(competing-risks labels; IPCW)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "v5_daily.png", bbox_inches="tight")
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
ax.set_title("Decision curves - final model")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "v5_dca.png", bbox_inches="tight")
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
(OUT / "boundary_v5.json").write_text(json.dumps(BD, indent=1))
print("boundary:", BD, flush=True)
print("v5 figures written:", sorted(p.name for p in FIG.glob("*.png")), flush=True)
print("STAGES_FIGURES_DONE", flush=True)

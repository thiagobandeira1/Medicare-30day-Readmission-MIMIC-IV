"""Fairness audit of the DEPLOYED V8 model (66 features, race excluded).

The previous audit described the V7 primary model. Since V8 is what the released
prototype serves, the fairness claims should describe that configuration: a model
that does not condition on race is still obliged to report how it performs across
racial groups, and removing the race feature did not remove the gap.

Retrains V8 on the paper's patient-grouped split, then reports per-subgroup AUROC
with bootstrap CIs, calibration, and operating-point behaviour across age, sex and
race.

Outputs: results/fairness_v8.json, figures/fairness_auroc.png
"""
import os
N_JOBS = 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(N_JOBS)

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PUB = REPO.parent
RESULTS, FIGS = REPO / "results", REPO / "figures"
DATA = PUB / "Dataset" / "mimic-parquet"
API_ART = PUB / "readmission-api" / "artifacts"
SEED = 42
rng = np.random.RandomState(SEED)
t0 = time.time()

V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)

serving = joblib.load(API_ART / "serving_artifacts.joblib")
FEATURES = list(serving["feature_order"])
assert "race_te" not in FEATURES and len(FEATURES) == 66
X = DF[FEATURES].copy()
for c in serving["categorical_features"]:
    X[c] = X[c].astype(str).fillna("__NA__").map(serving["cat_maps"][c]).astype(float)
X = X.apply(pd.to_numeric, errors="coerce")
y = DF["readmit_30d"].to_numpy().astype(int)

split = np.load(RESULTS / "v7_split_indices.npz")
tr, te = split["train_idx"], split["test_idx"]

clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.9,
                        colsample_bytree=0.9, random_state=SEED, n_jobs=N_JOBS,
                        eval_metric="auc", tree_method="hist")
clf.fit(X.loc[tr].to_numpy(np.float32), y[tr])
p = clf.predict_proba(X.loc[te].to_numpy(np.float32))[:, 1]
yt = y[te]
overall = roc_auc_score(yt, p)
THRESHOLD = float(json.loads((API_ART / "metadata.json").read_text())["metrics"]["operating_threshold"])
print(f"[{time.time()-t0:.0f}s] V8 overall test AUROC {overall:.4f}, threshold {THRESHOLD:.4f}", flush=True)

sub = DF.iloc[te].reset_index(drop=True)
race_raw = pd.read_parquet(DATA / "admissions.parquet", columns=["hadm_id", "race"])
sub = sub.merge(race_raw, on="hadm_id", how="left")


def age_band(a):
    if a < 65: return "<65"
    if a < 75: return "65-74"
    if a < 85: return "75-84"
    return "85+"


def race_cat(r):
    r = str(r).upper()
    if "WHITE" in r: return "White"
    if "BLACK" in r: return "Black"
    if "HISPANIC" in r or "LATINO" in r: return "Hispanic/Latino"
    if "ASIAN" in r: return "Asian"
    return "Other/Unknown"


sub["age_band"] = sub["age_at_admit"].apply(age_band)
sub["sex"] = sub["gender"].astype(str).str.upper().map({"F": "Female", "M": "Male"}).fillna("Unknown")
sub["race_cat"] = sub["race"].apply(race_cat)


def auroc_ci(yy, pp, n_boot=1000):
    pt = roc_auc_score(yy, pp)
    vals = []
    n = len(yy)
    for _ in range(n_boot):
        ix = rng.randint(0, n, n)
        if len(np.unique(yy[ix])) < 2:
            continue
        vals.append(roc_auc_score(yy[ix], pp[ix]))
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece(yy, pp, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    idx = np.digitize(pp, bins) - 1
    e = 0.0
    for b in range(nb):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(pp[m].mean() - yy[m].mean())
    return float(e)


out = {"model": "V8 (66 features, race excluded)", "overall_auroc": round(float(overall), 6),
       "threshold": THRESHOLD}
ORDER = {"age_band": ["<65", "65-74", "75-84", "85+"],
         "sex": ["Female", "Male"],
         "race_cat": ["White", "Black", "Hispanic/Latino", "Asian", "Other/Unknown"]}
for col, groups in ORDER.items():
    out[col] = {}
    print(f"\n{col}")
    for g in groups:
        m = (sub[col] == g).to_numpy()
        if m.sum() < 50 or len(np.unique(yt[m])) < 2:
            continue
        a, lo, hi = auroc_ci(yt[m], p[m])
        pred = (p[m] >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt[m], pred, labels=[0, 1]).ravel()
        out[col][g] = {"n": int(m.sum()), "auroc": round(a, 4), "ci": [round(lo, 4), round(hi, 4)],
                       "ece": round(ece(yt[m], p[m]), 4),
                       "mean_predicted": round(float(p[m].mean()), 4),
                       "observed": round(float(yt[m].mean()), 4),
                       "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) else float("nan"), 3),
                       "fpr": round(float(fp / (fp + tn)) if (fp + tn) else float("nan"), 3)}
        v = out[col][g]
        print(f"  {g:16s} n={v['n']:>6,}  AUROC {v['auroc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]"
              f"  ECE {v['ece']:.3f}  pred/obs {v['mean_predicted']:.3f}/{v['observed']:.3f}")

(RESULTS / "fairness_v8.json").write_text(json.dumps(out, indent=2))

# ------------------------------------------------------------------- figure
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 400, "axes.linewidth": 0.8})
fig, axes = plt.subplots(3, 1, figsize=(3.30, 5.35))
TITLES = {"age_band": "Age band", "sex": "Sex", "race_cat": "Race"}
for ax, col in zip(axes, ["age_band", "sex", "race_cat"]):
    gs = list(out[col].keys())
    xs = np.arange(len(gs))
    pts = [out[col][g]["auroc"] for g in gs]
    lo = [pts[i] - out[col][g]["ci"][0] for i, g in enumerate(gs)]
    hi = [out[col][g]["ci"][1] - pts[i] for i, g in enumerate(gs)]
    ax.errorbar(xs, pts, yerr=[lo, hi], fmt="o", color="#1F7A8C", capsize=2.5,
                markersize=4.5, lw=1.4, elinewidth=1.1)
    ax.axhline(overall, color="#E76F51", ls="--", lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels(gs, fontsize=6.8, rotation=18 if col == "race_cat" else 0,
                       ha="right" if col == "race_cat" else "center")
    ax.set_ylim(0.68, 0.92)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90])
    ax.tick_params(axis="y", labelsize=6.6)
    ax.tick_params(length=2)
    ax.set_ylabel("AUROC", fontsize=7.2, fontweight="bold")
    ax.set_title(TITLES[col], fontsize=7.6, fontweight="bold", pad=3)
    ax.grid(True, axis="y", ls=":", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for i, g in enumerate(gs):
        ax.annotate(f"{pts[i]:.3f}", (i, pts[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.0, fontweight="bold", color="#16303A")
axes[0].annotate(f"overall {overall:.3f}", xy=(0.985, 0.06), xycoords="axes fraction",
                 ha="right", fontsize=6.2, fontweight="bold", color="#E76F51")
fig.tight_layout(h_pad=1.1)
fig.savefig(FIGS / "fairness_auroc.png", bbox_inches="tight", pad_inches=0.03)
print(f"\nSaved figures/fairness_auroc.png and results/fairness_v8.json   TOTAL {time.time()-t0:.0f}s")

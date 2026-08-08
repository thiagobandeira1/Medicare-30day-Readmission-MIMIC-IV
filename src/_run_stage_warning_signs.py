"""Early-warning signs by post-discharge stage.

Dr. Poellabauer asked not only *how soon* patients return but *what the first
warning signs are*. This splits the 30-day window into three stages and asks, for
each, which features mark a patient for return in THAT window:

    Stage 1  days 1-7    early return
    Stage 2  days 8-14   intermediate
    Stage 3  days 15-30  late

Method: landmark analysis. Each stage is modelled on the patients still at risk at
the start of that stage (those already readmitted or dead are removed), so stage 2
answers "given this patient made it past day 7, what predicts a return in the next
week?" rather than re-describing the stage-1 population. One XGBoost classifier per
stage on the deployed V8 (race-excluded, 66) features and the paper's patient-grouped split;
TreeSHAP on the held-out test partition gives each stage's drivers.

Outputs:
  results/stage_warning_signs.json
  figures/stage_warning_signs.png
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
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PUB = REPO.parent
RESULTS, FIGS = REPO / "results", REPO / "figures"
DATA = PUB / "Dataset" / "mimic-parquet"
API_ART = PUB / "readmission-api" / "artifacts"
HORIZON, SEED = 30, 42
np.random.seed(SEED)
t0 = time.time()

# ---------------------------------------------------------- feature matrix
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
v7_unique = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7_unique].reset_index(drop=True)], axis=1)

serving = joblib.load(API_ART / "serving_artifacts.joblib")
FEATURES = serving["feature_order"]
assert len(FEATURES) == 66, "expects the deployed race-excluded set"
X = DF[FEATURES].copy()
for c in serving["categorical_features"]:
    X[c] = X[c].astype(str).fillna("__NA__").map(serving["cat_maps"][c]).astype(float)
Xv = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
print(f"[{time.time()-t0:.0f}s] matrix {Xv.shape}", flush=True)

# ---------------------------------------------------------- event times
ADM = pd.read_parquet(DATA / "admissions.parquet",
                      columns=["subject_id", "hadm_id", "admittime", "dischtime",
                               "hospital_expire_flag", "discharge_location"])
ADM = ADM.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
ADM["next_admittime"] = ADM.groupby("subject_id")["admittime"].shift(-1)
ADM["next_died"] = ADM.groupby("subject_id")["hospital_expire_flag"].shift(-1)
lab = DF[["hadm_id", "dischtime_dt"]].merge(
    ADM[["hadm_id", "next_admittime", "next_died", "hospital_expire_flag", "discharge_location"]]
    .rename(columns={"discharge_location": "dl_raw"}), on="hadm_id", how="left")
delta = (lab["next_admittime"] - lab["dischtime_dt"]).dt.total_seconds() / 86400.0
event = np.zeros(len(lab), dtype=np.int8)
ttime = np.full(len(lab), float(HORIZON))
in_hosp_death = (lab["hospital_expire_flag"] == 1) | (lab["dl_raw"] == "DIED")
event[in_hosp_death] = 2; ttime[in_hosp_death] = 0.0
m_re = (~in_hosp_death) & delta.notna() & (delta > 0) & (delta <= HORIZON) & (lab["next_died"].fillna(0) == 0)
event[m_re.values] = 1; ttime[m_re.values] = delta[m_re].values
m_cd = (~in_hosp_death) & delta.notna() & (delta > 0) & (delta <= HORIZON) & (lab["next_died"].fillna(0) == 1)
event[m_cd.values] = 2; ttime[m_cd.values] = delta[m_cd].values

split = np.load(RESULTS / "v7_split_indices.npz")
alive = ttime > 0
tr = split["train_idx"][alive[split["train_idx"]]]
te = split["test_idx"][alive[split["test_idx"]]]
print(f"[{time.time()-t0:.0f}s] alive-discharge train/test: {len(tr):,}/{len(te):,}", flush=True)

STAGES = [("Stage 1", "Days 1-7", 1, 7), ("Stage 2", "Days 8-14", 8, 14),
          ("Stage 3", "Days 15-30", 15, 30)]
LABELS = json.loads((REPO / "results" / "feature_labels.json").read_text()) \
    if (REPO / "results" / "feature_labels.json").exists() else {}


def pretty(f: str) -> str:
    return LABELS.get(f, f.replace("_", " "))


out = {"stages": []}
for name, span, lo, hi in STAGES:
    # Landmark: keep only patients still at risk when the stage opens.
    def stage_arrays(idx):
        t, e = ttime[idx], event[idx]
        at_risk = (t >= lo) | ((t < lo) & (e == 0))       # not already readmitted/dead
        at_risk = ~(((e == 1) | (e == 2)) & (t < lo))
        keep = idx[at_risk]
        y = ((event[keep] == 1) & (ttime[keep] >= lo) & (ttime[keep] <= hi)).astype(int)
        return keep, y

    tr_i, y_tr = stage_arrays(tr)
    te_i, y_te = stage_arrays(te)

    clf = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9,
        colsample_bytree=0.9, random_state=SEED, n_jobs=N_JOBS,
        eval_metric="auc", tree_method="hist",
    )
    clf.fit(Xv[tr_i], y_tr)
    p = clf.predict_proba(Xv[te_i])[:, 1]
    auc = roc_auc_score(y_te, p)

    booster = clf.get_booster()
    contribs = booster.predict(xgb.DMatrix(Xv[te_i], feature_names=FEATURES),
                               pred_contribs=True)[:, :-1]
    # Signed mean over the patients who DID return in this stage: the average
    # push each feature gave the people who actually came back then.
    ev = y_te == 1
    signed = contribs[ev].mean(axis=0)
    magnitude = np.abs(contribs).mean(axis=0)
    order = np.argsort(-magnitude)[:12]

    drivers = [{"feature": FEATURES[i], "label": pretty(FEATURES[i]),
                "mean_abs_shap": round(float(magnitude[i]), 5),
                "signed_shap_among_returners": round(float(signed[i]), 5)}
               for i in order]
    full_imp = {FEATURES[i]: round(float(magnitude[i]), 6) for i in range(len(FEATURES))}
    out["stages"].append({
        "full_importance": full_imp,
        "stage": name, "span": span, "day_from": lo, "day_to": hi,
        "n_at_risk_test": int(len(te_i)), "n_events_test": int(ev.sum()),
        "event_rate": round(float(ev.mean()), 4),
        "test_auroc": round(float(auc), 4), "drivers": drivers,
    })
    print(f"[{time.time()-t0:.0f}s] {name} {span}: at-risk {len(te_i):,}, events {int(ev.sum()):,}, "
          f"AUROC {auc:.4f}", flush=True)
    print("      top:", ", ".join(d["feature"] for d in drivers[:5]), flush=True)

# ------------------------------------------- clinical-domain attribution share
# Does the *kind* of signal change across the window? Group the 67 features into
# clinical domains and report each domain's share of total attribution per stage.
DOMAINS = {
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
for st in out["stages"]:
    imp = st["full_importance"]
    total = sum(imp.values()) or 1.0
    shares, assigned = {}, set()
    for dom, feats in DOMAINS.items():
        v = sum(imp.get(f, 0.0) for f in feats)
        shares[dom] = round(100.0 * v / total, 1)
        assigned |= {f for f in feats if f in imp}
    shares["Other (medications, orders, demographics)"] = round(
        100.0 * sum(v for k, v in imp.items() if k not in assigned) / total, 1)
    st["domain_share_pct"] = shares

print()
print("=== attribution share by clinical domain (% of total |SHAP|) ===")
doms = list(out["stages"][0]["domain_share_pct"].keys())
print(f"{'domain':44s}" + "".join(f"{s['span']:>13s}" for s in out["stages"]))
for dom in doms:
    print(f"{dom:44s}" + "".join(f"{s['domain_share_pct'][dom]:>12.1f}%" for s in out["stages"]))

(RESULTS / "stage_warning_signs.json").write_text(json.dumps(out, indent=2))

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "font.weight": "bold",
                     "axes.labelsize": 12.5, "axes.labelweight": "bold",
                     "axes.titlesize": 12.5, "axes.titleweight": "bold", "savefig.dpi": 300})
fig = plt.figure(figsize=(16.5, 9.4))
gs = fig.add_gridspec(2, 3, height_ratios=[2.15, 1.0], hspace=0.42, wspace=0.55)
COLORS = ["#E76F51", "#1F7A8C", "#2EC4B6"]

for k, (st, col) in enumerate(zip(out["stages"], COLORS)):
    ax = fig.add_subplot(gs[0, k])
    top = st["drivers"][:8][::-1]
    ax.barh(range(len(top)), [d["mean_abs_shap"] for d in top],
            color=col, edgecolor="black", linewidth=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([d["label"] for d in top], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{st['stage']}  ·  {st['span']}\n"
                 f"{st['n_events_test']:,} returns  ·  AUROC {st['test_auroc']:.3f}")
    ax.grid(True, axis="x", ls=":", alpha=0.4)

ax = fig.add_subplot(gs[1, :])
doms = ["Laboratory / physiology", "Prior utilisation",
        "Index-stay severity / disposition", "Other (medications, orders, demographics)"]
short = ["Laboratory /\nphysiology", "Prior\nutilisation",
         "Index-stay severity /\ndisposition", "Other"]
xs = np.arange(len(doms)); w = 0.26
for k, (st, col) in enumerate(zip(out["stages"], COLORS)):
    vals = [st["domain_share_pct"][d] for d in doms]
    b = ax.bar(xs + (k - 1) * w, vals, w, label=st["span"], color=col,
               edgecolor="black", linewidth=0.7)
    ax.bar_label(b, fmt="%.1f%%", fontsize=9.5, padding=2)
ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=10.5)
ax.set_ylabel("Share of total\nattribution (%)")
ax.set_ylim(0, 58)
ax.set_title("Where the signal comes from shifts across the window: "
             "physiology matters most for the earliest returns, prior utilisation for the latest")
ax.legend(title="Return window", fontsize=10.5, title_fontsize=10.5, loc="upper right")
ax.grid(True, axis="y", ls=":", alpha=0.4)

fig.suptitle("Warning signs by post-discharge stage (landmark models on the deployed V8 features)",
             fontsize=15, fontweight="bold", y=0.985)
fig.savefig(FIGS / "stage_warning_signs.png", bbox_inches="tight")
print("\nSaved figures/stage_warning_signs.png and results/stage_warning_signs.json")
print(f"TOTAL {time.time()-t0:.0f}s")

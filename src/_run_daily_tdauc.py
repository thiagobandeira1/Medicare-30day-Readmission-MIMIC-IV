"""Day-by-day time-dependent AUROC (days 1-29) for the survival models.

Extends the 3 landmark points (7/14/30) reported in the paper to a full daily curve,
so we can see exactly how discrimination evolves across the post-discharge window.

Replicates Survival_Analysis_v4.ipynb sections 3, 5, 7, 8, 9 exactly (same SEED,
same params, same patient-grouped split), then evaluates at t = 1..29.

Outputs:
  figures/survival_daily_tdauc.png
  results/survival_daily_tdauc.json
"""
import os
N_JOBS = 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(N_JOBS)

import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

HORIZON_DAYS, SEED = 30, 42
np.random.seed(SEED)

REPO = Path(__file__).resolve().parent.parent
PUB, RESULTS, FIGS = REPO.parent, REPO / "results", REPO / "figures"
DATA = PUB / "Dataset" / "mimic-parquet"
t0 = time.time()

# ---------------- 3. cohort ----------------
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
ADM = pd.read_parquet(DATA / "admissions.parquet",
                      columns=["subject_id", "hadm_id", "admittime", "dischtime",
                               "deathtime", "hospital_expire_flag", "discharge_location"])
ADM = ADM.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
ADM["next_admittime"] = ADM.groupby("subject_id")["admittime"].shift(-1)
ADM["next_died"] = ADM.groupby("subject_id")["hospital_expire_flag"].shift(-1)
ADM["prev_dischtime"] = ADM.groupby("subject_id")["dischtime"].shift(1)
ADM["days_since_last_discharge"] = (ADM["admittime"] - ADM["prev_dischtime"]).dt.total_seconds() / 86400.0

df = V7.merge(ADM[["hadm_id", "next_admittime", "next_died", "hospital_expire_flag",
                   "discharge_location", "days_since_last_discharge"]]
              .rename(columns={"discharge_location": "discharge_location_raw"}),
              on="hadm_id", how="left")
delta = (df["next_admittime"] - df["dischtime_dt"]).dt.total_seconds() / 86400.0
event = np.zeros(len(df), dtype=np.int8)
ttime = np.full(len(df), HORIZON_DAYS, dtype=np.float64)
in_hosp_death = (df["hospital_expire_flag"] == 1) | (df["discharge_location_raw"] == "DIED")
event[in_hosp_death] = 2; ttime[in_hosp_death] = 0.0
m_re = (~in_hosp_death) & delta.notna() & (delta > 0) & (delta <= HORIZON_DAYS) & (df["next_died"].fillna(0) == 0)
event[m_re.values] = 1; ttime[m_re.values] = delta[m_re].values
m_cd = (~in_hosp_death) & delta.notna() & (delta > 0) & (delta <= HORIZON_DAYS) & (df["next_died"].fillna(0) == 1)
event[m_cd.values] = 2; ttime[m_cd.values] = delta[m_cd].values
df["event_type"] = event; df["time_to_event"] = ttime
print(f"[{time.time()-t0:.0f}s] cohort built:", pd.Series(event).value_counts().sort_index().to_dict(), flush=True)

# ---------------- 5. features + split ----------------
binary_feat = json.loads((RESULTS / "v7_feature_cols.json").read_text())
feat_cols = binary_feat + ["days_since_last_discharge"]
df["days_since_last_discharge"] = df["days_since_last_discharge"].fillna(365.0).clip(0.0, 365.0)
for c in feat_cols:
    if df[c].dtype == "object":
        df[c] = LabelEncoder().fit_transform(df[c].astype(str).fillna("__NA__"))

split = np.load(RESULTS / "v7_split_indices.npz")
train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

def mask_alive(idx):
    return idx[(df.iloc[idx]["time_to_event"] > 0).values]

train_idx_s, val_idx_s, test_idx_s = mask_alive(train_idx), mask_alive(val_idx), mask_alive(test_idx)
test_alive_mask = (df.iloc[test_idx]["time_to_event"] > 0).values   # for binary-model alignment
print(f"[{time.time()-t0:.0f}s] split: {len(train_idx_s):,}/{len(val_idx_s):,}/{len(test_idx_s):,}", flush=True)

def mat(idx):
    X = df.iloc[idx][feat_cols].to_numpy(dtype=np.float64)
    med = np.nanmedian(X, axis=0)
    return X, med

X_train, med = mat(train_idx_s)
med = np.nan_to_num(med, nan=0.0)
def clean(X):
    X = np.where(np.isfinite(X), X, np.nan)
    ix = np.where(np.isnan(X))
    X[ix] = np.take(med, ix[1])
    return X
X_train = clean(X_train)
X_val = clean(df.iloc[val_idx_s][feat_cols].to_numpy(dtype=np.float64))
X_test = clean(df.iloc[test_idx_s][feat_cols].to_numpy(dtype=np.float64))

def yarr(idx):
    sub = df.iloc[idx]
    e = (sub["event_type"] == 1).to_numpy()          # readmission is the event of interest
    t = sub["time_to_event"].to_numpy(dtype=float)
    # event-aware whole-day discretization (events capped 29, censoring 30)
    t = np.where(e, np.clip(np.ceil(t), 1, HORIZON_DAYS - 1), HORIZON_DAYS)
    return np.array([(bool(a), float(b)) for a, b in zip(e, t)],
                    dtype=[("event", bool), ("time", float)])

y_train, y_val, y_test = yarr(train_idx_s), yarr(val_idx_s), yarr(test_idx_s)

# ---------------- fit models ----------------
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
import xgboost as xgb

print(f"[{time.time()-t0:.0f}s] fitting XGBoost-AFT...", flush=True)
lo = y_train["time"].copy()
hi = np.where(y_train["event"], y_train["time"], +np.inf)
dtr = xgb.DMatrix(X_train); dtr.set_float_info("label_lower_bound", lo); dtr.set_float_info("label_upper_bound", hi)
lo_v = y_val["time"].copy(); hi_v = np.where(y_val["event"], y_val["time"], +np.inf)
dva = xgb.DMatrix(X_val); dva.set_float_info("label_lower_bound", lo_v); dva.set_float_info("label_upper_bound", hi_v)
params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
          "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0,
          "tree_method": "hist", "learning_rate": 0.05, "max_depth": 5,
          "subsample": 0.9, "colsample_bytree": 0.9, "nthread": N_JOBS, "seed": SEED}
bst = xgb.train(params, dtr, num_boost_round=600, evals=[(dva, "val")],
                early_stopping_rounds=50, verbose_eval=False)
aft_time = bst.predict(xgb.DMatrix(X_test))      # predicted survival time
aft_risk = -aft_time                             # higher = riskier (time-independent marker)
print(f"[{time.time()-t0:.0f}s] AFT done.", flush=True)

print(f"[{time.time()-t0:.0f}s] fitting RSF...", flush=True)
rsf = RandomSurvivalForest(n_estimators=60, max_depth=6, min_samples_leaf=200,
                           max_features="sqrt", max_samples=0.6, n_jobs=N_JOBS, random_state=SEED)
rsf.fit(X_train, y_train)
print(f"[{time.time()-t0:.0f}s] fitting GBS...", flush=True)
gbs = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.05, max_depth=3,
                                       subsample=0.85, random_state=SEED)
gbs.fit(X_train, y_train)
print(f"[{time.time()-t0:.0f}s] models fitted.", flush=True)

# ---------------- daily time-dependent AUROC ----------------
DAYS = np.arange(1.0, 30.0)      # 1..29, strictly inside follow-up (max=30)

def surv_matrix(model, X, times):
    """P(T > t) for each row at each requested time."""
    arr = model.predict_survival_function(X, return_array=True)
    et = model.unique_times_
    out = np.empty((X.shape[0], len(times)))
    for i, t in enumerate(times):
        j = max(np.searchsorted(et, t, side="right") - 1, 0)
        out[:, i] = arr[:, j]
    return out

def daily_auc(estimate):
    """estimate: 1D (time-independent) or 2D (n_samples, n_times)."""
    aucs, _ = cumulative_dynamic_auc(y_train, y_test, estimate, DAYS)
    return np.asarray(aucs, dtype=float)

curves = {}
print(f"[{time.time()-t0:.0f}s] computing daily td-AUROC...", flush=True)
curves["XGBoost-AFT"] = daily_auc(aft_risk)
curves["Random Survival Forest"] = daily_auc(1.0 - surv_matrix(rsf, X_test, DAYS))
curves["Gradient Boosting Survival"] = daily_auc(1.0 - surv_matrix(gbs, X_test, DAYS))

# deployed binary model, restricted to the same alive-discharge test patients
p_bin = np.load(RESULTS / "v7_xgboost_test.npz")["preds"].mean(axis=1)[test_alive_mask]
curves["Binary XGBoost (deployed)"] = daily_auc(p_bin)
print(f"[{time.time()-t0:.0f}s] curves done.", flush=True)

# events per day (context panel)
ev_days = y_test["time"][y_test["event"]]
counts = np.array([(ev_days == d).sum() for d in DAYS])

out = {"days": DAYS.tolist(), "events_per_day": counts.tolist(),
       "curves": {k: v.tolist() for k, v in curves.items()},
       "landmarks_paper": {"day7": 7, "day14": 14, "day29": 29}}
(RESULTS / "survival_daily_tdauc.json").write_text(json.dumps(out, indent=2))

# ---------------- figure ----------------
plt.rcParams.update({"font.size": 12, "font.weight": "bold", "axes.labelsize": 13,
                     "axes.titlesize": 14, "axes.titleweight": "bold", "savefig.dpi": 300})
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
style = {
    "Binary XGBoost (deployed)": ("#8D99AE", "--", "o"),
    "XGBoost-AFT":               ("#E76F51", "-",  "o"),
    "Gradient Boosting Survival":("#1F7A8C", "-",  "s"),
    "Random Survival Forest":    ("#2EC4B6", "-",  "^"),
}
for name in ["Binary XGBoost (deployed)", "XGBoost-AFT",
             "Gradient Boosting Survival", "Random Survival Forest"]:
    c, ls, mk = style[name]
    ax.plot(DAYS, curves[name], ls, color=c, marker=mk, markersize=5,
            lw=2.4, label=name, zorder=4 if "AFT" in name else 3)

for d in (7, 14, 29):
    ax.axvline(d, color="#B0BEC5", ls=":", lw=1.4, zorder=1)
    ax.annotate(f"day {d}", xy=(d, 0.638), fontsize=10, color="#5B7280",
                ha="center", fontweight="bold")
ax.set_ylabel("Time-dependent AUROC")
ax.set_title("Day-by-day discrimination across the 30-day post-discharge window")
ax.set_ylim(0.63, 0.79)
ax.grid(True, axis="y", ls=":", alpha=0.45)
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)

ax2.bar(DAYS, counts, color="#1F7A8C", alpha=0.75, edgecolor="white", linewidth=0.6)
ax2.set_xlabel("Days since discharge")
ax2.set_ylabel("Readmissions\non that day")
ax2.grid(True, axis="y", ls=":", alpha=0.45)
ax2.set_xticks(np.arange(1, 30, 2))
fig.savefig(FIGS / "survival_daily_tdauc.png", bbox_inches="tight")
print("Saved figures/survival_daily_tdauc.png", flush=True)

print("\n=== DAILY td-AUROC ===")
print(f"{'day':>4} {'events':>7} " + " ".join(f"{k[:12]:>13}" for k in curves))
for i, d in enumerate(DAYS):
    print(f"{int(d):>4} {counts[i]:>7} " + " ".join(f"{curves[k][i]:>13.4f}" for k in curves))
print(f"\nTOTAL {time.time()-t0:.0f}s")

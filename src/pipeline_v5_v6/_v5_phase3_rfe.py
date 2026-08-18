"""v5 - leakage-safe staged RFE + comparators, DEVELOPMENT PARTITIONS ONLY.

Changes from v4 (both from the 2026-08-12 external audit):
  1. discharge_location_te REMOVED from the eligible pool. Empirical provenance:
     out-of-fold target encoding computed over train+val with a fixed map applied
     to test, folds not aligned with this pipeline's folds -> outcome leakage
     into nested-CV estimates. The raw discharge_location column stays and is
     target-encoded fold-locally like every other categorical.
  2. Nested selection runs on the DEVELOPMENT partitions (train+val) only, so
     the consensus feature set is chosen without any test-partition outcomes and
     the tertiary test evaluation is selection-independent.

Everything else is identical to v4 by design: 5 outer GroupKFold on subject_id,
3 grouped inner folds, fold-local smoothed target encodings (smoothing 20),
staged elimination grid, prespecified parsimony margin 0.002, seed 42,
XGBoost 400 trees / lr .05 / depth 5 / subsample+colsample .9 / hist.

Outputs -> results_v5/: eligible_pool_v5.json, rfe_trajectories.json,
fold_selected_features.json, oof_predictions.npz, model_comparison.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "8"
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.linear_model import LogisticRegression

t0 = time.time()
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT4 = REPO / "results_v4"
OUT = REPO / "results_v5"
OUT.mkdir(exist_ok=True)
R2 = REPO / "results_reanalysis"
ART = PUB / "readmission-api" / "artifacts"
SEED = 42
rng = np.random.RandomState(SEED)
MARGIN = 0.002
PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=8,
              eval_metric="auc", tree_method="hist", verbosity=0)


def log(m):
    print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)


# ---------------------------------------------------------------- pool v5
P4 = json.loads((OUT4 / "eligible_pool.json").read_text())
LEAK = "discharge_location_te"
assert LEAK in P4["eligible"]
ELIG = [f for f in P4["eligible"] if f != LEAK]
P5 = {"eligible": ELIG, "n_eligible": len(ELIG), "pool_total": P4["pool_total"],
      "excluded": P4["excluded"] + [{
          "feature": LEAK,
          "reason": ("excluded: precomputed outcome-derived encoding; empirical "
                     "audit shows out-of-fold encoding over train+val with a "
                     "fixed train+val map applied to test, fold structure not "
                     "aligned with this pipeline (leakage). Raw "
                     "discharge_location retained with fold-local encoding.")}]}
(OUT / "eligible_pool_v5.json").write_text(json.dumps(P5, indent=1))
GRID = [len(ELIG), 120, 100, 80, 65, 55, 45, 38, 32, 27, 22, 18]
log(f"eligible pool v5: {len(ELIG)} (v4 minus {LEAK})")

# ---------------------------------------------------------------- data
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
C = pd.read_parquet(R2 / "cohort_v2.parquet")
keep = ((~C["excl_index_death"])
        & C["partition"].isin(["train", "val"])).to_numpy()   # DEV ONLY
y_all = C["label_v2"].to_numpy()
s_all = C["subject_id"].to_numpy()

D = DF.loc[keep].reset_index(drop=True)
y = y_all[keep]
groups = s_all[keep]
log(f"development cohort {len(D):,} admissions, "
    f"{len(np.unique(groups)):,} patients, prevalence {y.mean():.4f}")

serving = joblib.load(ART / "serving_artifacts.joblib")
F66 = list(serving["feature_order"])
F50 = [f for f in F66 if f in set(ELIG)]   # billing-free AND leakage-free subset
log(f"comparators: 66-feature historical (billing+leaked TE) and "
    f"{len(F50)}-feature billing-free leakage-free subset")

OBJ = [c for c in ELIG if not pd.api.types.is_numeric_dtype(D[c])]
log(f"object columns target-encoded fold-locally: {OBJ}")


def encode_fit(cols, idx_fit):
    maps = {}
    gmean = y[idx_fit].mean()
    for c in cols:
        cats = D[c].astype(str).fillna("__NA__")
        g = pd.DataFrame({"c": cats.iloc[idx_fit], "y": y[idx_fit]}) \
            .groupby("c")["y"].agg(["mean", "count"])
        maps[c] = ((g["mean"] * g["count"] + gmean * 20) / (g["count"] + 20)), gmean
    return maps


def encode_apply(cols, maps, frame):
    X = frame.copy()
    for c in cols:
        sm, gmean = maps[c]
        X[c] = X[c].astype(str).fillna("__NA__").map(sm).fillna(gmean).astype(float)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def fit_score(feats, idx_tr, idx_ev, return_model=False, return_gain=False):
    obj = [c for c in feats if not pd.api.types.is_numeric_dtype(D[c])]
    maps = encode_fit(obj, idx_tr)
    Xtr = encode_apply(obj, maps, D[feats].iloc[idx_tr]).to_numpy(np.float32)
    Xev = encode_apply(obj, maps, D[feats].iloc[idx_ev]).to_numpy(np.float32)
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(Xtr, y[idx_tr])
    p = m.predict_proba(Xev)[:, 1]
    a = roc_auc_score(y[idx_ev], p)
    gain = None
    if return_gain:
        sc = m.get_booster().get_score(importance_type="gain")
        gain = np.array([sc.get(f"f{i}", 0.0) for i in range(len(feats))])
    return (a, p, m if return_model else None, gain)


# ---------------------------------------------------------------- outer loop
outer = GroupKFold(n_splits=5)
oof = {k: np.full(len(D), np.nan)
       for k in ("rfe", "all_eligible", "f50", "f66", "logit")}
fold_assign = np.full(len(D), -1)
trajectories, fold_feats, fold_counts = {}, {}, {}

for k, (otr, oev) in enumerate(outer.split(D, y, groups=groups), 1):
    fold_assign[oev] = k
    log(f"=== outer fold {k}: train {len(otr):,} eval {len(oev):,} ===")
    inner = GroupKFold(n_splits=3)
    inner_folds = list(inner.split(otr, y[otr], groups=groups[otr]))
    feats = list(ELIG)
    traj = []
    for target_n in GRID:
        aucs, gains = [], np.zeros(len(feats))
        for itr_i, iva_i in inner_folds:
            a, _, _, g = fit_score(feats, otr[itr_i], otr[iva_i],
                                   return_gain=True)
            aucs.append(a)
            gains += g
        traj.append({"n": len(feats), "mean_auc": round(float(np.mean(aucs)), 5),
                     "sd": round(float(np.std(aucs)), 5)})
        log(f"  n={len(feats):3d}  inner AUROC {np.mean(aucs):.4f} "
            f"+/- {np.std(aucs):.4f}")
        nxt = next((g2 for g2 in GRID if g2 < len(feats)), None)
        if nxt is None:
            break
        order = np.argsort(gains)[::-1]
        feats = [feats[i] for i in order[:nxt]]
    best = max(t["mean_auc"] for t in traj)
    chosen = min((t["n"] for t in traj if t["mean_auc"] >= best - MARGIN))
    feats = list(ELIG)
    for t_i, target_n in enumerate(GRID):
        if len(feats) == chosen:
            break
        gains = np.zeros(len(feats))
        for itr_i, iva_i in inner_folds:
            _, _, _, g = fit_score(feats, otr[itr_i], otr[iva_i], return_gain=True)
            gains += g
        nxt = next((g2 for g2 in GRID if g2 < len(feats)), None)
        if nxt is None or nxt < chosen:
            order = np.argsort(gains)[::-1]
            feats = [feats[i] for i in order[:chosen]]
            break
        order = np.argsort(gains)[::-1]
        feats = [feats[i] for i in order[:nxt]]
    fold_feats[k] = feats
    fold_counts[k] = chosen
    trajectories[k] = traj
    log(f"  fold {k} selected n={chosen} (rule: within {MARGIN} of best {best:.4f})")

    a, p, _, _ = fit_score(feats, otr, oev)
    oof["rfe"][oev] = p
    log(f"  outer-eval RFE AUROC {a:.4f}")
    for name, fl in (("all_eligible", ELIG), ("f50", F50), ("f66", F66)):
        a2, p2, _, _ = fit_score(fl, otr, oev)
        oof[name][oev] = p2
        log(f"  outer-eval {name} AUROC {a2:.4f}")
    obj = [c for c in ELIG if not pd.api.types.is_numeric_dtype(D[c])]
    maps = encode_fit(obj, otr)
    Xtr = encode_apply(obj, maps, D[ELIG].iloc[otr])
    Xev = encode_apply(obj, maps, D[ELIG].iloc[oev])
    med = Xtr.median(numeric_only=True)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    Xtr = ((Xtr.fillna(med) - mu) / sd).to_numpy(np.float32)
    Xev = ((Xev.fillna(med) - mu) / sd).to_numpy(np.float32)
    lr = LogisticRegression(C=1.0, max_iter=2000, n_jobs=8)
    lr.fit(Xtr, y[otr])
    p3 = lr.predict_proba(Xev)[:, 1]
    oof["logit"][oev] = p3
    log(f"  outer-eval logit AUROC {roc_auc_score(y[oev], p3):.4f}")

np.savez(OUT / "oof_predictions.npz", y=y, groups=groups,
         fold=fold_assign, **oof)
(OUT / "rfe_trajectories.json").write_text(json.dumps(trajectories, indent=1))
(OUT / "fold_selected_features.json").write_text(json.dumps(
    {str(k): v for k, v in fold_feats.items()}, indent=1))


# ---------------------------------------------------------------- metrics
def ecefun(yy, pp, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    idx = np.digitize(pp, bins) - 1
    return float(sum((idx == b).mean() * abs(pp[idx == b].mean() - yy[idx == b].mean())
                     for b in range(nb) if (idx == b).sum()))


def slope_int(yy, pp):
    eps = 1e-6
    lo = np.log(np.clip(pp, eps, 1 - eps) / np.clip(1 - pp, eps, 1 - eps)) \
        .reshape(-1, 1)
    m = LogisticRegression(C=1e6).fit(lo, yy)
    return float(m.coef_[0][0]), float(m.intercept_[0])


uniq = np.unique(groups)
rowsmap = {s: np.where(groups == s)[0] for s in uniq}
comp = {}
BOOT_IX = []          # shared bootstrap draws -> paired differences
for b in range(500):
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    BOOT_IX.append(np.concatenate([rowsmap[s] for s in pick]))
for name, pp in oof.items():
    a = roc_auc_score(y, pp)
    boots = []
    for ix in BOOT_IX:
        if len(np.unique(y[ix])) < 2:
            continue
        boots.append(roc_auc_score(y[ix], pp[ix]))
    sl, ic = slope_int(y, pp)
    k10 = int(len(pp) * 0.10)
    top10 = np.argsort(pp)[::-1][:k10]
    comp[name] = {
        "n_features": ({"rfe": f"per-fold {sorted(set(fold_counts.values()))}",
                        "all_eligible": len(ELIG), "f50": len(F50),
                        "f66": len(F66), "logit": len(ELIG)}[name]),
        "oof_auroc": round(float(a), 4),
        "auroc_ci95_cluster": [round(float(np.percentile(boots, 2.5)), 4),
                               round(float(np.percentile(boots, 97.5)), 4)],
        "ap": round(float(average_precision_score(y, pp)), 4),
        "brier": round(float(brier_score_loss(y, pp)), 4),
        "ece": round(ecefun(y, pp), 4),
        "cal_slope": round(sl, 3), "cal_intercept": round(ic, 3),
        "top10pct_events_captured": round(float(y[top10].sum() / y.sum()), 3),
        "top10pct_ppv": round(float(y[top10].mean()), 3)}
    log(f"{name}: OOF AUROC {a:.4f} {comp[name]['auroc_ci95_cluster']}")

# paired cluster-bootstrap AUROC differences vs the RFE procedure (same draws,
# same observations) - replaces any 'statistically indistinguishable' language
paired = {}
for name in ("all_eligible", "f50", "f66", "logit"):
    diffs = []
    for ix in BOOT_IX:
        if len(np.unique(y[ix])) < 2:
            continue
        diffs.append(roc_auc_score(y[ix], oof[name][ix])
                     - roc_auc_score(y[ix], oof["rfe"][ix]))
    d = np.array(diffs)
    p2s = 2 * min((d <= 0).mean(), (d >= 0).mean())
    paired[f"{name}_minus_rfe"] = {
        "point": round(float(roc_auc_score(y, oof[name])
                             - roc_auc_score(y, oof["rfe"])), 4),
        "ci95": [round(float(np.percentile(d, 2.5)), 4),
                 round(float(np.percentile(d, 97.5)), 4)],
        "p_bootstrap_2sided": round(float(min(1.0, max(p2s, 2 / len(d)))), 4)}
    log(f"paired {name}-rfe: {paired[f'{name}_minus_rfe']}")
comp["_paired_differences_vs_rfe"] = paired

# stability
sets = [set(v) for v in fold_feats.values()]
from itertools import combinations
jac = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
freq = {}
for s in sets:
    for f in s:
        freq[f] = freq.get(f, 0) + 1
consensus = sorted([f for f, c in freq.items() if c >= 3])
comp["_stability"] = {
    "fold_counts": fold_counts,
    "jaccard_mean": round(float(np.mean(jac)), 3),
    "jaccard_pairwise": [round(j, 3) for j in jac],
    "selection_frequency": dict(sorted(freq.items(), key=lambda kv: -kv[1])),
    "consensus_majority_features": consensus,
    "n_consensus": len(consensus)}
comp["_design"] = {
    "cohort": "development partitions only (train+val), test untouched",
    "n_dev": int(len(D)), "n_patients_dev": int(len(uniq)),
    "removed_from_pool": LEAK, "grid": GRID, "margin": MARGIN, "seed": SEED}
(OUT / "model_comparison.json").write_text(json.dumps(comp, indent=1))
log(f"stability: mean Jaccard {np.mean(jac):.3f}; consensus(>=3/5) "
    f"{len(consensus)} features")
log("DONE")

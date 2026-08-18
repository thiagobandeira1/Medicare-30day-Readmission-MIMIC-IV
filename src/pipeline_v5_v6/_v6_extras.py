"""v6 supplementary analyses (fifth external review).

Outputs -> results_v6/v6_extras.json + results_v6/final_model_v6.json
(canonical: v5 final-model JSON merged with v6 additions/corrections).

 1. Fixed-31 vs fixed-142 paired grouped CV on development data (same 5
    grouped folds, fold-local TE) -> direct fixed-vs-fixed paired difference.
 2. High-draw (5,000) patient-cluster contrasts with plus-one correction:
    LACE uplift, HOSPITAL-12m uplift (paired same-draw), White-Black AUROC
    gap (stratified: patients resampled within each subgroup).
 3. Death/readmission same-date tie-rule sensitivity: death-first (primary)
    vs readmission-first; AFT Harrell C, day-1/29 td-AUROC, AJ 30-day CIF.
 4. Transfer-band documentation: exact bounds and counts by partition.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "8"
import json, time, copy
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

t0 = time.time()
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
O5 = REPO / "results_v5"
O6 = REPO / "results_v6"
O6.mkdir(exist_ok=True)
R2 = REPO / "results_reanalysis"
SEED = 42
rng = np.random.RandomState(SEED)
PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=8,
              eval_metric="auc", tree_method="hist", verbosity=0)


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


FM = json.loads((O5 / "final_model_v5.json").read_text())
POOL = json.loads((O5 / "eligible_pool_v5.json").read_text())
F31 = FM["final_feature_set"]
ELIG = POOL["eligible"]

V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
C = pd.read_parquet(R2 / "cohort_v2.parquet")
keep = (~C["excl_index_death"]).to_numpy()
part = C["partition"].to_numpy()
y = C["label_v2"].to_numpy()
subj = C["subject_id"].to_numpy()
dev = keep & ((part == "train") | (part == "val"))


def te_encode(feats, idx_fit, idx_apply_list):
    obj = [c for c in feats if not pd.api.types.is_numeric_dtype(DF[c])]
    gmean = y[idx_fit].mean()
    maps = {}
    for c in obj:
        cats = DF[c].astype(str).fillna("__NA__")
        g = pd.DataFrame({"c": cats.iloc[idx_fit], "y": y[idx_fit]}) \
            .groupby("c")["y"].agg(["mean", "count"])
        maps[c] = ((g["mean"] * g["count"] + gmean * 20) / (g["count"] + 20),
                   gmean)
    outs = []
    for idx in idx_apply_list:
        X = DF[feats].iloc[idx].copy()
        for c in obj:
            sm, gm = maps[c]
            X[c] = X[c].astype(str).fillna("__NA__").map(sm).fillna(gm) \
                .astype(float)
        X = X.apply(pd.to_numeric, errors="coerce") \
            .replace([np.inf, -np.inf], np.nan)
        outs.append(X.to_numpy(np.float32))
    return outs


# ================ 1. fixed-31 vs fixed-142 paired grouped CV (dev) ==========
devidx = np.where(dev)[0]
ydev = y[devidx]; sdev = subj[devidx]
p31 = np.full(len(devidx), np.nan)
p142 = np.full(len(devidx), np.nan)
for f, (itr, ite) in enumerate(
        GroupKFold(5).split(devidx, ydev, groups=sdev), 1):
    tr_g, te_g = devidx[itr], devidx[ite]
    X31tr, X31te = te_encode(F31, tr_g, [tr_g, te_g])
    m1 = xgb.XGBClassifier(**PARAMS); m1.fit(X31tr, y[tr_g])
    p31[ite] = m1.predict_proba(X31te)[:, 1]
    Xatr, Xate = te_encode(ELIG, tr_g, [tr_g, te_g])
    m2 = xgb.XGBClassifier(**PARAMS); m2.fit(Xatr, y[tr_g])
    p142[ite] = m2.predict_proba(Xate)[:, 1]
    log(f"fixed-CV fold {f}: 31f {roc_auc_score(y[te_g], p31[ite]):.4f} "
        f"142f {roc_auc_score(y[te_g], p142[ite]):.4f}")

uniq_d = np.unique(sdev)
rows_d = {s: np.where(sdev == s)[0] for s in uniq_d}
B = 5000
diffs = np.empty(B)
kept = 0
for b in range(B):
    pick = rng.choice(uniq_d, size=len(uniq_d), replace=True)
    ix = np.concatenate([rows_d[s] for s in pick])
    if len(np.unique(ydev[ix])) < 2:
        continue
    diffs[kept] = (roc_auc_score(ydev[ix], p31[ix])
                   - roc_auc_score(ydev[ix], p142[ix]))
    kept += 1
diffs = diffs[:kept]
p2 = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
p2 = min(1.0, max(p2, 2 / (kept + 1)))
fixed_cmp = {
    "auroc_fixed31_cv": round(float(roc_auc_score(ydev, p31)), 4),
    "auroc_fixed142_cv": round(float(roc_auc_score(ydev, p142)), 4),
    "paired_diff_31_minus_142": round(float(
        roc_auc_score(ydev, p31) - roc_auc_score(ydev, p142)), 4),
    "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
             round(float(np.percentile(diffs, 97.5)), 4)],
    "p_two_sided_plus1": round(float(p2), 4), "n_boot": int(kept),
    "note": ("both fixed sets defined after selection, so both carry "
             "selection optimism; the DIFFERENCE is a fair fixed-vs-fixed "
             "contrast on identical development folds")}
log(f"fixed 31 vs 142: {fixed_cmp}")

# ================ 2. high-draw test-partition contrasts =====================
P5 = np.load(O5 / "final_preds_v5.npz")
p_te, y_te, s_te = P5["p_te"], P5["y_te"], P5["s_te"]
BS = np.load(R2 / "baselines_v2_scores.npz")
lace = BS["lace"]
# HOSPITAL-12m: reconstruct identically to phase57
ADM2 = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                       columns=["subject_id", "hadm_id", "admittime"])
ADM2 = ADM2.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
cnt12 = np.zeros(len(ADM2), dtype=int)
for _s, g in ADM2.groupby("subject_id", sort=False):
    t = g["admittime"].values.astype("datetime64[D]").astype("int64")
    cnt12[g.index.values] = np.arange(len(t)) - np.searchsorted(t, t - 365, "left")
ADM2["adm12"] = cnt12
hadm_te = C.loc[keep & (part == "test"), "hadm_id"].to_numpy()
dd2 = pd.DataFrame({"hadm_id": hadm_te}).merge(
    V10[["hadm_id", "los_days", "admission_type", "hemoglobin_last",
         "sodium_last", "n_procedures"]], on="hadm_id").merge(
    ADM2[["hadm_id", "adm12"]], on="hadm_id")
svc = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "services.parquet",
                      columns=["hadm_id", "curr_service"])
onc = dd2["hadm_id"].isin(set(svc.loc[svc["curr_service"] == "OMED",
                                      "hadm_id"])).astype(int) * 2
EL = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}
h12 = ((dd2["hemoglobin_last"] < 12).astype(int) + onc
       + (dd2["sodium_last"] < 135).astype(int)
       + (dd2["n_procedures"] >= 1).astype(int)
       + (~dd2["admission_type"].isin(EL)).astype(int)
       + dd2["adm12"].fillna(0).apply(lambda n: 0 if n <= 1 else (2 if n <= 5 else 5))
       + (dd2["los_days"] >= 5).astype(int) * 2).to_numpy()

uniq = np.unique(s_te)
rows = {s: np.where(s_te == s)[0] for s in uniq}
dl = []; dh = []
for b in range(B):
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    ix = np.concatenate([rows[s] for s in pick])
    yy = y_te[ix]
    if len(np.unique(yy)) < 2:
        continue
    am = roc_auc_score(yy, p_te[ix])
    dl.append(am - roc_auc_score(yy, lace[ix]))
    dh.append(am - roc_auc_score(yy, h12[ix]))


def summ(d):
    d = np.asarray(d)
    p2 = 2 * min((d <= 0).mean(), (d >= 0).mean())
    p2 = min(1.0, max(p2, 2 / (len(d) + 1)))
    return {"point": round(float(np.mean(d)), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "p_two_sided_plus1": float(f"{p2:.2g}"), "n_boot": int(len(d)),
            "design": "paired same-draw patient-cluster bootstrap"}


hd = {"lace_uplift": summ(dl), "hospital12_uplift": summ(dh)}
log(f"highdraw LACE {hd['lace_uplift']}")
log(f"highdraw HOSP12 {hd['hospital12_uplift']}")

# White-Black gap: stratified within-subgroup patient resampling
ADM = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                      columns=["hadm_id", "race"])
rr = pd.DataFrame({"hadm_id": hadm_te}).merge(ADM, on="hadm_id")["race"].astype(str)
race = np.select([rr.str.contains("WHITE"), rr.str.contains("BLACK")],
                 ["White", "Black"], "other")
wb_diffs = []
grp_rows = {}
for gname in ("White", "Black"):
    m = race == gname
    sg = s_te[m]
    ug = np.unique(sg)
    grp_rows[gname] = (np.where(m)[0], sg, ug,
                       {s: np.where(sg == s)[0] for s in ug})
for b in range(B):
    ok = True
    aucs = {}
    for gname in ("White", "Black"):
        gidx, sg, ug, rmap = grp_rows[gname]
        pick = rng.choice(ug, size=len(ug), replace=True)
        ix = gidx[np.concatenate([rmap[s] for s in pick])]
        yy = y_te[ix]
        if len(np.unique(yy)) < 2:
            ok = False
            break
        aucs[gname] = roc_auc_score(yy, p_te[ix])
    if ok:
        wb_diffs.append(aucs["White"] - aucs["Black"])
wb6 = summ(wb_diffs)
wb6["design"] = ("stratified patient-cluster bootstrap: patients resampled "
                 "within each subgroup independently; disjoint groups, not a "
                 "paired comparison")
hd["white_minus_black"] = wb6
log(f"highdraw W-B {wb6}")

# ================ 3. tie-rule sensitivity ===================================
EV = pd.read_parquet(R2 / "cohort_v3_events.parquet")
lab = C.set_index("hadm_id")["label_v2"]
ev_lab = EV["hadm_id"].map(lab).to_numpy()
tie = (EV["event_v3"].to_numpy() == 2) & (ev_lab == 1)
log(f"tie cases (event=death but binary readmission exists): "
    f"{int(tie.sum())} total, "
    f"{int((tie & (EV['partition']=='test').to_numpy()).sum())} on test")

D2 = DF.loc[DF["hadm_id"].isin(set(EV["hadm_id"]))].reset_index(drop=True)
assert (D2["hadm_id"].values == EV["hadm_id"].values).all()
maps5 = json.loads((O5 / "artifacts_v5" / "te_maps.json").read_text())
X31 = D2[F31].copy()
for c, mv in maps5.items():
    X31[c] = X31[c].astype(str).fillna("__NA__").map(mv["map"]) \
        .fillna(mv["global"]).astype(float)
X31 = X31.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
pt2 = EV["partition"].to_numpy()
tr2, va2, te2 = pt2 == "train", pt2 == "val", pt2 == "test"
from lifelines.utils import concordance_index
from lifelines import AalenJohansenFitter
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

tie_res = {}
for rule, flip in (("death_first_primary", False), ("readmission_first", True)):
    event = EV["event_v3"].to_numpy().copy()
    tt = EV["time_v3"].to_numpy().copy()
    if flip:
        event[tie] = 1
    lo2 = tt.copy(); hi2 = np.where(event == 1, tt, np.inf)

    def dmat(m):
        d = xgb.DMatrix(X31.loc[m].to_numpy(np.float32))
        d.set_float_info("label_lower_bound", lo2[m])
        d.set_float_info("label_upper_bound", hi2[m])
        return d

    ap_ = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
           "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0,
           "tree_method": "hist", "learning_rate": 0.05, "max_depth": 5,
           "subsample": 0.9, "colsample_bytree": 0.9, "seed": SEED, "nthread": 8}
    bst = xgb.train(ap_, dmat(tr2), num_boost_round=600,
                    evals=[(dmat(va2), "val")], early_stopping_rounds=50,
                    verbose_eval=False)
    predt = bst.predict(xgb.DMatrix(X31.loc[te2].to_numpy(np.float32)))
    hc = concordance_index(tt[te2], predt, (event[te2] == 1).astype(int))
    S_tr = Surv.from_arrays((event[tr2] == 1), tt[tr2])
    S_te = Surv.from_arrays((event[te2] == 1), tt[te2])
    days = np.arange(1, 30, 1.0)
    auc_d, _ = cumulative_dynamic_auc(S_tr, S_te, -predt, days)
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(tt[te2], event[te2], event_of_interest=1)
    cif30 = float(aj.cumulative_density_.iloc[-1, 0])
    aj2 = AalenJohansenFitter(calculate_variance=False)
    aj2.fit(tt[te2], event[te2], event_of_interest=2)
    cifd30 = float(aj2.cumulative_density_.iloc[-1, 0])
    tie_res[rule] = {"test_readmissions": int((event[te2] == 1).sum()),
                     "test_deaths": int((event[te2] == 2).sum()),
                     "harrell_c": round(float(hc), 4),
                     "tdauc_day1": round(float(auc_d[0]), 4),
                     "tdauc_day29": round(float(auc_d[-1]), 4),
                     "aj_cif30_readmission": round(cif30, 4),
                     "aj_cif30_death": round(cifd30, 4)}
    log(f"tie rule {rule}: {tie_res[rule]}")
tie_res["n_ties"] = int(tie.sum())
tie_res["n_ties_test"] = int((tie & te2).sum())
tie_res["note"] = ("registry death dates are day-granular; same-date "
                   "readmission/death order is not observable, so both "
                   "orderings are reported")

# ================ 4. transfer band ==========================================
d = C["delta_days"]
band = keep & d.notna().to_numpy() & (d <= 0).to_numpy() & (d > -2).to_numpy()
deep = keep & d.notna().to_numpy() & (d <= -2).to_numpy()
tb = {"delta_definition": ("next admittime minus index dischtime, exact "
                           "timestamps, in fractional days"),
      "primary_label": "readmission iff 0 < delta <= 30",
      "transfer_band": "-2 < delta <= 0 (treated as transfer/continuation)",
      "n_band_total": int(band.sum()),
      "n_band_by_partition": {p_: int((band & (part == p_)).sum())
                              for p_ in ("train", "val", "test")},
      "n_delta_below_minus2": int(deep.sum()),
      "rationale": ("a subsequent record beginning at or within 48 hours "
                    "before the index discharge timestamp reflects "
                    "inter-unit transfer or administrative re-registration, "
                    "not a community return; no records fall below the "
                    "-2-day bound in this cohort")}

EX = {"fixed31_vs_fixed142_paired_cv": fixed_cmp,
      "highdraw_contrasts_5000": hd,
      "tie_rule_sensitivity": tie_res,
      "transfer_band": tb,
      "boot_draws": B, "seed": SEED}
(O6 / "v6_extras.json").write_text(json.dumps(EX, indent=1))

# canonical v6 results file: v5 merged with corrections/additions
FM6 = copy.deepcopy(FM)
FM6["fairness"]["white_minus_black"] = {
    "point": FM["fairness"]["white_minus_black"]["point"],
    "ci95": wb6["ci95"], "p_bootstrap": wb6["p_two_sided_plus1"],
    "n_boot": wb6["n_boot"], "design": wb6["design"]}
FM6["baselines"]["uplift_lace_ci"] = hd["lace_uplift"]["ci95"]
FM6["baselines"]["uplift_lace_p"] = hd["lace_uplift"]["p_two_sided_plus1"]
FM6["baselines"]["uplift_hospital12_ci"] = hd["hospital12_uplift"]["ci95"]
FM6["baselines"]["uplift_hospital12_p"] = hd["hospital12_uplift"]["p_two_sided_plus1"]
FM6["v6_extras"] = EX
(O6 / "final_model_v6.json").write_text(json.dumps(FM6, indent=1))
log("final_model_v6.json + v6_extras.json written - DONE")

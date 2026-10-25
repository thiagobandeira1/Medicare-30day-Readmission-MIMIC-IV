"""v7 supplementary analyses -> canonical results file.

Outputs -> results_v12/extras_v12.json + results_v12/final_model_v12.json
(canonical: v7 base merged with these additions under key "extras").

 1. Fixed consensus vs fixed eligible-pool paired grouped CV (development
    data; same 5 grouped folds; fold-local encoding).
 2. High-draw (5,000) patient-cluster contrasts with plus-one correction:
    LACE uplift, HOSPITAL-12m uplift (paired same-draw), White vs Black
    AUROC gap (stratified: patients resampled within each subgroup).
 3. Same-date readmission/death tie-rule sensitivity (death-first primary
    vs readmission-first), using the canonical tdAUC estimand.
 4. Transfer-band documentation.
 5. Era-stratified test AUROC and payer scope of outcome ascertainment.
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
O7 = REPO / "results_v12"
R2 = REPO / "results_reanalysis"
D = PUB / "Dataset" / "mimic-parquet"
SEED = 42
rng = np.random.RandomState(SEED)
PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=8,
              eval_metric="auc", tree_method="hist", verbosity=0)


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


FM = json.loads((O7 / "final_model_v12_base.json").read_text())
POOL = json.loads((O7 / "eligible_pool_v12.json").read_text())
FC = FM["final_feature_set"]
ELIG = POOL["eligible"]

V7 = pd.read_parquet(PUB / "training_table_v12.parquet")
V10 = pd.read_parquet(PUB / "training_table_v12.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
C = pd.read_parquet(R2 / "cohort_v2.parquet")
assert (DF["hadm_id"].values == C["hadm_id"].values).all()
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


# ================ 1. fixed consensus vs fixed eligible pool (dev) ===========
devidx = np.where(dev)[0]
ydev = y[devidx]; sdev = subj[devidx]
pc = np.full(len(devidx), np.nan)
pe = np.full(len(devidx), np.nan)
for f, (itr, ite) in enumerate(
        GroupKFold(5).split(devidx, ydev, groups=sdev), 1):
    tr_g, te_g = devidx[itr], devidx[ite]
    Xa, Xb = te_encode(FC, tr_g, [tr_g, te_g])
    m1 = xgb.XGBClassifier(**PARAMS); m1.fit(Xa, y[tr_g])
    pc[ite] = m1.predict_proba(Xb)[:, 1]
    Xa, Xb = te_encode(ELIG, tr_g, [tr_g, te_g])
    m2 = xgb.XGBClassifier(**PARAMS); m2.fit(Xa, y[tr_g])
    pe[ite] = m2.predict_proba(Xb)[:, 1]
    log(f"fixed-CV fold {f}: consensus {roc_auc_score(y[te_g], pc[ite]):.4f} "
        f"eligible {roc_auc_score(y[te_g], pe[ite]):.4f}")

uniq_d = np.unique(sdev)
rows_d = {s: np.where(sdev == s)[0] for s in uniq_d}
B = 5000
diffs = []
for b in range(B):
    pick = rng.choice(uniq_d, size=len(uniq_d), replace=True)
    ix = np.concatenate([rows_d[s] for s in pick])
    if len(np.unique(ydev[ix])) < 2:
        continue
    diffs.append(roc_auc_score(ydev[ix], pc[ix]) - roc_auc_score(ydev[ix], pe[ix]))
diffs = np.array(diffs)
p2 = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
p2 = min(1.0, max(p2, 2 / (len(diffs) + 1)))
fixed_cmp = {
    "n_consensus": len(FC), "n_eligible": len(ELIG),
    "auroc_consensus_cv": round(float(roc_auc_score(ydev, pc)), 4),
    "auroc_eligible_cv": round(float(roc_auc_score(ydev, pe)), 4),
    "paired_diff_consensus_minus_eligible": round(float(
        roc_auc_score(ydev, pc) - roc_auc_score(ydev, pe)), 4),
    "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
             round(float(np.percentile(diffs, 97.5)), 4)],
    "p_two_sided_plus1": round(float(p2), 4), "n_boot": int(len(diffs))}
log(f"fixed consensus vs eligible: {fixed_cmp}")

# ================ 2. high-draw test-partition contrasts =====================
P7 = np.load(O7 / "final_preds_v12.npz")
p_te, y_te, s_te = P7["p_te"], P7["y_te"], P7["s_te"]
BS = np.load(O7 / "baseline_scores_test_v12.npz")
lace, h12 = BS["lace"], BS["hospital12"]
assert np.array_equal(BS["y_te"], y_te)
uniq = np.unique(s_te)
rows = {s: np.where(s_te == s)[0] for s in uniq}
dl, dh = [], []
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
hadm_te = C.loc[keep & (part == "test"), "hadm_id"].to_numpy()
ADM = pd.read_parquet(D / "admissions.parquet", columns=["hadm_id", "race"])
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
wb7 = summ(wb_diffs)
wb7["design"] = ("stratified patient-cluster bootstrap: patients resampled "
                 "within each subgroup independently; disjoint groups, not a "
                 "paired comparison")
hd["white_minus_black"] = wb7
log(f"highdraw W-B {wb7}")

# ================ 3. tie-rule sensitivity ===================================
EV = pd.read_parquet(R2 / "cohort_v3_events.parquet")
lab = C.set_index("hadm_id")["label_v2"]
ev_lab = EV["hadm_id"].map(lab).to_numpy()
tie = (EV["event_v3"].to_numpy() == 2) & (ev_lab == 1)
D2 = DF.loc[DF["hadm_id"].isin(set(EV["hadm_id"]))].reset_index(drop=True)
assert (D2["hadm_id"].values == EV["hadm_id"].values).all()
mapsj = json.loads((O7 / "artifacts_v12" / "te_maps.json").read_text())
XC = D2[FC].copy()
for c, mv in mapsj.items():
    XC[c] = XC[c].astype(str).fillna("__NA__").map(mv["map"]) \
        .fillna(mv["global"]).astype(float)
XC = XC.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
pt2 = EV["partition"].to_numpy()
tr2, va2, te2 = pt2 == "train", pt2 == "val", pt2 == "test"
from lifelines.utils import concordance_index
from lifelines import AalenJohansenFitter
tie_res = {}
for rule, flip in (("death_first_primary", False), ("readmission_first", True)):
    event = EV["event_v3"].to_numpy().copy()
    tt = EV["time_v3"].to_numpy().copy()
    if flip:
        event[tie] = 1
    lo2 = tt.copy(); hi2 = np.where(event == 1, tt, np.inf)

    def dmat(m):
        d = xgb.DMatrix(XC.loc[m].to_numpy(np.float32))
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
    predt = bst.predict(xgb.DMatrix(XC.loc[te2].to_numpy(np.float32)))
    hc = concordance_index(tt[te2], predt, (event[te2] == 1).astype(int))
    ev_te, tt_te = event[te2], tt[te2]
    a1 = roc_auc_score(((ev_te == 1) & (tt_te <= 1)).astype(int), -predt)
    a29 = roc_auc_score(((ev_te == 1) & (tt_te <= 29)).astype(int), -predt)
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(tt[te2], event[te2], event_of_interest=1)
    aj2 = AalenJohansenFitter(calculate_variance=False)
    aj2.fit(tt[te2], event[te2], event_of_interest=2)
    tie_res[rule] = {"test_readmissions": int((ev_te == 1).sum()),
                     "test_deaths": int((ev_te == 2).sum()),
                     "harrell_c": round(float(hc), 4),
                     "tdauc_day1": round(float(a1), 4),
                     "tdauc_day29": round(float(a29), 4),
                     "aj_cif30_readmission":
                         round(float(aj.cumulative_density_.iloc[-1, 0]), 4),
                     "aj_cif30_death":
                         round(float(aj2.cumulative_density_.iloc[-1, 0]), 4)}
    log(f"tie rule {rule}: {tie_res[rule]}")
tie_res["n_ties"] = int(tie.sum())
tie_res["n_ties_test"] = int((tie & te2).sum())

# ================ 4. transfer band ==========================================
d = C["delta_days"]
band = keep & d.notna().to_numpy() & (d <= 0).to_numpy() & (d > -2).to_numpy()
deep = keep & d.notna().to_numpy() & (d <= -2).to_numpy()
tb = {"primary_label": "readmission iff 0 < delta <= 30",
      "transfer_band": "-2 < delta <= 0 (treated as transfer/continuation)",
      "n_band_total": int(band.sum()),
      "n_band_by_partition": {p_: int((band & (part == p_)).sum())
                              for p_ in ("train", "val", "test")},
      "n_delta_below_minus2": int(deep.sum())}

# ================ 5. era bands + payer scope ================================
pat = pd.read_parquet(D / "patients.parquet",
                      columns=["subject_id", "anchor_year_group"])
mm = pd.DataFrame({"subject_id": s_te, "p": p_te, "y": y_te}) \
    .merge(pat, on="subject_id", how="left")
assert mm["anchor_year_group"].notna().all()
bands = []
for g, gdf in sorted(mm.groupby("anchor_year_group"), key=lambda kv: kv[0]):
    bands.append({"band": g, "n": int(len(gdf)), "events": int(gdf["y"].sum()),
                  "prevalence": round(float(gdf["y"].sum() / len(gdf)), 4),
                  "auroc": round(float(roc_auc_score(gdf["y"], gdf["p"])), 4)})
log(f"era bands: {[(b['band'], b['auroc']) for b in bands]}")
ADMP = pd.read_parquet(D / "admissions.parquet",
                       columns=["subject_id", "hadm_id", "admittime",
                                "dischtime", "insurance"])
adm = ADMP.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
adm["admittime"] = pd.to_datetime(adm["admittime"])
adm["dischtime"] = pd.to_datetime(adm["dischtime"])
nxt = adm.groupby("subject_id").shift(-1)
adm["delta_any"] = (nxt["admittime"] - adm["dischtime"]).dt.total_seconds() / 86400.0
adm["next_ins"] = nxt["insurance"]
med = adm[adm["insurance"] == "Medicare"].set_index("hadm_id")
j = EV.set_index("hadm_id").join(med[["delta_any", "next_ins"]], how="inner")
nonmed = j[(j["delta_any"] > 0) & (j["delta_any"] <= 30)
           & (j["next_ins"] != "Medicare")]
payer = {"n_30d_next_nonmedicare": int(len(nonmed)),
         "n_counted_as_readmission": int((nonmed["event_v3"] == 1).sum())}

EX = {"fixed_consensus_vs_eligible_cv": fixed_cmp,
      "highdraw_contrasts_5000": hd,
      "tie_rule_sensitivity": tie_res,
      "transfer_band": tb,
      "era_stratified_test": {"bands": bands},
      "payer_scope": payer,
      "boot_draws": B, "seed": SEED}
(O7 / "extras_v12.json").write_text(json.dumps(EX, indent=1))
FMC = copy.deepcopy(FM)
FMC["fairness"]["white_minus_black"] = {
    "point": FM["fairness"]["white_minus_black"]["point"],
    "ci95": wb7["ci95"], "p_bootstrap": wb7["p_two_sided_plus1"],
    "n_boot": wb7["n_boot"], "design": wb7["design"]}
FMC["baselines"]["uplift_lace_ci"] = hd["lace_uplift"]["ci95"]
FMC["baselines"]["uplift_lace_p"] = hd["lace_uplift"]["p_two_sided_plus1"]
FMC["baselines"]["uplift_hospital12_ci"] = hd["hospital12_uplift"]["ci95"]
FMC["baselines"]["uplift_hospital12_p"] = hd["hospital12_uplift"]["p_two_sided_plus1"]
FMC["extras"] = EX
(O7 / "final_model_v12.json").write_text(json.dumps(FMC, indent=1))
log("final_model_v12.json + extras_v12.json written - DONE")

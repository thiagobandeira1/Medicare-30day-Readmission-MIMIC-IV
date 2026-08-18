"""v5 final consensus model: artifacts + every downstream analysis.

Additions over the v4 equivalent (2026-08-12 external audit):
  * consensus set comes from the DEV-ONLY leak-free rerun (results_v5)
  * secondary CV runs on development partitions only (test untouched)
  * subgroup panel gains cluster-bootstrap CIs for sensitivity, specificity,
    PPV, NPV, FNR, Brier, and calibration slope per group
  * same-day-admission sensitivity analysis (overlapping 'transfer' next
    admissions counted as readmissions; refit + test evaluation)
  * planned-readmission sensitivity (label_sens_unplanned; evaluation of the
    final model and a refit variant)
  * elective-INDEX-admission subset evaluation
  * daily competing-risk event counts for the appendix
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "8"
import json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve)
from sklearn.linear_model import LogisticRegression

t0 = time.time()
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v5"
R2 = REPO / "results_reanalysis"
ART = OUT / "artifacts_v5"
ART.mkdir(exist_ok=True)
SEED = 42
rng = np.random.RandomState(SEED)
PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=8,
              eval_metric="auc", tree_method="hist", verbosity=0)


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


CMP = json.loads((OUT / "model_comparison.json").read_text())
FSEL = CMP["_stability"]["consensus_majority_features"]
NF = len(FSEL)
assert "discharge_location_te" not in FSEL
log(f"consensus features ({NF}): {sorted(FSEL)}")

V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
C = pd.read_parquet(R2 / "cohort_v2.parquet")
keep = (~C["excl_index_death"]).to_numpy()
part = C["partition"].to_numpy()
y = C["label_v2"].to_numpy()
subj = C["subject_id"].to_numpy()
tr, va, te = (keep & (part == p) for p in ("train", "val", "test"))
dev = keep & ((part == "train") | (part == "val"))

OBJ = [c for c in FSEL if not pd.api.types.is_numeric_dtype(DF[c])]
log(f"fold-local-encoded object columns in final set: {OBJ}")


def te_fit(idx, yy=None):
    yy = y if yy is None else yy
    maps, gmean = {}, yy[idx].mean()
    for c in OBJ:
        cats = DF[c].astype(str).fillna("__NA__")
        g = pd.DataFrame({"c": cats.iloc[idx], "y": yy[idx]}).groupby("c")["y"] \
            .agg(["mean", "count"])
        maps[c] = {"map": ((g["mean"] * g["count"] + gmean * 20)
                           / (g["count"] + 20)).to_dict(), "global": float(gmean)}
    return maps


def te_apply(maps, frame):
    X = frame.copy()
    for c in OBJ:
        X[c] = X[c].astype(str).fillna("__NA__") \
            .map(maps[c]["map"]).fillna(maps[c]["global"]).astype(float)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


# ------------------- secondary CV (fixed consensus set, DEV partitions only)
devidx = np.where(dev)[0]
Xk = DF[FSEL].iloc[devidx].reset_index(drop=True)
yk, sk = y[devidx], subj[devidx]
cv_scores = []
for f, (itr, ite) in enumerate(GroupKFold(5).split(Xk, yk, groups=sk), 1):
    m0 = te_fit(devidx[itr])
    Xf_tr = te_apply(m0, Xk.iloc[itr]).to_numpy(np.float32)
    Xf_te = te_apply(m0, Xk.iloc[ite]).to_numpy(np.float32)
    mm = xgb.XGBClassifier(**PARAMS)
    mm.fit(Xf_tr, yk[itr])
    cv_scores.append(float(roc_auc_score(yk[ite],
                                         mm.predict_proba(Xf_te)[:, 1])))
    log(f"consensus CV fold {f}: {cv_scores[-1]:.4f}")
cv_mean, cv_sd = float(np.mean(cv_scores)), float(np.std(cv_scores))
log(f"consensus-{NF} CV (dev-only): {cv_mean:.4f} +/- {cv_sd:.4f}")

# ------------------------------------------- final artifact (train-fitted)
maps = te_fit(np.where(tr)[0])
Xtr = te_apply(maps, DF[FSEL].loc[tr]).to_numpy(np.float32)
Xva = te_apply(maps, DF[FSEL].loc[va]).to_numpy(np.float32)
Xte = te_apply(maps, DF[FSEL].loc[te]).to_numpy(np.float32)
clf = xgb.XGBClassifier(**PARAMS)
clf.fit(Xtr, y[tr])
p_va = clf.predict_proba(Xva)[:, 1]
p_te = clf.predict_proba(Xte)[:, 1]
y_va, y_te, s_te = y[va], y[te], subj[te]
auroc_te = roc_auc_score(y_te, p_te)
fpr, tpr, thr = roc_curve(y_va, p_va)
THR = float(thr[np.argmax(tpr - fpr)])
log(f"final consensus-{NF}: test(tertiary) AUROC {auroc_te:.4f}; "
    f"val threshold {THR:.4f}")
clf.save_model(str(ART / f"model_v5_consensus{NF}.json"))
(ART / "te_maps.json").write_text(json.dumps(maps, indent=1))
(ART / "feature_order.json").write_text(json.dumps(FSEL, indent=1))
np.savez(OUT / "final_preds_v5.npz", p_te=p_te, y_te=y_te, s_te=s_te,
         p_va=p_va, y_va=y_va)


# ---------------------------------------------------- metric panel + boots
def ecef(yy, pp, nb=10):
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


def ops(yy, pp, t):
    pr = (pp >= t).astype(int)
    tp = ((pr == 1) & (yy == 1)).sum(); fn = ((pr == 0) & (yy == 1)).sum()
    fp = ((pr == 1) & (yy == 0)).sum(); tn = ((pr == 0) & (yy == 0)).sum()
    return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "ppv": tp / (tp + fp), "npv": tn / (tn + fn), "fnr": fn / (fn + tp)}


uniq = np.unique(s_te)
rows = {s: np.where(s_te == s)[0] for s in uniq}
stats = {k: [] for k in ("auroc", "ap", "brier", "ece", "slope", "intercept",
                         "sensitivity", "specificity", "ppv", "npv", "fnr")}
for b in range(1000):
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    ix = np.concatenate([rows[s] for s in pick])
    yy, pp = y_te[ix], p_te[ix]
    if len(np.unique(yy)) < 2:
        continue
    stats["auroc"].append(roc_auc_score(yy, pp))
    stats["ap"].append(average_precision_score(yy, pp))
    stats["brier"].append(brier_score_loss(yy, pp))
    stats["ece"].append(ecef(yy, pp))
    sl, ic = slope_int(yy, pp)
    stats["slope"].append(sl); stats["intercept"].append(ic)
    for k, v in ops(yy, pp, THR).items():
        stats[k].append(v)


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)]


point = {"auroc": auroc_te, "ap": average_precision_score(y_te, p_te),
         "brier": brier_score_loss(y_te, p_te), "ece": ecef(y_te, p_te)}
point["slope"], point["intercept"] = slope_int(y_te, p_te)
point.update(ops(y_te, p_te, THR))
panel = {k: {"point": round(float(point[k]), 4), "ci95": ci(v)}
         for k, v in stats.items()}
log(f"panel done: AUROC {panel['auroc']}")

# ------------------------------------------------------------- DCA/capacity
prev = y_te.mean()
ths = np.round(np.arange(0.05, 0.41, 0.01), 2)


def nb_(yy, pp, t):
    pr = pp >= t
    return (((pr) & (yy == 1)).sum() - ((pr) & (yy == 0)).sum() * t / (1 - t)) / len(yy)


dca = {"thresholds": ths.tolist(),
       "model": [round(float(nb_(y_te, p_te, t)), 4) for t in ths],
       "treat_all": [round(float(prev - (1 - prev) * t / (1 - t)), 4) for t in ths]}
cap = {}
for pct in (5, 10, 20):
    k = int(len(p_te) * pct / 100)
    ix = np.argsort(p_te)[::-1][:k]
    cap[f"top_{pct}"] = {"captured": round(float(y_te[ix].sum() / y_te.sum()), 3),
                         "ppv": round(float(y_te[ix].mean()), 3),
                         "n": k}

# ------------------------------------------------------------------- SHAP
dm = xgb.DMatrix(Xte, feature_names=FSEL)
contrib = clf.get_booster().predict(dm, pred_contribs=True)
ma = np.abs(contrib[:, :-1]).mean(axis=0)
shap_top = [{"feature": FSEL[i], "mean_abs_shap": round(float(ma[i]), 4)}
            for i in np.argsort(ma)[::-1]]
log(f"SHAP top-3: {[s['feature'] for s in shap_top[:3]]}")

# ------------------------------------------------------------- subgroups
ADM = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                      columns=["hadm_id", "race"])
hadm_te = C.loc[te, "hadm_id"].to_numpy()
rr = pd.DataFrame({"hadm_id": hadm_te}).merge(ADM, on="hadm_id")["race"].astype(str)
race = np.select([rr.str.contains("WHITE"), rr.str.contains("BLACK"),
                  rr.str.contains("HISPANIC|LATINO"), rr.str.contains("ASIAN")],
                 ["White", "Black", "Hispanic/Latino", "Asian"], "Other/Unknown")
V10d = pd.read_parquet(PUB / "training_table_v10.parquet",
                       columns=["hadm_id", "age_at_admit", "gender"])
dd = pd.DataFrame({"hadm_id": hadm_te}).merge(V10d, on="hadm_id")
age = dd["age_at_admit"].to_numpy()
sex = dd["gender"].astype(str).str.upper().map({"F": "Female", "M": "Male"}).to_numpy()
band = np.select([age < 65, age < 75, age < 85], ["<65", "65-74", "75-84"], "85+")
GROUPS = {"race": {g: race == g for g in
                   ["White", "Black", "Hispanic/Latino", "Asian", "Other/Unknown"]},
          "sex": {g: sex == g for g in ["Female", "Male"]},
          "age_band": {g: band == g for g in ["<65", "65-74", "75-84", "85+"]}}
GMET = ("auroc", "sensitivity", "specificity", "ppv", "npv", "fnr", "brier",
        "cal_slope")
fair = {}
boot_g = {c: {g: {m: [] for m in GMET} for g in d} for c, d in GROUPS.items()}
wb = []
for b in range(800):
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    ix = np.concatenate([rows[s] for s in pick])
    yy, pp = y_te[ix], p_te[ix]
    per = {"race": race[ix], "sex": sex[ix], "age_band": band[ix]}
    aw = ab = None
    for c, d in GROUPS.items():
        for g in d:
            m = per[c] == g
            if m.sum() < 50 or len(np.unique(yy[m])) < 2:
                continue
            a = roc_auc_score(yy[m], pp[m])
            bg = boot_g[c][g]
            bg["auroc"].append(a)
            bg["brier"].append(brier_score_loss(yy[m], pp[m]))
            try:
                sl, _ = slope_int(yy[m], pp[m])
                bg["cal_slope"].append(sl)
            except Exception:
                pass
            for k2, v2 in ops(yy[m], pp[m], THR).items():
                bg[k2].append(v2)
            if c == "race" and g == "White":
                aw = a
            if c == "race" and g == "Black":
                ab = a
    if aw is not None and ab is not None:
        wb.append(aw - ab)
for c, d in GROUPS.items():
    fair[c] = {}
    for g, m in d.items():
        yy, pp = y_te[m], p_te[m]
        e = {"patients": int(len(np.unique(s_te[m]))), "admissions": int(m.sum()),
             "events": int(yy.sum()),
             "prevalence": round(float(yy.mean()), 4),
             "auroc": round(float(roc_auc_score(yy, pp)), 4)
             if len(np.unique(yy)) > 1 else None,
             "ap": round(float(average_precision_score(yy, pp)), 4),
             "brier": round(float(brier_score_loss(yy, pp)), 4),
             "ece": round(ecef(yy, pp), 4)}
        sl, ic = slope_int(yy, pp)
        e["cal_slope"], e["cal_intercept"] = round(sl, 3), round(ic, 3)
        e.update({k: round(float(v2), 3) for k, v2 in ops(yy, pp, THR).items()})
        for mk in GMET:
            vv = boot_g[c][g][mk]
            if vv:
                e[f"{mk}_ci95"] = ci(vv)
        fair[c][g] = e
delta = fair["race"]["White"]["auroc"] - fair["race"]["Black"]["auroc"]
p2s = 2 * min((np.array(wb) <= 0).mean(), (np.array(wb) >= 0).mean())
fair["white_minus_black"] = {"point": round(float(delta), 4), "ci95": ci(wb),
                             "p_bootstrap": round(float(max(p2s, 2 / len(wb))), 4)}
log(f"fairness: W-B {delta:.4f} {fair['white_minus_black']['ci95']} p~{p2s:.4f}")

# ------------------------------------------------------- baselines vs final
BS = np.load(R2 / "baselines_v2_scores.npz")
lace = BS["lace"]
ADM2 = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                       columns=["subject_id", "hadm_id", "admittime"])
ADM2 = ADM2.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
cnt12 = np.zeros(len(ADM2), dtype=int)
for _s, g in ADM2.groupby("subject_id", sort=False):
    t = g["admittime"].values.astype("datetime64[D]").astype("int64")
    cnt12[g.index.values] = np.arange(len(t)) - np.searchsorted(t, t - 365, "left")
ADM2["adm12"] = cnt12
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
dl, dh = [], []
for b in range(600):
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    ix = np.concatenate([rows[s] for s in pick])
    yy = y_te[ix]
    if len(np.unique(yy)) < 2:
        continue
    am = roc_auc_score(yy, p_te[ix])
    dl.append(am - roc_auc_score(yy, lace[ix]))
    dh.append(am - roc_auc_score(yy, h12[ix]))
base = {"lace_auroc": round(float(roc_auc_score(y_te, lace)), 4),
        "hospital12_auroc": round(float(roc_auc_score(y_te, h12)), 4),
        "uplift_lace": round(float(auroc_te - roc_auc_score(y_te, lace)), 4),
        "uplift_lace_ci": ci(dl),
        "uplift_hospital12": round(float(auroc_te - roc_auc_score(y_te, h12)), 4),
        "uplift_hospital12_ci": ci(dh)}
log(f"baselines: LACE {base['lace_auroc']} HOSP12 {base['hospital12_auroc']}")

# ---------------------------------------------- outcome sensitivity analyses
sens = {}
delta_days = C["delta_days"].to_numpy()

# (a) same-day / overlapping next admissions counted as readmissions
y_alt = ((~np.isnan(delta_days)) & (delta_days > -2)
         & (delta_days <= 30)).astype(int)
clf_a = xgb.XGBClassifier(**PARAMS)
clf_a.fit(Xtr, y_alt[tr])
p_te_a = clf_a.predict_proba(Xte)[:, 1]
sens["sameday_as_readmission"] = {
    "definition": ("next admission counted as readmission when exact-time "
                   "delta <= 30 days INCLUDING the -2<delta<=0 band the "
                   "primary label treats as transfers/continuations"),
    "test_events_primary": int(y_te.sum()),
    "test_events_alt": int(y_alt[te].sum()),
    "test_prevalence_alt": round(float(y_alt[te].mean()), 4),
    "test_auroc_refit": round(float(roc_auc_score(y_alt[te], p_te_a)), 4),
    "test_auroc_final_model_vs_alt_label":
        round(float(roc_auc_score(y_alt[te], p_te)), 4)}
log(f"sensitivity same-day: {sens['sameday_as_readmission']}")

# (b) planned-readmission proxy: next admission typed ELECTIVE not counted
y_unpl = C["label_sens_unplanned"].to_numpy()
clf_b = xgb.XGBClassifier(**PARAMS)
clf_b.fit(Xtr, y_unpl[tr])
p_te_b = clf_b.predict_proba(Xte)[:, 1]
sens["unplanned_only"] = {
    "definition": ("readmission counted only when the subsequent admission "
                   "is not typed ELECTIVE (proxy for excluding planned "
                   "readmissions)"),
    "test_events_alt": int(y_unpl[te].sum()),
    "test_prevalence_alt": round(float(y_unpl[te].mean()), 4),
    "test_auroc_refit": round(float(roc_auc_score(y_unpl[te], p_te_b)), 4),
    "test_auroc_final_model_vs_alt_label":
        round(float(roc_auc_score(y_unpl[te], p_te)), 4)}
log(f"sensitivity unplanned: {sens['unplanned_only']}")

# (c) elective-INDEX admissions excluded from the test evaluation
adm_type = DF["admission_type"].astype(str).to_numpy()
nonel = ~np.isin(adm_type, list(EL))
m_ne = te & nonel
p_ne = clf.predict_proba(te_apply(maps, DF[FSEL].loc[m_ne]).to_numpy(np.float32))[:, 1]
sens["nonelective_index_only"] = {
    "test_admissions": int(m_ne.sum()),
    "test_events": int(y[m_ne].sum()),
    "test_prevalence": round(float(y[m_ne].mean()), 4),
    "test_auroc": round(float(roc_auc_score(y[m_ne], p_ne)), 4)}
log(f"sensitivity nonelective index: {sens['nonelective_index_only']}")

# ------------------------------------------------ survival + stages
EV = pd.read_parquet(R2 / "cohort_v3_events.parquet")
D2 = DF.loc[DF["hadm_id"].isin(set(EV["hadm_id"]))].reset_index(drop=True)
assert (D2["hadm_id"].values == EV["hadm_id"].values).all()
X33 = te_apply(maps, D2[FSEL])
event = EV["event_v3"].to_numpy(); tt = EV["time_v3"].to_numpy()
pt = EV["partition"].to_numpy()
tr2, va2, te2 = pt == "train", pt == "val", pt == "test"
lo2 = tt.copy(); hi2 = np.where(event == 1, tt, np.inf)

# daily competing-risk event counts (test partition) for the appendix
daily = []
for d_ in range(1, 31):
    daily.append({"day": d_,
                  "readmissions": int(((event == 1) & (tt > d_ - 1) & (tt <= d_)
                                       & te2).sum()),
                  "deaths_before_readmission":
                      int(((event == 2) & (tt > d_ - 1) & (tt <= d_) & te2).sum())})


def dmat(m):
    d = xgb.DMatrix(X33.loc[m].to_numpy(np.float32))
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
predt = bst.predict(xgb.DMatrix(X33.loc[te2].to_numpy(np.float32)))
from lifelines.utils import concordance_index
hc = concordance_index(tt[te2], predt, (event[te2] == 1).astype(int))
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
S_tr = Surv.from_arrays((event[tr2] == 1), tt[tr2])
S_te = Surv.from_arrays((event[te2] == 1), tt[te2])
days = np.arange(1, 30, 1.0)
auc_d, _ = cumulative_dynamic_auc(S_tr, S_te, -predt, days)
bst.save_model(str(ART / f"aft_v5_consensus{NF}.json"))
log(f"AFT-{NF}: C {hc:.4f}; day1 {auc_d[0]:.4f} day29 {auc_d[-1]:.4f}")

stages = []
for name, lo_d, hi_d in (("1-7", 0, 7), ("8-14", 7, 14), ("15-30", 14, 30)):
    at = ~(((event == 1) | (event == 2)) & (tt <= lo_d))
    ys = ((event == 1) & (tt > lo_d) & (tt <= hi_d)).astype(int)
    mm = xgb.XGBClassifier(**{**PARAMS, "n_estimators": 400})
    mm.fit(X33.loc[at & tr2].to_numpy(np.float32), ys[at & tr2])
    ps = mm.predict_proba(X33.loc[at & te2].to_numpy(np.float32))[:, 1]
    a = roc_auc_score(ys[at & te2], ps)
    stages.append({"stage": name, "at_risk": int((at & te2).sum()),
                   "events": int(ys[at & te2].sum()), "auroc": round(float(a), 4)})
    log(f"stage {name}: AUROC {a:.4f}")

RFEC = CMP["rfe"]
RES = {"final_feature_set": FSEL, "n_features": NF,
       "validation_hierarchy": {
           "primary_oof_procedure_dev_only": {
               "auroc": RFEC["oof_auroc"], "ci95": RFEC["auroc_ci95_cluster"],
               "note": ("out-of-fold performance of the complete selection "
                        "procedure on development partitions (fold-specific "
                        "feature sets); estimates the procedure, not the "
                        "fixed consensus model")},
           "secondary_consensus_cv_dev_only": {
               "folds": [round(s, 4) for s in cv_scores],
               "mean": round(cv_mean, 4), "sd": round(cv_sd, 4),
               "note": ("fixed consensus set; optimistic because the set "
                        "aggregates selection information across folds")},
           "tertiary_test_historically_exposed": panel["auroc"]},
       "threshold_validation": round(THR, 4), "metric_panel": panel,
       "dca": dca, "capacity": cap, "shap": shap_top, "fairness": fair,
       "baselines": base, "sensitivity": sens,
       "daily_event_counts_test": daily,
       "survival": {"aft_harrell_c": round(float(hc), 4),
                    "daily_tdauc": [round(float(a), 4) for a in auc_d]},
       "stages": stages,
       "env": {"python": platform.python_version(), "xgboost": xgb.__version__,
               "sklearn": sklearn.__version__, "pandas": pd.__version__,
               "numpy": np.__version__, "seed": SEED}}
(OUT / "final_model_v5.json").write_text(json.dumps(RES, indent=1))
log("final_model_v5.json written - DONE")

"""v7 analyses responding to the 2026-10-11 independent review.

Writes results_v11/review_v11.json (aggregates only) and merges it into
results_v11/final_model_v11.json under "review"; draws Figure 4 (daily AUROC
with bands, both continuation rules) and Figure 5 (subgroups incl. prior
utilization).

A  continuation band: returns within 1 hour of discharge
B  1-hour continuation rule as a sensitivity outcome (refit + final model)
C  daily cumulative/dynamic AUROC with patient-cluster bands, both rules;
   landmark models under the 1-hour rule
D  performance by prior-utilization stratum
E  White vs Black contrast within prior-utilization strata
F  era: eMAR coverage and within-stratum AUROC
G  LACE and HOSPITAL components refit on the training partition, with and
   without the coded (billing-derived) components
H  leakage ablation: apparent (cross-validated) vs held-out AUROC under
   common leaky pipeline choices
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "8"
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

t0 = time.time()
PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
O7 = REPO / "results_v11"
FIG = REPO / "figures_v11"
R2 = REPO / "results_reanalysis"
D = PUB / "Dataset" / "mimic-parquet"
SEED = 42
rng = np.random.RandomState(SEED)
PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=8,
              eval_metric="auc", tree_method="hist", verbosity=0)
CONT_H = 1.0
TEAL, CORAL = "#1F7A8C", "#E76F51"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 450,
                     "axes.spines.top": False, "axes.spines.right": False})


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)]


FM = json.loads((O7 / "final_model_v11.json").read_text())
FC = FM["final_feature_set"]
REV = {}

V7 = pd.read_parquet(PUB / "training_table_v11.parquet")
V10 = pd.read_parquet(PUB / "training_table_v11.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
C = pd.read_parquet(R2 / "cohort_v2.parquet")
assert (DF["hadm_id"].values == C["hadm_id"].values).all()
keep = (~C["excl_index_death"]).to_numpy()
part = C["partition"].to_numpy()
y = C["label_v2"].to_numpy()
subj = C["subject_id"].to_numpy()
tr, va, te = (keep & (part == p) for p in ("train", "val", "test"))
dev = keep & np.isin(part, ["train", "val"])
P7 = np.load(O7 / "final_preds_v11.npz")
p_te, y_te, s_te = P7["p_te"].astype(float), P7["y_te"], P7["s_te"]
assert np.array_equal(y_te, y[te])
uniq = np.unique(s_te)
rows = {s: np.where(s_te == s)[0] for s in uniq}
BOOT = [np.concatenate([rows[s] for s in rng.choice(uniq, len(uniq), True)])
        for _ in range(800)]
log("data loaded")

# ======================================================= A. continuation band
adm = pd.read_parquet(D / "admissions.parquet", columns=[
    "subject_id", "hadm_id", "admittime", "dischtime", "admission_location",
    "discharge_location", "edregtime", "race"])
for c_ in ("admittime", "dischtime", "edregtime"):
    adm[c_] = pd.to_datetime(adm[c_])
adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
g = adm.groupby("subject_id")
adm["prior_n"] = g.cumcount()
for c_ in ("admittime", "admission_location", "edregtime"):
    adm["next_" + c_] = g[c_].shift(-1)
CC = C[["hadm_id", "partition", "excl_index_death", "label_v2"]].merge(
    adm, on="hadm_id", how="left")
assert (CC.hadm_id.values == C.hadm_id.values).all()
CC["gap_h"] = (CC.next_admittime - CC.dischtime).dt.total_seconds() / 3600
ev = CC[keep & (CC.label_v2 == 1).to_numpy()]
band = ev[ev.gap_h <= CONT_H]
rates = {}
for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 6), (6, 12), (12, 24), (24, 72),
               (72, 168), (168, 720)]:
    n = int(((ev.gap_h > lo) & (ev.gap_h <= hi)).sum())
    rates[f"{lo}_{hi}h"] = {"events": n, "per_hour": round(n / (hi - lo), 1)}
pre_ed = band.next_edregtime.notna() & (band.next_edregtime <= band.dischtime)
REV["continuation_band"] = {
    "window_hours": CONT_H,
    "events_all": int(len(ev)),
    "events_in_band": int(len(band)),
    "share_of_events": round(len(band) / len(ev), 4),
    "events_in_band_test": int((band.partition == "test").sum()),
    "rate_per_hour": rates,
    "later_rate_range_per_hour": [
        min(v["per_hour"] for k, v in rates.items() if k != "0_1h"),
        max(v["per_hour"] for k, v in rates.items() if k != "0_1h")],
    "next_internal_psych_transfer": round(float(
        (band.next_admission_location == "INTERNAL TRANSFER TO OR FROM PSYCH").mean()), 4),
    "next_from_snf": round(float(
        (band.next_admission_location == "TRANSFER FROM SKILLED NURSING FACILITY").mean()), 4),
    "next_ed_registration_before_index_discharge": round(float(pre_ed.mean()), 4),
    "index_discharge_location_missing": round(float(band.discharge_location.isna().mean()), 4),
    "index_discharge_hospice": round(float((band.discharge_location == "HOSPICE").mean()), 4),
    "index_discharge_psych_facility": round(float(
        (band.discharge_location == "PSYCH FACILITY").mean()), 4)}
log(f"A: {REV['continuation_band']['events_in_band']} band events")

# ============================================ B. 1-hour rule sensitivity
gap_all = CC.gap_h.to_numpy()
y1h = y.copy()
y1h[(y == 1) & (gap_all <= CONT_H)] = 0
OBJ = [c for c in FC if not pd.api.types.is_numeric_dtype(DF[c])]
mapsj = json.loads((O7 / "artifacts_v11" / "te_maps.json").read_text())


def apply_maps(maps, frame, cols):
    X = frame.copy()
    for c_ in cols:
        X[c_] = X[c_].astype(str).fillna("__NA__").map(maps[c_]["map"]) \
            .fillna(maps[c_]["global"]).astype(float)
    return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


Xtr = apply_maps(mapsj, DF[FC].loc[tr], OBJ).to_numpy(np.float32)
Xte = apply_maps(mapsj, DF[FC].loc[te], OBJ).to_numpy(np.float32)
clf1 = xgb.XGBClassifier(**PARAMS).fit(Xtr, y1h[tr])
p1 = clf1.predict_proba(Xte)[:, 1]
REV["one_hour_rule"] = {
    "test_events": int(y1h[te].sum()),
    "test_prevalence": round(float(y1h[te].mean()), 4),
    "test_auroc_refit": round(float(roc_auc_score(y1h[te], p1)), 4),
    "test_auroc_final_model": round(float(roc_auc_score(y1h[te], p_te)), 4)}
log(f"B: {REV['one_hour_rule']}")

# ======================================================= C. timing with bands
EV = pd.read_parquet(R2 / "cohort_v3_events.parquet")
A = np.load(O7 / "aft_test_preds_v11.npz")
predt, ev_te, tt_te, hadm_aft, subj_aft = (A["predt"], A["event"], A["tt"],
                                           A["hadm"], A["subj"])
gap_map = pd.Series(gap_all, index=C.hadm_id.values)
gap_aft = gap_map.reindex(hadm_aft).to_numpy()
ev1h = ev_te.copy()
cont = (ev_te == 1) & (gap_aft <= CONT_H)
ev1h[cont] = 0
days = np.arange(1, 30)
ua = np.unique(subj_aft)
ra = {s: np.where(subj_aft == s)[0] for s in ua}
BA = [np.concatenate([ra[s] for s in rng.choice(ua, len(ua), True)])
      for _ in range(500)]
score = -predt


def daily(evv):
    pts = [roc_auc_score(((evv == 1) & (tt_te <= d)).astype(int), score)
           for d in days]
    bands = np.array([[roc_auc_score(((evv[ix] == 1) & (tt_te[ix] <= d)).astype(int),
                                     score[ix]) for d in days] for ix in BA])
    return ([round(float(v), 4) for v in pts],
            [round(float(v), 4) for v in np.percentile(bands, 2.5, axis=0)],
            [round(float(v), 4) for v in np.percentile(bands, 97.5, axis=0)])


prim = daily(ev_te)
onehr = daily(ev1h)
REV["timing"] = {
    "estimand": ("cumulative/dynamic AUROC of the AFT risk score; cases "
                 "readmitted by day t, controls all others including deaths "
                 "before t; empirical (no censoring before day 30); 95% "
                 "patient-cluster bootstrap bands, 500 draws"),
    "primary": {"auc": prim[0], "lo": prim[1], "hi": prim[2]},
    "one_hour_rule": {"auc": onehr[0], "lo": onehr[1], "hi": onehr[2]},
    "day1_events_primary": int(((ev_te == 1) & (tt_te <= 1)).sum()),
    "day1_events_one_hour_rule": int(((ev1h == 1) & (tt_te <= 1)).sum()),
    "reclassified_test_events": int(cont.sum())}
log(f"C: day1 primary {prim[0][0]} one-hour {onehr[0][0]}")
# landmark stages under the 1-hour rule
D2 = DF.loc[DF["hadm_id"].isin(set(EV["hadm_id"]))].reset_index(drop=True)
X33 = apply_maps(mapsj, D2[FC], OBJ)
eva = EV["event_v3"].to_numpy().copy(); tta = EV["time_v3"].to_numpy()
gapE = gap_map.reindex(EV.hadm_id.values).to_numpy()
eva[(eva == 1) & (gapE <= CONT_H)] = 0
ptE = EV["partition"].to_numpy()
st1h = []
for name, lo_d, hi_d in (("1-7", 0, 7), ("8-14", 7, 14), ("15-30", 14, 30)):
    at = ~(((eva == 1) | (eva == 2)) & (tta <= lo_d))
    ys = ((eva == 1) & (tta > lo_d) & (tta <= hi_d)).astype(int)
    trm, tem = at & (ptE == "train"), at & (ptE == "test")
    mm = xgb.XGBClassifier(**{**PARAMS, "n_estimators": 400})
    mm.fit(X33.loc[trm].to_numpy(np.float32), ys[trm])
    a = roc_auc_score(ys[tem], mm.predict_proba(X33.loc[tem].to_numpy(np.float32))[:, 1])
    st1h.append({"stage": name, "at_risk": int(tem.sum()),
                 "events": int(ys[tem].sum()), "auroc": round(float(a), 4)})
REV["timing"]["landmark_one_hour_rule"] = st1h
log(f"C landmarks 1h: {st1h}")

fig, ax = plt.subplots(figsize=(5.6, 3.7))
for key, col, lab in (("primary", CORAL, "Primary outcome definition"),
                      ("one_hour_rule", TEAL,
                       "Returns within 1 h reclassified as continuations")):
    r = REV["timing"][key]
    ax.fill_between(days, r["lo"], r["hi"], color=col, alpha=0.18, lw=0)
    ax.plot(days, r["auc"], "o-", color=col, lw=1.8, ms=3.2, label=lab)
ax.set_xlabel("Days since discharge")
ax.set_ylabel("Cumulative/dynamic AUROC")
ax.legend(fontsize=7.6, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "v11_daily.png", bbox_inches="tight")
plt.close(fig)

# =============================================== D. prior-utilization strata
T = C.loc[te, ["hadm_id"]].merge(adm[["hadm_id", "prior_n", "race"]], on="hadm_id")
prior_n = T.prior_n.to_numpy()
STR = {"0": prior_n == 0, "1 to 2": (prior_n >= 1) & (prior_n <= 2),
       "3 to 9": (prior_n >= 3) & (prior_n <= 9), "10 or more": prior_n >= 10}
r_ = T.race.astype(str)
race = np.select([r_.str.contains("WHITE"), r_.str.contains("BLACK")],
                 ["White", "Black"], "Other")
k10 = int(len(p_te) * 0.10)
top10 = np.zeros(len(p_te), bool); top10[np.argsort(p_te)[::-1][:k10]] = True
strata = {}
for name, m in STR.items():
    boots = [roc_auc_score(y_te[ix][m[ix]], p_te[ix][m[ix]]) for ix in BOOT]
    strata[name] = {"admissions": int(m.sum()),
                    "patients": int(len(np.unique(s_te[m]))),
                    "share": round(float(m.mean()), 4),
                    "prevalence": round(float(y_te[m].mean()), 4),
                    "auroc": round(float(roc_auc_score(y_te[m], p_te[m])), 4),
                    "auroc_ci95": ci(boots),
                    "share_of_top10pct_flags": round(float(m[top10].mean()), 4)}
REV["prior_utilization"] = strata
log(f"D: {[(k, v['auroc']) for k, v in strata.items()]}")

# ===================================== E. race within utilization strata
wmix = {k: float((m & (race == "White")).sum() / (race == "White").sum())
        for k, m in STR.items()}


def std_gap(ix=None):
    ix = np.arange(len(p_te)) if ix is None else ix
    yy, pp, rr = y_te[ix], p_te[ix], race[ix]
    gaps = {}
    for k, m in STR.items():
        mm_ = m[ix]
        w, b = mm_ & (rr == "White"), mm_ & (rr == "Black")
        gaps[k] = roc_auc_score(yy[w], pp[w]) - roc_auc_score(yy[b], pp[b])
    return sum(wmix[k] * gaps[k] for k in STR), gaps


sg, within = std_gap()
sgb = [std_gap(ix)[0] for ix in BOOT[:400]]
crude = FM["fairness"]["white_minus_black"]["point"]
REV["race_by_utilization"] = {
    "crude_gap": crude,
    "standardized_within_stratum_gap": round(float(sg), 4),
    "standardized_ci95": ci(sgb),
    "within_stratum_gap": {k: round(float(v), 4) for k, v in within.items()},
    "white_stratum_mix": {k: round(v, 4) for k, v in wmix.items()},
    "black_stratum_mix": {k: round(float((m & (race == "Black")).sum()
                                         / (race == "Black").sum()), 4)
                          for k, m in STR.items()},
    "median_prior_admissions": {"White": float(np.median(prior_n[race == "White"])),
                                "Black": float(np.median(prior_n[race == "Black"]))}}
log(f"E: {REV['race_by_utilization']}")

# =============================================== F. era: data completeness
pat = pd.read_parquet(D / "patients.parquet", columns=["subject_id", "anchor_year_group"])
era = pd.DataFrame({"subject_id": s_te}).merge(pat, on="subject_id",
                                               how="left").anchor_year_group.to_numpy()
emar_h = set(pd.read_parquet(D / "emar.parquet", columns=["hadm_id"])
             .hadm_id.dropna().astype("int64"))
has_emar = T.hadm_id.isin(emar_h).to_numpy()
era_out = []
for e in sorted(np.unique(era)):
    m = era == e
    era_out.append({"band": e.replace(" - ", " to "), "n": int(m.sum()),
                    "auroc": round(float(roc_auc_score(y_te[m], p_te[m])), 4),
                    "emar_coverage": round(float(has_emar[m].mean()), 4),
                    "first_admission_share": round(float((prior_n[m] == 0).mean()), 4),
                    "auroc_first_admissions": round(float(roc_auc_score(
                        y_te[m & (prior_n == 0)], p_te[m & (prior_n == 0)])), 4)})
REV["era"] = era_out
log(f"F: {[(e['band'], e['auroc'], e['emar_coverage']) for e in era_out]}")

# ============================================ G. refit LACE / HOSPITAL parts
ED = adm[["subject_id", "hadm_id", "admittime", "edregtime"]].copy()
ED["has_ed"] = ED.edregtime.notna().astype(int)
ed180 = np.zeros(len(ED), int); adm12 = np.zeros(len(ED), int)
for _, grp in ED.groupby("subject_id", sort=False):
    t = grp.admittime.values.astype("datetime64[s]").astype("int64") / 86400
    he = grp.has_ed.values
    idx = grp.index.values
    for i in range(len(grp)):
        j180 = np.searchsorted(t[:i], t[i] - 180, "left")
        ed180[idx[i]] = he[j180:i].sum()
    td = grp.admittime.values.astype("datetime64[D]").astype("int64")
    adm12[idx] = np.arange(len(td)) - np.searchsorted(td, td - 365, "left")
ED["prior_ed_180d"], ED["adm12"] = ed180, adm12
svc = pd.read_parquet(D / "services.parquet", columns=["hadm_id", "curr_service"])
omed = set(svc.loc[svc.curr_service == "OMED", "hadm_id"])
B_ = DF[["hadm_id", "los_days", "admission_type", "cci_score_total",
         "hemoglobin_last", "sodium_last", "n_procedures"]].merge(
    ED[["hadm_id", "prior_ed_180d", "adm12"]], on="hadm_id", how="left")
assert (B_.hadm_id.values == C.hadm_id.values).all()
EL = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}


def lace_L(x):
    if x < 1: return 0
    if x <= 3: return int(x)
    if x <= 6: return 4
    if x <= 13: return 5
    return 7


comp = pd.DataFrame({
    "L": B_.los_days.apply(lace_L),
    "A": (~B_.admission_type.isin(EL)).astype(int) * 3,
    "C": B_.cci_score_total.apply(lambda c_: min(int(c_), 3) if c_ < 4 else 5),
    "E": B_.prior_ed_180d.clip(0, 4),
    "H_hb": (B_.hemoglobin_last < 12).astype(int),
    "H_onc": B_.hadm_id.isin(omed).astype(int) * 2,
    "H_na": (B_.sodium_last < 135).astype(int),
    "H_proc": (B_.n_procedures >= 1).astype(int),
    "H_type": (~B_.admission_type.isin(EL)).astype(int),
    "H_adm": B_.adm12.apply(lambda n: 0 if n <= 1 else (2 if n <= 5 else 5)),
    "H_los": (B_.los_days >= 5).astype(int) * 2})
lace_tot = comp[["L", "A", "C", "E"]].sum(axis=1).to_numpy()
hosp_tot = comp[[c_ for c_ in comp if c_.startswith("H_")]].sum(axis=1).to_numpy()
BS = np.load(O7 / "baseline_scores_test_v11.npz")
lace_match = float((lace_tot[te] == BS["lace"]).mean())
hosp_match = float((hosp_tot[te] == BS["hospital12"]).mean())
log(f"G: reconstruction match LACE {lace_match:.4f} HOSPITAL {hosp_match:.4f}")


def refit(cols):
    Xd = pd.get_dummies(comp[cols].astype(str), drop_first=True).astype(float)
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(Xd[tr], y[tr])
    return lr.predict_proba(Xd[te])[:, 1]


LACE_ALL, LACE_DA = ["L", "A", "C", "E"], ["L", "A", "E"]
H_ALL = [c_ for c_ in comp if c_.startswith("H_")]
H_DA = [c_ for c_ in H_ALL if c_ != "H_proc"]
cmp_ = {"lace_original": BS["lace"].astype(float),
        "hospital_original": BS["hospital12"].astype(float),
        "lace_refit": refit(LACE_ALL), "lace_refit_no_coded": refit(LACE_DA),
        "hospital_refit": refit(H_ALL), "hospital_refit_no_coded": refit(H_DA),
        "lace_hospital_refit_no_coded": refit(LACE_DA + H_DA)}
G_ = {"reconstruction_match": {"lace": round(lace_match, 4),
                               "hospital": round(hosp_match, 4)}}
am = roc_auc_score(y_te, p_te)
for k, sc in cmp_.items():
    d_ = [roc_auc_score(y_te[ix], p_te[ix]) - roc_auc_score(y_te[ix], sc[ix])
          for ix in BOOT]
    G_[k] = {"auroc": round(float(roc_auc_score(y_te, sc)), 4),
             "uplift": round(float(am - roc_auc_score(y_te, sc)), 4),
             "uplift_ci95": ci(d_)}
REV["comparators"] = G_
log(f"G: {[(k, v['auroc']) for k, v in G_.items() if k != 'reconstruction_match']}")

# ============================================== H. leakage ablation
POOL = json.loads((O7 / "eligible_pool_v11.json").read_text())
BILL = [e["feature"] for e in POOL["excluded"] if "billing" in e["reason"]]
LEG_TE = ["discharge_location_te", "drg_code_te", "primary_dx_chapter_te",
          "last_drg_dispo_te"]
BILL_RAW = [b for b in BILL if b not in LEG_TE]
TREND = ["los_trend_180d", "los_trend_x_prior_6m", "los_trend_x_prior_admits"]
di = np.where(dev)[0]
ti, tei = np.where(tr)[0], np.where(te)[0]


def enc_maps(cols, idx):
    gm, mp = y[idx].mean(), {}
    for c_ in cols:
        cat = DF[c_].astype(str).fillna("__NA__")
        gg = pd.DataFrame({"c": cat.iloc[idx], "y": y[idx]}).groupby("c").y.agg(["mean", "count"])
        mp[c_] = (((gg["mean"] * gg["count"] + gm * 20) / (gg["count"] + 20)).to_dict(), gm)
    return mp


def mat(feats, mp, idx):
    X = DF[feats].iloc[idx].copy()
    for c_, (m_, gm) in mp.items():
        if c_ in X:
            X[c_] = X[c_].astype(str).fillna("__NA__").map(m_).fillna(gm).astype(float)
    return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan) \
        .to_numpy(np.float32)


def scheme(feats, split="patient", global_enc=False):
    obj = [c_ for c_ in feats if not pd.api.types.is_numeric_dtype(DF[c_])]
    folds = (GroupKFold(5).split(di, groups=subj[di]) if split == "patient"
             else KFold(5, shuffle=True, random_state=SEED).split(di))
    oof = np.full(len(di), np.nan)
    gmap = enc_maps(obj, di) if global_enc else None
    for itr, ite in folds:
        mp = gmap if global_enc else enc_maps(obj, di[itr])
        m_ = xgb.XGBClassifier(**PARAMS).fit(mat(feats, mp, di[itr]), y[di[itr]])
        oof[ite] = m_.predict_proba(mat(feats, mp, di[ite]))[:, 1]
    mp = enc_maps(obj, ti)
    m_ = xgb.XGBClassifier(**PARAMS).fit(mat(feats, mp, ti), y[ti])
    pt_ = m_.predict_proba(mat(feats, mp, tei))[:, 1]
    app, held = roc_auc_score(y[di], oof), roc_auc_score(y[tei], pt_)
    return {"apparent_cv_auroc": round(float(app), 4),
            "heldout_test_auroc": round(float(held), 4),
            "optimism": round(float(app - held), 4), "n_features": len(feats)}


ABL = {}
for key, kw in [("reference_leakage_safe", dict(feats=FC)),
                ("admission_level_split", dict(feats=FC, split="admission")),
                ("global_target_encoding", dict(feats=FC, global_enc=True)),
                ("plus_future_trend_features", dict(feats=FC + TREND)),
                ("plus_legacy_precomputed_encodings", dict(feats=FC + LEG_TE)),
                ("plus_billing_features", dict(feats=FC + BILL_RAW)),
                ("all_combined", dict(feats=FC + TREND + LEG_TE + BILL_RAW,
                                      split="admission", global_enc=True))]:
    ABL[key] = scheme(**kw)
    log(f"H {key}: {ABL[key]}")
REV["leakage_ablation"] = ABL

# ================================================= Figure 5 (2 x 2 panels)
FA = FM["fairness"]
overall = FM["metric_panel"]["auroc"]["point"]
PANELS = [("Age band", [(g.replace("-", " to "), FA["age_band"][g])
                        for g in ["<65", "65-74", "75-84", "85+"]]),
          ("Sex", [(g, FA["sex"][g]) for g in ["Female", "Male"]]),
          ("Race", [(g.replace("Hispanic/Latino", "Hispanic or\nLatino")
                     .replace("Other/Unknown", "Other or\nunknown"), FA["race"][g])
                    for g in ["White", "Black", "Hispanic/Latino", "Asian",
                              "Other/Unknown"]]),
          ("Prior admissions", [(k, {"auroc": v["auroc"], "auroc_ci95": v["auroc_ci95"]})
                                for k, v in strata.items()])]
allv = [e["auroc_ci95"][i] for _, its in PANELS for _, e in its for i in (0, 1)]
ylo, yhi = min(allv) - 0.02, max(allv) + 0.03
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
for ax, (title, its) in zip(axes.ravel(), PANELS):
    xs = np.arange(len(its))
    pts = [e["auroc"] for _, e in its]
    lo = [e["auroc"] - e["auroc_ci95"][0] for _, e in its]
    hi = [e["auroc_ci95"][1] - e["auroc"] for _, e in its]
    ax.errorbar(xs, pts, yerr=[lo, hi], fmt="o", color=TEAL, capsize=2.5, ms=4.5)
    ax.axhline(overall, color=CORAL, ls="--", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in its], fontsize=7)
    ax.set_xlim(-0.6, len(its) - 0.4); ax.set_ylim(ylo, yhi)
    ax.set_title(title, fontsize=9); ax.set_ylabel("AUROC", fontsize=8)
    ax.tick_params(axis="y", labelsize=7); ax.grid(axis="y", alpha=0.3, ls=":")
    for x, v in zip(xs, pts):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                    xytext=(7, 0), fontsize=6.5, va="center")
fig.tight_layout(); fig.savefig(FIG / "v11_fairness.png", bbox_inches="tight")
plt.close(fig)

(O7 / "review_v11.json").write_text(json.dumps(REV, indent=1))
FM["review"] = REV
(O7 / "final_model_v11.json").write_text(json.dumps(FM, indent=1))
log("review_v11.json written and merged - DONE")

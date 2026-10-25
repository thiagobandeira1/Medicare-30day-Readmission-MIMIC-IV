# -*- coding: utf-8 -*-
"""v10: 1-hour continuation sensitivity rule with the episode END as time origin.

Supersedes v9 (which timed returns from the first discharge timestamp).

The v7/v8 rule relabeled every readmission beginning within 1 hour of the
index discharge as a nonevent. That hides genuine returns that follow the
continuation. The episode rule treats the continuation as part of the index
episode: the outcome is 1 if the NEXT admission after the continuation begins
within 30 days of the index discharge (prediction horizon unchanged), with its
event time measured from the index discharge. Recomputes the binary AUROCs
(frozen and refit), daily cumulative/dynamic AUROC with bands, and landmark
models. Aggregate output: results_v11/episode_end_rule_v11.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
O7 = REPO / "results_v11"; R2 = REPO / "results_reanalysis"
SEED = 42; rng = np.random.default_rng(SEED)
PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.9,
              colsample_bytree=0.9, random_state=SEED, n_jobs=8,
              eval_metric="auc", tree_method="hist", verbosity=0)
CONT_H = 1.0
FM = json.loads((O7 / "final_model_v11.json").read_text())
FC = FM["final_feature_set"]
V7 = pd.read_parquet(PUB / "training_table_v11.parquet")
V10 = pd.read_parquet(PUB / "training_table_v11.parquet")
DF = pd.concat([V10.reset_index(drop=True),
                V7[[c for c in V7.columns if c not in V10.columns]].reset_index(drop=True)], axis=1)
C = pd.read_parquet(R2 / "cohort_v2.parquet")
assert (DF.hadm_id.values == C.hadm_id.values).all()
keep = (~C.excl_index_death).to_numpy(); part = C.partition.to_numpy()
y = C.label_v2.to_numpy()
tr, te = keep & (part == "train"), keep & (part == "test")

adm = pd.read_parquet(PUB / "Dataset" / "mimic-parquet" / "admissions.parquet",
                      columns=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"])
adm = adm.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
g = adm.groupby("subject_id")
for k in (1, 2):
    adm[f"adm{k}"] = g.admittime.shift(-k)
adm["dis1"] = g.dischtime.shift(-1); adm["death1"] = g.deathtime.shift(-1)
adm["dis_prev"] = g.dischtime.shift(1)
X = C[["hadm_id"]].merge(adm, on="hadm_id", how="left")
assert (X.hadm_id.values == C.hadm_id.values).all()
gap1_h = ((X.adm1 - X.dischtime).dt.total_seconds() / 3600).to_numpy()
# Chain every consecutive admission that begins within 1 h of the previous
# discharge into one episode; time origin = discharge of the last link; the
# outcome is the next admission after the chain (by construction > 1 h later).
at_ = adm.admittime.to_numpy(); dt_ = adm.dischtime.to_numpy()
dead_ = adm.deathtime.notna().to_numpy(); sid_ = adm.subject_id.to_numpy()
n_ = len(adm); end_dis = np.full(n_, np.datetime64("NaT"), "datetime64[ns]")
nxt_adm = np.full(n_, np.datetime64("NaT"), "datetime64[ns]")
chain_dead = np.zeros(n_, bool); chain_len = np.zeros(n_, int)
H1 = np.timedelta64(int(CONT_H * 3600), "s")
for i in range(n_):
    j = i
    while j + 1 < n_ and sid_[j + 1] == sid_[i] and np.timedelta64(0, "s") < at_[j + 1] - dt_[j] <= H1:
        j += 1
    chain_len[i] = j - i
    end_dis[i] = dt_[j]; chain_dead[i] = dead_[j] if j > i else False
    if j + 1 < n_ and sid_[j + 1] == sid_[i]:
        nxt_adm[i] = at_[j + 1]
m_ = pd.DataFrame({"hadm_id": adm.hadm_id, "end_dis": end_dis, "nxt": nxt_adm,
                   "chain_dead": chain_dead, "chain_len": chain_len})
X = X.merge(m_, on="hadm_id", how="left")
assert (X.hadm_id.values == C.hadm_id.values).all()
gap2_d = ((X.nxt - X.end_dis).dt.total_seconds() / 86400).to_numpy()
cont = (y == 1) & (gap1_h > 0) & (gap1_h <= CONT_H)
died = cont & X.chain_dead.to_numpy()          # episode ends in hospital death
# an admission that itself begins within 1 h of the previous discharge is a
# continuation link; under the episode rule it is not a separate index row
gap_prev_h = ((X.admittime - X.dis_prev).dt.total_seconds() / 3600).to_numpy()
is_link = (gap_prev_h > 0) & (gap_prev_h <= CONT_H)
keep0 = keep.copy()
drop = died | is_link
keep = keep & ~drop                            # deaths excluded, links merged
tr, te = keep & (part == "train"), keep & (part == "test")
y_ep = y.copy()
y_ep[cont] = ((gap2_d[cont] > 0) & (gap2_d[cont] <= 30)).astype(int)
y_1h = y.copy(); y_1h[cont] = 0
R = {"rule": "continuation (next admission within 1 h of index discharge) treated as part "
             "of the index episode; consecutive admissions within 1 h are chained; outcome = next admission within 30 days of the "
             "episode end (discharge of the last link); episodes ending in death excluded",
     "episodes_ending_in_death_excluded_all": int(died.sum()),
     "episodes_with_more_than_one_continuation": int((cont & (X.chain_len.to_numpy() > 1)).sum()),
     "episodes_ending_in_death_excluded_test": int((died & (part == "test") & ~C.excl_index_death.to_numpy()).sum()),
     "continuations_all": int(cont[keep].sum()),
     "continuations_test": int(cont[te].sum()),
     "continuations_with_later_return_all": int((cont & (y_ep == 1))[keep].sum()),
     "continuations_with_later_return_test": int((cont & (y_ep == 1))[te].sum()),
     "test_events_episode_rule": int(y_ep[te].sum()),
     "test_prevalence_episode_rule": round(float(y_ep[te].mean()), 4),
     "flow": {"test_admissions_primary": int((keep0 & (part == "test")).sum()),
              "test_events_primary": int(y[keep0 & (part == "test")].sum()),
              "test_rows_that_are_continuation_links_merged": int((is_link & keep0 & (part == "test")).sum()),
              "test_episodes_ending_in_death_excluded": int((died & keep0 & (part == "test")).sum()),
              "test_episodes_retained": int(te.sum()),
              "test_events_episode_rule": int(y_ep[te].sum()),
              "all_admissions_primary": int(keep0.sum()),
              "all_rows_that_are_continuation_links_merged": int((is_link & keep0).sum()),
              "all_episodes_ending_in_death_excluded": int((died & keep0).sum()),
              "all_episodes_retained": int(keep.sum()),
              "all_events_episode_rule": int(y_ep[keep].sum())}}

# binary AUROCs
maps = json.loads((O7 / "artifacts_v11" / "te_maps.json").read_text())
OBJ = [c for c in FC if not pd.api.types.is_numeric_dtype(DF[c])]


def enc(frame):
    Z = frame.copy()
    for c in OBJ:
        Z[c] = Z[c].astype(str).fillna("__NA__").map(maps[c]["map"]).fillna(maps[c]["global"]).astype(float)
    return Z.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)


P7 = np.load(O7 / "final_preds_v11.npz")
te0 = (~C.excl_index_death.to_numpy()) & (part == "test")
mte = ~drop[te0]
p_te, s_te = P7["p_te"][mte], P7["s_te"][mte]
assert np.array_equal(P7["y_te"], y[te0])
Xtr, Xte = enc(DF.loc[tr, FC]), enc(DF.loc[te, FC])
p_ref = xgb.XGBClassifier(**PARAMS).fit(Xtr, y_ep[tr]).predict_proba(Xte)[:, 1]
R["test_auroc_final_model"] = round(float(roc_auc_score(y_ep[te], p_te)), 4)
R["test_auroc_refit"] = round(float(roc_auc_score(y_ep[te], p_ref)), 4)
us = np.unique(s_te); rm = {u: np.where(s_te == u)[0] for u in us}
B = [np.concatenate([rm[u] for u in rng.choice(us, len(us))]) for _ in range(800)]
for key, pp in (("final_model", p_te), ("refit", p_ref)):
    v = [roc_auc_score(y_ep[te][ix], pp[ix]) for ix in B]
    R[f"test_auroc_{key}_ci95"] = [round(float(np.percentile(v, 2.5)), 4),
                                    round(float(np.percentile(v, 97.5)), 4)]
print(json.dumps(R, indent=1), flush=True)

# daily cumulative/dynamic AUROC (AFT score), same estimand as v7
A = np.load(O7 / "aft_test_preds_v11.npz")
predt, ev, tt, hadm_a, subj_a = A["predt"], A["event"].copy(), A["tt"].astype(float).copy(), A["hadm"], A["subj"]
gap1_map = pd.Series(gap1_h, index=C.hadm_id.values)
died_map = pd.Series(drop, index=C.hadm_id.values)
gap2_map = pd.Series(gap2_d, index=C.hadm_id.values)
g1, g2 = gap1_map.reindex(hadm_a).to_numpy(), gap2_map.reindex(hadm_a).to_numpy()
keep_a = ~died_map.reindex(hadm_a).fillna(False).to_numpy()
predt, ev, tt, hadm_a, subj_a, g1, g2 = (predt[keep_a], ev[keep_a], tt[keep_a], hadm_a[keep_a],
                                        subj_a[keep_a], g1[keep_a], g2[keep_a])
c_a = (ev == 1) & (g1 > 0) & (g1 <= CONT_H)
ret = c_a & (g2 > 0) & (g2 <= 30)
ev[c_a & ~ret] = 0
tt[ret] = g2[ret]
days = np.arange(1, 30); score = -predt
ua = np.unique(subj_a); ra = {u: np.where(subj_a == u)[0] for u in ua}
BA = [np.concatenate([ra[u] for u in rng.choice(ua, len(ua))]) for _ in range(500)]
pts = [roc_auc_score(((ev == 1) & (tt <= d)).astype(int), score) for d in days]
bands = np.array([[roc_auc_score(((ev[ix] == 1) & (tt[ix] <= d)).astype(int), score[ix])
                   for d in days] for ix in BA])
R["timing_episode_rule"] = {"auc": [round(float(v), 4) for v in pts],
                            "lo": [round(float(v), 4) for v in np.percentile(bands, 2.5, axis=0)],
                            "hi": [round(float(v), 4) for v in np.percentile(bands, 97.5, axis=0)]}
print("day1", pts[0], "min", min(pts), "day29", pts[-1], flush=True)

# landmark models under the episode rule
EV = pd.read_parquet(R2 / "cohort_v3_events.parquet")
D2 = DF.loc[DF.hadm_id.isin(set(EV.hadm_id))].reset_index(drop=True)
assert (D2.hadm_id.values == EV.hadm_id.values).all()
keepE = ~died_map.reindex(EV.hadm_id.values).fillna(False).to_numpy()
EV = EV.loc[keepE].reset_index(drop=True); D2 = D2.loc[keepE].reset_index(drop=True)
X33 = enc(D2[FC])
eva = EV.event_v3.to_numpy().copy(); tta = EV.time_v3.to_numpy().astype(float).copy()
gE1, gE2 = gap1_map.reindex(EV.hadm_id.values).to_numpy(), gap2_map.reindex(EV.hadm_id.values).to_numpy()
cE = (eva == 1) & (gE1 > 0) & (gE1 <= CONT_H)
rE = cE & (gE2 > 0) & (gE2 <= 30)
eva[cE & ~rE] = 0; tta[rE] = gE2[rE]
ptE = EV.partition.to_numpy(); st = []
for name, lo_d, hi_d in (("1-7", 0, 7), ("8-14", 7, 14), ("15-30", 14, 30)):
    at = ~(((eva == 1) | (eva == 2)) & (tta <= lo_d))
    ys = ((eva == 1) & (tta > lo_d) & (tta <= hi_d)).astype(int)
    trm, tem = at & (ptE == "train"), at & (ptE == "test")
    mm = xgb.XGBClassifier(**{**PARAMS, "n_estimators": 400}).fit(X33[trm], ys[trm])
    st.append({"stage": name, "at_risk": int(tem.sum()), "events": int(ys[tem].sum()),
               "auroc": round(float(roc_auc_score(ys[tem], mm.predict_proba(X33[tem])[:, 1])), 4)})
R["landmark_episode_rule"] = st
print(json.dumps(st), flush=True)
(O7 / "episode_end_rule_v11.json").write_text(json.dumps(R, indent=1))
print("saved")

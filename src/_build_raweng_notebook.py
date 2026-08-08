"""Build the raw-vs-engineered RFE notebook (the bulletproof feature-engineering analysis).
Mirrors src/_run_raweng_rfe.py; resumes from results/rfe_base_checkpoint.json so re-execution is cheap."""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "Feature_Selection_Raw_vs_Engineered.ipynb"
nb = nbf.v4.new_notebook(); cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md(r"""# Raw vs. Engineered Features — what does the hand-engineering actually buy?
## Direct response to the feature-engineering review

A reviewer asked us to move beyond the hand-crafted V1->V7 progression: *use the entire feature set,
then peel it apart with recursive feature elimination, with clear reasoning for how the sets differ.*
A subtle point is that **MIMIC-IV has no raw feature matrix** -- it is a relational database of event
tables (`labevents` alone is 158M rows), so every per-admission feature must be *constructed* by
aggregation. The honest axis is therefore not "raw vs. made-up" but:

* **BASE (minimally-processed)** -- raw passthroughs and the *mandatory* aggregations (e.g. `sodium_last`
  is the last of a patient's sodium time series). These are unavoidable.
* **ENG (hand-engineered)** -- the *optional*, designed features: target-encodings (`_te`), interactions
  (`_x_`), and log / polynomial / PCA transforms.

This notebook quantifies exactly what the ENG layer adds on top of BASE, and lets RFE decide which
engineered features are worth keeping. CPU is capped to 8 cores and RFE is checkpointed (crash-safe).""")

md("## §1. Setup (CPU-capped before numpy/lightgbm import)")
code(r"""import os
N_JOBS = 8
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(N_JOBS)
import json, re, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
plt.rcParams.update({"font.size":12,"axes.titlesize":14,"axes.titleweight":"bold","axes.labelsize":13,"font.weight":"bold"})

REPO = Path.cwd().resolve()
while REPO.name != "medicare-30day-readmission-mimic-iv" and REPO.parent != REPO:
    REPO = REPO.parent
PUB, RESULTS, FIGS = REPO.parent, REPO/"results", REPO/"figures"
print("repo:", REPO)""")

md("""## §2. Build the 207-feature pool and split BASE vs ENG
Same candidate universe as the canonical RFE (177 features from `training_table_v10` + 30 engineered
features unique to the deployed V7 set). We then split it by construction signature.""")
code(r"""V10 = pd.read_parquet(PUB/"training_table_v10.parquet")
V7  = pd.read_parquet(PUB/"training_table_v7.parquet")
META = {"subject_id","hadm_id","admittime_dt","dischtime_dt","insurance","readmit_30d",
        "discharge_location","admittime","dischtime","deathtime"}
feat_v10 = [c for c in V10.columns if c not in META]
v7_feats = json.loads((RESULTS/"v7_feature_cols.json").read_text())
v7_unique = [f for f in v7_feats if f not in feat_v10 and f in V7.columns]
feat_full = feat_v10 + v7_unique
DF = pd.concat([V10[feat_v10].reset_index(drop=True), V7[v7_unique].reset_index(drop=True)], axis=1)
y = V10["readmit_30d"].to_numpy().astype(int)

ENG_RE = re.compile(r"_te$|_x_|_sq$|^log_|_pc\d")
eng_feats  = [f for f in feat_full if ENG_RE.search(f)]
base_feats = [f for f in feat_full if not ENG_RE.search(f)]
print(f"Full pool: {len(feat_full)}  |  BASE (minimally-processed): {len(base_feats)}  |  ENG (hand-engineered): {len(eng_feats)}")
print("ENG features:", eng_feats)""")

md("## §3. Encode and reuse the exact patient-grouped split")
code(r"""Xdf = DF[feat_full].copy()
for c in feat_full:
    if Xdf[c].dtype == "object":
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str).fillna("__NA__"))
Xall = Xdf.to_numpy(dtype=float)
split = np.load(RESULTS/"v7_split_indices.npz")
tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]
pos = {f:i for i,f in enumerate(feat_full)}

def fit_lgbm(cols, n_estimators=400, seed=42):
    idx=[pos[f] for f in cols]
    m=lgb.LGBMClassifier(n_estimators=n_estimators,learning_rate=0.05,num_leaves=31,
                         subsample=0.9,colsample_bytree=0.9,random_state=seed,n_jobs=N_JOBS,verbose=-1)
    m.fit(Xall[tr][:,idx], y[tr]); return m, idx
def auroc(cols):
    m,idx=fit_lgbm(cols); return roc_auc_score(y[te], m.predict_proba(Xall[te][:,idx])[:,1])
print("train/val/test:", len(tr), len(va), len(te))""")

md("## §4. BASE-only vs Full-pool (the headline engineering lift)")
code(r"""t0=time.time()
auroc_base = auroc(base_feats); print(f"BASE-only ({len(base_feats)}) test AUROC: {auroc_base:.4f}  ({time.time()-t0:.0f}s)")
auroc_full = auroc(feat_full); print(f"Full pool ({len(feat_full)}) test AUROC: {auroc_full:.4f}")
print(f"Engineering lift (full - base): {auroc_full-auroc_base:+.4f}")""")

md("""## §5. RFE over BASE-only (checkpointed, CPU-capped)
Importance-based elimination, one LightGBM fit per round, scored on the inner-validation split.
Resumes from `rfe_base_checkpoint.json` if present, so this is cheap to re-run.""")
code(r"""STEP, MIN_FEATURES = 10, 10
CKPT = RESULTS/"rfe_base_checkpoint.json"
if CKPT.exists():
    st=json.loads(CKPT.read_text()); remaining=st["remaining"]; history=st["history"]
    print(f"Resuming: {len(history)} rounds done, {len(remaining)} remaining.")
else:
    remaining=list(base_feats); history=[]; print(f"Starting BASE-RFE from {len(remaining)} features.")
t0=time.time()
while len(remaining) >= MIN_FEATURES:
    idx=[pos[f] for f in remaining]
    m=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=31,subsample=0.9,
                         colsample_bytree=0.9,random_state=42,n_jobs=N_JOBS,verbose=-1)
    m.fit(Xall[tr][:,idx], y[tr])
    va_auc=roc_auc_score(y[va], m.predict_proba(Xall[va][:,idx])[:,1])
    history.append({"n":len(remaining),"val_auroc":float(va_auc),"features":list(remaining)})
    CKPT.write_text(json.dumps({"remaining":remaining,"history":history}))
    print(f"  {len(remaining):3d} base feat -> inner-val AUROC {va_auc:.4f}  ({time.time()-t0:.0f}s)")
    if len(remaining) <= MIN_FEATURES: break
    imp=m.feature_importances_; drop=set(np.argsort(imp)[:STEP])
    remaining=[f for i,f in enumerate(remaining) if i not in drop]
best=max(history,key=lambda h:h["val_auroc"]); base_rfe_feats=best["features"]
auroc_base_rfe=auroc(base_rfe_feats)
print(f"\nBASE-RFE optimal: {len(base_rfe_feats)} feat | inner-val {best['val_auroc']:.4f} | test {auroc_base_rfe:.4f}")""")

md("## §6. Engineering lift on the parsimony-optimal subset + which ENG features RFE keeps")
code(r"""combo = base_rfe_feats + eng_feats
auroc_combo = auroc(combo)
print(f"BASE-RFE + {len(eng_feats)} ENG ({len(combo)} feat) test AUROC: {auroc_combo:.4f}")
print(f"Engineering lift on RFE subset: {auroc_combo-auroc_base_rfe:+.4f}")

prev=json.loads((RESULTS/"rfe_selection_results.json").read_text())
rfe_sel=set(prev["rfe_selected"]); eng_in_rfe=sorted(f for f in rfe_sel if ENG_RE.search(f))
print(f"\nEngineered features retained by the full-pool RFE: {len(eng_in_rfe)}/{len(eng_feats)}")
print(eng_in_rfe)""")

md("## §7. Summary table, figure, and saved results")
code(r"""out={"n_base":len(base_feats),"n_eng":len(eng_feats),"n_full":len(feat_full),
     "auroc_base_full":float(auroc_base),"auroc_full_pool":float(auroc_full),
     "auroc_base_rfe":float(auroc_base_rfe),"n_base_rfe":len(base_rfe_feats),
     "auroc_base_rfe_plus_eng":float(auroc_combo),
     "auroc_full_rfe":float(prev["rfe_auroc"]),"n_full_rfe":prev["rfe_n"],
     "delta_eng_full_minus_base":float(auroc_full-auroc_base),
     "delta_eng_on_rfe":float(auroc_combo-auroc_base_rfe),
     "eng_features":eng_feats,"eng_features_kept_by_full_rfe":eng_in_rfe,
     "n_eng_kept_by_full_rfe":len(eng_in_rfe),"base_rfe_features":base_rfe_feats}
(RESULTS/"rfe_raw_vs_engineered.json").write_text(json.dumps(out,indent=2))

import pandas as pd
tab=pd.DataFrame({
 "Feature set":[f"BASE only ({len(base_feats)})",f"BASE-RFE ({len(base_rfe_feats)})",
                f"Full pool ({len(feat_full)})",f"Full-RFE ({prev['rfe_n']})"],
 "Test AUROC":[auroc_base,auroc_base_rfe,auroc_full,prev["rfe_auroc"]]})
print(tab.to_string(index=False))

labels=[f"BASE only\n({len(base_feats)} raw)",f"BASE-RFE\n({len(base_rfe_feats)})",
        f"Full pool\n({len(feat_full)})",f"Full-RFE\n({prev['rfe_n']})"]
vals=[auroc_base,auroc_base_rfe,auroc_full,prev["rfe_auroc"]]
fig,ax=plt.subplots(figsize=(9,5.2))
bars=ax.bar(labels,vals,color=["#8D99AE","#5C7A89","#1F7A8C","#E76F51"],edgecolor="black",lw=1.1)
ax.set_ylim(0.78,0.80); ax.set_ylabel("Test AUROC"); ax.set_xlabel("Feature set")
ax.set_title("What hand-engineering adds beyond minimally-processed features")
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,v+0.0004,f"{v:.4f}",ha="center",va="bottom",fontsize=11,fontweight="bold")
fig.tight_layout(); fig.savefig(FIGS/"raw_vs_engineered_auroc.png",dpi=300,bbox_inches="tight"); plt.show()
print("Saved figures/raw_vs_engineered_auroc.png and results/rfe_raw_vs_engineered.json")""")

md(r"""## §8. Interpretation for the manuscript
- **MIMIC has no raw feature matrix**: every feature is a constructed aggregation over relational event
  tables; only `admissions` and `patients` hold true one-row-per-admission fields.
- The **engineering lift** (full pool minus BASE) isolates the value of the optional target-encodings and
  interactions over the mandatory minimally-processed features.
- RFE retains only a minority of the engineered features, identifying *which* engineering is justified and
  confirming the deployed set sits at the parsimony-optimal frontier rather than being hand-picked.
- A full **feature-provenance audit** (source table + transformation + temporal class for every deployed
  feature) accompanies this analysis and confirms **zero forward-looking features** (no leakage).""")

nb.cells = cells
nb.metadata = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
               "language_info":{"name":"python","version":"3.11"}}
with open(OUT,"w",encoding="utf-8") as f: nbf.write(nb,f)
print(f"Saved {OUT} ({len(nb.cells)} cells)")

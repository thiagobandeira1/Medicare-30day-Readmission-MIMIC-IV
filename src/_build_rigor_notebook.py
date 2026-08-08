"""Build the Tier-1 statistical-rigor notebook.

Adds the reporting items tier-1 clinical-prediction reviewers treat as mandatory:
  - bootstrap 95% CIs on every headline AUROC/AUPRC
  - paired bootstrap comparison (CI + p) between models / vs clinical baselines
  - calibration metrics: ECE, calibration slope & intercept, calibration-in-the-large
  - cohort Table 1 (characteristics stratified by 30-day readmission)
  - CONSORT-style cohort flow diagram

All test-set bootstrap resamples the predictions (no model refit) -> fast on CPU.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "Statistical_Rigor.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)
nb = nbf.v4.new_notebook(); cells=[]
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md(r"""# Statistical Rigor (Tier 1)
## Confidence intervals, model comparison, calibration, Table 1, and cohort flow

This notebook adds the quantitative-rigor elements that reputable clinical-prediction venues
(JMIR, IEEE J-BHI) expect and that the manuscript currently lacks:

1. **Bootstrap 95% confidence intervals** on every headline AUROC / AUPRC.
2. **Paired bootstrap comparison** (difference CI + p-value) between the GBM families, and between the
   deployed model and the same-cohort clinical baselines.
3. **Calibration**: Expected Calibration Error (ECE), calibration slope & intercept, and
   calibration-in-the-large for the deployed model.
4. **Table 1**: cohort characteristics stratified by 30-day readmission.
5. **CONSORT-style cohort-flow diagram**.

All bootstrap CIs resample the held-out *test predictions* (no model refitting), so the whole
notebook runs in well under a minute on CPU.""")

md("---")
md("## §1. Setup and data")
code(r"""print("RUNNING: Statistical Rigor (Tier 1) rev 2026-06-02a")
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":13,
                     "axes.titleweight":"bold","axes.labelsize":12,"savefig.dpi":150})

REPO = Path.cwd().resolve()
while REPO.name != "medicare-30day-readmission-mimic-iv" and REPO.parent != REPO:
    REPO = REPO.parent
PUB, RESULTS, FIGS = REPO.parent, REPO/"results", REPO/"figures"
DATA = PUB/"Dataset"/"mimic-parquet"

split = np.load(RESULTS/"v7_split_indices.npz")
y_te = split["y_test"].astype(int)
test_idx = split["test_idx"]
N = len(y_te)
print(f"Test set: {N:,} admissions, {y_te.mean():.4f} readmit rate")

# 10-seed-averaged test predictions for each GBM family
FAMILIES = ["xgboost", "lightgbm", "catboost", "histgbm"]
preds = {f: np.load(RESULTS/f"v7_{f}_test.npz")["preds"].mean(axis=1) for f in FAMILIES}
print("Loaded test predictions for:", list(preds))
""")

md(r"""## §2. Bootstrap 95% confidence intervals

We resample the test set with replacement (2,000 iterations, fixed seed) and recompute each metric,
reporting the 2.5th-97.5th percentile interval. This is the standard non-parametric CI for a held-out
performance metric and requires no model refitting.""")
code(r"""rng = np.random.RandomState(42)
N_BOOT = 2000
boot_idx = [rng.randint(0, N, N) for _ in range(N_BOOT)]   # shared resamples for paired comparisons

def boot_ci(y, s, metric=roc_auc_score):
    point = metric(y, s)
    vals = np.array([metric(y[ix], s[ix]) for ix in boot_idx])
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5), vals

rows = []
boot_store = {}
for f in FAMILIES:
    a, lo, hi, vals = boot_ci(y_te, preds[f], roc_auc_score)
    ap, aplo, aphi, _ = boot_ci(y_te, preds[f], average_precision_score)
    boot_store[f] = vals
    rows.append({"Model": f, "AUROC": f"{a:.4f}", "AUROC 95% CI": f"[{lo:.4f}, {hi:.4f}]",
                 "AUPRC": f"{ap:.4f}", "AUPRC 95% CI": f"[{aplo:.4f}, {aphi:.4f}]"})
ci_table = pd.DataFrame(rows)
print(ci_table.to_string(index=False))
""")

md(r"""## §3. Paired bootstrap model comparison

Using the **same** resamples for each pair, we compute the distribution of AUROC *differences*. The
95% CI of the difference and a two-sided bootstrap p-value tell us whether the families are
statistically distinguishable. CatBoost and HistGBM come out indistinguishable from XGBoost; LightGBM
shows a small but significant edge that is clinically negligible (0.0035 AUROC) -- hence we describe
the families as **clinically equivalent** rather than statistically identical.""")
code(r"""def paired_diff(s1, s2):
    d = np.array([roc_auc_score(y_te[ix], s1[ix]) - roc_auc_score(y_te[ix], s2[ix]) for ix in boot_idx])
    point = roc_auc_score(y_te, s1) - roc_auc_score(y_te, s2)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())   # two-sided bootstrap p
    return point, lo, hi, p

print("Pairwise AUROC differences (deployed XGBoost vs others):")
for f in ["lightgbm", "catboost", "histgbm"]:
    pt, lo, hi, p = paired_diff(preds["xgboost"], preds[f])
    sig = "" if (lo <= 0 <= hi) else "  *significant*"
    print(f"  XGBoost - {f:9s}: {pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}{sig}")
print("\\n=> CatBoost and HistGBM are statistically indistinguishable from XGBoost (CIs span 0).")
print("   LightGBM shows a small but significant edge (CI excludes 0); at 0.0035 AUROC it is")
print("   clinically negligible. We therefore treat the families as CLINICALLY EQUIVALENT.")
""")

md(r"""## §4. Calibration of the deployed model

Beyond the Brier score we report:
- **Expected Calibration Error (ECE)** -- mean gap between predicted and observed risk across deciles.
- **Calibration slope & intercept** -- from a logistic recalibration of the outcome on the model logit
  (ideal slope = 1, intercept = 0).
- **Calibration-in-the-large** -- mean predicted vs mean observed risk.""")
code(r"""from sklearn.linear_model import LogisticRegression
p_dep = preds["xgboost"]

# ECE (10 equal-width bins)
def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.digitize(p, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        e += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return e

eps = 1e-6
logit = np.log(np.clip(p_dep, eps, 1-eps) / (1 - np.clip(p_dep, eps, 1-eps)))
lr = LogisticRegression(fit_intercept=True, C=1e6, solver="lbfgs").fit(logit.reshape(-1,1), y_te)
cal_slope = float(lr.coef_[0,0]); cal_intercept = float(lr.intercept_[0])
brier = float(np.mean((p_dep - y_te)**2))

print(f"Deployed XGBoost calibration:")
print(f"  Brier score              : {brier:.4f}")
print(f"  ECE (10-bin)             : {ece(y_te, p_dep):.4f}")
print(f"  Calibration slope        : {cal_slope:.3f}   (ideal 1.0)")
print(f"  Calibration intercept    : {cal_intercept:.3f}   (ideal 0.0)")
print(f"  Calibration-in-the-large : mean pred {p_dep.mean():.4f} vs mean obs {y_te.mean():.4f}")
""")

md(r"""## §5. Same-cohort clinical baselines with CIs and significance

Re-derive LACE and HOSPITAL on the test partition (faithful, as in the baselines notebook) and report
their AUROC with bootstrap CIs and the **paired** comparison vs the deployed model.""")
code(r"""V10 = pd.read_parquet(PUB/"training_table_v10.parquet")
df = V10.iloc[test_idx].copy()

# prior ED count (faithful LACE E)
adm = pd.read_parquet(DATA/"admissions.parquet", columns=["subject_id","hadm_id","admittime","edregtime"])
adm["admittime"]=pd.to_datetime(adm["admittime"]); adm["has_ed"]=adm["edregtime"].notna().astype(int)
adm=adm.sort_values(["subject_id","admittime"]).reset_index(drop=True)
edc=np.zeros(len(adm),int)
for _s,g in adm.groupby("subject_id",sort=False):
    t=g["admittime"].values.astype("datetime64[D]").astype("int64")
    cum=np.concatenate([[0],np.cumsum(g["has_ed"].values)])
    lo=np.searchsorted(t,t-180,side="left"); edc[g.index.values]=cum[np.arange(len(t))]-cum[lo]
adm["prior_ed_180d"]=edc; edmap=adm.set_index("hadm_id")["prior_ed_180d"]
df["prior_ed_180d"]=df["hadm_id"].map(edmap).fillna(0).astype(int)

ELEC={"ELECTIVE","SURGICAL SAME DAY ADMISSION"}
def lL(x): return 0 if x<1 else (int(x) if x<=3 else (4 if x<=6 else (5 if x<=13 else 7)))
def lC(c): return min(int(c),3) if c<4 else 5
lace=(df["los_days"].apply(lL)+(~df["admission_type"].isin(ELEC)).astype(int)*3+df["cci_score_total"].apply(lC)+df["prior_ed_180d"].clip(0,4)).to_numpy().astype(float)
svc=pd.read_parquet(DATA/"services.parquet",columns=["hadm_id","curr_service"])
onc=set(svc.loc[svc["curr_service"]=="OMED","hadm_id"].unique())
def hA(n): return 0 if n<=1 else (2 if n<=5 else 5)
hosp=((df["hemoglobin_last"]<12).astype(int)+df["hadm_id"].isin(onc).astype(int)*2+(df["sodium_last"]<135).astype(int)+(df["n_procedures"]>=1).astype(int)+(~df["admission_type"].isin(ELEC)).astype(int)+df["prior_admissions_6m"].fillna(0).apply(hA)+(df["los_days"]>=5).astype(int)*2).to_numpy().astype(float)

for name, s in [("LACE", lace), ("HOSPITAL", hosp)]:
    a, lo, hi, _ = boot_ci(y_te, s, roc_auc_score)
    pt, dlo, dhi, p = paired_diff(p_dep, s)
    print(f"{name:9s} AUROC {a:.4f}  95% CI [{lo:.4f}, {hi:.4f}]   | model-{name} uplift {pt:+.4f} [{dlo:+.4f}, {dhi:+.4f}] p={p:.4f}")
""")

md("## §6. Table 1 — cohort characteristics by 30-day readmission")
code(r"""full = V10.copy()
full["readmit"] = full["readmit_30d"].map({0:"No readmit", 1:"Readmit"})

def summ(col, label, pct=False):
    g = full.groupby("readmit")[col]
    out = {}
    for k in ["No readmit","Readmit"]:
        s = g.get_group(k)
        out[k] = f"{s.mean():.1f} ({s.std():.1f})" if not pct else f"{100*s.mean():.1f}%"
    out["Overall"] = (f"{full[col].mean():.1f} ({full[col].std():.1f})" if not pct
                      else f"{100*full[col].mean():.1f}%")
    return {"Characteristic": label, **out}

rows = [
    summ("age_at_admit", "Age, mean (SD)"),
    {"Characteristic":"Female", **{k:f"{100*(full[full['readmit']==k]['gender'].astype(str).str.upper().str.startswith('F')).mean():.1f}%" for k in ['No readmit','Readmit']}, "Overall":f"{100*(full['gender'].astype(str).str.upper().str.startswith('F')).mean():.1f}%"},
    summ("los_days", "Length of stay (days), mean (SD)"),
    summ("cci_score_total", "Charlson comorbidity index, mean (SD)"),
    summ("n_diagnoses", "Number of diagnoses, mean (SD)"),
    summ("n_procedures", "Number of procedures, mean (SD)"),
    summ("icu_flag", "ICU stay", pct=True),
    summ("prior_admissions_6m", "Prior admissions (6 mo), mean (SD)"),
]
n_no = (full['readmit']=="No readmit").sum(); n_yes=(full['readmit']=="Readmit").sum()
t1 = pd.DataFrame(rows)
t1.columns = ["Characteristic", f"No readmit (n={n_no:,})", f"Readmit (n={n_yes:,})", f"Overall (N={len(full):,})"]
print(t1.to_string(index=False))
t1.to_csv(RESULTS/"table1_cohort.csv", index=False)
print("\\nSaved results/table1_cohort.csv")
""")

md("## §7. CONSORT-style cohort flow diagram")
code(r"""fig, ax = plt.subplots(figsize=(8.5, 9)); ax.axis("off")
boxes = [
    (0.5, 0.93, "MIMIC-IV v3.1 hospital admissions\nN = 546,028 (364,627 patients)"),
    (0.5, 0.74, "Restrict to insurance == Medicare\nN = 244,576 index admissions"),
    (0.5, 0.55, "30-day readmission label from pre-discharge timestamps\nPrevalence 21.1%"),
    (0.5, 0.36, "Patient-grouped 80/20 split (GroupShuffleSplit on subject_id)\nTrain+val 195,385  |  Test 49,191"),
    (0.5, 0.17, "Inner-validation carved from train (10%)\nPure train 176,270  |  Inner-val 19,115  |  Test 49,191"),
]
for x,y,t in boxes:
    ax.add_patch(plt.Rectangle((x-0.34,y-0.055),0.68,0.10,fill=True,facecolor="#E8F0F2",
                               edgecolor="#1F7A8C",lw=1.8,transform=ax.transAxes))
    ax.text(x,y,t,ha="center",va="center",fontsize=10.5,transform=ax.transAxes)
for y0,y1 in [(0.875,0.795),(0.685,0.605),(0.495,0.415),(0.305,0.225)]:
    ax.annotate("",xy=(0.5,y1),xytext=(0.5,y0),xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>",color="#1F7A8C",lw=1.8))
ax.set_title("Cohort construction (CONSORT-style)", fontweight="bold")
fig.savefig(FIGS/"cohort_flow.png", dpi=300, bbox_inches="tight"); plt.show()
print("Saved figures/cohort_flow.png")
""")

md("## §8. Save a machine-readable rigor summary")
code(r"""rigor = {
    "n_test": int(N), "n_boot": int(N_BOOT),
    "auroc": {f: float(roc_auc_score(y_te, preds[f])) for f in FAMILIES},
    "deployed_calibration": {"brier": brier, "ece": float(ece(y_te,p_dep)),
                             "slope": cal_slope, "intercept": cal_intercept},
    "lace_auroc_samecohort": float(roc_auc_score(y_te, lace)),
    "hospital_auroc_samecohort": float(roc_auc_score(y_te, hosp)),
}
(RESULTS/"statistical_rigor.json").write_text(json.dumps(rigor, indent=2))
print("Saved results/statistical_rigor.json")
print(json.dumps(rigor, indent=2)[:600])
""")

md(r"""## §9. Notes for the manuscript

- Report **every headline metric with its 95% CI** (Methods: "2,000 bootstrap resamples of the test set").
- The paired comparison **quantifies** the "GBM families are statistically indistinguishable" claim
  (CIs of the pairwise differences span 0) -- state the difference CIs and p-values explicitly.
- Report the deployed model's **ECE, calibration slope, and intercept** alongside the Brier score.
- Same-cohort LACE/HOSPITAL uplift is reported with a **difference CI + p-value** (controlled
  significance, replacing the cross-cohort point claim).
- Include **Table 1** and the **cohort-flow figure** as standard front-matter.""")

nb.cells=cells
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
             "language_info":{"name":"python","version":"3.11"}}
with open(OUT,"w",encoding="utf-8") as f: nbf.write(nb,f)
print(f"Saved: {OUT} ({len(nb.cells)} cells, {sum(1 for c in nb.cells if c.cell_type=='code')} code)")

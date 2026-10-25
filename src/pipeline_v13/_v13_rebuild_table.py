# -*- coding: utf-8 -*-
"""v13 feature table = v12 table with documentation-time availability enforced
on the remaining event-derived features: eMAR features (charttime and
storetime on or before discharge), chart-event features (charttime and
storetime), prescription features (starttime and pharmacy entertime), and the
discharge interval recomputed from any earlier admission (missing only for
first observed admissions). For every rebuilt feature the no-cutoff
recomputation is compared with the stored value to confirm the definition,
then the cutoff version replaces it. Writes training_table_v13.parquet (local)
and results_v13/table_rebuild_v13.json (aggregate, includes v11 and v12)."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, duckdb

PUB = Path(__file__).resolve().parent.parent
REPO = PUB / "medicare-30day-readmission-mimic-iv"
OUT = REPO / "results_v13"; OUT.mkdir(exist_ok=True)
RAW = Path(r"C:\Users\Thiago\Documents\Coursework\Capstone Project\Dataset\mimic-iv-3.1 - Full\mimic-iv-3.1")
PQ = PUB / "Dataset" / "mimic-parquet"
t0 = time.time()
DF = pd.read_parquet(PUB / "training_table_v12.parquet"); OLD = DF.copy()
adm = pd.read_parquet(PQ / "admissions.parquet", columns=["subject_id", "hadm_id", "admittime", "dischtime"])
coh = DF[["hadm_id"]].merge(adm[["hadm_id", "dischtime"]], on="hadm_id", how="left")
con = duckdb.connect(); con.register("cohort", coh)
R = json.loads((REPO / "results_v12" / "table_rebuild_v12.json").read_text())
R["v13_documentation_time_enforced"] = {}


def rd(path, cols):
    return (f"read_csv('{str(path).replace(chr(92), '/')}', header=true, auto_detect=true, quote='\"', "
            f"escape='\"', strict_mode=false, columns={cols})")


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def compare(name, stored, nocut, cut, tol=1e-4):
    s_ = pd.to_numeric(stored, errors="coerce").fillna(0).to_numpy()
    a_ = nocut.astype(float).fillna(0).to_numpy(); b_ = cut.astype(float).fillna(0).to_numpy()
    R["v13_documentation_time_enforced"][name] = {
        "definition_match_without_documentation_cutoff": round(float(np.isclose(s_, a_, atol=tol).mean()), 4),
        "share_changed_by_documentation_cutoff": round(float((~np.isclose(a_, b_, atol=tol)).mean()), 4),
        "mean_before": round(float(a_.mean()), 4), "mean_after": round(float(b_.mean()), 4)}


# ------------------------------------------------------------------ eMAR
EMAR = rd(RAW / "hosp" / "emar.csv.gz",
          "{'subject_id':'BIGINT','hadm_id':'BIGINT','emar_id':'VARCHAR','emar_seq':'INTEGER','poe_id':'VARCHAR',"
          "'pharmacy_id':'BIGINT','enter_provider_id':'VARCHAR','charttime':'TIMESTAMP','medication':'VARCHAR',"
          "'event_txt':'VARCHAR','scheduletime':'TIMESTAMP','storetime':'TIMESTAMP'}")
con.execute(f"""CREATE TABLE emar_c AS
SELECT e.hadm_id, e.emar_id, e.subject_id, lower(e.medication) AS med, lower(e.event_txt) AS ev,
       COALESCE(e.storetime, e.charttime) <= c.dischtime AS stored_ok
FROM {EMAR} e INNER JOIN cohort c ON e.hadm_id = c.hadm_id WHERE e.charttime <= c.dischtime""")
n_all, n_late = con.execute("SELECT COUNT(*), SUM(CASE WHEN stored_ok THEN 0 ELSE 1 END) FROM emar_c").fetchone()
R["v13_documentation_time_enforced"]["_emar_rows"] = {"charted_before_discharge": int(n_all), "stored_after_discharge": int(n_late)}
log(f"emar loaded {n_all:,} rows, {n_late:,} stored after discharge")
ABX = ["vancomycin", "piperacillin", "meropenem", "ceftriaxone", "ciprofloxacin", "azithromycin", "levofloxacin",
       "metronidazole", "cefazolin", "ampicillin", "gentamicin", "doxycycline", "cefepime", "sulfamethoxazole"]
HRM = ["insulin", "warfarin", "heparin", "morphine", "fentanyl", "hydromorphone", "oxycodone", "methadone"]
E = con.execute("SELECT hadm_id, med, ev, stored_ok FROM emar_c").fetchdf()
E["missed"] = E.ev.fillna("").str.contains("not given|held|refused", regex=True)
E["hrm"] = E.med.fillna("").str.contains("|".join(HRM), regex=True)
E["abx"] = E.med.fillna("").str.contains("|".join(ABX), regex=True)
EMF = {}
for tag, sub in (("nocut", E), ("cut", E[E.stored_ok])):
    g = sub.groupby("hadm_id")
    EMF[tag] = pd.DataFrame({"n_med_administrations": g.size(), "missed_doses_rate": g.missed.mean(),
                             "high_risk_med_flag": g.hrm.max().astype(int),
                             "antibiotic_flag": g.abx.max().astype(int)}).reset_index()
del E
log("emar flags done")
EMD = rd(RAW / "hosp" / "emar_detail.csv.gz",
         "{'subject_id':'BIGINT','emar_id':'VARCHAR','emar_seq':'INTEGER','parent_field_ordinal':'VARCHAR',"
         "'administration_type':'VARCHAR','pharmacy_id':'BIGINT','barcode_type':'VARCHAR','reason_for_no_barcode':'VARCHAR',"
         "'complete_dose_not_given':'VARCHAR','dose_due':'VARCHAR','dose_due_unit':'VARCHAR','dose_given':'VARCHAR',"
         "'dose_given_unit':'VARCHAR','will_remainder_of_dose_be_given':'VARCHAR','product_amount_given':'VARCHAR',"
         "'product_unit':'VARCHAR','product_code':'VARCHAR','product_description':'VARCHAR','product_description_other':'VARCHAR',"
         "'prior_infusion_rate':'VARCHAR','infusion_rate':'VARCHAR','infusion_rate_adjustment':'VARCHAR',"
         "'infusion_rate_adjustment_amount':'VARCHAR','infusion_rate_unit':'VARCHAR','route':'VARCHAR','infusion_complete':'VARCHAR',"
         "'completion_interval':'VARCHAR','new_iv_bag_hung':'VARCHAR','continued_infusion_in_other_location':'VARCHAR',"
         "'restart_interval':'VARCHAR','side':'VARCHAR','site':'VARCHAR','non_formulary_visual_verification':'VARCHAR'}")
con.execute(f"""CREATE TABLE emd_c AS
SELECT e.hadm_id, e.stored_ok,
       lower(ed.complete_dose_not_given) IN ('true','t','1') AS incomplete,
       (contains(upper(ed.route), 'IV') OR contains(upper(ed.administration_type), 'IV')) AS iv,
       ed.route
FROM emar_c e INNER JOIN {EMD} ed ON e.emar_id = ed.emar_id AND e.subject_id = ed.subject_id""")
log("emar detail joined")
EMDF = {}
for tag, cond in (("nocut", "TRUE"), ("cut", "stored_ok")):
    EMDF[tag] = con.execute(f"""SELECT hadm_id, COUNT(*) AS n_emar_details,
        SUM(CASE WHEN incomplete THEN 1 ELSE 0 END) AS n_incomplete_doses,
        SUM(CASE WHEN incomplete THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS incomplete_dose_rate,
        SUM(CASE WHEN iv THEN 1 ELSE 0 END) AS n_iv_administrations,
        SUM(CASE WHEN iv THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS iv_admin_rate,
        COUNT(DISTINCT route) AS n_distinct_routes
        FROM emd_c WHERE {cond} GROUP BY hadm_id""").fetchdf()
con.execute("DROP TABLE emd_c")
log("emar detail features done")

# ------------------------------------------------------ prescriptions / pharmacy
RX = rd(RAW / "hosp" / "prescriptions.csv.gz",
        "{'subject_id':'BIGINT','hadm_id':'BIGINT','pharmacy_id':'BIGINT','poe_id':'VARCHAR','poe_seq':'INTEGER',"
        "'order_provider_id':'VARCHAR','starttime':'TIMESTAMP','stoptime':'TIMESTAMP','drug_type':'VARCHAR','drug':'VARCHAR',"
        "'formulary_drug_cd':'VARCHAR','gsn':'VARCHAR','ndc':'VARCHAR','prod_strength':'VARCHAR','form_rx':'VARCHAR',"
        "'dose_val_rx':'VARCHAR','dose_unit_rx':'VARCHAR','form_val_disp':'VARCHAR','form_unit_disp':'VARCHAR',"
        "'doses_per_24_hrs':'DOUBLE','route':'VARCHAR'}")
PH = rd(RAW / "hosp" / "pharmacy.csv.gz",
        "{'subject_id':'BIGINT','hadm_id':'BIGINT','pharmacy_id':'BIGINT','poe_id':'VARCHAR','starttime':'TIMESTAMP',"
        "'stoptime':'TIMESTAMP','medication':'VARCHAR','proc_type':'VARCHAR','status':'VARCHAR','entertime':'TIMESTAMP',"
        "'verifiedtime':'TIMESTAMP','route':'VARCHAR','frequency':'VARCHAR','disp_sched':'VARCHAR','infusion_type':'VARCHAR',"
        "'sliding_scale':'VARCHAR','lockout_interval':'VARCHAR','basal_rate':'DOUBLE','one_hr_max':'VARCHAR',"
        "'doses_per_24_hrs':'DOUBLE','duration':'DOUBLE','duration_interval':'VARCHAR','expiration_value':'DOUBLE',"
        "'expiration_unit':'VARCHAR','expirationdate':'TIMESTAMP','dispensation':'VARCHAR','fill_quantity':'VARCHAR'}")
con.execute(f"CREATE TABLE ph AS SELECT pharmacy_id, entertime FROM {PH} p WHERE p.hadm_id IN (SELECT hadm_id FROM cohort)")
X = con.execute(f"""SELECT r.hadm_id, lower(r.drug) AS drug,
       (r.stoptime IS NULL OR r.stoptime >= c.dischtime) AS at_disch,
       COALESCE(p.entertime, r.starttime) <= c.dischtime AS entered_ok
FROM {RX} r INNER JOIN cohort c ON r.hadm_id = c.hadm_id LEFT JOIN ph p ON r.pharmacy_id = p.pharmacy_id
WHERE r.starttime <= c.dischtime""").fetchdf()
R["v13_documentation_time_enforced"]["_prescription_rows"] = {"started_before_discharge": int(len(X)),
                                                              "entered_after_discharge": int((~X.entered_ok).sum())}
log(f"prescriptions loaded {len(X):,} rows, {int((~X.entered_ok).sum()):,} entered after discharge")
OPI = ["oxycodone", "morphine", "hydromorphone", "fentanyl", "tramadol", "codeine", "methadone"]
ANT = ["warfarin", "heparin", "enoxaparin", "rivaroxaban", "apixaban", "dabigatran"]
DIU = ["furosemide", "bumetanide", "spironolactone", "hydrochlorothiazide"]
X["drug"] = X.drug.fillna("")
X["opi"] = X.drug.str.contains("|".join(OPI), regex=True) & X.at_disch
X["ant"] = X.drug.str.contains("|".join(ANT), regex=True) & X.at_disch
X["ins"] = X.drug.str.contains("insulin") & X.at_disch
X["diu"] = X.drug.str.contains("|".join(DIU), regex=True) & X.at_disch
RXF = {}
for tag, sub in (("nocut", X), ("cut", X[X.entered_ok])):
    g = sub.groupby("hadm_id")
    nd = sub[sub.at_disch].groupby("hadm_id").drug.nunique()
    f = pd.DataFrame({"n_meds_total": g.size(), "distinct_drugs": g.drug.nunique(),
                      "n_discharge_drugs": nd.reindex(g.size().index).fillna(0),
                      "opioid_at_discharge": g.opi.max().astype(int), "anticoag_at_discharge": g.ant.max().astype(int),
                      "insulin_at_discharge": g.ins.max().astype(int), "diuretic_at_discharge": g.diu.max().astype(int)})
    f["severe_polypharmacy"] = (f.n_discharge_drugs >= 10).astype(int)
    RXF[tag] = f.reset_index()
del X
log("prescription features done")

# ---------------------------------------------------------------- chart events
CH = rd(RAW / "icu" / "chartevents.csv.gz",
        "{'subject_id':'BIGINT','hadm_id':'BIGINT','stay_id':'BIGINT','caregiver_id':'BIGINT','charttime':'TIMESTAMP',"
        "'storetime':'TIMESTAMP','itemid':'INTEGER','value':'VARCHAR','valuenum':'DOUBLE','valueuom':'VARCHAR','warning':'INTEGER'}")
VENT = [224385, 224686, 224687, 224688, 224422, 224689, 224690]
items = [220045, 220050, 220739, 223900, 223901] + VENT
con.execute(f"""CREATE TABLE ch_c AS
SELECT v.hadm_id, v.charttime, v.itemid, v.valuenum, COALESCE(v.storetime, v.charttime) <= c.dischtime AS stored_ok
FROM {CH} v INNER JOIN cohort c ON v.hadm_id = c.hadm_id
WHERE v.itemid IN ({",".join(map(str, items))}) AND v.charttime <= c.dischtime""")
c_all, c_late = con.execute("SELECT COUNT(*), SUM(CASE WHEN stored_ok THEN 0 ELSE 1 END) FROM ch_c").fetchone()
R["v13_documentation_time_enforced"]["_chart_rows"] = {"charted_before_discharge": int(c_all), "stored_after_discharge": int(c_late)}
log(f"chartevents loaded {c_all:,} rows, {c_late:,} stored after discharge")
CHF = {}
for tag, cond in (("nocut", "TRUE"), ("cut", "stored_ok")):
    CHF[tag] = con.execute(f"""
WITH g AS (SELECT hadm_id, charttime,
    MAX(CASE WHEN itemid=220739 THEN valuenum END) AS e, MAX(CASE WHEN itemid=223900 THEN valuenum END) AS vb,
    MAX(CASE WHEN itemid=223901 THEN valuenum END) AS m
  FROM ch_c WHERE itemid IN (220739,223900,223901) AND valuenum IS NOT NULL AND {cond} GROUP BY hadm_id, charttime),
gt AS (SELECT hadm_id, MIN(COALESCE(e,1)+COALESCE(vb,1)+COALESCE(m,1)) AS gcs_min FROM g
       WHERE e IS NOT NULL OR vb IS NOT NULL OR m IS NOT NULL GROUP BY hadm_id),
s AS (SELECT hadm_id, STDDEV(CASE WHEN itemid=220045 AND valuenum > 0 THEN valuenum END) AS hr_std,
             STDDEV(CASE WHEN itemid=220050 AND valuenum > 0 THEN valuenum END) AS sbp_std
      FROM ch_c WHERE {cond} AND valuenum IS NOT NULL GROUP BY hadm_id),
vt AS (SELECT DISTINCT hadm_id, 1 AS ventilator_flag FROM ch_c WHERE itemid IN ({",".join(map(str, VENT))}) AND {cond})
SELECT c.hadm_id, s.hr_std, s.sbp_std, gt.gcs_min, COALESCE(vt.ventilator_flag, 0) AS ventilator_flag
FROM cohort c LEFT JOIN s ON s.hadm_id=c.hadm_id LEFT JOIN gt ON gt.hadm_id=c.hadm_id LEFT JOIN vt ON vt.hadm_id=c.hadm_id""").fetchdf()
log("chart features done")

# -------------------------------------------------------------- assemble
base = DF[["hadm_id"]]
NC = base.merge(EMF["nocut"], on="hadm_id", how="left").merge(EMDF["nocut"], on="hadm_id", how="left") \
         .merge(RXF["nocut"], on="hadm_id", how="left").merge(CHF["nocut"], on="hadm_id", how="left")
CT = base.merge(EMF["cut"], on="hadm_id", how="left").merge(EMDF["cut"], on="hadm_id", how="left") \
         .merge(RXF["cut"], on="hadm_id", how="left").merge(CHF["cut"], on="hadm_id", how="left")
ZERO_FILL = {"n_med_administrations", "missed_doses_rate", "high_risk_med_flag", "antibiotic_flag", "n_emar_details",
             "n_incomplete_doses", "incomplete_dose_rate", "n_iv_administrations", "iv_admin_rate", "n_distinct_routes",
             "n_meds_total", "distinct_drugs", "n_discharge_drugs", "severe_polypharmacy", "opioid_at_discharge",
             "anticoag_at_discharge", "insulin_at_discharge", "diuretic_at_discharge", "ventilator_flag"}
for c in [c for c in CT.columns if c != "hadm_id" and c in DF.columns]:
    nc, ct = NC[c].astype(float), CT[c].astype(float)
    if c in ZERO_FILL:
        nc, ct = nc.fillna(0), ct.fillna(0)
    compare(c, DF[c], nc, ct)
    DF[c] = ct
log("assembled")

# ----------------------------------------- discharge interval from any admission
a = adm.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
a["prev_dis"] = a.groupby("subject_id").dischtime.shift(1)
a["tsld_any"] = ((a.admittime - a.prev_dis).dt.total_seconds() / 86400).clip(lower=0)
m = DF[["hadm_id", "prior_admissions_all"]].merge(a[["hadm_id", "tsld_any"]], on="hadm_id", how="left")
assert (m.hadm_id.values == DF.hadm_id.values).all()
old_t = pd.to_numeric(DF.time_since_last_discharge, errors="coerce")
R["v13_documentation_time_enforced"]["time_since_last_discharge"] = {
    "definition": "days from the previous discharge of any insurance to the index admission; missing if none",
    "share_missing_new": round(float(m.tsld_any.isna().mean()), 4),
    "share_missing_old": round(float(old_t.isna().mean()), 4),
    "share_old_fill_80_22_remaining": round(float(np.isclose(old_t.fillna(-1), 80.22013888888888, atol=1e-6).mean()), 4),
    "share_changed": round(float((~np.isclose(old_t.fillna(-9), m.tsld_any.fillna(-9), atol=0.01)).mean()), 4)}
assert (m.tsld_any.isna() == (DF.prior_admissions_all == 0)).all()
DF["time_since_last_discharge"] = m.tsld_any
DF["log_time_since_discharge"] = np.log1p(DF.time_since_last_discharge)
DF["freq_x_recency"] = np.where(DF.prior_admissions_all == 0, 0.0, DF.prior_admissions_all / (DF.time_since_last_discharge + 1.0))
R["v13_seconds"] = round(time.time() - t0)
DF.to_parquet(PUB / "training_table_v13.parquet", index=False)
(OUT / "table_rebuild_v13.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R["v13_documentation_time_enforced"], indent=1))

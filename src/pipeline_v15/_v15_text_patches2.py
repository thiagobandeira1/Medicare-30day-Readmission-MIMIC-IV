# -*- coding: utf-8 -*-
"""v15, second patch: one timestamp rule per final predictor family in Audit_207."""
from pathlib import Path

H = Path(__file__).resolve().parent


def patch(fn, pairs):
    p = H / fn; s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        if s.count(a) == 0 and s.count(b) >= 1:
            continue
        assert s.count(a) == 1, (fn, s.count(a), a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8"); print("patched", fn)


patch("_v15_appendices.py", [
 (' "n_meds_total": ("Prescriptions started on or before discharge (v12)", "index stay, on or before discharge", 1.0),',
  ' "n_meds_total": ("Prescriptions started on or before discharge whose pharmacy entry time is also on or before discharge (v12 start time; v13 entry time)", "index stay, on or before discharge", 1.0),'),
 ('''for f_ in ("n_labs_total", "lab_abnormal_rate", "n_lab_item_types", "n_critical_labs", "n_meds_total", "log_n_labs",''',
  '''for f_ in ("n_labs_total", "lab_abnormal_rate", "n_lab_item_types", "n_critical_labs", "log_n_labs",'''),
 ('''audit.loc[audit["feature_name"] == "n_order_types", "timestamp_used"] = (
    "ordertime on or before discharge (source extraction filter; spot-checked)")''',
  '''audit.loc[audit["feature_name"] == "n_order_types", "timestamp_used"] = (
    "ordertime on or before discharge (source extraction filter; spot-checked)")
# one timestamp rule per feature family for the remaining final predictors (replaces the generic v4 audit label)
_TS = {
 "ordertime on or before discharge (source extraction filter; spot-checked)":
     ["orders_per_day", "n_lab_orders", "n_late_orders", "late_order_rate", "late_order_burden",
      "consult_per_day", "provider_fragmentation_x_los", "orders_last_6h"],
 "admittime and dischtime of earlier admissions relative to the index admittime (v8 recomputation from raw timestamps)":
     ["prior_admissions_6m", "prior_mean_los_6m", "prior_admissions_all", "prior_admits_6m_sq",
      "log_prior_readmit_count"],
 "admittime and dischtime of earlier admissions (any insurance) relative to the index admittime (v13 interval)":
     ["freq_x_recency", "log_time_since_discharge"],
 "admittime and dischtime of the index admission (admissions table fields fixed at discharge)":
     ["admission_type", "admission_location", "discharge_location", "age_at_admit", "los_days",
      "log_los_days", "discharge_surge"],
 "index admittime and dischtime; earlier admissions by admittime and dischtime":
     ["los_per_prior_admit"],
 "transfer intime on or before discharge (source extraction filter)":
     ["stepdown_flag"],
 "albumin: charttime and storetime on or before discharge (v11 to v12); length of stay from admittime and dischtime":
     ["albumin_x_los"],
 "prescription starttime (v12) and pharmacy entertime (v13) on or before discharge":
     ["n_meds_total"],
}
for _lab, _fs in _TS.items():
    audit.loc[audit["feature_name"].isin(_fs), "timestamp_used"] = _lab
_WIN = {"all earlier admissions": ["freq_x_recency", "log_time_since_discharge", "prior_admissions_all"],
        "prior 365 days": ["log_prior_readmit_count"],
        "prior 180 days": ["prior_admits_6m_sq"],
        "index stay and all earlier admissions": ["los_per_prior_admit"]}
for _w, _fs in _WIN.items():
    audit.loc[audit["feature_name"].isin(_fs), "measurement_window"] = _w'''),
])

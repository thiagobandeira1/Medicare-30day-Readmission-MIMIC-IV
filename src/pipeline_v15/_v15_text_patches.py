# -*- coding: utf-8 -*-
"""v15 reporting corrections from review round 5 (no model change; results_v13)."""
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


patch("_build_jamia_supp_v15.py", [
 # S13: the crude gap is the observed White minus Black difference from the joint analysis (0.0029)
 ('''  f"crude {RU['crude_gap']:.4f}. Median prior admissions were "''',
  '''  f"crude (observed, unstandardized) "
  f"{SC['white_minus_black_auroc_joint']['point_observed']:.4f}. Median prior admissions were "'''),
 # S3: the v12 count rebuild rows live in Cutoff_Rebuild
 ('''  f"Appendix 1, Table_Rebuild). Two features that were exact copies of "''',
  '''  f"Appendix 1, Cutoff_Rebuild). Two features that were exact copies of "'''),
])

patch("_v15_appendices.py", [
 # specificity contrast is descriptive: no P value in the workbook
 ('''                    "p_two_sided_plus1": v["p_two_sided_plus1"], "n_boot": v["n_boot"], "design": v["design"],''',
  '''                    "p_two_sided_plus1": (v["p_two_sided_plus1"] if k == "white_minus_black_auroc_joint"
                                          else None),
                    "n_boot": v["n_boot"], "design": v["design"],'''),
 # Cutoff_Rebuild: drop the legacy 80.22 column (the dict spread copied it before the pop)
 ('''                 [{"feature": k, "version": "v13 (documentation-time cutoff)",
                   **({**v, "share_old_fill_80_22_before_v13_rebuild": v.pop("share_old_fill_80_22_remaining"),
                       "share_fill_80_22_after_v13_rebuild": 0.0} if "share_old_fill_80_22_remaining" in v else v)}
                  for k, v in TB.get("v13_documentation_time_enforced", {}).items()]''',
  '''                 [{"feature": k, "version": "v13 (documentation-time cutoff)",
                   **{kk: vv for kk, vv in v.items() if kk != "share_old_fill_80_22_remaining"},
                   **({"share_old_fill_80_22_before_v13_rebuild": v["share_old_fill_80_22_remaining"],
                       "share_fill_80_22_after_v13_rebuild": 0.0} if "share_old_fill_80_22_remaining" in v else {})}
                  for k, v in TB.get("v13_documentation_time_enforced", {}).items()]'''),
 # Audit_207 timestamp labels must match the feature definitions
 ('''k = audit["feature_name"] == "los_trend_180d"
audit.loc[k, "verified_definition_v8"] = ("Per-patient summary over the whole record (constant within "''',
  '''audit.loc[audit["feature_name"] == "time_since_last_discharge", "timestamp_used"] = (
    "dischtime of the previous admission (any insurance) and admittime of the index admission (v13)")
audit.loc[audit["feature_name"] == "n_order_types", "timestamp_used"] = (
    "ordertime on or before discharge (source extraction filter; spot-checked)")
k = audit["feature_name"] == "los_trend_180d"
audit.loc[k, "verified_definition_v8"] = ("Per-patient summary over the whole record (constant within "'''),
])

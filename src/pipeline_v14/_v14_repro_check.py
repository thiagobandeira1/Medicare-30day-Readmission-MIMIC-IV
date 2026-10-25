# -*- coding: utf-8 -*-
"""Regenerate headline values from stored predictions and compare with the
v10 manuscript text; compare the rerun episode-end JSON with the original."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from docx import Document
PUB = Path(__file__).resolve().parent.parent
O7 = PUB / "medicare-30day-readmission-mimic-iv" / "results_v13"
txt = "\n".join(p.text for p in Document(PUB / "Paper JAMIA Submission v14.docx").paragraphs)
sup = "\n".join(p.text for p in Document(PUB / "Paper JAMIA Supplementary Material v14.docx").paragraphs)
rng = np.random.default_rng(42)
F = np.load(O7 / "final_preds_v13.npz"); O = np.load(O7 / "oof_predictions.npz"); Bz = np.load(O7 / "baseline_scores_test_v13.npz")
def boot(y, p, s, B=1000):
    us = np.unique(s); m = {u: np.where(s == u)[0] for u in us}
    v = [roc_auc_score(y[ix], p[ix]) for ix in (np.concatenate([m[u] for u in rng.choice(us, len(us))]) for _ in range(B))]
    return np.percentile(v, [2.5, 97.5])
chk = []
def add(name, val, fmt, where=txt):
    s = fmt.format(val); chk.append((name, s, s in where))
add("OOF AUROC", roc_auc_score(O["y"], O["rfe"]), "{:.4f}")
add("test AUROC", roc_auc_score(F["y_te"], F["p_te"]), "{:.4f}")
lo, hi = boot(F["y_te"], F["p_te"], F["s_te"])
FMJ = json.loads((O7 / "final_model_v13.json").read_text()); _c = FMJ["metric_panel"]["auroc"]["ci95"]
chk.append(("test AUROC CI (regenerated vs reported)", f"{lo:.4f} to {hi:.4f} vs {_c[0]:.4f} to {_c[1]:.4f}", abs(lo-_c[0]) < 0.002 and abs(hi-_c[1]) < 0.002))
add("LACE AUROC", roc_auc_score(Bz["y_te"], Bz["lace"]), "{:.4f}")
add("HOSPITAL AUROC", roc_auc_score(Bz["y_te"], Bz["hospital12"]), "{:.4f}")
A = np.load(O7 / "aft_test_preds_v13.npz")
d1 = roc_auc_score(((A["event"] == 1) & (A["tt"] <= 1)).astype(int), -A["predt"])
add("day-1 AUROC primary", d1, "{:.3f}")
E1 = json.loads((O7 / "episode_end_rule_v13.json").read_text())
E2 = json.loads((O7 / "episode_end_rule_v13_rerun.json").read_text())
chk.append(("episode-end JSON identical on rerun", "", E1 == E2))
add("episode frozen AUROC", E1["test_auroc_final_model"], "{:.4f}")
add("episode refit AUROC", E1["test_auroc_refit"], "{:.4f}")
add("episode day-1", E1["timing_episode_rule"]["auc"][0], "{:.3f}")
add("episode landmarks", E1["landmark_episode_rule"][2]["auroc"], "{:.3f}")
add("episode returns (supp)", E1["continuations_with_later_return_all"], "{}", sup)
for r in chk: print(("PASS " if r[2] else "FAIL ") + r[0] + " " + r[1])

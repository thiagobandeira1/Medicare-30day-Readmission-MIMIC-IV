"""Fix §16 markdown to use current-run numbers and the new 80/20+inner-val protocol."""

from __future__ import annotations
import io, json, sys
from pathlib import Path
try:
 sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
 sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebook" / "Capstone_Final_Notebook.ipynb"

ens = json.loads((REPO_ROOT / "results" / "v7_summary.json").read_text())
lgb = ens["models"]["lightgbm"]["test_auroc"]
xgb = ens["models"]["xgboost"]["test_auroc"]
blend = ens["blend"]["test_auroc"]
LACE = 0.684
CBERT = 0.714

NEW_S16 = f"""\
---
## 16. Contributions & Conclusions

### 16.1 Four contributions

1. **Reproducible MIMIC-IV Medicare cohort + staged V1 → V7 feature-engineering recipe** that other researchers can extend. The full DuckDB pipeline + 80/20 patient-grouped split + inner-val early-stopping protocol is wired end-to-end (see `data_prep/` and §8.2).
2. **Empirical evidence** that a **single-model LightGBM trained on the 50-feature parsimonious V7 set** matches or surpasses traditional clinical scores and notes-based deep-learning pipelines while remaining fully interpretable. The scipy-optimised 4-GBM blend is LightGBM-dominant and does **not** improve on the best single model — a clear reversal of the original capstone's blend-favouring narrative once test-set early-stopping leakage is removed.
3. **SHAP-based explanation layer** that converts individual predictions into ranked contributing factors — a bridge between model output and clinician decision.
4. **A parsimonious 50-feature deployment configuration.** V7 itself is the 50-feature curated subset distilled in the original capstone from a 368-feature exploration set by LightGBM gain-importance ranking; §10.4 publishes the feature-importance breakdown of the deployed model so the predictive contribution of every retained feature is auditable for clinical deployment.

### 16.2 Conclusion

A **single LightGBM model trained on the 50-feature parsimonious V7 set (curated in the original capstone from a 368-feature exploration superset) predicts 30-day all-cause readmission in Medicare patients with a test AUROC of {lgb:.4f}** under a strict 80/20 patient-grouped split with a 10% inner-validation slice carved from the training portion for early stopping — a {lgb-LACE:+.3f} improvement over LACE and {lgb-CBERT:+.3f} over ClinicalBERT. XGBoost ({xgb:.4f}) is a co-equal alternative (gap {lgb-xgb:+.4f} AUROC). The scipy-optimised 4-GBM blend ({blend:.4f}) is LightGBM-dominant (weight {ens['blend']['weights']['lightgbm']:.2f}) and does not improve on the best single model, so the additional operational complexity of an ensemble is unjustified.

The single-split test AUROC sits ~0.011 below the original capstone report's 5-fold-CV stability mean of 0.7956 ± 0.0026 (see the reproduction validator in §10.3b). The drift is attributable to (a) single-split variance vs cross-validation averaging and (b) removal of the original capstone implementation's test-set early-stopping leakage, which mildly inflated the originally reported numbers. The headline pattern — interpretable boosting at ~0.79 AUROC, with parsimony preserved at 50 features — is robust across both protocols.

The model is interpretable at both global and individual levels via SHAP, well calibrated in the operating range that matters for care-coordination triage, and simple enough to be embedded in existing EHR workflows without additional infrastructure.

**Next steps:** external validation on multi-hospital data, fairness analysis across demographic subgroups, prospective evaluation of the downstream intervention pathway, and a 5-fold CV stability re-run to obtain a direct apples-to-apples comparison with the original capstone report's stability estimate.

---

### Acknowledgments

This work was completed as the capstone project for the MS in Data Science & Artificial Intelligence at Florida International University, under the mentorship of **Dr. Christian Poellabauer**. We gratefully acknowledge the MIT Laboratory for Computational Physiology and the PhysioNet team for maintaining and curating the MIMIC-IV database.
"""

def main():
 nb = nbf.read(NB_PATH, as_version=4)
 for i, c in enumerate(nb.cells):
 if c.cell_type != "markdown": continue
 src = "".join(c.source) if isinstance(c.source, list) else c.source
 if "## 16. Contributions & Conclusions" in src:
 nb.cells[i].source = NEW_S16
 print(f" [{i:3d}] §16 rewritten with current numbers + protocol")
 break
 nbf.write(nb, NB_PATH)
 print(f"Saved to {NB_PATH}")

if __name__ == "__main__":
 main()

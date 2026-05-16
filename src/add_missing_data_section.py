"""Insert §5.3 Missing-Data Profile and Handling Strategy section.

Inserts a markdown cell + code cell between §5 (cohort load) and §6 (EDA).
The new section documents:
  - Sources of raw-data missingness in MIMIC-IV
  - Two-stage handling: upstream conditional imputation in the DuckDB feature
    engineering pipeline + native NaN handling by tree models at the modelling
    layer
  - A code cell that profiles the V7 parquet to count NaN, characterise the
    zero-share footprint, and write the per-column profile to results/

Idempotent — re-running replaces the section's contents in place.

Usage:
    python scripts/add_missing_data_section.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebook" / "Capstone_Final_Notebook.ipynb"


MD_CELL = """\
### 5.3 Missing-Data Profile and Handling Strategy

Raw MIMIC-IV lab / vital / medication tables contain extensive missingness — clinicians order tests selectively, so any given lab (e.g. BNP, lactate, albumin) is undefined for the majority of admissions. We handle missingness in **two stages**.

#### Stage 1 — Upstream feature-engineering pipeline (DuckDB)

The V7 (V17) training table is constructed with **deliberate, feature-class-specific** handling of source missingness. Rather than applying one global imputation rule, each feature class is treated according to whether its missingness carries clinical meaning:

| Feature class | Examples (V7) | Source missingness | Strategy in V7 |
|---|---|---|---|
| **Count / event features** | `prior_admissions_6m`, `orders_last_6h`, `orders_first_24h`, `n_order_types`, `prior_readmission_count`, `n_emar_details` | Undefined when nothing measured | **Zero** — semantically valid ("no event observed → count is 0") |
| **Aggregate measurements** | `bmi_last`, `sodium_last`, `bilirubin_max`, `bp_diastolic_outpatient` | Undefined when the lab / vital was never ordered | **NaN preserved** — passed through to the tree models so the split learner can branch on missingness |
| **Derived deltas / trends** | `los_trend_180d`, `los_trend_x_prior_6m`, `los_trend_x_prior_admits` | Undefined when there is no comparison point | **Zero** — represents "no change observed" |
| **Target encodings** | `discharge_location_te`, `drg_code_te`, `primary_dx_chapter_te`, `last_drg_dispo_te`, `race_te` | Undefined when category never seen in the training fold | **Global cohort positive rate** as a smoothing prior (out-of-fold encoding to avoid leakage) |
| **Interaction features** | `prior_admits_x_age`, `renal_risk_x_age`, `high_risk_meds_x_age`, `freq_x_recency`, `severity_x_readmit`, `los_x_n_diagnoses`, `anemia_x_prior_admits` | Inherit missingness from constituent base columns | **NaN preserved** when either base is missing |
| **Log / polynomial transforms** | `log_n_labs`, `log_prior_admits_6m`, `log_prior_readmit_count`, `log_time_since_discharge`, `prior_admits_6m_sq` | Inherit from base | `log1p` / `square` of the imputed-zero base, so naturally zero |

#### Missingness as signal

Unlike a fully pre-imputed table, **V7 deliberately preserves `NaN` where missingness is itself a clinical signal** (most labs and vitals — a "missing creatinine" tells the model the clinician did not order one, which is informative risk evidence). The dataset therefore contains literal `NaN` values, which the modelling layer is built to consume natively.

#### Stage 2 — Downstream modelling pipeline

- **Tree-based models** (LightGBM, XGBoost, CatBoost, HistGradientBoosting) handle `NaN` natively. Each split learns whether `NaN` routes to the left or right child during fitting, treating missingness as an informative branch rather than a value to be filled. **This is the primary reason V7 selected gradient boosting** — it lets the model exploit missingness as evidence rather than discard it through imputation.
- **Logistic regression and MLP** baselines (§9) wrap inputs in a `sklearn.pipeline.Pipeline` that begins with `SimpleImputer(strategy="median")` followed by `StandardScaler`. The imputer's median is fit on the training split only, so no value statistics leak across the train / validation / test boundary.

#### Empirical confirmation

The code cell below profiles the V7 training table — counts the total `NaN` entries, identifies which columns carry them and at what rate, and characterises the implicit-imputation footprint (proportion of zero-valued entries by feature) for columns where the upstream pipeline filled missing values with zero.
"""


CODE_CELL = """\
# ── 5.3 Missing-data profile (V7) ──────────────────────────────────────────
# Profiles literal NaN content (preserved-as-signal) and zero-share content
# (proxy for "where the upstream pipeline filled missingness with 0").

df_v7_profile = pd.read_parquet(PATHS["v7"])
total_nan = df_v7_profile.isna().sum().sum()
print(f"V7 training table: {df_v7_profile.shape[0]:,} admissions x {df_v7_profile.shape[1]} columns")
print(f"Total literal NaN values: {total_nan:,}")
print(f"  -> retained where source missingness is itself informative (labs / vitals).")
print()

# Per-column NaN summary
nan_per_col = df_v7_profile.isna().mean().sort_values(ascending=False)
nan_cols = nan_per_col[nan_per_col > 0]
print(f"Columns carrying NaN: {len(nan_cols)} / {df_v7_profile.shape[1]}")
if len(nan_cols) > 0:
    print("\\nTop columns by NaN rate:")
    for col, rate in nan_cols.head(15).items():
        print(f"  {col:32s}  {rate*100:6.2f}% missing")

# Zero-share for numeric columns (proxy for "imputed zero where event absent")
numeric_cols = [c for c in df_v7_profile.select_dtypes(include="number").columns
                if c not in {"subject_id", "hadm_id", "readmit_30d"}]

rows = []
for c in numeric_cols:
    s = df_v7_profile[c]
    s_nonan = s.dropna()
    rows.append({
        "feature": c,
        "n_unique": int(s.nunique(dropna=True)),
        "pct_nan":  float(s.isna().mean()) * 100.0,
        "pct_zero_of_nonnan": float((s_nonan == 0).mean()) * 100.0 if len(s_nonan) else 0.0,
        "min":    float(s_nonan.min())    if len(s_nonan) else float("nan"),
        "median": float(s_nonan.median()) if len(s_nonan) else float("nan"),
        "max":    float(s_nonan.max())    if len(s_nonan) else float("nan"),
    })
profile_df = pd.DataFrame(rows).sort_values("pct_zero_of_nonnan", ascending=False)

zero_heavy = (profile_df.pct_zero_of_nonnan > 50).sum()
zero_med   = ((profile_df.pct_zero_of_nonnan > 10) & (profile_df.pct_zero_of_nonnan <= 50)).sum()
zero_low   = ((profile_df.pct_zero_of_nonnan > 0)  & (profile_df.pct_zero_of_nonnan <= 10)).sum()
zero_none  = (profile_df.pct_zero_of_nonnan == 0).sum()

print()
print("Implicit-zero footprint (zero-share among non-NaN values):")
print(f"  >50% zeros : {int(zero_heavy):3d} features  (likely count / event features where 0 = none observed)")
print(f"  10-50%     : {int(zero_med):3d} features")
print(f"  0-10%      : {int(zero_low):3d} features")
print(f"  exactly 0% : {int(zero_none):3d} features  (continuous quantities — never imputed-to-zero)")

# Persist for reference
ART_DIR.mkdir(parents=True, exist_ok=True)
profile_df.to_csv(ART_DIR / "missing_data_profile_v7.csv", index=False)
print(f"\\nFull per-feature profile saved to {ART_DIR / 'missing_data_profile_v7.csv'}")

del df_v7_profile  # free memory before EDA cells load V1
"""


def find_cell(nb, needle):
    for i, c in enumerate(nb.cells):
        src = "".join(c.source) if isinstance(c.source, list) else c.source
        if needle in src:
            return i
    raise LookupError(f"cell containing {needle!r} not found")


def main():
    nb = nbf.read(NB_PATH, as_version=4)
    print(f"Loaded {NB_PATH.name}: {len(nb.cells)} cells")

    try:
        md_idx = find_cell(nb, "### 5.3 Missing-Data Profile")
        code_idx = find_cell(nb, "# ── 5.3 Missing-data profile (V7)")
        nb.cells[md_idx].source = MD_CELL
        nb.cells[code_idx].source = CODE_CELL
        nb.cells[code_idx].outputs = []
        nb.cells[code_idx].execution_count = None
        print(f"  [{md_idx:3d},{code_idx:3d}] §5.3 markdown + code refreshed (already present)")
    except LookupError:
        anchor = find_cell(nb, "# Load V1 for exploratory analysis")
        md_cell  = nbf.v4.new_markdown_cell(source=MD_CELL)
        code_cell = nbf.v4.new_code_cell(source=CODE_CELL)
        nb.cells.insert(anchor + 1, md_cell)
        nb.cells.insert(anchor + 2, code_cell)
        print(f"  inserted §5.3 (markdown at {anchor + 1}, code at {anchor + 2})")

    nbf.write(nb, NB_PATH)
    print(f"\nSaved to {NB_PATH}  (now {len(nb.cells)} cells)")


if __name__ == "__main__":
    main()

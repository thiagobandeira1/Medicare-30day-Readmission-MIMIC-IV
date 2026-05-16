"""V7 50-feature parsimony analysis (validates the manuscript's parsimony claim).

The original capstone claims V7 = 50 parsimonious features derived from a
larger 368-feature superset. The published `training_table_v7.parquet`
actually contains 90 features after dropping IDs/datetimes/target. To validate
the parsimony claim with the data we have, this script:

  1. Loads the seed-0 LightGBM model from results/v7_seed0_lightgbm.pkl
     (trained on the full 90 features by scripts/run_v7_ensemble.py).
  2. Ranks features by LightGBM gain importance and keeps the top 50.
  3. Re-trains all 4 GBM families on the 50-feature subset (N_SEEDS=10,
     same patient-grouped 60/20/20 split, subprocess per model).
  4. Saves predictions to results/v7_50feat_<model>_{val,test}.npz.
  5. Writes results/v7_50feat_summary.json with side-by-side 90-feat vs
     50-feat AUROCs, the selected feature list, and the LightGBM-importance
     rank that produced it.
  6. Produces figures/v7_parsimony_comparison.png.

Usage:
    python scripts/run_v7_50feature_analysis.py
    python scripts/run_v7_50feature_analysis.py --top-k 50
    python scripts/run_v7_50feature_analysis.py --top-k 30
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "figures"

TARGET = "readmit_30d"
ID_COLS = ["subject_id", "hadm_id"]
DT_COLS = ["admittime_dt", "dischtime_dt"]
RANDOM_STATE = 42

MODELS = ["lightgbm", "xgboost", "catboost", "histgbm"]


def _base_dir() -> Path:
    env = os.environ.get("MEDICARE_READMIT_BASE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    fallback = Path(r"C:/Users/Thiago/Documents/Coursework/Capstone Project")
    if (fallback / "Dataset" / "mimic-parquet").is_dir():
        return fallback
    return Path.cwd().resolve()


def _v7_path() -> Path:
    base = _base_dir()
    root_v7 = base / "training_table_v7.parquet"
    if root_v7.exists():
        return root_v7
    return base / "Dataset" / "mimic-parquet" / "training_table_v7.parquet"


def _load_split_arrays(top_features: list[str]):
    """Reload the canonical V7 split filtered to top_features only."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_parquet(_v7_path())
    feature_cols = json.loads((RESULTS / "v7_feature_cols.json").read_text())

    X = df[feature_cols].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Subset to top features
    X = X[top_features]

    split = np.load(RESULTS / "v7_split_indices.npz")
    tr_idx, va_idx, te_idx = split["train_idx"], split["val_idx"], split["test_idx"]

    X_train = X.iloc[tr_idx].reset_index(drop=True).values.astype(np.float32, copy=False)
    X_val   = X.iloc[va_idx].reset_index(drop=True).values.astype(np.float32, copy=False)
    X_test  = X.iloc[te_idx].reset_index(drop=True).values.astype(np.float32, copy=False)
    return X_train, split["y_train"], X_val, split["y_val"], X_test, split["y_test"]


def _train_lightgbm(Xtr, ytr, Xva, yva, Xte, n_seeds):
    import gc, numpy as np
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc",
             num_leaves=63, learning_rate=0.03, n_estimators=1500,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
             min_child_samples=50, reg_alpha=0.1, reg_lambda=1.0,
             verbose=-1, n_jobs=4)
    pv = np.zeros((len(Xva),  n_seeds), dtype=np.float32)
    pt = np.zeros((len(Xte),  n_seeds), dtype=np.float32)
    for s in range(n_seeds):
        m = lgb.LGBMClassifier(**{**p, "random_state": 42 + s})
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pv[:, s] = m.predict_proba(Xva)[:, 1]
        pt[:, s] = m.predict_proba(Xte)[:, 1]
        del m; gc.collect()
        print(f"  LightGBM seed {s}: done")
    return pv, pt


def _train_xgboost(Xtr, ytr, Xva, yva, Xte, n_seeds):
    import gc, numpy as np
    from xgboost import XGBClassifier
    pv = np.zeros((len(Xva),  n_seeds), dtype=np.float32)
    pt = np.zeros((len(Xte),  n_seeds), dtype=np.float32)
    for s in range(n_seeds):
        m = XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            max_depth=6, learning_rate=0.03, n_estimators=1500,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=50,
            tree_method="hist", verbosity=0, n_jobs=2,
            random_state=42 + s, early_stopping_rounds=100,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pv[:, s] = m.predict_proba(Xva)[:, 1]
        pt[:, s] = m.predict_proba(Xte)[:, 1]
        del m; gc.collect()
        print(f"  XGBoost seed {s}: done")
    return pv, pt


def _train_catboost(Xtr, ytr, Xva, yva, Xte, n_seeds):
    import gc, numpy as np
    from catboost import CatBoostClassifier
    pv = np.zeros((len(Xva),  n_seeds), dtype=np.float32)
    pt = np.zeros((len(Xte),  n_seeds), dtype=np.float32)
    for s in range(n_seeds):
        m = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
            random_seed=42 + s, verbose=0, eval_metric="AUC",
            early_stopping_rounds=100, thread_count=4,
        )
        m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=0)
        pv[:, s] = m.predict_proba(Xva)[:, 1]
        pt[:, s] = m.predict_proba(Xte)[:, 1]
        del m; gc.collect()
        print(f"  CatBoost seed {s}: done")
    return pv, pt


def _train_histgbm(Xtr, ytr, Xva, yva, Xte, n_seeds):
    import gc, numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    pv = np.zeros((len(Xva),  n_seeds), dtype=np.float32)
    pt = np.zeros((len(Xte),  n_seeds), dtype=np.float32)
    for s in range(n_seeds):
        m = HistGradientBoostingClassifier(
            max_iter=1500, learning_rate=0.03, max_leaf_nodes=63,
            min_samples_leaf=50, l2_regularization=1.0,
            random_state=42 + s, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=100, verbose=0,
        )
        m.fit(Xtr, ytr)
        pv[:, s] = m.predict_proba(Xva)[:, 1]
        pt[:, s] = m.predict_proba(Xte)[:, 1]
        del m; gc.collect()
        print(f"  HistGBM seed {s}: done")
    return pv, pt


TRAINERS = {
    "lightgbm": _train_lightgbm,
    "xgboost":  _train_xgboost,
    "catboost": _train_catboost,
    "histgbm":  _train_histgbm,
}


def _select_top_k(k: int) -> tuple[list[str], list[tuple[str, float]]]:
    """Return (selected_features, ranking_with_importances)."""
    import numpy as np
    feature_cols = json.loads((RESULTS / "v7_feature_cols.json").read_text())
    with open(RESULTS / "v7_seed0_lightgbm.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    importances = np.asarray(lgb_model.booster_.feature_importance(importance_type="gain"))
    ranking = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    selected = [name for name, _ in ranking[:k]]
    return selected, ranking


def _run_model_mode(model_name: str, n_seeds: int, k: int):
    print(f"=== subprocess: training {model_name} on top-{k} features ===")
    t0 = time.time()
    selected, _ = _select_top_k(k)
    Xtr, ytr, Xva, yva, Xte, yte = _load_split_arrays(selected)
    print(f"  prep done in {time.time()-t0:.1f}s  shapes train={Xtr.shape} val={Xva.shape} test={Xte.shape}")
    pv, pt = TRAINERS[model_name](Xtr, ytr, Xva, yva, Xte, n_seeds)
    import numpy as np
    np.savez_compressed(RESULTS / f"v7_50feat_{model_name}_val.npz",  preds=pv)
    np.savez_compressed(RESULTS / f"v7_50feat_{model_name}_test.npz", preds=pt)
    print(f"  wrote v7_50feat_{model_name}_{{val,test}}.npz")
    print(f"=== subprocess: {model_name} done in {time.time()-t0:.1f}s ===")
    return 0


def _build_comparison_figure(summary):
    """Bar chart comparing 90-feat vs 50-feat test AUROC per model."""
    import matplotlib.pyplot as plt
    import numpy as np

    PALETTE = {"teal": "#0F7B8A", "mint": "#00C9A7", "gray": "#888888"}
    FIGS.mkdir(parents=True, exist_ok=True)

    models = ["lightgbm", "xgboost", "catboost", "histgbm"]
    full = [summary["models_full"][m]["test_auroc"] for m in models]
    sub  = [summary["models_50"][m]["test_auroc"]   for m in models]

    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_full = ax.bar(x - width/2, full, width, color=PALETTE["gray"],
                       label=f"Full ({summary['n_features_full']} features)")
    bars_sub  = ax.bar(x + width/2, sub,  width, color=PALETTE["teal"],
                       label=f"Top-{summary['top_k']} ({summary['top_k']} features by LightGBM gain)")

    for bars in (bars_full, bars_sub):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0015,
                    f"{bar.get_height():.4f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(min(full + sub) - 0.005, max(full + sub) + 0.010)
    ax.set_title(
        f"V7 Parsimony Analysis — Test AUROC at full vs top-{summary['top_k']} feature set",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGS / "v7_parsimony_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def _orchestrate(k: int, n_seeds: int):
    print(f"V7 50-feature parsimony analysis: top-{k}, N_SEEDS={n_seeds}")

    selected, ranking = _select_top_k(k)
    feature_cols = json.loads((RESULTS / "v7_feature_cols.json").read_text())
    print(f"\nTop-{k} features by LightGBM gain importance (selected from {len(feature_cols)} total):")
    for i, (name, imp) in enumerate(ranking[:k], 1):
        print(f"  {i:2d}. {name:32s}  gain={imp:.0f}")

    # Train each model in its own subprocess
    for m in MODELS:
        print(f"\n[{m}] launching subprocess ...")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)),
             "--run-model", m, "--seeds", str(n_seeds), "--top-k", str(k)],
            capture_output=False, text=True,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"[{m}] FAILED with exit {proc.returncode} after {elapsed:.1f}s")
            return 1
        print(f"[{m}] done in {elapsed:.1f}s")

    # Aggregate AUROCs (vs full-feature baseline)
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score
    split = np.load(RESULTS / "v7_split_indices.npz")
    y_val, y_test = split["y_val"], split["y_test"]

    full_summary = json.loads((RESULTS / "v7_summary.json").read_text())

    summary = {
        "top_k": k,
        "n_features_full": len(feature_cols),
        "selected_features": selected,
        "ranking_first_25": [(name, float(imp)) for name, imp in ranking[:25]],
        "models_full": full_summary["models"],
        "models_50":   {},
    }
    preds_val_50, preds_test_50 = {}, {}
    for m in MODELS:
        pv = np.load(RESULTS / f"v7_50feat_{m}_val.npz")["preds"].mean(axis=1)
        pt = np.load(RESULTS / f"v7_50feat_{m}_test.npz")["preds"].mean(axis=1)
        preds_val_50[m], preds_test_50[m] = pv, pt
        summary["models_50"][m] = {
            "val_auroc":  float(roc_auc_score(y_val, pv)),
            "test_auroc": float(roc_auc_score(y_test, pt)),
        }

    # Blend on val
    order = MODELS
    val_stack  = np.stack([preds_val_50[m]  for m in order], axis=1)
    test_stack = np.stack([preds_test_50[m] for m in order], axis=1)

    def neg_auc(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return 0.0
        w = w / w.sum()
        return -roc_auc_score(y_val, val_stack @ w)

    res = minimize(neg_auc, x0=np.full(4, 0.25),
                   method="Nelder-Mead", bounds=[(0.05, 0.5)] * 4)
    w_opt = res.x / res.x.sum()
    summary["blend_50"] = {
        "weights": {m: float(w) for m, w in zip(order, w_opt)},
        "val_auroc":  float(roc_auc_score(y_val,  val_stack @ w_opt)),
        "test_auroc": float(roc_auc_score(y_test, test_stack @ w_opt)),
    }

    out = RESULTS / "v7_50feat_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")

    # Side-by-side
    print()
    print("=" * 78)
    print(f"V7 PARSIMONY ANALYSIS — {len(feature_cols)} features (full) vs {k} features (top by LGB gain)")
    print("=" * 78)
    print(f"{'Model':<12} {'Full val':>10} {'Full test':>10} {'Top-50 val':>12} {'Top-50 test':>13} {'Δ test':>9}")
    for m in MODELS:
        full = summary["models_full"][m]
        sub  = summary["models_50"][m]
        print(f"{m:<12} {full['val_auroc']:>10.4f} {full['test_auroc']:>10.4f} "
              f"{sub['val_auroc']:>12.4f} {sub['test_auroc']:>13.4f} "
              f"{sub['test_auroc'] - full['test_auroc']:>+9.4f}")
    full_blend = full_summary.get("blend", {})
    sub_blend  = summary["blend_50"]
    if full_blend:
        print(f"{'blend':<12} {full_blend.get('val_auroc', 0):>10.4f} {full_blend.get('test_auroc', 0):>10.4f} "
              f"{sub_blend['val_auroc']:>12.4f} {sub_blend['test_auroc']:>13.4f} "
              f"{sub_blend['test_auroc'] - full_blend.get('test_auroc', 0):>+9.4f}")

    _build_comparison_figure(summary)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--run-model", default=None, choices=MODELS)
    args = ap.parse_args()
    if args.run_model:
        sys.exit(_run_model_mode(args.run_model, args.seeds, args.top_k))
    sys.exit(_orchestrate(args.top_k, args.seeds))


if __name__ == "__main__":
    main()

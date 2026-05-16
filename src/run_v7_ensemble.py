"""Train the V7 4-GBM ensemble (§10.1) in isolated subprocesses.

The notebook kernel reliably crashes when training LightGBM + XGBoost + CatBoost
+ HistGBM × 10 seeds in one Python session (Fatal Python error related to
XGBoost's eval_set callback and the GIL). This script trains each model family
in its own subprocess so the OS reclaims memory + closes XGBoost's thread pool
between families.

Outputs (in results/):
 v7_split_indices.npz — train/val/test row indices (for reproducibility)
 v7_feature_cols.json — column names used as features
 v7_predictions.npz — predictions per model, per seed, on val + test
 v7_seed0_lightgbm.pkl — seed-0 LightGBM model (for §12 SHAP)
 v7_seed0_xgboost.pkl — seed-0 XGBoost model (for §11.1 plots)
 v7_summary.json — AUROCs per model + optimised blend

Usage:
 python scripts/run_v7_ensemble.py # full N_SEEDS=10
 python scripts/run_v7_ensemble.py --seeds 3 # quick
 python scripts/run_v7_ensemble.py --models lgb xgb # subset

Subprocess mode (internal):
 python scripts/run_v7_ensemble.py --run-model lightgbm --seeds 10
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

TARGET = "readmit_30d"
ID_COLS = ["subject_id", "hadm_id"]
DT_COLS = ["admittime_dt", "dischtime_dt"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10 # inner-val carved from the 80% train (no test leakage)

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
 # 50-feature parquet (V7) lives at the project root, not in Dataset/mimic-parquet/.
 root_v7 = base / "training_table_v7.parquet"
 if root_v7.exists():
 return root_v7
 return base / "Dataset" / "mimic-parquet" / "training_table_v7.parquet"


def _prepare_data():
 """Load V7, do 80/20 + 10% inner-val split, return arrays + persist split."""
 import numpy as np
 import pandas as pd
 from sklearn.model_selection import GroupShuffleSplit
 from sklearn.preprocessing import LabelEncoder

 df = pd.read_parquet(_v7_path())
 drop_cols = set(ID_COLS + DT_COLS + ["insurance", TARGET])
 feature_cols = [c for c in df.columns if c not in drop_cols]
 X = df[feature_cols].copy()
 for col in X.select_dtypes(include=["object", "category"]).columns:
 X[col] = LabelEncoder().fit_transform(X[col].astype(str))
 y = df[TARGET].astype(int).values
 groups = df["subject_id"].values

 tv_splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
 tv_idx, te_idx = next(tv_splitter.split(X, y, groups=groups))
 g_tv = groups[tv_idx]
 val_splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
 tr_idx_local, va_idx_local = next(val_splitter.split(X.iloc[tv_idx], y[tv_idx], groups=g_tv))
 tr_idx = tv_idx[tr_idx_local]
 va_idx = tv_idx[va_idx_local]

 X_train = X.iloc[tr_idx].reset_index(drop=True).values.astype(np.float32, copy=False)
 X_val = X.iloc[va_idx].reset_index(drop=True).values.astype(np.float32, copy=False)
 X_test = X.iloc[te_idx].reset_index(drop=True).values.astype(np.float32, copy=False)
 y_train = y[tr_idx]
 y_val = y[va_idx]
 y_test = y[te_idx]

 RESULTS.mkdir(parents=True, exist_ok=True)
 np.savez(RESULTS / "v7_split_indices.npz",
 train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx,
 y_train=y_train, y_val=y_val, y_test=y_test)
 (RESULTS / "v7_feature_cols.json").write_text(json.dumps(feature_cols))

 return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def _train_lightgbm(X_train, y_train, X_val, y_val, X_test, n_seeds):
 import gc, numpy as np
 import lightgbm as lgb
 params = dict(
 objective="binary", metric="auc", boosting_type="gbdt",
 num_leaves=63, learning_rate=0.03,
 feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
 min_child_samples=50, reg_alpha=0.1, reg_lambda=1.0,
 n_estimators=1500, verbose=-1, n_jobs=4,
 )
 pv = np.zeros((len(X_val), n_seeds), dtype=np.float32)
 pt = np.zeros((len(X_test), n_seeds), dtype=np.float32)
 seed0_model = None
 for s in range(n_seeds):
 p = {**params, "random_state": 42 + s}
 m = lgb.LGBMClassifier(**p)
 m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
 callbacks=[lgb.early_stopping(100, verbose=False),
 lgb.log_evaluation(0)])
 pv[:, s] = m.predict_proba(X_val)[:, 1]
 pt[:, s] = m.predict_proba(X_test)[:, 1]
 if s == 0:
 seed0_model = m
 else:
 del m; gc.collect()
 print(f" LightGBM seed {s}: done")
 return pv, pt, seed0_model


def _train_xgboost(X_train, y_train, X_val, y_val, X_test, n_seeds):
 import gc, numpy as np
 from xgboost import XGBClassifier
 pv = np.zeros((len(X_val), n_seeds), dtype=np.float32)
 pt = np.zeros((len(X_test), n_seeds), dtype=np.float32)
 seed0_model = None
 for s in range(n_seeds):
 m = XGBClassifier(
 objective="binary:logistic", eval_metric="auc",
 max_depth=6, learning_rate=0.03, n_estimators=1500,
 subsample=0.8, colsample_bytree=0.7,
 reg_alpha=0.1, reg_lambda=1.0, min_child_weight=50,
 tree_method="hist", verbosity=0, n_jobs=2,
 random_state=42 + s, early_stopping_rounds=100,
 )
 m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
 pv[:, s] = m.predict_proba(X_val)[:, 1]
 pt[:, s] = m.predict_proba(X_test)[:, 1]
 if s == 0:
 seed0_model = m
 else:
 del m; gc.collect()
 print(f" XGBoost seed {s}: done")
 return pv, pt, seed0_model


def _train_catboost(X_train, y_train, X_val, y_val, X_test, n_seeds):
 import gc, numpy as np
 from catboost import CatBoostClassifier
 pv = np.zeros((len(X_val), n_seeds), dtype=np.float32)
 pt = np.zeros((len(X_test), n_seeds), dtype=np.float32)
 for s in range(n_seeds):
 m = CatBoostClassifier(
 iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
 random_seed=42 + s, verbose=0, eval_metric="AUC",
 early_stopping_rounds=100, thread_count=4,
 )
 m.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
 pv[:, s] = m.predict_proba(X_val)[:, 1]
 pt[:, s] = m.predict_proba(X_test)[:, 1]
 del m; gc.collect()
 print(f" CatBoost seed {s}: done")
 return pv, pt, None


def _train_histgbm(X_train, y_train, X_val, y_val, X_test, n_seeds):
 import gc, numpy as np
 from sklearn.ensemble import HistGradientBoostingClassifier
 pv = np.zeros((len(X_val), n_seeds), dtype=np.float32)
 pt = np.zeros((len(X_test), n_seeds), dtype=np.float32)
 for s in range(n_seeds):
 m = HistGradientBoostingClassifier(
 max_iter=1500, learning_rate=0.03, max_leaf_nodes=63,
 min_samples_leaf=50, l2_regularization=1.0,
 random_state=42 + s, early_stopping=True,
 validation_fraction=0.1, n_iter_no_change=100, verbose=0,
 )
 m.fit(X_train, y_train)
 pv[:, s] = m.predict_proba(X_val)[:, 1]
 pt[:, s] = m.predict_proba(X_test)[:, 1]
 del m; gc.collect()
 print(f" HistGBM seed {s}: done")
 return pv, pt, None


TRAINERS = {
 "lightgbm": _train_lightgbm,
 "xgboost": _train_xgboost,
 "catboost": _train_catboost,
 "histgbm": _train_histgbm,
}


def _run_model_mode(model_name: str, n_seeds: int):
 """Subprocess entry: train one model family, persist predictions + seed-0 model."""
 print(f"=== subprocess: training {model_name} with N_SEEDS={n_seeds} ===")
 t0 = time.time()
 Xtr, ytr, Xva, yva, Xte, yte, _ = _prepare_data()
 print(f" prep done in {time.time()-t0:.1f}s shapes train={Xtr.shape} val={Xva.shape} test={Xte.shape}")
 pv, pt, model = TRAINERS[model_name](Xtr, ytr, Xva, yva, Xte, n_seeds)
 out_pv = RESULTS / f"v7_{model_name}_val.npz"
 out_pt = RESULTS / f"v7_{model_name}_test.npz"
 import numpy as np
 np.savez_compressed(out_pv, preds=pv)
 np.savez_compressed(out_pt, preds=pt)
 print(f" wrote {out_pv.name}, {out_pt.name}")
 if model is not None:
 with open(RESULTS / f"v7_seed0_{model_name}.pkl", "wb") as f:
 pickle.dump(model, f)
 print(f" wrote v7_seed0_{model_name}.pkl")
 print(f"=== subprocess: {model_name} done in {time.time()-t0:.1f}s ===")
 return 0


def _orchestrate(models, n_seeds):
 print(f"Orchestrator: models={models}, N_SEEDS={n_seeds}")
 RESULTS.mkdir(parents=True, exist_ok=True)

 # Persist canonical split once (used by all subprocess workers).
 _ = _prepare_data()
 print("Canonical split + feature_cols persisted.")

 summary = {"models": {}, "n_seeds": n_seeds, "random_state": RANDOM_STATE,
 "test_size": TEST_SIZE, "val_size": VAL_SIZE}

 for m in models:
 print(f"\n[{m}] launching subprocess (N_SEEDS={n_seeds}) ...")
 t0 = time.time()
 proc = subprocess.run(
 [sys.executable, str(Path(__file__)),
 "--run-model", m, "--seeds", str(n_seeds)],
 capture_output=False, text=True,
 )
 elapsed = time.time() - t0
 if proc.returncode != 0:
 print(f"[{m}] ❌ subprocess exited {proc.returncode} after {elapsed:.1f}s")
 return 1
 print(f"[{m}] ✅ done in {elapsed:.1f}s")

 # Aggregate AUROCs + run blend optimisation.
 import numpy as np
 from scipy.optimize import minimize
 from sklearn.metrics import roc_auc_score
 split = np.load(RESULTS / "v7_split_indices.npz")
 y_val, y_test = split["y_val"], split["y_test"]

 preds_val, preds_test = {}, {}
 for m in models:
 preds_val[m] = np.load(RESULTS / f"v7_{m}_val.npz")["preds"].mean(axis=1)
 preds_test[m] = np.load(RESULTS / f"v7_{m}_test.npz")["preds"].mean(axis=1)
 summary["models"][m] = {
 "val_auroc": float(roc_auc_score(y_val, preds_val[m])),
 "test_auroc": float(roc_auc_score(y_test, preds_test[m])),
 }
 print(f" {m}: val={summary['models'][m]['val_auroc']:.4f} "
 f"test={summary['models'][m]['test_auroc']:.4f}")

 if set(models) >= set(MODELS):
 # Run full 4-way blend optimisation on val
 order = MODELS
 val_stack = np.stack([preds_val[m] for m in order], axis=1)
 test_stack = np.stack([preds_test[m] for m in order], axis=1)

 def neg_auc(w):
 w = np.clip(w, 0, None)
 if w.sum() == 0:
 return 0.0
 w = w / w.sum()
 return -roc_auc_score(y_val, val_stack @ w)

 res = minimize(neg_auc, x0=np.full(4, 0.25),
 method="Nelder-Mead", bounds=[(0.05, 0.5)] * 4)
 w_opt = res.x / res.x.sum()
 blend_val = float(roc_auc_score(y_val, val_stack @ w_opt))
 blend_test = float(roc_auc_score(y_test, test_stack @ w_opt))
 summary["blend"] = {
 "weights": {m: float(w) for m, w in zip(order, w_opt)},
 "val_auroc": blend_val,
 "test_auroc": blend_test,
 }
 print(f"\nBlend weights: " + " ".join(f"{m}={w:.3f}" for m, w in zip(order, w_opt)))
 print(f"Blend val={blend_val:.6f} test={blend_test:.6f}")
 print(f"XGBoost solo test={summary['models']['xgboost']['test_auroc']:.6f}")

 out = RESULTS / "v7_summary.json"
 out.write_text(json.dumps(summary, indent=2))
 print(f"\nWrote {out}")
 return 0


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--seeds", type=int, default=10)
 ap.add_argument("--models", nargs="+", default=MODELS,
 choices=MODELS)
 ap.add_argument("--run-model", default=None, choices=MODELS,
 help="internal subprocess mode")
 args = ap.parse_args()

 if args.run_model:
 sys.exit(_run_model_mode(args.run_model, args.seeds))
 sys.exit(_orchestrate(args.models, args.seeds))


if __name__ == "__main__":
 main()

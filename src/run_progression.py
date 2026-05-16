"""Run the V1→V7 model progression (§9.1–9.4) as a standalone script.

Why standalone? Running the progression inline in the notebook reliably exhausts
memory on a 244k-row dataset when training LightGBM + XGBoost + MLP across
five feature versions in one kernel session. This script runs in a fresh
subprocess per version (one Python process per version), writes results to
results/progression_*.json, and the notebook then loads those JSONs.

Outputs:
 results/progression_logreg.json
 results/progression_gbm.json

Usage:
 python scripts/run_progression.py # all available versions
 python scripts/run_progression.py --versions v1 v2 v3 v6 v7

Each version is trained in its own subprocess via --run-version, so memory is
reclaimed by process exit between versions.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

try:
 sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
 sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
DEFAULT_VERSIONS = ["v1", "v2", "v3", "v6", "v7"]

TARGET = "readmit_30d"
ID_COLS = ["subject_id", "hadm_id"]
DT_COLS = ["admittime_dt", "dischtime_dt"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10 # inner-val carved from the 80% train (no test leakage)

PROGRESSION_N_EST = 400
PROGRESSION_ES = 40


def _base_dir() -> Path:
 env = os.environ.get("MEDICARE_READMIT_BASE_DIR")
 if env:
 return Path(env).expanduser().resolve()
 fallback = Path(r"C:/Users/Thiago/Documents/Coursework/Capstone Project")
 if (fallback / "Dataset" / "mimic-parquet").is_dir():
 return fallback
 return Path.cwd().resolve()


def _paths(base: Path) -> dict[str, Path]:
 data = base / "Dataset" / "mimic-parquet"
 # 50-feature parquet (V7) lives at the project root, not in Dataset/mimic-parquet/.
 root_v7 = base / "training_table_v7.parquet"
 return {
 "v1": data / "training_table_v1.parquet",
 "v2": data / "training_table_v2.parquet",
 "v3": data / "training_table_v3.parquet",
 "v4": data / "training_table_v4.parquet",
 "v5": data / "training_table_v5.parquet",
 "v6": data / "training_table_v6_full.parquet",
 "v7": root_v7 if root_v7.exists() else data / "training_table_v7.parquet",
 }


def _split_patients(v7_path: Path):
 """Recreate the same train/val/test patient sets the notebook uses (§8.2)."""
 import pandas as pd
 from sklearn.model_selection import GroupShuffleSplit
 df = pd.read_parquet(v7_path)
 g = df["subject_id"].values
 tv_splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
 tv_idx, te_idx = next(tv_splitter.split(df, df[TARGET], groups=g))
 test_patients = set(g[te_idx])
 tv_g = g[tv_idx]
 val_splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
 tr_idx, va_idx = next(val_splitter.split(df.iloc[tv_idx], df.iloc[tv_idx][TARGET], groups=tv_g))
 train_patients = set(tv_g[tr_idx])
 val_patients = set(tv_g[va_idx])
 return {"train": train_patients, "val": val_patients, "test": test_patients}


def _prep_xy(df_v, patient_split):
 """Return X_train_inner, y_train_inner, X_val_inner, y_val_inner, X_test, y_test, n_feat."""
 import numpy as np
 import pandas as pd
 from sklearn.model_selection import GroupShuffleSplit
 from sklearn.preprocessing import LabelEncoder

 drop_cols = set(ID_COLS + DT_COLS + ["insurance", TARGET])
 feat = [c for c in df_v.columns if c not in drop_cols]
 X = df_v[feat].copy()
 for col in X.select_dtypes(include=["object", "category"]).columns:
 X[col] = LabelEncoder().fit_transform(X[col].astype(str))
 y = df_v[TARGET].astype(int).values
 g = df_v["subject_id"].values

 train_mask = np.fromiter((gi in patient_split["train"] for gi in g), dtype=bool, count=len(g))
 test_mask = np.fromiter((gi in patient_split["test"] for gi in g), dtype=bool, count=len(g))
 Xtr, ytr, g_tr = X[train_mask].reset_index(drop=True), y[train_mask], g[train_mask]
 Xte, yte = X[test_mask].reset_index(drop=True), y[test_mask]

 inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
 tr_idx, va_idx = next(inner.split(Xtr, ytr, groups=g_tr))
 return (
 Xtr.iloc[tr_idx].reset_index(drop=True), ytr[tr_idx],
 Xtr.iloc[va_idx].reset_index(drop=True), ytr[va_idx],
 Xte, yte,
 len(feat),
 )


def run_one_version(v_key: str, paths: dict[str, Path], patient_split):
 """Train LogReg + LGB + XGB + MLP on one version. Return dict of AUROCs."""
 import gc as _gc
 import numpy as np
 import pandas as pd
 from sklearn.metrics import roc_auc_score

 df = pd.read_parquet(paths[v_key])
 Xtr, ytr, Xva, yva, Xte, yte, n_feat = _prep_xy(df, patient_split)
 del df; _gc.collect()

 out = {"version": v_key.upper(), "n_features": n_feat}

 # --- LogReg
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 from sklearn.pipeline import Pipeline
 from sklearn.impute import SimpleImputer
 pipe_lr = Pipeline([
 ("impute", SimpleImputer(strategy="median")),
 ("scale", StandardScaler()),
 ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
 random_state=RANDOM_STATE, n_jobs=2)),
 ])
 pipe_lr.fit(Xtr, ytr)
 out["logreg"] = float(roc_auc_score(yte, pipe_lr.predict_proba(Xte)[:, 1]))
 del pipe_lr; _gc.collect()

 # --- LightGBM
 import lightgbm as lgb
 m_lgb = lgb.LGBMClassifier(
 objective="binary", metric="auc",
 num_leaves=63, learning_rate=0.05, n_estimators=PROGRESSION_N_EST,
 feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
 min_child_samples=50, reg_alpha=0.1, reg_lambda=1.0,
 verbose=-1, n_jobs=4, random_state=RANDOM_STATE,
 )
 m_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)],
 callbacks=[lgb.early_stopping(PROGRESSION_ES, verbose=False),
 lgb.log_evaluation(0)])
 out["lightgbm"] = float(roc_auc_score(yte, m_lgb.predict_proba(Xte)[:, 1]))
 del m_lgb; _gc.collect()

 # --- XGBoost (reduced n_jobs, approx tree_method for memory)
 from xgboost import XGBClassifier
 m_xgb = XGBClassifier(
 objective="binary:logistic", eval_metric="auc",
 max_depth=6, learning_rate=0.05, n_estimators=PROGRESSION_N_EST,
 subsample=0.8, colsample_bytree=0.7,
 reg_alpha=0.1, reg_lambda=1.0, min_child_weight=50,
 tree_method="hist", verbosity=0, n_jobs=2,
 random_state=RANDOM_STATE, early_stopping_rounds=PROGRESSION_ES,
 )
 m_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
 out["xgboost"] = float(roc_auc_score(yte, m_xgb.predict_proba(Xte)[:, 1]))
 del m_xgb; _gc.collect()

 # --- MLP
 from sklearn.neural_network import MLPClassifier
 pipe_mlp = Pipeline([
 ("impute", SimpleImputer(strategy="median")),
 ("scale", StandardScaler()),
 ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=80,
 early_stopping=True, validation_fraction=0.10,
 random_state=RANDOM_STATE)),
 ])
 pipe_mlp.fit(Xtr, ytr)
 out["mlp"] = float(roc_auc_score(yte, pipe_mlp.predict_proba(Xte)[:, 1]))
 del pipe_mlp, Xtr, ytr, Xva, yva, Xte, yte; _gc.collect()

 return out


def _build_plots(results: list[dict]):
 """Build the two progression figures from collected results."""
 import matplotlib.pyplot as plt

 PALETTE = {"teal": "#0F7B8A", "gold": "#F4B942", "coral": "#E8636F",
 "mint": "#00C9A7", "blue": "#4C72B0", "orange": "#DD8452"}

 figs = REPO_ROOT / "figures"
 figs.mkdir(parents=True, exist_ok=True)

 # LogReg plot
 fig, ax = plt.subplots(figsize=(10, 4.5))
 versions = [r["version"] for r in results]
 lr_vals = [r["logreg"] for r in results]
 ax.plot(versions, lr_vals, "o-", color=PALETTE["blue"], lw=2.5, markersize=10)
 ax.axhline(0.70, color="gray", ls="--", lw=1)
 for v, s in zip(versions, lr_vals):
 ax.text(v, s + 0.002, f"{s:.4f}", ha="center", fontsize=10, fontweight="bold")
 ax.set_ylim(min(lr_vals) - 0.01, max(lr_vals) + 0.015)
 ax.set_ylabel("Test AUROC")
 ax.set_title("Logistic Regression — test AUROC across feature versions",
 fontsize=13, fontweight="bold")
 plt.tight_layout()
 plt.savefig(figs / "fig_b_logreg_v1v6.png", dpi=150, bbox_inches="tight")
 plt.close()

 # GBM/MLP plot
 fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
 for ax, name, color in zip(
 axes,
 ["lightgbm", "xgboost", "mlp"],
 [PALETTE["teal"], PALETTE["mint"], PALETTE["gold"]],
 ):
 vals = [r[name] for r in results]
 ax.plot(versions, vals, "o-", color=color, lw=2.5, markersize=10)
 for v, s in zip(versions, vals):
 ax.text(v, s + 0.002, f"{s:.3f}", ha="center", fontsize=9, fontweight="bold")
 ax.set_title(f"{name.upper()} — V_n progression", fontsize=13, fontweight="bold")
 ax.set_ylabel("Test AUROC")
 ax.set_ylim(0.65, 0.80)
 plt.suptitle("Per-family AUROC progression across feature-engineering versions",
 fontsize=14, fontweight="bold", y=1.02)
 plt.tight_layout()
 plt.savefig(figs / "fig_cde_lgbm_xgb_mlp_v1v6.png", dpi=150, bbox_inches="tight")
 plt.close()


def _orchestrate(versions: Iterable[str]):
 base = _base_dir()
 paths = _paths(base)
 print(f"BASE_DIR: {base}")

 # Build patient split once and persist for use by subprocess workers
 print("Computing patient split from V7...")
 patient_split = _split_patients(paths["v7"])
 split_cache = RESULTS / "_patient_split.json"
 RESULTS.mkdir(parents=True, exist_ok=True)
 split_cache.write_text(json.dumps({
 k: sorted(int(p) for p in s) for k, s in patient_split.items()
 }))
 print(f" cached patient split to {split_cache}")

 results = []
 for v in versions:
 path = paths.get(v)
 if path is None or not path.exists():
 print(f"\n[{v}] SKIP — parquet missing at {path}")
 continue
 print(f"\n[{v}] launching subprocess ...")
 t0 = time.time()
 proc = subprocess.run(
 [sys.executable, str(Path(__file__)), "--run-version", v],
 capture_output=True, text=True, encoding="utf-8",
 )
 elapsed = time.time() - t0
 if proc.returncode != 0:
 print(f" ❌ subprocess failed in {elapsed:.1f}s (exit {proc.returncode})")
 print(" STDERR:", (proc.stderr or "")[-2000:])
 continue
 # The subprocess prints a single JSON line on success.
 line = next((l for l in proc.stdout.splitlines()[::-1]
 if l.startswith("__RESULT__:")), None)
 if not line:
 print(f" ❌ subprocess returned no __RESULT__ line. stdout tail:")
 print((proc.stdout or "")[-2000:])
 continue
 rec = json.loads(line.removeprefix("__RESULT__:"))
 results.append(rec)
 print(f" ✅ {v} done in {elapsed:.1f}s LR={rec['logreg']:.4f} "
 f"LGB={rec['lightgbm']:.4f} XGB={rec['xgboost']:.4f} MLP={rec['mlp']:.4f}")

 if not results:
 print("\nNo results collected — nothing to write.")
 return 1

 # Persist
 payload = {
 "logreg": {r["version"]: {"n_features": r["n_features"],
 "test_auroc": r["logreg"]} for r in results},
 "lightgbm": {r["version"]: r["lightgbm"] for r in results},
 "xgboost": {r["version"]: r["xgboost"] for r in results},
 "mlp": {r["version"]: r["mlp"] for r in results},
 "_meta": {
 "n_estimators_progression": PROGRESSION_N_EST,
 "random_state": RANDOM_STATE,
 "test_size": TEST_SIZE,
 "val_size": VAL_SIZE,
 },
 }
 out_path = RESULTS / "progression.json"
 out_path.write_text(json.dumps(payload, indent=2))
 print(f"\nWrote {out_path}")

 _build_plots(results)
 print("Wrote progression figures to figures/")
 return 0


def _run_version_mode(v_key: str):
 """Subprocess entry point: trains on one version, prints JSON to stdout."""
 base = _base_dir()
 paths = _paths(base)
 # Reuse cached patient split if present, otherwise compute (slow).
 cache = RESULTS / "_patient_split.json"
 if cache.exists():
 ps_raw = json.loads(cache.read_text())
 patient_split = {k: set(v) for k, v in ps_raw.items()}
 else:
 patient_split = _split_patients(paths["v7"])
 rec = run_one_version(v_key, paths, patient_split)
 # Marker line so orchestrator can find result on stdout
 print(f"__RESULT__:{json.dumps(rec)}")
 return 0


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS)
 ap.add_argument("--run-version", default=None,
 help="internal: subprocess mode, trains on a single version and emits JSON")
 args = ap.parse_args()

 if args.run_version:
 sys.exit(_run_version_mode(args.run_version))
 sys.exit(_orchestrate(args.versions))


if __name__ == "__main__":
 main()

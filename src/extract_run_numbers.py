"""Extract AUROCs and weights from an executed notebook and write them to results/.

Usage:
    python scripts/extract_run_numbers.py
    python scripts/extract_run_numbers.py --notebook path/to/Capstone_Final_Notebook.ipynb

Outputs:
    results/run_numbers.json
    results/run_numbers.md   (markdown summary for paste-in)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NB = REPO_ROOT / "notebook" / "Capstone_Final_Notebook.ipynb"


def get_stdout(cell) -> str:
    parts = []
    for out in cell.get("outputs", []) or []:
        if out.get("output_type") == "stream":
            parts.append(out.get("text", ""))
    return "".join(parts)


def find_cells(nb, marker: str):
    out = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        src = "".join(c.source) if isinstance(c.source, list) else c.source
        if marker in src:
            out.append((i, c))
    return out


def extract(nb):
    res = {
        "split": {},
        "logreg_progression": {},
        "gbm_progression": {"LightGBM": {}, "XGBoost": {}, "MLP": {}},
        "v7": {},  # final V7 model results
        "blend": {},
    }

    # 1. Split sizes
    for _, c in find_cells(nb, "Patient-grouped 60/20/20"):
        text = get_stdout(c)
        for split in ["Train", "Val", "Test"]:
            m = re.search(rf"{split}:\s+([\d,]+) admissions \(([\d,]+) patients\)\s+\|\s+pos rate = ([\d.]+)", text)
            if m:
                res["split"][split.lower()] = {
                    "admissions": int(m.group(1).replace(",", "")),
                    "patients":  int(m.group(2).replace(",", "")),
                    "pos_rate":  float(m.group(3)),
                }
        m = re.search(r"Features:\s+(\d+)\s+\((\d+)\s+categorical\)", text)
        if m:
            res["split"]["n_features"] = int(m.group(1))
            res["split"]["n_categorical"] = int(m.group(2))

    # 2. LogReg progression
    for _, c in find_cells(nb, "Logistic regression V1 -> V6"):
        for m in re.finditer(r"(V\d+):\s+(\d+) features -> test AUROC = ([\d.]+)", get_stdout(c)):
            res["logreg_progression"][m.group(1)] = {
                "n_features": int(m.group(2)),
                "test_auroc": float(m.group(3)),
            }

    # 3. GBM progression
    for _, c in find_cells(nb, "9.2/9.3/9.4 Per-family progression"):
        text = get_stdout(c)
        for m in re.finditer(
            r"(V\d+)\s+\(\s*(\d+) feat\)\s+->\s+LGB\s+([\d.]+)\s+XGB\s+([\d.]+)\s+MLP\s+([\d.]+)",
            text
        ):
            v = m.group(1)
            res["gbm_progression"]["LightGBM"][v] = float(m.group(3))
            res["gbm_progression"]["XGBoost"][v] = float(m.group(4))
            res["gbm_progression"]["MLP"][v]      = float(m.group(5))

    # 4. V7 final model AUROCs
    patterns = {
        "LightGBM": r"LightGBM \d+-seed avg AUROC: val = ([\d.]+)\s+\|\s+test = ([\d.]+)",
        "XGBoost":  r"XGBoost \d+-seed avg AUROC: val = ([\d.]+)\s+\|\s+test = ([\d.]+)",
        "CatBoost": r"CatBoost \d+-seed avg AUROC: val = ([\d.]+)\s+\|\s+test = ([\d.]+)",
        "HistGBM":  r"HistGBM \d+-seed avg AUROC: val = ([\d.]+)\s+\|\s+test = ([\d.]+)",
    }
    for name, pat in patterns.items():
        for _, c in find_cells(nb, name):
            m = re.search(pat, get_stdout(c))
            if m:
                res["v7"][name] = {"val": float(m.group(1)), "test": float(m.group(2))}
                break

    # 5. Blend weights + AUROCs
    for _, c in find_cells(nb, "SCIPY-OPTIMIZED"):
        text = get_stdout(c)
        weights = {}
        for k in ["LightGBM", "XGBoost", "CatBoost", "HistGBM"]:
            m = re.search(rf"{k}\s+weight:\s+([\d.]+)", text)
            if m:
                weights[k] = float(m.group(1))
        res["blend"]["weights"] = weights
        for label, key in [
            (r"Blend AUROC on val.*: ([\d.]+)", "val"),
            (r"Blend AUROC on test.*: ([\d.]+)", "test"),
            (r"XGBoost solo \(test\):\s+([\d.]+)", "xgb_solo_test"),
        ]:
            m = re.search(label, text)
            if m:
                res["blend"][key] = float(m.group(1))

    return res


def to_markdown(res) -> str:
    lines = ["# Run numbers", ""]
    if res["split"]:
        s = res["split"]
        lines += [
            "## Split (60/20/20 patient-grouped)",
            f"- Train: {s['train']['admissions']:,} adm ({s['train']['patients']:,} pts, pos={s['train']['pos_rate']:.4f})",
            f"- Val:   {s['val']['admissions']:,} adm ({s['val']['patients']:,} pts, pos={s['val']['pos_rate']:.4f})",
            f"- Test:  {s['test']['admissions']:,} adm ({s['test']['patients']:,} pts, pos={s['test']['pos_rate']:.4f})",
            f"- Features: {s.get('n_features', '?')} ({s.get('n_categorical', '?')} categorical)",
            "",
        ]
    if res["logreg_progression"]:
        lines += ["## §9.1 LogReg progression", "| Version | N feat | Test AUROC |", "|---|---:|---:|"]
        for v, d in res["logreg_progression"].items():
            lines.append(f"| {v} | {d['n_features']} | {d['test_auroc']:.4f} |")
        lines += [""]
    if res["gbm_progression"]["LightGBM"]:
        versions = list(res["gbm_progression"]["LightGBM"].keys())
        lines += ["## §9.2-9.4 GBM/MLP progression",
                  "| Version | LightGBM | XGBoost | MLP |", "|---|---:|---:|---:|"]
        for v in versions:
            lgb = res["gbm_progression"]["LightGBM"].get(v, "?")
            xgb = res["gbm_progression"]["XGBoost"].get(v, "?")
            mlp = res["gbm_progression"]["MLP"].get(v, "?")
            lines.append(f"| {v} | {lgb:.4f} | {xgb:.4f} | {mlp:.4f} |")
        lines += [""]
    if res["v7"]:
        lines += ["## §10.1 V7 final-model results (multi-seed average)",
                  "| Model | Val AUROC | Test AUROC |", "|---|---:|---:|"]
        for k, d in res["v7"].items():
            lines.append(f"| {k} | {d['val']:.4f} | {d['test']:.4f} |")
        lines += [""]
    if res["blend"]:
        b = res["blend"]
        lines += ["## §10.2 Scipy-optimized blend",
                  "| Weight | Value |", "|---|---:|"]
        for k, v in (b.get("weights") or {}).items():
            lines.append(f"| {k} | {v:.3f} |")
        lines += [
            "",
            f"- Blend AUROC (val):  **{b.get('val', '?'):.4f}**",
            f"- Blend AUROC (test): **{b.get('test', '?'):.4f}**  ← single-shot eval",
            f"- XGBoost solo (test): {b.get('xgb_solo_test', '?'):.4f}",
        ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=str(DEFAULT_NB))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    nb = nbf.read(args.notebook, as_version=4)
    res = extract(nb)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run_numbers.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (out_dir / "run_numbers.md").write_text(md, encoding="utf-8")

    print(md)
    print()
    print(f"\nWrote {out_dir / 'run_numbers.json'} and {out_dir / 'run_numbers.md'}")


if __name__ == "__main__":
    main()

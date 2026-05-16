"""Execute the publication notebook end-to-end.

Modes:
    --smoke   : N_SEEDS=2, §9 progression on [v1, v7] only, fast sanity check.
    (default) : full run, N_SEEDS=10, progression on [v1..v6], save outputs in place.

Usage:
    python scripts/execute_notebook.py --smoke
    python scripts/execute_notebook.py
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebook" / "Capstone_Final_Notebook.ipynb"


def apply_smoke_overrides(nb):
    """In-memory overrides for fast sanity check."""
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        src = "".join(c.source) if isinstance(c.source, list) else c.source

        # 1) N_SEEDS=2 in §10.1
        if "N_SEEDS = 10" in src:
            src = src.replace("N_SEEDS = 10", "N_SEEDS = 2  # SMOKE")

        # 2) Limit progression loops to v1 + v7 (cuts §9 time ~3-6x)
        if 'for v_key in ["v1", "v2", "v3", "v4", "v5", "v6"]:' in src:
            src = src.replace(
                'for v_key in ["v1", "v2", "v3", "v4", "v5", "v6"]:',
                'for v_key in ["v1", "v7"]:  # SMOKE'
            )

        # 3) Skip cells that build big plots from progression (they reference V6)
        # — those plots still work with just v1,v7 keys; matplotlib won't crash.

        c.source = src


def cell_preview(cell, max_chars=120):
    src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
    return src.split("\n", 1)[0][:max_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="apply smoke-test overrides (N_SEEDS=2, fewer progression versions)")
    ap.add_argument("--output", default=None,
                    help="output notebook path (default: in-place; smoke writes to *_smoke.ipynb)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-cell timeout in seconds (default 3600)")
    args = ap.parse_args()

    os.environ.setdefault(
        "MEDICARE_READMIT_BASE_DIR",
        r"C:/Users/Thiago/Documents/Coursework/Capstone Project",
    )

    print(f"NB_PATH:      {NB_PATH}")
    print(f"BASE_DIR:     {os.environ.get('MEDICARE_READMIT_BASE_DIR')}")
    print(f"mode:         {'SMOKE' if args.smoke else 'FULL'}")
    print()

    nb = nbf.read(NB_PATH, as_version=4)

    if args.smoke:
        apply_smoke_overrides(nb)

    output_path = Path(args.output) if args.output else (
        NB_PATH.with_name("Capstone_Final_Notebook_smoke.ipynb") if args.smoke else NB_PATH
    )

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT / "notebook")}},
    )

    t0 = time.time()
    try:
        client.execute()
        elapsed = time.time() - t0
        print(f"\n✅ Notebook executed cleanly in {elapsed:.1f}s")
        status = 0
    except CellExecutionError as e:
        elapsed = time.time() - t0
        # Identify the failing cell
        failing_idx = None
        for i, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            for out in (c.get("outputs") or []):
                if out.get("output_type") == "error":
                    failing_idx = i
                    break
            if failing_idx is not None:
                break

        print(f"\n❌ Cell execution failed after {elapsed:.1f}s")
        if failing_idx is not None:
            print(f"   Failing cell: [{failing_idx:3d}] {cell_preview(nb.cells[failing_idx])}")
            err_outs = [o for o in nb.cells[failing_idx].get("outputs") or [] if o.get("output_type") == "error"]
            if err_outs:
                err = err_outs[0]
                print(f"   {err.get('ename', '?')}: {err.get('evalue', '?')}")
                tb = err.get("traceback") or []
                # strip ANSI
                import re
                ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
                for line in tb[-25:]:
                    print(f"     {ansi.sub('', line)}")
        status = 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
        status = 2
    finally:
        nbf.write(nb, output_path)
        print(f"\nSaved notebook to {output_path}")

    sys.exit(status)


if __name__ == "__main__":
    main()

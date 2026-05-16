"""Rewrite plt.savefig calls in the notebook to use FIG_DIR instead of bare filenames.

Affects EDA cells (§6.1-6.6) and any other cells that save figures with a literal
filename string. Cells that already use FIG_DIR (the cells we refactored in
refactor_notebook.py) are left alone.

Idempotent.

Usage:
    python scripts/fix_figure_paths.py
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebook" / "Capstone_Final_Notebook.ipynb"


def rewrite_savefig(src: str) -> tuple[str, int]:
    """Replace plt.savefig("foo.png", ...) with plt.savefig(FIG_DIR / "foo.png", ...).

    Skips calls that already use FIG_DIR or any Path-like.
    """
    n = 0

    def repl(match):
        nonlocal n
        prefix = match.group(1)
        path_arg = match.group(2)
        rest = match.group(3)
        # Skip if already wrapped
        if "FIG_DIR" in path_arg or "Path(" in path_arg or "/" in path_arg or "\\\\" in path_arg:
            return match.group(0)
        n += 1
        return f'{prefix}FIG_DIR / "{path_arg}"{rest}'

    pat = re.compile(r'(plt\.savefig\(\s*)"([^"]+\.png)"(\s*[,\)])')
    new_src = pat.sub(repl, src)
    return new_src, n


def main():
    nb = nbf.read(NB_PATH, as_version=4)
    total = 0
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        src = "".join(c.source) if isinstance(c.source, list) else c.source
        if "plt.savefig" not in src:
            continue
        new_src, n = rewrite_savefig(src)
        if n > 0:
            c.source = new_src
            print(f"  [{i:3d}] rewrote {n} savefig call(s)")
            total += n
    print(f"\nTotal savefig calls rewritten: {total}")
    nbf.write(nb, NB_PATH)


if __name__ == "__main__":
    main()

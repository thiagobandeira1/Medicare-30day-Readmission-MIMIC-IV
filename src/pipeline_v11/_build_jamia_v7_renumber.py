# -*- coding: utf-8 -*-
"""Renumber bracket citations by first appearance and drop uncited refs.
Validates that first appearances end strictly increasing."""
import re


def renumber(doc, refs):
    pat = re.compile(r"\[([0-9][0-9,\-]*)\]")

    def parse(group):
        out = []
        for part in group.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                if not (a.isdigit() and b.isdigit()):
                    return None
                out.extend(range(int(a), int(b) + 1))
            elif part.isdigit():
                out.append(int(part))
            else:
                return None
        if any(n < 1 or n > len(refs) for n in out):
            return None
        return out

    def paragraphs():
        for par in doc.paragraphs:
            yield par
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        yield par

    order, seen = [], set()
    for par in paragraphs():
        for mm in pat.finditer(par.text):
            nums = parse(mm.group(1))
            if nums is None:
                continue
            for n in nums:
                if n not in seen:
                    seen.add(n)
                    order.append(n)
    mapping = {old: new for new, old in enumerate(order, 1)}

    def rewrite(group):
        nums = parse(group)
        if nums is None:
            return f"[{group}]"
        new = sorted(mapping[n] for n in nums)
        parts, i = [], 0
        while i < len(new):
            j = i
            while j + 1 < len(new) and new[j + 1] == new[j] + 1:
                j += 1
            parts.append(str(new[i]) if j == i else
                         (f"{new[i]},{new[j]}" if j == i + 1 else
                          f"{new[i]}-{new[j]}"))
            i = j + 1
        return "[" + ",".join(parts) + "]"

    for par in paragraphs():
        if not pat.search(par.text):
            continue
        for run in par.runs:
            if pat.search(run.text):
                run.text = pat.sub(lambda mm: rewrite(mm.group(1)), run.text)
    seen2, mx = set(), 0
    for par in paragraphs():
        for mm in pat.finditer(par.text):
            nums = parse(mm.group(1)) if False else None
    # validate on the rewritten text against the new numbering
    new_refs = [refs[old - 1] for old in order]
    pat2 = re.compile(r"\[([0-9][0-9,\-]*)\]")
    for par in paragraphs():
        for mm in pat2.finditer(par.text):
            nums = []
            for part in mm.group(1).split(","):
                if "-" in part:
                    a, b = part.split("-", 1)
                    nums.extend(range(int(a), int(b) + 1))
                else:
                    nums.append(int(part))
            for n in nums:
                assert 1 <= n <= len(new_refs), f"dangling citation [{n}]"
                if n not in seen2:
                    seen2.add(n)
                    assert n >= mx, f"citation order broken at [{n}]"
                    mx = n
    return new_refs

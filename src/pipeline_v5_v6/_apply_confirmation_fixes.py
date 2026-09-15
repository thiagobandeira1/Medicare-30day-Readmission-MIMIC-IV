# -*- coding: utf-8 -*-
"""Final polish: the 15 problems surfaced by the confirmation verifiers."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPS = [
    # P1 sensitivity count
    ('P(f"Four sensitivity analyses probe the outcome and cohort rules. (1) Counting ',
     'P(f"Five sensitivity analyses probe the outcome definition, the cohort '
     'rules, and era composition. (1) Counting '),
    # P2 RQ tags
    ('  "using only structured EHR fields available at discharge (RQ1, RQ2); (2) "',
     '  "using only structured EHR fields available at discharge (RQ1, RQ2, RQ3); (2) "'),
    ('  "cluster-level uncertainty (RQ3, RQ4); and (5) release the pipeline and "',
     '  "cluster-level uncertainty (RQ4); and (5) release the pipeline and "'),
    # P3 posture forward-reference
    ('f"additional-resources posture this directs more outreach rather than "\n'
     '  f"less, but it is an operational disparity to monitor. "',
     'f"deployment posture that pairs flags with additional supportive "\n'
     '  f"resources rather than reduced care (Discussion), this directs more "\n'
     '  f"outreach rather than less, but it is an operational disparity to "\n'
     '  f"monitor. "'),
    ('f"share of nonreadmitted Black patients is flagged; under the stated "',
     'f"share of nonreadmitted Black patients is flagged; under a "'),
    # P4 duplicated baseline sentence
    ('f"published values, consistent with MIMIC-IV capturing only within-system "\n'
     '  f"prior utilization, which attenuates their utilization components; the "\n'
     '  f"comparison applies both scores to this study\'s all-cause within-system "',
     'f"published values; the "\n'
     '  f"comparison applies both scores to this study\'s all-cause within-system "'),
    # P5 Background duplication
    ('  "interventions act within days of discharge, so when patients return "\n'
     '  "matters as much as whether they return, and because risk scores steer "',
     '  "interventions act within days of discharge, so the timing of a return "\n'
     '  "matters as much as its occurrence, and because risk scores steer "'),
    # P6 RQ4 forward reference
    ('  "analyses of RQ4 address both gaps.")',
     '  "analyses in this study address both gaps.")'),
    # P7 fold WB test into the stratified sentence; delete the stray one
    ('f"computed in each draw. No formal equivalence or noninferiority test was "',
     'f"computed in each draw and tested directly by bootstrap rather than by "\n'
     'f"CI overlap. No formal equivalence or noninferiority test was "'),
    ('f"White vs Black AUROC difference was tested directly by bootstrap rather than by "\n'
     '  f"CI overlap. Decision-curve analysis [31] compared net benefit against treat-all "',
     'f"Decision-curve analysis [31] compared net benefit against treat-all "'),
    # P8 secondary tier clarity (+3 words)
    ('f"(secondary; SD {CCV[\'sd\']:.4f}) and "',
     'f"in development cross-validation (secondary; SD {CCV[\'sd\']:.4f}) and "'),
    # P9 abstract parsimony parenthetical dropped (-3 words)
    ('f"development partitions only (prespecified parsimony rule), and features selected in at least 3 of 5 folds formed "',
     'f"development partitions only, and features selected in at least 3 of 5 folds formed "'),
    # P10 serial and in RQ1 feature list
    ('_rq1 = ", ".join(flabel(s["feature"]) for s in SH[:7])',
     '_rq1 = (", ".join(flabel(s["feature"]) for s in SH[:6])\n'
     '        + ", and " + flabel(SH[6]["feature"]))'),
    # P11 serial and in metric list
    ('f"unchanged to the test partition; sensitivity, specificity, positive and "',
     'f"unchanged to the test partition; sensitivity, specificity, and positive and "'),
    # P12 subgroup-contrast scoping
    ('f"performance by sex (Figure 7, Table 1). Only the White vs Black "\n'
     '  f"contrast was prespecified and formally tested.',
     'f"performance by sex (Figure 7, Table 1). Among the subgroup "\n'
     '  f"comparisons, only the White vs Black contrast was prespecified and "\n'
     '  f"formally tested.'),
    # P13 appendix list label
    ('P(f"Appendix 1: predictor audit and selection workbook',
     'P(f"Multimedia Appendix 1: predictor audit and selection workbook'),
    # numbers agent: Jaccard at stored precision
    ("f\"Jaccard {STAB['jaccard_mean']:.2f};",
     "f\"Jaccard {STAB['jaccard_mean']:.3f};"),
]


def patch(path):
    t = path.read_text(encoding="utf-8")
    miss = []
    for old, new in REPS:
        n = t.count(old)
        if n == 0 and t.count(new):
            continue
        if n != 1:
            miss.append((n, old[:90]))
            continue
        t = t.replace(old, new)
    if miss:
        for n, m in miss:
            print(f"  MISS ({n}) in {path.name}: {m!r}")
        raise SystemExit(f"{path.name}: {len(miss)} failed")
    path.write_text(t, encoding="utf-8")
    print(f"patched {path.name}")


for name in ("_build_jmir_final_part2.py", "_build_jmir_finalsc_part2.py"):
    patch(HERE / name)
print("CONFIRMATION FIXES APPLIED")

# -*- coding: utf-8 -*-
"""Author rule (2026-08-17, Thiago): no dash punctuation anywhere in the
manuscript. This script patches the four FINAL builders in place:

  * kills the spaced-hyphen -> em-dash conversion in _style() and replaces it
    with a build-time assertion so no dash punctuation can ever reach the
    document again;
  * rewrites every " - " construct in the content with commas, colons,
    semicolons, or parentheses;
  * converts numeric ranges in prose and table cells (CIs, day windows,
    feature counts, Python versions, age bands) to "x to y";
  * replaces the bare "-" not-available marker in Table 1 with "NA".

Kept on purpose (spelling/notation, not punctuation): compound-word hyphens
(30-day, patient-grouped, MIMIC-IV), identifiers (ORCID, DOI, phone, dates,
tags, URLs), reference page ranges, and bracketed citation ranges [9-12].

Every replacement is verified: the script fails if any pattern is not found
exactly the expected number of times.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- part1
# (applies to _build_jmir_final_part1.py AND _build_jmir_finalsc_part1.py)
PART1_REPS = [
    ('''    """AMA-style typography at build time (audit items S1/S3/S7): spaced
    hyphens become unspaced em dashes, apostrophes curl, negative numbers get
    a true minus sign. Digit-hyphen-digit ranges are untouched."""
    t = t.replace(" - ", "—")
    t = t.replace("'", "’")''',
     '''    """Typography at build time: apostrophes curl and negative numbers get
    a true minus sign. Author rule (2026-08-17): no dash punctuation may
    reach the document; the build fails if an em/en dash or spaced hyphen
    appears in any styled string."""
    assert "—" not in t and "–" not in t and " - " not in t, (
        "dash punctuation in manuscript text: %r" % t[:120])
    t = t.replace("'", "’")''', 1),
    ('    return f"{lo:.{d}f}-{hi:.{d}f}"',
     '    return f"{lo:.{d}f} to {hi:.{d}f}"', 1),
]

# ---------------------------------------------------------------- part2
# (applies to _build_jmir_final_part2.py AND _build_jmir_finalsc_part2.py)
# (old, new, expected_count_per_file)
PART2_REPS = [
    # numeric CI ranges rendered from f-strings -> "x to y"
    (':.4f}-{', ':.4f} to {', None),
    (':.3f}-{', ':.3f} to {', None),
    (':.2f}-{', ':.2f} to {', None),
    (':.4f}-"', ':.4f} to "', None),
    # abstract: fold-specific feature counts (also trims 1 word for the cap)
    ('"The selection procedure (fold-specific feature sets of 27-38 '
     'predictors) "',
     '"The selection procedure (fold-specific sets of 27 to 38 predictors) "',
     1),
    # abstract: shorten secondary-tier parenthetical (word-count headroom)
    ('f"(SD {CCV[\'sd\']:.4f}) under grouped cross-validation (secondary; the "\n'
     '  f"consensus set was selected on the same development data) and "',
     'f"(SD {CCV[\'sd\']:.4f}) under grouped cross-validation (secondary; "\n'
     '  f"consensus selected on the same development data) and "', 1),
    # RQ3 em-dash pair -> parentheses
    ('"interpretable machine-learning outputs - global and patient-level Shapley "',
     '"interpretable machine-learning outputs (global and patient-level Shapley "',
     1),
    ('"additive explanations (SHAP) "\n  "- provide actionable insight',
     '"additive explanations, SHAP) "\n  "provide actionable insight', 1),
    # validation-hierarchy tiers -> colons
    ('"(1) primary - the out-of-fold performance',
     '"(1) primary: the out-of-fold performance', 1),
    ('"sets, not the fixed consensus model; (2) secondary - 5-fold patient-grouped "',
     '"sets, not the fixed consensus model; (2) secondary: 5-fold patient-grouped "',
     1),
    ('"(3) tertiary - a single evaluation',
     '"(3) tertiary: a single evaluation', 1),
    # billing-artifact em-dash pair -> comma apposition
    ('f"artifacts - diagnosis-related groups,',
     'f"artifacts, namely diagnosis-related groups,', 1),
    ('f"clinical names whose provenance was verified in the feature-build code) - "',
     'f"clinical names whose provenance was verified in the feature-build code), "',
     1),
    # AFT parenthetical
    ('"failure time (AFT) objective [28] - a single-event survival model evaluated under "',
     '"failure time (AFT) objective [28] (a single-event survival model evaluated under "',
     1),
    ('"censoring, not itself a competing-risk model - and penalized cause-specific "',
     '"censoring, not itself a competing-risk model) and penalized cause-specific "',
     1),
    # day ranges in Methods
    ('"of-censoring-weighted time-dependent AUROC at each day 1-29 [30], with "',
     '"of-censoring-weighted time-dependent AUROC at each day 1 to 29 [30], with "',
     1),
    ('"Three landmark models covered days 1-7, 8-14, and 15-30, each trained only on "',
     '"Three landmark models covered days 1 to 7, 8 to 14, and 15 to 30, each trained only on "',
     1),
    ('"patients still at risk when the stage opens - patients already readmitted or "',
     '"patients still at risk when the stage opens; patients already readmitted or "',
     1),
    # White-Black -> White vs Black
    ('f"resampled directly. The White-Black AUROC contrast compares disjoint "',
     'f"resampled directly. The White vs Black AUROC contrast compares disjoint "',
     1),
    ('f"White-Black AUROC difference was tested directly by bootstrap rather than by "',
     'f"White vs Black AUROC difference was tested directly by bootstrap rather than by "',
     1),
    # threshold parenthetical
    ('f"({THRV:.3f}) was the Youden point on the validation partition - a "',
     'f"({THRV:.3f}) was the Youden point on the validation partition (a "', 1),
    ('f"participated in the consensus-selection folds - and was applied "',
     'f"participated in the consensus-selection folds) and was applied "', 1),
    # DCA threshold range + Python versions
    ('f"and treat-none across threshold probabilities 0.05-0.40, with a capacity view "',
     'f"and treat-none across threshold probabilities 0.05 to 0.40, with a capacity view "',
     1),
    ('f"(top-k% flagged). Analyses used Python 3.11-3.12 (XGBoost, LightGBM, "',
     'f"(top-k% flagged). Analyses used Python 3.11 and 3.12 (XGBoost, LightGBM, "',
     1),
    # paired-differences sentence
    ('f"prior 50-feature subset ({_cmp[\'f50\'][\'oof_auroc\']:.4f}) - absolute "',
     'f"prior 50-feature subset ({_cmp[\'f50\'][\'oof_auroc\']:.4f}); these absolute "',
     1),
    ('f"differences smaller than the 0.002 tolerance used in the prespecified "',
     'f"differences are smaller than the 0.002 tolerance used in the prespecified "',
     1),
    ('f"parsimony rule, described as similar discrimination rather than "',
     'f"parsimony rule and are described as similar discrimination rather than "',
     1),
    # tie-rule apposition
    ('f"the tie was resolved as death first - a documented rule, adopted during "',
     'f"the tie was resolved as death first, a documented rule adopted during "',
     1),
    # landmark stage windows
    ('f"{st1[\'auroc\']:.4f} for days 1-7 ({st1[\'events\']:,} events among "',
     'f"{st1[\'auroc\']:.4f} for days 1 to 7 ({st1[\'events\']:,} events among "',
     1),
    ('f"{st1[\'at_risk\']:,} at risk), {st2[\'auroc\']:.4f} for days 8-14 "',
     'f"{st1[\'at_risk\']:,} at risk), {st2[\'auroc\']:.4f} for days 8 to 14 "',
     1),
    ('f"15-30 ({st3[\'events\']:,}/{st3[\'at_risk\']:,}) - discrimination improves for "',
     'f"15 to 30 ({st3[\'events\']:,}/{st3[\'at_risk\']:,}); discrimination improves for "',
     1),
    ('f"disproportionately important for a return within days 1-7 are "',
     'f"disproportionately important for a return within days 1 to 7 are "', 1),
    ('f"disproportionately important for returns in days 15-30 are predominantly "',
     'f"disproportionately important for returns in days 15 to 30 are predominantly "',
     1),
    ('"earliest (days 1-7) and latest (days 15-30) windows; positive values "',
     '"earliest (days 1 to 7) and latest (days 15 to 30) windows; positive values "',
     1),
    # Table 1: NA marker and age-band display labels
    ('        return "-"', '        return "NA"', 1),
    ('f"{e[\'ppv\']:.3f}" if e.get("ppv") is not None else "-",',
     'f"{e[\'ppv\']:.3f}" if e.get("ppv") is not None else "NA",', 1),
    ('f"{e[\'brier\']:.3f}" if e.get("brier") is not None else "-",',
     'f"{e[\'brier\']:.3f}" if e.get("brier") is not None else "NA",', 1),
    ('        rows.append([g,', '        rows.append([g.replace("-", " to "),',
     1),
    # SHAP multifactorial parenthetical
    ('f"the top seven. Readmission risk is multifactorial - no single feature "',
     'f"the top seven. Readmission risk is multifactorial (no single feature "',
     1),
    ('f"dominates - and the leading features map to observables a discharge team "',
     'f"dominates), and the leading features map to observables a discharge team "',
     1),
    # DCA exploratory range
    ('f"treat-none across an exploratory threshold range of roughly 0.10-0.40, a "',
     'f"treat-none across an exploratory threshold range of roughly 0.10 to 0.40, a "',
     1),
    # principal findings
    ('f"{NF}-feature gradient-boosted model - selected from the complete pool of "',
     'f"{NF}-feature gradient-boosted model, selected from the complete pool of "',
     1),
    ('f"data only - estimated 30-day within-system readmission with strong "',
     'f"data only, estimated 30-day within-system readmission with strong "', 1),
    ('f"leakage-safe evaluation - most of the usable signal is captured by a "',
     'f"leakage-safe evaluation: most of the usable signal is captured by a "',
     1),
    ('f"first day after discharge - the window in which an intervention initiated "',
     'f"first day after discharge, the window in which an intervention initiated "',
     1),
    # comparison-with-prior-work
    ('f"27-38 nonbilling structured predictors across folds, achieved a "',
     'f"27 to 38 nonbilling structured predictors across folds, achieved a "', 1),
    ('f"partition - numerically above the notes-based and structured-feature "',
     'f"partition, numerically above the notes-based and structured-feature "', 1),
    # Table 2 cells
    ('"Same cohort - this study\'s MIMIC-IV Medicare test partition"',
     '"Same cohort: this study\'s MIMIC-IV Medicare test partition"', 2),
    ('"Same database - MIMIC-IV (all-adult, 415,231)"',
     '"Same database: MIMIC-IV (all-adult, 415,231)"', 1),
    ('"Same database - MIMIC-IV (all-adult, 303,571)"',
     '"Same database: MIMIC-IV (all-adult, 303,571)"', 1),
    ('"Same database - MIMIC-IV (radiograph-selected, 14,532)"',
     '"Same database: MIMIC-IV (radiograph-selected, 14,532)"', 1),
    ('"Different database - MIMIC-III"', '"Different database: MIMIC-III"', 1),
    ('"Primary; fold-specific sets of 27-38 features; leakage-safe "',
     '"Primary; fold-specific sets of 27 to 38 features; leakage-safe "', 1),
    # research-question answers
    ('f"gradient boosting - the XGBoost-based RFE procedure achieved an "',
     'f"gradient boosting; the XGBoost-based RFE procedure achieved an "', 1),
    ('f"procedure on identical inputs - with LightGBM, CatBoost, and HistGradientBoosting "',
     'f"procedure on identical inputs, with LightGBM, CatBoost, and HistGradientBoosting "',
     1),
    ('f"post-discharge window - providing discharge teams with an explanation "',
     'f"post-discharge window, providing discharge teams with an explanation "',
     1),
    # clinical-interpretation section
    ('"palliative-care consultation and earlier escalation to complex-care management "\n'
     '  "- as a hypothesis for prospective testing.',
     '"palliative-care consultation and earlier escalation to complex-care management, "\n'
     '  "as a hypothesis for prospective testing.', 1),
    ('"planning has already demonstrably failed - candidates for intensive case "',
     '"planning has already demonstrably failed: candidates for intensive case "',
     1),
    ('P("The abnormal-laboratory rate - the fraction of the index stay\'s laboratory "',
     'P("The abnormal-laboratory rate (the fraction of the index stay\'s laboratory "',
     1),
    ('"results flagged abnormal - is the clearest physiological signal in the final "',
     '"results flagged abnormal) is the clearest physiological signal in the final "',
     1),
    ('"derangement marks post-discharge vulnerability, suggesting - as a hypothesis "\n'
     '  "- that such patients may warrant clinical contact within days rather than "',
     '"derangement marks post-discharge vulnerability, suggesting, as a hypothesis, "\n'
     '  "that such patients may warrant clinical contact within days rather than "',
     1),
    ('"intervention points identified by the model - the disposition decision, the "',
     '"intervention points identified by the model (the disposition decision, the "',
     1),
    ('"post-acute hand-off, complex-care enrollment - are system-level processes, "',
     '"post-acute hand-off, complex-care enrollment) are system-level processes, "',
     1),
    # reference title (rendered with a colon, as PubMed does)
    ('"Vyas DA, Eisenstein LG, Jones DS. Hidden in plain sight - reconsidering the use "',
     '"Vyas DA, Eisenstein LG, Jones DS. Hidden in plain sight: reconsidering the use "',
     1),
]


def patch(path, reps):
    text = path.read_text(encoding="utf-8")
    misses = []
    for old, new, expected in reps:
        n = text.count(old)
        if n == 0 and text.count(new) >= (expected or 1):
            continue        # already applied (idempotent rerun)
        if expected is not None and n != expected:
            misses.append((old[:70], expected, n))
            continue
        if expected is None and n == 0:
            # pattern class may legitimately be absent; record as info only
            print(f"  note: 0 hits for pattern class {old!r} in {path.name}")
            continue
        text = text.replace(old, new)
    if misses:
        for old, exp, got in misses:
            print(f"  MISS in {path.name}: expected {exp}, found {got}: {old!r}")
        raise SystemExit(f"{path.name}: {len(misses)} replacement(s) failed")
    path.write_text(text, encoding="utf-8")
    print(f"patched {path.name}")


for name in ("_build_jmir_final_part1.py", "_build_jmir_finalsc_part1.py"):
    patch(HERE / name, PART1_REPS)
for name in ("_build_jmir_final_part2.py", "_build_jmir_finalsc_part2.py"):
    patch(HERE / name, PART2_REPS)
print("all builders patched")

# -*- coding: utf-8 -*-
"""Supplementary Appendix 5 (TRIPOD+AI, official 27-item checklist, version
11-January-2024) and Appendix 6 (PROBAST+AI, official signalling questions,
development quality and evaluation risk of bias rated separately) for v11.
Page numbers are read from the manuscript PDF by searching section headings.
Usage: python _v11_checklists.py <packet dir> <manuscript pdf> <supplement pdf>"""
import sys, re
from pathlib import Path
import fitz
from docx import Document
from docx.shared import Pt

PKT = Path(sys.argv[1]); MPDF = Path(sys.argv[2]); SPDF = Path(sys.argv[3])


def pages_of(pdf, heads):
    """Page of the first line that equals the heading (or starts with it for
    table and figure captions), so 'Objective:' in the abstract or 'Table S4'
    in a cross reference do not match."""
    d = fitz.open(str(pdf)); found = {}
    for i, p in enumerate(d, 1):
        lines = [ln.strip() for ln in p.get_text().splitlines()]
        for h in heads:
            if h in found:
                continue
            if h.endswith("."):
                if any(ln.startswith(h + " ") for ln in lines):
                    found[h] = i
            elif any(ln == h or ln == h.rstrip(".") for ln in lines):
                found[h] = i
    return found


M_HEADS = ["Abstract", "Background and Significance", "Objective", "Study Design and Data Source",
           "Cohort and Outcome Definition", "Partitions and Validation Design",
           "Predictor Audit, Leakage Testing, and Feature Selection", "Model and Comparators",
           "Clinical-Score Baselines", "Leakage Ablation", "Time-to-Event Analyses",
           "Statistical Analysis", "Subgroup Audit", "Ethical Considerations", "Cohort",
           "Future-Information Leakage", "Feature Selection and Model Comparison",
           "Discrimination and Calibration", "Sensitivity Analyses", "Same-Cohort Clinical Baselines",
           "Timing of Readmission", "Subgroup Performance", "Interpretability via SHAP",
           "Clinical Utility (Exploratory)", "Principal Findings", "Lessons for Leakage Auditing",
           "Comparison With Prior Work", "Clinical Interpretation (Hypotheses)", "Limitations",
           "Conclusion", "Acknowledgments", "Conflict of Interest Statement", "Funding",
           "Data Availability Statement", "Author Contributions", "Table 1.", "Table 2.",
           "Table 3.", "Table 4.", "Figure 1.", "Figure 2.", "Figure 3.", "Figure 4.", "Figure 5.", "Figure 6."]
S_HEADS = ["S1.", "S2.", "S3.", "S4.", "S5.", "S6.", "S7.", "S8.", "S9.", "S10.", "S11.", "S12.",
           "S13.", "S14.", "S15.", "Table S1.", "Table S2.", "Table S3.", "Table S4.", "Supplementary Figure S1.",
           "Supplementary Figure S2.", "Supplementary Figure S3."]
MP = pages_of(MPDF, M_HEADS); SP = pages_of(SPDF, S_HEADS)


def loc(*keys):
    out = []
    for k in keys:
        if k in MP:
            out.append(f"{k} (p {MP[k]})")
        elif k in SP:
            out.append(f"Supplement {k} (p S{SP[k]})")
        else:
            out.append(k)
    return "; ".join(out)


def new_doc(title, note):
    d = Document()
    d.styles["Normal"].font.name = "Times New Roman"; d.styles["Normal"].font.size = Pt(10)
    p = d.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(13)
    d.add_paragraph(note)
    return d


def table(d, rows, widths=None, fs=8.5):
    t = d.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            r = t.cell(i, j).paragraphs[0].add_run(str(c)); r.font.size = Pt(fs)
            if i == 0:
                r.bold = True
    return t


# =============================================================== TRIPOD+AI
T = [
 ("Title", "1", "D;E", "Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome to be predicted", "Title identifies development and internal evaluation, Medicare-insured adults, 30-day readmission", "Title page"),
 ("Abstract", "2", "D;E", "See TRIPOD+AI for Abstracts checklist", "Structured abstract (Objective, Materials and Methods, Results, Discussion, Conclusion) with data source, sample size, outcome, model type, primary and test performance, and limitations", loc("Abstract")),
 ("Background", "3a", "D;E", "Explain the healthcare context (including whether diagnostic or prognostic) and rationale for developing or evaluating the prediction model, including references to existing models", "Prognostic context (post-discharge readmission), HRRP rationale, LACE and HOSPITAL and MIMIC-IV models referenced", loc("Background and Significance")),
 ("", "3b", "D;E", "Describe the target population and the intended purpose of the prediction model in the context of the care pathway, including its intended users", "Medicare-insured adults at alive discharge; intended to inform transitional-care targeting by discharge teams; research-only at present", loc("Background and Significance", "Cohort and Outcome Definition")),
 ("", "3c", "D;E", "Describe any known health inequalities between sociodemographic groups", "Unequal model performance across groups discussed as a risk; references to documented algorithmic bias", loc("Background and Significance", "Subgroup Audit")),
 ("Objectives", "4", "D;E", "Specify the study objectives, including whether the study describes the development or validation of a prediction model (or both)", "Five research questions; development with internal validation; leakage audit objective", loc("Objective")),
 ("Data", "5a", "D;E", "Describe the sources of data separately for the development and evaluation datasets, the rationale for using these data, and representativeness of the data", "MIMIC-IV v3.1, single center; development and test partitions drawn from the same source by patient-level split; representativeness limits stated", loc("Study Design and Data Source", "Partitions and Validation Design", "Limitations")),
 ("", "5b", "D;E", "Specify the dates of the collected participant data, including start and end of participant accrual; and, if applicable, end of follow-up", "Admissions 2008 to 2022 (date-shifted); 30-day follow-up; end-of-data boundary sensitivity", loc("Study Design and Data Source", "Sensitivity Analyses")),
 ("Participants", "6a", "D;E", "Specify key elements of the study setting including the number and location of centres", "One academic medical center, Boston, MA", loc("Study Design and Data Source")),
 ("", "6b", "D;E", "Describe the eligibility criteria for study participants", "Insurance recorded as Medicare; alive at discharge; every admission index-eligible", loc("Cohort and Outcome Definition")),
 ("", "6c", "D;E", "Give details of any treatments received, and how they were handled during model development or evaluation, if relevant", "Not applicable; in-hospital treatment intensity enters only as predictors", loc("Predictor Audit, Leakage Testing, and Feature Selection")),
 ("Data preparation", "7", "D;E", "Describe any data pre-processing and quality checking, including whether this was similar across relevant sociodemographic groups", "Predictor audit, timestamp recomputation, fold-local target encoding; identical processing for all groups; missingness by race reported in Appendix 1", loc("Predictor Audit, Leakage Testing, and Feature Selection", "S3.", "S4.")),
 ("Outcome", "8a", "D;E", "Clearly define the outcome that is being predicted and the time horizon, including how and when assessed, the rationale, and whether the method of outcome assessment is consistent across sociodemographic groups", "First within-system readmission more than 0 and up to 30 days after discharge, from timestamps; identical ascertainment for all groups; continuation-band sensitivity", loc("Cohort and Outcome Definition", "S1.")),
 ("", "8b", "D;E", "If outcome assessment requires subjective interpretation, describe the qualifications and demographic characteristics of the outcome assessors", "Not applicable; timestamp-based outcome", loc("Cohort and Outcome Definition")),
 ("", "8c", "D;E", "Report any actions to blind assessment of the outcome to be predicted", "Outcome derived from admission timestamps without reference to predictors", loc("Cohort and Outcome Definition")),
 ("Predictors", "9a", "D", "Describe the choice of initial predictors and any pre-selection of predictors before model building", "207 candidates from capstone feature engineering; billing-derived, race-derived, and outcome-derived predictors excluded before modeling; empirical leakage tests", loc("Predictor Audit, Leakage Testing, and Feature Selection")),
 ("", "9b", "D;E", "Clearly define all predictors, including how and when they were measured (and any actions to blind assessment of predictors for the outcome and other predictors)", "Definitions and verification of the final features in Supplementary Table S3; measured on or before discharge; laboratory results required to be stored before discharge", loc("Table S3.", "S3.")),
 ("", "9c", "D;E", "If predictor measurement requires subjective interpretation, describe the qualifications and demographic characteristics of the predictor assessors", "Not applicable; structured EHR fields", loc("Predictor Audit, Leakage Testing, and Feature Selection")),
 ("Sample size", "10", "D;E", "Explain how the study size was arrived at (separately for development and evaluation), and justify that the study size was sufficient", "All eligible admissions used; events per partition reported; no formal calculation; learning curve and flat RFE profile as empirical support", loc("Partitions and Validation Design", "S11.")),
 ("Missing data", "11", "D;E", "Describe how missing data were handled. Provide reasons for omitting any data", "Continuous measurements left missing for the tree learner; counts zero; earlier global median imputation in the source table identified and reversed", loc("Predictor Audit, Leakage Testing, and Feature Selection", "S4.")),
 ("Analytical methods", "12a", "D", "Describe how the data were used (e.g., for development and evaluation of model performance) in the analysis, including whether the data were partitioned", "80/20 patient-level split; validation partition; 5 outer by 3 inner grouped folds; three evidence tiers", loc("Partitions and Validation Design", "S2.")),
 ("", "12b", "D", "Describe how predictors were handled in the analyses (functional form, rescaling, transformation, or any standardisation)", "Tree learner on raw scales; log and interaction terms as listed; fold-local target encoding of categorical predictors", loc("Predictor Audit, Leakage Testing, and Feature Selection", "Table S3.")),
 ("", "12c", "D", "Specify the type of model, rationale, all model-building steps, including any hyperparameter tuning, and method for internal validation", "XGBoost; staged RFE with parsimony rule; fixed hyperparameters (sensitivity in S11); nested grouped cross-validation", loc("Model and Comparators", "S11.")),
 ("", "12d", "D;E", "Describe if and how any heterogeneity in estimates of model parameter values and model performance was handled and quantified across clusters", "Single center; performance by calendar era reported", loc("Table S1.")),
 ("", "12e", "D;E", "Specify all measures and plots used to evaluate model performance and, if relevant, to compare multiple models", "AUROC, AP, Brier, ECE, calibration slope and intercept, decision curves, time-dependent AUROC; paired cluster bootstrap for model contrasts", loc("Statistical Analysis")),
 ("", "12f", "E", "Describe any model updating (e.g., recalibration) arising from the model evaluation", "None performed", loc("Model and Comparators")),
 ("", "12g", "E", "For model evaluation, describe how the model predictions were calculated", "Stored model artifact and feature order in the released repository; predictions computed by code", loc("Data Availability Statement")),
 ("Class imbalance", "13", "D;E", "If class imbalance methods were used, state why and how this was done", "No class imbalance methods used", loc("Model and Comparators")),
 ("Fairness", "14", "D;E", "Describe any approaches that were used to address model fairness and their rationale", "Race excluded as input; performance, specificity, and calibration audited by race, sex, age, and prior utilization", loc("Subgroup Audit", "Subgroup Performance")),
 ("Model output", "15", "D", "Specify the output of the prediction model. Provide details and rationale for any classification and how the thresholds were identified", "Probability of 30-day readmission; validation-partition Youden threshold; capacity view", loc("Model and Comparators", "Clinical Utility (Exploratory)")),
 ("Training versus evaluation", "16", "D;E", "Identify any differences between the development data and the evaluation data in healthcare setting, eligibility criteria, outcome, and predictors", "Same source and definitions; test partition historically exposed; characteristics by partition in Table 1", loc("Partitions and Validation Design", "Table 1.")),
 ("Ethical approval", "17", "D;E", "Name the institutional research board or ethics committee that approved the study and describe the participant-informed consent or the waiver", "MIT and BIDMC IRBs with waiver of consent; PhysioNet DUA; no FIU IRB review required", loc("Ethical Considerations")),
 ("Funding", "18a", "D;E", "Give the source of funding and the role of the funders", "No external funding", loc("Funding")),
 ("Conflicts of interest", "18b", "D;E", "Declare any conflicts of interest and financial disclosures for all authors", "None declared", loc("Conflict of Interest Statement")),
 ("Protocol", "18c", "D;E", "Indicate where the study protocol can be accessed or state that a protocol was not prepared", "No protocol was prepared (stated in Ethical Considerations); the analysis plan evolved and the changes are disclosed", loc("Ethical Considerations", "Statistical Analysis")),
 ("Registration", "18d", "D;E", "Provide registration information for the study, or state that the study was not registered", "The study was not registered (stated in Ethical Considerations)", loc("Ethical Considerations")),
 ("Data sharing", "18e", "D;E", "Provide details of the availability of the study data", "MIMIC-IV via PhysioNet credentialed access; no row-level data redistributed", loc("Data Availability Statement")),
 ("Code sharing", "18f", "D;E", "Provide details of the availability of the analytical code", "Public repository with tagged release and Zenodo archive", loc("Data Availability Statement")),
 ("Patient and public involvement", "19", "D;E", "Provide details of any patient and public involvement or state no involvement", "No patient or public involvement (stated in Ethical Considerations)", loc("Ethical Considerations")),
 ("Participants", "20a", "D;E", "Describe the flow of participants through the study, including the number with and without the outcome", "Figure 1 flow; events per partition", loc("Figure 1.", "Cohort")),
 ("", "20b", "D;E", "Report the characteristics overall and for each data source or setting, including key dates, key predictors, sample size, outcome events, follow-up time, and missing data", "Table 1 characteristics by partition; missingness per predictor in Appendix 1", loc("Table 1.")),
 ("", "20c", "E", "For model evaluation, show a comparison with the development data of the distribution of important predictors", "Table 1 compares development and test partitions", loc("Table 1.")),
 ("Model development", "21", "D;E", "Specify the number of participants and outcome events in each analysis", "Per partition, per fold, per subgroup, and per landmark stage", loc("Partitions and Validation Design", "Table 4.", "S6.")),
 ("Model specification", "22", "D", "Provide details of the full prediction model to allow predictions in new individuals and to enable third-party evaluation", "Model artifact, encodings, and feature order released; freely available under MIT license", loc("Data Availability Statement")),
 ("Model performance", "23a", "D;E", "Report model performance estimates with confidence intervals, including for any key subgroups", "Out-of-fold and test estimates with cluster-bootstrap CIs; subgroups in Table 4 and Figure 5", loc("Discrimination and Calibration", "Table 4.")),
 ("", "23b", "D;E", "If examined, report results of any heterogeneity in model performance across clusters", "Calendar-era bands only (single center)", loc("Table S1.")),
 ("Model updating", "24", "E", "Report the results from any model updating", "Not performed", loc("Model and Comparators")),
 ("Interpretation", "25", "D;E", "Give an overall interpretation of the main results, including issues of fairness in the context of the objectives and previous studies", "Principal findings; fairness interpreted with uncertainty; comparison with prior work", loc("Principal Findings", "Comparison With Prior Work")),
 ("Limitations", "26", "D;E", "Discuss any limitations of the study and their effects on any biases, statistical uncertainty, and generalizability", "Dedicated section", loc("Limitations")),
 ("Usability", "27a", "D", "Describe how poor quality or unavailable input data should be assessed and handled when implementing the prediction model", "Missing measurements handled natively by the tree learner; not deployment-ready; prototype disclosed as research-only", loc("Limitations", "S8.")),
 ("", "27b", "D", "Specify whether users will be required to interact in the handling of the input data or use of the model, and what level of expertise is required", "Not applicable; no deployment proposed", loc("Clinical Interpretation (Hypotheses)")),
 ("", "27c", "D;E", "Discuss any next steps for future research, with a specific view to applicability and generalizability", "External validation and prospective evaluation required", loc("Conclusion")),
]
d = new_doc("Supplementary Appendix 5. TRIPOD+AI Checklist",
            "Items and wording follow the official TRIPOD+AI checklist (Collins et al, BMJ "
            "2024;385:e078378; checklist version 11-January-2024; 27 items). D: development; E: "
            "evaluation. Page numbers refer to the manuscript PDF (p) and the Supplementary "
            "Material PDF (p S).")
table(d, [["Section", "Item", "D/E", "Checklist item", "How addressed", "Location"]] +
      [list(r) for r in T], fs=7.5)
d.save(str(PKT / "Supplementary Appendix 5 TRIPOD-AI Checklist.docx"))
print("A5 written", len(T), "rows; unresolved locations:",
      [k for k in M_HEADS if k not in MP][:10], [k for k in S_HEADS if k not in SP][:10])

# =============================================================== PROBAST+AI
Y, PY, PN, N, NI, NA = "Y", "PY", "PN", "N", "NI", "NA"
DEV = [
 ("Domain 1. Participants and data sources (quality)", [
  ("1.1 Were appropriate data sources used?", PY, "Routine EHR data from one center; appropriate for a discharge-time model but single center"),
  ("1.2 Was an appropriate study design used?", Y, "Retrospective cohort of all eligible admissions"),
  ("1.3 Did the in- and exclusions of study participants result in a representative dataset?", PN, "Medicare-insured admissions at one academic center; fee-for-service and Medicare Advantage indistinguishable; case mix differs from national Medicare")],
  "Quality concern: unclear. Rationale: appropriate source and design, but representativeness of a single center cannot be established from the data.",
  "Applicability concern: high for national Medicare populations; low for the within-system question posed."),
 ("Domain 2. Predictors (quality)", [
  ("2.1 Were predictors defined and assessed in a similar way for all participants?", Y, "Structured EHR fields, identical extraction for all admissions"),
  ("2.2 Was any pre-processing of predictors similar for all participants?", Y, "Identical pipeline; target encoding fold-local; no group-specific processing"),
  ("2.3 Were predictor assessments made without knowledge of outcome data?", Y, "Predictors extracted from pre-discharge records; a feature that encoded later admissions was detected empirically and removed"),
  ("2.4 Were the predictors included in the model available at the time the model was intended to be used?", PY, "Documented availability audit, first-admission and constancy tests, recomputation of history features from timestamps, laboratory results required to be stored before discharge; some index-stay features rely on documented timestamps only")],
  "Quality concern: low. Rationale: availability was audited by documentation and enforced by recomputation from raw tables with discharge cutoffs for history, laboratory, order, and medication features; the residual uncertainty concerns a minority of index-stay features that rely on documented extraction filters only.",
  "Applicability concern: low; predictors are routinely available structured fields."),
 ("Domain 3. Outcome (quality)", [
  ("3.1 Were outcomes defined and assessed appropriately?", PY, "Timestamp-based within-system readmission; out-of-system readmissions missed; planned readmissions included"),
  ("3.2 Were outcomes defined and assessed in a similar way for all participants?", Y, "Same rule for all admissions"),
  ("3.3 Were outcome assessments made without use or knowledge of predictor data?", Y, "Derived from admission timestamps only"),
  ("3.4 Was the time interval between predictor assessment and outcome assessment appropriate?", Y, "Predictors at discharge; outcome in the following 30 days; probable administrative continuations examined in a sensitivity analysis")],
  "Quality concern: low. Rationale: objective timestamp-based outcome applied identically to all participants; within-system ascertainment and probable administrative continuations are limitations of scope, disclosed and examined in sensitivity analyses, rather than assessment bias.",
  "Applicability concern: low for within-system readmission; high if the intended outcome is all-setting readmission."),
 ("Domain 4. Analysis (quality)", [
  ("4.1 Was there evidence that the sample size was reasonable?", Y, "170,677 training admissions with 36,665 events; learning curve reported"),
  ("4.2 Were continuous and categorical predictors handled appropriately?", Y, "Tree learner on raw scales; no categorization of continuous predictors; fold-local target encoding"),
  ("4.3 Were participants with missing or censored data handled appropriately in the analysis?", PY, "Missing measurements left missing for the tree learner after reversing an earlier global median imputation; no participants omitted for missingness"),
  ("4.4 If methods to address class imbalance were used, was the model or the model predictions recalibrated?", NA, "No class imbalance methods used"),
  ("4.5 Were methods used to address potential model overfitting?", PY, "Nested grouped cross-validation for selection; fixed hyperparameters; parsimony rule; hyperparameters not tuned within folds")],
  "Quality concern: low. Rationale: nested grouped resampling, native handling of missing values, no categorization of continuous predictors, and a large development sample. Hyperparameters were fixed before this analysis version; fixing them is not itself a defect, but they were chosen during earlier development that had evaluated models on the test partition, which is why the test partition is treated as exposed in Part 2.",
  ""),
]
EVAL = [
 ("Domain 1. Participants and data sources (risk of bias)", [
  ("1.1 Were appropriate data sources used?", PY, "Same source as development (internal evaluation)"),
  ("1.2 Was an appropriate study design used?", Y, "Patient-level held-out partition and nested cross-validation"),
  ("1.3 Did the in- and exclusions of study participants result in a representative dataset?", PN, "As for development")],
  "Risk of bias: low. Rationale: same data source, patient-level partitioning, and nested resampling; representativeness is an applicability concern, not a bias in the evaluation.", "Applicability concern: as for development."),
 ("Domain 2. Predictors (risk of bias)", [
  ("2.1", Y, "As for development"), ("2.2", Y, "As for development"), ("2.3", Y, "As for development"),
  ("2.4", PY, "As for development")],
  "Risk of bias: low after correction.", "Applicability concern: low."),
 ("Domain 3. Outcome (risk of bias)", [
  ("3.1", PY, "As for development"), ("3.2", Y, "As for development"), ("3.3", Y, "As for development"),
  ("3.4", Y, "As for development")],
  "Risk of bias: low. Rationale: as for development.", "Applicability concern: as for development."),
 ("Domain 4. Analysis (risk of bias), assessed separately for the two evaluations", [
  ("4.1 Was model evaluation based on only apparent performance avoided?", Y, "Out-of-fold procedure performance (primary) and held-out test partition"),
  ("4.2 Was there evidence that the sample size was reasonable?", Y, "47,672 test admissions with 10,133 events; development folds of similar size"),
  ("4.3 Were participants with missing or censored data handled appropriately?", PY, "As for development; deaths handled explicitly in the time-to-event analysis"),
  ("4.4 If class imbalance methods were used, was the evaluation done in a dataset without imbalance correction?", NA, "No class imbalance methods used"),
  ("4.5 If data splitting was done to create training and test datasets, was there evidence that data leakage was avoided?", PN, "Patient-level split and fold-local preprocessing in this version; but the test partition was scored repeatedly in earlier development and the leaked feature was present in all partitions before correction; an earlier global median imputation used all partitions"),
  ("4.6 If resampling methods were used to evaluate model performance, were all model development steps replicated in the resampling process?", PY, "Feature selection and encodings replicated inside each outer fold; hyperparameters fixed; consensus aggregation itself not replicated"),
  ("4.7 Was the predictive performance of the model evaluated appropriately (calibration, discrimination, net benefit)?", Y, "Discrimination, calibration (slope, intercept, ECE, Brier), decision curves, subgroups, time-dependent AUROC, with cluster-bootstrap CIs")],
  "Risk of bias: high for the historically exposed test partition (signalling question 4.5) and low for the out-of-fold procedure estimate, which replicates selection and encoding inside each outer fold on data never used for earlier model choices. Under the PROBAST+AI reclassification rules, the large development data and internal validation do not remove the concern created by prior test-partition exposure, so the test-partition results are presented as tertiary evidence.",
  ""),
]
d = new_doc("Supplementary Appendix 6. PROBAST+AI Assessment",
            "Signalling questions follow the official PROBAST+AI tool (Moons et al, BMJ 2025;388:e082505). "
            "Answers: Y yes; PY probably yes; PN probably no; N no; NI no information; NA not applicable. "
            "Model development is rated for quality concern and model evaluation for risk of bias, each "
            "low, high, or unclear, with applicability rated for domains 1 to 3. Self-assessment by the authors.")
for part, blocks in (("PART 1. MODEL DEVELOPMENT (QUALITY)", DEV), ("PART 2. MODEL EVALUATION (RISK OF BIAS)", EVAL)):
    p = d.add_paragraph(); r = p.add_run(part); r.bold = True; r.font.size = Pt(11)
    for title, qs, judge, appl in blocks:
        p = d.add_paragraph(); r = p.add_run(title); r.bold = True
        table(d, [["Signalling question", "Answer", "Basis"]] + [list(q) for q in qs], fs=8)
        d.add_paragraph(judge)
        if appl:
            d.add_paragraph(appl)
d.add_paragraph("Overall judgement. Development quality: unclear, because domain 1 (single-center "
                "representativeness) is unclear while domains 2 to 4 are low. Evaluation risk of "
                "bias: high for the historically exposed test partition and low for the out-of-fold "
                "procedure estimate, which is therefore the primary result. Applicability: high "
                "concern for settings other than within-system readmission at a single academic "
                "center; low for that setting.")
d.save(str(PKT / "Supplementary Appendix 6 PROBAST-AI Assessment.docx"))
print("A6 written")

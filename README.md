# nuMoM2b SMM Prediction Pipeline

Severe Maternal Morbidity (SMM) prediction study using the nuMoM2b longitudinal pregnancy cohort. This repository contains the full preprocessing, feature engineering, outcome construction, and machine learning pipeline.

---

## Overview

Severe Maternal Morbidity (SMM) is a composite indicator of life-threatening complications during delivery hospitalization. This project builds a binary SMM classifier using prenatal clinical and sociodemographic data collected at Visits 1 and 3 of the nuMoM2b study, with the goal of early-pregnancy risk stratification.

| Item | Detail |
|---|---|
| Dataset | nuMoM2b Merged Dataset (restricted access) |
| Outcome | SMM (binary 0/1), constructed from coded variables and free-text SP fields |
| Predictors | 35 engineered variables covering demographics, medical history, prenatal labs, and behavioral factors |
| Languages | R (preprocessing, outcome, descriptive statistics), Python (ML modeling) |
| Train / Test | 80 / 20 stratified split, random seed 42 |

---

## Repository Structure

```
scripts/
    s1_selecting_blue_variables_63_checking_variables.Rmd
    s2_selecting_pink_variables.Rmd
    s3_processing_pink_variables.Rmd
    s4_ml_pipeline.ipynb
    s4_ml_pipeline_v2.ipynb
    s4_ml_pipeline_v2_cvloop.ipynb
    s4_ml_pipeline_v3.ipynb
    s4_ml_pipeline_v4.ipynb
    s5_table1_descriptive.Rmd

data/
    general/
        nuMoM2b_Merged_Dataset.csv          # not included (restricted)
        HIGHLIGHTED_nuMoM2b_Merged_Dataset_Codebook_9.14.25.xlsx

outputs/
    s1/    # blue predictor variables
    s2/    # pink outcome variables
    s3/    # SMM outcome + merged modeling dataset
    s4/    # ML model results (v1–v4)
    s5/    # Table 1 descriptive statistics
```

---

## Pipeline

The five scripts run sequentially. Each script reads from the previous script's outputs directory.

```
nuMoM2b_Merged_Dataset.csv
        │
        ├── s1_selecting_blue_variables
        │       Select 63 → engineer → 35 predictor variables
        │       outputs/s1/v1_1_blue_variables.csv
        │
        ├── s2_selecting_pink_variables
        │       Select 131 pink variables, exclude CMAE14=1 rows
        │       outputs/s2/v1_2_pink_variables_clean.csv
        │
        └── s3_processing_pink_variables
                Construct SMM from coded vars + text matching
                Merge with blue predictors
                outputs/s3/v2_1_blue_with_SMM.csv
                        │
                        ├── s4_ml_pipeline (v1–v4)
                        │       KNN impute → scale → tune → evaluate
                        │       outputs/s4/{version}/
                        │
                        └── s5_table1_descriptive
                                Table 1 descriptive statistics
                                outputs/s5/Table1_descriptive_statistics.docx
```

---

## Scripts

### s1 — Selecting Blue Variables

Selects 63 predictor variables from the raw nuMoM2b dataset, engineers them into 35 model-ready features, and saves the processed dataset.

Key steps:
- Variable existence check against raw dataset columns
- Codebook lookup for Variable_Type, Variable_Unit, Variable_Code_List_if_Coded
- Missing rate computation at each transformation stage
- BMI derivation: combines weight in kg (CMAB01a1) and lbs (CMAB01a2); computes BMI = weight_kg / (height_m)^2; drops raw height/weight columns
- Variable merging: 11 composite binary flags derived from sub-variables using `any_flag()` (returns 1 if any sub-variable = 1, NA if all sub-variables are NA, 0 otherwise). Groups: Ins, Diabetes, Sleep_Apnea, Cardiac_Disease, Autoimmune_Disease, Thyroid_Disorder, Prior_Uterine_Surgery, Bleeding_Disorder, Tobacco_Use, Alcohol_Use, Drug_Use
- NA recoding by variable type: Yes_No_v1 variables recode 2 and NA to 0; Checked variables recode NA to 0
- Non-numeric values (e.g., skip codes `D`, `R`) replaced with NA
- One-hot encoding: Race (reference = Non-Hispanic White) and Marital Status (reference = Married/Living with partner) with first level dropped

Output: `outputs/s1/v1_1_blue_variables.csv` (35 columns including Race and Marital dummies)

---

### s2 — Selecting Pink Variables

Selects 131 outcome-relevant (pink) variables and filters rows using CMAE14 as an exclusion criterion.

- CMAE14 = 1 rows are saved separately for record-keeping (not used in modeling)
- Remaining rows form the clean pink dataset passed to s3

Output: `outputs/s2/v1_2_pink_variables_clean.csv`

---

### s3 — Processing Pink Variables and Creating SMM Column

Constructs the binary SMM outcome from two complementary approaches, then merges with the blue predictors from s1.

**Step 1 — Variable removal.** Drops 16 variables before SMM coding: 8 that would render SMM always NA (CMAE11e_SP, CMAE12c, CMAE13c1–c4, CMAE13c4_SP, CMAI01A_INT), 7 risk-factor variables excluded from SMM coding (CMBJ04 x4, CMDA08c, CMEA03), and A04A05.

**Step 2 — Coded variable rules.**
- Event_Related variables: value 1 or 2 triggers SMM = 1
- Adverse_Event_v1, Checked, Yes_No_v1, Yes_No_v4, Yes_No_v5 variables: value = 1 triggers SMM = 1

**Step 3 — Free-text SP field matching.**
- Five SP variables (CMAE15a_sp, CMAE15b_sp, CMAE15c_sp, CMAE15d_sp, CMAJ01d5_SP) use exact-value matching against pre-approved clinical term lists (case-insensitive, whitespace-trimmed)
- Four SP variables (CMAD01h_SP, CMAD01i_SP, CMAJ01d1_SP, A04A05_1) use keyword matching against an `smm_dict` covering 17 clinical domains: SMM core terms, transfusion/hemorrhage, hypertensive disorders, infection/sepsis, thromboembolic events, amniotic fluid embolism, sickle cell disease, DIC, respiratory failure, cardiac arrest/cardiomyopathy, puerperal cerebrovascular disorders, neurological, renal/hepatic failure, aneurysm, hysterectomy, tracheostomy, anesthesia-related complications

**Final SMM column:** SMM = 1 if any coded rule or text match fires. An attribution table reports how many individuals were identified by each method (numeric only, text only, both).

Outputs:
- `outputs/s3/v2_1_blue_with_SMM.csv` — primary modeling dataset (35 predictors + SMM)
- `outputs/s3/v2_2_blue_with_SMM_extended.csv` — extended with CMAC15 ICD-based delivery complication columns

---

### s4 — ML Pipeline (v1–v4)

Trains and evaluates ML classifiers for SMM prediction. Four versions with progressively refined imbalance-handling strategy.

**Shared steps (all versions):**
1. Load `v2_1_blue_with_SMM.csv`
2. KNN imputation (`n_neighbors=5, weights='uniform'`) on the full dataset before splitting; one-hot post-processing (argmax within Race_ and Marital_ dummy groups to restore valid one-hot encoding after continuous KNN interpolation)
3. Stratified 80/20 train/test split
4. Min-Max scaling fit on train set only
5. `RandomizedSearchCV` with `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)`
6. Threshold optimization: sweep 0.10–0.89 to maximize F1; report metrics at both default (0.50) and optimized thresholds
7. Metrics: AUC-ROC, AUC-PR, F1-macro, F1-weighted, Sensitivity, Specificity, PPV, NPV

**Version comparison:**

| Version | Models | Imbalance Handling | Scoring | N_ITER |
|---|---|---|---|---|
| v1 | RF + XGBoost | SMOTE-Tomek | f1_weighted | 50 |
| v2 | RF + XGBoost | SMOTE-Tomek | f1_weighted | 50 |
| v3 | RF + XGBoost | class_weight / scale_pos_weight | recall | 30 |
| v4 | LightGBM | is_unbalance=True | f1_weighted | 50 |

---

### s5 — Table 1: Descriptive Statistics

Produces an unstratified Table 1 for the full modeling cohort using `gtsummary::tbl_summary()`. Continuous variables reported as Mean (SD); categorical as n (%). Exported to Word via `flextable::save_as_docx()`.

Variable groupings: sociodemographic/SDoH, baseline medical history, late pregnancy clinical status, behavioral factors.

Output: `outputs/s5/Table1_descriptive_statistics.docx`

---

## Predictor Variables (35 features)

| Domain | Variables |
|---|---|
| Demographics | AgeCat_V1, Education, Race (one-hot: Race_2–Race_9), Marital_Status (one-hot: Marital_2–Marital_5), V1AF10 (English proficiency), V1AF09 (years in US) |
| Insurance | Ins (any insurance, binary) |
| Obstetric history | GravCat |
| Chronic conditions | ChronHTN, Diabetes, VXXB01ab_FA (asthma), Sleep_Apnea, VXXB01al_FA (renal disease), Cardiac_Disease, Autoimmune_Disease, VXXB01ac_FA (seizure disorder), Thyroid_Disorder, Prior_Uterine_Surgery, VXXB01ap_FA (thrombocytopenia), Bleeding_Disorder, VXXB01ar_FA (blood clots/stroke) |
| Prenatal clinical | BMI, CMAC03_wks (GA at delivery), V3BA02a1/b1 (systolic/diastolic BP), CMAH01b4/b6/b1 (hematocrit/platelets/WBC), CMDA04a/b (new-onset HTN/proteinuria), CMBG01 (cerclage), A09A03b3 (suspected preeclampsia/HELLP) |
| Behavioral | Tobacco_Use, Alcohol_Use, Drug_Use |

---

## Requirements

**R packages**

```r
install.packages(c(
  "readr", "dplyr", "readxl", "writexl",
  "here", "knitr", "ggplot2", "purrr",
  "stringr", "gtsummary", "flextable"
))
```

**Python packages**

```bash
pip install numpy pandas scikit-learn xgboost lightgbm scipy matplotlib tqdm
```

---

## Notes

- The nuMoM2b dataset is not publicly available. Access requires application through the study data coordinating center.
- All scripts use `here::i_am()` for reproducible relative paths; run from the project root.
- `set.seed()` / `random_state=42` used throughout for reproducibility.
- s4 notebooks are designed to run on Google Colab Pro (A100/L4/T4) due to the computational cost of `RandomizedSearchCV` with 15 CV fits × 30–50 hyperparameter candidates.

---

## Reference

Haas DM, et al. (2015). The Nulliparous Pregnancy Outcomes Study: Monitoring Mothers-to-Be (nuMoM2b): the rationale and protocol. *American Journal of Obstetrics and Gynecology*, 212(4), 539.e1–539.e24.

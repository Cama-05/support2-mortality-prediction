# Dataset Introduction — SUPPORT2

**Source:** UCI Machine Learning Repository — [SUPPORT2](https://archive.ics.uci.edu/dataset/880/support2) (id: 880)  
**Area:** Health and Medicine  
**Year:** 1995 (studies 1989–1991 and 1992–1994)  
**Creator:** Frank Harrell, Vanderbilt University Department of Biostatistics  
**DOI:** 10.3886/ICPSR02957.v2

---

## Overview

SUPPORT2 (Study to Understand Prognoses Preferences Outcomes and Risks of Treatment) is a prospective study of **9,105 seriously ill hospitalised adults** across 5 U.S. medical centres. The goal of the original study was to develop and validate prognostic models for estimating 180-day survival, and to improve end-of-life decision-making.

Patients were enrolled if they had one of nine serious conditions:

- Acute Respiratory Failure (ARF)
- Chronic Obstructive Pulmonary Disease (COPD)
- Congestive Heart Failure (CHF)
- Liver disease
- Coma
- Colon cancer
- Lung cancer
- Multiple Organ System Failure with malignancy (MOSFM)
- Multiple Organ System Failure with sepsis (MOSFS)

Baseline measurements were recorded at **day 3 of the hospital stay** — the point at which a meaningful prognosis can be made. Data were collected from medical records, personal interviews, and the National Death Index (NDI).

---

## Target variable

| Column | Description |
|---|---|
| `death` | 1 if the patient died within **180 days** of enrolment, 0 if survived. Rate ≈ 68%. |

## Goals

1. **Data Cleaning and Tidying:** Clean and tidy the dataset appropriately to detect possible pitfalls, with an appropriate simplified description (e.g. the distribution of the values in the columns, etc.) 
2. **Data Quality Assessment:** Evaluate the quality of the dataset, including completeness for missing values.
3. **Data Visualization:** 
   - Produce a data visualization to explore the possible relationship between the mortality rate and the comorbity conditions. Are some groups lethal, despite low comorbidity? 
   In particular, show the mean number of comorbidities (`num.co`) per disease group and the proportion of patients who died vs survived, for the same groups. Comment.
   - Does the economic cost of hospitalisation vary with a patient's income and cancer status? Produce an appropriate visualization to answer, and comment.
   - Produce an **interactive** visualization showing how the distribution of key physiological markers varies at day 3 differ across disease groups.
4. **Data Analysis:** Create a classification pipeline that predicts, given a sample, what is the predicted `death`. 

---

## Feature schema

### Demographics & administrative

| Column | Type | Description |
|---|---|---|
| `age` | float | Age at enrolment (years) |
| `sex` | str |  `"male"` / `"female"` |
| `race` | str | Race/ethnicity (`white`, `black`, `asian`, `hispanic`, `other`) |
| `edu` | float | Years of education |
| `income` | str | Annual income category (`"under $11k"`, `"$11-$25k"`, `"$25-$50k"`, `">$50k"`) |

### Diagnosis & comorbidities

| Column | Type | Description |
|---|---|---|
| `dzgroup` | str | Specific disease group (9 categories, see above) |
| `dzclass` | str | Broader disease class (`ARF`, `MOSF`, `CHF`, `COPD`, `Coma`, `Cancer`) |
| `num.co` | int | Number of comorbid conditions |
| `diabetes` | int | Diabetes comorbidity flag (0/1) |
| `dementia` | int | Dementia comorbidity flag (0/1) |
| `ca` | str | Cancer status: `"no"`, `"yes"` (local), `"metastatic"` |

### Physiology — measured at day 3

| Column | Type | Description |
|---|---|---|
| `meanbp` | float | Mean arterial blood pressure (mmHg) |
| `hrt` | float | Heart rate (beats/min) |
| `resp` | float | Respiratory rate (breaths/min) |
| `temp` | float | Body temperature (°C) |
| `wblc` | float | White blood cell count (thousands/µL) |
| `pafi` | float | PaO₂/FiO₂ ratio — oxygenation index; lower = worse lung function |
| `alb` | float | Serum albumin (g/dL); marker of nutrition and severity |
| `bili` | float | Serum bilirubin (mg/dL); marker of liver function |
| `crea` | float | Serum creatinine (mg/dL); marker of kidney function |
| `sod` | float | Serum sodium (mEq/L) |
| `ph` | float | Arterial blood pH |
| `glucose` | float | Serum glucose (mg/dL) |
| `bun` | float | Blood urea nitrogen (mg/dL); marker of kidney function |
| `urine` | float | Urine output (mL/day) |

### Severity & prognosis scores

| Column | Type | Description |
|---|---|---|
| `scoma` | float | Glasgow Coma Score at day 3 (3–15; lower = deeper coma) |
| `sps` | float | SUPPORT Physiology Score — study's own composite severity index |
| `aps` | float | APACHE III physiology score at day 3 — standard ICU severity score |
| `avtisst` | float | Average TISS score (days 3–25); measures ICU treatment intensity |
| `hday` | int | Day in hospital at study entry (always 3 in most cases) |

### Physician survival estimates — recorded at day 3

| Column | Type | Description |
|---|---|---|
| `surv2m` | float | Physician's estimate of 2-month survival probability (0–1) |
| `surv6m` | float | Physician's estimate of 6-month survival probability (0–1) |
| `prg2m` | float | Model-based 2-month survival probability from SUPPORT prognostic model |
| `prg6m` | float | Model-based 6-month survival probability from SUPPORT prognostic model |

### DNR (Do Not Resuscitate) orders
A DNR order is a medical directive — written by a doctor, usually after discussing with the patient or family — that instructs healthcare staff not to attempt resuscitation (CPR, defibrillation, intubation) if the patient's heart or breathing stops. 

| Column | Type | Description |
|---|---|---|
| `dnr` | str | DNR order status at day 3: `"no dnr"`, `"dnr before enroll"`, `"dnr after enroll"` |
| `dnrday` | float | Day on which the DNR order was written (NaN if no DNR) |

### Activities of Daily Living (ADL)

| Column | Type | Description |
|---|---|---|
| `adlp` | float | ADL score reported by the **patient** — ability to perform daily tasks (0–7; higher = more dependent) |
| `adls` | float | ADL score reported by a **surrogate** (caregiver/family) |
| `adlsc` | int | Imputed/calibrated ADL score used by the SUPPORT model |

### Economic outcomes (recorded during/after stay)

| Column | Type | Description |
|---|---|---|
| `charges` | float | Total hospital charges in USD |
| `totcst` | float | Total RCC (Resource Cost Charge) cost |
| `totmcst` | float | Total micro-cost estimate |

---

## Key observations for machine learning
  
1. **Class imbalance** — `death` is ≈68% positive (the positive label is the majority class)
2. **Leakage candidates** — cost columns (`charges`, `totcst`, `totmcst`) and functional outcomes (`sfdm2`, `adlp`) accumulate or are measured *after* the hospital stay and are unavailable at admission-time prediction. `dnr` status is assigned *during* the stay and is correlated with outcome by design.
3. **Multicollinearity** — `sps`, `aps`, `surv2m`, `surv6m`, `prg2m`, `prg6m` are all composite severity/prognosis scores; they overlap substantially with the raw physiology features.
4. **Nominal encoding** — `dzgroup`, `dzclass`, `income`, `race`, `ca`, `dnr` require encoding before use in linear models.
5. **Sentinel values** — inspect `scoma` for `-999` (used in some versions as a missing-data sentinel) and `sex` for `"unknown"` strings.

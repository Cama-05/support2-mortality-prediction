# Dataset Hints — death_v1

**Target variable:** `death` (1 = patient died within 180 days of enrolment, 0 = survived)

## What to investigate

1. **Feature availability at prediction time** — some columns describe outcomes only
   known *after* the patient's stay ends (total hospital charges, functional follow-up
   status measured at 2 months). These would be unavailable at the time a clinician
   needs to make a prediction at day 3 of the stay.

2. **Data integrity** — check whether any rows appear more than once. What happens to
   your train/test evaluation if the same patient appears in both sets?

3. **Evaluation metric** — the class distribution is strongly skewed toward one class.
   Which direction? What does the confusion matrix look like for a trivial baseline?

4. **Missing values** — multiple columns have significant missingness with right-skewed
   distributions. One categorical column contains a non-standard string value instead
   of NaN.

## Expert Knowledge

A medical informatics specialist has reviewed this dataset and provided the following guidance:

- **`bun`** — Blood urea nitrogen, a marker of kidney function. The distribution is
  right-skewed with a long upper tail in patients with renal impairment.
  Clinical reference: median BUN in this population is approximately **6.51 mg/dL**.
  *Recommended imputation strategy: median.*

- **`alb`** — Serum albumin, a marker of nutritional status and illness severity.
  Outliers are present (severely malnourished patients and measurement artefacts).
  Clinical reference: median albumin in this population is approximately **3.5 g/dL**.
  *Recommended imputation strategy: median (more robust than mean when outliers are present).*

# Study protocol 2: dengue severity, state by state

Frozen on 2026-11-26, in a commit that comes before the study code. Changes
after that go in the "Deviations" section, with a date and a reason.

The first study ([protocol.md](protocol.md), [findings.md](findings.md)) had
ten transfers. Most of its results were already known (see
[related-work.md](related-work.md)). This study asks two questions that ten
targets could not answer.

## Questions

1. **When does a labelled update hurt?** Is the number of local severe cases in
   the sample a better guide than the number of patients?
2. **Can a clinic tell, without labels, that the label-free repair (EM) will
   fail?**

## What was looked at before freezing

- The first study's results, including that EM made log loss worse in 7 of 10
  targets and that the dengue repair sizes looked closer in cases (9 to 48)
  than in patients (250 to 2,500). Question 1 came from that second
  observation, so the first study cannot count as evidence for it.
- Counts of lab-confirmed cases and severe cases per state and year, for 2023
  and 2025. These set the inclusion rule below. No model was scored on any
  state.

Nothing else about the states was looked at.

## Data and targets

SINAN dengue, lab-confirmed cases, the same label and the same 25 inputs as the
served dengue model (`ml/clinical_data.py`). A target is one state in one year.

**Inclusion rule:** a state-year is a target if it has at least 1,000
confirmed cases and at least 100 severe cases. On the counts above that gives
14 targets in 2023 and 15 in 2025, 29 in all. Smaller state-years are listed in
the report with their counts and left out.

**2024, decided now:** the 2024 file is not on disk. If it is downloaded before
stage A starts, its state-years join stage B under the same inclusion rule.
If not, the study runs on 2023 and 2025 and this is not a deviation.

## Source models

For each target state, one model trained on a stratified sample of up to
200,000 confirmed 2023 cases from every other state. It never sees the target
state, in any year. The same source model serves that state's 2023 and 2025
targets, so a 2025 target moves in place and in time.

Gradient boosting with the served settings and the served sigmoid
calibration. Those settings were chosen on 2023 rows that include every state.
That is a small leak, the same one the first study had.

## Updating methods and design

The six methods of the first study, unchanged: none, em, intercept, platt,
refit, rate (see [protocol.md](protocol.md) and `study/updating.py`).

Each repeat splits the target in half at random, stratified on the label. One
half is the pool, the other the test half, capped at 20,000 rows.

- n = 100, 250, 500, 1000, 2500, each used only where the pool holds at least
  n patients. Draws are not stratified, so the number of local severe cases
  varies from draw to draw. That variation is what question 1 uses.
- 50 repeats. Seed 42 plus the repeat number. Methods in one repeat share rows.
- Each draw records its number of local severe cases.

Primary measure: log loss on the test half, as the paired difference from
`none`. A draw is **hurt** if the update's log loss is above `none`'s. Helps,
helps reliably and repair size are defined as in the first study.

## The label-free check

For each target, a **shift score**: the cross-validated AUC (5 folds) of a
gradient-boosted classifier that tells the source training rows from the
target's pool rows, using the 25 inputs and no labels. EM assumes the inputs
look the same within each class in both places. A high shift score means they
probably do not.

## Stages

- **Stage A:** the 2023 targets. Fit the EM guard (below) here.
- **Stage B:** the 2025 targets (and 2024 if present). The guard threshold is
  written to the results file before stage B runs, and is not changed after.

## Hypotheses and how each can fail

All four are confirmatory. Every verdict is reported, supported or not.

- **S1. Few cases, more harm.** For `intercept`, at every n pooled, compare the
  share of hurt draws with fewer than 10 local severe cases against draws with
  20 or more. Supported if the first share is higher in at least 75% of the
  targets that have at least 20 draws in each group.
- **S2. Cases travel better than patients.** For targets with a repair size for
  `intercept`, express it in patients and in local severe cases (the mean over
  draws at that n). Supported if the coefficient of variation across targets
  is smaller in cases than in patients, and the bootstrap interval of the
  ratio (cases CV over patients CV) lies below 1.
- **S3. The shift score warns of EM failure.** At each target's largest n,
  take EM's mean paired log-loss change. Supported if its Spearman correlation
  with the shift score, over all targets, is above 0 and the bootstrap
  interval excludes 0.
- **S4. The guard works on new targets.** On stage A, choose the shift-score
  threshold that minimizes the summed log loss of "EM below the threshold,
  none above". On stage B, supported if the guard is worse than `none` in
  fewer targets than plain EM is, and helps (beats `none`) in at least one.

Two replications of the first study, reported with the same pass marks:
`intercept` helps at n = 500 in at least 75% of targets whose pool allows it,
and EM at the largest n is worse than `none` in at least 25% of targets.

## Uncertainty

Targets are not independent. One state appears in two years and shares a
source model, and all source models share most training rows. Every interval
across targets comes from a bootstrap that resamples **states** (1,000
times), keeping a state's years together. Spread over repeats describes the
luck of the draw within a target only.

## Limits fixed in advance

- Only 29 targets meet the rule without 2024. S3 and S4 rest on a correlation
  over 29 points and a test on 15, so a null result may mean low power. The
  report says so if it happens.
- States may classify severe dengue differently. A model that breaks and a
  label that changed cannot be told apart here, as in the first study.
- The data is administrative. Results are about method behaviour on these
  records, not about care.
- The served settings leak a little information about every state.

## Deviations

None yet.

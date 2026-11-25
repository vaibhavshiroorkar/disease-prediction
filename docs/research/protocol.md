# Study protocol: moving a disease model to a new place

Frozen on 2026-11-25, in a commit that comes before the study code. Changes
after that go in the "Deviations" section, with a date and a reason.

## Question

When a disease model moves to a hospital, region or year it never saw, what
breaks, and how few local labelled patients repair it?

## What was already known when this was written

The README reported AUC, the average prediction and the real rate for every
left-out hospital and region, and for dengue 2025. So H1 cannot be a fresh
prediction. It is **exploratory**. Nothing in this project had tried
recalibration or a local refit, so H2 to H5 are **confirmatory**. The audits
are exploratory.

## Transfers

Ten, each with a source model that never sees target rows:

| Family | Targets | Source model |
|---|---|---|
| heart (4) | Cleveland, Hungary, Switzerland, VA | logistic regression, re-tuned on the other three |
| dengue (5) | five Brazilian regions, 2023 | gradient boosting, settings fixed from the main search |
| dengue (1) | 2025, all regions | same, trained on the 2023 sample |

These are the served families, with the served sigmoid calibration. The dengue
settings were chosen on 2023 rows that include each target region. With one
small grid and 160,000 rows this leak should be small, but it is there.

## Updating methods

All work on the logit of the source probability. `n` is the number of local
patients, drawn at random and not stratified, as a clinic would meet them.

| Method | Local labels used | What it learns |
|---|---|---|
| none | 0 | nothing |
| em | 0 (n unlabelled) | the new rate, by expectation maximization (Saerens 2002) |
| intercept | n | one shift |
| platt | n | shift and slope |
| refit | n | a new logistic model, C = 1, same inputs, source ignored |
| rate | n | the local rate only, same risk for everyone, (cases + 0.5) / (n + 1) |

Intercept and platt use weak normal priors (shift sd 2.5 around 0, slope sd 1
around 1) so a sample with one class still has a finite answer. Refit on one
class falls back to the rate method. The report counts how often that happens.

## Design

Each repeat splits the target in half at random, stratified on the label. One
half is the pool that `n` patients come from. The other half is the test half.

- heart: n = 10, 25, 50; 200 repeats
- dengue: n = 100, 250, 500, 1000, 2500; 50 repeats. The test half is capped
  at 20,000 rows.

Seed 42 plus the repeat number. Methods in one repeat share the same rows, so
comparisons are paired.

## Measures

Primary: log loss on the test half, as the paired difference from `none`.
Secondary: AUC, average predicted minus observed rate, calibration slope,
and sensitivity and specificity at the source threshold.

An update **helps** a transfer at `n` if the mean difference is below 0. It
helps **reliably** if at least 90% of repeats are below 0. The repair size is
the smallest `n` where it helps reliably, there and at every larger `n` tested.

Spread over repeats describes the luck of the draw within one target. It does
not describe uncertainty about new hospitals: there are only ten transfers,
and they are not independent.

## Hypotheses and how each can fail

- **H1 (exploratory).** Rates break more often than ranking. Compare each
  target with out-of-fold source predictions, by bootstrap (1,000). Count the
  targets whose AUC drop has a 95% interval above 0, and the targets whose
  observed/expected ratio has an interval that excludes 1. Supported if the
  second count is higher.
- **H2.** A one-number shift repairs most transfers: `intercept` helps in at
  least 8 of 10, at n = 50 (heart) and n = 500 (dengue).
- **H3.** The slope costs more data than it returns at first: at the smallest
  n, `intercept` has lower mean log loss than `platt` in at least 8 of 10.
- **H4.** Reusing the old model beats starting again: `intercept` has lower
  mean log loss than `refit` at every tested n in at least 8 of 10.
- **H5.** The label-free repair is not safe: `em` at the largest n makes log
  loss worse than `none` in at least 2 of 10.

Every verdict is reported, supported or not. No significance test is claimed
across transfers.

## Audits (exploratory)

- **Missingness:** how well the pattern of missing values alone predicts the
  label, and for heart the hospital, by 5-fold cross-validated AUC. Boosted
  trees on yes/no "is missing" columns. Heart is also run on the raw 13 inputs.
- **Site shortcut:** for heart and dengue, the AUC from the site's own rate
  alone, and the out-of-fold AUC pooled against within each site.

## Limits fixed in advance

- Switzerland has 8 patients without disease. Its AUC, slope and specificity
  are shown but carry no conclusion.
- The data is old (heart, 1980s) or administrative (dengue). Results are about
  method behaviour on these records, not about care today.
- Labels can differ between sites. A "broken" model and a changed label cannot
  be told apart here.

## Deviations

None yet.

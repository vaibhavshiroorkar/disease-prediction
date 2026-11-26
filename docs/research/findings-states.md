# Findings: dengue severity, state by state

Run on 2026-11-26 by the [frozen protocol](protocol-states.md), on commit
54acac7, in 20 minutes on 12 cores. Every number is in
`results/states/report.json` (frozen tests) or `exploratory.json` (checks made
after the results were seen, by `python -m study.states.explore`). Both files
record the commit. The report also records the data hashes, seed and library
versions. **Verified** means measured in this run. **Assumed** means our
reading of it.

## Short answer

The label-free repair (EM) made predictions worse in 41 of 47 state-years.
The shift score could not tell in advance where it would fail, and the guard
built on it chose never to use EM. A labelled one-number shift helped in 37 of
47 at 500 local patients. Small local samples often hurt, and having more
severe cases in the sample helped more than having more patients, but only
partly. Of the two questions in the protocol, the answer to the second is
"not with this score". A later check found why EM failed: its estimate of the
local rate was wrong. Given the true rate, the same correction helped in 40
of 47.

## Targets

47 state-years passed the inclusion rule: 14 from 2023 (stage A), 18 from 2024
and 15 from 2025 (stage B). 34 were left out for having too few cases.
Each state's model never saw that state.

- Rates broke in most places: observed over expected was outside 0.8 to 1.25
  in 38 of 47 (verified). The model under-predicted by more than double in 13.
- Ranking moved less: AUC fell by more than 0.02 in 20 of 47 (verified). The
  worst was MS 2025, from 0.765 to 0.615.

## Verdicts

| Test | Result | Verdict |
|---|---|---|
| S1: fewer than 10 local cases hurts more often than 20 or more | 45 of 47, bar 75% | supported, but see the caveat below |
| S2: repair size varies less in cases than in patients | CV ratio 0.84, interval 0.68 to 1.02 | not supported, the interval crosses 1 |
| S3: shift score correlates with EM's harm | Spearman 0.07, interval -0.31 to 0.40 | not supported |
| S4: a guard fitted on 2023 works on 2024 and 2025 | guard threshold 0 (never EM) | not supported |
| R1: shift helps at n = 500 | 37 of 47, bar 75% | replicated |
| R2: EM worse than no update | 41 of 47, bar 25% | replicated, far past the bar |

All verified from `report.json`. Intervals resample states, keeping a state's
years together.

## What each result means

**S1 needs a caveat.** It pooled draws across every n. A draw with fewer than
10 cases is usually a small draw, so S1 mostly compares small samples with
large ones. As frozen, it cannot separate cases from patients. That is a
weakness in the protocol, found after the run.

An exploratory check keeps the state and n fixed and asks whether draws with
more cases did better. They did, in 68% of the 226 target-and-n cells (median
Spearman -0.08). The effect is clearest at n = 100 (79% of cells) and gone by
n = 2,500 (53%). So cases matter beyond n in small samples, and the effect is
modest (verified in `exploratory.json`; the reading is assumed).

**S2 failed narrowly.** Repair sizes ran from 100 to 2,500 patients, or from
3 to 80 local cases. Neither is a universal number, which agrees with Yao et
al. (see [related-work.md](related-work.md)).

**S3 and S4 failed clearly.** The shift score ranged from 0.58 to 0.79, so the
inputs differed from the source everywhere, yet the score did not track EM's
harm. In stage A, EM made every 2023 target worse, so the guard learned "never
use EM". That guard did no harm in stage B, but it could never help, and the
protocol required it to help at least once.

**Small samples hurt.** At n = 100 the shift raised mean log loss in 28 of 47
targets (verified).

## Seen after the results, not tested

These are hypotheses for a next study. They were not frozen in advance.

- **Repair only pays when the model is off.** The 8 targets that no sample
  size repaired all had observed over expected between 0.85 and 1.12. All 39
  that were repaired lay outside that band. Within the band, there was little
  to fix and the update added noise (verified in `exploratory.json`, reading
  assumed). Checking the band needs labels, so this is not a label-free rule.
- **EM helped only under large under-prediction.** It helped in 6 targets, all
  with observed over expected above 2.1. It helped in none of the 34 with a
  ratio of 2 or less (verified). EM may only be worth its risk when the rate
  shift is large.

## Is it the estimate or the correction?

EM does two things. It estimates the local rate from unlabelled predictions,
then shifts every prediction toward that rate. A later check,
`python -m study.states.true_rate`, gave the same shift the true local rate instead.
It reused the saved test halves and EM samples. All 47 refitted source
models matched the saved run to within 0.0001 in log loss (verified).

The decision rule was committed before the first run (1196e55). If the true
rate repaired at least 75% of the targets where EM hurt, the estimate failed.
It repaired 34 of 41, so **the estimate failed** (verified in
`results/states/true_rate.json`).

- EM's rate was wrong by a lot. Its median estimate was 1.56 times the true
  rate. It was more than double in 19 of 47 targets and under half in 12.
  It was within 0.8 to 1.25 of the truth in only 4 (verified).
- EM's harm tracked that error closely: Spearman 0.91 between how far off
  the rate was and how much log loss rose (verified).
- The correction with the true rate helped, but only a little: a median log
  loss change of -0.002. EM's median harm was +0.040 (verified).
- Shifting predictions until their mean matched the true rate helped in 46 of
  47 (verified). It does not assume the model was calibrated to the source
  rate on these patients, which may be why it beat the prior formula (40 of
  47). That reading is assumed.
- With 2,500 labelled patients, the one-number labelled shift (intercept) did
  about as well as the true-rate shift. It beat it in 21 of 47, and both had
  a median change near -0.002 (verified). So most of what local labels buy
  here is the rate itself.

What this means (assumed): the reweighting step is safe under this shift, if
the rate is right. EM's estimate is what breaks. When the inputs shift, the
model's scores move for reasons other than the rate, and EM reads that
movement as a rate change. A better label-free estimate of the rate would be
worth more than a better correction. Even a few labelled patients give one.

This check was planned after the results were seen, so it explains the
failure but does not test a frozen hypothesis.

## Limits

- 47 targets from 27 states over three years share most of their training
  rows, so they are not independent. The state-level bootstrap helps, but it
  cannot fix that.
- S1 was badly specified, as above.
- The smoke run showed 4 targets' numbers before the full run. No rule
  changed after it (see the protocol).
- States may classify severity differently. A label that changed and a model
  that broke cannot be told apart.
- The data is administrative. These results describe method behaviour on
  these records, not care.

## Is this a paper now?

It is closer. The literature check (see [related-work.md](related-work.md))
found that EM failing under input shift is expected and has been shown on
images. What remains is narrower. It is a pre-registered measurement on
tabular clinical data with natural shift: EM hurt in 87% of new state-years,
and a covariate-shift score gave no warning. That suits a workshop or student
venue as a negative result, not a journal as a new method (assumed, not
checked against any venue's criteria). The true-rate check adds a mechanism:
the rate estimate fails and the correction does not. That matches Chandra H et
al. (2026), who corrected with the known rate and saw calibration improve. A
fresh frozen test of the post-hoc patterns and of this finding, for example on
2026 data when it is published, would strengthen it most.

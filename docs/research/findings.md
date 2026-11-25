# Findings: moving a disease model to a new place

Run on 2026-11-25 by the [frozen protocol](protocol.md). Every number is in
`results/transfer/report.json` or `audits.json`, which also record the code
commit, data hashes, seed and library versions. Labels follow the project
rule: **verified** means measured in that run, **assumed** means our reading of it.

## Short answer

Probabilities usually broke. Ranking mostly survived. A one-number shift,
learned from local patients, repaired 8 of 10 targets at the sizes fixed in
advance. It needed 10 to 50 patients for three of the heart hospitals and 250
to 2,500 for dengue. It did nothing useful at VA Long Beach, where the rate
was already right and the ranking was what broke.

## What broke

| Target | AUC at home | AUC there | Predicted | Real | Slope |
|---|---|---|---|---|---|
| heart: Cleveland | 0.869 | 0.868 | 54.4% | 45.9% | 1.13 |
| heart: Hungary | 0.852 | 0.887 | 50.2% | 36.2% | 1.16 |
| heart: Switzerland | 0.881 | 0.729 (8 negatives) | 62.9% | 93.5% | 0.77 |
| heart: VA | 0.887 | 0.751 | 71.7% | 74.4% | 0.67 |
| dengue: Center-West | 0.763 | 0.779 | 2.4% | 1.9% | 1.08 |
| dengue: North | 0.765 | 0.837 | 2.3% | 1.6% | 1.30 |
| dengue: Northeast | 0.760 | 0.786 | 3.5% | 5.3% | 0.96 |
| dengue: South | 0.778 | 0.741 | 1.9% | 3.4% | 0.91 |
| dengue: Southeast | 0.771 | 0.758 | 3.1% | 1.9% | 0.96 |
| dengue: 2025 | 0.766 | 0.734 | 2.6% | 3.9% | 0.87 |

A slope below 1 means the predictions are too extreme there.

## Verdicts

| Hypothesis | Result | Verdict |
|---|---|---|
| H1, exploratory: rates break more often than ranking | rates 9 of 10, ranking 3 of 10 (VA, South, 2025) | supported |
| H2: a shift helps at n = 50 / 500 | 8 of 10, bar 8 | supported, exactly at the bar |
| H3: at the smallest n, shift beats shift and slope | 10 of 10, bar 8 | supported |
| H4: shift beats a new local model at every n | 10 of 10, bar 8 | supported |
| H5: EM makes at least 2 targets worse | 7 of 10, bar 2 | supported |

H2 fails at VA (0.517 against 0.511 with no update) and Center-West (0.0843
against 0.0838). One more failure would have flipped the verdict.

## How many local patients

Smallest n where the shift helps in at least 90% of draws, there and above:

| Target | n | Expected cases among them |
|---|---|---|
| heart: Switzerland | 10 | 9 |
| heart: Hungary | 25 | 9 |
| heart: Cleveland | 50 | 23 |
| heart: VA | never, up to 50 | |
| dengue: Northeast | 250 | 13 |
| dengue: South | 500 | 17 |
| dengue: Southeast | 500 | 10 |
| dengue: North | 1,000 | 16 |
| dengue: 2025 | 1,000 | 39 |
| dengue: Center-West | 2,500 | 48 |

The patient counts differ 250-fold but the cases stay between 9 and 48.
Counting cases may be the better way to plan a sample. This was seen after the
run, so it is a hypothesis for the next study and not a tested result.

Small samples also hurt. At the smallest n the shift made log loss worse on
average in 8 of 10 targets. The smaller the original error, the more data it
took to beat leaving the model alone.

## The label-free repair fails without warning

EM landed near the truth in Switzerland (0.939 against 0.935 real). At VA it
went to 100% and log loss rose from 0.51 to 3.14. In the Northeast it estimated
32% against 5.3% real. A direct grid search over the likelihood gave the same
answers, so this is the method and not our code (**verified**;
`tests/test_study.py` keeps the check). Its assumption, that patients in each
class look the same everywhere, does not hold in these targets (**assumed**
cause). From unlabelled data alone, a clinic could not tell the good case from
the bad one.

## Audits

- **Site shortcut, mostly not there.** The hospital's rate alone scores AUC
  0.704 on heart. But the model's pooled AUC is 0.874 and its within-site AUC
  is 0.861, so mixing hospitals adds about 0.013. On dengue it adds 0.003.
  We expected a bigger effect on heart and did not find it.
- **Missing values identify the hospital.** On all 13 raw heart inputs, the
  missing pattern alone predicts the hospital at AUC 0.948 and the label at
  0.741. With the project's 10 inputs these fall to 0.755 and 0.614. Dropping
  `slope`, `ca` and `thal` helped, and did not remove the leak. The served
  model gets no missing-value columns, though imputed values can still carry
  some of it (**unknown** how much).
- On diabetes, liver and dengue the missing pattern predicts nothing (0.48 to 0.51).

## What these results do not show

- **Ten targets are few, and related.** The five regions share one data system,
  and each target's training data overlaps the others'. Repeats measure the
  luck of the draw within a target.
- **H1 favours rates.** Large dengue samples flag tiny rate gaps, while the
  small hospitals can hardly show an AUC drop. The sizes matter more than the count.
- **H4 covers small samples only.** Refit kept improving. In the South at 2,500
  it beat no update in 72% of draws, and it may pass the shift with more data.
  A lasso or boosted refit was not tried.
- **Dengue settings saw the target regions.** The protocol records this leak.
- **A changed label looks like a broken model.** Regions may record "warning
  signs" differently. This data cannot separate the two.
- **Lower log loss is not better care.** No decision or outcome was studied.
- **A repaired model is not ready to use.** The repair fixes the average. It
  cannot fix ranking, which is what failed at VA.

## Next questions

Open ideas live in [../PENDING.md](../PENDING.md).

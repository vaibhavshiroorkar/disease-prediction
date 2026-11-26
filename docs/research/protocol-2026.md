# Study protocol 3: a frozen retest on 2026

Frozen on 2026-11-26, before the 2026 file was downloaded and before the code
that will analyse it was written. Changes after that go in "Deviations", with
a date and a reason.

The state study ([protocol-states.md](protocol-states.md),
[findings-states.md](findings-states.md)) produced findings that were seen
before they were tested. This protocol turns each of them into a rule with a
pass mark, and tests them on a year no model or person here has looked at.

## What was looked at before freezing

- Every result of the state study on 2023, 2024 and 2025, including the
  exploratory checks and the true-rate check (`results/states/`).
- The 2026 file's size and date from the server's headers (15,864,337 bytes,
  last modified 2026-11-25). No row of it was read.
- Not yet the BBSE check (`study/states/bbse.py`). Its decision rule was
  committed before this protocol, and its result was not known when this was
  written. T7 below does not depend on which way it came out.

## Data and targets

- SINAN dengue 2026 (`DENGBR26.csv.zip`) from the same OpenDataSUS address as
  the other years. Its SHA-256 goes in "Recorded before analysis", committed
  before any model scores a 2026 row.
- Targets: 2026 state-years that pass the state study's inclusion rule, at
  least 1,000 lab-confirmed cases and at least 100 severe cases.
- If fewer than 8 states pass, every test below is reported as "not
  testable" and only counts are given.

## Procedure

Exactly the state study's, with 2026 as the target year:

- Source models: each state's model trained on every other state's 2023 cases,
  as before. These fits are deterministic, so they are the same models.
- Per target: the shift score, 50 repeats at each size the pool allows, and
  the six updating methods. Then, at the target's largest n and on the same
  draws, the true-rate correction (`true_prior`) and the BBSE estimates.
- The code is written after this protocol and committed before the download.
  It reuses `study/states/` and must reproduce the saved 2023 to 2025 numbers
  in its tests or checks before it runs on 2026.

## Hypotheses and how each can fail

"Helps" means a lower mean log loss than no update over the 50 repeats at the
target's largest n. "Hurts" means higher.

| Test | Claim, from the earlier finding | Supported when | Not testable when |
|---|---|---|---|
| T1 | EM hurts in most new state-years (41 of 47 before) | EM hurts in at least 75% of targets | |
| T2 | EM's correction works with the true rate (34 of 41 before) | Among targets where EM hurts, `true_prior` helps in at least 75% | EM hurts in fewer than 4 |
| T3 | EM's harm tracks how wrong its rate is (Spearman 0.91 before) | Spearman of abs(log(EM's median rate / true rate)) against EM's change is at least 0.5, and its 95% bootstrap interval stays above 0 | |
| T4 | A labelled repair pays only when the model is off (8 of 8 unrepaired had O/E 0.85 to 1.12) | With "inside" meaning O/E between 0.85 and 1.12: the share of outside targets with a repair size is at least 0.25 above the share of inside targets with one | Fewer than 3 inside or 3 outside |
| T5 | EM helps only under large under-prediction (0 of 34 with O/E at most 2) | Among targets with O/E at most 2, EM helps in at most 10% | Fewer than 5 such targets |
| T6 | More local cases help within a fixed n (68% of cells before) | Over (target, n) cells with more than 2 distinct case counts, the share with a negative Spearman of local cases against the shift's log-loss change is at least 0.6 | Fewer than 20 cells |
| T7 | The BBSE result generalizes | The BBSE decision rule in `study/states/bbse.py` gives the same answer on 2026 as on 2023 to 2025 | |

T6 replaces the state study's S1, which pooled sizes and so mostly compared
small samples with large ones.

## Uncertainty

Each state appears once in 2026, so the bootstrap for T3 resamples targets,
1,000 times, seed 42. Counts are reported with the targets that make them up.

## Limits fixed in advance

- The 2026 file is preliminary. Cases still under investigation have no final
  class and drop out of the confirmed filter. If severe cases are classified
  later than mild ones, the 2026 severe rate is too low, and that looks like a
  rate shift. The test reports the share of rows with no final class.
- The source models still train on 2023. A shift found in 2026 mixes three
  years of change with geography.
- Every finding tested here came from the same people and code. A pass is a
  replication on new data, not an independent one.

## Deviations

- 2026-11-27: added 2026 to `DENGUE_STUDY_YEARS` in `ml/config.py`, so the
  downloader fetches the file and the provenance hashes it. The protocol
  named this file from the start; no analysis code or pass mark changed.

## Recorded before analysis

Recorded on 2026-11-27, after the download and before any model scored a
2026 row. Reading the file to count cases and apply the inclusion rule is all
that was done with it.

- File: `DENGBR26.csv.zip`, saved as `data/raw/dengue_sinan_2026.csv.zip`,
  15,864,337 bytes (the size the server's headers gave before freezing).
- SHA-256: `0c9ba29ff33573431b5198949c982a285b7233f26c09ef81dbb068460775e909`.
- Rows: 489,197 notifications. 140,693 are lab-confirmed dengue under the
  study's filter, 6,518 of them severe (4.6%).
- No final class yet: 6.11% of rows.
- Targets: 15 states pass, so every test is testable on count. They are PA,
  TO, MA, PI, CE, RN, PE, BA, MG, ES, RJ, SP, PR, MT and GO. The nearest
  misses were PB (2,341 cases, 85 severe) and AL (1,990, 64).
- Analysis code: `study/states/retest.py` as committed in 027f362, unchanged.
  It runs on the commit that records this section.

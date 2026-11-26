# Changelog

Semantic versions, newest first.

## Unreleased

- Froze the protocol for the state-by-state dengue study before any of its
  code: four hypotheses and a held-out test of an EM guard.
- Added SINAN 2024 as a study-only download (187 MB). With it the state study
  has 47 targets. The served model still uses 2023 and 2025 only.
- State study code, now in `study/states/`, with tests.
  It reads SINAN in chunks, fits one source model per state, writes the EM
  guard before stage B and resumes after a stop. Every draw in both studies
  now records its local severe cases.
- Ran the state study (47 targets, 20 minutes). S1 and both replications
  passed. S2, S3 and S4 did not. EM hurt in 41 of 47. Findings, exploratory
  checks (`study/states/explore.py`) and results committed.
- Checked the label-shift literature. EM failing under input shift is not
  new (Roschewitz et al. 2024). The scale on clinical data and the failed
  warning score may be.
- True-rate check (`study/states/true_rate.py`, exploratory, decision rule committed
  before the run). The same correction with the true local rate helped in 40
  of 47 targets, and in 34 of the 41 where EM hurt. EM fails at estimating
  the rate, not at correcting for it.
- Split `study/` into `shared/`, `transfer/` and `states/`, and grouped tests
  by area. The BBSE check reproduced every saved number on the new code.
- BBSE check (`study/states/bbse.py`, rule committed before the run): a second
  label-free rate estimate helped in 0 of 47 targets (soft version 9 of 47).
  The shared assumption fails, not one algorithm.
- Froze a retest protocol for 2026 (`docs/research/protocol-2026.md`) and
  committed its code (`study/states/retest.py`) before downloading the data.
- Lab page: a state-by-state section with the four repairs, a chart of rate
  error against harm, the six verdicts and every target.
- Read Chandra H et al. (2026) in full and the Esuli et al. abstract.
- Local CI uses installed Edge on Windows, so the pre-push hook passes.
- Compared with the closest published work. Withdrew the claim that the local
  sample-size question was new, and recorded what could still make a paper.

## 1.1.0 (2026-11-25)

- Transfer study: protocol frozen before the code, ten transfers, six updating
  methods, two shortcut audits, results with commit and data hashes.
- Lab page: cutoff explorer, score, ranking and calibration charts, transfer
  and repair charts, verdicts and audits. Works on phones and in dark mode.
- Documentation: handbook, findings, related work, model card, security,
  decision log, pending list and this changelog. The README was split up.
- Local CI runner and `pre-push` hook. `BROWSER_CHANNEL` picks an installed browser.

## 1.0.0

The six models, API, website and tests as they stood before this log. This
version was never tagged.

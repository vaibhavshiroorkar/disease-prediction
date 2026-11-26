# Changelog

Semantic versions, newest first.

## Unreleased

- Froze the protocol for the state-by-state dengue study before any of its
  code: four hypotheses and a held-out test of an EM guard.
- Added SINAN 2024 as a study-only download (187 MB). With it the state study
  has 47 targets. The served model still uses 2023 and 2025 only.
- State study code: `study/states.py` and `study/run_states.py`, with tests.
  It reads SINAN in chunks, fits one source model per state, writes the EM
  guard before stage B and resumes after a stop. Every draw in both studies
  now records its local severe cases.
- Ran the state study (47 targets, 20 minutes). S1 and both replications
  passed. S2, S3 and S4 did not. EM hurt in 41 of 47. Findings, exploratory
  checks (`study/explore_states.py`) and results committed.
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

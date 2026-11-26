# Handbook

Start here. Each fact lives in one place and this page links to it.

## The project in one minute

Six models predict disease from public data, served by one API and website.
The models are the easy part. The study asks what happens when such a model
moves to a hospital, region or year it never saw, and how many local patients
repair it. The answer is in [research/findings.md](research/findings.md).

## Where things are

| I want | Go to |
|---|---|
| To run or test it | [README](../README.md) |
| The question, rules and pass marks | [research/protocol.md](research/protocol.md) |
| The state-by-state study, frozen before its code | [research/protocol-states.md](research/protocol-states.md) |
| Its results | [research/findings-states.md](research/findings-states.md) |
| The results and their limits | [research/findings.md](research/findings.md) |
| What others found | [research/related-work.md](research/related-work.md) |
| Each model's data, scores and limits | [model-card.md](model-card.md) |
| The datasets and their defects (data card) | [data/README.md](../data/README.md) |
| API fields | [api.md](api.md) |
| Hosting | [deployment.md](deployment.md) |
| Stored predictions and reviews | [reviewing.md](reviewing.md) |
| Weaknesses and countermeasures | [SECURITY.md](SECURITY.md) |
| Why a choice was made | [DECISIONS.md](DECISIONS.md) |
| What is open, what is done | [PENDING.md](PENDING.md), [CHANGELOG.md](CHANGELOG.md) |

## How the parts connect

```
data/raw --> ml/ (train) --> models/*.joblib --> app/ (API and website)
   |                              |                  ^
   |                              v                  |
   +--> study/transfer/ --> results/transfer/ --> study/lab_data.py --> app/static/lab-data.json
   +--> study/states/   --> results/states/   ----^
        (both use study/shared/: draws, updates, provenance)
```

Tests mirror the code: `tests/app/`, `tests/ml/`, `tests/study/`, and
`tests/browser/` for the website.

`app/` never imports `ml/` or `study/`.

## Ideas you need

- **Ranking (AUC):** does a sick patient get a higher score than a well one?
- **Calibration:** of patients given 30%, do 30 in 100 have the disease?
  A model can rank well and still be wrong here.
- **Shift:** adding one number to every score on the logit scale. It moves the
  average and never changes the ranking.
- **Slope:** how spread out the scores should be. Below 1 means too extreme.
- **Log loss:** one score for honest probabilities. Confident mistakes cost most.

## Reproduce the results

```powershell
python -m scripts.download_data
python -m study.transfer.run
python -m study.lab_data
python -m study.states.run
python -m pytest
```

`study.states.run` is the state study. It saves each target as it finishes and
resumes from there. `--smoke DIR` runs four targets with three repeats to check
the code. Three checks run after it, on the same draws, and stop unless they
reproduce its saved numbers:

- `study.states.true_rate`: EM's correction given the true local rate (6 minutes).
- `study.states.bbse`: a second label-free rate estimate, BBSE (about 25 minutes).
- `study.states.retest`: the frozen 2026 retest. Needs the 2026 file.

`study.transfer.run` took 12 minutes on 12 cores. It refuses to run on uncommitted code,
and records the commit and a SHA-256 of each raw file. Use the library versions
named in the report. Another scikit-learn may give slightly different numbers.

## Mistakes log

| What went wrong | How it was caught | Cost | Fix |
|---|---|---|---|
| The first EM results looked like a bug (log loss 3.1 at VA) | Checked against a separate grid search | one check | It was real. The grid check is now a test |
| The lab page scrolled sideways on phones | The browser test | one fix | Grid columns may shrink; tables scroll inside their box |
| Chart text was unreadable on phones and "100%" was cut off | Looking at screenshots | one fix | Narrower drawing on phones, wider margin |
| Local browser tests could not start: Chromium not installed | The run failed, and later a pre-push hook | one change, then one push retry | `scripts/ci_local.sh` uses installed Edge on Windows |
| The findings draft said small samples hurt in 7 of 10 targets | Recounting from the report: 8 | one fix | Recount each claim from `report.json` before committing |
| State study S1 pooled draws across every n, so "few cases" mostly meant "few patients" | Reading the result critically after the run | S1 cannot answer its question | An exploratory within-n check; the next protocol compares within each n |

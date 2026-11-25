# Disease Prediction

Six machine learning models behind one API and website, and a study of what
happens when such a model moves to a place it never saw.

**Live demo:** [disease-prediction-f4qa.onrender.com](https://disease-prediction-f4qa.onrender.com)
(free plan, so the first request after a quiet spell takes about a minute).
For learning, not diagnosis.

## The study

We trained heart and dengue models without one hospital, region or year, then
tested them there. Ten targets. The [protocol](docs/research/protocol.md) and
its pass marks were committed before the code.

- **Probabilities broke in 9 of 10 targets. Ranking broke in 3.** In Switzerland
  the heart model predicted 63% on average and 93% had the disease.
- **One number, learned locally, repaired 8 of 10**, exactly the bar set in
  advance. It took 10 to 50 patients for three of the heart hospitals and 250
  to 2,500 for dengue to help reliably.
- **It could not repair VA Long Beach**, where ranking broke and not the rate.
- **The repair that needs no labels made 7 of 10 worse**, with no warning sign.

Full numbers and limits: [findings](docs/research/findings.md). The site's Lab
page lets you move each model's cutoff and explore these results.

## The models

| Model | Data | Test AUC (95% interval) | At its cutoff |
|---|---|---|---|
| heart | UCI, 918 patients, 4 hospitals | 0.866 (0.808 to 0.918) | catches 87%, specificity 68% |
| diabetes | Pima, 768 women | 0.815 (0.744 to 0.877) | catches 89%, specificity 60% |
| liver | Indian patients, 570 | 0.817 (0.739 to 0.887) | catches 89%, specificity 33% |
| dengue severity | Brazil 2023, 200,000 cases | 0.776 (0.760 to 0.792) | catches 90%, specificity 35% |
| malaria cell | NIH, 27,558 cells, 200 patients | 0.9915 (0.9882 to 0.9942) | catches 93%, specificity 98% |
| symptoms | Kaggle, 304 distinct patterns | 100% accuracy | misleading: the data is too easy |

How each was trained, and why the symptom score means little: [model card](docs/model-card.md).

## Run locally

Use Python 3.12 or later. The trained models are included, so you can run the
API without downloading data or retraining.

On Windows, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements-runtime.txt
python -m uvicorn app.main:app --reload
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-runtime.txt
python -m uvicorn app.main:app --reload
```

Then open http://localhost:8000 for the web interface. Interactive API docs are
at `/docs`, and service information is at `/api`.

`requirements-runtime.txt` is the hash-locked set the Docker image and CI use.
Training and the tests need more; `requirements-dev.txt` adds them (see Tests).

For hosting, see [docs/deployment.md](docs/deployment.md). The included Docker
image runs the frontend and API together and reads the host's `PORT` setting.

To download the source data and retrain all six models:

```bash
python -m scripts.download_data
python -m ml.train --model all
```

The download is about 450 MB, mostly malaria images and two years of dengue
records. Training time depends on your machine. Dengue takes much of that time
because its nested cross-validation runs on 160,000 training rows.

## Test it

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

Before a push, `bash scripts/ci_local.sh` runs the CI steps. Turn on the hook
that calls it with `git config core.hooksPath .githooks`. Browser tests need
Playwright's Chromium, or `BROWSER_CHANNEL=msedge` for installed Edge.

Database tests need `TEST_DATABASE_URL` set to a disposable Postgres database,
because they empty its prediction tables. Without it they skip.

## Repository layout

```
app/     the API. reads models/, never imports ml/
  static/         frontend HTML, CSS and JavaScript
  security.py     administrator keys, request limits and privacy headers
  cells.py        turns a blood cell image into the 181 numbers the malaria model reads
  db.py           postgres logging, review queues, feedback for retraining
  registry.py     loading the artifacts and refusing to load a bad one
  routers/risk.py the risk endpoints and their explanations
study/   the transfer study. run.py writes results/, lab_data.py feeds the lab page
results/ study outputs, with the commit and data hashes behind them
ml/      training
  datasets.py       symptom data loading and cleaning
  clinical_data.py  heart, diabetes, liver and dengue loading, cleaning and input specs
  malaria.py        malaria images, patient-grouped training, the example figure
  model.py          the served symptom model and reference classifiers
  clinical_model.py the four candidate families and the selection rule
  evaluate.py       symptom scoring and the noise tests
  clinical_eval.py  risk scores, bootstrap intervals, threshold, calibration
  train.py          run this to produce everything in models/
models/  the trained artifacts and a <name>_summary.json for each
docs/figures/  ROC, precision-recall and calibration plots, redrawn by training
tests/   API, data cleaning, model evaluation and database tests
```

The API loads joblib artifacts without importing the training package. It checks
their `artifact_version` before using them. Symptom retraining imports `app.db`
inside the feedback loader, so ordinary training does not need a database.
Malaria training imports `app.cells` to share the serving feature extractor.
The malaria route rejects artifacts whose feature version does not match the code.

The tabular request schemas are written by hand for readable API docs. A test
checks that their fields and ranges match the artifacts.

## More

- [Handbook](docs/HANDBOOK.md): the entry point, with the mistakes log.
- [API reference](docs/api.md), [deployment](docs/deployment.md), [security](docs/SECURITY.md).
- [Data card](data/README.md), [related work](docs/research/related-work.md).
- [Decisions](docs/DECISIONS.md), [pending](docs/PENDING.md), [changelog](docs/CHANGELOG.md).

Licensed under [MIT](LICENSE). The datasets belong to their publishers.

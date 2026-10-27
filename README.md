# Disease Prediction System

Live demo: [disease-prediction-f4qa.onrender.com](https://disease-prediction-f4qa.onrender.com).
It runs on Render's free plan, so the first request after a quiet spell takes
about a minute to wake the server.

This project serves six machine learning models through one FastAPI service. You can
send a list of symptoms and get a ranked list from 41 conditions. You can also
submit clinical measurements for heart disease, diabetes and liver disease risk
estimates. The responses include model scores, the inputs that affected the
result, and any missing measurements the model filled in.

The dengue model estimates severity among patients who already have confirmed
dengue. It trains on a sample of Brazil's 2023 case records. The malaria model
classifies images of individual red blood cells as parasitized or uninfected.

The symptom model uses a small Kaggle dataset. Its perfect test scores say more
about the dataset than its ability to handle real symptoms. The other models
use patient records or microscope images, with separate tests for calibration,
data leakage and performance across hospitals or years.

The web interface has a home page, a checks page for all six models, a models
page with each model's data and test results, and an about page on reading
results and privacy. Checks cover searchable symptoms, clinical forms and blood
cell image uploads, and each result shows the model's cutoff, its test accuracy
and any inputs it filled in. FastAPI serves the interface and API from the same
address.
Postgres logging is optional; when enabled, it stores predictions for later review.

## Run locally

Use Python 3.12 or later. The trained models are included, so you can run the
API without downloading data or retraining.

On Windows, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Then open http://localhost:8000 for the web interface. Interactive API docs are
at `/docs`, and service information is at `/api`.

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

The following request examples use Bash-style quoting. On Windows, you can
make the same requests through `/docs`.

```bash
curl -X POST http://localhost:8000/predict/heart \
  -H "Content-Type: application/json" \
  -d '{"age": 61, "sex": 1, "cp": 4, "thalach": 115, "exang": 1, "oldpeak": 2.5}'
```

```bash
curl --data-binary @cell.png -H "Content-Type: image/png" http://localhost:8000/predict/malaria
```

```bash
curl -X POST http://localhost:8000/predict/symptoms \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"], "top_k": 3}'
```

### Enable prediction logging

The API can make predictions without a database. To store them, start the local
Postgres container and set `DATABASE_URL` in the same shell as the API.

In PowerShell:

```powershell
docker compose up -d
$env:DATABASE_URL = "postgresql://dps:devpass@localhost:5432/disease_predictions"
python -m uvicorn app.main:app --reload
```

In Bash:

```bash
docker compose up -d
export DATABASE_URL=postgresql://dps:devpass@localhost:5432/disease_predictions
python -m uvicorn app.main:app --reload
```

The API creates its tables on startup. If it cannot connect to Postgres, it logs
a warning and keeps serving predictions without recording them.

`docker-compose.yml` starts only the development database.
`compose.production.yml` runs the application. History, review and statistics
routes require an administrator key in the `X-API-Key` header. Set `ADMIN_API_KEY`
on the server to enable them; they return 503 when no key is configured.

## Model results

| Model | Data | Size | Test AUC (95% CI) | At its threshold |
|---|---|---|---|---|
| heart | UCI Heart Disease, 4 hospitals | 918 patients | 0.866 (0.808 to 0.918) | sensitivity 87%, specificity 68% |
| diabetes | Pima Indians Diabetes | 768 patients | 0.815 (0.744 to 0.877) | sensitivity 89%, specificity 60% |
| liver | Indian Liver Patient Dataset | 570 patients | 0.817 (0.739 to 0.887) | sensitivity 89%, specificity 33% |
| dengue | Brazil's SINAN register, 2023 | 200,000 confirmed cases | 0.776 (0.760 to 0.792) | sensitivity 90%, specificity 35% |
| malaria | NIH thin smear images | 27,558 cells, 200 patients | 0.9915 (0.9882 to 0.9942) | sensitivity 93%, specificity 98% |

AUC measures how well a model ranks positive cases above negative ones. A score
of 0.5 is chance-level ranking; 1.0 means perfect separation on that test set.
Sensitivity is the share of positive cases flagged. Specificity is the share of
negative cases left unflagged. The first three models have wide confidence
intervals because their test sets contain only 114 to 184 patients.

The symptom model reaches 100% accuracy on 304 unique patterns. That is a
multiclass accuracy score, so it is separate from the binary AUC results above.
The [symptom model section](#the-symptom-model) explains why that score is misleading.

## Training and evaluation

Heart, diabetes, liver and dengue share a training pipeline. Malaria uses a
similar process with image features and patient-based splits.

1. Clean the data and log the changes. Each loader handles the defects listed
   in the dataset sections below.
2. Set aside 20% for testing before choosing or tuning a model.
3. Compare a prevalence-only baseline, logistic regression, random forest and
   gradient boosting on the remaining 80%. The outer cross-validation uses
   5 folds repeated 3 times, with a tuning search inside each fold. Dengue skips
   the forest and uses one outer repeat to keep training time manageable.
4. Choose the model family with the best mean AUC. Prefer logistic regression
   if its score is within half the best model's standard deviation.
5. Calibrate the probabilities. The aim is for a group assigned 30% risk to
   have an observed positive rate near 30%. Calibration still needs checking
   on the test set and on any new population.
6. Pick the highest threshold that reaches 90% sensitivity on out-of-fold
   training predictions. The test set does not choose the threshold.
7. Evaluate on the test set, with 1,000 bootstrap resamples for confidence intervals.

The pipeline fills missing values using medians from its training rows. Each
cross-validation fold fits its own imputer, so validation rows do not affect
those medians.

### Thresholds and false alarms

The training target prioritizes sensitivity: flag 9 in 10 positive cases, then
measure how many negative cases also get flagged. The 90% target is a project
choice, not a clinically validated operating point. It also does not guarantee
90% sensitivity on new data. The results above show the false-alarm tradeoff
for each model.

### Comparing model families

Logistic regression, random forest and gradient boosting finished within about
one standard deviation of each other on the three smaller datasets:

| | heart | diabetes | liver |
|---|---|---|---|
| logistic regression | 0.879 +/- 0.027 | **0.837** +/- 0.036 | **0.729** +/- 0.047 |
| random forest | **0.879** +/- 0.026 | 0.833 +/- 0.038 | 0.718 +/- 0.040 |
| gradient boosting | 0.877 +/- 0.022 | 0.832 +/- 0.028 | 0.701 +/- 0.046 |
| dummy (floor) | 0.500 | 0.500 | 0.500 |

These are mean cross-validated AUCs, with standard deviations across 15 folds.
On heart disease, the forest leads by 0.0008, so the selection rule chooses
logistic regression. It also chooses logistic regression for diabetes and liver
disease. These training splits contain roughly 450 to 730 rows and 8 to 10 inputs;
the more complex models did not produce a clear gain.

Gradient boosting performed better on dengue and malaria:

| | dengue | malaria |
|---|---|---|
| gradient boosting | **0.764** +/- 0.005 | **0.990** +/- 0.004 |
| logistic regression | 0.740 +/- 0.007 | 0.970 +/- 0.008 |
| dummy (floor) | 0.500 | 0.500 |

On dengue, boosting leads logistic regression by 0.024 AUC, compared with a
fold-to-fold standard deviation of about 0.005. The larger gap supports choosing
boosting for this dataset. It does not establish that boosting will always win
on larger datasets.

### Log-transforming lab values

Liver enzyme measurements are skewed. An ALT value of 2,000 can have much more
influence on a linear model than a typical value around 35. A log transform
reduces that difference:

| | liver | diabetes |
|---|---|---|
| logistic, raw values | 0.716 | 0.834 |
| logistic, skewed values logged | **0.729** | **0.837** |

The transform adds about 0.013 AUC on liver and 0.003 on diabetes, where it
applies to insulin and the family-history score. On liver, that moves logistic
regression from below the forest to above it. Training keeps an untransformed
version in the comparison table so the effect stays visible.

## Heart disease

The heart model uses records from 918 patients at four sources: the Cleveland
Clinic, the Hungarian Institute of Cardiology, the Zurich and Basel university
hospitals (pooled as one Swiss source), and the Long Beach VA. The target is
whether an angiogram found more than 50% narrowing in a major vessel.

![heart curves](docs/figures/heart_curves.png)

On the test set, it flags 89 of 102 patients with disease and misses 13. It also
flags 26 of the 82 without disease. Its expected calibration error is 0.04,
measured across probability bins. That average does not guarantee an accurate
probability for an individual patient. Chest pain type has the strongest
effect, followed by exercise-induced ST depression and angina.

Every Swiss record and 49 VA records list cholesterol as 0. The loader treats
those values as missing. The `slope`, `ca` and `thal` columns are nearly complete
at Cleveland but 50 to 99% missing elsewhere. They are excluded because their
missingness can identify the hospital. Cholesterol stays, but without a
missing-value indicator, for the same reason.

A separate evaluation trains on three sources and tests on the fourth. This
checks how well the model transfers to a hospital outside its training data:

| Left-out hospital | Patients | Real rate | Predicted on average | AUC |
|---|---|---|---|---|
| Cleveland | 303 | 46% | 54% | 0.87 |
| Hungary | 293 | 36% | 50% | 0.89 |
| Switzerland | 123 | 94% | 63% | 0.73 |
| VA | 199 | 74% | 72% | 0.75 |

Ranking stays close to the original test score at Cleveland and Hungary. AUC
drops to 0.73 at Switzerland and 0.75 at the VA. For Switzerland, the average
predicted risk is 63%, while the observed rate is 94%. The model's probabilities
transfer poorly even where it retains some ranking ability. A new site would
need its own validation and calibration.

## Diabetes

The diabetes model uses records from 768 women of Pima heritage, aged 21 and
over. Its target is the dataset's diabetes outcome.

![diabetes curves](docs/figures/diabetes_curves.png)

It flags 48 of the 54 positive cases and leaves 60 of the 100 negative cases
unflagged. Glucose is the strongest input, with about five times the importance
of BMI, the next input. The threshold is about 0.21. Only 55% of flagged cases
are positive, while 91% of unflagged cases are negative. The flag reflects the
chosen sensitivity target; it is not a diagnosis.

Five measurement columns use 0 for missing data. Insulin is 0 in 374 records,
and skin thickness is 0 in 227. The loader replaces these values with missing
values before the pipeline fits its imputer.

The training data covers only Pima women. The API has no sex field for this
model and does not enforce that population limit. It will return a score for
inputs from other groups, despite having no validation for them.

## Liver disease

The liver model uses 570 patient records from Andhra Pradesh. It predicts the
dataset's label: liver patient or non-patient.

![liver curves](docs/figures/liver_curves.png)

This is the weakest of the three smaller clinical models. Its cross-validated
AUC is 0.73, compared with 0.82 on the 114-patient test set. The test interval
runs from 0.74 to 0.89. That uncertainty makes the higher test score a poor
basis for claiming a large improvement over cross-validation.

At its threshold, it flags 89% of positive cases but also 22 of the 33 negative
cases. The precision-recall score is 0.93, against a prevalence baseline of
about 0.71. Expected calibration error is 0.10, the highest of the three smaller
clinical models.

These ten measurements give limited separation, particularly at a threshold
that prioritizes sensitivity. The false-positive rate remains a limitation.

The loader removes 13 exact duplicate rows before splitting. It also maps the
original labels, 1 for patient and 2 for non-patient, to 1 and 0.

## Dengue

The dengue model predicts severity rather than diagnosis. The public SINAN
files used here contain about 1.6 million notifications per year, but omit
cases that tested negative. They cannot support a model that distinguishes
dengue from other causes of similar symptoms.

Among lab-confirmed cases, the target is a recorded outcome of dengue with
warning signs or severe dengue. These labels can include outcomes such as
bleeding, persistent vomiting, abdominal pain and a falling platelet count.
They describe severity after confirmation, not evidence that someone has dengue.

Inputs come from the initial notification form: age, sex, pregnancy, days since
symptom onset, 14 signs and 7 chronic conditions. The signs include fever, rash,
petechiae and the tourniquet test. The loader excludes outcome fields, including
the warning-sign labels, hospitalization and death, along with lab-result
fields that could leak later information.

The 2023 file contains 654,045 lab-confirmed cases. Training uses a stratified
sample of 200,000, with 2.4% positive outcomes. A separate sample from 2025 tests
the model on another year's records.

![dengue curves](docs/figures/dengue_curves.png)

| Test | Cases | AUC |
|---|---|---|
| held-out 2023 cases | 40,000 | 0.776 (0.760 to 0.792) |
| 2025, a different year | 100,000 | 0.733 |

The strongest input is the form's leukopenia indicator, followed by days since
onset, vomiting and age. For example, a 34-year-old woman on day 3 with fever,
headache and muscle aches receives a score around 0.9%. Adding petechiae, a
positive tourniquet test and leukopenia raises that example to about 11%.

The low positive rate makes the threshold difficult to use. Reaching 90%
sensitivity flags almost two thirds of the test set. Only 3.3% of flagged cases
have the positive outcome. Among the roughly one third left unflagged, 99.3%
have a negative outcome. Those figures describe this test set. They do not
establish that the model can safely rule out deterioration or guide discharge.

On the 2025 sample, AUC falls from 0.776 to 0.733. The observed positive rate
is 3.9%, while the mean predicted risk is 2.6%. The data does not establish
whether that change comes from the epidemic, reporting practices or both.
As with the heart model's hospital tests, useful ranking does not ensure
accurate probabilities in a different population.

A separate evaluation leaves out each of Brazil's five regions in turn:

| Left-out region | Cases | Real rate | Predicted on average | AUC |
|---|---|---|---|---|
| Center-West | 26,261 | 1.9% | 2.4% | 0.78 |
| North | 5,484 | 1.6% | 2.3% | 0.84 |
| Northeast | 8,237 | 5.3% | 3.5% | 0.79 |
| South | 46,839 | 3.4% | 1.9% | 0.74 |
| Southeast | 113,179 | 1.9% | 3.1% | 0.76 |

Regional AUC ranges from 0.74 to 0.84, but the predicted and observed rates
differ. The model underestimates the Northeast and South and overestimates the
other regions. Region is excluded as an input to reduce reliance on location
and possible reporting differences. These tests reuse the main search's chosen
settings rather than repeating the tuning search for each region.

## Malaria

The malaria model uses the NIH dataset of 27,558 single-cell images from
Giemsa-stained thin smears. The images come from 200 patients at Chittagong
Medical College Hospital in Bangladesh. Half the cells are labelled parasitized.
This task classifies an already cropped cell; it does not read a whole slide.

The model uses scikit-learn rather than a neural network. The feature extractor
turns each cell into 181 measurements: colour statistics, hue and saturation
histograms, edge directions and dark-pixel features. The dark-pixel features
capture contrast between stained regions and the rest of the cell.

The test split, cross-validation, calibration and threshold selection all keep
patients separate. Cells from one patient can share staining and lighting, so a
random cell split can reward recognition of slide conditions. A comparison
gets 0.9927 AUC with random cell splits and 0.9894 with patient splits. The gap
is small here, but the patient split is the more relevant evaluation.

![malaria curves](docs/figures/malaria_curves.png)

On 40 held-out patients (5,555 cells), AUC is 0.9915, with a 95% interval of
0.9882 to 0.9942. Bootstrap resampling uses patients rather than individual
cells, since cells from the same person are related observations. The threshold
flags 92.5% of parasitized cells and 1.6% of uninfected cells.

The patient-level breakdown shows variation hidden by the pooled score:

- For the 11 patients with no labelled parasitized cells, the median flagged
  fraction was 1.4%. The worst was 4.5%.
- For the 29 patients with labelled parasitized cells, median sensitivity was
  92%. For the worst patient, it was about one third.

The test set is roughly half parasitized cells. If only 1% of cells on another
slide were parasitized, the same sensitivity and false-positive rate would make
most flagged cells false positives. A slide-level system would need a separate
evaluation of cell counts and slide-level decisions. Cell accuracy alone does
not answer that question.

![malaria examples](docs/figures/malaria_examples.png)

The example figure includes the highest-confidence misses in the middle row
and false positives in the bottom row. Some misses have no obvious dark spot;
some false positives do. Appearance alone cannot establish a labelling error.
These cases would need review by a qualified microscopist before drawing
conclusions about label quality.

Inputs must be single cells, cropped and centred like the NIH examples. A whole
microscope field needs cell detection before classification. The training data
covers P. falciparum from one hospital, and the model does not distinguish species.

## The symptom model

The symptom model uses logistic regression over 131 yes/no symptom inputs. Its
fixed settings match the baseline that performed better on noisy held-out
patterns. Training also evaluates a reference forest and Naive Bayes.

### Why 100% accuracy is misleading

The clean evaluation reaches 100% accuracy, but the dataset makes this task
unusually easy. Duplicate rows are part of the problem.

The training file has 4,920 rows but only 304 distinct symptom patterns. Each
condition has 5 to 10 patterns, repeated until every class has 120 rows. A
random split before deduplication can put identical patterns in both training
and test sets, making memorization look like generalization.

`unique_symptom_patterns()` removes duplicates before splitting. Even then,
the model reaches 1.000 accuracy on the clean evaluations. The remaining
patterns still separate neatly by condition. They do not test how the model
handles people describing symptoms in their own words.

### Testing incomplete and noisy symptoms

The noise test drops symptoms from held-out patterns and adds unrelated ones.
It compares the forest with logistic regression and Naive Bayes. This gives a
harder test than the clean data, though the simulated noise is still no
substitute for real patient descriptions.

| Symptoms dropped and added | Forest | Logistic regression | Naive Bayes |
|---|---|---|---|
| none | 1.000 | 1.000 | 0.977 |
| 1 and 1 | 0.977 | **0.991** | 0.911 |
| 2 and 2 | 0.863 | **0.921** | 0.833 |
| 3 and 3 | 0.752 | **0.845** | 0.793 |
| 3 and 3, right answer in top 3 | 0.921 | **0.976** | 0.941 |

These are accuracies on held-out patterns, using 5 folds and 5 noise trials per fold.

The older noise test perturbs training patterns after fitting. That is easier
than perturbing unseen patterns. For the served logistic model, it reports
about 0.95 accuracy with two symptoms dropped and two added. The held-out
evaluation gives 0.92. Both results remain in the summary, with the held-out
results under `comparison`.

Logistic regression beats the forest at every nonzero noise level. With three
symptoms dropped and three added, it keeps the correct condition in the top
three 97.6% of the time, compared with 92.1% for the forest. The API now serves
logistic regression. The artifact stores normalized mean absolute coefficients
for global feature rankings. Per-prediction explanations still use the
dataset's symptom matches.

### Choosing the search metric

The reference forest's grid search uses macro recall, giving each of the 41 classes equal weight.
This keeps performance on one condition from disappearing inside an average
dominated by more common classes. Top-three accuracy is reported separately
because the API returns a ranked list.

The forest search does not identify a clear winner on this dataset: 16 of 32 candidates
tie, and the worst scores 0.983. Training reports the tie count alongside the
chosen settings so the nominal winner is not mistaken for a meaningful gain.

### Misspellings

The API normalizes symptom names, so `Skin Rash`, `skin rash` and `skin_rash`
map to the same key. It predicts using recognized symptoms and returns
suggestions for unknown names: `hedache`, for example, suggests `headache`.
If none of the supplied symptoms are recognized, it returns a 422 response
with suggestions instead of a prediction.

## Reviewing predictions

With database logging and administrator access enabled, you can attach a review to a stored prediction.
For symptoms, record whether the result was correct and, if known, the actual condition:

```bash
curl -X POST http://localhost:8000/predictions/42/review \
  -H "X-API-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"was_correct": false, "actual_disease": "Acne"}'
```

Running `python -m ml.train --with-feedback` adds corrected symptom predictions
to the training data. These rows do not receive extra weight. A result marked
wrong without an `actual_disease` counts in the review statistics but cannot
become a training example because it has no corrected label.

For the binary models, record whether the positive outcome occurred:

```bash
curl -X POST http://localhost:8000/risk-predictions/7/review \
  -H "X-API-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"had_condition": true}'
```

`/risk-predictions/stats` compares predicted probabilities with recorded
outcomes for each model. A gap can point to calibration drift, though the
reviewed cases may not represent all requests. The risk models do not retrain
on these outcomes; the records support later evaluation.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Web interface |
| GET | `/api` | Service description and endpoint list |
| GET | `/health` | Model load status and database logging status |
| GET | `/ready` | Returns 503 if any model is unavailable |
| GET | `/models` | Every model with its headline scores |
| GET | `/models/{name}` | One model's inputs, ranges, full scores and data source |
| POST | `/predict/heart` | Heart disease risk |
| POST | `/predict/diabetes` | Diabetes risk |
| POST | `/predict/liver` | Liver disease risk |
| POST | `/predict/dengue` | Risk that a confirmed dengue case turns serious |
| POST | `/predict/malaria` | Is this blood cell infected (send the image as the body) |
| POST | `/predict/symptoms` | Ranked conditions from symptoms |
| GET | `/symptoms` | The 131 symptom keys you can send |
| GET | `/diseases` | The 41 conditions the symptom model knows |
| GET | `/diseases/{name}/symptoms` | Symptoms recorded for one condition |
| GET | `/predictions` | Recent symptom predictions |
| GET | `/predictions/wrong` | Symptom predictions marked wrong |
| GET | `/predictions/stats` | Counts and accuracy among reviewed symptom predictions |
| POST | `/predictions/{id}/review` | Record a symptom prediction review |
| GET | `/risk-predictions` | Recent risk predictions |
| GET | `/risk-predictions/stats` | Live calibration and threshold performance per model |
| POST | `/risk-predictions/{id}/review` | Record whether the patient had the condition |

See [docs/api.md](docs/api.md) for request and response fields.

All `/predictions` and `/risk-predictions` routes require `X-API-Key`. Public
prediction requests need no key. Prediction and administrator routes share a
per-client rate limit of 60 requests per minute by default.

- The tabular models fill omitted optional measurements with training medians.
  Each response lists the fields it filled.
- The tabular request schemas reject unknown fields. Sending `cholesterol`
  instead of `chol` returns a 422 response.
- Tabular risk responses include up to three influential supplied inputs. The
  explanation replaces each input with a typical value and measures the change
  in the score. These are model effects, not evidence of medical causation.

## Repository layout

```
app/     the API. reads models/, never imports ml/
  static/         frontend HTML, CSS and JavaScript
  security.py     administrator keys, request limits and privacy headers
  cells.py        turns a blood cell image into the 181 numbers the malaria model reads
  db.py           postgres logging, review queues, feedback for retraining
  registry.py     loading the artifacts and refusing to load a bad one
  routers/risk.py the risk endpoints and their explanations
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

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Create a separate, disposable Postgres database for the database tests. Their
fixtures truncate both prediction tables. Set `TEST_DATABASE_URL` to that
database before running the suite.

In PowerShell:

```powershell
$env:TEST_DATABASE_URL = "postgresql://dps:devpass@localhost:5432/disease_predictions_test"
python -m pytest
```

In Bash:

```bash
TEST_DATABASE_URL=postgresql://dps:devpass@localhost:5432/disease_predictions_test python -m pytest
```

Without that variable, database tests skip. Browser tests are opt-in through
`RUN_BROWSER_TESTS=1` and require Playwright's Chromium installation. See the
[deployment guide](docs/deployment.md#checks-before-changes-ship) for those commands.

The database tests use real Postgres to check `text[]` round trips, `jsonb`
columns and partial indexes. Those behaviours depend on the database and would
not be covered by mocking its responses.

Data-cleaning tests create small fixtures rather than downloading datasets.
They check defects such as zero cholesterol values, duplicate rows and the
liver dataset's label encoding.

## Model compatibility

`requirements.txt` pins scikit-learn and joblib to the versions used by the
committed artifacts.

The joblib files contain pickled scikit-learn objects. Internal module changes
between versions can stop an artifact from loading, sometimes with errors such
as `No module named '_loss'`. If you change the scikit-learn version, retrain
and commit the matching artifacts in the same change. The API reports loading
failures as 503 responses with a retraining hint.

## Data

`scripts/download_data.py` fetches eleven source files. Raw data stays outside
Git. The loaders handle empty columns, missing measurements encoded as zero,
duplicate rows, inconsistent labels, packed age codes and patient identifiers.
See [data/README.md](data/README.md) for sources and cleaning rules.

## Disclaimer

This is an educational project trained on public datasets. It is not a validated
medical device and should not guide diagnosis or treatment. The heart data is
from the 1980s. The diabetes data covers one population of women, and the liver
data comes from one region of India. The dengue model covers confirmed cases
in Brazil. The malaria model classifies individual cells from one hospital's
slides. Every prediction response includes an educational-use disclaimer.

## License

MIT. See [LICENSE](LICENSE).

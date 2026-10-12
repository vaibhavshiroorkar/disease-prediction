# Disease Prediction System

Four models behind one FastAPI service. Tell it your symptoms and it ranks 41
conditions that match, with a confidence for each. Or give it clinical
measurements (blood pressure, glucose, liver enzymes and so on) and it gives
back a calibrated risk of heart disease, diabetes or liver disease, along with
what the risk was based on and which numbers it had to guess.

The symptom model came first. It trains on a well-known Kaggle file, and it
scores 100% on it, which tells you nothing. The three risk models are what I
built once I wanted a problem where the numbers mean something. They train on
real patient records from UCI, and none of them get anywhere near 100%.

Every prediction gets written to Postgres, so I can go back later and check it
against what actually happened.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt

python -m scripts.download_data
python -m ml.train --model all
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs.

The trained models are committed, so if you only want to run the API you can
skip the two middle steps. Training everything takes a few minutes. Most of
that is the nested cross-validation on the risk models.

```bash
curl -X POST http://localhost:8000/predict/heart \
  -H "Content-Type: application/json" \
  -d '{"age": 61, "sex": 1, "cp": 4, "thalach": 115, "exang": 1, "oldpeak": 2.5}'
```

```bash
curl -X POST http://localhost:8000/predict/symptoms \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"], "top_k": 3}'
```

### Turning on the logging

It runs fine without a database, it just doesn't record anything. If you want
the logging, point it at any Postgres:

```bash
docker compose up -d
export DATABASE_URL=postgresql://dps:devpass@localhost:5432/disease_predictions
uvicorn app.main:app --reload
```

It creates its own tables on startup. If Postgres isn't there it prints a
warning and carries on serving predictions, which is deliberate. Logging is for
me, not for the person waiting on a response.

## The models at a glance

| Model | Data | Patients | Test AUC (95% CI) | At its threshold |
|---|---|---|---|---|
| heart | UCI Heart Disease, 4 hospitals | 918 | 0.866 (0.808 to 0.918) | catches 87% of cases, clears 68% of healthy |
| diabetes | Pima Indians Diabetes | 768 | 0.815 (0.744 to 0.877) | catches 89% of cases, clears 60% of healthy |
| liver | Indian Liver Patient Dataset | 570 | 0.817 (0.739 to 0.887) | catches 89% of cases, clears 33% of healthy |
| symptom | Kaggle disease-symptom file | 304 unique rows | 1.000, and that's the problem | see below |

AUC is the chance that the model ranks a random sick patient above a random
healthy one. 0.5 is a coin flip and 1.0 is perfect. The intervals are wide
because each test set is only 114 to 184 patients. I'd rather show that than
quote three decimals and pretend.

## How the risk models are built

All three go through the same steps.

1. Clean the file. Each dataset has its own problems (below), and the loader
   logs every fix it makes, so training prints what it changed in the raw data.
2. Set aside 20% as a test set and leave it alone until the end.
3. Compare four kinds of model on the other 80% with nested cross-validation:
   5 folds, repeated 3 times, with a small tuning search inside each fold. The
   four are a dummy that guesses the average for everyone (the floor, always
   0.5 AUC), logistic regression, a random forest, and gradient boosting.
4. Pick a winner. If logistic regression is within half a standard deviation
   of the best, it wins. On this little data a gap that small could flip with a
   different random seed, and I'd rather trust the simpler model.
5. Calibrate it, so that "30% risk" means about 30 out of 100 such patients
   actually have it.
6. Choose a threshold: the highest cut-off that still catches 90% of real
   cases, picked on cross-validated predictions from the training set.
7. Score the test set once, with 1000 bootstrap resamples for the intervals.

Missing values are filled in inside the model pipeline, never in the loader.
That way each fold only ever sees medians computed from its own training rows.

### Why a threshold, and why 90%

Same reasoning as the symptom model. Missing someone who has the disease is a
worse mistake than flagging someone who doesn't, because a false alarm costs a
follow-up test. So the threshold is set to catch 9 in 10 real cases, and the
price shows up as false alarms. How big that price is depends a lot on the
model, and the table above shows it plainly.

### What won: logistic regression, every time

I expected the forest or the boosting to win at least once. They didn't. On all
three datasets the three real candidates finished within about one standard
deviation of each other:

| | heart | diabetes | liver |
|---|---|---|---|
| logistic regression | 0.879 +/- 0.027 | **0.837** +/- 0.036 | **0.729** +/- 0.047 |
| random forest | **0.879** +/- 0.026 | 0.833 +/- 0.038 | 0.718 +/- 0.040 |
| gradient boosting | 0.877 +/- 0.022 | 0.832 +/- 0.028 | 0.701 +/- 0.046 |
| dummy (floor) | 0.500 | 0.500 | 0.500 |

Cross-validated AUC on the training split, mean +/- standard deviation over 15
folds. On heart the forest is ahead by 0.0008, which is nothing, so the tie rule
picks logistic regression. With 450 to 730 training rows and 8 to 10 inputs,
there just isn't enough data for the fancier models to find anything the linear
one misses. It's a common result on small tabular medical data, and it was
good to see it happen rather than assume it.

### An improvement that held up: logging the lab values

Liver enzymes are wildly skewed. Most patients have an ALT around 35, and a few
have 2000. A linear model treats that 2000 as a huge lever. Taking the log first
pulls those values back in line:

| | liver | diabetes |
|---|---|---|
| logistic, raw values | 0.716 | 0.834 |
| logistic, skewed values logged | **0.729** | **0.837** |

That's a gain of 0.013 on liver and next to nothing on diabetes, which only has
two skewed inputs (insulin and the family-history score). It's small, but it's
the difference between logistic regression trailing the forest on liver and
leading it. Every training run keeps the raw version in the comparison table as
a reference row, so if this stops paying off, it'll be obvious.

## Heart disease

918 patients from four sources: the Cleveland Clinic, the Hungarian Institute of
Cardiology in Budapest, the university hospitals of Zurich and Basel (pooled as
one Swiss file), and the Long Beach VA. The question is whether an angiogram
found a major vessel more than 50% narrowed.

![heart curves](docs/figures/heart_curves.png)

On the test set it catches 89 of 102 patients with disease and misses 13. It
also flags 26 of the 82 without it. Calibration is good: the expected
calibration error is 0.04, so when it says 70%, about 70% of those people really
do have it. The input that matters most by far is the type of chest pain.
Exercise-induced ST depression and angina come next.

The raw files needed some fixing. Every Swiss patient has a cholesterol of 0,
and so do 49 at the VA. That means "not measured", not "no cholesterol". Three
columns (`slope`, `ca`, `thal`) are nearly complete at Cleveland and 50 to 99%
missing everywhere else. I dropped them, because the model can't use those
columns without also learning which hospital a patient came from. Cholesterol
has the same problem in a milder form, so it stays, but without a "was missing"
flag.

The test I care about most is training on three hospitals and testing on the
fourth. That's the closest this data gets to "what happens at a hospital it has
never seen":

| Left-out hospital | Patients | Real rate | Predicted on average | AUC |
|---|---|---|---|---|
| Cleveland | 303 | 46% | 54% | 0.87 |
| Hungary | 293 | 36% | 50% | 0.89 |
| Switzerland | 123 | 94% | 63% | 0.73 |
| VA | 199 | 74% | 72% | 0.75 |

At Cleveland and Hungary it ranks patients as well as it does on its own test
set. At Switzerland and the VA the AUC drops to the mid 0.7s. At Switzerland
the average prediction is also off by 30 points: 94% of the Swiss patients had
disease, and no model trained on the other three hospitals expects that. So the
ranking gets worse but survives, while the probabilities don't travel at all.
Before using this anywhere new you'd want to recalibrate it on local patients.

## Diabetes

768 women of Pima heritage, aged 21 and over. The question is whether they were
diagnosed with diabetes within five years.

![diabetes curves](docs/figures/diabetes_curves.png)

It catches 48 of the 54 who developed diabetes and clears 60 of the 100 who
didn't. Glucose does most of the work. Its importance is five times that of the
next input (BMI). The threshold sits low, at 0.21. That's what catching 90% of
cases costs when only a third of the women have the condition. So a flag here
means "test properly", not "probably diabetic": only 55% of flagged women had
it. On the other hand, 91% of the women it cleared really were fine.

The file uses 0 for "not measured" in five columns where 0 is impossible.
Insulin is "0" for 374 of the 768 women and skin thickness for 227. Leave those
in, and the model learns that an insulin of zero is normal and common.

I wouldn't trust it outside that group. It has only ever seen Pima women. It has
no idea what to do with a man, and the API will happily score one, because
nothing in the inputs says sex.

## Liver disease

570 patients from Andhra Pradesh, some recorded as liver patients and some not.
The question is which group a patient belongs to.

![liver curves](docs/figures/liver_curves.png)

This is the weak one. Cross-validated AUC is 0.73. The test set came out at
0.82, which is better than it deserves. With 114 patients a lucky split can do
that, and the interval (0.74 to 0.89) is wide enough to cover both. I'd trust
the cross-validated number.

The cost of the threshold is steep here. To catch 89% of patients, it also flags
22 of the 33 non-patients. Its precision-recall score of 0.93 looks impressive
until you notice that 71% of this dataset are patients, so a model that guessed
at random would score 0.71. Calibration is the loosest of the three (error
0.10).

Most published results on this dataset land in the same range, so I don't think
there's much left to find with these ten numbers.

The file had 13 exact duplicate rows, which I drop before the split for the
same reason as the symptom data. The label is coded 1 for patient and 2 for
not, which I map to 1 and 0.

## The symptom model

This is where the project started. It's a Random Forest over 131 yes/no
symptoms, tuned on macro recall.

### The 100% accuracy trap

Nearly every writeup of this dataset reports 100% accuracy. It took me a while
to work out why, and it isn't because the models are good.

The file has 4920 rows, but only **304 of them are actually different**. Each
disease has between 5 and 10 real symptom patterns, and those get repeated until
every class has exactly 120 rows. So if you do a normal random split, the same
row ends up in both train and test, and you're scoring the model on data it has
literally memorised.

`unique_symptom_patterns()` strips it down to those 304 distinct rows before
anything gets split. Even after that, the model scores 1.000 on every clean
measure. Each disease has a distinctive set of symptoms, so the task is just
easy. A perfect score here means the dataset separates cleanly. It does not mean
this would survive real patients describing their own symptoms.

### Making it hard on purpose

So the test that matters is the noisy one. I take a held-out pattern, drop a few
of its symptoms and add a few unrelated ones, which is what real people do. Then
I compare the forest with two simpler models:

| Symptoms dropped and added | Forest | Logistic regression | Naive Bayes |
|---|---|---|---|
| none | 1.000 | 1.000 | 0.977 |
| 1 and 1 | 0.977 | **0.991** | 0.911 |
| 2 and 2 | 0.863 | **0.921** | 0.833 |
| 3 and 3 | 0.752 | **0.845** | 0.793 |
| 3 and 3, right answer in top 3 | 0.921 | **0.976** | 0.941 |

Accuracy on held-out patterns, out of 5 folds and 5 rounds of noise each.

Two things came out of this.

First, my original noise test was too kind. It perturbed the same patterns the
forest had trained on, and it reported 0.94 at "2 and 2". On patterns the forest
had never seen, the same test gives 0.86.

Second, logistic regression beats the forest at every noise level. It degrades
more slowly, and it keeps the right answer in the top 3 almost every time, even
with three real symptoms missing and three wrong ones added. The API still
serves the forest for now. Swapping it means reworking the artifact and the
explanation code, which currently lean on the forest's feature importances.

### Why recall and not accuracy

The grid search scores candidates on macro recall. If the model misses a
condition, that condition isn't in the list at all and you never see it. If it
adds a wrong one, you read an extra line and move on. Macro recall weights all
41 classes equally, so the search can't give up recall on a rare disease to gain
a fraction of a point on the common ones.

Did it help? Honestly, no. 16 of the 32 candidates tied, and even the worst
scored 0.983. The training output prints the tie count, so I don't fool myself
into treating the "best" settings as a real finding.

### Misspellings

If you misspell something it doesn't throw the request away. It predicts on
whatever it recognised and hands back the names it didn't, with close matches:
send `hedache` and it'll suggest `headache`. `Skin Rash`, `skin rash` and
`skin_rash` all land on the same key.

## Looking at what it got wrong

Every prediction is stored. For the symptom model, I can record what the
condition really was:

```bash
curl -X POST http://localhost:8000/predictions/42/review \
  -H "Content-Type: application/json" \
  -d '{"was_correct": false, "actual_disease": "Acne"}'
```

Then `python -m ml.train --with-feedback` pulls those corrections in as extra
training rows. They're worth more than ordinary rows because they're cases the
model already got wrong. If I mark something wrong but don't know the right
answer, it's counted but not trained on. Making up a label to fill the gap would
just add noise.

For the risk models, the review is simpler: did the patient turn out to have it
or not.

```bash
curl -X POST http://localhost:8000/risk-predictions/7/review \
  -H "Content-Type: application/json" \
  -d '{"had_condition": true}'
```

`/risk-predictions/stats` then compares the average predicted risk with how
often people really had the condition, for each model. That's the cheapest live
calibration check I know of. If the two drift apart, the people using it don't
look like the training data. The risk models don't retrain on these outcomes
yet. They're recorded so I can see when that's needed.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Are the models loaded, is logging on |
| GET | `/models` | Every model with its headline scores |
| GET | `/models/{name}` | One model's inputs, ranges, full scores and data source |
| POST | `/predict/heart` | Heart disease risk |
| POST | `/predict/diabetes` | Diabetes risk |
| POST | `/predict/liver` | Liver disease risk |
| POST | `/predict/symptoms` | Ranked conditions from symptoms |
| GET | `/symptoms` | The 131 symptom keys you can send |
| GET | `/diseases` | The 41 conditions the symptom model knows |
| GET | `/diseases/{name}/symptoms` | Symptoms recorded for one condition |
| GET | `/predictions` | Recent symptom predictions |
| GET | `/predictions/wrong` | Symptom predictions I've marked wrong |
| GET | `/predictions/stats` | Counts, and how often it's actually right |
| POST | `/predictions/{id}/review` | Tell it what the answer should have been |
| GET | `/risk-predictions` | Recent risk predictions |
| GET | `/risk-predictions/stats` | Live calibration and threshold performance per model |
| POST | `/risk-predictions/{id}/review` | Record whether the patient had the condition |

Request and response shapes are in [docs/api.md](docs/api.md). A few behaviours
worth knowing:

- Optional measurements you leave out get filled with the training median, and
  the response tells you which ones it filled.
- A field it doesn't know is an error. Send `cholesterol` instead of `chol` and
  you get a 422, not a risk with your cholesterol quietly ignored.
- Each risk comes with the three inputs that moved it most. For each one, the
  patient is rescored with that value swapped for a typical one.

## Layout

```
app/     the API. reads models/, never imports ml/
  db.py           postgres logging, review queues, feedback for retraining
  registry.py     loading the artifacts and refusing to load a bad one
  routers/risk.py the three risk endpoints and their explanations
ml/      training
  datasets.py       symptom data loading and cleaning
  clinical_data.py  heart, diabetes and liver loading, cleaning and input specs
  model.py          the symptom forest, the recall search, the baselines
  clinical_model.py the four candidate families and the selection rule
  evaluate.py       symptom scoring and the noise tests
  clinical_eval.py  risk scores, bootstrap intervals, threshold, calibration
  train.py          run this to produce everything in models/
models/  the trained artifacts and a <name>_summary.json for each
docs/figures/  ROC, precision-recall and calibration plots, redrawn by training
tests/   128 of them
```

The API and the training code don't import each other. They meet at the joblib
files, which carry an `artifact_version` the API checks before trusting them.
The one exception is `ml.train --with-feedback`, which reaches into `app.db` to
fetch corrections. It does that import inside the function, so training still
works when there's no database around.

The request schemas for the risk models are written out by hand, so `/docs`
reads well. A test checks that their fields and ranges still match what the
artifacts say.

## Tests

```bash
pytest
TEST_DATABASE_URL=postgresql://dps:devpass@localhost:5432/disease_predictions pytest
```

Without the env var the 20 database tests skip and 108 run. With it, all 128
run.

The database tests hit a real Postgres rather than a mock. Mocking felt
pointless here, because the things most likely to break are the `text[]` round
trip, the `jsonb` columns and the partial indexes, and a mock would happily
pretend all three work.

The data tests don't need the downloaded files. Each one writes a few rows that
reproduce a specific defect (a zero cholesterol, a duplicate row, a label coded
2) and checks that the loader fixes it.

## One thing that will bite you

scikit-learn is pinned to an exact version in requirements.txt, not a range.

The models are pickles, and scikit-learn shuffles its internal modules between
minor releases. An artifact trained on 1.8 and loaded on 1.9 blows up with
`No module named '_loss'`, which is not a helpful thing to read at 1am. I spent
a while on that one. If you bump the version you have to retrain and commit the
new artifacts in the same change. The API will at least tell you that's what
happened instead of passing the raw ImportError through.

## Data

Eight files, fetched by `scripts/download_data.py`, not checked into the repo.
Every one of them arrives with problems: a phantom column, zeros that mean
"missing", duplicate rows, labels coded 1 and 2. What they are and how they're
handled is in [data/README.md](data/README.md).

## Disclaimer

This is a learning project trained on small public datasets. It is not a medical
device and you should not make decisions with it. The heart data is from the
1980s, the diabetes data covers one population of women, and the liver data
comes from one region of India. The disclaimer is in every API response too, so
it travels with the prediction.

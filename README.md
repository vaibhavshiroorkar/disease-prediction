# Disease Prediction System

Tell it your symptoms, it gives you back a ranked list of conditions that match,
with a confidence score for each one. 41 conditions, 131 symptoms, a Random
Forest behind it and FastAPI on top.

Every prediction gets written to Postgres. That was the whole reason I added a
database: I wanted to be able to go back and look at what it got wrong.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt

python -m scripts.download_data
python -m ml.train
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs.

The trained model is committed, so you can skip the two middle steps if you just
want to run the API. You only need them if you want to retrain it yourself.

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

It creates its own table on startup. If Postgres isn't there it prints a warning
and carries on serving predictions, which is deliberate. Logging is for me, not
for the person waiting on a response.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Is the model loaded, is logging on |
| GET | `/symptoms` | The 131 symptom keys you can send |
| GET | `/diseases` | The 41 conditions it knows |
| GET | `/diseases/{name}/symptoms` | Symptoms recorded for one condition |
| POST | `/predict/symptoms` | The actual prediction |
| GET | `/predictions` | What it's been asked lately |
| GET | `/predictions/wrong` | Everything I've marked wrong |
| GET | `/predictions/stats` | Counts, and how often it's actually right |
| POST | `/predictions/{id}/review` | Tell it what the answer should have been |

Request and response shapes are in [docs/api.md](docs/api.md).

## How it works

Your symptoms become a vector of 131 zeros and ones, one slot per symptom the
model knows. The forest scores all 41 conditions and the top few come back with
their probabilities.

If you misspell something it doesn't throw the request away. It predicts on
whatever it recognised and hands back the names it didn't, with close matches:
send `hedache` and it'll suggest `headache`. Pydantic normalises the input first,
so `Skin Rash`, `skin rash` and `skin_rash` all land on the same key.

## Why recall and not accuracy

The grid search scores candidates on macro recall and refits on the best one.

The reasoning: if the model misses a condition, that condition isn't in the list
at all and you never see it. If it adds a wrong one, you read an extra line and
move on. Those aren't the same size of mistake. Optimising for accuracy would let
the search give up recall on some rare disease to pick up a fraction of a point
on the common ones, and macro recall doesn't allow that because it weights all 41
classes equally.

Did it help? Honestly, no. 16 of the 32 candidates ended up tied at the same
macro recall, so the "winner" is just whichever one came first. Even the worst
one scored 0.983. The training output prints the tie count so I don't fool myself
into treating those best_params as a real result.

## The 100% accuracy trap

Nearly every writeup of this dataset reports 100% accuracy. It took me a while to
work out why, and it isn't because the models are good.

The file has 4920 rows, but only **304 of them are actually different**. Each
disease has somewhere between 5 and 10 real symptom patterns, and those get
repeated until every class has exactly 120 rows. So if you do a normal random
split, the same row ends up in both train and test, and you're scoring the model
on data it has literally memorised.

`unique_symptom_patterns()` strips it down to those 304 distinct rows before
anything gets split. Every number below is measured after that.

## Results

| | Accuracy | Macro recall | Top-3 |
|---|---|---|---|
| 5-fold CV, unique patterns | 1.000 | 1.000 | 1.000 |
| Held-out 42-row split | 1.000 | 1.000 | 1.000 |
| Drop 1 symptom, add 1 wrong one | 0.998 | 0.998 | 1.000 |
| Drop 2, add 2 | 0.941 | 0.921 | 0.992 |

All 41 classes get perfect recall on the held-out split. The per-class breakdown
is in `models/summary.json`.

I'd rather flag this than dress it up: those top two rows being 1.000 is not me
showing off. Deduplication kills the memorisation, but the task is still easy.
Each disease has a distinctive set of symptoms and the recorded patterns are
mostly subsets of it, so the forest pulls them apart without much trouble. A
perfect score here means the dataset separates cleanly. It does not mean this
would survive contact with real patients describing their own symptoms.

The bottom two rows are the ones I actually care about. Those simulate what
people really do: forget a symptom, throw in one that's unrelated. Two wrong
symptoms out of roughly seven and accuracy drops six points, recall drops eight.
But top-3 hardly budges, staying above 0.99. The right answer is still in the
list even when it isn't first, which is exactly why the API returns a ranking
instead of one confident-sounding diagnosis.

## Looking at what it got wrong

Every prediction is stored with its symptoms, the full ranking, the confidence
and how long it took. When one is wrong I record the real answer:

```bash
curl -X POST http://localhost:8000/predictions/42/review \
  -H "Content-Type: application/json" \
  -d '{"was_correct": false, "actual_disease": "Acne"}'
```

`/predictions/wrong` is the running list. Then:

```bash
python -m ml.train --with-feedback
```

pulls those corrections in as extra training rows. They're worth more than
ordinary rows because they're cases the model has already demonstrably failed.

If I mark something wrong but don't know what it should have been, it stays in
the table and gets counted, but training skips it. A row that says "this was
wrong" without saying what's right can't teach the model anything, and making up
a label to fill the gap would just add noise.

## Layout

```
app/     the API. reads models/, never imports ml/
  db.py       postgres logging, review queue, feedback for retraining
  registry.py loading the artifact and refusing to load a bad one
ml/      training
  datasets.py the loading and cleaning
  model.py    the forest and the recall search
  evaluate.py scoring, per-class recall, the noise tests
  train.py    run this to produce models/symptom.joblib
tests/   68 of them
```

The API and the training code don't import each other. They meet at the joblib
file, which carries an `artifact_version` the API checks before trusting it. The
one exception is `ml.train --with-feedback`, which reaches into `app.db` to fetch
corrections, and it does that import inside the function so training still works
when there's no database around.

## Tests

```bash
pytest
TEST_DATABASE_URL=postgresql://dps:devpass@localhost:5432/disease_predictions pytest
```

Without the env var the 15 database tests skip and 53 run. With it, all 68 run.

The database tests hit a real Postgres rather than a mock. Mocking felt pointless
here, because the things most likely to break are the `text[]` round trip, the
`jsonb` column and the partial indexes, and a mock would happily pretend all
three work.

## One thing that will bite you

scikit-learn is pinned to an exact version in requirements.txt, not a range.

The model is a pickle, and scikit-learn shuffles its internal modules between
minor releases. An artifact trained on 1.8 and loaded on 1.9 blows up with
`No module named '_loss'`, which is not a helpful thing to read at 1am. I spent
a while on that one. If you bump the version you have to retrain and commit the
new artifact in the same change. The API will at least tell you that's what
happened instead of passing the raw ImportError through.

## Data

One CSV, fetched by `scripts/download_data.py`, not checked into the repo. It
arrives with a few problems (a phantom column, a duplicated symptom, names with
stray spaces in them). What they are and how they're handled is in
[data/README.md](data/README.md).

## Disclaimer

This is a learning project trained on a small public dataset. It is not a medical
device and you should not make decisions with it. The disclaimer is in the API
response body too, so it travels with the prediction.

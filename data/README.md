# Data

`data/raw/` is gitignored. These files belong to their publishers and are not
vendored here. Fetch them with:

```bash
python -m scripts.download_data
```

| File | Source | Used by |
|---|---|---|
| `symptom_training.csv` | [Disease-Prediction-from-Symptoms](https://github.com/anujdutt9/Disease-Prediction-from-Symptoms) mirror of the Kaggle disease-symptom dataset | symptom model |
| `symptom_testing.csv` | 42-row held-out split published alongside it | symptom holdout score |

URLs live in `ml/config.py`. Adding a dataset means adding a `Source` there and
a loader in `ml/datasets.py`.

## Known defects in the raw files

Handled in `ml/datasets.py`, tested in `tests/test_datasets.py`.

**Symptom training file**
- A trailing comma on every row produces an unnamed empty column.
- `fluid_overload` appears twice; pandas renames the second to
  `fluid_overload.1`. The first copy is entirely zero and is dropped as constant.
- Several names carry stray spaces: `dischromic _patches`,
  `spotting_ urination`, `foul_smell_of urine`.

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
| `heart_{cleveland,hungarian,switzerland,va}.data` | [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), the processed files from four hospitals | heart model |
| `pima_diabetes.csv` | [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) mirror of the Pima Indians Diabetes data, which UCI no longer hosts | diabetes model |
| `ilpd_liver.csv` | [UCI Indian Liver Patient Dataset](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset) | liver model |

URLs live in `ml/config.py`. Adding a dataset means adding a `Source` there and
a loader in `ml/datasets.py` or `ml/clinical_data.py`.

## Known defects in the raw files

Handled in `ml/datasets.py` and `ml/clinical_data.py`, tested in
`tests/test_datasets.py` and `tests/test_clinical_data.py`.

**Symptom training file**
- A trailing comma on every row produces an unnamed empty column.
- `fluid_overload` appears twice; pandas renames the second to
  `fluid_overload.1`. The first copy is entirely zero and is dropped as constant.
- Several names carry stray spaces: `dischromic _patches`,
  `spotting_ urination`, `foul_smell_of urine`.
- 4920 rows collapse to 304 distinct symptom patterns. Each disease's handful
  of real patterns is repeated to pad every class to 120 rows. Splitting before
  deduplication leaks identical rows across train and test, which is where the
  widely quoted 100% accuracy comes from.

**Heart disease (920 patients, 4 hospitals)**
- No header. `?` marks a missing value.
- Cholesterol is recorded as 0 when it was not measured: all 123 Swiss
  patients and 49 at VA. One VA patient has a resting blood pressure of 0. All
  of these become missing.
- `slope`, `ca` and `thal` are mostly complete at Cleveland and 50-99% missing
  at the other three sites. They are dropped, because their missingness tells
  the model which hospital a patient came from.
- Disease prevalence differs by site: Hungary 36%, Cleveland 46%, VA 74%,
  Switzerland 93%.
- The target `num` runs 0 to 4. It is collapsed to `num > 0`.
- 2 exact duplicate rows.

**Pima diabetes (768 women)**
- No header.
- 0 stands for "not measured" in five columns where 0 is impossible: glucose
  (5 rows), diastolic blood pressure (35), skin thickness (227), insulin (374)
  and BMI (11). All become missing. A zero pregnancy count is real and kept.

**Indian liver patients (583 patients)**
- No header. Sex is the text `Male`/`Female`.
- The label is 1 for a liver patient and 2 for not.
- 13 exact duplicate rows.
- 4 patients have no albumin/globulin ratio.

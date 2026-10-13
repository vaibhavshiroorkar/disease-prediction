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
| `dengue_sinan_{2023,2025}.csv.zip` | Brazil's national notification register (SINAN), published by the Ministry of Health on [OpenDataSUS](https://dadosabertos.saude.gov.br/) | dengue model: 2023 trains it, 2025 checks it |
| `malaria_cell_images.zip` | [NIH malaria cell images](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html), from the Lister Hill National Center for Biomedical Communications | malaria model |

The full download is about 450 MB, nearly all of it the malaria images and the
two dengue years. To fetch only some files:

```bash
python -m scripts.download_data --only heart_cleveland diabetes
```

Training the malaria model also writes `malaria_features_v1_<size>.npz` here.
It caches the 181 numbers per cell, so the second run skips reading 27,558
images.

URLs live in `ml/config.py`. Adding a dataset means adding a `Source` there and
a loader in `ml/datasets.py`, `ml/clinical_data.py` or `ml/malaria.py`.

## Known defects in the raw files

Handled in `ml/datasets.py`, `ml/clinical_data.py` and `ml/malaria.py`, tested
in `tests/test_datasets.py`, `tests/test_clinical_data.py` and
`tests/test_malaria.py`.

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

**Dengue notifications (SINAN, Brazil)**
- One row per reported case, about 1.6 million a year. Column names and codes
  are in Portuguese: `FEBRE` is fever, `LACO` the tourniquet test.
- Yes/no fields use 1 for yes, 2 for no and 9 or blank for unknown. 9 becomes
  missing.
- Age is packed into one number, a unit digit then three digits of value:
  `4031` is 31 years, `3006` is 6 months. An impossible value like `4999`
  becomes missing.
- Pregnancy is coded 1 to 3 for the trimester, 4 for "age unknown", 5 for not
  pregnant, 6 for not applicable (men and children) and 9 for not recorded.
- The gap between notification and symptom onset is sometimes negative or
  months long. Anything outside 0 to 30 days becomes missing.
- The public release has no cases that tested negative. They are removed
  before publication, which is why this data trains a severity model and not
  a diagnosis model.
- 2023 has 654,045 lab-confirmed cases, 2.4% of them with warning signs or
  severe dengue. 2025 has 511,555, at 3.9%. Training uses a stratified sample
  of 200,000 from 2023 and checks on 100,000 from 2025.

**Malaria cell images (NIH)**
- 27,558 PNG crops, one red blood cell each, in `Parasitized/` and
  `Uninfected/` folders. Sizes vary from about 50 to 250 pixels a side, so
  every crop is resized to 64x64 before features are taken.
- The cells come from 200 patients. The patient is the `C` number at the start
  of the filename, but the rest of the prefix is spelled several ways:
  `C100P61ThinF`, `C37BP2_thinF`, `C42P5thin_original`, `C12NThinF`.
- Infected patients contribute both parasitized and uninfected cells. Healthy
  patients contribute uninfected cells only.
- Cells from one patient share a slide, a stain batch and the lighting.
  Splitting by cell puts the same slide on both sides of the split, so every
  split here is by patient.

# Related work

Checked on 2026-11-25 through search results and abstracts. Full texts were not
read, so each row says only what the abstract or index showed. "Verified" means
the claim appeared there. "Assumed" means it comes from general knowledge.

## What others found

| Work | Finding that matters here | Status |
|---|---|---|
| Detrano et al., 1989, *American Journal of Cardiology* 64:304-310 | Tested a coronary disease model in Budapest, Long Beach and Zurich/Basel. It overpredicted disease in Hungary and Long Beach and underpredicted in Switzerland | verified ([index record](https://www.scilit.com/publications/3e4cece298ed5e00301ec7962e40df5f)) |
| Zech et al., 2018, *PLoS Medicine* | Sorting X-rays by hospital alone gave AUC 0.861 for pneumonia, because the hospitals' rates differ. External results were worse in 3 of 5 comparisons | verified ([PMC6219764](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764)) |
| Wong et al., 2021, *JAMA Internal Medicine* | The Epic sepsis model scored AUC 0.63 at an outside hospital, against 0.76 to 0.83 reported | verified ([PMC8218233](https://pmc.ncbi.nlm.nih.gov/articles/PMC8218233)) |
| Van Calster et al., 2019, *BMC Medicine* 17:230 | Calibration is assessed far less often than ranking, and drift calls for updating | verified ([PMC6912996](https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996)) |
| Vergouwe et al., 2017, *Statistics in Medicine* 36(28) | Compares re-estimating the intercept, intercept and slope, or everything, with a test that picks one. In their example the intercept was enough, and full revision overfitted | verified ([PMC6857513](https://pmc.ncbi.nlm.nih.gov/articles/PMC6857513) and [Twente record](https://research.utwente.nl/en/publications/a-closed-testing-procedure-to-select-an-appropriate-method-for-up/)) |
| Saerens et al., 2002, *Neural Computation* 14:21-41 | EM estimates a new class rate from unlabelled data | verified ([record](https://mlanthology.org/neco/2002/saerens2002neco-adjusting)) |
| Saerens et al., the assumption behind it | It holds only if the inputs look the same within each class in both places | assumed: it follows from the derivation, which we did not read. `tests/test_study.py` checks our version where the assumption holds |
| Snell et al., 2021 | Precise calibration estimates often need more than 200 events and 200 non-events | verified ([PMC8352630](https://pmc.ncbi.nlm.nih.gov/articles/PMC8352630)) |
| Severe dengue from SINAN without lab data (SBC proceedings) | A random forest classified severity from non-lab fields | verified that it exists ([paper](https://sol.sbc.org.br/index.php/sbcas/article/download/42790/42557/)); its split is unknown |

Two arXiv papers use Cleveland and Hungary as source and Switzerland and Long
Beach as target ([2212.02742](https://arxiv.org/pdf/2212.02742),
[2506.05047](https://arxiv.org/pdf/2506.05047)). They detect harmful shift. They
do not measure how many local patients repair it (assumed, from abstracts).

## What is new here, and what is not

- **Not new:** that this kind of model miscalibrates at these hospitals.
  Detrano reported it in 1989. H1 confirms it on the current pipeline.
- **Not new:** the updating methods. All six are standard.
- **What this project adds:** one fixed protocol over hospitals, regions and a
  later year, with the smallest local sample that repairs each, and a
  label-free method in the same comparison. We found no study that does this
  on SINAN severity. That is a limited search, not proof that none exists.

## Choices the literature settled

- Report ranking and calibration separately.
- Try the intercept first. Vergouwe's order goes from least change to most.
- Keep the Swiss AUC out of every conclusion. It has 8 patients without
  disease, far below the sample sizes above.

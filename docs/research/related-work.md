# Related work

Checked on 2026-11-25. The first table rests on search results and abstracts.
The second says how much of each paper was read. Each row says only what that showed. "Verified" means
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
| Severe dengue from SINAN without lab data (SBC proceedings) | A random forest classified severity from non-lab fields | verified that it exists ([paper](https://sol.sbc.org.br/index.php/sbcas/article/download/42790/42557/)); random 85/15 split, see below |

Two arXiv papers use Cleveland and Hungary as source and Switzerland and Long
Beach as target ([2212.02742](https://arxiv.org/pdf/2212.02742),
[2506.05047](https://arxiv.org/pdf/2506.05047)). They detect harmful shift. They
do not measure how many local patients repair it (assumed, from abstracts).

## The closest work, read on the second pass

| Work | What it did | Overlap with this study | Read |
|---|---|---|---|
| Yao, Fang and Zhu, preprint posted 1 Nov 2026 ([rs-11054375](https://www.researchsquare.com/article/rs-11054375/v1)) | ICU deterioration, MIMIC-IV (31,054 stays) to eICU (90,393) and NWICU (16,562). Intercept, or intercept and slope, from 20, 50, 100 or 200 local events. Finds miscalibration more consistent than ranking loss, and no universal event count | Very high: the question, two methods, and counting events. Earlier and larger than ours | abstract |
| Ebrahim, 28 Oct 2026 ([2610.39065](https://arxiv.org/pdf/2610.39065)) | Pre-registered simulation plus four clinical datasets. 500 local rows repaired most rate shift. 100 rows usually made calibration worse | High: small samples hurt, recalibration repairs, rules fixed in advance | key sections |
| Davis et al., 2019 ([PMC6857513](https://pmc.ncbi.nlm.nih.gov/articles/PMC6857513/)) | A bootstrap test picks no update, intercept, recalibration or refit. 1,000 to 10,000 rows. Small sets favour simple updates | High for H3 and H4 | part |
| Brown et al., 2025 ([PMC12204302](https://pmc.ncbi.nlm.nih.gov/articles/PMC12204302/)) | Rural hospitals lack the rows to recalibrate: 15,299 to 16,461 needed, 905 to 3,187 available | Same practical question | part |
| Alexandari et al., 2020 ([1901.06852](https://arxiv.org/abs/1901.06852)) | EM for a changed rate needs a calibrated source model | Explains one way EM fails. Our sources were sigmoid-calibrated, and EM recovered their own rate in the 3 we checked, yet it failed on targets | abstract |
| SINAN severity study (SBC proceedings, linked above) | 700,000 cases, random 85/15 split, no mention of calibration | Same data and nearly the same label. No transfer test | methods |
| [dengue-risk-prediction](https://github.com/worldofjiayi/dengue-risk-prediction) | 9.45 million SINAN records. Severe against other dengue AUC 0.810 on balanced tests, scores not probabilities | Same data. No transfer or calibration test of its own model | README |

Steyerberg 2004 and Janssen 2008 are the standard early papers on updating.
Their pages could not be opened, so nothing is claimed from them.

## Label-shift work, checked for the state study (2026-11-26)

The question: has a label-free rate correction such as EM been tested on many
real clinical shifts, and has anyone predicted where it fails?

| Work | What it did | Bearing on our result | Read |
|---|---|---|---|
| Esuli, Molinari and Sebastiani, *ACM TOIS*, 2021 ([record](https://iris.cnr.it/handle/20.500.14243/420688)) | Reassessed EM (they call it SLD) on text data | EM helps only with few classes and a calibrated classifier, and otherwise hurts. Ours had two classes and calibrated sources, yet it hurt, so calibration alone does not explain our failures | search summary |
| Godau et al., arXiv 2023, *Medical Image Analysis* 2025 ([2303.12540](https://arxiv.org/abs/2303.12540)) | 30 medical image tasks. Rate correction improved calibration | The shifts were made by resampling one test set, so inputs within a class could not change. They used the true rates with noise, not a label-free estimate | method and results sections |
| Roschewitz, Mehta, Jones and Glocker, arXiv 2024 ([2411.07940](https://arxiv.org/abs/2411.07940)) | Images from 4 datasets, mixes of real scanners and sites. Tells rate shift from input shift with two tests | A label-free rate correction made calibration "much worse" once inputs also shifted. Same mechanism as ours, shown in plots, on images. They do not claim their tests predict whether the correction will help | full text |
| Chandra H, Shivhare and Sahu, Research Square preprint, July 2026 ([rs-10254564](https://www.researchsquare.com/article/rs-10254564/v1)) | MIMIC-IV to 185 eICU hospitals. A rate correction cut mean calibration error from 0.071-0.085 to 0.027-0.037 | It used each hospital's true mortality rate, not an estimate from unlabelled predictions. So knowing the rate fixed most of the error there. Estimating it was not tested | abstract |
| Garg et al., NeurIPS 2020 ([2003.07554](https://arxiv.org/abs/2003.07554)) | EM (MLLS) with calibration beats BBSE under pure rate shift | Benchmarks on simulated rate shift only | search summary |

Not read: the full text of Esuli et al. and of Chandra H et al. (the PDF could
not be turned into text here).

## What is new here, and what is not

- **False, and withdrawn:** that measuring the smallest local sample is this
  project's addition. Yao et al. did it first, with more data.
- **Not new:** H1, H2, H3 and H4. They confirm earlier work on other diseases.
- **Not new:** that small updates can hurt (Ebrahim), or planning by events
  (Yao). Our 9 to 48 cases is one more data point, seen after the fact.
- **Possibly new (assumed, limited search):** the first leave-one-region-out
  and later-year test of SINAN severity models with calibration measured; and
  EM compared with labelled updates on the same clinical targets, where it
  failed on 7 of 10 although the source models were sigmoid-calibrated. The
  state study repeated this at scale: EM failed on 41 of 47 state-years, and
  a covariate-shift score did not predict where
  ([findings-states.md](findings-states.md)).
- **Not new, checked 2026-11-26:** that EM can fail when inputs shift as well
  as rates. It follows from EM's assumption, and Roschewitz et al. (2024)
  showed it on images.
- **Possibly new after that check (assumed, one day of searching):** the
  size of the failure on tabular clinical data with natural shift (41 of 47
  state-years, paired against labelled updates), and the negative result that
  a domain-classifier score did not predict where EM fails. Roschewitz et al.
  identify the type of shift but do not test whether that predicts the
  correction's success. Chandra H et al. suggest that the true rate repairs
  most of the error. If that holds here too, EM fails because its rate
  estimate is wrong, not because rate correction is wrong. We have not tested
  that.

## Choices the literature settled

- Report ranking and calibration separately.
- Try the intercept first. Vergouwe's order goes from least change to most.
- Keep the Swiss AUC out of every conclusion. It has 8 patients without
  disease, far below the sample sizes above.

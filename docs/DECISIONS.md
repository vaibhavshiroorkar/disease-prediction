# Decision log

## 2026-11-25: not a novel research paper yet

- **Decided:** treat 1.1.0 as a solid replication, suitable for a student or
  regional conference if it is framed that way. Do not submit it as novel.
- **Why:** Yao et al. (1 November 2026) asked the same question with far more
  data. H1 to H4 confirm known results. See
  [research/related-work.md](research/related-work.md).
- **What could make a paper:** SINAN by state and year gives dozens of targets
  instead of ten. Questions left open there: when does updating hurt, and can a
  clinic tell in advance that a label-free repair will fail?
- **Would change it:** a paper that already does this on SINAN.

## 2026-11-25: study transfer, not a higher score

- **Decided:** the research question is what breaks when a model moves, and how
  few local patients repair it.
- **Why:** the small datasets cannot separate model families (0.877 to 0.879 on
  heart), so chasing accuracy would measure noise. Heart has four hospitals and
  dengue five regions and two years, so transfer can be measured.
- **Considered:** a larger model search (gains below the noise), new disease
  datasets (breadth without depth), deep learning on malaria (already 0.99 AUC,
  and one hospital so no transfer test).
- **Would change it:** a second malaria site, or a much larger heart dataset.

## 2026-11-25: freeze the protocol first

- **Decided:** commit the protocol before any study code, with pass marks.
- **Why:** choosing pass marks after seeing results would prove nothing. H1 was
  partly known from the README, so it is marked exploratory.

## 2026-11-25: plain JavaScript and SVG for the lab

- **Decided:** hand-written SVG in the existing site.
- **Considered:** Chart.js or D3 (a new dependency, and the security policy
  allows own files only), Streamlit (a second server on a free plan), static
  images (no interaction).
- **Why:** few chart types, no build step, works on the current host.

## 2026-11-25: publish counts, not scores

- **Decided:** `lab-data.json` holds counts per cutoff.
- **Why:** counts give every metric, and no row of patient data is published.
  The builder stops if the AUC differs from the committed summary.

## 2026-11-25: keep the served models unchanged

- **Decided:** no retraining for this version.
- **Why:** the study measures these model families and settings. Changing them
  would break the link between the results and what the site serves.

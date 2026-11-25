# Decision log

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

# Decision log

## 2026-11-26: report the state study as run, flaws included

- **Decided:** keep every frozen verdict as computed. S1 stays "supported" in
  the table, with a caveat that its design cannot separate cases from
  patients. The two patterns seen afterwards are labelled post hoc.
- **Why:** rewriting a test after its result is the failure the protocol
  exists to stop. A known flaw, stated, is worth more than a quiet fix.
- **Would change it:** nothing for this run. The next protocol fixes S1.

## 2026-11-26: the state study design

- **Decided:** targets are state-years with at least 1,000 confirmed and 100
  severe cases: 29 from 2023 and 2025. Each state gets a source model trained
  on the other states in 2023. The EM guard is fitted on 2023 and tested on
  2025. Protocol: [research/protocol-states.md](research/protocol-states.md).
- **Why:** below about 50 severe cases per half, the test log loss is too noisy
  to compare methods. Keeping the guard's threshold away from the targets it
  is judged on avoids grading it on its own training data.
- **False, corrected:** "about 80 targets". The 2023 and 2025 files give 54
  state-years, and only 29 pass the rule. 2024 could add up to 27 more.
- **Considered:** the five regions again (too few targets), every state-year
  with any severe case (many have under 10), a fixed national model for all
  targets (it would have seen each target state).
- **Update, same day:** 2024 was downloaded before any study code. It adds
  18 targets: 47 in all, 33 in stage B.
- **Would change it:** if S3 fails with wide
  intervals, more years are the fix, not a new hypothesis.

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

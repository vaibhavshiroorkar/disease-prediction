# Security

The public site takes form values and blood cell images from strangers, and the
server loads pickled models. Review this file whenever an input path or a
dependency changes. Checked 2026-11-25 by reading the code and running the tests.

## What is worth protecting

- The server and its `ADMIN_API_KEY` and `DATABASE_URL`.
- Anything visitors type in, if prediction logging is on.
- The trained models and raw data, which must not be swapped.

## Weaknesses and countermeasures

| Weakness | Countermeasure | Status |
|---|---|---|
| A joblib file runs code when loaded | Only files committed in `models/` are loaded, never an upload or a download. Anyone who can change `models/` can run code on the server, so review model changes like code | Load paths verified in `app/registry.py` and `study/lab_data.py`. Artifacts are not signed: open |
| Hostile image | 2 MB and 4 million pixel limits; PNG and JPEG only | verified in `app/cells.py`. A decoder bug in Pillow is unknown risk |
| Large request bodies | Refused above 2.1 MB while streaming | verified by `tests/app/test_hosting.py` |
| Flooding | 60 requests a minute per client on prediction and admin routes, counted in one process | verified by `tests/app/test_hosting.py`. Resets on restart and does not cover several replicas |
| Reading stored predictions | Admin key needed; routes return 503 when no key is set | verified by `tests/app/test_hosting.py` |
| Script injection | Content-Security-Policy allows own files only; the pages set text, never HTML | verified in `app/security.py`; the lab builds nodes with `textContent` |
| Lab data leaks patients | `lab-data.json` holds counts, bin averages and study means only | verified in `study/lab_data.py`. Small bins can hold few rows, from public datasets |
| Altered downloads | The study records a SHA-256 for every raw file it read | verified in `results/transfer/report.json`. `scripts/download_data.py` does not check hashes: open |
| Vulnerable dependency | The runtime set is hash-locked | verified in `requirements-runtime.txt`. No scheduled vulnerability scan: open |
| A visitor treats a score as a diagnosis | Notices on every page, cutoffs and error rates with each result | verified on the site. Whether people read them is unknown |

## Reporting a problem

Use GitHub's private vulnerability reporting on this repository if it is
enabled, or open an issue that asks for a private contact. Do not put exploit
details in a public issue.

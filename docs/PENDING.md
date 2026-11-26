# Pending work

Finished items move to [CHANGELOG.md](CHANGELOG.md).

## For Vaibhav

- Review and merge the `research-study` branch. Merging to `main` may redeploy
  the Render demo, so it was left as your call.
- After merging, tag the merge commit `v1.1.0` and push the tag.


## State study (run, findings written)

[Findings](research/findings-states.md).

- Run the frozen 2026 retest ([protocol](research/protocol-2026.md)). Needs
  `DENGBR26.csv.zip` (15.9 MB, preliminary) downloaded with Vaibhav's OK, its
  hash recorded in the protocol in its own commit, then
  `python -m study.states.retest`.
- Read Esuli et al. 2021 in full. The open copy is a PDF download.

## Research ideas (suggestions, not started)


- Read Yao et al. 2026, Steyerberg 2004 and Janssen 2008 in full before writing.
- Find why VA's ranking broke: compare each input's effect there with elsewhere.
- Larger local samples for the refit, with stronger penalties, to find where it
  passes the shift.
- Measure how much the imputed cholesterol still gives away the hospital.
- A second malaria source, so that model can have a transfer test.

## Engineering

- Lock the development dependencies. Only the runtime set is locked.
- Check raw data hashes in `scripts/download_data.py`.
- Sign or hash the model artifacts, and scan dependencies on a schedule.
- Link the findings from the About page once the branch is merged.

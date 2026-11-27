# Pending work

Finished items move to [CHANGELOG.md](CHANGELOG.md).

## For Vaibhav

- Review and merge the `research-study` branch. Merging to `main` may redeploy
  the Render demo, so it was left as your call.
- After merging, tag the merge commit `v1.1.0` and push the tag.


## State study (run, findings written)

[Findings](research/findings-states.md).

- Read Esuli et al. 2021 in full. The open copy is a PDF download.

## Research ideas (suggestions, not started)


- Show the 2026 retest on the Lab page, next to the state study.
- Rerun the retest on the final 2026 file once SINAN closes the year, to see
  whether late classification moved any verdict. T4 needs more targets inside
  the O/E band than 2026 gave.
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

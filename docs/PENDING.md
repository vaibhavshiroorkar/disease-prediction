# Pending work

Finished items move to [CHANGELOG.md](CHANGELOG.md).

## For Vaibhav

- Review and merge the `research-study` branch. Merging to `main` may redeploy
  the Render demo, so it was left as your call.
- After merging, tag the merge commit `v1.1.0` and push the tag.

## Research ideas (suggestions, not started)

- **Main path to a paper:** rerun on SINAN by state and year (27 states, 2023
  to 2025, about 80 targets at most; small states may drop out). Freeze a new protocol first:
  when does updating hurt, and does a label-free check warn that EM will fail?
- In the same run, test whether local cases, not patients, predict the repair
  size, with events on the axis like Yao et al.
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

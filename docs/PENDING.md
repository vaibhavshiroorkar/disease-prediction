# Pending work

Finished items move to [CHANGELOG.md](CHANGELOG.md).

## For Vaibhav

- Review and merge the `research-study` branch. Merging to `main` may redeploy
  the Render demo, so it was left as your call.
- After merging, tag the merge commit `v1.1.0` and push the tag.


## State study (in progress)

Protocol: [research/protocol-states.md](research/protocol-states.md).

- Run `python -m study.run_states` detached, then write the findings, with
  commit and data hashes. Add the results to the Lab page after that.

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

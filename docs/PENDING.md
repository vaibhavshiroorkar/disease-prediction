# Pending work

Finished items move to [CHANGELOG.md](CHANGELOG.md).

## For Vaibhav

- Review and merge the `research-study` branch. Merging to `main` may redeploy
  the Render demo, so it was left as your call.
- After merging, tag the merge commit `v1.1.0` and push the tag.


## State study (run, findings written)

[Findings](research/findings-states.md).

- Suggestion: try a label-free rate estimate built for input shift (for
  example black-box shift estimation, Lipton et al. 2018), since the true-rate
  check found EM's estimate is what fails.
- Read Chandra H et al. 2026 and Esuli et al. 2021 in full.
- A frozen test of the two post-hoc patterns (repair only pays outside an
  observed over expected band; EM only helps when the ratio passes 2), on new
  data such as SINAN 2026 once it is published.
- A fixed version of S1 for that test: compare cases within each n, not pooled.
- Add the state results to the Lab page.

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

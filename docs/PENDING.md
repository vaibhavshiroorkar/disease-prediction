# Pending work

Finished items move to [CHANGELOG.md](CHANGELOG.md).

## For Vaibhav

- Review and merge the `research-study` branch. Merging to `main` may redeploy
  the Render demo, so it was left as your call.
- After merging, tag the merge commit `v1.1.0` and push the tag.

- Decide whether to download SINAN 2024 (the file is not on disk). It must
  happen before stage A runs, or the study goes ahead without it.

## State study (in progress)

Protocol: [research/protocol-states.md](research/protocol-states.md).

1. State loader: keep the state code, apply the inclusion rule, list the
   excluded state-years with counts.
2. Shift score: source rows against target pool rows, 5-fold AUC.
3. Runner: stage A, write the EM guard threshold, then stage B. Record the
   local severe cases in every draw.
4. Verdicts S1 to S4, the two replications, state-level bootstrap. Tests for
   each piece.
5. Run, then write the findings, with commit and data hashes.

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

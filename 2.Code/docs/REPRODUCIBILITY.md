# Reproducibility

`run_experiment.py` creates an immutable run folder under `outputs/runs/` containing:

- scenario and controller snapshots;
- manifest with timestamp, Python/platform, dependencies, commit and seeds;
- per-run summary, time series, predictions and event log;
- aggregate statistics, comparison CSV, PNG and self-contained HTML.

Re-run from the snapshots rather than from a later edited global configuration. Archive the whole
run directory with a manuscript or benchmark release. Do not remove failed trials from aggregate
results; classify and report them.

Native Monte Carlo release evidence has an additional source gate. A clean Git
commit (or immutable installed distribution) is required by default, and the
manifest records a deterministic hash of validation-relevant source/config
files. Resume requires both the protocol fingerprint and source snapshot to
match. Dirty-source execution is available only through an explicit smoke-test
override and is never release-eligible.

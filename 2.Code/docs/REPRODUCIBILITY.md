# Reproducibility

`run_experiment.py` creates an immutable run folder under `outputs/runs/` containing:

- scenario and controller snapshots;
- manifest with timestamp, Python/platform, dependencies, commit and seeds;
- per-run summary, time series, predictions and event log;
- aggregate statistics, comparison CSV, PNG and self-contained HTML.

Re-run from the snapshots rather than from a later edited global configuration. Archive the whole
run directory with a manuscript or benchmark release. Do not remove failed trials from aggregate
results; classify and report them.

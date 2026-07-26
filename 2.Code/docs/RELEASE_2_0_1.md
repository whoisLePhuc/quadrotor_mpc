# Release 2.0.1 Stabilization

## Release statement

Version 2.0.1 closes the planned Phase 1–8 implementation as a
feature-complete research workbench. It does not extend the controller
algorithm or upgrade the scientific claim. The current claim remains:

> Empirically validated with documented limitations; not production-ready,
> not proven episode-wide probabilistically safe, and not 20 Hz real-time.

## Stabilization changes

- Root documentation now describes measured repository behavior rather than
  copying performance claims from the source papers.
- Broken scenario/notebook commands were replaced with files that exist.
- The CC-MPC command-acceptance deadline is `50 ms`, equal to the controller
  period. Configuration validation rejects a longer deadline when late
  solutions are configured for rejection.
- GitHub Actions runs lint, the full Python test suite, all native config
  checks, a real Qt panel lifecycle smoke test, distribution build and an
  installed-wheel smoke test.
- `uv.lock` freezes the complete cross-platform dependency resolution.
- The wheel installs native top-level modules, six console scripts, native YAML
  configurations and the licensed Crazyflie MuJoCo assets.
- Root and wheel distributions include the MIT license.
- Native Monte Carlo manifests record clean/dirty source state and an exact
  validation-source SHA-256. Release campaigns refuse a dirty Git tree by
  default and verify the snapshot again on resume.

## Release gates

Run from `2.Code/`:

```bash
uv sync --all-extras --locked
uv run ruff check .
uv run pytest
uv run python scripts/release_check.py
uv run python scripts/smoke_native_panel.py
uv run python -m build
```

Validate the installed wheel outside the source checkout:

```bash
python -m venv /tmp/quadrotor-wheel-test
/tmp/quadrotor-wheel-test/bin/python -m pip install dist/*.whl
cd /tmp
/tmp/quadrotor-wheel-test/bin/quadrotor-mpc-native --validate-config
/tmp/quadrotor-wheel-test/bin/quadrotor-mpc-monte-carlo --validate-config
```

Run the 50-seed release campaign only after committing the intended source:

```bash
uv run quadrotor-mpc-monte-carlo \
  --config config/native_monte_carlo.yaml \
  --workers 3 \
  --batch-size 5
```

The campaign has 450 episodes:

```text
50 paired seeds × 3 covariance levels × 3 controller modes
```

`CLEAN_GIT_COMMIT` or a fingerprinted, non-editable installed distribution is
release-eligible. Installed payloads receive the same deterministic snapshot
hash treatment as source checkouts. `DIRTY_GIT_SNAPSHOT` remains reproducible
through its snapshot hash, while editable distributions and unverified loose
source trees are also explicitly non-release-eligible. Native MJCF/OBJ model
files are included in the validation-source fingerprint.

## Manual desktop acceptance

Automated CI verifies that the Qt safety panel can be constructed, updated,
reset and closed. Before creating the GitHub release, perform one native Linux
desktop session and check:

1. MuJoCo and the Qt panel open together.
2. Pause, single-step, Reset and Run again work.
3. Positive slack never appears as guarantee-eligible.
4. A deadline miss or fallback is visible in the cards, plots and transition
   log.
5. Closing either configured window terminates the session cleanly.
6. Replay opens without invoking the solver.

This manual check is a release acceptance test, not a new scientific
validation claim.

## Tagging

After CI and the manual desktop acceptance pass on the committed release
revision:

```bash
git tag -a v2.0.1 -m "quadrotor MPC research workbench v2.0.1"
git push origin v2.0.1
```

Attach the source archive, validation report and checksum to the GitHub
Release. Tagging and publication are intentionally performed only after the
user reviews and commits the stabilization changes.

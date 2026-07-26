#!/usr/bin/env python3
"""Run or resume Stage 8 paired Monte Carlo validation on native MuJoCo."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from quadrotor_mpc.application.validation.monte_carlo import (
    NativeMonteCarloRunner,
    append_trial_checkpoint,
    create_validation_directory,
    finalize_validation_artifacts,
    load_native_monte_carlo_protocol,
    load_trial_checkpoint,
    protocol_fingerprint,
    run_native_trial_batch,
)
from quadrotor_mpc.infrastructure.resources import resolve_input_path
from quadrotor_mpc.interfaces.desktop.viewer import load_native_mujoco_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/native_monte_carlo.yaml")
    parser.add_argument(
        "--output-dir",
        help="override the protocol output directory",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_DIRECTORY",
        help="resume an interrupted directory after fingerprint validation",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="override trials per controller/noise level",
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        help="override the first paired seed",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel worker processes; default 1 is maximally reproducible",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="seeds per worker batch; each batch reuses one compiled controller",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="validate protocol and base native config, then exit",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow a non-release smoke campaign from an uncommitted source snapshot",
    )
    return parser


def _with_overrides(protocol, args):
    mapping = protocol.to_mapping()
    if args.output_dir:
        mapping["output_dir"] = str(Path(args.output_dir).resolve())
    if args.trials is not None:
        mapping["trials"] = args.trials
    if args.first_seed is not None:
        mapping["first_seed"] = args.first_seed
    # Avoid writing an implicit temporary configuration: reconstruct directly.
    from quadrotor_mpc.application.validation.monte_carlo import NativeMonteCarloProtocol

    levels = protocol.noise_levels
    return NativeMonteCarloProtocol(
        name=protocol.name,
        base_config_path=protocol.base_config_path,
        output_dir=Path(mapping["output_dir"]),
        modes=protocol.modes,
        noise_levels=levels,
        trials=int(mapping["trials"]),
        first_seed=int(mapping["first_seed"]),
        confidence_level=protocol.confidence_level,
        minimum_trials_for_claim=protocol.minimum_trials_for_claim,
        empirical_collision_rate_limit=protocol.empirical_collision_rate_limit,
        require_zero_positive_slack=protocol.require_zero_positive_slack,
        require_zero_fallback=protocol.require_zero_fallback,
        require_zero_budget_failures=protocol.require_zero_budget_failures,
        timing_percentile=protocol.timing_percentile,
    )


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", message="The ONNX feature is not available.*")
    warnings.filterwarnings("ignore", message="The opcua feature is not available.*")
    warnings.filterwarnings("ignore", message="The approximateMPC feature requires PyTorch.*")
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    protocol = _with_overrides(
        load_native_monte_carlo_protocol(resolve_input_path(args.config)),
        args,
    )
    base_config = load_native_mujoco_config(protocol.base_config_path)
    expected = protocol.trials * len(protocol.modes) * len(protocol.noise_levels)
    print(f"protocol:       {protocol.name}")
    print(f"base config:    {protocol.base_config_path}")
    print(f"paired seeds:   {protocol.seeds[0]}..{protocol.seeds[-1]}")
    print(f"planned trials: {expected}")
    if args.validate_config:
        print("configuration is valid")
        return 0

    if args.resume:
        directory = Path(args.resume).resolve()
        manifest_path = directory / "manifest.yaml"
        if not manifest_path.exists():
            raise SystemExit(f"resume manifest not found: {manifest_path}")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        current = protocol_fingerprint(protocol, base_config)
        if manifest.get("protocol_fingerprint") != current:
            raise SystemExit("resume rejected: protocol fingerprint does not match")
        from quadrotor_mpc.application.validation.monte_carlo import source_provenance

        recorded_source = manifest.get("source", {})
        current_source = source_provenance()
        if (
            recorded_source.get("source_snapshot_sha256")
            and recorded_source.get("source_snapshot_sha256")
            != current_source.get("source_snapshot_sha256")
        ):
            raise SystemExit("resume rejected: validation source snapshot does not match")
        trials = load_trial_checkpoint(directory)
        print(f"resume:         {directory} ({len(trials)}/{expected} complete)")
    else:
        directory = create_validation_directory(
            protocol,
            base_config,
            command=[sys.executable, *sys.argv],
            allow_dirty_source=args.allow_dirty_source,
        )
        trials = []
        print(f"output:         {directory}")

    completed = {
        (trial.noise_label, trial.mode, trial.seed)
        for trial in trials
    }
    def save_trial(trial) -> None:
        trials.append(trial)
        append_trial_checkpoint(directory, trial)
        completed.add((trial.noise_label, trial.mode, trial.seed))
        print(
            f"[{len(completed):03d}/{expected:03d}] "
            f"{trial.noise_label:<13} {trial.mode:<13} seed={trial.seed:<5d} "
            f"collision={int(trial.collision)} "
            f"clearance={trial.min_clearance_m:.3f}m "
            f"slack={trial.max_slack_m:.3f}m "
            f"p99={trial.p99_solver_ms:.1f}ms",
            flush=True,
        )

    if args.workers == 1:
        runner = NativeMonteCarloRunner(base_config)
        for level in protocol.noise_levels:
            for seed in protocol.seeds:
                for mode in protocol.modes:
                    key = (level.label, mode, seed)
                    if key in completed:
                        continue
                    save_trial(
                        runner.run_trial(
                            mode=mode,
                            noise_level=level,
                            seed=seed,
                        )
                    )
    else:
        batches = []
        for level in protocol.noise_levels:
            for mode in protocol.modes:
                seeds = [
                    seed
                    for seed in protocol.seeds
                    if (level.label, mode, seed) not in completed
                ]
                batches.extend(
                    (level, mode, seeds[index : index + args.batch_size])
                    for index in range(0, len(seeds), args.batch_size)
                )
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    run_native_trial_batch,
                    base_config.to_mapping(),
                    mode=mode,
                    noise_label=level.label,
                    covariance_scale=level.covariance_scale,
                    seeds=seeds,
                ): (level.label, mode, seeds)
                for level, mode, seeds in batches
            }
            for future in as_completed(futures):
                label, mode, seeds = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"native Monte Carlo batch failed: "
                        f"{label}/{mode}/seeds={seeds}"
                    ) from exc
                from quadrotor_mpc.application.validation.monte_carlo import NativeTrialResult

                for row in rows:
                    save_trial(NativeTrialResult(**row))

    artifacts = finalize_validation_artifacts(
        directory,
        trials,
        protocol,
        base_config,
    )
    aggregate = yaml.safe_load(artifacts["aggregate"].read_text(encoding="utf-8"))
    print(f"status:         {aggregate['overall']['stage_status']}")
    print(f"report:         {artifacts['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

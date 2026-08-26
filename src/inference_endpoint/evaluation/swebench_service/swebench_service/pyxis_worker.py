# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_bytes
from .pyxis_environment import PyxisStepLimits, resolve_image, run_srun_step
from .runner import RunnerError, load_pyxis_placement

_PRINT_LOCK = threading.Lock()
_INFRASTRUCTURE_FAILURE = ".pyxis_infrastructure_failure"
# This mirrors the per-instance portion of SWE-bench 4.1.0's Docker evaluator:
# apply the patch with the same three fallbacks, run eval.sh with its timeout, and
# preserve the test output for get_eval_report(). Pyxis mounts these files and
# executes them with srun instead of copying them through the Docker API.
_EVAL_SCRIPT = r"""set -eu

patch_path=$1
eval_path=$2
output_path=$3
timeout_s=$4

cd /testbed
if git apply --verbose "$patch_path" || \
    git apply --verbose --reject "$patch_path" || \
    patch --batch --fuzz=5 -p1 -i "$patch_path"; then
    echo ">>>>> Applied Patch"
else
    echo ">>>>> Patch Apply Failed"
    exit 1
fi

set +e
timeout "$timeout_s" /bin/bash "$eval_path" >"$output_path" 2>&1
status=$?
set -e
cat "$output_path"
if [[ $status -eq 124 ]]; then
    echo "Timeout error: $timeout_s seconds exceeded." >>"$output_path"
    exit 124
fi
exit 0
"""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def _run_agent(args: argparse.Namespace) -> None:
    # Generation-only dependencies are loaded only in the agent worker mode.
    from minisweagent.environments import get_environment
    from minisweagent.run.benchmarks import swebench

    failure_path = args.output / _INFRASTRUCTURE_FAILURE
    failure_path.unlink(missing_ok=True)

    def get_pyxis_environment(config: dict, instance: dict):
        environment_config = copy.deepcopy(config.get("environment", {}))
        environment_config["image"] = resolve_image(
            args.image_registry, instance["instance_id"]
        )
        if args.placement is not None:
            environment_config["node"] = args.placement.node_for(
                instance["instance_id"]
            )
            environment_config["shared_runtime_root"] = str(args.shared_runtime_root)
        environment_config["infrastructure_failure_path"] = str(failure_path)
        environment_config["step_limits"] = args.step_limits
        return get_environment(environment_config)

    if not hasattr(swebench, "get_sb_environment"):
        raise RuntimeError(
            "installed mini-swe-agent does not expose get_sb_environment"
        )
    original_get_sb_environment = swebench.get_sb_environment
    try:
        swebench.get_sb_environment = get_pyxis_environment
        swebench.main(
            subset=args.subset,
            split=args.split,
            slice_spec="",
            filter_spec=args.filter,
            shuffle=False,
            output=str(args.output),
            workers=args.workers,
            model=args.model,
            model_class=None,
            redo_existing=False,
            config_spec=[str(args.config)],
            environment_class="swebench_service.pyxis_environment.PyxisEnvironment",
        )
        if failure_path.exists():
            raise RunnerError("Pyxis infrastructure failure during agent execution")
    finally:
        swebench.get_sb_environment = original_get_sb_environment


def _evaluate_instance(
    *,
    test_spec: Any,
    prediction: dict[str, Any],
    image: str | Path,
    output_dir: Path,
    run_id: str,
    timeout_s: int,
    node: str | None = None,
    step_limits: PyxisStepLimits | None = None,
) -> None:
    instance_id = test_spec.instance_id
    safe_model = prediction["model_name_or_path"].replace("/", "__")
    log_dir = output_dir / "logs" / "run_evaluation" / run_id / safe_model / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    patch_path = log_dir / "patch.diff"
    eval_path = log_dir / "eval.sh"
    output_path = log_dir / "test_output.txt"
    report_path = log_dir / "report.json"
    patch_path.write_text(prediction["model_patch"])
    eval_path.write_text(test_spec.eval_script)
    output_path.write_text("")
    report_path.unlink(missing_ok=True)

    mounts = [
        (patch_path, "/tmp/swebench_patch.diff"),
        (eval_path, "/tmp/swebench_eval.sh"),
        (output_path, "/tmp/swebench_test_output.txt"),
    ]
    status_path = log_dir / ".mlperf_srun_status"
    mounts.append((status_path, "/tmp/.mlperf_srun_status"))
    result = run_srun_step(
        image=image,
        create=True,
        mounts=mounts,
        workdir="/testbed",
        status_path=status_path,
        timeout_s=timeout_s + 30,
        stderr=subprocess.PIPE,
        argv=[
            "bash",
            "-c",
            _EVAL_SCRIPT,
            "pyxis-eval",
            "/tmp/swebench_patch.diff",
            "/tmp/swebench_eval.sh",
            "/tmp/swebench_test_output.txt",
            str(timeout_s),
        ],
        node=node,
        step_limits=step_limits,
    )
    with _PRINT_LOCK:
        print(f"[{instance_id}]\n{result.stdout}{result.stderr}", flush=True)
    if result.returncode in {1, 124}:
        return
    if result.returncode != 0:
        raise RunnerError(
            f"unexpected Pyxis evaluation exit code for {instance_id}: "
            f"{result.returncode}"
        )

    # Grading is an evaluation-only dependency and is not needed in agent mode.
    from swebench.harness.grading import get_eval_report

    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path=output_path,
        include_tests_status=True,
    )
    atomic_write_bytes(report_path, (json.dumps(report, indent=4) + "\n").encode())


def _run_eval(args: argparse.Namespace) -> None:
    # Evaluation-only SWE-bench harness dependencies are not needed in agent mode.
    from swebench.harness.reporting import make_run_report
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.utils import (
        get_predictions_from_file,
        load_swebench_dataset,
    )

    predictions = {
        prediction["instance_id"]: prediction
        for prediction in get_predictions_from_file(
            str(args.predictions_path), args.dataset_name, args.split
        )
        if prediction["instance_id"] in args.instance_ids
    }
    rows = load_swebench_dataset(args.dataset_name, args.split, args.instance_ids)
    images = {
        instance_id: resolve_image(args.image_registry, instance_id)
        for instance_id in args.instance_ids
    }
    payloads = []
    for row in rows:
        instance_id = row["instance_id"]
        prediction = predictions.get(instance_id)
        if prediction is None or prediction.get("model_patch") in {"", None}:
            continue
        payloads.append(
            {
                "test_spec": make_test_spec(row, arch="arm64"),
                "prediction": prediction,
                "image": images[instance_id],
                "output_dir": args.output_dir,
                "run_id": args.run_id,
                "timeout_s": args.timeout,
                "node": (
                    args.placement.node_for(instance_id)
                    if args.placement is not None
                    else None
                ),
                "step_limits": args.step_limits,
            }
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(_evaluate_instance, **payload): payload[
                "test_spec"
            ].instance_id
            for payload in payloads
        }
        failures = []
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with _PRINT_LOCK:
                    print(f"Pyxis evaluation failed: {exc}", flush=True)
                failures.append(futures[future])
        if failures:
            raise RunnerError(
                "Pyxis infrastructure failure evaluating: "
                + ", ".join(sorted(failures))
            )

    output_dir = args.output_dir.resolve()
    with contextlib.chdir(output_dir):
        make_run_report(
            predictions,
            [{"instance_id": instance_id} for instance_id in args.instance_ids],
            args.run_id,
            client=None,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    def add_step_limit_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--max-concurrent-creates", type=_positive_int)
        command_parser.add_argument("--max-concurrent-srun-steps", type=_positive_int)
        command_parser.add_argument(
            "--srun-launch-grace-s", type=_nonnegative_int, default=30
        )

    agent_parser = commands.add_parser("agent")
    agent_parser.add_argument("--model", required=True)
    agent_parser.add_argument("--config", type=Path, required=True)
    agent_parser.add_argument("--subset", required=True)
    agent_parser.add_argument("--split", required=True)
    agent_parser.add_argument("--filter", required=True)
    agent_parser.add_argument("--workers", type=int, required=True)
    agent_parser.add_argument("--output", type=Path, required=True)
    agent_parser.add_argument("--image-registry", required=True)
    agent_parser.add_argument("--placement-file", type=Path)
    agent_parser.add_argument("--shared-runtime-root", type=Path)
    add_step_limit_arguments(agent_parser)

    eval_parser = commands.add_parser("eval")
    eval_parser.add_argument("--dataset-name", required=True)
    eval_parser.add_argument("--split", required=True)
    eval_parser.add_argument("--predictions-path", type=Path, required=True)
    eval_parser.add_argument("--max-workers", type=int, required=True)
    eval_parser.add_argument("--run-id", required=True)
    eval_parser.add_argument("--image-registry", required=True)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument("--timeout", type=int, default=1800)
    eval_parser.add_argument("--instance-ids", nargs="+", required=True)
    eval_parser.add_argument("--placement-file", type=Path)
    eval_parser.add_argument("--shared-runtime-root", type=Path)
    add_step_limit_arguments(eval_parser)
    args = parser.parse_args(argv)

    if (args.placement_file is None) != (args.shared_runtime_root is None):
        parser.error("--placement-file and --shared-runtime-root must be used together")
    if args.shared_runtime_root is not None:
        if not args.shared_runtime_root.is_absolute():
            parser.error("--shared-runtime-root must be absolute")
        args.shared_runtime_root = args.shared_runtime_root.resolve()
        artifact_path = (
            args.output if args.command == "agent" else args.output_dir
        ).resolve()
        try:
            artifact_path.relative_to(args.shared_runtime_root)
        except ValueError:
            parser.error(
                "Pyxis placement requires agent and evaluation artifacts under "
                "--shared-runtime-root"
            )
    args.placement = (
        load_pyxis_placement(args.placement_file)
        if args.placement_file is not None
        else None
    )
    args.step_limits = PyxisStepLimits(
        max_concurrent_creates=args.max_concurrent_creates,
        max_concurrent_srun_steps=args.max_concurrent_srun_steps,
        launch_grace_s=args.srun_launch_grace_s,
    )

    if args.command == "agent":
        _run_agent(args)
    else:
        _run_eval(args)


if __name__ == "__main__":
    main()

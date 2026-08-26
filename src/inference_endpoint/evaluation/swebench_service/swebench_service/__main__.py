# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from pathlib import Path

from aiohttp import web

from .config import ServiceConfig
from .runner import create_runner
from .server import create_app


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SWE-bench service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--artifact-root", default="swebench_service_artifacts")
    parser.add_argument("--max-concurrent-runs", type=int, default=1)
    parser.add_argument("--subprocess-timeout-s", type=int, default=24 * 60 * 60)
    parser.add_argument("--runtime", choices=("docker", "pyxis"), default="docker")
    parser.add_argument("--image-registry")
    parser.add_argument(
        "--pyxis-placement-file",
        type=Path,
        help="trusted TSV mapping each requested SWE-bench instance ID to a Slurm node",
    )
    parser.add_argument(
        "--pyxis-shared-runtime-root",
        type=Path,
        help="absolute shared filesystem root for multi-node Pyxis artifacts",
    )
    parser.add_argument(
        "--pyxis-max-concurrent-creates",
        type=_positive_int,
        help="maximum simultaneous Pyxis container creates in each worker process",
    )
    parser.add_argument(
        "--pyxis-max-concurrent-srun-steps",
        type=_positive_int,
        help="maximum simultaneous nested srun steps in each worker process",
    )
    parser.add_argument(
        "--pyxis-srun-launch-grace-s",
        type=_nonnegative_int,
        default=30,
        help="extra time in the outer srun deadline for Slurm step startup",
    )
    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument("--auth-token")
    auth_group.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--max-stored-runs", type=int, default=100)
    args = parser.parse_args()

    config = ServiceConfig(
        host=args.host,
        port=args.port,
        artifact_root=Path(args.artifact_root),
        max_concurrent_runs=args.max_concurrent_runs,
        subprocess_timeout_s=args.subprocess_timeout_s,
        auth_token=args.auth_token,
        allow_unauthenticated=args.allow_unauthenticated,
        max_stored_runs=args.max_stored_runs,
        pyxis_placement_file=args.pyxis_placement_file,
        pyxis_shared_runtime_root=args.pyxis_shared_runtime_root,
        pyxis_max_concurrent_creates=args.pyxis_max_concurrent_creates,
        pyxis_max_concurrent_srun_steps=args.pyxis_max_concurrent_srun_steps,
        pyxis_srun_launch_grace_s=args.pyxis_srun_launch_grace_s,
    )
    runner = create_runner(
        args.runtime,
        project_root=Path(__file__).resolve().parents[1],
        subprocess_timeout_s=config.subprocess_timeout_s,
        image_registry=args.image_registry,
        pyxis_placement_file=config.pyxis_placement_file,
        pyxis_shared_runtime_root=config.pyxis_shared_runtime_root,
        pyxis_max_concurrent_creates=config.pyxis_max_concurrent_creates,
        pyxis_max_concurrent_srun_steps=config.pyxis_max_concurrent_srun_steps,
        pyxis_srun_launch_grace_s=config.pyxis_srun_launch_grace_s,
    )
    web.run_app(create_app(config, runner=runner), host=config.host, port=config.port)


if __name__ == "__main__":
    main()

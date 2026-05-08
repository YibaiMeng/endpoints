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

"""push subcommands: `inference-endpoint push run <path>`."""

from __future__ import annotations

import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Annotated

import cyclopts
import httpx
import yaml

from inference_endpoint.exceptions import ExecutionError, InputValidationError

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False

logger = logging.getLogger(__name__)

push_app = cyclopts.App(
    name="push",
    help="Push benchmark artifacts to the MLCommons endpoint.",
)

_REQUIRED_FILES = [
    "config.yaml",
    "result_summary.json",
    "runtime_settings.json",
    "events.jsonl",
]

_DEFAULT_API_URL = "http://localhost:8082"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_run_path(path: Path) -> Path:
    """Resolve the run folder from a directory or YAML file path."""
    if path.is_dir():
        return path
    if path.suffix in (".yaml", ".yml") and path.is_file():
        with path.open() as fh:
            cfg = yaml.safe_load(fh) or {}
        report_dir = cfg.get("report_dir")
        if report_dir:
            resolved = Path(report_dir)
            if resolved.is_dir():
                return resolved
        logger.warning(
            "report_dir not found in %s — falling back to cwd", path
        )
        return Path.cwd()
    if not path.exists():
        raise InputValidationError(f"Path does not exist: {path}")
    raise InputValidationError(
        f"Path must be a run result directory or a YAML config file: {path}"
    )


def _validate_run_dir(run_dir: Path) -> None:
    missing = [f for f in _REQUIRED_FILES if not (run_dir / f).exists()]
    if missing:
        raise InputValidationError(
            f"Run folder is missing required files: {', '.join(missing)}\n"
            f"  Run folder: {run_dir}"
        )


def _create_archive(run_dir: Path, dest: Path) -> None:
    """Pack the run folder contents (not the folder itself) into a .tar.gz."""
    with tarfile.open(dest, "w:gz") as tf:
        for item in run_dir.iterdir():
            tf.add(item, arcname=item.name)


def _masked_token(token: str) -> str:
    return f"****{token[-4:]}" if len(token) >= 4 else "****"


def _upload(archive_path: Path, token: str, api_url: str) -> httpx.Response:
    """Synchronous HTTP upload with a progress indicator."""
    url = f"{api_url.rstrip('/')}/push_run"
    total = archive_path.stat().st_size

    try:
        with archive_path.open("rb") as fh:
            if _HAS_TQDM:
                assert _tqdm is not None
                with _tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc="Uploading",
                    ncols=72,
                ) as pbar:

                    class _ProgressReader:
                        def read(self, n: int = -1) -> bytes:
                            chunk = fh.read(n)
                            pbar.update(len(chunk))
                            return chunk

                    return httpx.post(
                        url,
                        params={"token": token},
                        files={"archive": (archive_path.name, _ProgressReader())},
                        timeout=300.0,
                    )
            else:
                print("Uploading...")
                return httpx.post(
                    url,
                    params={"token": token},
                    files={"archive": (archive_path.name, fh)},
                    timeout=300.0,
                )
    except httpx.TimeoutException as exc:
        raise ExecutionError(f"Upload timed out: {exc}") from exc
    except httpx.RequestError as exc:
        raise ExecutionError(f"Upload failed - could not reach {url}: {exc}") from exc


def _handle_response(resp: httpx.Response) -> None:
    """Map HTTP response codes to CLI exceptions or success output."""
    status = resp.status_code

    if status == 201:
        try:
            run_id = resp.json().get("id", "<unknown>")
        except Exception:
            run_id = "<unknown>"
        print(f"Run submitted successfully. Run ID: {run_id}")
        return

    try:
        body = resp.json()
        detail = body.get("detail", resp.text)
    except Exception:
        detail = resp.text

    _map: dict[int, tuple[type[Exception], str]] = {
        400: (
            InputValidationError,
            "Token format is invalid (must be 68 chars, mlc_ prefix)",
        ),
        403: (
            InputValidationError,
            "Token not authorized for the Endpoints service",
        ),
        500: (
            ExecutionError,
            "Server error — this is likely a bug, please report it",
        ),
    }

    if status in _map:
        exc_cls, msg = _map[status]
        raise exc_cls(msg)

    if status == 401:
        raise InputValidationError(f"Authentication failed: {detail}")

    if status == 422:
        raise InputValidationError(f"Validation failed: {detail}")

    if status == 429:
        retry_after = 0
        if isinstance(detail, dict):
            retry_after = detail.get("retry_after_seconds", 0)
        raise ExecutionError(f"Rate limited — retry after {retry_after}s")

    if status == 502:
        raise ExecutionError(f"Upstream service unavailable: {detail}")

    raise ExecutionError(f"Push failed [{status}]: {detail}")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@push_app.command
def run(
    *,
    path: Annotated[
        Path,
        cyclopts.Parameter(
            name=["--path", "-p"],
            help="Path to the run result folder (or a config.yaml whose report_dir points to it).",
        ),
    ],
    token: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--token", "-t"],
            env_var="ENDPOINTS_TOKEN",
            help="PRISM API key. Also read from ENDPOINTS_TOKEN env var.",
        ),
    ] = None,
    api_url: Annotated[
        str,
        cyclopts.Parameter(
            name="--api-url",
            help="Override the push API base URL.",
        ),
    ] = _DEFAULT_API_URL,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(
            name="--dry-run",
            help="Package and validate locally; do NOT upload.",
        ),
    ] = False,
) -> None:
    """Package and push a benchmark run folder to the MLCommons endpoint service."""
    # 1. Resolve token
    resolved_token = token or os.environ.get("ENDPOINTS_TOKEN", "")
    if not resolved_token:
        raise InputValidationError(
            "Token is required. Pass --token or set ENDPOINTS_TOKEN env var"
        )

    # 2. Resolve run directory
    run_dir = _resolve_run_path(path)
    logger.info("Run folder: %s", run_dir)

    # 3. Pre-flight validation
    _validate_run_dir(run_dir)

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / f"{run_dir.name}.tar.gz"
        _create_archive(run_dir, archive_path)
        size_mb = archive_path.stat().st_size / 1024 / 1024

        if dry_run:
            print(f"Dry run — no upload will be performed.")
            print(f"  Archive : {archive_path.name}")
            print(f"  Size    : {size_mb:.2f} MB")
            print(f"  Token   : {_masked_token(resolved_token)}")
            print("  Contents:")
            with tarfile.open(archive_path, "r:gz") as tf:
                for member in tf.getmembers():
                    print(f"    {member.name}")
            return

        # 4. Upload
        try:
            resp = _upload(archive_path, resolved_token, api_url)
        finally:
            # archive lives inside tempdir so it's cleaned up automatically,
            # but explicit cleanup ensures clarity
            if archive_path.exists():
                archive_path.unlink()

        _handle_response(resp)

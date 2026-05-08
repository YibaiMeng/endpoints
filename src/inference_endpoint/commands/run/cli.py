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

"""Run subcommands: list / get / delete / pin / unpin.

Each verb is its own cyclopts.App registered in main.py:
    inference-endpoint list   run --token <tok>
    inference-endpoint get    run --token <tok> --run_id <id>
    inference-endpoint delete run --token <tok> --run_id <id>
    inference-endpoint pin    run --token <tok> --run_id <id>
    inference-endpoint unpin  run --token <tok> --run_id <id>
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated

import cyclopts
import httpx

from inference_endpoint.exceptions import ExecutionError, InputValidationError

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "http://localhost:8082"

# ---------------------------------------------------------------------------
# Apps — one per verb so the CLI reads: inference-endpoint <verb> run ...
# ---------------------------------------------------------------------------

list_app = cyclopts.App(name="list", help="List benchmark artifacts.")
get_app = cyclopts.App(name="get", help="Get a benchmark artifact.")
delete_app = cyclopts.App(name="delete", help="Delete a benchmark artifact.")
pin_app = cyclopts.App(name="pin", help="Pin a benchmark artifact.")
unpin_app = cyclopts.App(name="unpin", help="Unpin a benchmark artifact.")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_token(token: str | None) -> str:
    resolved = token or os.environ.get("ENDPOINTS_TOKEN", "")
    if not resolved:
        raise InputValidationError(
            "Token is required. Pass --token or set ENDPOINTS_TOKEN env var"
        )
    return resolved


def _handle_error(resp: httpx.Response) -> None:
    """Raise an appropriate CLI exception for any non-success response."""
    status = resp.status_code
    try:
        body = resp.json()
        detail = body.get("detail", resp.text)
    except Exception:  # noqa: BLE001 — best-effort JSON parse
        detail = resp.text

    if status == 400:
        raise InputValidationError(f"Bad request: {detail}")
    if status == 401:
        raise InputValidationError(f"Authentication failed: {detail}")
    if status == 403:
        raise InputValidationError("Token not authorized for the Endpoints service")
    if status == 404:
        raise InputValidationError(f"Run not found: {detail}")
    if status == 429:
        retry_after = 0
        if isinstance(detail, dict):
            retry_after = detail.get("retry_after_seconds", 0)
        raise ExecutionError(f"Rate limited — retry after {retry_after}s")
    if status == 500:
        raise ExecutionError(f"Server error: {detail}")
    if status == 502:
        raise ExecutionError(f"Upstream service unavailable: {detail}")
    raise ExecutionError(f"Request failed [{status}]: {detail}")


def _get(url: str, params: dict) -> httpx.Response:
    try:
        return httpx.get(url, params=params, timeout=30.0)
    except httpx.TimeoutException as exc:
        raise ExecutionError(f"Request timed out: {exc}") from exc
    except httpx.RequestError as exc:
        raise ExecutionError(f"Could not reach API: {exc}") from exc


def _delete(url: str, params: dict) -> httpx.Response:
    try:
        return httpx.delete(url, params=params, timeout=30.0)
    except httpx.TimeoutException as exc:
        raise ExecutionError(f"Request timed out: {exc}") from exc
    except httpx.RequestError as exc:
        raise ExecutionError(f"Could not reach API: {exc}") from exc


def _patch(url: str, params: dict) -> httpx.Response:
    try:
        return httpx.patch(url, params=params, timeout=30.0)
    except httpx.TimeoutException as exc:
        raise ExecutionError(f"Request timed out: {exc}") from exc
    except httpx.RequestError as exc:
        raise ExecutionError(f"Could not reach API: {exc}") from exc


# ---------------------------------------------------------------------------
# list run
# ---------------------------------------------------------------------------


def list_run(
    token: str | None = None,
    api_url: str = _DEFAULT_API_URL,
) -> None:
    """List all benchmark runs for the authenticated user."""
    resolved_token = _resolve_token(token)
    resp = _get(f"{api_url.rstrip('/')}/runs", {"token": resolved_token})
    if resp.status_code == 200:
        runs = resp.json()
        if not runs:
            print("No runs found.")
            return
        print(json.dumps(runs, indent=2))
        return
    _handle_error(resp)


@list_app.command(name="run")
def _list_run_cmd(
    *,
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
        cyclopts.Parameter(name="--api-url", help="Override the API base URL."),
    ] = _DEFAULT_API_URL,
) -> None:
    """List all benchmark runs for the authenticated user."""
    list_run(token=token, api_url=api_url)


# ---------------------------------------------------------------------------
# get run
# ---------------------------------------------------------------------------


def get_run(
    token: str | None = None,
    run_id: str = "",
    api_url: str = _DEFAULT_API_URL,
) -> None:
    """Get a single benchmark run by ID."""
    resolved_token = _resolve_token(token)
    resp = _get(
        f"{api_url.rstrip('/')}/runs/{run_id}",
        {"token": resolved_token},
    )
    if resp.status_code == 200:
        print(json.dumps(resp.json(), indent=2))
        return
    _handle_error(resp)


@get_app.command(name="run")
def _get_run_cmd(
    *,
    token: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--token", "-t"],
            env_var="ENDPOINTS_TOKEN",
            help="PRISM API key. Also read from ENDPOINTS_TOKEN env var.",
        ),
    ] = None,
    run_id: Annotated[
        str,
        cyclopts.Parameter(name="--run_id", help="Run UUID to retrieve."),
    ],
    api_url: Annotated[
        str,
        cyclopts.Parameter(name="--api-url", help="Override the API base URL."),
    ] = _DEFAULT_API_URL,
) -> None:
    """Get a single benchmark run by ID."""
    get_run(token=token, run_id=run_id, api_url=api_url)


# ---------------------------------------------------------------------------
# delete run
# ---------------------------------------------------------------------------


def delete_run(
    token: str | None = None,
    run_id: str = "",
    api_url: str = _DEFAULT_API_URL,
) -> None:
    """Delete a benchmark run by ID."""
    resolved_token = _resolve_token(token)
    resp = _delete(
        f"{api_url.rstrip('/')}/runs/{run_id}",
        {"token": resolved_token},
    )
    if resp.status_code == 200:
        print(f"Deleted run: {run_id}")
        return
    _handle_error(resp)


@delete_app.command(name="run")
def _delete_run_cmd(
    *,
    token: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--token", "-t"],
            env_var="ENDPOINTS_TOKEN",
            help="PRISM API key. Also read from ENDPOINTS_TOKEN env var.",
        ),
    ] = None,
    run_id: Annotated[
        str,
        cyclopts.Parameter(name="--run_id", help="Run UUID to delete."),
    ],
    api_url: Annotated[
        str,
        cyclopts.Parameter(name="--api-url", help="Override the API base URL."),
    ] = _DEFAULT_API_URL,
) -> None:
    """Delete a benchmark run by ID."""
    delete_run(token=token, run_id=run_id, api_url=api_url)


# ---------------------------------------------------------------------------
# pin run
# ---------------------------------------------------------------------------


def pin_run(
    token: str | None = None,
    run_id: str = "",
    api_url: str = _DEFAULT_API_URL,
) -> None:
    """Pin a benchmark run so it is not auto-expired."""
    resolved_token = _resolve_token(token)
    resp = _patch(
        f"{api_url.rstrip('/')}/runs/{run_id}/pin",
        {"token": resolved_token},
    )
    if resp.status_code == 200:
        print(f"Pinned run: {run_id}")
        return
    _handle_error(resp)


@pin_app.command(name="run")
def _pin_run_cmd(
    *,
    token: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--token", "-t"],
            env_var="ENDPOINTS_TOKEN",
            help="PRISM API key. Also read from ENDPOINTS_TOKEN env var.",
        ),
    ] = None,
    run_id: Annotated[
        str,
        cyclopts.Parameter(name="--run_id", help="Run UUID to pin."),
    ],
    api_url: Annotated[
        str,
        cyclopts.Parameter(name="--api-url", help="Override the API base URL."),
    ] = _DEFAULT_API_URL,
) -> None:
    """Pin a benchmark run so it is not auto-expired."""
    pin_run(token=token, run_id=run_id, api_url=api_url)


# ---------------------------------------------------------------------------
# unpin run
# ---------------------------------------------------------------------------


def unpin_run(
    token: str | None = None,
    run_id: str = "",
    api_url: str = _DEFAULT_API_URL,
) -> None:
    """Unpin a benchmark run, allowing it to be auto-expired."""
    resolved_token = _resolve_token(token)
    resp = _patch(
        f"{api_url.rstrip('/')}/runs/{run_id}/unpin",
        {"token": resolved_token},
    )
    if resp.status_code == 200:
        print(f"Unpinned run: {run_id}")
        return
    _handle_error(resp)


@unpin_app.command(name="run")
def _unpin_run_cmd(
    *,
    token: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--token", "-t"],
            env_var="ENDPOINTS_TOKEN",
            help="PRISM API key. Also read from ENDPOINTS_TOKEN env var.",
        ),
    ] = None,
    run_id: Annotated[
        str,
        cyclopts.Parameter(name="--run_id", help="Run UUID to unpin."),
    ],
    api_url: Annotated[
        str,
        cyclopts.Parameter(name="--api-url", help="Override the API base URL."),
    ] = _DEFAULT_API_URL,
) -> None:
    """Unpin a benchmark run, allowing it to be auto-expired."""
    unpin_run(token=token, run_id=run_id, api_url=api_url)

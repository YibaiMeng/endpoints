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

"""Run and submission subcommands: list / get / delete / pin / unpin / (submission list+get).

Each verb is its own cyclopts.App registered in main.py:
    inference-endpoint list   run        --token <tok>
    inference-endpoint get    run        --token <tok> --run_id <id>
    inference-endpoint delete run        --token <tok> --run_id <id>
    inference-endpoint pin    run        --token <tok> --run_id <id>
    inference-endpoint unpin  run        --token <tok> --run_id <id>
    inference-endpoint list   submission --token <tok>
    inference-endpoint get    submission --token <tok> --submission_id <id>
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Annotated, Any

import cyclopts
import httpx
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from inference_endpoint.exceptions import ExecutionError, InputValidationError

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "http://localhost:8082"
_console = Console()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_dt(s: str | None) -> str:
    dt = _parse_dt(s)
    if dt is None:
        return "—"
    return dt.astimezone(tz=None).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(started: str | None, finished: str | None) -> str:
    s, f = _parse_dt(started), _parse_dt(finished)
    if s is None or f is None:
        return "—"
    secs = int((f - s).total_seconds())
    if secs < 60:
        return f"{secs}s"
    m, s2 = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s2}s"
    h, m2 = divmod(m, 60)
    return f"{h}h {m2}m"


def _dash(v: Any) -> str:
    if v is None:
        return "—"
    return str(v)


def _print_list_runs(runs: list[dict[str, Any]]) -> None:
    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Concurrency", justify="right")
    table.add_column("Started", no_wrap=True, min_width=19)
    table.add_column("Finished", no_wrap=True, min_width=19)
    table.add_column("Duration", justify="right")

    for i, run in enumerate(runs, 1):
        table.add_row(
            str(i),
            run.get("id", "—"),
            _dash(run.get("model")),
            _dash(run.get("concurrency")),
            _fmt_dt(run.get("started_at")),
            _fmt_dt(run.get("finished_at")),
            _fmt_duration(run.get("started_at"), run.get("finished_at")),
        )

    _console.print(table)


def _print_run(run: dict[str, Any]) -> None:
    _console.print(
        Syntax(json.dumps(run, indent=2), "json", theme="monokai", background_color="default")
    )


def _print_list_submissions(subs: list[dict[str, Any]]) -> None:
    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Division", no_wrap=True)
    table.add_column("Availability", no_wrap=True)
    table.add_column("Version", no_wrap=True)
    table.add_column("Cycle", no_wrap=True)
    table.add_column("Runs", justify="right")

    for i, sub in enumerate(subs, 1):
        run_ids = sub.get("run_ids") or []
        table.add_row(
            str(i),
            sub.get("id", "—"),
            sub.get("status", "—"),
            sub.get("division", "—"),
            sub.get("availability", "—"),
            sub.get("benchmark_version", "—"),
            sub.get("publication_cycle", "—"),
            str(len(run_ids)),
        )

    _console.print(table)


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
        raise InputValidationError(f"Not found: {detail}")
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
    json_output: bool = False,
) -> None:
    """List all benchmark runs for the authenticated user."""
    resolved_token = _resolve_token(token)
    resp = _get(f"{api_url.rstrip('/')}/runs", {"token": resolved_token})
    if resp.status_code == 200:
        runs = resp.json()
        if not runs:
            print("No runs found.")
            return
        if json_output:
            print(json.dumps(runs, indent=2))
        else:
            _print_list_runs(runs)
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
    json_output: Annotated[
        bool,
        cyclopts.Parameter(name=["-j", "--json"], help="Output raw JSON."),
    ] = False,
) -> None:
    """List all benchmark runs for the authenticated user."""
    list_run(token=token, api_url=api_url, json_output=json_output)


# ---------------------------------------------------------------------------
# get run
# ---------------------------------------------------------------------------


def get_run(
    token: str | None = None,
    run_id: str = "",
    api_url: str = _DEFAULT_API_URL,
    json_output: bool = False,
) -> None:
    """Get a single benchmark run by ID."""
    resolved_token = _resolve_token(token)
    resp = _get(
        f"{api_url.rstrip('/')}/runs/{run_id}",
        {"token": resolved_token},
    )
    if resp.status_code == 200:
        run = resp.json()
        if json_output:
            print(json.dumps(run, indent=2))
        else:
            _print_run(run)
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
    json_output: Annotated[
        bool,
        cyclopts.Parameter(name=["-j", "--json"], help="Output raw JSON."),
    ] = False,
) -> None:
    """Get a single benchmark run by ID."""
    get_run(token=token, run_id=run_id, api_url=api_url, json_output=json_output)


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


# ---------------------------------------------------------------------------
# list submission
# ---------------------------------------------------------------------------


def list_submissions(
    token: str | None = None,
    api_url: str = _DEFAULT_API_URL,
    json_output: bool = False,
) -> None:
    """List all submissions for the authenticated user."""
    resolved_token = _resolve_token(token)
    resp = _get(f"{api_url.rstrip('/')}/submissions", {"token": resolved_token})
    if resp.status_code == 200:
        subs = resp.json()
        if not subs:
            print("No submissions found.")
            return
        if json_output:
            print(json.dumps(subs, indent=2))
        else:
            _print_list_submissions(subs)
        return
    _handle_error(resp)


@list_app.command(name="submission")
def _list_submission_cmd(
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
    json_output: Annotated[
        bool,
        cyclopts.Parameter(name=["-j", "--json"], help="Output raw JSON."),
    ] = False,
) -> None:
    """List all MLCommons submissions for the authenticated user."""
    list_submissions(token=token, api_url=api_url, json_output=json_output)


# ---------------------------------------------------------------------------
# get submission
# ---------------------------------------------------------------------------


def get_submission(
    token: str | None = None,
    submission_id: str = "",
    include_runs: bool = True,
    api_url: str = _DEFAULT_API_URL,
    json_output: bool = False,
) -> None:
    """Get a single submission by ID."""
    resolved_token = _resolve_token(token)
    resp = _get(
        f"{api_url.rstrip('/')}/submissions/{submission_id}",
        {"token": resolved_token, "include_runs": str(include_runs).lower()},
    )
    if resp.status_code == 200:
        sub = resp.json()
        if json_output:
            print(json.dumps(sub, indent=2))
        else:
            _print_run(sub)
        return
    _handle_error(resp)


@get_app.command(name="submission")
def _get_submission_cmd(
    *,
    token: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--token", "-t"],
            env_var="ENDPOINTS_TOKEN",
            help="PRISM API key. Also read from ENDPOINTS_TOKEN env var.",
        ),
    ] = None,
    submission_id: Annotated[
        str,
        cyclopts.Parameter(name="--submission_id", help="Submission UUID to retrieve."),
    ],
    include_runs: Annotated[
        str,
        cyclopts.Parameter(
            name="--include_runs", help="Include embedded run details (true/false)."
        ),
    ] = "true",
    api_url: Annotated[
        str,
        cyclopts.Parameter(name="--api-url", help="Override the API base URL."),
    ] = _DEFAULT_API_URL,
    json_output: Annotated[
        bool,
        cyclopts.Parameter(name=["-j", "--json"], help="Output raw JSON."),
    ] = False,
) -> None:
    """Get a single MLCommons submission by ID."""
    get_submission(
        token=token,
        submission_id=submission_id,
        include_runs=include_runs.lower() not in ("false", "0", "no"),
        api_url=api_url,
        json_output=json_output,
    )

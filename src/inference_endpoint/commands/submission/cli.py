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

"""Submission subcommands: create / update / withdraw.

Each verb is its own cyclopts.App registered in main.py:
    inference-endpoint create   submission --token <tok> ...
    inference-endpoint update   submission --token <tok> --submission_id <id> ...
    inference-endpoint withdraw submission --token <tok> --submission_id <id>
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any

import cyclopts
import httpx
from rich.console import Console
from rich.syntax import Syntax

from inference_endpoint.exceptions import ExecutionError, InputValidationError

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "http://localhost:8082"
_console = Console()

create_app = cyclopts.App(name="create", help="Create a benchmark artifact.")
update_app = cyclopts.App(name="update", help="Update a benchmark artifact.")
withdraw_app = cyclopts.App(name="withdraw", help="Withdraw a benchmark artifact.")

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
        raise InputValidationError(f"Submission not found: {detail}")
    if status == 422:
        raise InputValidationError(f"Validation error: {detail}")
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


def _post(url: str, params: dict, json_body: dict) -> httpx.Response:
    try:
        return httpx.post(url, params=params, json=json_body, timeout=30.0)
    except httpx.TimeoutException as exc:
        raise ExecutionError(f"Request timed out: {exc}") from exc
    except httpx.RequestError as exc:
        raise ExecutionError(f"Could not reach API: {exc}") from exc


def _patch(url: str, params: dict, json_body: dict) -> httpx.Response:
    try:
        return httpx.patch(url, params=params, json=json_body, timeout=30.0)
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


def _print_submission(sub: dict[str, Any]) -> None:
    _console.print(
        Syntax(
            json.dumps(sub, indent=2),
            "json",
            theme="monokai",
            background_color="default",
        )
    )


# ---------------------------------------------------------------------------
# create submission
# ---------------------------------------------------------------------------


def create_submission(
    token: str | None = None,
    availability: str = "available",
    benchmark_version: str = "",
    division: str = "standardized",
    early_publish: bool = False,
    publication_cycle: str | None = None,
    target_availability_date: str | None = None,
    run_ids: list[str] | None = None,
    api_url: str = _DEFAULT_API_URL,
) -> None:
    """Create a new MLCommons submission."""
    resolved_token = _resolve_token(token)
    body: dict[str, Any] = {
        "availability": availability,
        "benchmark_version": benchmark_version,
        "division": division,
        "early_publish": early_publish,
        "run_ids": run_ids or [],
    }
    if publication_cycle is not None:
        body["publication_cycle"] = publication_cycle
    if target_availability_date is not None:
        body["target_availability_date"] = target_availability_date
    resp = _post(
        f"{api_url.rstrip('/')}/submissions",
        {"token": resolved_token},
        body,
    )
    if resp.status_code in (200, 201):
        sub = resp.json()
        print(f"Created submission: {sub.get('id', '?')}")
        _print_submission(sub)
        return
    _handle_error(resp)


@create_app.command(name="submission")
def _create_submission_cmd(
    *,
    token: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--token", "-t"],
            env_var="ENDPOINTS_TOKEN",
            help="PRISM API key. Also read from ENDPOINTS_TOKEN env var.",
        ),
    ] = None,
    availability: Annotated[
        str,
        cyclopts.Parameter(name="--availability", help="Availability (e.g. available)."),
    ] = "available",
    benchmark_version: Annotated[
        str,
        cyclopts.Parameter(name="--benchmark_version", help="Benchmark version string."),
    ],
    division: Annotated[
        str,
        cyclopts.Parameter(name="--division", help="Division (e.g. standardized)."),
    ] = "standardized",
    early_publish: Annotated[
        bool,
        cyclopts.Parameter(name="--early_publish", help="Allow early publication."),
    ] = False,
    publication_cycle: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--publication_cycle", help="Publication cycle (e.g. 2025-04-C1)."
        ),
    ] = None,
    target_availability_date: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--target_availability_date",
            help="ISO date for target availability (e.g. 2025-06-01).",
        ),
    ] = None,
    run_ids: Annotated[
        list[str] | None,
        cyclopts.Parameter(name="--run_id", help="Run UUID to include. Repeatable."),
    ] = None,
    api_url: Annotated[
        str,
        cyclopts.Parameter(name="--api-url", help="Override the API base URL."),
    ] = _DEFAULT_API_URL,
) -> None:
    """Create a new MLCommons submission."""
    create_submission(
        token=token,
        availability=availability,
        benchmark_version=benchmark_version,
        division=division,
        early_publish=early_publish,
        publication_cycle=publication_cycle,
        target_availability_date=target_availability_date,
        run_ids=run_ids,
        api_url=api_url,
    )


# ---------------------------------------------------------------------------
# update submission
# ---------------------------------------------------------------------------


def update_submission(
    token: str | None = None,
    submission_id: str = "",
    status: str | None = None,
    availability_qualified_at: str | None = None,
    compliance_passed_at: str | None = None,
    first_published_at: str | None = None,
    peer_review_started_at: str | None = None,
    objection_resolution_started_at: str | None = None,
    finalized_at: str | None = None,
    pr_url: str | None = None,
    pr_number: int | None = None,
    archive_uri: str | None = None,
    publication_cycle: str | None = None,
    target_availability_date: str | None = None,
    api_url: str = _DEFAULT_API_URL,
) -> None:
    """Update a submission."""
    resolved_token = _resolve_token(token)
    candidates: dict[str, Any] = {
        "status": status,
        "availability_qualified_at": availability_qualified_at,
        "compliance_passed_at": compliance_passed_at,
        "first_published_at": first_published_at,
        "peer_review_started_at": peer_review_started_at,
        "objection_resolution_started_at": objection_resolution_started_at,
        "finalized_at": finalized_at,
        "pr_url": pr_url,
        "pr_number": pr_number,
        "archive_uri": archive_uri,
        "publication_cycle": publication_cycle,
        "target_availability_date": target_availability_date,
    }
    body = {k: v for k, v in candidates.items() if v is not None}
    resp = _patch(
        f"{api_url.rstrip('/')}/submissions/{submission_id}",
        {"token": resolved_token},
        body,
    )
    if resp.status_code == 200:
        sub = resp.json()
        print(f"Updated submission: {submission_id}")
        _print_submission(sub)
        return
    _handle_error(resp)


@update_app.command(name="submission")
def _update_submission_cmd(
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
        cyclopts.Parameter(name="--submission_id", help="Submission UUID to update."),
    ],
    status: Annotated[
        str | None,
        cyclopts.Parameter(name="--status", help="New status string."),
    ] = None,
    availability_qualified_at: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--availability_qualified_at",
            help="ISO datetime when availability was qualified.",
        ),
    ] = None,
    compliance_passed_at: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--compliance_passed_at",
            help="ISO datetime when compliance passed.",
        ),
    ] = None,
    first_published_at: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--first_published_at",
            help="ISO datetime of first publication.",
        ),
    ] = None,
    peer_review_started_at: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--peer_review_started_at",
            help="ISO datetime when peer review started.",
        ),
    ] = None,
    objection_resolution_started_at: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--objection_resolution_started_at",
            help="ISO datetime when objection resolution started.",
        ),
    ] = None,
    finalized_at: Annotated[
        str | None,
        cyclopts.Parameter(name="--finalized_at", help="ISO datetime when finalized."),
    ] = None,
    pr_url: Annotated[
        str | None,
        cyclopts.Parameter(name="--pr_url", help="GitHub PR URL."),
    ] = None,
    pr_number: Annotated[
        int | None,
        cyclopts.Parameter(name="--pr_number", help="GitHub PR number."),
    ] = None,
    archive_uri: Annotated[
        str | None,
        cyclopts.Parameter(name="--archive_uri", help="Archive URI (e.g. S3 path)."),
    ] = None,
    publication_cycle: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--publication_cycle", help="Publication cycle (e.g. 2025-04-C1)."
        ),
    ] = None,
    target_availability_date: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--target_availability_date",
            help="ISO date for target availability.",
        ),
    ] = None,
    api_url: Annotated[
        str,
        cyclopts.Parameter(name="--api-url", help="Override the API base URL."),
    ] = _DEFAULT_API_URL,
) -> None:
    """Update a submission's status, timestamps, PR link, or publication details."""
    update_submission(
        token=token,
        submission_id=submission_id,
        status=status,
        availability_qualified_at=availability_qualified_at,
        compliance_passed_at=compliance_passed_at,
        first_published_at=first_published_at,
        peer_review_started_at=peer_review_started_at,
        objection_resolution_started_at=objection_resolution_started_at,
        finalized_at=finalized_at,
        pr_url=pr_url,
        pr_number=pr_number,
        archive_uri=archive_uri,
        publication_cycle=publication_cycle,
        target_availability_date=target_availability_date,
        api_url=api_url,
    )


# ---------------------------------------------------------------------------
# withdraw submission
# ---------------------------------------------------------------------------


def withdraw_submission(
    token: str | None = None,
    submission_id: str = "",
    api_url: str = _DEFAULT_API_URL,
) -> None:
    """Withdraw (delete) a submission."""
    resolved_token = _resolve_token(token)
    resp = _delete(
        f"{api_url.rstrip('/')}/submissions/{submission_id}",
        {"token": resolved_token},
    )
    if resp.status_code == 200:
        print(f"Withdrawn submission: {submission_id}")
        return
    _handle_error(resp)


@withdraw_app.command(name="submission")
def _withdraw_submission_cmd(
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
        cyclopts.Parameter(name="--submission_id", help="Submission UUID to withdraw."),
    ],
    api_url: Annotated[
        str,
        cyclopts.Parameter(name="--api-url", help="Override the API base URL."),
    ] = _DEFAULT_API_URL,
) -> None:
    """Withdraw a submission."""
    withdraw_submission(token=token, submission_id=submission_id, api_url=api_url)

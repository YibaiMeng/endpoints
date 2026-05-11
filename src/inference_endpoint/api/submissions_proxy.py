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

"""Proxy routes for the /submissions API — authenticate with PRISM then forward upstream."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict

from inference_endpoint.api.auth import PRISMIdentity, require_auth

logger = logging.getLogger(__name__)

_RUNS_API_BASE_URL: str = os.environ.get(
    "RUNS_API_BASE_URL", "http://localhost:8081"
).rstrip("/")

router = APIRouter(tags=["submissions"])

# ---------------------------------------------------------------------------
# Payload models
# ---------------------------------------------------------------------------


class _SubmissionCreate(BaseModel):
    model_config = ConfigDict(frozen=True)

    availability: str
    benchmark_version: str
    division: str
    early_publish: bool
    publication_cycle: str
    run_ids: list[str]


class _SubmissionUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str | None = None
    availability_qualified_at: str | None = None
    compliance_passed_at: str | None = None
    first_published_at: str | None = None
    peer_review_started_at: str | None = None
    objection_resolution_started_at: str | None = None
    finalized_at: str | None = None
    pr_url: str | None = None
    pr_number: int | None = None
    archive_uri: str | None = None
    publication_cycle: str | None = None
    target_availability_date: str | None = None


# ---------------------------------------------------------------------------
# Shared upstream helpers
# ---------------------------------------------------------------------------


def _upstream_error(exc: httpx.RequestError, route: str) -> HTTPException:
    logger.error("Submissions API unreachable (%s): %s", route, exc)
    return HTTPException(
        http_status.HTTP_502_BAD_GATEWAY,
        detail=f"Submissions API unavailable: {exc}",
    )


def _proxy_response(upstream: httpx.Response) -> Response:
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/submissions", response_model=None, status_code=http_status.HTTP_201_CREATED)
async def create_submission(
    body: _SubmissionCreate,
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """Create a new submission."""
    logger.info(
        "create_submission: user_id=%s benchmark_version=%s run_ids=%s",
        identity.user_id,
        body.benchmark_version,
        body.run_ids,
    )
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.post(
                f"{_RUNS_API_BASE_URL}/submissions",
                params={"user_id": identity.user_id},
                json=body.model_dump(),
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, "POST /submissions") from exc
    logger.info("create_submission: upstream responded %d", upstream.status_code)
    return _proxy_response(upstream)


@router.get("/submissions", response_model=None)
async def list_submissions(
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """List all submissions for the authenticated user."""
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.get(
                f"{_RUNS_API_BASE_URL}/submissions",
                params={"user_id": identity.user_id},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, "GET /submissions") from exc
    return _proxy_response(upstream)


@router.get("/submissions/{submission_id}", response_model=None)
async def get_submission(
    submission_id: str,
    include_runs: bool = True,
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """Get a single submission by ID."""
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.get(
                f"{_RUNS_API_BASE_URL}/submissions/{submission_id}",
                params={"user_id": identity.user_id, "include_runs": include_runs},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, f"GET /submissions/{submission_id}") from exc
    return _proxy_response(upstream)


@router.patch("/submissions/{submission_id}", response_model=None)
async def update_submission(
    submission_id: str,
    body: _SubmissionUpdate,
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """Update a submission's status, PR link, or compliance timestamp."""
    payload: dict[str, Any] = body.model_dump(exclude_none=True)
    logger.info(
        "update_submission: submission_id=%s user_id=%s fields=%s",
        submission_id,
        identity.user_id,
        list(payload.keys()),
    )
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.patch(
                f"{_RUNS_API_BASE_URL}/submissions/{submission_id}",
                params={"user_id": identity.user_id},
                json=payload,
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, f"PATCH /submissions/{submission_id}") from exc
    return _proxy_response(upstream)


@router.delete("/submissions/{submission_id}", response_model=None)
async def withdraw_submission(
    submission_id: str,
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """Withdraw (delete) a submission."""
    logger.info(
        "withdraw_submission: submission_id=%s user_id=%s",
        submission_id,
        identity.user_id,
    )
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.delete(
                f"{_RUNS_API_BASE_URL}/submissions/{submission_id}",
                params={"user_id": identity.user_id},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, f"DELETE /submissions/{submission_id}") from exc
    return _proxy_response(upstream)

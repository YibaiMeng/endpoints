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

"""Proxy routes for /runs — authenticate with PRISM then forward to the MLCommons Endpoints Backend."""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi import status as http_status

from inference_endpoint.api.auth import PRISMIdentity, require_auth

logger = logging.getLogger(__name__)

_BACKEND_BASE_URL: str = os.environ.get(
    "RUNS_API_BASE_URL", "http://localhost:8080"
).rstrip("/")

router = APIRouter(tags=["runs"])

# ---------------------------------------------------------------------------
# Shared upstream helper
# ---------------------------------------------------------------------------


def _upstream_error(exc: httpx.RequestError, route: str) -> HTTPException:
    logger.error("MLCommons Endpoints Backend unreachable (%s): %s", route, exc)
    return HTTPException(
        http_status.HTTP_502_BAD_GATEWAY,
        detail=f"MLCommons Endpoints Backend unavailable: {exc}",
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


@router.get("/runs", response_model=None)
async def list_runs(
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """List all runs for the authenticated user."""
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.get(
                f"{_BACKEND_BASE_URL}/runs",
                params={"user_id": identity.user_id},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, "GET /runs") from exc
    return _proxy_response(upstream)


@router.get("/runs/{run_id}", response_model=None)
async def get_run(
    run_id: str,
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """Get a single run by ID (user validated via X-User-Id header)."""
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.get(
                f"{_BACKEND_BASE_URL}/runs/{run_id}",
                headers={"X-User-Id": identity.user_id},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, f"GET /runs/{run_id}") from exc
    return _proxy_response(upstream)


@router.delete("/runs/{run_id}", response_model=None)
async def delete_run(
    run_id: str,
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """Delete a run (user validated via X-User-Id header)."""
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.delete(
                f"{_BACKEND_BASE_URL}/runs/{run_id}",
                headers={"X-User-Id": identity.user_id},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, f"DELETE /runs/{run_id}") from exc
    return _proxy_response(upstream)


@router.patch("/runs/{run_id}/pin", response_model=None)
async def pin_run(
    run_id: str,
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """Pin a run (user validated via user_id query param)."""
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.patch(
                f"{_BACKEND_BASE_URL}/runs/{run_id}/pin",
                params={"user_id": identity.user_id},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, f"PATCH /runs/{run_id}/pin") from exc
    return _proxy_response(upstream)


@router.patch("/runs/{run_id}/unpin", response_model=None)
async def unpin_run(
    run_id: str,
    identity: PRISMIdentity = Depends(require_auth),
) -> Response:
    """Unpin a run (user validated via user_id query param)."""
    async with httpx.AsyncClient() as client:
        try:
            upstream = await client.patch(
                f"{_BACKEND_BASE_URL}/runs/{run_id}/unpin",
                params={"user_id": identity.user_id},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise _upstream_error(exc, f"PATCH /runs/{run_id}/unpin") from exc
    return _proxy_response(upstream)

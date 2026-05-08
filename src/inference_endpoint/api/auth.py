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

"""Shared PRISM authentication layer for all API routes."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import HTTPException, Query
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

_PRISM_BASE = "https://prism.mlcommons.org/functions/v1"
_PRISM_USER_TOKEN: str | None = os.environ.get("PRISM_USER_TOKEN")

_PRISM_ERROR_MAP: dict[str, tuple[int, str]] = {
    "INVALID_FORMAT": (400, "Malformed API token format"),
    "INVALID_KEY": (401, "Invalid API token"),
    "INACTIVE_KEY": (401, "API token is inactive"),
    "EXPIRED_KEY": (401, "API token has expired"),
    "SERVICE_MISMATCH": (403, "Token not authorized for the Endpoints service"),
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class PRISMAuthError(Exception):
    def __init__(self, status: int, detail: str, extra: dict | None = None) -> None:
        self.status = status
        self.detail = detail
        self.extra = extra or {}
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Identity model
# ---------------------------------------------------------------------------


class PRISMIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    email: str
    company_name: str
    company_external_id: str


# ---------------------------------------------------------------------------
# Core verification function (pure async, raises PRISMAuthError)
# ---------------------------------------------------------------------------


def _prism_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_PRISM_USER_TOKEN}",
        "Content-Type": "application/json",
    }


async def verify_token(token: str) -> PRISMIdentity:
    """Validate *token* against PRISM and return the resolved identity.

    All error paths raise :class:`PRISMAuthError` — callers decide how to surface
    those to HTTP clients.
    """
    async with httpx.AsyncClient() as client:
        # --- Step 1: validate the API key ---
        try:
            validate_resp = await client.post(
                f"{_PRISM_BASE}/validate-api-key",
                headers=_prism_headers(),
                json={"api_key": token, "service_id": "endpoints"},
                timeout=10.0,
            )
        except httpx.TimeoutException as exc:
            logger.error("PRISM timeout during token validation: %s", exc)
            raise PRISMAuthError(502, f"PRISM unavailable: timeout ({exc})") from exc
        except httpx.RequestError as exc:
            logger.error("PRISM network error during token validation: %s", exc)
            raise PRISMAuthError(502, f"PRISM unavailable: {exc}") from exc

        if validate_resp.status_code == 401:
            logger.error(
                "PRISM rejected our bearer token — PRISM_USER_TOKEN misconfigured"
            )
            raise PRISMAuthError(500, "Server auth misconfiguration — contact admin")

        if validate_resp.status_code == 429:
            body: dict[str, Any] = {}
            try:
                body = validate_resp.json()
            except Exception:  # noqa: BLE001
                pass
            retry_after = body.get("retry_after_seconds", 0)
            raise PRISMAuthError(
                429, "Rate limited", {"retry_after_seconds": retry_after}
            )

        if validate_resp.status_code == 400:
            logger.error(
                "PRISM returned 400 — bug in our request body: %s",
                validate_resp.text[:200],
            )
            raise PRISMAuthError(500, "Internal error constructing PRISM request")

        try:
            validate_body = validate_resp.json()
        except Exception as exc:
            logger.error("PRISM validate response is not valid JSON: %s", exc)
            raise PRISMAuthError(
                502, "PRISM unavailable: invalid JSON response"
            ) from exc

        if validate_body.get("valid") is not True:
            error_code = validate_body.get("error_code", "")
            status, detail = _PRISM_ERROR_MAP.get(
                error_code, (401, f"Token rejected: {error_code}")
            )
            raise PRISMAuthError(status, detail)

        user_id: str = validate_body.get("id", "")

        # --- Step 2: fetch user metadata ---
        try:
            meta_resp = await client.get(
                f"{_PRISM_BASE}/validate-api-key",
                headers=_prism_headers(),
                params={"user_id": user_id},
                timeout=10.0,
            )
        except httpx.RequestError as exc:
            logger.error("PRISM user metadata fetch failed: %s", exc)
            raise PRISMAuthError(
                502, f"Failed to fetch user metadata from PRISM: {exc}"
            ) from exc

        if not meta_resp.is_success:
            logger.error(
                "PRISM metadata endpoint returned %d: %s",
                meta_resp.status_code,
                meta_resp.text[:200],
            )
            raise PRISMAuthError(
                502, f"Failed to fetch user metadata: HTTP {meta_resp.status_code}"
            )

        try:
            meta = meta_resp.json()
        except Exception as exc:
            logger.error("PRISM metadata response is not valid JSON: %s", exc)
            raise PRISMAuthError(
                502, "Invalid JSON from PRISM user metadata endpoint"
            ) from exc

        return PRISMIdentity(
            user_id=user_id,
            email=meta.get("email", ""),
            company_name=meta.get("company_name", ""),
            company_external_id=meta.get("company_external_id", ""),
        )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def require_auth(
    token: str = Query(..., description="PRISM API key"),
) -> PRISMIdentity:
    """FastAPI dependency: resolve PRISM token → PRISMIdentity or raise HTTPException."""
    try:
        return await verify_token(token)
    except PRISMAuthError as exc:
        detail: str | dict = (
            exc.detail if not exc.extra else {"detail": exc.detail, **exc.extra}
        )
        raise HTTPException(status_code=exc.status, detail=detail) from exc

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

"""Unit tests for the shared PRISM auth module (auth.py)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

try:
    from fastapi import HTTPException
    from inference_endpoint.api.auth import (
        PRISMAuthError,
        PRISMIdentity,
        require_auth,
        verify_token,
    )

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

_api_only = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi extras not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Resp:
    """Minimal httpx.Response stand-in for mocking."""

    def __init__(self, status: int, body: Any = None) -> None:
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if body is not None else ""
        self.is_success = 200 <= status < 300

    def json(self) -> Any:
        return self._body


_VALID_PRISM_RESP = _Resp(200, {"valid": True, "id": "user-uuid-123"})
_USER_META_RESP = _Resp(
    200,
    {
        "id": "user-uuid-123",
        "email": "alice@example.com",
        "company_name": "Acme Corp",
        "company_external_id": "ext-456",
    },
)

_EXPECTED_IDENTITY = (
    PRISMIdentity(
        user_id="user-uuid-123",
        email="alice@example.com",
        company_name="Acme Corp",
        company_external_id="ext-456",
    )
    if HAS_FASTAPI
    else None
)  # type: ignore[assignment]


@contextmanager
def _mock_prism(post_resp: Any, get_resp: Any = _USER_META_RESP):
    """Patch httpx.AsyncClient inside auth.py with configured responses."""
    mock_client = AsyncMock()
    mock_client.post.return_value = post_resp
    mock_client.get.return_value = get_resp

    class _Ctx:
        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, *_):
            return False

    with patch("inference_endpoint.api.auth.httpx.AsyncClient", return_value=_Ctx()):
        yield mock_client


# ---------------------------------------------------------------------------
# verify_token — happy path
# ---------------------------------------------------------------------------


@_api_only
class TestVerifyTokenSuccess:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_prism_identity_on_success(self) -> None:
        with _mock_prism(_VALID_PRISM_RESP, _USER_META_RESP):
            identity = await verify_token("mlc_good_token_1234")

        assert identity == _EXPECTED_IDENTITY
        assert identity.user_id == "user-uuid-123"
        assert identity.email == "alice@example.com"
        assert identity.company_name == "Acme Corp"
        assert identity.company_external_id == "ext-456"


# ---------------------------------------------------------------------------
# verify_token — PRISM error_code variants
# ---------------------------------------------------------------------------


@_api_only
class TestVerifyTokenPrismErrorCodes:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_format_raises_400(self) -> None:
        with _mock_prism(_Resp(200, {"valid": False, "error_code": "INVALID_FORMAT"})):
            with pytest.raises(PRISMAuthError) as exc_info:
                await verify_token("bad")
        assert exc_info.value.status == 400
        assert "Malformed" in exc_info.value.detail

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_key_raises_401(self) -> None:
        with _mock_prism(_Resp(200, {"valid": False, "error_code": "INVALID_KEY"})):
            with pytest.raises(PRISMAuthError) as exc_info:
                await verify_token("mlc_bad_key")
        assert exc_info.value.status == 401
        assert "Invalid" in exc_info.value.detail

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_inactive_key_raises_401(self) -> None:
        with _mock_prism(_Resp(200, {"valid": False, "error_code": "INACTIVE_KEY"})):
            with pytest.raises(PRISMAuthError) as exc_info:
                await verify_token("mlc_inactive_key")
        assert exc_info.value.status == 401
        assert "inactive" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_expired_key_raises_401(self) -> None:
        with _mock_prism(_Resp(200, {"valid": False, "error_code": "EXPIRED_KEY"})):
            with pytest.raises(PRISMAuthError) as exc_info:
                await verify_token("mlc_expired_key")
        assert exc_info.value.status == 401
        assert "expired" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_service_mismatch_raises_403(self) -> None:
        with _mock_prism(
            _Resp(200, {"valid": False, "error_code": "SERVICE_MISMATCH"})
        ):
            with pytest.raises(PRISMAuthError) as exc_info:
                await verify_token("mlc_wrong_service")
        assert exc_info.value.status == 403
        assert "Endpoints" in exc_info.value.detail


# ---------------------------------------------------------------------------
# verify_token — HTTP-level PRISM errors
# ---------------------------------------------------------------------------


@_api_only
class TestVerifyTokenHttpErrors:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_prism_http_401_raises_500(self) -> None:
        """PRISM HTTP 401 = our bearer token is wrong → 500 server misconfiguration."""
        with _mock_prism(_Resp(401)):
            with pytest.raises(PRISMAuthError) as exc_info:
                await verify_token("mlc_any_token")
        assert exc_info.value.status == 500
        assert "misconfiguration" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_prism_http_429_raises_429_with_retry_after(self) -> None:
        with _mock_prism(
            _Resp(429, {"error": "Rate limit exceeded", "retry_after_seconds": 45})
        ):
            with pytest.raises(PRISMAuthError) as exc_info:
                await verify_token("mlc_any_token")
        assert exc_info.value.status == 429
        assert exc_info.value.extra.get("retry_after_seconds") == 45

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_network_timeout_raises_502(self) -> None:
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")

        class _Ctx:
            async def __aenter__(self):
                return mock_client

            async def __aexit__(self, *_):
                return False

        with patch(
            "inference_endpoint.api.auth.httpx.AsyncClient", return_value=_Ctx()
        ):
            with pytest.raises(PRISMAuthError) as exc_info:
                await verify_token("mlc_any_token")
        assert exc_info.value.status == 502
        assert "unavailable" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# require_auth — FastAPI dependency converts PRISMAuthError to HTTPException
# ---------------------------------------------------------------------------


@_api_only
class TestRequireAuthDependency:
    """require_auth must map PRISMAuthError → HTTPException with correct status."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_valid_token_returns_identity(self) -> None:
        with _mock_prism(_VALID_PRISM_RESP, _USER_META_RESP):
            identity = await require_auth("mlc_good_token")
        assert identity.user_id == "user-uuid-123"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_prism_error_raises_http_exception(self) -> None:
        with _mock_prism(_Resp(200, {"valid": False, "error_code": "INVALID_KEY"})):
            with pytest.raises(HTTPException) as exc_info:
                await require_auth("mlc_bad_key")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_rate_limit_raises_http_429_with_retry_after_in_detail(self) -> None:
        with _mock_prism(
            _Resp(429, {"error": "Rate limit exceeded", "retry_after_seconds": 20})
        ):
            with pytest.raises(HTTPException) as exc_info:
                await require_auth("mlc_any_token")
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["retry_after_seconds"] == 20

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_server_misconfiguration_raises_http_500(self) -> None:
        with _mock_prism(_Resp(401)):
            with pytest.raises(HTTPException) as exc_info:
                await require_auth("mlc_any_token")
        assert exc_info.value.status_code == 500

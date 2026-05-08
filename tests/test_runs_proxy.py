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

"""Integration tests for the /runs proxy routes."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

try:
    import httpx
    from fastapi import HTTPException
    from httpx import ASGITransport, AsyncClient
    from inference_endpoint.api.app import app
    from inference_endpoint.api.auth import PRISMIdentity, require_auth

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

_api_only = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi extras not installed")

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_TEST_IDENTITY = (
    PRISMIdentity(
        user_id="test-user-uuid",
        email="test@example.com",
        company_name="TestCo",
        company_external_id="ext-test",
    )
    if HAS_FASTAPI
    else None
)

_RUN_ID = "run-abc-123"


class _UpstreamResp:
    """Minimal httpx.Response stand-in for upstream mock."""

    def __init__(self, status: int, body: Any = None) -> None:
        self.status_code = status
        self._body = body
        self.content = json.dumps(body).encode() if body is not None else b""
        self.text = json.dumps(body) if body is not None else ""
        self.is_success = 200 <= status < 300
        self.headers = {"content-type": "application/json"}

    def json(self) -> Any:
        return self._body


@contextmanager
def _auth_ok():
    """Override require_auth to return the test identity."""
    app.dependency_overrides[require_auth] = lambda: _TEST_IDENTITY
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_auth, None)


@contextmanager
def _auth_fail(status: int = 401, detail: str = "Invalid API token"):
    """Override require_auth to raise HTTPException."""

    async def _failing():
        raise HTTPException(status_code=status, detail=detail)

    app.dependency_overrides[require_auth] = _failing
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_auth, None)


@contextmanager
def _mock_upstream(method: str, response: Any):
    """Patch httpx.AsyncClient inside runs_proxy for the given HTTP method."""
    mock_client = AsyncMock()
    getattr(mock_client, method).return_value = response

    class _Ctx:
        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, *_):
            return False

    with patch(
        "inference_endpoint.api.runs_proxy.httpx.AsyncClient", return_value=_Ctx()
    ):
        yield mock_client


def _asgi_client() -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )


TOKEN_PARAM = "?token=mlc_test_token_1234"

# ---------------------------------------------------------------------------
# GET /runs — list runs
# ---------------------------------------------------------------------------


@_api_only
class TestListRuns:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_auth_failure_returns_401(self) -> None:
        with _auth_fail(401, "Invalid API token"):
            async with _asgi_client() as client:
                resp = await client.get(f"/runs{TOKEN_PARAM}")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_happy_path_returns_upstream_body(self) -> None:
        runs = [{"id": _RUN_ID, "user_id": "test-user-uuid"}]
        with _auth_ok():
            with _mock_upstream("get", _UpstreamResp(200, runs)) as mock_client:
                async with _asgi_client() as client:
                    resp = await client.get(f"/runs{TOKEN_PARAM}")
                call_params = mock_client.get.call_args.kwargs.get("params", {})
        assert resp.status_code == 200
        assert resp.json() == runs
        assert call_params.get("user_id") == "test-user-uuid"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_error_proxied_as_is(self) -> None:
        with _auth_ok():
            with _mock_upstream("get", _UpstreamResp(503, {"detail": "DB down"})):
                async with _asgi_client() as client:
                    resp = await client.get(f"/runs{TOKEN_PARAM}")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_unavailable_returns_502(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("connection refused")

        class _Ctx:
            async def __aenter__(self):
                return mock_client

            async def __aexit__(self, *_):
                return False

        with _auth_ok():
            with patch(
                "inference_endpoint.api.runs_proxy.httpx.AsyncClient",
                return_value=_Ctx(),
            ):
                async with _asgi_client() as client:
                    resp = await client.get(f"/runs{TOKEN_PARAM}")
        assert resp.status_code == 502
        assert "unavailable" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_passes_user_id_as_query_param(self) -> None:
        """list_runs must use user_id query param, not a header."""
        with _auth_ok():
            with _mock_upstream("get", _UpstreamResp(200, [])) as mock_client:
                async with _asgi_client() as client:
                    await client.get(f"/runs{TOKEN_PARAM}")
                call = mock_client.get.call_args
        params = call.kwargs.get("params", {})
        headers = call.kwargs.get("headers", {})
        assert params.get("user_id") == "test-user-uuid"
        assert "X-User-Id" not in headers


# ---------------------------------------------------------------------------
# GET /runs/{run_id} — get single run
# ---------------------------------------------------------------------------


@_api_only
class TestGetRun:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_auth_failure_returns_401(self) -> None:
        with _auth_fail(401):
            async with _asgi_client() as client:
                resp = await client.get(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_happy_path_returns_run(self) -> None:
        run = {"id": _RUN_ID, "user_id": "test-user-uuid"}
        with _auth_ok():
            with _mock_upstream("get", _UpstreamResp(200, run)) as mock_client:
                async with _asgi_client() as client:
                    resp = await client.get(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
                call_headers = mock_client.get.call_args.kwargs.get("headers", {})
        assert resp.status_code == 200
        assert resp.json() == run
        assert call_headers.get("X-User-Id") == "test-user-uuid"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_404_proxied(self) -> None:
        with _auth_ok():
            with _mock_upstream("get", _UpstreamResp(404, {"detail": "Run not found"})):
                async with _asgi_client() as client:
                    resp = await client.get(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_unavailable_returns_502(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")

        class _Ctx:
            async def __aenter__(self):
                return mock_client

            async def __aexit__(self, *_):
                return False

        with _auth_ok():
            with patch(
                "inference_endpoint.api.runs_proxy.httpx.AsyncClient",
                return_value=_Ctx(),
            ):
                async with _asgi_client() as client:
                    resp = await client.get(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_passes_x_user_id_header_not_query_param(self) -> None:
        """get_run must use X-User-Id header, not user_id query param."""
        with _auth_ok():
            with _mock_upstream("get", _UpstreamResp(200, {})) as mock_client:
                async with _asgi_client() as client:
                    await client.get(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
                call = mock_client.get.call_args
        headers = call.kwargs.get("headers", {})
        params = call.kwargs.get("params", {})
        assert headers.get("X-User-Id") == "test-user-uuid"
        assert "user_id" not in params


# ---------------------------------------------------------------------------
# DELETE /runs/{run_id} — delete run
# ---------------------------------------------------------------------------


@_api_only
class TestDeleteRun:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_auth_failure_returns_401(self) -> None:
        with _auth_fail(401):
            async with _asgi_client() as client:
                resp = await client.delete(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_happy_path_returns_run_id(self) -> None:
        with _auth_ok():
            with _mock_upstream("delete", _UpstreamResp(200, _RUN_ID)) as mock_client:
                async with _asgi_client() as client:
                    resp = await client.delete(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
                call_headers = mock_client.delete.call_args.kwargs.get("headers", {})
        assert resp.status_code == 200
        assert call_headers.get("X-User-Id") == "test-user-uuid"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_404_proxied(self) -> None:
        with _auth_ok():
            with _mock_upstream(
                "delete", _UpstreamResp(404, {"detail": "Run not found"})
            ):
                async with _asgi_client() as client:
                    resp = await client.delete(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_unavailable_returns_502(self) -> None:
        mock_client = AsyncMock()
        mock_client.delete.side_effect = httpx.ConnectError("refused")

        class _Ctx:
            async def __aenter__(self):
                return mock_client

            async def __aexit__(self, *_):
                return False

        with _auth_ok():
            with patch(
                "inference_endpoint.api.runs_proxy.httpx.AsyncClient",
                return_value=_Ctx(),
            ):
                async with _asgi_client() as client:
                    resp = await client.delete(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_passes_x_user_id_header_not_query_param(self) -> None:
        with _auth_ok():
            with _mock_upstream("delete", _UpstreamResp(200, _RUN_ID)) as mock_client:
                async with _asgi_client() as client:
                    await client.delete(f"/runs/{_RUN_ID}{TOKEN_PARAM}")
                call = mock_client.delete.call_args
        assert call.kwargs.get("headers", {}).get("X-User-Id") == "test-user-uuid"
        assert "user_id" not in call.kwargs.get("params", {})


# ---------------------------------------------------------------------------
# PATCH /runs/{run_id}/pin
# ---------------------------------------------------------------------------


@_api_only
class TestPinRun:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_auth_failure_returns_401(self) -> None:
        with _auth_fail(401):
            async with _asgi_client() as client:
                resp = await client.patch(f"/runs/{_RUN_ID}/pin{TOKEN_PARAM}")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_happy_path_passes_user_id_query_param(self) -> None:
        with _auth_ok():
            with _mock_upstream("patch", _UpstreamResp(200, _RUN_ID)) as mock_client:
                async with _asgi_client() as client:
                    resp = await client.patch(f"/runs/{_RUN_ID}/pin{TOKEN_PARAM}")
                call = mock_client.patch.call_args
        assert resp.status_code == 200
        assert call.kwargs.get("params", {}).get("user_id") == "test-user-uuid"
        assert "X-User-Id" not in call.kwargs.get("headers", {})

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_404_proxied(self) -> None:
        with _auth_ok():
            with _mock_upstream(
                "patch", _UpstreamResp(404, {"detail": "Run not found"})
            ):
                async with _asgi_client() as client:
                    resp = await client.patch(f"/runs/{_RUN_ID}/pin{TOKEN_PARAM}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_unavailable_returns_502(self) -> None:
        mock_client = AsyncMock()
        mock_client.patch.side_effect = httpx.ConnectError("refused")

        class _Ctx:
            async def __aenter__(self):
                return mock_client

            async def __aexit__(self, *_):
                return False

        with _auth_ok():
            with patch(
                "inference_endpoint.api.runs_proxy.httpx.AsyncClient",
                return_value=_Ctx(),
            ):
                async with _asgi_client() as client:
                    resp = await client.patch(f"/runs/{_RUN_ID}/pin{TOKEN_PARAM}")
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# PATCH /runs/{run_id}/unpin
# ---------------------------------------------------------------------------


@_api_only
class TestUnpinRun:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_auth_failure_returns_401(self) -> None:
        with _auth_fail(401):
            async with _asgi_client() as client:
                resp = await client.patch(f"/runs/{_RUN_ID}/unpin{TOKEN_PARAM}")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_happy_path_passes_user_id_query_param(self) -> None:
        with _auth_ok():
            with _mock_upstream("patch", _UpstreamResp(200, _RUN_ID)) as mock_client:
                async with _asgi_client() as client:
                    resp = await client.patch(f"/runs/{_RUN_ID}/unpin{TOKEN_PARAM}")
                call = mock_client.patch.call_args
        assert resp.status_code == 200
        assert call.kwargs.get("params", {}).get("user_id") == "test-user-uuid"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_404_proxied(self) -> None:
        with _auth_ok():
            with _mock_upstream(
                "patch", _UpstreamResp(404, {"detail": "Run not found"})
            ):
                async with _asgi_client() as client:
                    resp = await client.patch(f"/runs/{_RUN_ID}/unpin{TOKEN_PARAM}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_upstream_unavailable_returns_502(self) -> None:
        mock_client = AsyncMock()
        mock_client.patch.side_effect = httpx.ConnectError("refused")

        class _Ctx:
            async def __aenter__(self):
                return mock_client

            async def __aexit__(self, *_):
                return False

        with _auth_ok():
            with patch(
                "inference_endpoint.api.runs_proxy.httpx.AsyncClient",
                return_value=_Ctx(),
            ):
                async with _asgi_client() as client:
                    resp = await client.patch(f"/runs/{_RUN_ID}/unpin{TOKEN_PARAM}")
        assert resp.status_code == 502

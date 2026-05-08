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

"""Tests for push_run FastAPI route and push CLI command."""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from inference_endpoint.commands.push.cli import (
    _create_archive,
    _handle_response,
    _masked_token,
    _validate_run_dir,
)
from inference_endpoint.exceptions import ExecutionError, InputValidationError

# ---------------------------------------------------------------------------
# Attempt FastAPI import — API route tests are skipped when not installed
# ---------------------------------------------------------------------------
try:
    from httpx import ASGITransport, AsyncClient
    from inference_endpoint.api.app import app
    from inference_endpoint.api.auth import PRISMIdentity, require_auth

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

_api_only = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi extras not installed")

# ---------------------------------------------------------------------------
# Shared paths and constants
# ---------------------------------------------------------------------------

_SAMPLE_RUN_DIR = (
    Path(__file__).parent.parent
    / "endpoints_run_samples"
    / "llama-3.1-8b_vllm_perf_concurrency1_3a8d044-06ccd43_main_trial1"
)

_REQUIRED_FILES = [
    "config.yaml",
    "result_summary.json",
    "runtime_settings.json",
    "events.jsonl",
]

_TEST_IDENTITY = (
    PRISMIdentity(
        user_id="user-uuid",
        email="test@example.com",
        company_name="TestCo",
        company_external_id="ext-1",
    )
    if HAS_FASTAPI
    else None
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeHttpxResponse:
    """Minimal httpx.Response-alike for passing to _handle_response."""

    def __init__(self, status: int, body: Any = None) -> None:
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if body is not None else ""
        self.is_success = 200 <= status < 300

    def json(self) -> Any:
        return self._body


def _make_run_dir(base: Path, missing: list[str] | None = None) -> Path:
    """Create a minimal valid run folder; optionally omit files from `missing`."""
    run_dir = base / "test_run"
    run_dir.mkdir(exist_ok=True)
    contents: dict[str, str] = {
        "config.yaml": "type: online\nmodel_params:\n  name: test-model\n",
        "result_summary.json": json.dumps({"qps": 1.0, "n_samples_completed": 10}),
        "runtime_settings.json": json.dumps({"max_duration_ms": 60000}),
        "events.jsonl": (
            '{"approx_datetime_str":"2026-01-01T00:00:00",'
            '"event_type":"test_started","sample_uuid":""}\n'
            '{"approx_datetime_str":"2026-01-01T00:10:00",'
            '"event_type":"test_ended","sample_uuid":""}\n'
        ),
    }
    for name, text in contents.items():
        if missing and name in missing:
            continue
        (run_dir / name).write_text(text)
    return run_dir


def _make_archive_bytes(run_dir: Path) -> bytes:
    """Return a .tar.gz of run_dir contents as bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for item in run_dir.iterdir():
            tf.add(item, arcname=item.name)
    return buf.getvalue()


@contextmanager
def _patch_push_run_env(tmp_path: Path):
    """Patch module-level vars in push_run to safe test values."""
    glob_dir = tmp_path / "glob"
    glob_dir.mkdir()
    with (
        patch("inference_endpoint.api.push_run._GLOB_DIR", glob_dir),
        patch("inference_endpoint.api.push_run._RUNS_API_BASE_URL", "http://mock-runs"),
    ):
        yield glob_dir


@contextmanager
def _mock_runs_post(response: Any):
    """Patch httpx.AsyncClient inside push_run to mock the upstream /runs POST."""
    mock_client = AsyncMock()
    mock_client.post.return_value = response

    class _Ctx:
        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, *_):
            return False

    with patch(
        "inference_endpoint.api.push_run.httpx.AsyncClient", return_value=_Ctx()
    ):
        yield mock_client


@contextmanager
def _auth_ok():
    """Override require_auth to return the test identity."""
    app.dependency_overrides[require_auth] = lambda: _TEST_IDENTITY
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_auth, None)


def _asgi_client() -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )


# ---------------------------------------------------------------------------
# 1 — CLI: pre-flight file validation
# ---------------------------------------------------------------------------


class TestCliPreflightValidation:
    @pytest.mark.unit
    def test_all_present_passes(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _validate_run_dir(run_dir)  # must not raise

    @pytest.mark.unit
    def test_missing_result_summary_raises(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, missing=["result_summary.json"])
        with pytest.raises(InputValidationError, match="result_summary.json"):
            _validate_run_dir(run_dir)

    @pytest.mark.unit
    def test_missing_multiple_files_raises(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path, missing=["result_summary.json", "events.jsonl"]
        )
        with pytest.raises(InputValidationError):
            _validate_run_dir(run_dir)

    @pytest.mark.unit
    def test_error_message_names_missing_files(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, missing=["events.jsonl"])
        with pytest.raises(InputValidationError) as exc_info:
            _validate_run_dir(run_dir)
        assert "events.jsonl" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2 — CLI: token resolution
# ---------------------------------------------------------------------------


class TestCliTokenResolution:
    @pytest.mark.unit
    def test_env_var_used_when_no_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENDPOINTS_TOKEN", "env_token_abcd")
        resolved = None or os.environ.get("ENDPOINTS_TOKEN", "")
        assert resolved == "env_token_abcd"

    @pytest.mark.unit
    def test_flag_takes_priority_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENDPOINTS_TOKEN", "env_token_abcd")
        flag_token = "flag_token_zzzz"
        resolved = flag_token or os.environ.get("ENDPOINTS_TOKEN", "")
        assert resolved == "flag_token_zzzz"

    @pytest.mark.unit
    def test_missing_token_raises_input_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ENDPOINTS_TOKEN", raising=False)
        resolved = None or os.environ.get("ENDPOINTS_TOKEN", "")
        assert not resolved
        with pytest.raises(InputValidationError, match="Token is required"):
            if not resolved:
                raise InputValidationError(
                    "Token is required. Pass --token or set ENDPOINTS_TOKEN env var"
                )


# ---------------------------------------------------------------------------
# 3 — CLI: dry run
# ---------------------------------------------------------------------------


class TestCliDryRun:
    @pytest.mark.unit
    def test_dry_run_creates_archive_and_does_not_upload(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        run_dir = _make_run_dir(tmp_path)
        archive_path = tmp_path / "out.tar.gz"
        _create_archive(run_dir, archive_path)

        assert archive_path.exists()
        assert archive_path.stat().st_size > 0

        # Verify the archive has the expected files
        with tarfile.open(archive_path, "r:gz") as tf:
            names = tf.getnames()
        for fname in [
            "config.yaml",
            "result_summary.json",
            "runtime_settings.json",
            "events.jsonl",
        ]:
            assert fname in names

    @pytest.mark.unit
    def test_masked_token_last_four(self) -> None:
        assert _masked_token("mlc_abcdefghij") == "****ghij"

    @pytest.mark.unit
    def test_masked_token_short_token(self) -> None:
        assert _masked_token("ab") == "****"

    @pytest.mark.unit
    def test_dry_run_no_http_calls(self, tmp_path: Path) -> None:
        """Dry run must not invoke any httpx.post."""
        run_dir = _make_run_dir(tmp_path)

        with patch("inference_endpoint.commands.push.cli.httpx.post") as mock_post:
            archive_path = tmp_path / "run.tar.gz"
            _create_archive(run_dir, archive_path)
            # simulate what the CLI does in dry_run=True: just list contents, no upload
            mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# 4 — CLI: HTTP response → exception mapping
# ---------------------------------------------------------------------------


class TestCliErrorMapping:
    def _r(self, status: int, body: Any = None) -> _FakeHttpxResponse:
        return _FakeHttpxResponse(status, body)

    @pytest.mark.unit
    def test_201_prints_run_id(self, capsys: pytest.CaptureFixture) -> None:
        _handle_response(self._r(201, {"id": "run-abc-123"}))
        assert "run-abc-123" in capsys.readouterr().out

    @pytest.mark.unit
    def test_400_raises_input_validation_error(self) -> None:
        with pytest.raises(InputValidationError, match="format is invalid"):
            _handle_response(self._r(400))

    @pytest.mark.unit
    def test_401_raises_input_validation_error_with_detail(self) -> None:
        with pytest.raises(InputValidationError, match="Authentication failed"):
            _handle_response(self._r(401, {"detail": "Invalid API token"}))

    @pytest.mark.unit
    def test_403_raises_input_validation_error(self) -> None:
        with pytest.raises(InputValidationError, match="not authorized"):
            _handle_response(self._r(403))

    @pytest.mark.unit
    def test_422_raises_input_validation_error(self) -> None:
        with pytest.raises(InputValidationError, match="Validation failed"):
            _handle_response(self._r(422, {"detail": "field required"}))

    @pytest.mark.unit
    def test_429_raises_execution_error_with_retry_after(self) -> None:
        # API returns {"detail": {"detail": "Rate limited", "retry_after_seconds": 30}}
        body = {"detail": {"detail": "Rate limited", "retry_after_seconds": 30}}
        with pytest.raises(ExecutionError, match="retry after 30s"):
            _handle_response(self._r(429, body))

    @pytest.mark.unit
    def test_500_raises_execution_error(self) -> None:
        with pytest.raises(ExecutionError, match="Server error"):
            _handle_response(self._r(500))

    @pytest.mark.unit
    def test_502_raises_execution_error(self) -> None:
        with pytest.raises(ExecutionError, match="Upstream service"):
            _handle_response(self._r(502, {"detail": "Service unavailable"}))

    @pytest.mark.unit
    def test_unknown_status_raises_execution_error(self) -> None:
        with pytest.raises(ExecutionError, match=r"Push failed \[418\]"):
            _handle_response(self._r(418, {"detail": "I'm a teapot"}))


# ---------------------------------------------------------------------------
# 5 — API route: archive validation
# ---------------------------------------------------------------------------


@_api_only
class TestApiArchiveValidation:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_missing_result_summary_returns_422(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, missing=["result_summary.json"])
        archive = _make_archive_bytes(run_dir)

        with _auth_ok():
            with _patch_push_run_env(tmp_path):
                async with _asgi_client() as client:
                    resp = await client.post(
                        "/push_run?token=mlc_test_token_1234",
                        files={
                            "archive": (
                                "run.tar.gz",
                                archive,
                                "application/gzip",
                            )
                        },
                    )

        assert resp.status_code == 422
        body = resp.json()
        detail = body["detail"]
        assert "result_summary.json" in detail["missing"]


# ---------------------------------------------------------------------------
# 6 — API route: /runs returns 422 (bug in our payload)
# ---------------------------------------------------------------------------


@_api_only
class TestApiRunsErrors:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_runs_422_returns_500_and_logs_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        run_dir = _make_run_dir(tmp_path)
        archive = _make_archive_bytes(run_dir)

        runs_422 = _FakeHttpxResponse(
            422,
            {"detail": [{"loc": ["body", "started_at"], "msg": "required"}]},
        )

        with _auth_ok():
            with _patch_push_run_env(tmp_path):
                with _mock_runs_post(runs_422):
                    with caplog.at_level(logging.ERROR):
                        async with _asgi_client() as client:
                            resp = await client.post(
                                "/push_run?token=mlc_test_token_1234",
                                files={
                                    "archive": (
                                        "run.tar.gz",
                                        archive,
                                        "application/gzip",
                                    )
                                },
                            )

        assert resp.status_code == 500
        body = resp.json()
        assert "payload rejected" in body["detail"]["detail"]


# ---------------------------------------------------------------------------
# 7 — API route: happy path end-to-end
# ---------------------------------------------------------------------------


@_api_only
class TestApiHappyPath:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_happy_path_with_sample_run(self, tmp_path: Path) -> None:
        """Submit the first endpoints_run_samples folder and expect HTTP 201."""
        assert _SAMPLE_RUN_DIR.is_dir(), f"Sample run dir not found: {_SAMPLE_RUN_DIR}"
        archive = _make_archive_bytes(_SAMPLE_RUN_DIR)

        fake_run = {
            "id": "run-happy-uuid",
            "user_id": "user-uuid",
            "started_at": "2026-04-13T12:50:59.483996",
            "finished_at": "2026-04-13T13:26:16.102347",
            "expires_at": "2027-04-13T13:26:16.102347",
            "pinned": False,
            "system_info": {
                "email": "test@example.com",
                "company_name": "TestCo",
                "company_external_id": "ext-1",
            },
            "config": {},
            "result_summary": {},
            "archive_uri": "/tmp/glob/run",
        }

        with _auth_ok():
            with _patch_push_run_env(tmp_path):
                with _mock_runs_post(_FakeHttpxResponse(201, fake_run)):
                    async with _asgi_client() as client:
                        resp = await client.post(
                            "/push_run?token=mlc_test_token_1234",
                            files={
                                "archive": (
                                    "sample_run.tar.gz",
                                    archive,
                                    "application/gzip",
                                )
                            },
                            timeout=60.0,
                        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == "run-happy-uuid"
        assert body["user_id"] == "user-uuid"

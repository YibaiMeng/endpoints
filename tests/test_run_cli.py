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

"""Unit tests for the run management CLI commands (list/get/delete/pin/unpin)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

from inference_endpoint.commands.run.cli import (
    _handle_error,
    _resolve_token,
    delete_run,
    get_run,
    list_run,
    pin_run,
    unpin_run,
)
from inference_endpoint.exceptions import ExecutionError, InputValidationError

_RUN_ID = "run-abc-123"
_TOKEN = "mlc_" + "x" * 64
_API_URL = "http://localhost:8082"

# ---------------------------------------------------------------------------
# Minimal httpx.Response stand-in
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status: int, body: Any = None) -> None:
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if body is not None else ""
        self.is_success = 200 <= status < 300

    def json(self) -> Any:
        return self._body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _mock_http(method: str, response: _Resp):
    """Patch httpx.<method> inside run/cli.py."""
    with patch(f"inference_endpoint.commands.run.cli.httpx.{method}") as mock:
        mock.return_value = response
        yield mock


# ---------------------------------------------------------------------------
# _resolve_token
# ---------------------------------------------------------------------------


class TestResolveToken:
    @pytest.mark.unit
    def test_returns_explicit_token(self) -> None:
        assert _resolve_token("mytoken") == "mytoken"

    @pytest.mark.unit
    def test_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENDPOINTS_TOKEN", "env_token")
        assert _resolve_token(None) == "env_token"

    @pytest.mark.unit
    def test_raises_when_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ENDPOINTS_TOKEN", raising=False)
        with pytest.raises(InputValidationError, match="Token is required"):
            _resolve_token(None)


# ---------------------------------------------------------------------------
# _handle_error — status code → exception mapping
# ---------------------------------------------------------------------------


class TestHandleError:
    @pytest.mark.unit
    def test_400_raises_input_validation(self) -> None:
        with pytest.raises(InputValidationError, match="Bad request"):
            _handle_error(_Resp(400, {"detail": "bad"}))

    @pytest.mark.unit
    def test_401_raises_input_validation(self) -> None:
        with pytest.raises(InputValidationError, match="Authentication failed"):
            _handle_error(_Resp(401, {"detail": "Invalid token"}))

    @pytest.mark.unit
    def test_403_raises_input_validation(self) -> None:
        with pytest.raises(InputValidationError, match="not authorized"):
            _handle_error(_Resp(403))

    @pytest.mark.unit
    def test_404_raises_input_validation(self) -> None:
        with pytest.raises(InputValidationError, match="Run not found"):
            _handle_error(_Resp(404, {"detail": "no such run"}))

    @pytest.mark.unit
    def test_429_raises_execution_error_with_retry_after(self) -> None:
        with pytest.raises(ExecutionError, match="retry after 30s"):
            _handle_error(_Resp(429, {"detail": {"retry_after_seconds": 30}}))

    @pytest.mark.unit
    def test_500_raises_execution_error(self) -> None:
        with pytest.raises(ExecutionError, match="Server error"):
            _handle_error(_Resp(500, {"detail": "oops"}))

    @pytest.mark.unit
    def test_502_raises_execution_error(self) -> None:
        with pytest.raises(ExecutionError, match="Upstream service unavailable"):
            _handle_error(_Resp(502, {"detail": "upstream down"}))

    @pytest.mark.unit
    def test_unknown_raises_execution_error(self) -> None:
        with pytest.raises(ExecutionError, match=r"Request failed \[418\]"):
            _handle_error(_Resp(418))


# ---------------------------------------------------------------------------
# list_run
# ---------------------------------------------------------------------------


class TestListRun:
    @pytest.mark.unit
    def test_happy_path_prints_runs(self, capsys: pytest.CaptureFixture) -> None:
        runs = [{"id": _RUN_ID, "user_id": "u1"}]
        with _mock_http("get", _Resp(200, runs)):
            list_run(token=_TOKEN, api_url=_API_URL)
        assert _RUN_ID in capsys.readouterr().out

    @pytest.mark.unit
    def test_empty_list_prints_message(self, capsys: pytest.CaptureFixture) -> None:
        with _mock_http("get", _Resp(200, [])):
            list_run(token=_TOKEN, api_url=_API_URL)
        assert "No runs found" in capsys.readouterr().out

    @pytest.mark.unit
    def test_passes_token_as_query_param(self) -> None:
        with _mock_http("get", _Resp(200, [])) as mock:
            list_run(token=_TOKEN, api_url=_API_URL)
        call_params = mock.call_args.kwargs.get("params", {})
        assert call_params.get("token") == _TOKEN

    @pytest.mark.unit
    def test_401_raises_input_validation(self) -> None:
        with _mock_http("get", _Resp(401, {"detail": "bad token"})):
            with pytest.raises(InputValidationError):
                list_run(token=_TOKEN, api_url=_API_URL)

    @pytest.mark.unit
    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ENDPOINTS_TOKEN", raising=False)
        with pytest.raises(InputValidationError, match="Token is required"):
            list_run(token=None, api_url=_API_URL)


# ---------------------------------------------------------------------------
# get_run
# ---------------------------------------------------------------------------


class TestGetRun:
    @pytest.mark.unit
    def test_happy_path_prints_run(self, capsys: pytest.CaptureFixture) -> None:
        run = {"id": _RUN_ID, "user_id": "u1"}
        with _mock_http("get", _Resp(200, run)):
            get_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)
        assert _RUN_ID in capsys.readouterr().out

    @pytest.mark.unit
    def test_url_contains_run_id(self) -> None:
        with _mock_http("get", _Resp(200, {})) as mock:
            get_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)
        url = mock.call_args.args[0]
        assert _RUN_ID in url

    @pytest.mark.unit
    def test_404_raises_input_validation(self) -> None:
        with _mock_http("get", _Resp(404, {"detail": "not found"})):
            with pytest.raises(InputValidationError, match="Run not found"):
                get_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)

    @pytest.mark.unit
    def test_502_raises_execution_error(self) -> None:
        with _mock_http("get", _Resp(502, {"detail": "upstream down"})):
            with pytest.raises(ExecutionError, match="Upstream service unavailable"):
                get_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)


# ---------------------------------------------------------------------------
# delete_run
# ---------------------------------------------------------------------------


class TestDeleteRun:
    @pytest.mark.unit
    def test_happy_path_prints_confirmation(self, capsys: pytest.CaptureFixture) -> None:
        with _mock_http("delete", _Resp(200, _RUN_ID)):
            delete_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)
        assert _RUN_ID in capsys.readouterr().out

    @pytest.mark.unit
    def test_url_contains_run_id(self) -> None:
        with _mock_http("delete", _Resp(200, _RUN_ID)) as mock:
            delete_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)
        url = mock.call_args.args[0]
        assert _RUN_ID in url

    @pytest.mark.unit
    def test_404_raises_input_validation(self) -> None:
        with _mock_http("delete", _Resp(404, {"detail": "not found"})):
            with pytest.raises(InputValidationError, match="Run not found"):
                delete_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)


# ---------------------------------------------------------------------------
# pin_run
# ---------------------------------------------------------------------------


class TestPinRun:
    @pytest.mark.unit
    def test_happy_path_prints_confirmation(self, capsys: pytest.CaptureFixture) -> None:
        with _mock_http("patch", _Resp(200, _RUN_ID)):
            pin_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)
        assert _RUN_ID in capsys.readouterr().out

    @pytest.mark.unit
    def test_url_contains_pin(self) -> None:
        with _mock_http("patch", _Resp(200, _RUN_ID)) as mock:
            pin_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)
        url = mock.call_args.args[0]
        assert "/pin" in url

    @pytest.mark.unit
    def test_404_raises_input_validation(self) -> None:
        with _mock_http("patch", _Resp(404, {"detail": "not found"})):
            with pytest.raises(InputValidationError, match="Run not found"):
                pin_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)


# ---------------------------------------------------------------------------
# unpin_run
# ---------------------------------------------------------------------------


class TestUnpinRun:
    @pytest.mark.unit
    def test_happy_path_prints_confirmation(self, capsys: pytest.CaptureFixture) -> None:
        with _mock_http("patch", _Resp(200, _RUN_ID)):
            unpin_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)
        assert _RUN_ID in capsys.readouterr().out

    @pytest.mark.unit
    def test_url_contains_unpin(self) -> None:
        with _mock_http("patch", _Resp(200, _RUN_ID)) as mock:
            unpin_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)
        url = mock.call_args.args[0]
        assert "/unpin" in url

    @pytest.mark.unit
    def test_404_raises_input_validation(self) -> None:
        with _mock_http("patch", _Resp(404, {"detail": "not found"})):
            with pytest.raises(InputValidationError, match="Run not found"):
                unpin_run(token=_TOKEN, run_id=_RUN_ID, api_url=_API_URL)

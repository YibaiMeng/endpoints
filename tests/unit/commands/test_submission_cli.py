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

"""Unit tests for submission CLI commands (create / update / withdraw / list / get)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from inference_endpoint.commands.run.cli import (
    get_submission,
    list_submissions,
)
from inference_endpoint.commands.submission.cli import (
    _handle_error,
    _resolve_token,
    create_submission,
    update_submission,
    withdraw_submission,
)
from inference_endpoint.exceptions import ExecutionError, InputValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_SUB = {
    "id": "sub-123",
    "availability": "available",
    "benchmark_version": "a1b2c3d",
    "division": "standardized",
    "early_publish": False,
    "publication_cycle": "2025-04-C1",
    "run_ids": ["run-abc"],
    "status": "COMPLIANCE_CHECKING",
    "user_id": "u_prism_7f3a9b",
    "created_at": "2025-04-28T12:00:00",
}


def _mock_resp(status: int, body) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.text = json.dumps(body) if isinstance(body, dict) else str(body)
    return resp


# ---------------------------------------------------------------------------
# _resolve_token
# ---------------------------------------------------------------------------


class TestResolveToken:
    @pytest.mark.unit
    def test_uses_explicit_token(self):
        assert _resolve_token("my-token") == "my-token"

    @pytest.mark.unit
    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("ENDPOINTS_TOKEN", "env-token")
        assert _resolve_token(None) == "env-token"

    @pytest.mark.unit
    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("ENDPOINTS_TOKEN", raising=False)
        with pytest.raises(InputValidationError, match="Token is required"):
            _resolve_token(None)


# ---------------------------------------------------------------------------
# _handle_error
# ---------------------------------------------------------------------------


class TestHandleError:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "status, exc_type, match",
        [
            (400, InputValidationError, "Bad request"),
            (401, InputValidationError, "Authentication failed"),
            (403, InputValidationError, "not authorized"),
            (404, InputValidationError, "not found"),
            (422, InputValidationError, "Validation error"),
            (500, ExecutionError, "Server error"),
            (502, ExecutionError, "Upstream service"),
            (503, ExecutionError, r"\[503\]"),
        ],
    )
    def test_status_code_mapping(self, status, exc_type, match):
        resp = _mock_resp(status, {"detail": "oops"})
        with pytest.raises(exc_type, match=match):
            _handle_error(resp)

    @pytest.mark.unit
    def test_rate_limit_includes_retry_after(self):
        resp = _mock_resp(429, {"detail": {"retry_after_seconds": 42}})
        with pytest.raises(ExecutionError, match="42s"):
            _handle_error(resp)


# ---------------------------------------------------------------------------
# list_submissions (in commands/run/cli.py)
# ---------------------------------------------------------------------------


class TestListSubmissions:
    @pytest.mark.unit
    @patch("inference_endpoint.commands.run.cli._get")
    def test_success_table(self, mock_get, capsys):
        mock_get.return_value = _mock_resp(200, [_FAKE_SUB])
        list_submissions(token="tok")
        mock_get.assert_called_once()
        # Rich table output goes to _console, not capsys; just assert no exception

    @pytest.mark.unit
    @patch("inference_endpoint.commands.run.cli._get")
    def test_success_json(self, mock_get, capsys):
        mock_get.return_value = _mock_resp(200, [_FAKE_SUB])
        list_submissions(token="tok", json_output=True)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["id"] == "sub-123"

    @pytest.mark.unit
    @patch("inference_endpoint.commands.run.cli._get")
    def test_empty_list(self, mock_get, capsys):
        mock_get.return_value = _mock_resp(200, [])
        list_submissions(token="tok")
        out = capsys.readouterr().out
        assert "No submissions found" in out

    @pytest.mark.unit
    @patch("inference_endpoint.commands.run.cli._get")
    def test_error_propagates(self, mock_get):
        mock_get.return_value = _mock_resp(401, {"detail": "bad token"})
        with pytest.raises(InputValidationError, match="Authentication failed"):
            list_submissions(token="tok")

    @pytest.mark.unit
    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("ENDPOINTS_TOKEN", raising=False)
        with pytest.raises(InputValidationError, match="Token is required"):
            list_submissions(token=None)


# ---------------------------------------------------------------------------
# get_submission (in commands/run/cli.py)
# ---------------------------------------------------------------------------


class TestGetSubmission:
    @pytest.mark.unit
    @patch("inference_endpoint.commands.run.cli._get")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_resp(200, _FAKE_SUB)
        get_submission(token="tok", submission_id="sub-123")
        mock_get.assert_called_once()
        url, params = mock_get.call_args.args
        assert "sub-123" in url

    @pytest.mark.unit
    @patch("inference_endpoint.commands.run.cli._get")
    def test_success_json(self, mock_get, capsys):
        mock_get.return_value = _mock_resp(200, _FAKE_SUB)
        get_submission(token="tok", submission_id="sub-123", json_output=True)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["id"] == "sub-123"

    @pytest.mark.unit
    @patch("inference_endpoint.commands.run.cli._get")
    def test_include_runs_param(self, mock_get):
        mock_get.return_value = _mock_resp(200, _FAKE_SUB)
        get_submission(token="tok", submission_id="sub-123", include_runs=False)
        _, params = mock_get.call_args.args
        assert params["include_runs"] == "false"

    @pytest.mark.unit
    @patch("inference_endpoint.commands.run.cli._get")
    def test_not_found(self, mock_get):
        mock_get.return_value = _mock_resp(404, {"detail": "not found"})
        with pytest.raises(InputValidationError, match="not found"):
            get_submission(token="tok", submission_id="nope")


# ---------------------------------------------------------------------------
# create_submission
# ---------------------------------------------------------------------------


class TestCreateSubmission:
    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._post")
    def test_success(self, mock_post, capsys):
        mock_post.return_value = _mock_resp(201, _FAKE_SUB)
        create_submission(
            token="tok",
            benchmark_version="a1b2c3d",
            publication_cycle="2025-04-C1",
            run_ids=["run-abc"],
        )
        out = capsys.readouterr().out
        assert "Created submission" in out
        assert "sub-123" in out

    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._post")
    def test_payload_fields(self, mock_post):
        mock_post.return_value = _mock_resp(201, _FAKE_SUB)
        create_submission(
            token="tok",
            availability="closed",
            benchmark_version="v2",
            division="open",
            early_publish=True,
            publication_cycle="2025-06-C1",
            run_ids=["r1", "r2"],
        )
        _, params, body = mock_post.call_args.args
        assert body["availability"] == "closed"
        assert body["division"] == "open"
        assert body["early_publish"] is True
        assert body["run_ids"] == ["r1", "r2"]

    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._post")
    def test_empty_run_ids_defaults_to_list(self, mock_post):
        mock_post.return_value = _mock_resp(201, _FAKE_SUB)
        create_submission(token="tok", benchmark_version="v1", publication_cycle="2025-04-C1")
        _, params, body = mock_post.call_args.args
        assert body["run_ids"] == []

    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._post")
    def test_error_propagates(self, mock_post):
        mock_post.return_value = _mock_resp(422, {"detail": "bad payload"})
        with pytest.raises(InputValidationError, match="Validation error"):
            create_submission(token="tok", benchmark_version="v1", publication_cycle="c1")

    @pytest.mark.unit
    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("ENDPOINTS_TOKEN", raising=False)
        with pytest.raises(InputValidationError, match="Token is required"):
            create_submission(token=None, benchmark_version="v1", publication_cycle="c1")


# ---------------------------------------------------------------------------
# update_submission
# ---------------------------------------------------------------------------


class TestUpdateSubmission:
    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._patch")
    def test_success(self, mock_patch, capsys):
        mock_patch.return_value = _mock_resp(200, _FAKE_SUB)
        update_submission(token="tok", submission_id="sub-123", status="PEER_REVIEW_PENDING")
        out = capsys.readouterr().out
        assert "Updated submission" in out

    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._patch")
    def test_only_set_fields_sent(self, mock_patch):
        mock_patch.return_value = _mock_resp(200, _FAKE_SUB)
        update_submission(token="tok", submission_id="sub-123", pr_url="https://github.com/pr/1")
        _, params, body = mock_patch.call_args.args
        assert "pr_url" in body
        assert "status" not in body
        assert "pr_number" not in body

    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._patch")
    def test_all_fields(self, mock_patch):
        mock_patch.return_value = _mock_resp(200, _FAKE_SUB)
        update_submission(
            token="tok",
            submission_id="sub-123",
            status="PASSED",
            availability_qualified_at="2025-05-02T10:00:00",
            compliance_passed_at="2025-05-01T14:30:00",
            first_published_at="2025-05-03T09:00:00",
            peer_review_started_at="2025-05-04T08:00:00",
            objection_resolution_started_at="2025-05-05T07:00:00",
            finalized_at="2025-05-06T12:00:00",
            pr_url="https://github.com/pr/42",
            pr_number=42,
            archive_uri="s3://bucket/path.tar.gz",
            publication_cycle="2025-04-C1",
            target_availability_date="2025-06-01",
        )
        _, params, body = mock_patch.call_args.args
        assert body["status"] == "PASSED"
        assert body["pr_number"] == 42
        assert body["compliance_passed_at"] == "2025-05-01T14:30:00"
        assert body["availability_qualified_at"] == "2025-05-02T10:00:00"
        assert body["archive_uri"] == "s3://bucket/path.tar.gz"
        assert body["publication_cycle"] == "2025-04-C1"
        assert body["target_availability_date"] == "2025-06-01"

    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._patch")
    def test_none_fields_excluded(self, mock_patch):
        mock_patch.return_value = _mock_resp(200, _FAKE_SUB)
        update_submission(token="tok", submission_id="sub-123", status="PASSED")
        _, params, body = mock_patch.call_args.args
        assert set(body.keys()) == {"status"}

    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._patch")
    def test_error_propagates(self, mock_patch):
        mock_patch.return_value = _mock_resp(404, {"detail": "not found"})
        with pytest.raises(InputValidationError, match="not found"):
            update_submission(token="tok", submission_id="nope", status="PASSED")


# ---------------------------------------------------------------------------
# withdraw_submission
# ---------------------------------------------------------------------------


class TestWithdrawSubmission:
    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._delete")
    def test_success(self, mock_delete, capsys):
        mock_delete.return_value = _mock_resp(200, {"id": "sub-123"})
        withdraw_submission(token="tok", submission_id="sub-123")
        out = capsys.readouterr().out
        assert "Withdrawn submission" in out
        assert "sub-123" in out

    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._delete")
    def test_calls_correct_url(self, mock_delete):
        mock_delete.return_value = _mock_resp(200, {"id": "sub-123"})
        withdraw_submission(token="tok", submission_id="sub-123", api_url="http://localhost:9000")
        url, params = mock_delete.call_args.args
        assert "http://localhost:9000/submissions/sub-123" == url

    @pytest.mark.unit
    @patch("inference_endpoint.commands.submission.cli._delete")
    def test_error_propagates(self, mock_delete):
        mock_delete.return_value = _mock_resp(403, {"detail": "forbidden"})
        with pytest.raises(InputValidationError, match="not authorized"):
            withdraw_submission(token="tok", submission_id="sub-123")

    @pytest.mark.unit
    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("ENDPOINTS_TOKEN", raising=False)
        with pytest.raises(InputValidationError, match="Token is required"):
            withdraw_submission(token=None, submission_id="sub-123")

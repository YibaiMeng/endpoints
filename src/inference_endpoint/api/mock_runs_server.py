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

"""Local-dev mock of the MLCommons Endpoints Backend (port 8081 by default).

Simulates the backend that runs in GCP behind a proxy on port 8080.
Has NO PRISM auth — auth is enforced by the proxy API (app.py) on port 8082.

Start with:
    python -m inference_endpoint.api.mock_runs_server

Stores runs and submissions in memory only — not connected to any real database.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Mock /runs server — local dev only")

# In-memory store: run_id -> run dict
_runs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class _SystemInfoIn(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    email: str = ""
    company_name: str = ""
    company_external_id: str = ""


class _RunCreateIn(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    started_at: str
    finished_at: str
    expires_at: str | None = None
    pinned: bool = False
    system_info: _SystemInfoIn = _SystemInfoIn()
    config: dict[str, Any] = {}
    result_summary: dict[str, Any] = {}
    archive_uri: str = ""


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _require_user_match(run: dict[str, Any], user_id: str | None, label: str) -> None:
    """Return 403 when a non-None user_id doesn't match the run's owner."""
    if user_id is not None and user_id != run.get("user_id"):
        raise HTTPException(
            http_status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: {label} does not own this run",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/runs", status_code=http_status.HTTP_201_CREATED)
def create_run(user_id: str, body: _RunCreateIn) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    expires_at = (
        body.expires_at or (datetime.now(tz=UTC) + timedelta(days=365)).isoformat()
    )

    run: dict[str, Any] = {
        "id": run_id,
        "user_id": user_id,
        "started_at": body.started_at,
        "finished_at": body.finished_at,
        "expires_at": expires_at,
        "pinned": body.pinned,
        "system_info": body.system_info.model_dump(),
        "config": body.config,
        "result_summary": body.result_summary,
        "archive_uri": body.archive_uri,
    }
    _runs[run_id] = run
    return run


@app.get("/runs")
def list_runs(user_id: str | None = None) -> list[dict[str, Any]]:
    """Return runs, optionally filtered to a single owner."""
    if user_id is not None:
        return [r for r in _runs.values() if r["user_id"] == user_id]
    return list(_runs.values())


@app.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    if run_id not in _runs:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail="Run not found")
    run = _runs[run_id]
    x_user_id = request.headers.get("X-User-Id")
    _require_user_match(run, x_user_id, "X-User-Id")
    return run


@app.delete("/runs/{run_id}")
def delete_run(run_id: str, request: Request) -> str:
    if run_id not in _runs:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail="Run not found")
    run = _runs[run_id]
    x_user_id = request.headers.get("X-User-Id")
    _require_user_match(run, x_user_id, "X-User-Id")
    del _runs[run_id]
    return run_id


@app.patch("/runs/{run_id}/pin")
def pin_run(run_id: str, user_id: str | None = None) -> str:
    if run_id not in _runs:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail="Run not found")
    run = _runs[run_id]
    _require_user_match(run, user_id, "user_id")
    _runs[run_id] = {**run, "pinned": True}
    return run_id


@app.patch("/runs/{run_id}/unpin")
def unpin_run(run_id: str, user_id: str | None = None) -> str:
    if run_id not in _runs:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail="Run not found")
    run = _runs[run_id]
    _require_user_match(run, user_id, "user_id")
    _runs[run_id] = {**run, "pinned": False}
    return run_id


@app.delete("/runs", status_code=http_status.HTTP_204_NO_CONTENT)
def clear_all_runs() -> None:
    """Remove all stored runs — for test resets only."""
    _runs.clear()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_BANNER = """\
┌───────────────────────────────────────────────┐
│  Mock /runs server — local dev only (NO AUTH)    │
│  Listening on http://localhost:{port:<17}│
│  NOT connected to any real database            │
│  Auth proxy runs separately on port 8082       │
└───────────────────────────────────────────────┘\
"""

if __name__ == "__main__":
    port = int(os.environ.get("MOCK_PORT", "8081"))
    print(_BANNER.format(port=port))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

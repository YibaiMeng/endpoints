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

"""FastAPI router: POST /push_run — extract archive, forward to /runs."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict

from inference_endpoint.api.auth import PRISMIdentity, require_auth

logger = logging.getLogger(__name__)

_RUNS_API_BASE_URL: str = os.environ.get(
    "RUNS_API_BASE_URL", "http://localhost:8081"
).rstrip("/")
_GLOB_DIR: Path = Path(os.environ.get("GLOB_DIR", "./glob"))

router = APIRouter(prefix="/push_run", tags=["push"])

# ---------------------------------------------------------------------------
# Payload models
# ---------------------------------------------------------------------------


class _SystemInfo(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    email: str = ""
    company_name: str = ""
    company_external_id: str = ""


class _RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    started_at: str
    finished_at: str
    expires_at: str | None
    pinned: bool
    system_info: _SystemInfo
    config: dict[str, Any]
    result_summary: dict[str, Any]
    archive_uri: str
    model: str | None = None
    concurrency: int | None = None


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

_REQUIRED_FILES = frozenset(
    ["config.yaml", "result_summary.json", "runtime_settings.json", "events.jsonl"]
)


def _extract_archive(upload_path: Path, extract_dir: Path) -> None:
    if not tarfile.is_tarfile(upload_path):
        raise HTTPException(
            http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is not a valid tar archive",
        )
    with tarfile.open(upload_path, "r:gz") as tf:
        members = []
        for m in tf.getmembers():
            m.name = m.name.lstrip("/").replace("..", "__")
            members.append(m)
        tf.extractall(extract_dir, members=members)  # noqa: S202


def _find_run_root(extract_dir: Path) -> Path:
    """Handle archives where files sit at root or inside a single top-level folder."""
    if all((extract_dir / f).exists() for f in _REQUIRED_FILES):
        return extract_dir
    subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        candidate = subdirs[0]
        if all((candidate / f).exists() for f in _REQUIRED_FILES):
            return candidate
    return extract_dir


def _validate_run_files(run_root: Path) -> list[str]:
    return [f for f in _REQUIRED_FILES if not (run_root / f).exists()]


def _parse_run_files(
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        with (run_root / "config.yaml").open() as fh:
            config = yaml.safe_load(fh)
    except Exception as exc:
        raise HTTPException(
            http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed run data: config.yaml — {exc}",
        ) from exc

    try:
        with (run_root / "result_summary.json").open() as fh:
            result_summary = json.load(fh)
    except Exception as exc:
        raise HTTPException(
            http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed run data: result_summary.json — {exc}",
        ) from exc

    try:
        events: list[dict[str, Any]] = []
        with (run_root / "events.jsonl").open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception as exc:
        raise HTTPException(
            http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed run data: events.jsonl — {exc}",
        ) from exc

    return config, result_summary, events


def _extract_timestamps(events: list[dict[str, Any]]) -> tuple[str, str]:
    if not events:
        raise HTTPException(
            http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed run data: events.jsonl is empty",
        )
    started_at = events[0].get("approx_datetime_str", "")
    finished_at = events[-1].get("approx_datetime_str", "")
    if not started_at or not finished_at:
        raise HTTPException(
            http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed run data: missing approx_datetime_str in events",
        )
    return started_at, finished_at


def _extract_model_concurrency(
    config: dict[str, Any],
) -> tuple[str | None, int | None]:
    """Pull model name and concurrency from a parsed config.yaml dict."""
    model: str | None = (config.get("model_params") or {}).get("name")
    lp: dict[str, Any] = (config.get("settings") or {}).get("load_pattern") or {}
    concurrency: int | None = None
    if lp.get("type") == "concurrency":
        raw = lp.get("target_concurrency")
        concurrency = int(raw) if raw is not None else None
    return model, concurrency


def _store_in_glob(run_root: Path, folder_name: str) -> Path:
    dest = _GLOB_DIR / folder_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(run_root, dest)
    return dest.resolve()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("", status_code=http_status.HTTP_201_CREATED)
async def push_run(
    archive: UploadFile,
    identity: PRISMIdentity = Depends(require_auth),
) -> dict[str, Any]:
    """Accept a run archive and forward it to the runs API."""
    logger.info(
        "push_run: auth OK — user_id=%s email=%s", identity.user_id, identity.email
    )

    system_info = _SystemInfo(
        email=identity.email,
        company_name=identity.company_name,
        company_external_id=identity.company_external_id,
    )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            upload_path = tmp_path / "upload.tar.gz"

            content = await archive.read()
            upload_path.write_bytes(content)
            logger.info(
                "push_run: archive received — filename=%s size=%d bytes",
                archive.filename,
                len(content),
            )

            extract_dir = tmp_path / "extracted"
            extract_dir.mkdir()
            _extract_archive(upload_path, extract_dir)
            logger.info("push_run: archive extracted OK")

            run_root = _find_run_root(extract_dir)
            missing = _validate_run_files(run_root)
            if missing:
                logger.error("push_run: archive missing required files: %s", missing)
                raise HTTPException(
                    http_status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"detail": "Missing required files", "missing": missing},
                )

            config, result_summary, events = _parse_run_files(run_root)
            logger.info(
                "push_run: run files parsed OK — %d events", len(events)
            )

            started_at, finished_at = _extract_timestamps(events)
            logger.info(
                "push_run: timestamps — started=%s finished=%s", started_at, finished_at
            )

            folder_name = (
                Path(archive.filename).stem.removesuffix(".tar")
                if archive.filename
                else "run"
            )
            archive_path = _store_in_glob(run_root, folder_name)
            logger.info("push_run: stored in glob at %s", archive_path)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("push_run: unexpected error during archive processing: %s", exc)
        raise HTTPException(
            http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error processing archive: {exc}",
        ) from exc

    model, concurrency = _extract_model_concurrency(config)
    logger.info("push_run: extracted model=%s concurrency=%s", model, concurrency)

    run_create = _RunCreate(
        started_at=started_at,
        finished_at=finished_at,
        expires_at=None,
        pinned=False,
        system_info=system_info,
        config=config,
        result_summary=result_summary,
        archive_uri=str(archive_path),
        model=model,
        concurrency=concurrency,
    )

    upstream_url = f"{_RUNS_API_BASE_URL}/runs"
    logger.info(
        "push_run: forwarding to upstream %s (user_id=%s)", upstream_url, identity.user_id
    )

    async with httpx.AsyncClient() as client:
        try:
            runs_resp = await client.post(
                upstream_url,
                params={"user_id": identity.user_id},
                json=run_create.model_dump(),
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            logger.error("push_run: upstream unreachable — %s", exc)
            raise HTTPException(
                http_status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to create run: runs API unreachable — {exc}",
            ) from exc

    logger.info(
        "push_run: upstream responded %d — body: %s",
        runs_resp.status_code,
        runs_resp.text[:500],
    )

    if runs_resp.status_code == http_status.HTTP_201_CREATED:
        return runs_resp.json()

    if runs_resp.status_code == http_status.HTTP_422_UNPROCESSABLE_ENTITY:
        try:
            upstream_errors = runs_resp.json().get("detail", [])
        except Exception:
            upstream_errors = runs_resp.text
        logger.error("push_run: upstream rejected payload (422): %s", upstream_errors)
        raise HTTPException(
            http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": "Run creation failed — payload rejected by runs API",
                "upstream_errors": upstream_errors,
            },
        )

    logger.error(
        "push_run: upstream returned unexpected %d: %s",
        runs_resp.status_code,
        runs_resp.text[:500],
    )
    raise HTTPException(
        http_status.HTTP_502_BAD_GATEWAY,
        detail=f"Failed to create run: {runs_resp.status_code} {runs_resp.text[:200]}",
    )

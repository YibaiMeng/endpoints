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

"""Top-level FastAPI application — wires push and proxy routers together.

Start with:
    PRISM_USER_TOKEN=<token> uvicorn inference_endpoint.api.app:app --port 8082
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from inference_endpoint.api.push_run import _GLOB_DIR
from inference_endpoint.api.push_run import router as push_router
from inference_endpoint.api.runs_proxy import router as runs_proxy_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLCommons Endpoints API",
    version="0.1.0",
    description=(
        "Auth proxy for pushing benchmark runs and managing them via PRISM-validated tokens. "
        "Point RUNS_API_BASE_URL at the real DB or leave unset to use the local mock on 8081."
    ),
)

app.include_router(push_router)
app.include_router(runs_proxy_router)


@app.on_event("startup")
async def _startup() -> None:
    prism_token = os.environ.get("PRISM_USER_TOKEN")
    if not prism_token:
        raise RuntimeError(
            "PRISM_USER_TOKEN environment variable is required but not set. "
            "This is the server-side bearer token for calling PRISM."
        )
    _GLOB_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "app: startup OK (RUNS_API_BASE_URL=%s, GLOB_DIR=%s)",
        os.environ.get("RUNS_API_BASE_URL", "http://localhost:8081"),
        _GLOB_DIR,
    )

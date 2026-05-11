# Push Run — Architecture & Testing Guide

How `inference-endpoint push run` works end-to-end, and how to test it at every level:
automated tests, full local dev with the mock server, and raw `curl`.

---

## Architecture overview

The feature is split into four modules that each own one concern.

```
src/inference_endpoint/api/
├── app.py            ← top-level FastAPI app (mounts the two routers)
├── auth.py           ← shared PRISM auth layer
├── push_run.py       ← POST /push_run  (upload + forward to /runs)
└── runs_proxy.py     ← GET|DELETE|PATCH /runs/* (authenticated proxy)

src/inference_endpoint/api/mock_runs_server.py  ← local dev mock (port 8081)
src/inference_endpoint/commands/push/cli.py     ← CLI: inference-endpoint push run
```

### Component diagram

```
  User / CI
     │
     │  inference-endpoint push run --path <path> --token <mlc_key>
     ▼
┌──────────────────────────────────────────────────────────┐
│                   CLI  (push/cli.py)                     │
│  validate dir → create .tar.gz → POST /push_run?token=… │
└──────────────────────────┬───────────────────────────────┘
                           │  multipart/form-data
                           ▼
┌──────────────────────────────────────────────────────────┐
│        Auth proxy  (app.py on port 8082)                 │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  auth.py  ─  require_auth(token)                 │   │
│  │    1. POST PRISM /validate-api-key                │   │
│  │    2. GET  PRISM /validate-api-key?user_id=…      │   │
│  │    → PRISMIdentity {user_id, email, company_name} │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  push_run.py                                      │   │
│  │    extract archive → validate files               │   │
│  │    store in GLOB_DIR                              │   │
│  │    POST /runs?user_id=… → backend                 │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  runs_proxy.py  (5 routes)                        │   │
│  │    GET    /runs              → GET  upstream + ?user_id   │
│  │    GET    /runs/{id}         → GET  upstream + X-User-Id  │
│  │    DELETE /runs/{id}         → DEL  upstream + X-User-Id  │
│  │    PATCH  /runs/{id}/pin     → PATCH upstream + ?user_id  │
│  │    PATCH  /runs/{id}/unpin   → PATCH upstream + ?user_id  │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────┘
                           │  HTTP
                           ▼
              ┌──────────────────────────────────────────┐
              │  MLCommons Endpoints Backend  (port 8080) │
              │  GCP-hosted; local mock on port 8081      │
              └──────────────────────────────────────────┘
```

---

## Auth flow (per request)

Every route goes through the same `require_auth` dependency before any business logic runs.

```
Client request
  ?token=mlc_…
       │
       ▼
  require_auth()  ──── PRISMAuthError ──────────────► HTTP 400/401/403/429/500
       │
       │  POST https://prism.mlcommons.org/…/validate-api-key
       │         {api_key: token, service_id: "endpoints"}
       │
       │  response.valid == False?
       │     INVALID_FORMAT   → 400
       │     INVALID_KEY      → 401
       │     INACTIVE_KEY     → 401
       │     EXPIRED_KEY      → 401
       │     SERVICE_MISMATCH → 403
       │
       │  response.status == 401  → 500 (server misconfiguration)
       │  response.status == 429  → 429 (rate limited, retry_after_seconds)
       │
       │  GET https://prism.mlcommons.org/…/validate-api-key?user_id=…
       │       → {id, email, company_name, company_external_id}
       │
       ▼
  PRISMIdentity{user_id, email, company_name, company_external_id}
       │
       ▼
  route handler continues
```

---

## Push run flow (`POST /push_run`)

```
CLI                       push_run.py                    MLCommons Endpoints Backend (8080)
 │                              │                              │
 │── POST /push_run?token=… ───►│                              │
 │   (multipart: archive file)  │                              │
 │                              │── require_auth ──────────────►│ (PRISM, not shown)
 │                              │◄─ PRISMIdentity ─────────────│
 │                              │                              │
 │                              │  extract .tar.gz             │
 │                              │  validate required files     │
 │                              │  (config.yaml,               │
 │                              │   result_summary.json,       │
 │                              │   runtime_settings.json,     │
 │                              │   events.jsonl)              │
 │                              │                              │
 │                              │  copy to GLOB_DIR/           │
 │                              │  parse timestamps from events│
 │                              │                              │
 │                              │── POST /runs?user_id=… ─────►│
 │                              │   {started_at, finished_at,  │
 │                              │    system_info, config,      │
 │                              │    result_summary,           │
 │                              │    archive_uri}              │
 │                              │◄─ 201 {id, user_id, …} ─────│
 │◄─ 201 {run object} ─────────│                              │
```

---

## Proxy routes flow (`GET|DELETE|PATCH /runs/…`)

```
Client                    runs_proxy.py                  MLCommons Endpoints Backend (8080)
 │                              │                              │
 │── GET /runs?token=… ────────►│                              │
 │                              │── require_auth ──────────────► (PRISM)
 │                              │◄─ PRISMIdentity ─────────────│
 │                              │── GET /runs?user_id=… ──────►│
 │◄─ 200 [...runs...] ─────────│◄─ 200 [...] ─────────────────│

 │── GET /runs/{id}?token=… ──►│
 │                              │── GET /runs/{id}            ►│
 │                              │   headers: X-User-Id: <uid>  │
 │◄─ 200 {run} ────────────────│◄─ 200 {run} ─────────────────│

 │── DELETE /runs/{id}?token=… ►│
 │                              │── DELETE /runs/{id}         ►│
 │                              │   headers: X-User-Id: <uid>  │
 │◄─ 200 "<run_id>" ───────────│◄─ 200 "<run_id>" ────────────│

 │── PATCH /runs/{id}/pin ─────►│
 │   ?token=…                   │── PATCH /runs/{id}/pin      ►│
 │                              │   params: user_id=<uid>      │
 │◄─ 200 "<run_id>" ───────────│◄─ 200 "<run_id>" ────────────│
```

Note the difference in upstream auth:

- **List, pin, unpin** → `?user_id=` query param (route uses `user_id` in URL)
- **Get, delete** → `X-User-Id` header (route uses path segment, needs header auth)

---

## Prerequisites

```bash
# Using uv (recommended — matches the lockfile exactly)
uv sync --extra api --extra test

# Using pip
pip install -e ".[api,test]"
```

---

## 1. Automated tests

88 tests across four files, all tagged `@pytest.mark.unit`.

```bash
# All four files at once
pytest tests/test_auth.py tests/test_runs_proxy.py tests/test_push_run.py tests/test_run_cli.py -v

# Or by marker
pytest tests/test_auth.py tests/test_runs_proxy.py tests/test_push_run.py tests/test_run_cli.py -m unit -v
```

### What each file covers

| File                 | Classes                                                                                                                                                       | Count | Focus                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | -------------------------------------------------------------------------------------- |
| `test_auth.py`       | `TestVerifyTokenSuccess` `TestVerifyTokenPrismErrorCodes` `TestVerifyTokenHttpErrors` `TestRequireAuthDependency`                                             | 13    | PRISM token validation, all error codes, FastAPI dependency mapping                    |
| `test_runs_proxy.py` | `TestListRuns` `TestGetRun` `TestDeleteRun` `TestPinRun` `TestUnpinRun`                                                                                       | 19    | Auth gate, upstream forwarding, 502 on network failure, correct header/param per route |
| `test_push_run.py`   | `TestCliPreflightValidation` `TestCliTokenResolution` `TestCliDryRun` `TestCliErrorMapping` `TestApiArchiveValidation` `TestApiRunsErrors` `TestApiHappyPath` | 23    | CLI validation, dry run, HTTP→exception mapping, archive validation, full push flow    |
| `test_run_cli.py`    | `TestResolveToken` `TestHandleError` `TestListRun` `TestGetRun` `TestDeleteRun` `TestPinRun` `TestUnpinRun`                                                   | 29    | Token resolution, error mapping, all five run-management CLI commands                  |

### How auth is mocked in tests

`test_auth.py` tests the auth layer itself (patches `httpx.AsyncClient` inside `auth.py`).

`test_runs_proxy.py` and the API tests in `test_push_run.py` bypass auth entirely:

```python
# In the test file:
app.dependency_overrides[require_auth] = lambda: PRISMIdentity(
    user_id="test-uuid",
    email="test@example.com",
    company_name="TestCo",
    company_external_id="ext-1",
)
```

This means proxy/push route tests only exercise _their own_ logic — not PRISM.

---

## 2. Manual local flow (three terminals)

No real credentials needed. Everything runs in-process on localhost.

### Terminal 1 — mock backend (port 8081)

The real MLCommons Endpoints Backend runs in GCP on port 8080. For local dev, the mock
simulates it on port 8081 so there is no conflict if you also have a GCP tunnel open.

```bash
python -m inference_endpoint.api.mock_runs_server
```

```
┌───────────────────────────────────────────────────┐
│  Mock MLCommons Endpoints Backend — local dev only │
│  Listening on http://localhost:8081  (NO AUTH)     │
│  NOT connected to any real database                │
│  Auth proxy runs separately on port 8082           │
└───────────────────────────────────────────────────┘
```

### Terminal 2 — auth proxy (port 8082)

```bash
PRISM_USER_TOKEN=<your-server-prism-token> \
RUNS_API_BASE_URL=http://localhost:8081 \
uvicorn inference_endpoint.api.app:app --port 8082 --reload
```

`RUNS_API_BASE_URL` defaults to `http://localhost:8080` (the GCP proxy). Override it to
`http://localhost:8081` when running the local mock instead.

> `PRISM_USER_TOKEN` is the **server-side** bearer credential your service uses to call PRISM
> — not a user API key. The app startup will fail loudly if it's missing.

### Terminal 3 — CLI

All commands accept `--token` (or set `ENDPOINTS_TOKEN` env var) and `--api-url` (default `http://localhost:8082`).

**Push a run:**

```bash
inference-endpoint push run \
  --path ./endpoints_run_samples/llama-3.1-8b_vllm_perf_concurrency1_3a8d044-06ccd43_main_trial1 \
  --token <your-prism-api-key> \
  --api-url http://localhost:8082
```

**Token from env (no `--token` flag):**

```bash
export ENDPOINTS_TOKEN=<your-prism-api-key>
inference-endpoint push run \
  --path ./endpoints_run_samples/llama-3.1-8b_vllm_perf_concurrency1_3a8d044-06ccd43_main_trial1 \
  --api-url http://localhost:8082
```

**List your runs:**

```bash
inference-endpoint list run --token <your-prism-api-key> --api-url http://localhost:8082
```

**Get a single run:**

```bash
inference-endpoint get run \
  --token <your-prism-api-key> \
  --run_id <uuid-from-push-response> \
  --api-url http://localhost:8082
```

**Pin / unpin a run:**

```bash
inference-endpoint pin run \
  --token <your-prism-api-key> \
  --run_id <uuid-from-push-response> \
  --api-url http://localhost:8082

inference-endpoint unpin run \
  --token <your-prism-api-key> \
  --run_id <uuid-from-push-response> \
  --api-url http://localhost:8082
```

**Delete a run:**

```bash
inference-endpoint delete run \
  --token <your-prism-api-key> \
  --run_id <uuid-from-push-response> \
  --api-url http://localhost:8082
```

**Dry run — no servers, no network:**

```bash
inference-endpoint push run \
  --path ./endpoints_run_samples/llama-3.1-8b_vllm_perf_concurrency1_3a8d044-06ccd43_main_trial1 \
  --token dummy-token \
  --dry-run
```

```
Dry run — no upload will be performed.
  Archive : llama-3.1-8b_vllm_perf_concurrency1_3a8d044-06ccd43_main_trial1.tar.gz
  Size    : 18.42 MB
  Token   : ****1234
  Contents:
    config.yaml
    result_summary.json
    runtime_settings.json
    events.jsonl
```

---

## 3. Environment variable reference

| Variable            | Component   | Default                 | Description                                                                                                  |
| ------------------- | ----------- | ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| `PRISM_USER_TOKEN`  | API server  | _(required)_            | Server-side bearer token for calling PRISM. App startup fails if missing.                                    |
| `RUNS_API_BASE_URL` | API server  | `http://localhost:8080` | Base URL of the MLCommons Endpoints Backend (GCP proxy on port 8080). Set to `http://localhost:8081` for the local mock. |
| `GLOB_DIR`          | API server  | `./glob`                | Directory where extracted run archives are stored.                                                           |
| `ENDPOINTS_TOKEN`   | CLI         | _(none)_                | User PRISM API key — fallback when `--token` is not passed.                                                  |
| `MOCK_PORT`         | mock server | `8081`                  | Port for the local mock backend (distinct from the real GCP port 8080).                                      |

---

## 4. Common errors

| Symptom                                                                          | Cause                                  | Fix                                                                  |
| -------------------------------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------- |
| `RuntimeError: PRISM_USER_TOKEN environment variable is required`                | Not set before `uvicorn`               | `export PRISM_USER_TOKEN=<token>` before starting the server         |
| `InputValidationError: Token format is invalid`                                  | API key is not `mlc_` + 64 chars       | Check your key in the MLCommons portal                               |
| `InputValidationError: Authentication failed: Invalid API token`                 | Wrong or revoked key                   | Re-generate the key in the MLCommons portal                          |
| `InputValidationError: Token not authorized for the Endpoints service`           | Key was issued for a different service | Request an Endpoints-scoped key                                      |
| `ExecutionError: Upload failed — could not reach http://localhost:8082/push_run` | Push API not running                   | Start `uvicorn inference_endpoint.api.app:app --port 8082`           |
| `InputValidationError: Missing required files: events.jsonl`                     | Incomplete run directory               | Ensure the run completed and all output files were written           |
| `ExecutionError: Upstream service unavailable`                                   | MLCommons Endpoints Backend unreachable | Check `RUNS_API_BASE_URL` and that the mock or GCP backend is reachable |
| API tests skipped                                                                | FastAPI extras not installed           | `pip install -e ".[api,test]"` or `uv sync --extra api --extra test` |

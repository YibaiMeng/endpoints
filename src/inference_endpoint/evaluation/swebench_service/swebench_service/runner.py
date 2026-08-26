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

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse, urlunparse

import msgspec.json
import yaml

from .artifacts import atomic_write_bytes, redact_secrets, redact_text
from .schemas import RunRequest, TemplateName

logger = logging.getLogger(__name__)


class RunnerError(RuntimeError):
    pass


class RunCancelled(RunnerError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            process = self._process
        if process is not None:
            _terminate_process(process)

    def attach(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._process = process
            cancelled = self._event.is_set()
        if cancelled:
            _terminate_process(process)

    def detach(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            if self._process is process:
                self._process = None


TEMPLATE_FILES: dict[TemplateName, str] = {
    "default": "swebench_template.yaml",
    "qwen_tools": "swebench_qwen_tools_template.yaml",
}

_LOG_TAIL_MAX_BYTES = 64 * 1024
_LOG_TAIL_MAX_LINES = 50
_RUN_LABEL = "com.mlcommons.endpoints.swebench-run"
_PROCESS_TERMINATE_TIMEOUT_S = 10
_SWEBENCH_DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
}


@dataclass(frozen=True)
class PyxisPlacement:
    """Trusted assignment of a SWE-bench instance to an allocated Slurm node."""

    nodes_by_instance: dict[str, str]

    def node_for(self, instance_id: str) -> str:
        try:
            return self.nodes_by_instance[instance_id]
        except KeyError as exc:
            raise RunnerError(
                f"Pyxis placement does not contain requested instance {instance_id}"
            ) from exc

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.nodes_by_instance.values())))

    def require_exact_instances(self, instance_ids: list[str]) -> None:
        expected = set(instance_ids)
        actual = set(self.nodes_by_instance)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing[:10]))
            if extra:
                details.append("unexpected: " + ", ".join(extra[:10]))
            raise RunnerError(
                "Pyxis placement must match requested instances exactly ("
                + "; ".join(details)
                + ")"
            )

    def write_snapshot(self, path: Path, instance_ids: list[str]) -> None:
        self.require_exact_instances(instance_ids)
        contents = "".join(
            f"{instance_id}\t{self.node_for(instance_id)}\n"
            for instance_id in instance_ids
        )
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(contents)
            path.chmod(0o444)
        except FileExistsError as exc:
            raise RunnerError(
                f"Pyxis placement snapshot already exists: {path}"
            ) from exc
        except OSError as exc:
            raise RunnerError(
                f"could not write Pyxis placement snapshot {path}: {exc}"
            ) from exc


def load_pyxis_placement(path: Path) -> PyxisPlacement:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RunnerError(f"could not read Pyxis placement file {path}: {exc}") from exc

    nodes_by_instance: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        instance_id, separator, node = line.partition("\t")
        if not separator or not instance_id or not node or "\t" in node:
            raise RunnerError(
                f"invalid Pyxis placement at {path}:{line_number}; "
                "expected instance_id<TAB>node"
            )
        if instance_id in nodes_by_instance:
            raise RunnerError(
                f"duplicate instance ID in Pyxis placement at {path}:{line_number}: "
                f"{instance_id}"
            )
        if any(character.isspace() for character in node):
            raise RunnerError(
                f"invalid Slurm node in Pyxis placement at {path}:{line_number}: {node}"
            )
        nodes_by_instance[instance_id] = node
    if not nodes_by_instance:
        raise RunnerError(f"Pyxis placement file {path} is empty")
    return PyxisPlacement(nodes_by_instance)


def _prepare_eval(request: RunRequest, run_dir: Path) -> tuple[str, str]:
    run_id = f"endpoints_{uuid.uuid4().hex[:8]}"
    (run_dir / "swe_bench_eval_run_id.txt").write_text(run_id)
    dataset_name = _SWEBENCH_DATASETS.get(request.subset)
    if dataset_name is None:
        raise RunnerError(f"unknown SWE-bench subset: {request.subset}")
    return run_id, dataset_name


def _resolve_eval_result(output_dir: Path, model_name: str, run_id: str) -> Path:
    result_path = output_dir / f"{model_name.replace('/', '__')}.{run_id}.json"
    if result_path.exists():
        return result_path
    candidates = sorted(output_dir.rglob(f"*{run_id}*.json"))
    if not candidates:
        raise RunnerError(f"SWE-bench result file not found for run_id={run_id}")
    if len(candidates) > 1:
        raise RunnerError(f"multiple SWE-bench result files found for run_id={run_id}")
    return candidates[0]


def _normalize_endpoint_base(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    hostname = parsed.hostname or ""
    if hostname == "localhost":
        hostname = "127.0.0.1"
    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunparse(
        parsed._replace(netloc=netloc, path=path, params="", query="", fragment="")
    )


def _exact_instance_filter(instance_ids: list[str]) -> str:
    return (
        "^(?:" + "|".join(re.escape(instance_id) for instance_id in instance_ids) + ")$"
    )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate the local process group; containers are cleaned separately."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("SWE-bench subprocess did not exit after SIGKILL")


def _run_subprocess(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
    cancel_token: CancellationToken | None = None,
) -> None:
    if cancel_token is not None and cancel_token.is_cancelled():
        raise RunCancelled(f"subprocess cancelled before start: {cmd}")
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(cwd),
                env=env,
                start_new_session=os.name != "nt",
            )
            if cancel_token is not None:
                cancel_token.attach(process)
            deadline = time.monotonic() + timeout_s
            while True:
                if cancel_token is not None and cancel_token.is_cancelled():
                    _terminate_process(process)
                    raise RunCancelled(f"subprocess cancelled: {cmd}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_process(process)
                    raise RunnerError(f"subprocess timed out after {timeout_s}s: {cmd}")
                try:
                    process.communicate(timeout=min(0.5, remaining))
                    if cancel_token is not None and cancel_token.is_cancelled():
                        raise RunCancelled(f"subprocess cancelled: {cmd}")
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        if process is not None and cancel_token is not None:
            cancel_token.detach(process)

    if process.returncode != 0:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - _LOG_TAIL_MAX_BYTES))
            tail_bytes = log_file.read()
        tail = "\n".join(
            tail_bytes.decode("utf-8", errors="replace").splitlines()[
                -_LOG_TAIL_MAX_LINES:
            ]
        )
        raise RunnerError(
            f"subprocess exited with code {process.returncode}: {cmd}\n{tail}"
        )


class RunnerProtocol(Protocol):
    """Structural interface used by the service to execute a SWE-bench run."""

    def run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]: ...


class SweBenchRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        subprocess_timeout_s: int,
    ):
        self.project_root = project_root.resolve()
        self.subprocess_timeout_s = subprocess_timeout_s

    def run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        try:
            return self._run(request, run_dir, cancel_token)
        finally:
            try:
                cleanup_kwargs: dict[str, Any] = {}
                eval_run_id_path = run_dir / "swe_bench_eval_run_id.txt"
                if eval_run_id_path.exists():
                    eval_run_id = eval_run_id_path.read_text().strip()
                    if eval_run_id:
                        cleanup_kwargs = {
                            "eval_run_id": eval_run_id,
                            "instance_ids": request.evaluated_instance_ids,
                        }
                self._cleanup_containers(run_dir.name, **cleanup_kwargs)
            except Exception:
                logger.warning(
                    "Could not clean up SWE-bench containers for run %s",
                    run_dir.name,
                    exc_info=True,
                )

    def _run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        secret_values = (
            {request.endpoint_api_key} if request.endpoint_api_key else set()
        )
        (run_dir / "request.json").write_bytes(
            msgspec.json.encode(
                redact_secrets(request.model_dump(), secret_values=secret_values)
            )
        )

        output_dir = run_dir / "swe_bench_output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        with tempfile.TemporaryDirectory(prefix="swebench_config_") as config_tmp:
            patched_config = self._patch_config(
                Path(config_tmp),
                request,
                run_id=run_dir.name,
            )
            self._run_agent(
                request,
                patched_config,
                output_dir,
                run_dir,
                secret_values,
                cancel_token,
            )

        preds_path = output_dir / "preds.json"
        if not preds_path.exists():
            raise RunnerError("mini-extra did not produce preds.json")
        self._validate_prediction_ids(request, preds_path)
        shutil.copy2(preds_path, run_dir / "preds.json")

        eval_failures_path = output_dir / "eval_infrastructure_failures.json"
        try:
            result_path = self._run_eval(
                request, preds_path, output_dir, run_dir, secret_values, cancel_token
            )
        finally:
            if eval_failures_path.exists():
                shutil.copy2(eval_failures_path, run_dir / eval_failures_path.name)
        shutil.copy2(result_path, run_dir / "swe_bench_results.json")
        return msgspec.json.decode(result_path.read_bytes(), type=dict)

    def _load_template(self, request: RunRequest) -> dict[str, Any]:
        template_path = self._template_dir / TEMPLATE_FILES[request.template]
        with template_path.open() as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise RunnerError("swebench template must be a YAML mapping")
        model_cfg = loaded.get("model")
        if not isinstance(model_cfg, dict):
            raise RunnerError("swebench template must define model")
        if not isinstance(model_cfg.get("model_kwargs"), dict):
            raise RunnerError("swebench template must define model.model_kwargs")
        return loaded

    @property
    def _template_dir(self) -> Path:
        return Path(__file__).resolve().parent / "templates"

    def _patch_config(
        self, config_dir: Path, request: RunRequest, *, run_id: str
    ) -> Path:
        cfg = self._load_template(request)
        model_cfg = cfg["model"]
        model_kwargs = model_cfg["model_kwargs"]

        model_cfg["model_name"] = request.model_name
        if request.template == "qwen_tools":
            model_cfg["model_class"] = (
                "swebench_service.qwen_tools_model.QwenToolsModel"
            )
        else:
            model_cfg.pop("model_class", None)
        if request.endpoint_urls:
            base = _normalize_endpoint_base(str(request.endpoint_urls[0]))
            model_kwargs["api_base"] = base + "/v1"
        else:
            base = ""
            model_kwargs["api_base"] = ""

        model_kwargs.pop("api_key", None)

        for field in (
            "temperature",
            "seed",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            val = request.generation_params.get(field)
            if val is not None:
                model_kwargs[field] = val
            else:
                model_kwargs.pop(field, None)

        if (
            max_new_tokens := request.generation_params.get("max_new_tokens")
        ) is not None:
            model_kwargs["max_tokens"] = max_new_tokens
        else:
            model_kwargs.pop("max_tokens", None)

        if (
            chat_tmpl := request.generation_params.get("chat_template_kwargs")
        ) is not None:
            model_kwargs["chat_template_kwargs"] = chat_tmpl
        else:
            model_kwargs.pop("chat_template_kwargs", None)

        environment_cfg = cfg.get("environment")
        if not isinstance(environment_cfg, dict):
            raise RunnerError("swebench template must define environment")
        self._configure_environment(environment_cfg, run_id)
        if request.agent_command_timeout_s is not None:
            environment_cfg["timeout"] = request.agent_command_timeout_s
        if request.agent_create_timeout_s is not None:
            environment_cfg["pull_timeout"] = request.agent_create_timeout_s
        if request.model_request_timeout_s is not None:
            model_kwargs["timeout"] = request.model_request_timeout_s

        config_dir.mkdir(parents=True, exist_ok=True)
        patched_path = config_dir / "swebench_patched.yaml"
        with patched_path.open("w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        return patched_path

    def _configure_environment(
        self, environment_cfg: dict[str, Any], run_id: str
    ) -> None:
        environment_cfg["run_args"] = [
            "--rm",
            "--label",
            f"{_RUN_LABEL}={run_id}",
        ]

    def _run_agent(
        self,
        request: RunRequest,
        patched_config: Path,
        output_dir: Path,
        run_dir: Path,
        secret_values: set[str],
        cancel_token: CancellationToken | None = None,
    ) -> None:
        instance_filter = _exact_instance_filter(request.evaluated_instance_ids)
        cmd = [
            "mini-extra",
            "swebench",
            "--model",
            request.model_name,
            "--config",
            str(patched_config),
            "--subset",
            request.subset,
            "--split",
            request.split,
            "--filter",
            instance_filter,
            "--workers",
            str(request.workers),
            "--output",
            str(output_dir),
        ]
        self._run_logged_subprocess(
            cmd,
            run_dir / "swe_bench_agent.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            env=self._base_env(request),
            secret_values=secret_values,
            cancel_token=cancel_token,
        )

    @staticmethod
    def _run_logged_subprocess(
        cmd: list[str],
        public_log_path: Path,
        *,
        cwd: Path,
        timeout_s: int,
        env: dict[str, str],
        secret_values: set[str],
        cancel_token: CancellationToken | None,
    ) -> None:
        raw_log_path = public_log_path.with_name(f".{public_log_path.name}.raw")
        try:
            _run_subprocess(
                cmd,
                raw_log_path,
                cwd=cwd,
                timeout_s=timeout_s,
                env=env,
                cancel_token=cancel_token,
            )
        finally:
            try:
                if raw_log_path.exists():
                    atomic_write_bytes(
                        public_log_path,
                        redact_text(
                            raw_log_path.read_text(errors="replace"), secret_values
                        ).encode(),
                    )
            finally:
                raw_log_path.unlink(missing_ok=True)

    def _base_env(self, request: RunRequest) -> dict[str, str]:
        env = dict(os.environ)
        no_proxy = {"127.0.0.1", "localhost"}
        for endpoint in request.endpoint_urls:
            host = urlparse(str(endpoint)).hostname
            if host:
                no_proxy.add(host)
        existing = env.get("NO_PROXY") or env.get("no_proxy")
        if existing:
            no_proxy.update(
                part.strip() for part in existing.split(",") if part.strip()
            )
        no_proxy_value = ",".join(sorted(no_proxy))
        env["NO_PROXY"] = no_proxy_value
        env["no_proxy"] = no_proxy_value
        endpoint_host = (
            urlparse(str(request.endpoint_urls[0])).hostname
            if request.endpoint_urls
            else None
        )
        if request.endpoint_api_key:
            env["OPENAI_API_KEY"] = request.endpoint_api_key
        elif endpoint_host in {"localhost", "127.0.0.1", "::1"}:
            env["OPENAI_API_KEY"] = "EMPTY"
        else:
            env.pop("OPENAI_API_KEY", None)
        return env

    def _cleanup_containers(
        self,
        run_id: str,
        *,
        eval_run_id: str | None = None,
        instance_ids: list[str] | None = None,
    ) -> None:
        docker = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
        label_filter = f"label={_RUN_LABEL}={run_id}"
        try:
            listed = subprocess.run(
                [docker, "ps", "-aq", "--filter", label_filter],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            container_ids = listed.stdout.split()
            if eval_run_id is not None:
                expected_names = {
                    f"sweb.eval.{instance_id.lower()}.{eval_run_id}"
                    for instance_id in instance_ids or []
                }
                listed_eval = subprocess.run(
                    [
                        docker,
                        "ps",
                        "-a",
                        "--filter",
                        f"name={eval_run_id}",
                        "--format",
                        "{{.ID}}\t{{.Names}}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                for line in listed_eval.stdout.splitlines():
                    container_id, separator, container_name = line.partition("\t")
                    if (
                        separator
                        and container_id
                        and container_name in expected_names
                        and container_id not in container_ids
                    ):
                        container_ids.append(container_id)
            if container_ids:
                subprocess.run(
                    [docker, "rm", "-f", *container_ids],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RunnerError(
                f"failed to clean up Docker containers for SWE-bench run {run_id}"
            ) from exc

    def _validate_prediction_ids(self, request: RunRequest, preds_path: Path) -> None:
        try:
            preds = msgspec.json.decode(preds_path.read_bytes(), type=dict)
        except msgspec.DecodeError as exc:
            raise RunnerError("mini-extra produced invalid preds.json") from exc
        expected = set(request.evaluated_instance_ids)
        actual = {str(instance_id) for instance_id in preds}
        unexpected = sorted(actual - expected)
        if unexpected:
            raise RunnerError(
                "mini-extra produced predictions for unexpected SWE-bench "
                f"instances: {', '.join(unexpected[:10])}"
            )
        missing = sorted(expected - actual)
        if missing:
            logger.warning(
                "mini-extra omitted predictions for %d expected SWE-bench "
                "instances: %s",
                len(missing),
                ", ".join(missing[:10]),
            )

    def _run_eval(
        self,
        request: RunRequest,
        preds_path: Path,
        output_dir: Path,
        run_dir: Path,
        secret_values: set[str],
        cancel_token: CancellationToken | None = None,
    ) -> Path:
        run_id, dataset_name = _prepare_eval(request, run_dir)
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            request.split,
            "--predictions_path",
            str(preds_path),
            "--max_workers",
            str(request.max_eval_workers),
            "--run_id",
            run_id,
            "--instance_ids",
            *request.evaluated_instance_ids,
        ]
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)
        self._run_logged_subprocess(
            cmd,
            run_dir / "swe_bench_eval.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            env=env,
            secret_values=secret_values,
            cancel_token=cancel_token,
        )
        return _resolve_eval_result(output_dir, request.model_name, run_id)


class PyxisSweBenchRunner(SweBenchRunner):
    def __init__(
        self,
        *,
        project_root: Path,
        subprocess_timeout_s: int,
        image_registry: str,
        pyxis_placement_file: Path | None = None,
        pyxis_shared_runtime_root: Path | None = None,
        pyxis_max_concurrent_creates: int | None = None,
        pyxis_max_concurrent_srun_steps: int | None = None,
        pyxis_srun_launch_grace_s: int = 30,
    ):
        super().__init__(
            project_root=project_root,
            subprocess_timeout_s=subprocess_timeout_s,
        )
        self.image_registry = image_registry
        self.pyxis_max_concurrent_creates = pyxis_max_concurrent_creates
        self.pyxis_max_concurrent_srun_steps = pyxis_max_concurrent_srun_steps
        self.pyxis_srun_launch_grace_s = pyxis_srun_launch_grace_s
        if pyxis_placement_file is None:
            if pyxis_shared_runtime_root is not None:
                raise ValueError(
                    "--pyxis-shared-runtime-root requires --pyxis-placement-file"
                )
            self._placement_file = None
            self._shared_runtime_root = None
        else:
            if pyxis_shared_runtime_root is None:
                raise ValueError(
                    "--pyxis-placement-file requires --pyxis-shared-runtime-root"
                )
            if not pyxis_shared_runtime_root.is_absolute():
                raise ValueError("--pyxis-shared-runtime-root must be absolute")
            self._placement_file = pyxis_placement_file.resolve()
            self._shared_runtime_root = pyxis_shared_runtime_root.resolve()
        self._placements_by_run: dict[str, PyxisPlacement] = {}
        self._placement_lock = threading.Lock()

    def run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        if self._placement_file is None:
            return super().run(request, run_dir, cancel_token)

        run_dir.mkdir(parents=True, exist_ok=True)
        resolved_run_dir = run_dir.resolve()
        assert self._shared_runtime_root is not None
        try:
            resolved_run_dir.relative_to(self._shared_runtime_root)
        except ValueError as exc:
            raise RunnerError(
                "Pyxis placement requires the service artifact root to be under "
                f"the shared runtime root {self._shared_runtime_root}"
            ) from exc

        placement = load_pyxis_placement(self._placement_file)
        snapshot_path = run_dir / "pyxis_placement.tsv"
        placement.write_snapshot(snapshot_path, request.evaluated_instance_ids)
        snapshot = load_pyxis_placement(snapshot_path)
        with self._placement_lock:
            self._placements_by_run[run_dir.name] = snapshot
        try:
            return super().run(request, run_dir, cancel_token)
        finally:
            with self._placement_lock:
                self._placements_by_run.pop(run_dir.name, None)

    def _placement_for_run(self, run_dir: Path) -> PyxisPlacement | None:
        with self._placement_lock:
            return self._placements_by_run.get(run_dir.name)

    def _configure_environment(
        self, environment_cfg: dict[str, Any], run_id: str
    ) -> None:
        # ``pull_timeout`` is the template's image-acquisition budget and is
        # exactly what the Pyxis container-create step needs, so it is carried
        # over rather than dropped. Without it the create step fell back to the
        # per-command ``timeout`` (300s in both templates) and every image
        # import slower than that was killed as an "infrastructure failure".
        # ``run_args``/``container_timeout`` stay dropped: both are docker-only.
        for key in ("run_args", "container_timeout"):
            environment_cfg.pop(key, None)
        environment_cfg["environment_class"] = (
            "swebench_service.pyxis_environment.PyxisEnvironment"
        )
        environment_cfg["run_id"] = run_id

    def _run_agent(
        self,
        request: RunRequest,
        patched_config: Path,
        output_dir: Path,
        run_dir: Path,
        secret_values: set[str],
        cancel_token: CancellationToken | None = None,
    ) -> None:
        placement = self._placement_for_run(run_dir)
        command = [
            sys.executable,
            "-m",
            "swebench_service.pyxis_worker",
            "agent",
            "--model",
            request.model_name,
            "--config",
            str(patched_config),
            "--subset",
            request.subset,
            "--split",
            request.split,
            "--filter",
            _exact_instance_filter(request.evaluated_instance_ids),
            "--workers",
            str(request.workers),
            "--output",
            str(output_dir),
            "--image-registry",
            self.image_registry,
            "--srun-launch-grace-s",
            str(self.pyxis_srun_launch_grace_s),
        ]
        if self.pyxis_max_concurrent_creates is not None:
            command.extend(
                [
                    "--max-concurrent-creates",
                    str(self.pyxis_max_concurrent_creates),
                ]
            )
        if self.pyxis_max_concurrent_srun_steps is not None:
            command.extend(
                [
                    "--max-concurrent-srun-steps",
                    str(self.pyxis_max_concurrent_srun_steps),
                ]
            )
        if placement is not None:
            assert self._shared_runtime_root is not None
            command.extend(
                [
                    "--placement-file",
                    str(run_dir / "pyxis_placement.tsv"),
                    "--shared-runtime-root",
                    str(self._shared_runtime_root),
                ]
            )
        try:
            self._run_logged_subprocess(
                command,
                run_dir / "swe_bench_agent.log",
                cwd=output_dir,
                timeout_s=self.subprocess_timeout_s,
                env=self._base_env(request),
                secret_values=secret_values,
                cancel_token=cancel_token,
            )
        except RunCancelled:
            raise
        except Exception as exc:
            agent_error_path = run_dir / "agent_phase_error.txt"
            atomic_write_bytes(
                agent_error_path,
                redact_text(str(exc), secret_values).encode(),
            )
            if not (output_dir / "preds.json").exists():
                raise
            logger.warning(
                "Pyxis agent phase failed after producing predictions; continuing "
                "with evaluation. Details are in %s",
                agent_error_path,
            )

    def _run_eval(
        self,
        request: RunRequest,
        preds_path: Path,
        output_dir: Path,
        run_dir: Path,
        secret_values: set[str],
        cancel_token: CancellationToken | None = None,
    ) -> Path:
        run_id, dataset_name = _prepare_eval(request, run_dir)
        placement = self._placement_for_run(run_dir)
        command = [
            sys.executable,
            "-m",
            "swebench_service.pyxis_worker",
            "eval",
            "--dataset-name",
            dataset_name,
            "--split",
            request.split,
            "--predictions-path",
            str(preds_path),
            "--max-workers",
            str(request.max_eval_workers),
            "--run-id",
            run_id,
            "--image-registry",
            self.image_registry,
            "--output-dir",
            str(output_dir),
            "--timeout",
            str(request.eval_timeout_s or 1800),
            "--srun-launch-grace-s",
            str(self.pyxis_srun_launch_grace_s),
        ]
        if self.pyxis_max_concurrent_creates is not None:
            command.extend(
                [
                    "--max-concurrent-creates",
                    str(self.pyxis_max_concurrent_creates),
                ]
            )
        if self.pyxis_max_concurrent_srun_steps is not None:
            command.extend(
                [
                    "--max-concurrent-srun-steps",
                    str(self.pyxis_max_concurrent_srun_steps),
                ]
            )
        if placement is not None:
            assert self._shared_runtime_root is not None
            command.extend(
                [
                    "--placement-file",
                    str(run_dir / "pyxis_placement.tsv"),
                    "--shared-runtime-root",
                    str(self._shared_runtime_root),
                ]
            )
        command.extend(["--instance-ids", *request.evaluated_instance_ids])
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)
        self._run_logged_subprocess(
            command,
            run_dir / "swe_bench_eval.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            env=env,
            secret_values=secret_values,
            cancel_token=cancel_token,
        )
        return _resolve_eval_result(output_dir, request.model_name, run_id)

    def _cleanup_containers(
        self,
        run_id: str,
        *,
        eval_run_id: str | None = None,
        instance_ids: list[str] | None = None,
    ) -> None:
        # Local import avoids the runner <-> Pyxis environment import cycle.
        from .pyxis_environment import build_srun_command, safe_srun_env

        del eval_run_id, instance_ids
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)[:24]
        container_name = re.compile(
            rf"^pyxis_(?:[0-9]+_)?mswe_{re.escape(safe_run_id)}_"
        )
        with self._placement_lock:
            placement = self._placements_by_run.get(run_id)
        nodes: tuple[str | None, ...] = placement.nodes if placement else (None,)
        try:
            for node in nodes:
                listed = subprocess.run(
                    build_srun_command(argv=["enroot", "list", "-f"], node=node),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=safe_srun_env(),
                )
                for line in listed.stdout.splitlines():
                    fields = line.split(maxsplit=1)
                    name = fields[0] if fields else ""
                    if container_name.match(name):
                        subprocess.run(
                            build_srun_command(
                                argv=["enroot", "remove", "-f", name], node=node
                            ),
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=30,
                            env=safe_srun_env(),
                        )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RunnerError(
                f"failed to clean up Pyxis containers for SWE-bench run {run_id}"
            ) from exc


def create_runner(
    runtime: Literal["docker", "pyxis"],
    *,
    project_root: Path,
    subprocess_timeout_s: int,
    image_registry: str | None,
    pyxis_placement_file: Path | None = None,
    pyxis_shared_runtime_root: Path | None = None,
    pyxis_max_concurrent_creates: int | None = None,
    pyxis_max_concurrent_srun_steps: int | None = None,
    pyxis_srun_launch_grace_s: int = 30,
) -> RunnerProtocol:
    if runtime == "docker":
        return SweBenchRunner(
            project_root=project_root,
            subprocess_timeout_s=subprocess_timeout_s,
        )
    if runtime == "pyxis":
        if image_registry is None:
            raise ValueError("Pyxis runtime requires an image registry")
        return PyxisSweBenchRunner(
            project_root=project_root,
            subprocess_timeout_s=subprocess_timeout_s,
            image_registry=image_registry,
            pyxis_placement_file=pyxis_placement_file,
            pyxis_shared_runtime_root=pyxis_shared_runtime_root,
            pyxis_max_concurrent_creates=pyxis_max_concurrent_creates,
            pyxis_max_concurrent_srun_steps=pyxis_max_concurrent_srun_steps,
            pyxis_srun_launch_grace_s=pyxis_srun_launch_grace_s,
        )
    raise ValueError(f"unknown SWE-bench runtime: {runtime}")

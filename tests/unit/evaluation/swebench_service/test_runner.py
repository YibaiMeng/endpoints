# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import stat
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Literal, get_type_hints

import msgspec.json
import pytest
import yaml
from inference_endpoint.evaluation.swebench_service.swebench_service import (
    pyxis_worker as worker_mod,
)
from inference_endpoint.evaluation.swebench_service.swebench_service import (
    pyxis_environment as pyxis_environment_mod,
    runner as runner_mod,
)
from inference_endpoint.evaluation.swebench_service.swebench_service.pyxis_environment import (
    PyxisEnvironment,
    build_srun_command,
    resolve_image,
    run_srun_step,
    safe_srun_env,
)
from inference_endpoint.evaluation.swebench_service.swebench_service.runner import (
    CancellationToken,
    EndpointOutageError,
    PyxisSweBenchRunner,
    RunCancelled,
    RunnerError,
    SweBenchRunner,
    create_runner,
)
from inference_endpoint.evaluation.swebench_service.swebench_service.schemas import (
    RunRequest,
)

pytestmark = pytest.mark.unit


def test_pyxis_implementation_is_confined_to_environment_and_worker_modules():
    package_dir = Path(runner_mod.__file__).parent

    assert {path.name for path in package_dir.glob("pyxis_*") if path.is_file()} == {
        "pyxis_environment.py",
        "pyxis_worker.py",
    }


def _request(endpoints: list[str]) -> RunRequest:
    return RunRequest(
        model_name="test-model",
        endpoint_urls=endpoints,
        endpoint_api_key=None,
        generation_params={"name": "test-model"},
        subset="lite",
        split="test",
        num_instances=1,
        workers=1,
        max_eval_workers=1,
        evaluated_instance_ids=["repo__repo-1"],
    )


def test_run_subprocess_streams_output_to_log(tmp_path):
    log_path = tmp_path / "subprocess.log"

    runner_mod._run_subprocess(
        [sys.executable, "-c", "print('first'); print('second')"],
        log_path,
        cwd=tmp_path,
        timeout_s=5,
    )

    assert log_path.read_text() == "first\nsecond\n"


def test_run_subprocess_reports_bounded_failure_tail(tmp_path):
    log_path = tmp_path / "subprocess.log"
    script = (
        "import sys\n"
        "print('early-marker')\n"
        "for i in range(700): print(f'{i:04d}-' + 'x' * 100)\n"
        "print('final-marker')\n"
        "sys.exit(7)\n"
    )

    with pytest.raises(RunnerError, match="exited with code 7") as exc_info:
        runner_mod._run_subprocess(
            [sys.executable, "-c", script],
            log_path,
            cwd=tmp_path,
            timeout_s=5,
        )

    assert "early-marker" in log_path.read_text()
    failure_tail = str(exc_info.value).partition("\n")[2]
    assert "final-marker" in failure_tail
    assert "early-marker" not in failure_tail


def test_run_subprocess_timeout_preserves_partial_log(monkeypatch, tmp_path):
    log_path = tmp_path / "subprocess.log"
    communicate_timeouts: list[float | None] = []
    original_communicate = subprocess.Popen.communicate

    def communicate_with_spy(process, *args, **kwargs):
        communicate_timeouts.append(kwargs.get("timeout"))
        return original_communicate(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", communicate_with_spy)

    with pytest.raises(RunnerError, match="timed out after 1s"):
        runner_mod._run_subprocess(
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            log_path,
            cwd=tmp_path,
            timeout_s=1,
        )

    assert log_path.read_text() == "started\n"
    assert communicate_timeouts
    assert all(timeout is not None for timeout in communicate_timeouts)


def test_run_subprocess_cancellation_preserves_partial_log(tmp_path):
    log_path = tmp_path / "subprocess.log"
    cancel_token = CancellationToken()
    cancel_timer = threading.Timer(0.2, cancel_token.cancel)
    cancel_timer.start()
    try:
        with pytest.raises(RunCancelled, match="subprocess cancelled"):
            runner_mod._run_subprocess(
                [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                log_path,
                cwd=tmp_path,
                timeout_s=5,
                cancel_token=cancel_token,
            )
    finally:
        cancel_timer.cancel()
        cancel_timer.join()

    assert log_path.read_text() == "started\n"


def test_base_env_keeps_proxies_and_sets_no_proxy_for_loopback(monkeypatch, tmp_path):
    monkeypatch.setenv("http_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("all_proxy", "socks5://proxy.example:1080")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.example:1080")
    monkeypatch.setenv("NO_PROXY", "intel.com")

    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    env = runner._base_env(_request(["http://localhost:30000"]))

    assert env["http_proxy"] == "http://proxy.example:8080"
    assert env["https_proxy"] == "http://proxy.example:8080"
    assert env["HTTP_PROXY"] == "http://proxy.example:8080"
    assert env["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert env["all_proxy"] == "socks5://proxy.example:1080"
    assert env["ALL_PROXY"] == "socks5://proxy.example:1080"
    assert {"127.0.0.1", "localhost", "intel.com"} <= set(env["NO_PROXY"].split(","))
    assert env["NO_PROXY"] == env["no_proxy"]


def test_base_env_keeps_proxies_for_non_loopback_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("https_proxy", "http://proxy.example:8080")

    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    env = runner._base_env(_request(["http://swebench-host:30000"]))

    assert env["https_proxy"] == "http://proxy.example:8080"
    assert "swebench-host" in env["NO_PROXY"].split(",")


@pytest.mark.parametrize(
    ("endpoint", "expected_api_base"),
    [
        ("http://localhost:30000", "http://127.0.0.1:30000/v1"),
        (
            "https://user:pass@endpoint.example:8443/proxy/v1?token=secret#fragment",
            "https://endpoint.example:8443/proxy/v1",
        ),
    ],
)
def test_patch_config_normalizes_api_base(tmp_path, endpoint, expected_api_base):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    patched = runner._patch_config(
        tmp_path,
        _request([endpoint]),
        run_id="run-123",
    )

    text = patched.read_text()
    cfg = yaml.safe_load(text)
    assert cfg["model"]["model_kwargs"]["api_base"] == expected_api_base
    assert "user:pass" not in text
    assert "token=secret" not in text
    assert "fragment" not in text
    assert "model_class" not in cfg["model"]
    assert "api_key" not in cfg["model"]["model_kwargs"]
    assert cfg["environment"]["run_args"] == [
        "--rm",
        "--label",
        "com.mlcommons.endpoints.swebench-run=run-123",
    ]


def test_patch_config_keeps_api_key_out_of_yaml_and_forwards_generation(tmp_path):
    request = _request(["http://endpoint:30000"])
    request.endpoint_api_key = "real-secret"
    request.generation_params = {
        "temperature": 0.2,
        "seed": 23,
        "max_new_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    patched = runner._patch_config(tmp_path, request, run_id="run-1")

    text = patched.read_text()
    cfg = yaml.safe_load(text)
    model_kwargs = cfg["model"]["model_kwargs"]
    assert "real-secret" not in text
    assert "api_key" not in model_kwargs
    assert model_kwargs["temperature"] == 0.2
    assert model_kwargs["seed"] == 23
    assert model_kwargs["max_tokens"] == 2048
    assert model_kwargs["chat_template_kwargs"] == {"enable_thinking": False}


def test_base_env_supplies_api_key_only_to_agent_subprocess(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    authenticated = _request(["http://endpoint:30000"])
    authenticated.endpoint_api_key = "real-secret"
    loopback = _request(["http://localhost:30000"])

    assert runner._base_env(loopback)["OPENAI_API_KEY"] == "EMPTY"

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    unauthenticated = _request(["http://endpoint:30000"])
    assert "OPENAI_API_KEY" not in runner._base_env(unauthenticated)


def test_run_agent_filters_exact_instance_ids(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []

    def fake_run_subprocess(cmd, *args, **kwargs):
        commands.append(cmd)
        envs.append(kwargs["env"])

    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run_subprocess)
    request = _request(["http://endpoint:30000"])
    request.endpoint_api_key = "agent-secret"
    request.evaluated_instance_ids = ["repo__repo-1", "repo.with.regex+chars"]
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    runner._run_agent(
        request, tmp_path / "config.yaml", tmp_path, tmp_path, {"agent-secret"}
    )

    cmd = commands[0]
    assert "--slice" not in cmd
    assert cmd[cmd.index("--filter") + 1] == (
        "^(?:repo__repo\\-1|repo\\.with\\.regex\\+chars)$"
    )
    assert envs[0]["OPENAI_API_KEY"] == "agent-secret"


def test_qwen_template_selects_model_without_mutating_pythonpath(monkeypatch, tmp_path):
    envs: list[dict[str, str]] = []
    request = _request(["http://endpoint:30000"])
    request.template = "qwen_tools"
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    def fake_run_subprocess(cmd, log_path, *, env, **kwargs):
        envs.append(env)

    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run_subprocess)

    patched = runner._patch_config(tmp_path, request, run_id="run-qwen")
    cfg = yaml.safe_load(patched.read_text())
    runner._run_agent(request, patched, tmp_path, tmp_path, set())

    assert cfg["model"]["model_class"] == (
        "swebench_service.qwen_tools_model.QwenToolsModel"
    )
    assert envs[0]["PYTHONPATH"] == "/existing/path"


def test_validate_prediction_ids_rejects_unexpected_instances(tmp_path):
    request = _request(["http://endpoint:30000"])
    request.evaluated_instance_ids = ["repo__repo-1"]
    preds = tmp_path / "preds.json"
    preds.write_bytes(
        msgspec.json.encode({"repo__repo-1": "patch", "repo__repo-2": "patch"})
    )
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    with pytest.raises(RunnerError, match="unexpected SWE-bench"):
        runner._validate_prediction_ids(request, preds)


def test_validate_prediction_ids_logs_missing_instances(tmp_path, caplog):
    request = _request(["http://endpoint:30000"])
    request.evaluated_instance_ids = ["repo__repo-1", "repo__repo-2"]
    preds = tmp_path / "preds.json"
    preds.write_bytes(msgspec.json.encode({"repo__repo-1": "patch"}))
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    runner._validate_prediction_ids(request, preds)

    assert "omitted predictions" in caplog.text
    assert "repo__repo-2" in caplog.text


@pytest.mark.parametrize("agent_raises", [False, True])
def test_run_classifies_terminal_service_unavailable_status_before_prediction_validation(
    monkeypatch, tmp_path, agent_raises
):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    def fake_run_agent(
        request, patched_config, output_dir, run_dir, secret_values, cancel_token=None
    ):
        (output_dir / "exit_statuses_0001.yaml").write_text(
            "instances_by_exit_status:\n"
            "  ServiceUnavailableError:\n"
            "    - repo__repo-1\n"
        )
        if agent_raises:
            raise RunnerError("agent exited unsuccessfully")

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "_cleanup_containers", lambda *args, **kwargs: None)

    with pytest.raises(EndpointOutageError, match="endpoint outage"):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run")


def test_run_ignores_transient_agent_log_outages_when_final_status_is_submitted(
    monkeypatch, tmp_path
):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    def fake_run_agent(
        request, patched_config, output_dir, run_dir, secret_values, cancel_token=None
    ):
        (run_dir / "swe_bench_agent.log").write_text(
            "ServiceUnavailableError: Error code: 503\n"
            "ServiceUnavailableError: no_available_workers\n"
            "All workers are unavailable (circuit breaker open or unhealthy)\n"
        )
        (output_dir / "exit_statuses_0001.yaml").write_text(
            "instances_by_exit_status:\n"
            "  Submitted:\n"
            "    - repo__repo-1\n"
        )
        (output_dir / "preds.json").write_text('{"repo__repo-1":"patch"}')

    def fake_run_eval(
        request, preds_path, output_dir, run_dir, secret_values, cancel_token=None
    ):
        result_path = output_dir / "result.json"
        result_path.write_text('{"resolved_instances":1,"submitted_instances":1}')
        return result_path

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "_run_eval", fake_run_eval)
    monkeypatch.setattr(runner, "_cleanup_containers", lambda *args, **kwargs: None)

    result = runner.run(_request(["http://endpoint:30000"]), tmp_path / "run")

    assert result == {"resolved_instances": 1, "submitted_instances": 1}


def test_run_ignores_stale_service_unavailable_exit_status(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    def fake_run_agent(
        request, patched_config, output_dir, run_dir, secret_values, cancel_token=None
    ):
        stale_path = output_dir / "exit_statuses_0001.yaml"
        stale_path.write_text(
            "instances_by_exit_status:\n"
            "  ServiceUnavailableError:\n"
            "    - repo__repo-1\n"
        )
        final_path = output_dir / "exit_statuses_0002.yaml"
        final_path.write_text(
            "instances_by_exit_status:\n"
            "  Submitted:\n"
            "    - repo__repo-1\n"
        )
        os.utime(stale_path, (1, 1))
        os.utime(final_path, (2, 2))
        (output_dir / "preds.json").write_text('{"repo__repo-1":"patch"}')

    def fake_run_eval(
        request, preds_path, output_dir, run_dir, secret_values, cancel_token=None
    ):
        result_path = output_dir / "result.json"
        result_path.write_text('{"resolved_instances":1,"submitted_instances":1}')
        return result_path

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "_run_eval", fake_run_eval)
    monkeypatch.setattr(runner, "_cleanup_containers", lambda *args, **kwargs: None)

    result = runner.run(_request(["http://endpoint:30000"]), tmp_path / "run")

    assert result == {"resolved_instances": 1, "submitted_instances": 1}


def test_run_classifies_mixed_terminal_service_unavailable_statuses(
    monkeypatch, tmp_path
):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    def fake_run_agent(
        request, patched_config, output_dir, run_dir, secret_values, cancel_token=None
    ):
        (output_dir / "exit_statuses_0001.yaml").write_text(
            "instances_by_exit_status:\n"
            "  Submitted:\n"
            "    - repo__repo-1\n"
            "  ServiceUnavailableError:\n"
            "    - repo__repo-2\n"
        )

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "_cleanup_containers", lambda *args, **kwargs: None)

    with pytest.raises(EndpointOutageError, match="endpoint outage"):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run")


def test_run_preserves_agent_error_without_terminal_outage_status(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    def fake_run_agent(*args, **kwargs):
        raise RunnerError("agent failed after transient ServiceUnavailableError")

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "_cleanup_containers", lambda *args, **kwargs: None)

    with pytest.raises(RunnerError, match="agent failed"):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run")


def test_logged_subprocess_publishes_redacted_log(tmp_path):
    public_log = tmp_path / "agent.log"

    SweBenchRunner._run_logged_subprocess(
        [sys.executable, "-c", "print('Authorization: Bearer abc')"],
        public_log,
        cwd=tmp_path,
        timeout_s=5,
        env={},
        secret_values={"abc"},
        cancel_token=None,
    )

    assert "abc" not in public_log.read_text()
    assert "<redacted>" in public_log.read_text()
    assert not (tmp_path / ".agent.log.raw").exists()


@pytest.mark.parametrize("ambiguous_fallback", [False, True])
def test_run_eval_persists_harness_run_id(monkeypatch, tmp_path, ambiguous_fallback):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    request = _request(["http://endpoint:30000"])
    output_dir = tmp_path / "output"
    run_dir = tmp_path / "run"
    output_dir.mkdir()
    run_dir.mkdir()
    preds_path = output_dir / "preds.json"
    preds_path.write_text('{"repo__repo-1":"patch"}')

    def fake_run_subprocess(cmd, log_path, *, cwd, env, **kwargs):
        assert cmd[:3] == [sys.executable, "-m", "swebench.harness.run_evaluation"]
        assert "OPENAI_API_KEY" not in env
        run_id = cmd[cmd.index("--run_id") + 1]
        assert (run_dir / "swe_bench_eval_run_id.txt").read_text() == run_id
        if ambiguous_fallback:
            for directory in ("first", "second"):
                candidate_dir = cwd / directory
                candidate_dir.mkdir()
                (candidate_dir / f"{directory}.{run_id}.json").write_text("{}")
        else:
            (cwd / f"test-model.{run_id}.json").write_text(
                '{"resolved_instances":1,"submitted_instances":1}'
            )

    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run_subprocess)

    if ambiguous_fallback:
        with pytest.raises(RunnerError, match="multiple SWE-bench result files"):
            runner._run_eval(request, preds_path, output_dir, run_dir, set())
    else:
        result_path = runner._run_eval(request, preds_path, output_dir, run_dir, set())
        assert result_path.exists()

    assert (run_dir / "swe_bench_eval_run_id.txt").read_text().startswith("endpoints_")


def _stub_successful_run(monkeypatch, runner: SweBenchRunner) -> None:
    def fake_run_agent(
        request, patched_config, output_dir, run_dir, secret_values, cancel_token=None
    ):
        (output_dir / "preds.json").write_text('{"repo__repo-1":"patch"}')

    def fake_run_eval(
        request, preds_path, output_dir, run_dir, secret_values, cancel_token=None
    ):
        result_path = output_dir / "result.json"
        result_path.write_text('{"resolved_instances":1,"submitted_instances":1}')
        return result_path

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "_run_eval", fake_run_eval)


def test_run_cleans_labeled_containers_after_success(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    _stub_successful_run(monkeypatch, runner)
    cleaned: list[str] = []
    monkeypatch.setattr(runner, "_cleanup_containers", cleaned.append)

    result = runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-1")

    assert result == {"resolved_instances": 1, "submitted_instances": 1}
    assert cleaned == ["run-1"]


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (RuntimeError("agent failed"), "agent failed"),
        (RunnerError("subprocess timed out"), "timed out"),
        (RunCancelled("subprocess cancelled"), "cancelled"),
    ],
)
def test_run_cleans_labeled_containers_after_failure(
    monkeypatch, tmp_path, error, match
):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    cleaned: list[str] = []

    def fail_agent(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner, "_run_agent", fail_agent)
    monkeypatch.setattr(runner, "_cleanup_containers", cleaned.append)

    with pytest.raises(type(error), match=match):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-2")

    assert cleaned == ["run-2"]


def test_run_cleans_harness_containers_after_eval_cancellation(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    cleaned: list[tuple[str, dict]] = []

    def fake_run_agent(
        request, patched_config, output_dir, run_dir, secret_values, cancel_token=None
    ):
        (output_dir / "preds.json").write_text('{"repo__repo-1":"patch"}')

    def cancel_eval(
        request, preds_path, output_dir, run_dir, secret_values, cancel_token=None
    ):
        (run_dir / "swe_bench_eval_run_id.txt").write_text("endpoints_cancelled")
        raise RunCancelled("subprocess cancelled")

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "_run_eval", cancel_eval)
    monkeypatch.setattr(
        runner,
        "_cleanup_containers",
        lambda run_id, **kwargs: cleaned.append((run_id, kwargs)),
    )

    with pytest.raises(RunCancelled, match="cancelled"):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-cancelled")

    assert cleaned == [
        (
            "run-cancelled",
            {
                "eval_run_id": "endpoints_cancelled",
                "instance_ids": ["repo__repo-1"],
            },
        )
    ]


def test_cleanup_failure_does_not_fail_successful_run(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    _stub_successful_run(monkeypatch, runner)
    monkeypatch.setattr(
        runner,
        "_cleanup_containers",
        lambda run_id: (_ for _ in ()).throw(RunnerError("cleanup failed")),
    )

    result = runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-3")

    assert result == {"resolved_instances": 1, "submitted_instances": 1}


def test_pyxis_cleanup_failure_fails_successful_run(monkeypatch, tmp_path):
    runner = PyxisSweBenchRunner(
        project_root=tmp_path,
        subprocess_timeout_s=30,
        image_registry=_PYXIS_IMAGE_REGISTRY,
    )
    _stub_successful_run(monkeypatch, runner)
    monkeypatch.setattr(
        runner,
        "_cleanup_containers",
        lambda run_id: (_ for _ in ()).throw(
            RunnerError(
                "failed to clean up Pyxis containers for SWE-bench run run-3"
            )
        ),
    )

    with pytest.raises(RunnerError, match="failed to clean up Pyxis containers"):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-3")


def test_cleanup_failure_does_not_mask_primary_failure(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    monkeypatch.setattr(
        runner,
        "_run_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(RunCancelled("cancelled")),
    )
    monkeypatch.setattr(
        runner,
        "_cleanup_containers",
        lambda run_id: (_ for _ in ()).throw(RunnerError("cleanup failed")),
    )

    with pytest.raises(RunCancelled, match="cancelled"):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-4")


def test_cleanup_uses_exact_run_label_and_leaves_unrelated_containers(
    monkeypatch, tmp_path
):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        stdout = "matched-1\nmatched-2\n" if cmd[1:3] == ["ps", "-aq"] else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    runner._cleanup_containers("run-exact")

    assert calls == [
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.mlcommons.endpoints.swebench-run=run-exact",
        ],
        ["docker", "rm", "-f", "matched-1", "matched-2"],
    ]


def test_cleanup_exactly_matches_harness_container_names(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["ps", "-aq"]:
            stdout = "agent-container\n"
        elif cmd[1:3] == ["ps", "-a"]:
            stdout = (
                "eval-container\tsweb.eval.repo__repo-1.endpoints_eval\n"
                "other-instance\tsweb.eval.repo__repo-2.endpoints_eval\n"
                "other-run\tsweb.eval.repo__repo-1.endpoints_other\n"
                "unrelated\tunrelated.endpoints_eval\n"
            )
        else:
            stdout = ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    runner._cleanup_containers(
        "run-exact",
        eval_run_id="endpoints_eval",
        instance_ids=["Repo__Repo-1"],
    )

    assert calls == [
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.mlcommons.endpoints.swebench-run=run-exact",
        ],
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=endpoints_eval",
            "--format",
            "{{.ID}}\t{{.Names}}",
        ],
        ["docker", "rm", "-f", "agent-container", "eval-container"],
    ]


def _pyxis_request(instance_ids: list[str] | None = None) -> RunRequest:
    instance_ids = instance_ids or ["repo__repo-1"]
    return RunRequest(
        model_name="test-model",
        endpoint_urls=["http://endpoint:30000"],
        generation_params={"temperature": 0.2},
        subset="lite",
        split="test",
        num_instances=len(instance_ids),
        workers=2,
        max_eval_workers=2,
        evaluated_instance_ids=instance_ids,
    )


_PYXIS_IMAGE_REGISTRY = "gitlab-master.nvidia.com:5005/hvagadia/swebench-arm64-images"


def _srun_status_mount(command: list[str]) -> tuple[Path, str] | None:
    mount_argument = next(
        (argument for argument in command if argument.startswith("--container-mounts=")),
        None,
    )
    if mount_argument is None:
        return None
    for mount in mount_argument.removeprefix("--container-mounts=").split(","):
        source, destination = mount.split(":", 1)
        if destination.startswith("/tmp/.mlperf_srun_status."):
            return Path(source), destination
    return None


def _required_srun_status_mount(command: list[str]) -> tuple[Path, str]:
    status_mount = _srun_status_mount(command)
    assert status_mount is not None, "srun command does not mount its status file"
    return status_mount


def _finish_srun_step(command: list[str], returncode: int) -> None:
    status_mount = _srun_status_mount(command)
    if status_mount is None:
        return
    status_mount[0].write_text(f"finished:{returncode}\n")


def _install_fake_minisweagent(monkeypatch, swebench) -> None:
    minisweagent = types.ModuleType("minisweagent")
    environments = types.ModuleType("minisweagent.environments")
    environments.get_environment = lambda config: config  # type: ignore[attr-defined]
    run = types.ModuleType("minisweagent.run")
    benchmarks = types.ModuleType("minisweagent.run.benchmarks")
    benchmarks.swebench = swebench  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "minisweagent", minisweagent)
    monkeypatch.setitem(sys.modules, "minisweagent.environments", environments)
    monkeypatch.setitem(sys.modules, "minisweagent.run", run)
    monkeypatch.setitem(sys.modules, "minisweagent.run.benchmarks", benchmarks)


def _pyxis_agent_args(tmp_path) -> list[str]:
    return [
        "agent",
        "--model",
        "test-model",
        "--config",
        str(tmp_path / "config.yaml"),
        "--subset",
        "verified",
        "--split",
        "test",
        "--filter",
        "repo__repo-1",
        "--workers",
        "1",
        "--output",
        str(tmp_path),
        "--image-registry",
        _PYXIS_IMAGE_REGISTRY,
    ]


def test_create_runner_defaults_to_docker(tmp_path):
    runner = create_runner(
        "docker",
        project_root=tmp_path,
        subprocess_timeout_s=30,
        image_registry=None,
    )

    assert type(runner) is SweBenchRunner


def test_create_runner_runtime_is_typed():
    assert get_type_hints(create_runner)["runtime"] == Literal["docker", "pyxis"]


def test_create_runner_requires_image_registry_for_pyxis(tmp_path):
    with pytest.raises(ValueError, match="image registry"):
        create_runner(
            "pyxis",
            project_root=tmp_path,
            subprocess_timeout_s=30,
            image_registry=None,
        )


def test_create_runner_selects_pyxis(tmp_path):
    runner = create_runner(
        "pyxis",
        project_root=tmp_path,
        subprocess_timeout_s=30,
        image_registry=_PYXIS_IMAGE_REGISTRY,
    )

    assert isinstance(runner, PyxisSweBenchRunner)


def test_pyxis_patch_config_selects_pyxis_environment(tmp_path):
    runner = PyxisSweBenchRunner(
        project_root=tmp_path,
        subprocess_timeout_s=30,
        image_registry=_PYXIS_IMAGE_REGISTRY,
    )

    patched = runner._patch_config(tmp_path, _pyxis_request(), run_id="run-1")

    environment = yaml.safe_load(patched.read_text())["environment"]
    assert environment["environment_class"] == (
        "swebench_service.pyxis_environment.PyxisEnvironment"
    )
    assert environment["cwd"] == "/testbed"
    assert environment["run_id"] == "run-1"
    assert "run_args" not in environment
    assert "container_timeout" not in environment
    # Carried over, not dropped: the Pyxis create step *is* the image pull.
    assert environment["pull_timeout"] == 3600


def test_pyxis_qwen_command_preserves_inner_deadline_with_outer_grace(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    qwen_template = yaml.safe_load(
        (
            Path(runner_mod.__file__).parent
            / "templates"
            / "swebench_qwen_tools_template.yaml"
        ).read_text()
    )
    command_timeout_s = qwen_template["environment"]["timeout"]
    environment = PyxisEnvironment(
        image=tmp_path / "task.sqsh",
        run_id="run-1",
        timeout=command_timeout_s,
    )

    environment.execute({"command": "pytest -q"})
    environment.cleanup()

    assert command_timeout_s == 300
    assert pyxis_environment_mod.SRUN_TEARDOWN_GRACE_S == 300
    assert pyxis_environment_mod.SRUN_PYXIS_WORKLOAD_TEARDOWN_GRACE_S == 900
    assert pyxis_environment_mod.PYXIS_CLEANUP_TIMEOUT_S == 300
    assert 0 < calls[1][1]["timeout"] <= 1200
    assert "300" in calls[1][0]
    assert calls[-1][1]["timeout"] == pyxis_environment_mod.PYXIS_CLEANUP_TIMEOUT_S


@pytest.mark.parametrize(
    ("mode", "uses_workload_grace"),
    [
        ("image", True),
        ("name", True),
        ("image_and_name", False),
        ("host", False),
    ],
)
def test_pyxis_outer_grace_is_scoped_to_workload_steps(
    monkeypatch, tmp_path, mode, uses_workload_grace
):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    monkeypatch.setattr(pyxis_environment_mod.time, "monotonic", lambda: 100.0)
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if mode != "host":
            _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_kwargs = {
        "image": {"image": tmp_path / "task.sqsh"},
        "name": {"name": "existing-container"},
        "image_and_name": {
            "image": tmp_path / "task.sqsh",
            "name": "new-container",
        },
        "host": {},
    }[mode]

    if mode == "host":
        with pytest.raises(RunnerError, match="before the command completed"):
            run_srun_step(
                argv=["true"],
                status_path=Path(pyxis_environment_mod._STEP_STATUS),
                timeout_s=30,
                **run_kwargs,
            )
    else:
        assert (
            run_srun_step(
                argv=["true"],
                status_path=Path(pyxis_environment_mod._STEP_STATUS),
                timeout_s=30,
                **run_kwargs,
            ).returncode
            == 0
        )

    grace_s = (
        pyxis_environment_mod.SRUN_PYXIS_WORKLOAD_TEARDOWN_GRACE_S
        if uses_workload_grace
        else pyxis_environment_mod.SRUN_TEARDOWN_GRACE_S
    )
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 30 + grace_s
    assert "30" in calls[0][0]


def test_pyxis_resolves_registry_image_from_instance_id():
    assert resolve_image(_PYXIS_IMAGE_REGISTRY, "repo__repo-1") == (
        "gitlab-master.nvidia.com:5005#hvagadia/swebench-arm64-images/"
        "sweb.eval.arm64.repo__repo-1:v4.1.0-arm64"
    )
    with pytest.raises(RunnerError, match="invalid SWE-bench instance ID"):
        resolve_image(_PYXIS_IMAGE_REGISTRY, "../repo__repo-1")


def test_pyxis_normalizes_instance_id_for_registry_image():
    assert resolve_image(_PYXIS_IMAGE_REGISTRY, "Repo__Repo-1").endswith(
        "/sweb.eval.arm64.repo__repo-1:v4.1.0-arm64"
    )


def test_pyxis_resolves_file_image_from_instance_id():
    assert resolve_image("file:///shared/swebench-images", "Repo__Repo-1") == (
        "/shared/swebench-images/sweb.eval.arm64.repo__repo-1.sqsh"
    )


def test_pyxis_image_registry_requires_repository():
    with pytest.raises(RunnerError, match="must include a repository"):
        resolve_image("registry.example.com", "repo__repo-1")


def test_pyxis_builds_one_node_srun_command(monkeypatch, tmp_path):
    image = resolve_image(_PYXIS_IMAGE_REGISTRY, "repo__repo-1")
    source = tmp_path / "input.json"
    source.touch()
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")

    command = build_srun_command(
        image=image,
        name="minisweagent-run-1",
        mounts=[(source, "/swebench/input.json")],
        workdir="/testbed",
        argv=["python", "worker.py"],
    )

    assert command == [
        "srun",
        "--overlap",
        "--jobid=1738605",
        "-N1",
        "-n1",
        "--nodelist=gb-nvl-053-compute04",
        f"--container-image={image}",
        "--container-name=minisweagent-run-1",
        "--container-writable",
        "--container-remap-root",
        "--no-container-mount-home",
        f"--container-mounts={source.resolve()}:/swebench/input.json",
        "--container-workdir=/testbed",
        "python",
        "worker.py",
    ]


def test_pyxis_rejects_commas_in_mount_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")

    with pytest.raises(RunnerError, match="mount paths cannot contain commas"):
        build_srun_command(
            image="registry.example.com#images/task:latest",
            mounts=[(tmp_path / "input,with-comma.json", "/tmp/input.json")],
            argv=["true"],
        )


def test_pyxis_srun_requires_allocation(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(RunnerError, match="SLURM_JOB_ID"):
        build_srun_command(
            image=resolve_image(_PYXIS_IMAGE_REGISTRY, "repo__repo-1"),
            name=None,
            mounts=[],
            workdir="/testbed",
            argv=["true"],
        )


def test_pyxis_srun_requires_current_node(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)

    with pytest.raises(RunnerError, match="SLURMD_NODENAME"):
        build_srun_command(argv=["true"])


def test_pyxis_builds_host_srun_command(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")

    command = build_srun_command(argv=["enroot", "list", "-f"])

    assert command == [
        "srun",
        "--overlap",
        "--jobid=1738605",
        "-N1",
        "-n1",
        "--nodelist=gb-nvl-053-compute04",
        "enroot",
        "list",
        "-f",
    ]


def test_pyxis_srun_environment_does_not_forward_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret")
    monkeypatch.setenv("SWEBENCH_SERVICE_AUTH_TOKEN", "service-secret")
    monkeypatch.setenv("HF_TOKEN", "dataset-secret")

    environment = safe_srun_env()

    assert "OPENAI_API_KEY" not in environment
    assert "SWEBENCH_SERVICE_AUTH_TOKEN" not in environment
    assert "HF_TOKEN" not in environment


def test_pyxis_environment_reuses_named_writable_container(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret")
    image = tmp_path / "task.sqsh"
    image.touch()
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(
        image=image,
        run_id="run-1",
        cwd="/testbed",
        env={"PAGER": "cat"},
        timeout=30,
        interpreter=["bash", "-c"],
    )
    assert environment.config.timeout_s == 30
    assert environment.serialize()["info"]["config"]["environment"]["timeout"] == 30

    with caplog.at_level(
        logging.DEBUG,
        logger="inference_endpoint.evaluation.swebench_service.swebench_service.pyxis_environment",
    ):
        first = environment.execute({"command": "touch state"})
    second = environment.execute({"command": "test -f state"})
    environment.cleanup()

    container_name = next(
        argument.split("=", 1)[1]
        for argument in calls[0][0]
        if argument.startswith("--container-name=")
    )
    assert f"--container-image={image.resolve()}" in calls[0][0]
    for command, kwargs in calls[1:3]:
        assert f"--container-name={container_name}" in command
        assert not any(arg.startswith("--container-image=") for arg in command)
        assert "--no-container-mount-home" in command
        assert kwargs["env"].get("OPENAI_API_KEY") is None
    assert calls[1][0][-5:] == ["env", "PAGER=cat", "bash", "-c", "touch state"]
    assert any(
        "unshare --pid --fork --mount-proc" in argument for argument in calls[1][0]
    )
    assert calls[-1][0][-4:] == [
        "enroot",
        "remove",
        "-f",
        f"pyxis_{container_name}",
    ]
    assert first["returncode"] == second["returncode"] == 0
    assert "Executing Pyxis command: touch state" in caplog.text


def test_pyxis_environment_mounts_persistent_tmp_on_every_step(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    image = tmp_path / "task.sqsh"
    image.touch()
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(image=image, run_id="run-1")
    environment.execute({"command": "touch /tmp/state"})
    environment.execute({"command": "test -f /tmp/state"})

    mount_specs = [
        next(arg for arg in command if arg.startswith("--container-mounts="))
        .removeprefix("--container-mounts=")
        .split(",")
        for command in calls[:3]
    ]
    tmp_mounts = [
        next(spec for spec in specs if spec.endswith(":/tmp"))
        for specs in mount_specs
    ]
    assert tmp_mounts[0] == tmp_mounts[1] == tmp_mounts[2]
    source, destination = tmp_mounts[0].split(":", 1)
    assert destination == "/tmp"
    persistent_tmp = Path(source)
    assert persistent_tmp.is_dir()
    assert stat.S_IMODE(persistent_tmp.stat().st_mode) == 0o1777
    status_mounts = [
        next(spec for spec in specs if ":/tmp/.mlperf_srun_status." in spec)
        for specs in mount_specs
    ]
    assert len(set(status_mounts)) == 3
    assert all(not spec.startswith(f"{persistent_tmp}/") for spec in status_mounts)

    environment.cleanup()

    assert not persistent_tmp.exists()


def test_pyxis_environment_extracts_submission(monkeypatch, tmp_path):
    class Submitted(Exception):
        pass

    minisweagent = types.ModuleType("minisweagent")
    exceptions = types.ModuleType("minisweagent.exceptions")
    exceptions.Submitted = Submitted
    monkeypatch.setitem(sys.modules, "minisweagent", minisweagent)
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions)
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        output = (
            "ok\n"
            if calls == 1
            else "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/a b/a\n"
        )
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(image=tmp_path / "task.sqsh", run_id="run-1")

    with pytest.raises(Submitted) as exc_info:
        environment.execute({"command": "submit"})

    assert exc_info.value.args[0]["extra"]["submission"] == "diff --git a/a b/a\n"
    environment.cleanup()


def test_pyxis_environment_extracts_submission_after_srun_requeue_preamble(
    monkeypatch, tmp_path
):
    class Submitted(Exception):
        pass

    minisweagent = types.ModuleType("minisweagent")
    exceptions = types.ModuleType("minisweagent.exceptions")
    exceptions.Submitted = Submitted
    monkeypatch.setitem(sys.modules, "minisweagent", minisweagent)
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions)
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        output = (
            "ok\n"
            if calls == 1
            else "srun: lua: Checking requeue policy with options:\nordinary output\n"
            if calls == 2
            else (
                "srun: lua: Checking requeue policy with options:\n"
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
                "diff --git a/a b/a\n"
            )
        )
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(image=tmp_path / "task.sqsh", run_id="run-1")

    output = environment.execute({"command": "normal"})

    assert output["output"] == (
        "srun: lua: Checking requeue policy with options:\nordinary output\n"
    )

    with pytest.raises(Submitted) as exc_info:
        environment.execute({"command": "submit"})

    assert exc_info.value.args[0]["extra"]["submission"] == "diff --git a/a b/a\n"
    environment.cleanup()


def test_pyxis_environment_decodes_timeout_output(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            _finish_srun_step(command, 124)
            return subprocess.CompletedProcess(
                command, 124, stdout="partial�", stderr=""
            )
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(image=tmp_path / "task.sqsh", run_id="run-1")

    output = environment.execute({"command": "sleep 60"})

    assert output["returncode"] == -1
    assert output["output"] == "partial�"
    assert output["extra"]["exception_type"] == "TimeoutExpired"
    environment.cleanup()


def test_pyxis_environment_raises_when_srun_never_starts_command(monkeypatch, tmp_path):
    failure_path = tmp_path / ".pyxis_infrastructure_failure"
    environment = object.__new__(PyxisEnvironment)
    environment.config = types.SimpleNamespace(
        cwd="/testbed",
        env={},
        timeout_s=30,
        interpreter=["bash", "-c"],
        infrastructure_failure_path=failure_path,
    )
    environment.name = "mswe_run-1_abcd1234"
    environment._tmp_dir = tmp_path

    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(
        RunnerError,
        match=rf"exceeded its "
        rf"{30 + pyxis_environment_mod.SRUN_PYXIS_WORKLOAD_TEARDOWN_GRACE_S}s deadline",
    ):
        environment.execute({"command": "pytest -q"})

    assert failure_path.exists()
    assert calls == 1


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "mode"),
    [
        (167, "Job credential expired", "", "existing_container"),
        (139, "", "Header lengths are longer than data received", "evaluator"),
    ],
)
def test_pyxis_retries_exact_pending_transient_srun_launch_failure(
    monkeypatch, tmp_path, returncode, stdout, stderr, mode
):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            _finish_srun_step(command, 0)
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        assert _required_srun_status_mount(command)[0].read_text() == "pending\n"
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_kwargs = (
        {"name": "existing-container"}
        if mode == "existing_container"
        else {"image": tmp_path / "task.sqsh"}
    )

    result = run_srun_step(
        argv=["true"],
        status_path=Path(pyxis_environment_mod._STEP_STATUS),
        timeout_s=30,
        stderr=subprocess.PIPE,
        **run_kwargs,
    )

    assert result.returncode == 0
    assert len(calls) == 2
    first_mount = _required_srun_status_mount(calls[0])
    second_mount = _required_srun_status_mount(calls[1])
    assert first_mount[0] != second_mount[0]
    assert first_mount[1] != second_mount[1]


def test_pyxis_marks_retryable_srun_failure_only_after_retry_exhaustion(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    failure_path = tmp_path / ".pyxis_infrastructure_failure"
    marker_existed_before_second_attempt: list[bool] = []
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            marker_existed_before_second_attempt.append(failure_path.exists())
        return subprocess.CompletedProcess(
            command, 167, stdout="Job credential expired", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RunnerError, match="Job credential expired"):
        run_srun_step(
            argv=["true"],
            status_path=Path(pyxis_environment_mod._STEP_STATUS),
            timeout_s=30,
            failure_path=failure_path,
            image=tmp_path / "task.sqsh",
        )

    assert calls == 2
    assert marker_existed_before_second_attempt == [False]
    assert failure_path.exists()


def test_pyxis_retry_uses_only_the_remaining_original_deadline(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    monotonic_times = iter((100.0, 100.0, 130.0))
    monkeypatch.setattr(pyxis_environment_mod.time, "monotonic", lambda: next(monotonic_times))
    timeouts: list[float] = []

    def fake_run(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        if len(timeouts) == 2:
            _finish_srun_step(command, 0)
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        return subprocess.CompletedProcess(
            command, 167, stdout="Job credential expired", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        run_srun_step(
            argv=["true"],
            status_path=Path(pyxis_environment_mod._STEP_STATUS),
            timeout_s=30,
            image=tmp_path / "task.sqsh",
        ).returncode
        == 0
    )
    assert timeouts == [930.0, 900.0]


def test_pyxis_skips_retry_when_the_original_deadline_is_exhausted(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    monotonic_times = iter((0.0, 0.0, 930.0))
    monkeypatch.setattr(pyxis_environment_mod.time, "monotonic", lambda: next(monotonic_times))
    failure_path = tmp_path / ".pyxis_infrastructure_failure"
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            command, 167, stdout="Job credential expired", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RunnerError, match=r"exceeded its 930s deadline before retry"):
        run_srun_step(
            argv=["true"],
            status_path=Path(pyxis_environment_mod._STEP_STATUS),
            timeout_s=30,
            failure_path=failure_path,
            image=tmp_path / "task.sqsh",
        )

    assert calls == 1
    assert failure_path.exists()


def test_pyxis_does_not_retry_unmounted_host_srun(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    failure_path = tmp_path / ".pyxis_infrastructure_failure"
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            command, 167, stdout="Job credential expired", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RunnerError, match="before the command completed"):
        run_srun_step(
            argv=["true"],
            status_path=Path(pyxis_environment_mod._STEP_STATUS),
            timeout_s=30,
            failure_path=failure_path,
        )

    assert calls == 1
    assert failure_path.exists()


@pytest.mark.parametrize(
    ("status", "returncode", "output"),
    [
        ("started", 167, "Job credential expired"),
        ("finished", 167, "Job credential expired"),
        ("pending", 167, "unrelated launch failure"),
        ("pending", 1, "Job credential expired"),
    ],
)
def test_pyxis_does_not_retry_nonexact_srun_launch_failures(
    monkeypatch, tmp_path, status, returncode, output
):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    failure_path = tmp_path / ".pyxis_infrastructure_failure"
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if status == "finished":
            _finish_srun_step(command, returncode)
        elif status == "started":
            _required_srun_status_mount(command)[0].write_text("started\n")
        return subprocess.CompletedProcess(command, returncode, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    step = {
        "argv": ["true"],
        "status_path": Path(pyxis_environment_mod._STEP_STATUS),
        "timeout_s": 30,
        "failure_path": failure_path,
        "image": tmp_path / "task.sqsh",
    }

    if status == "finished":
        assert run_srun_step(**step).returncode == returncode
        assert not failure_path.exists()
    else:
        with pytest.raises(RunnerError):
            run_srun_step(**step)
        assert failure_path.exists()
    assert calls == 1


def test_pyxis_does_not_retry_named_container_creation(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    failure_path = tmp_path / ".pyxis_infrastructure_failure"
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            command, 167, stdout="Job credential expired", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RunnerError, match="Job credential expired"):
        run_srun_step(
            argv=["true"],
            status_path=Path(pyxis_environment_mod._STEP_STATUS),
            timeout_s=30,
            failure_path=failure_path,
            image=tmp_path / "task.sqsh",
            name="new-container",
        )

    assert calls == 1
    assert failure_path.exists()


def test_pyxis_container_create_uses_the_pull_budget_not_the_command_budget(
    monkeypatch, tmp_path
):
    """Creating the container is an image import, not a shell command.

    Under Pyxis, `--container-image` triggers an enroot import of a multi-GB
    SWE-bench image from a remote registry. Charging that against the
    per-command `timeout` (300s for the default template) killed the create step as
    soon as enough concurrent workers shared the registry -- 96 srun steps of
    one 200-instance run were SIGKILLed at a uniform ~5m50s (= 300 + 30 grace),
    which the service reported as an undiagnosable "failed to start Pyxis
    container" and cost 17 of 20 units.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    timeouts: list[float] = []

    def fake_run(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(
        image=tmp_path / "task.sqsh",
        run_id="run-1",
        timeout=300,
        pull_timeout=3600,
    )
    environment.execute({"command": "pytest -q"})

    create_timeout, command_timeout = timeouts[0], timeouts[1]
    assert 0 < create_timeout <= 3600 + pyxis_environment_mod.SRUN_TEARDOWN_GRACE_S
    assert (
        0
        < command_timeout
        <= 300 + pyxis_environment_mod.SRUN_PYXIS_WORKLOAD_TEARDOWN_GRACE_S
    )
    assert create_timeout > command_timeout, (
        "container creation must not be bounded by the per-command timeout"
    )
    environment.cleanup()


def test_pyxis_container_create_budget_defaults_without_a_template_value(
    monkeypatch, tmp_path
):
    """A template that never mentions pull_timeout still gets a pull budget."""
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    timeouts: list[float] = []

    def fake_run(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(
        image=tmp_path / "task.sqsh", run_id="run-1", timeout=300
    )

    assert 0 < timeouts[0] <= 3600 + pyxis_environment_mod.SRUN_TEARDOWN_GRACE_S
    environment.cleanup()


def test_pyxis_records_create_timing_when_enabled(monkeypatch, tmp_path):
    """Container-create cost must be measurable without re-deriving it.

    Creation was only ever observable after the fact, as a uniform block of
    SIGKILLed steps in `sacct` -- by which point the run was already lost.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    timing = tmp_path / "creates.jsonl"
    monkeypatch.setenv("SWEBENCH_PYXIS_CREATE_TIMING_PATH", str(timing))

    def fake_run(command, **kwargs):
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(image=tmp_path / "task.sqsh", run_id="run-1")
    environment.cleanup()

    records = [json.loads(line) for line in timing.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["ok"] is True
    assert records[0]["secs"] >= 0
    assert records[0]["image"].endswith("task.sqsh")


def test_pyxis_records_create_timing_for_a_failed_create(monkeypatch, tmp_path):
    """A create that failed is the one whose duration matters most."""
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    timing = tmp_path / "creates.jsonl"
    monkeypatch.setenv("SWEBENCH_PYXIS_CREATE_TIMING_PATH", str(timing))

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, stdout=""),
    )

    with pytest.raises(RunnerError):
        PyxisEnvironment(image=tmp_path / "task.sqsh", run_id="run-1")

    records = [json.loads(line) for line in timing.read_text().splitlines()]
    assert [r["ok"] for r in records] == [False]


def test_pyxis_create_timing_is_off_by_default(monkeypatch, tmp_path):
    """No env var, no writes, no behaviour change on a normal run."""
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    monkeypatch.delenv("SWEBENCH_PYXIS_CREATE_TIMING_PATH", raising=False)

    def fake_run(command, **kwargs):
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(image=tmp_path / "task.sqsh", run_id="run-1")
    environment.cleanup()

    assert list(tmp_path.glob("*.jsonl")) == []


def test_pyxis_create_timing_never_fails_the_run(monkeypatch, tmp_path):
    """An unwritable sink degrades to nothing; it does not lose the container."""
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    monkeypatch.setenv(
        "SWEBENCH_PYXIS_CREATE_TIMING_PATH", str(tmp_path / "nope" / "creates.jsonl")
    )

    def fake_run(command, **kwargs):
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(image=tmp_path / "task.sqsh", run_id="run-1")
    environment.cleanup()


def test_pyxis_failure_carries_srun_output(monkeypatch, tmp_path):
    """srun's own words must survive into the error.

    Without them every distinct infrastructure failure -- import failure, no
    space left, a step that never got resources -- collapses into one
    indistinguishable message and cannot be diagnosed from the artifacts.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")

    def fake_run(command, **kwargs):
        # Step never wrote its status file: srun died before the command ran.
        return subprocess.CompletedProcess(
            command, 1, stdout="slurmstepd: error: pyxis: no space left\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RunnerError, match="no space left") as exc_info:
        PyxisEnvironment(image=tmp_path / "task.sqsh", run_id="run-1")

    assert "failed to start Pyxis container" in str(exc_info.value)


def test_pyxis_environment_preserves_command_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        returncode = 0 if calls == 1 else 1
        _finish_srun_step(command, returncode)
        return subprocess.CompletedProcess(
            command, returncode, stdout="command failed\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = PyxisEnvironment(image=tmp_path / "task.sqsh", run_id="run-1")

    output = environment.execute({"command": "false"})

    assert output["returncode"] == 1
    assert output["output"] == "command failed\n"
    environment.cleanup()


def test_pyxis_cleanup_is_best_effort_outside_allocation(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")

    def fake_run(command, **kwargs):
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    image = tmp_path / "task.sqsh"
    image.touch()
    environment = PyxisEnvironment(image=image, run_id="run-1")
    persistent_tmp = environment._tmp_dir
    monkeypatch.delenv("SLURM_JOB_ID")

    environment.cleanup()

    assert not persistent_tmp.exists()


def test_pyxis_finalizer_logs_cleanup_failure(monkeypatch, caplog):
    environment = object.__new__(PyxisEnvironment)

    def fail_cleanup():
        raise OSError("cleanup failed")

    monkeypatch.setattr(environment, "cleanup", fail_cleanup)

    environment.__del__()

    assert "Could not clean up Pyxis environment" in caplog.text


def test_pyxis_start_failure_removes_persistent_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    image = tmp_path / "task.sqsh"
    image.touch()
    persistent_tmp: Path | None = None

    def fake_run(command, **kwargs):
        nonlocal persistent_tmp
        if any(arg.startswith("--container-image=") for arg in command):
            mount = next(
                arg for arg in command if arg.startswith("--container-mounts=")
            )
            source = mount.removeprefix("--container-mounts=").split(":", 1)[0]
            persistent_tmp = Path(source)
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RunnerError, match="failed to start Pyxis container"):
        PyxisEnvironment(image=image, run_id="run-1")

    assert persistent_tmp is not None
    assert not persistent_tmp.exists()


def test_pyxis_cleanup_removes_only_exact_run_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = ""
        if command[-3:] == ["enroot", "list", "-f"]:
            output = (
                "NAME PID COMM STATE STARTED TIME MNTNS USERNS COMMAND\n"
                "pyxis_mswe_run-1_abcd1234                         \n"
                "pyxis_mswe_run-10_ffff0000                        \n"
                "unrelated                                         \n"
            )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = PyxisSweBenchRunner(
        project_root=tmp_path,
        subprocess_timeout_s=30,
        image_registry=_PYXIS_IMAGE_REGISTRY,
    )

    runner._cleanup_containers("run-1")

    assert [command[-4:] for command, _ in calls[1:]] == [
        ["enroot", "remove", "-f", "pyxis_mswe_run-1_abcd1234"]
    ]
    assert [kwargs["timeout"] for _, kwargs in calls] == [
        pyxis_environment_mod.PYXIS_CLEANUP_TIMEOUT_S,
        pyxis_environment_mod.PYXIS_CLEANUP_TIMEOUT_S,
    ]


def test_pyxis_agent_command_uses_host_model_and_image_registry(monkeypatch, tmp_path):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, log_path, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run)
    request = _pyxis_request(["repo__repo-1", "repo__repo-2"])
    request.endpoint_api_key = "model-secret"
    runner = PyxisSweBenchRunner(
        project_root=tmp_path,
        subprocess_timeout_s=30,
        image_registry=_PYXIS_IMAGE_REGISTRY,
    )

    runner._run_agent(
        request,
        tmp_path / "config.yaml",
        tmp_path,
        tmp_path,
        {"model-secret"},
    )

    command, kwargs = calls[0]
    assert command[1:4] == ["-m", "swebench_service.pyxis_worker", "agent"]
    assert command[command.index("--image-registry") + 1] == _PYXIS_IMAGE_REGISTRY
    assert command[command.index("--filter") + 1] == (
        "^(?:repo__repo\\-1|repo__repo\\-2)$"
    )
    assert kwargs["env"]["OPENAI_API_KEY"] == "model-secret"


def test_pyxis_agent_requires_upstream_environment_hook(monkeypatch, tmp_path):
    swebench = types.SimpleNamespace(main=lambda **kwargs: None)
    _install_fake_minisweagent(monkeypatch, swebench)

    with pytest.raises(RuntimeError, match="get_sb_environment"):
        worker_mod.main(_pyxis_agent_args(tmp_path))


def test_pyxis_agent_restores_upstream_environment_hook(monkeypatch, tmp_path):
    original = object()
    swebench = types.SimpleNamespace(
        get_sb_environment=original,
        main=lambda **kwargs: None,
    )
    _install_fake_minisweagent(monkeypatch, swebench)

    worker_mod.main(_pyxis_agent_args(tmp_path))

    assert swebench.get_sb_environment is original


def test_pyxis_agent_propagates_environment_infrastructure_failure(
    monkeypatch, tmp_path
):
    original = object()

    def fake_main(**kwargs):
        (tmp_path / ".pyxis_infrastructure_failure").touch()

    swebench = types.SimpleNamespace(
        get_sb_environment=original,
        main=fake_main,
    )
    _install_fake_minisweagent(monkeypatch, swebench)

    with pytest.raises(RunnerError, match="infrastructure failure"):
        worker_mod.main(_pyxis_agent_args(tmp_path))


def test_pyxis_eval_command_does_not_forward_model_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    calls: list[tuple[list[str], dict]] = []
    output_dir = tmp_path / "output"
    run_dir = tmp_path / "run"
    output_dir.mkdir()
    run_dir.mkdir()
    preds_path = output_dir / "preds.json"
    preds_path.write_text("{}")

    def fake_run(command, log_path, *, cwd, env, **kwargs):
        calls.append((command, {"cwd": cwd, "env": env}))
        run_id = command[command.index("--run-id") + 1]
        (cwd / f"test-model.{run_id}.json").write_text("{}")

    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run)
    runner = PyxisSweBenchRunner(
        project_root=tmp_path,
        subprocess_timeout_s=30,
        image_registry=_PYXIS_IMAGE_REGISTRY,
    )

    result = runner._run_eval(
        _pyxis_request(), preds_path, output_dir, run_dir, {"ambient-secret"}
    )

    command, kwargs = calls[0]
    assert command[1:4] == ["-m", "swebench_service.pyxis_worker", "eval"]
    assert command[command.index("--max-workers") + 1] == "2"
    assert command[command.index("--image-registry") + 1] == _PYXIS_IMAGE_REGISTRY
    assert command[command.index("--instance-ids") + 1 :] == ["repo__repo-1"]
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert result.exists()


def test_pyxis_eval_finds_nested_result(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    run_dir = tmp_path / "run"
    output_dir.mkdir()
    run_dir.mkdir()
    preds_path = output_dir / "preds.json"
    preds_path.write_text("{}")

    def fake_run(command, log_path, *, cwd, **kwargs):
        run_id = command[command.index("--run-id") + 1]
        nested = cwd / "nested"
        nested.mkdir()
        (nested / f"result.{run_id}.json").write_text("{}")

    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run)
    runner = PyxisSweBenchRunner(
        project_root=tmp_path,
        subprocess_timeout_s=30,
        image_registry=_PYXIS_IMAGE_REGISTRY,
    )

    result = runner._run_eval(_pyxis_request(), preds_path, output_dir, run_dir, set())

    assert result.parent.name == "nested"


def test_pyxis_eval_does_not_mount_report_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    image = tmp_path / "task.sqsh"
    image.touch()
    report = (
        tmp_path
        / "logs"
        / "run_evaluation"
        / "run-1"
        / "test-model"
        / "repo__repo-1"
        / "report.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text('{"repo__repo-1":{"resolved":true}}')
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _finish_srun_step(command, 1)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    test_spec = types.SimpleNamespace(instance_id="repo__repo-1", eval_script="true")

    worker_mod._evaluate_instance(
        test_spec=test_spec,
        prediction={
            "model_name_or_path": "test-model",
            "model_patch": "diff --git a/a b/a",
        },
        image=image,
        output_dir=tmp_path,
        run_id="run-1",
        timeout_s=30,
    )

    mounts = next(arg for arg in calls[0] if arg.startswith("--container-mounts="))
    assert "report.json" not in mounts
    assert f"{report.parent.resolve()}:/" not in mounts
    assert not report.exists()


def test_pyxis_evaluate_instance_writes_upstream_report(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1738605")
    monkeypatch.setenv("SLURMD_NODENAME", "gb-nvl-053-compute04")
    grading = types.ModuleType("swebench.harness.grading")
    grading.get_eval_report = lambda **kwargs: {"repo__repo-1": {"resolved": True}}
    swebench = types.ModuleType("swebench")
    harness = types.ModuleType("swebench.harness")
    monkeypatch.setitem(sys.modules, "swebench", swebench)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness)
    monkeypatch.setitem(sys.modules, "swebench.harness.grading", grading)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _finish_srun_step(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout="tests passed", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    worker_mod._evaluate_instance(
        test_spec=types.SimpleNamespace(
            instance_id="repo__repo-1", eval_script="pytest -q"
        ),
        prediction={
            "model_name_or_path": "test-model",
            "model_patch": "diff --git a/a b/a",
        },
        image=tmp_path / "task.sqsh",
        output_dir=tmp_path,
        run_id="run-1",
        timeout_s=30,
    )

    report = tmp_path / "logs/run_evaluation/run-1/test-model/repo__repo-1/report.json"
    assert msgspec.json.decode(report.read_bytes()) == {
        "repo__repo-1": {"resolved": True}
    }
    mounts = next(arg for arg in calls[0] if arg.startswith("--container-mounts="))
    assert "patch.diff:/tmp/swebench_patch.diff" in mounts
    assert "eval.sh:/tmp/swebench_eval.sh" in mounts
    assert "test_output.txt:/tmp/swebench_test_output.txt" in mounts


def test_pyxis_worker_uses_upstream_report(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    run_id = "endpoints_test"
    predictions = {
        "repo__repo-1": {
            "model_name_or_path": "test-model",
            "instance_id": "repo__repo-1",
            "model_patch": "diff --git a/a b/a",
        },
        "repo__repo-2": {
            "model_name_or_path": "test-model",
            "instance_id": "repo__repo-2",
            "model_patch": "",
        },
    }
    output_dir.mkdir()
    calls = []

    def fake_make_run_report(predictions, dataset, run_id, client):
        calls.append((predictions, dataset, run_id, client))
        result = tmp_path / "output" / "test-model.endpoints_test.json"
        result.write_text("{}")
        return result.name

    swebench = types.ModuleType("swebench")
    harness = types.ModuleType("swebench.harness")
    reporting = types.ModuleType("swebench.harness.reporting")
    reporting.make_run_report = fake_make_run_report
    test_spec = types.ModuleType("swebench.harness.test_spec")
    test_spec_module = types.ModuleType("swebench.harness.test_spec.test_spec")
    test_spec_module.make_test_spec = lambda row, arch: row
    utils = types.ModuleType("swebench.harness.utils")
    utils.get_predictions_from_file = lambda path, dataset, split: predictions.values()
    utils.load_swebench_dataset = lambda dataset, split, instance_ids: []
    monkeypatch.setitem(sys.modules, "swebench", swebench)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness)
    monkeypatch.setitem(sys.modules, "swebench.harness.reporting", reporting)
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec", test_spec)
    monkeypatch.setitem(
        sys.modules, "swebench.harness.test_spec.test_spec", test_spec_module
    )
    monkeypatch.setitem(sys.modules, "swebench.harness.utils", utils)

    worker_mod.main(
        [
            "eval",
            "--dataset-name",
            "princeton-nlp/SWE-bench_Verified",
            "--split",
            "test",
            "--predictions-path",
            str(output_dir / "preds.json"),
            "--max-workers",
            "1",
            "--run-id",
            run_id,
            "--image-registry",
            _PYXIS_IMAGE_REGISTRY,
            "--output-dir",
            str(output_dir),
            "--instance-ids",
            "repo__repo-1",
            "repo__repo-2",
            "repo__repo-3",
        ]
    )

    assert (output_dir / "test-model.endpoints_test.json").exists()
    assert calls == [
        (
            predictions,
            [
                {"instance_id": "repo__repo-1"},
                {"instance_id": "repo__repo-2"},
                {"instance_id": "repo__repo-3"},
            ],
            run_id,
            None,
        )
    ]


def test_pyxis_worker_propagates_evaluation_infrastructure_failure(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    predictions = {
        "repo__repo-1": {
            "model_name_or_path": "test-model",
            "instance_id": "repo__repo-1",
            "model_patch": "diff --git a/a b/a",
        }
    }

    swebench = types.ModuleType("swebench")
    harness = types.ModuleType("swebench.harness")
    reporting = types.ModuleType("swebench.harness.reporting")
    reporting.make_run_report = lambda *args, **kwargs: None
    test_spec = types.ModuleType("swebench.harness.test_spec")
    test_spec_module = types.ModuleType("swebench.harness.test_spec.test_spec")
    test_spec_module.make_test_spec = lambda row, arch: types.SimpleNamespace(
        instance_id=row["instance_id"], eval_script="pytest -q"
    )
    utils = types.ModuleType("swebench.harness.utils")
    utils.get_predictions_from_file = lambda *args: predictions.values()
    utils.load_swebench_dataset = lambda *args: [{"instance_id": "repo__repo-1"}]
    monkeypatch.setitem(sys.modules, "swebench", swebench)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness)
    monkeypatch.setitem(sys.modules, "swebench.harness.reporting", reporting)
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec", test_spec)
    monkeypatch.setitem(
        sys.modules, "swebench.harness.test_spec.test_spec", test_spec_module
    )
    monkeypatch.setitem(sys.modules, "swebench.harness.utils", utils)
    def fail_with_marker(**kwargs):
        failure_path = worker_mod._eval_infrastructure_failure_path(
            kwargs["output_dir"],
            kwargs["run_id"],
            kwargs["prediction"],
            kwargs["test_spec"].instance_id,
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.touch()
        raise RunnerError("Pyxis infrastructure failure")

    monkeypatch.setattr(worker_mod, "_evaluate_instance", fail_with_marker)

    with pytest.raises(RunnerError, match="repo__repo-1"):
        worker_mod.main(
            [
                "eval",
                "--dataset-name",
                "princeton-nlp/SWE-bench_Verified",
                "--split",
                "test",
                "--predictions-path",
                str(output_dir / "preds.json"),
                "--max-workers",
                "1",
                "--run-id",
                "endpoints_test",
                "--image-registry",
                _PYXIS_IMAGE_REGISTRY,
                "--output-dir",
                str(output_dir),
                "--instance-ids",
                "repo__repo-1",
            ]
        )


def test_pyxis_worker_preserves_non_infrastructure_evaluation_failure(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    predictions = {
        "repo__repo-1": {
            "model_name_or_path": "test-model",
            "instance_id": "repo__repo-1",
            "model_patch": "diff --git a/a b/a",
        }
    }

    swebench = types.ModuleType("swebench")
    harness = types.ModuleType("swebench.harness")
    reporting = types.ModuleType("swebench.harness.reporting")
    reporting.make_run_report = lambda *args, **kwargs: None
    test_spec = types.ModuleType("swebench.harness.test_spec")
    test_spec_module = types.ModuleType("swebench.harness.test_spec.test_spec")
    test_spec_module.make_test_spec = lambda row, arch: types.SimpleNamespace(
        instance_id=row["instance_id"], eval_script="pytest -q"
    )
    utils = types.ModuleType("swebench.harness.utils")
    utils.get_predictions_from_file = lambda *args: predictions.values()
    utils.load_swebench_dataset = lambda *args: [{"instance_id": "repo__repo-1"}]
    monkeypatch.setitem(sys.modules, "swebench", swebench)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness)
    monkeypatch.setitem(sys.modules, "swebench.harness.reporting", reporting)
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec", test_spec)
    monkeypatch.setitem(
        sys.modules, "swebench.harness.test_spec.test_spec", test_spec_module
    )
    monkeypatch.setitem(sys.modules, "swebench.harness.utils", utils)
    monkeypatch.setattr(
        worker_mod,
        "_evaluate_instance",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("malformed report")),
    )

    with pytest.raises(
        RunnerError, match="SWE-bench evaluation failed for: repo__repo-1"
    ):
        worker_mod.main(
            [
                "eval",
                "--dataset-name",
                "princeton-nlp/SWE-bench_Verified",
                "--split",
                "test",
                "--predictions-path",
                str(output_dir / "preds.json"),
                "--max-workers",
                "1",
                "--run-id",
                "endpoints_test",
                "--image-registry",
                _PYXIS_IMAGE_REGISTRY,
                "--output-dir",
                str(output_dir),
                "--instance-ids",
                "repo__repo-1",
            ]
        )

    assert not (output_dir / ".pyxis_infrastructure_failure").exists()

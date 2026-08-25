# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and registration of the fleet scorer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_endpoint.config.schema import ScorerMethod
from inference_endpoint.evaluation import swe_bench_fleet_scorer as fleet_scorer_mod
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.evaluation.swe_bench_fleet_scorer import SWEBenchFleetScorer
from inference_endpoint.evaluation.swe_bench_distributed.fleet import build_gates
from inference_endpoint.evaluation.swe_bench_distributed.gates import ToolCallGate
from inference_endpoint.evaluation.swe_bench_scorer import SWEBenchScorer
from inference_endpoint.exceptions import SetupError

pytestmark = pytest.mark.unit

URLS = ["http://svc-a:18080", "http://svc-b:18080"]


class TestRegistration:
    def test_the_scorer_is_registered(self):
        assert Scorer.get("swe_bench_fleet") is SWEBenchFleetScorer

    def test_the_scorer_method_enum_is_in_sync(self):
        assert ScorerMethod.SWE_BENCH_FLEET.value in Scorer.available_scorers()

    def test_it_skips_the_endpoint_phase(self):
        # Like the single-service scorer, this one drives the run itself rather
        # than consuming responses collected by the load generator.
        assert SWEBenchFleetScorer.SKIP_ENDPOINT_PHASE
        assert not SWEBenchFleetScorer.REQUIRES_EXTRACTOR


class TestOptions:
    def test_service_urls_are_normalised(self):
        options = SWEBenchFleetScorer._resolve_options(
            {"swebench_service_urls": ["http://svc-a:18080/"]}
        )
        assert options["service_urls"] == ["http://svc-a:18080/"]

    def test_a_comma_separated_string_is_accepted(self):
        options = SWEBenchFleetScorer._resolve_options(
            {"swebench_service_urls": "http://svc-a:18080, http://svc-b:18080"}
        )
        assert len(options["service_urls"]) == 2

    def test_the_single_service_key_still_works(self):
        options = SWEBenchFleetScorer._resolve_options(
            {"swebench_service_url": "http://svc-a:18080"}
        )
        assert options["service_urls"] == ["http://svc-a:18080/"]

    def test_no_service_urls_is_a_setup_error(self):
        with pytest.raises(SetupError, match="swebench_service_urls is required"):
            SWEBenchFleetScorer._resolve_options({})

    def test_duplicate_service_urls_are_refused(self):
        # Two entries for one host is not extra capacity; it is two concurrent
        # runs contending for the same container runtime.
        with pytest.raises(SetupError, match="duplicate"):
            SWEBenchFleetScorer._resolve_options(
                {"swebench_service_urls": ["http://svc-a:18080", "http://svc-a:18080/"]}
            )

    def test_defaults_are_sane(self):
        options = SWEBenchFleetScorer._resolve_options(
            {"swebench_service_urls": URLS}
        )
        assert options["shard_size"] == 10
        assert options["max_attempts"] == 3
        assert options["max_consecutive_env_faults"] == 3
        assert options["env_fault_backoff_s"] == 60
        # The tool-call gate's floor must stay at SWE-bench prompt scale.
        assert options["min_prompt_tokens"] == 2000
        assert options["tool_call_timeout_s"] == 180

    def test_tool_call_timeout_must_be_positive(self):
        with pytest.raises(SetupError, match="tool_call_timeout_s"):
            SWEBenchFleetScorer._resolve_options(
                {"swebench_service_urls": URLS, "tool_call_timeout_s": 0}
            )

    def test_a_bad_shard_size_is_rejected(self):
        with pytest.raises(SetupError, match="shard_size"):
            SWEBenchFleetScorer._resolve_options(
                {"swebench_service_urls": URLS, "shard_size": 0}
            )

    def test_max_consecutive_env_faults_must_be_positive(self):
        with pytest.raises(SetupError, match="max_consecutive_env_faults"):
            SWEBenchFleetScorer._resolve_options(
                {"swebench_service_urls": URLS, "max_consecutive_env_faults": 0}
            )

    def test_env_fault_backoff_s_must_be_nonnegative(self):
        with pytest.raises(SetupError, match="env_fault_backoff_s"):
            SWEBenchFleetScorer._resolve_options(
                {"swebench_service_urls": URLS, "env_fault_backoff_s": -1}
            )


class TestDispatcherConfiguration:
    def test_environment_fault_options_are_forwarded(self, monkeypatch, tmp_path):
        captured: dict[str, object] = {}
        gate_options: dict[str, object] = {}

        class FakeDispatcher:
            quarantined: dict[str, str] = {}

            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self):
                pass

        scorer = object.__new__(SWEBenchFleetScorer)
        scorer.report_dir = tmp_path
        scorer.options = SWEBenchFleetScorer._resolve_options(
            {
                "swebench_service_urls": URLS,
                "max_consecutive_env_faults": 5,
                "env_fault_backoff_s": 17,
                "tool_call_timeout_s": 321,
            }
        )

        monkeypatch.setattr(
            fleet_scorer_mod,
            "load_benchmark_config",
            lambda _: {
                "model_params": {"name": "test-model"},
                "endpoint_config": {"endpoints": ["http://endpoint"]},
            },
        )
        monkeypatch.setattr(
            SWEBenchFleetScorer, "_instance_ids", lambda _: ["instance-1"]
        )
        monkeypatch.setattr(
            fleet_scorer_mod,
            "build_gates",
            lambda **kwargs: (
                gate_options.update(kwargs) or [],
                SimpleNamespace(fingerprint=lambda _: "fingerprint"),
            ),
        )
        monkeypatch.setattr(fleet_scorer_mod, "run_gates", lambda *_: None)
        monkeypatch.setattr(
            SWEBenchScorer, "_generation_params", staticmethod(lambda _: {})
        )
        monkeypatch.setattr(
            fleet_scorer_mod,
            "plan_units",
            lambda *_args, **_kwargs: SimpleNamespace(run_id="test-run", digest="digest"),
        )
        monkeypatch.setattr(fleet_scorer_mod, "WorkQueue", lambda *_: object())
        monkeypatch.setattr(fleet_scorer_mod, "FleetDispatcher", FakeDispatcher)
        monkeypatch.setattr(
            fleet_scorer_mod,
            "merge_run",
            lambda *_: SimpleNamespace(
                resolved_instances=1,
                total_instances=1,
                resolved_rate=1.0,
                unit_count=1,
                to_dict=lambda: {},
            ),
        )
        monkeypatch.setattr(fleet_scorer_mod, "write_merge_artifacts", lambda *_: None)

        assert scorer.score() == (1.0, 1)
        assert captured["max_consecutive_env_faults"] == 5
        assert captured["env_fault_backoff_s"] == 17
        assert gate_options["tool_call_timeout_s"] == 321


class TestGateConfiguration:
    def test_tool_call_timeout_is_applied_only_to_the_tool_call_gate(self):
        gates, _ = build_gates(
            expected_model="Org/Model",
            tool_call_model="Org/Model",
            min_prompt_tokens=2000,
            tool_call_timeout_s=321,
        )
        tool_call_gate = next(gate for gate in gates if isinstance(gate, ToolCallGate))
        assert tool_call_gate.timeout_s == 321

    def test_preflight_forwards_the_tool_call_timeout(self, monkeypatch):
        gate_options: dict[str, object] = {}
        monkeypatch.setattr(SWEBenchScorer, "_check_health", lambda *_: None)
        monkeypatch.setattr(
            fleet_scorer_mod,
            "build_gates",
            lambda **kwargs: (
                gate_options.update(kwargs) or [],
                SimpleNamespace(fingerprint=lambda _: "fingerprint"),
            ),
        )
        monkeypatch.setattr(fleet_scorer_mod, "run_gates", lambda *_: None)

        SWEBenchFleetScorer.preflight(
            {
                "swebench_service_urls": URLS,
                "endpoint_urls": ["http://endpoint"],
                "model_name": "Org/Model",
                "tool_call_timeout_s": 321,
            }
        )

        assert gate_options["tool_call_timeout_s"] == 321


class TestPollUnit:
    @staticmethod
    def _scorer(*, service_timeout_s: float = 10.0) -> SWEBenchFleetScorer:
        scorer = object.__new__(SWEBenchFleetScorer)
        scorer.options = {
            "auth_token": None,
            "poll_interval_s": 0.0,
            "service_timeout_s": service_timeout_s,
        }
        return scorer

    def test_timeouts_continue_polling_the_same_service_run(self, monkeypatch):
        calls: list[str] = []
        responses = iter([TimeoutError(), {"status": "succeeded"}])
        cancel_calls: list[tuple[str, str, str | None]] = []

        def fake_http_json(url, **kwargs):
            calls.append(url)
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return response

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        monkeypatch.setattr(
            SWEBenchScorer,
            "_cancel_service_run",
            lambda *args: cancel_calls.append(args),
        )
        monkeypatch.setattr(fleet_scorer_mod.time, "sleep", lambda _: None)
        monotonic = iter([0.0, 0.0, 1.0])
        monkeypatch.setattr(
            fleet_scorer_mod.time, "monotonic", lambda: next(monotonic)
        )

        status = self._scorer()._poll_unit("http://svc-a:18080/", "run-42")

        assert status == {"status": "succeeded"}
        assert calls == [
            "http://svc-a:18080/v1/runs/run-42",
            "http://svc-a:18080/v1/runs/run-42",
        ]
        assert cancel_calls == []

    def test_timeout_deadline_still_cancels_after_a_poll_timeout(self, monkeypatch):
        cancel_calls: list[tuple[str, str, str | None]] = []

        def timeout_http_json(url, **kwargs):
            raise TimeoutError

        monkeypatch.setattr(SWEBenchScorer, "_http_json", timeout_http_json)
        monkeypatch.setattr(
            SWEBenchScorer,
            "_cancel_service_run",
            lambda *args: cancel_calls.append(args),
        )
        monkeypatch.setattr(fleet_scorer_mod.time, "sleep", lambda _: None)
        monotonic = iter([0.0, 0.0, 1.0])
        monkeypatch.setattr(
            fleet_scorer_mod.time, "monotonic", lambda: next(monotonic)
        )

        with pytest.raises(SetupError, match="timed out waiting"):
            self._scorer(service_timeout_s=1.0)._poll_unit(
                "http://svc-a:18080/", "run-42"
            )

        assert cancel_calls == [("http://svc-a:18080/", "run-42", None)]

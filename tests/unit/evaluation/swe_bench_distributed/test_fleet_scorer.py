# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and registration of the fleet scorer."""

from __future__ import annotations

import pytest

from inference_endpoint.config.schema import ScorerMethod
from inference_endpoint.evaluation import swe_bench_fleet_scorer as fleet_scorer_mod
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.evaluation.swe_bench_fleet_scorer import SWEBenchFleetScorer
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
        # The tool-call gate's floor must stay at SWE-bench prompt scale.
        assert options["min_prompt_tokens"] == 2000

    def test_a_bad_shard_size_is_rejected(self):
        with pytest.raises(SetupError, match="shard_size"):
            SWEBenchFleetScorer._resolve_options(
                {"swebench_service_urls": URLS, "shard_size": 0}
            )


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

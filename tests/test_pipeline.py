import hashlib

import httpx
import pytest

import pulso_pipeline.submit_current_cycle as spc
from pulso_pipeline.submit_current_cycle import _predictions_hash
from pulso_transmi import PulsoTransmiApiError, PulsoTransmiClient, PulsoTransmiError


def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/forecast-cycles/current":
        if request.headers.get("X-Cycle-Open") == "1":
            return httpx.Response(200, json={"cycle_id": "cyc_test", "expected_predictions": 1})
        return httpx.Response(404, json={"detail": {"code": "no_open_cycle", "message": "none"}})
    if request.url.path == "/v1/submissions":
        assert request.headers["Idempotency-Key"] == "stable-key"
        return httpx.Response(201, json={"submission_id": "sub_123", "status": "accepted"})
    return httpx.Response(404, json={"detail": "not found"})


def client() -> PulsoTransmiClient:
    return PulsoTransmiClient(base_url="https://example.test", transport=httpx.MockTransport(handler))


def test_current_cycle_returns_none_on_404() -> None:
    with client() as api:
        assert api.current_cycle() is None


def test_current_cycle_returns_payload_when_open() -> None:
    with client() as api:
        api._client.headers["X-Cycle-Open"] = "1"
        cycle = api.current_cycle()
    assert cycle["cycle_id"] == "cyc_test"


def test_submit_sends_idempotency_header_and_returns_receipt() -> None:
    with client() as api:
        receipt = api.submit(
            cycle_id="cyc_test",
            client_run_id="run-1",
            data_cutoff="2026-09-10T03:00:00Z",
            model={"version": "v1"},
            predictions=[{"station_id": "02300", "target_at": "2026-09-10T03:15:00Z", "value": 10.0}],
            idempotency_key="stable-key",
        )
    assert receipt["submission_id"] == "sub_123"


def test_submit_raises_api_error_with_status_code() -> None:
    def conflict_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": {"code": "cycle_closed"}})

    with PulsoTransmiClient(base_url="https://example.test", transport=httpx.MockTransport(conflict_handler)) as api:
        with pytest.raises(PulsoTransmiApiError) as exc_info:
            api.submit(
                cycle_id="cyc_test", client_run_id="run-1", data_cutoff="2026-09-10T03:00:00Z",
                model={"version": "v1"}, predictions=[], idempotency_key="k",
            )
    assert exc_info.value.status_code == 409


def test_predictions_hash_is_stable_for_same_payload() -> None:
    payload = [{"station_id": "02300", "target_at": "2026-09-10T03:15:00Z", "value": 10.0}]
    assert _predictions_hash(payload) == _predictions_hash(list(payload))


def test_predictions_hash_changes_with_payload() -> None:
    a = [{"station_id": "02300", "target_at": "2026-09-10T03:15:00Z", "value": 10.0}]
    b = [{"station_id": "02300", "target_at": "2026-09-10T03:15:00Z", "value": 11.0}]
    assert _predictions_hash(a) != _predictions_hash(b)


def test_fetch_page_with_retries_recovers_after_transient_failure(monkeypatch) -> None:
    monkeypatch.setattr(spc.time, "sleep", lambda _: None)
    calls = []

    def flaky(cursor):
        calls.append(cursor)
        if len(calls) < 2:
            raise PulsoTransmiError("GET /v1/stream/observations failed: timed out")
        return {"data": [], "next_cursor": None}

    result = spc._fetch_page_with_retries(flaky, cursor="cur-1")
    assert result == {"data": [], "next_cursor": None}
    assert calls == ["cur-1", "cur-1"]


def test_fetch_page_with_retries_raises_after_max_attempts(monkeypatch) -> None:
    monkeypatch.setattr(spc.time, "sleep", lambda _: None)
    calls = []

    def always_fails(cursor):
        calls.append(cursor)
        raise PulsoTransmiError("GET /v1/stream/observations failed: timed out")

    with pytest.raises(PulsoTransmiError):
        spc._fetch_page_with_retries(always_fails, cursor=None)
    assert len(calls) == spc.RETRYABLE_MAX_ATTEMPTS


# --------------------------------------------------------- gate de promoción
import pulso_pipeline.train_and_promote as tap  # noqa: E402


def _report(same_block):
    return {"horizons": {f"h{h}": {"model": 85.0, **({"same_block": same_block} if same_block else {})}
                         for h in (1, 2, 3, 4)}}


def test_same_block_gate_uses_both_models_on_the_same_targets() -> None:
    sb = {"n_targets": 5000, "model": 87.0, "reference": 86.0}
    assert tap.same_block_accuracies(_report(sb)) == (87.0, 86.0, 5000)


def test_same_block_gate_falls_back_when_champion_saw_the_block() -> None:
    assert tap.same_block_accuracies(_report(None)) is None
    few = {"n_targets": tap.SAME_BLOCK_MIN_TARGETS - 1, "model": 87.0, "reference": 86.0}
    assert tap.same_block_accuracies(_report(few)) is None


# ------------------------------------------------- alerta de contexto atrasado
def test_context_alert_skipped_while_context_never_published_in_competition(monkeypatch) -> None:
    saved = []
    monkeypatch.setattr(spc.db, "latest_context_at", lambda _db: "2026-09-09T04:45:00+00:00")
    monkeypatch.setattr(spc.db, "get_cursor_updated_at", lambda _db, _r: "2026-09-01T00:00:00+00:00")
    monkeypatch.setattr(spc.db, "save_drift_signals", lambda _db, rows: saved.extend(rows))
    spc.check_context_freshness(object(), "run")
    assert saved == []


def test_context_alert_fires_when_feed_stalls_during_competition(monkeypatch) -> None:
    saved = []
    monkeypatch.setattr(spc.db, "latest_context_at", lambda _db: "2026-09-12T00:00:00+00:00")
    monkeypatch.setattr(spc.db, "get_cursor_updated_at", lambda _db, _r: "2026-09-01T00:00:00+00:00")
    monkeypatch.setattr(spc.db, "save_drift_signals", lambda _db, rows: saved.extend(rows))
    spc.check_context_freshness(object(), "run")
    assert len(saved) == 1 and saved[0]["triggered"] is True


# ------------------------------------------------------------------ MLflow
import pulso_pipeline.tracking as tracking  # noqa: E402


def test_tracking_is_a_no_op_without_server(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.log_training(report=None, params_by_h=None, version="v", decision="keep", reason="",
                                 git_commit="x", data_cutoff="c", joblib_bytes=b"model") is None


def test_tracking_failure_never_breaks_training(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # nadie escucha
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
    assert tracking.log_training(report=None, params_by_h=None, version="v", decision="keep", reason="",
                                 git_commit="x", data_cutoff="c") is None


def test_tracking_flattens_params_and_metrics() -> None:
    params = tracking._flat_params({1: {"hgb": {"learning_rate": 0.05}, "w_hgb": 0.6}})
    assert params == {"h1.hgb.learning_rate": 0.05, "h1.w_hgb": 0.6}
    report = {"horizons": {"h1": {"model": 88.0, "por_estacion": {"02300": 86.5},
                                  "same_block": {"model": 87.5, "reference": 87.7}},
                           "h2": {"model": 86.0}}}
    m = tracking._metrics(report)
    assert m["accuracy"] == 87.0 and m["h1.station_02300"] == 86.5 and m["h1.same_block.champion"] == 87.7


# --------------------------------------------- reentrenamiento automático en sombra
from datetime import datetime as _dt, timedelta as _td, timezone as _tz  # noqa: E402

import pandas as _pd  # noqa: E402

import pulso_pipeline.shadow as sh  # noqa: E402

_NOW = _dt(2026, 10, 1, tzinfo=_tz.utc)


def test_trigger_when_recent_rank_is_three_below_cumulative() -> None:
    assert sh.should_trigger(2, 5, False, None, _NOW)[0] is True
    assert sh.should_trigger(2, 4, False, None, _NOW)[0] is False


def test_trigger_blocked_by_pending_shadow_or_cooldown() -> None:
    assert sh.should_trigger(2, 6, True, None, _NOW)[0] is False
    assert sh.should_trigger(2, 6, False, _NOW - _td(minutes=30), _NOW)[0] is False
    assert sh.should_trigger(2, 6, False, _NOW - _td(hours=3), _NOW)[0] is True
    assert sh.should_trigger(None, 6, False, None, _NOW)[0] is False


def test_challenge_uses_the_competition_metric_per_station() -> None:
    f = _pd.DataFrame({"station_id": ["a", "a", "b", "b"], "y": [100, 100, 10, 10],
                       "champion": [90, 110, 5, 15], "shadow": [100, 100, 9, 11]})
    champ, shadow = sh.challenge_accuracy(f)
    assert champ == 70.0 and shadow == 95.0  # a: 90 vs 100, b: 50 vs 90, promedio por estación


def test_decide_promotes_keeps_and_rolls_back() -> None:
    assert sh.decide(80.0, 82.0, shadow_is_previous_champion=False) == ("promote", "shadow")
    assert sh.decide(82.0, 80.0, shadow_is_previous_champion=False) == ("discard", "rejected")
    assert sh.decide(80.0, 82.0, shadow_is_previous_champion=True) == ("promote", "rejected")
    assert sh.decide(82.0, 80.0, shadow_is_previous_champion=True) == ("discard", "retired")

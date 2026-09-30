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

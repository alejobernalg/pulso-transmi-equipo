import hashlib

import httpx
import pytest

from pulso_pipeline.submit_current_cycle import _predictions_hash
from pulso_transmi import PulsoTransmiApiError, PulsoTransmiClient


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

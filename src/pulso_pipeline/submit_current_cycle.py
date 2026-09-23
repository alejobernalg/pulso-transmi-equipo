"""Workflow de inferencia y entrega: sincroniza, consulta el ciclo, infiere y
entrega. Sigue el flujo de la guía operativa v2.0 (pág. 7). Se ejecuta cada
10 minutos desde `.github/workflows/predict.yml`; despertar tres veces no
significa enviar tres veces (guarda de idempotencia + recibo en Supabase).

Uso:
    python -m pulso_pipeline.submit_current_cycle [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO

import joblib

from pulso_forecast import forecast_for_targets, wide_from_frames
from pulso_transmi import PulsoTransmiApiError, PulsoTransmiClient

from . import db

STEP_MIN = 15
RETRYABLE_MAX_ATTEMPTS = 3


def git_commit() -> str | None:
    import os

    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def _synthesize_cursor(fields: list) -> str:
    """Reconstruye un cursor con la misma codificación observada en la API
    (base64 urlsafe de un JSON array de campos clave, sin padding).

    La API nunca entrega un `next_cursor` en la última página de un drenado
    (es null por construcción: "no hay más *ahora mismo*"). En operación
    normal (ticks de 10 min) el backlog nuevo casi siempre cabe en una sola
    página, así que sin esto el cursor guardado nunca avanzaría y cada corrida
    repaginaría el stream completo desde el principio -- barato hoy, pero cada
    vez más caro conforme crece el historial de la competencia. Se verificó
    empíricamente (2026-09-22) que reanudar con un cursor así construido
    devuelve exactamente lo esperado (0 filas si no hay nada nuevo). Es un
    mecanismo no documentado por la API: si cambia el formato, esto deja de
    avanzar el cursor y se degrada a repaginar desde el último cursor válido
    conocido (sigue siendo correcto, solo menos eficiente).
    """
    return base64.urlsafe_b64encode(json.dumps(fields).encode()).decode().rstrip("=")


def _sync_paginated(database, resource: str, fetch_page, upsert, resume_key) -> int:
    """Pagina desde el cursor guardado y solo lo avanza tras un upsert confirmado."""
    cursor = db.get_cursor(database, resource)
    total = 0
    last_row: dict | None = None
    while True:
        page = fetch_page(cursor)
        rows = page["data"]
        upsert(rows)
        total += len(rows)
        if rows:
            last_row = rows[-1]
        next_cursor = page.get("next_cursor")
        if next_cursor is not None:
            cursor = next_cursor
            db.set_cursor(database, resource, cursor)
            continue
        if last_row is not None:
            db.set_cursor(database, resource, _synthesize_cursor(resume_key(last_row)))
        return total


def sync_observations(client: PulsoTransmiClient, database) -> int:
    return _sync_paginated(
        database, "observations_stream",
        lambda cursor: client.stream_observations_page(cursor=cursor, limit=5000),
        lambda rows: db.upsert_observations(database, rows),
        lambda row: [row["released_at"], row["observed_at"], row["station_id"]],
    )


def sync_context(client: PulsoTransmiClient, database) -> int:
    return _sync_paginated(
        database, "context",
        lambda cursor: client.context_page(cursor=cursor, limit=5000),
        lambda rows: db.upsert_context(database, rows),
        lambda row: [row["observed_at"]],
    )


def _predictions_hash(payload: list[dict]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="arma y valida el batch, no lo envía")
    args = parser.parse_args(argv)

    database = db.get_client()
    commit = git_commit()
    run_id = db.start_run(database, commit or "unknown")

    try:
        with PulsoTransmiClient() as client:
            n_obs = sync_observations(client, database)
            n_ctx = sync_context(client, database)
            print(f"sync: {n_obs} observaciones, {n_ctx} periodos de contexto")

            cycle = client.current_cycle()
            if cycle is None:
                print("no hay ciclo abierto (404 no_open_cycle): fin en verde")
                db.finish_run(database, run_id, status="success", decision="keep",
                              decision_reason="no_open_cycle")
                return 0

            model_row = db.get_active_model(database)
            if model_row is None:
                raise db.PipelineDBError(
                    "no hay modelo activo en model_versions; correr train_and_promote primero"
                )

            if db.receipt_exists(database, cycle["cycle_id"], model_row["model_id"]):
                print(f"ya existe recibo para {cycle['cycle_id']} / {model_row['version']}: fin en verde")
                db.finish_run(database, run_id, status="success", decision="keep",
                              decision_reason="receipt_exists")
                return 0

            bundle = joblib.load(BytesIO(db.download_model(database, model_row["artifact_uri"])))

            obs_df, ctx_df, stations_df = db.fetch_history(database)
            y, ctx, stations = wide_from_frames(obs_df, ctx_df, stations_df)
            cutoff = cycle["data_cutoff"]
            y = y.loc[:cutoff]
            ctx = ctx.loc[:cutoff]

            targets = [(t["station_id"], t["target_at"]) for t in cycle["targets"]]
            preds = forecast_for_targets(bundle["models"], y, ctx, stations, cutoff, targets)

            if len(preds) != cycle["expected_predictions"]:
                raise ValueError(
                    f"se esperaban {cycle['expected_predictions']} predicciones, se generaron {len(preds)}"
                )
            expected_pairs = {(t["station_id"], t["target_at"]) for t in cycle["targets"]}
            got_pairs = {(r.station_id, r.target_at.isoformat().replace("+00:00", "Z")) for r in preds.itertuples()}
            if got_pairs != expected_pairs:
                raise ValueError("los pares (station_id, target_at) no coinciden exactamente con los targets")
            if not all(math.isfinite(v) and v >= 0 for v in preds["value"]):
                raise ValueError("hay valores no finitos o negativos en las predicciones")

            predictions_payload = [
                {
                    "station_id": r.station_id,
                    "target_at": r.target_at.isoformat().replace("+00:00", "Z"),
                    "value": float(r.value),
                }
                for r in preds.sort_values(["station_id", "target_at"]).itertuples()
            ]
            preds_hash = _predictions_hash(predictions_payload)
            idempotency_key = hashlib.sha256(
                f"{cycle['cycle_id']}:{model_row['version']}:{preds_hash}".encode()
            ).hexdigest()

            model_trace = {
                "version": model_row["version"],
                "trained_at": model_row.get("created_at"),
                "training_data_end": model_row.get("train_cutoff"),
                "git_commit": model_row.get("git_commit"),
            }

            if args.dry_run:
                print("DRY RUN: no se envía. Payload:")
                print(json.dumps({
                    "cycle_id": cycle["cycle_id"],
                    "client_run_id": run_id,
                    "data_cutoff": cutoff,
                    "model": model_trace,
                    "idempotency_key": idempotency_key,
                    "n_predictions": len(predictions_payload),
                    "sample": predictions_payload[:4],
                }, indent=2, default=str))
                db.finish_run(database, run_id, status="success", decision="keep",
                              decision_reason="dry_run")
                return 0

            receipt = None
            last_error: Exception | None = None
            for attempt in range(1, RETRYABLE_MAX_ATTEMPTS + 1):
                try:
                    receipt = client.submit(
                        cycle_id=cycle["cycle_id"],
                        client_run_id=run_id,
                        data_cutoff=cutoff,
                        model=model_trace,
                        predictions=predictions_payload,
                        idempotency_key=idempotency_key,
                    )
                    break
                except PulsoTransmiApiError as exc:
                    last_error = exc
                    if exc.status_code in (409, 422):
                        raise
                    if attempt == RETRYABLE_MAX_ATTEMPTS:
                        raise
                    print(f"intento {attempt} falló ({exc.status_code}), reintentando con la misma llave")
                    time.sleep(2 ** attempt)
            if receipt is None:
                raise last_error or RuntimeError("no se pudo enviar la submission")

            now = datetime.now(timezone.utc).isoformat()
            db.save_receipt(
                database,
                run_id=run_id,
                cycle_id=cycle["cycle_id"],
                model_id=model_row["model_id"],
                attempt=receipt.get("attempt", 1),
                idempotency_key=idempotency_key,
                predictions_hash=preds_hash,
                http_status=200,
                submission_id=receipt.get("submission_id"),
                status=receipt.get("status", "accepted"),
                accepted_at=now,
            )
            db.save_predictions(database, [
                {
                    "run_id": run_id,
                    "model_id": model_row["model_id"],
                    "station_id": r.station_id,
                    "issued_at": cutoff,
                    "target_at": r.target_at.isoformat(),
                    "horizon_steps": int(r.horizon_steps),
                    "y_pred": float(r.value),
                    "submitted_at": now,
                    "submission_status": receipt.get("status", "accepted"),
                    "cycle_id": cycle["cycle_id"],
                }
                for r in preds.itertuples()
            ])
            db.finish_run(database, run_id, status="success", decision="keep",
                          decision_reason="submitted", data_cutoff=cutoff)
            print(f"entregado: {cycle['cycle_id']} -> {receipt.get('submission_id')}")
            return 0
    except Exception as exc:  # noqa: BLE001 - se registra y se re-lanza como fallo del job
        db.finish_run(database, run_id, status="failed", failed_stage="predict", error_message=str(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())

"""Envoltorio delgado sobre Supabase: única fuente de estado entre corridas.

Todo lo que debe sobrevivir entre despertares del workflow (cursores, modelo
activo, recibos, predicciones, métricas, señales de drift) vive aquí. Usa
siempre la service key: las tablas tienen RLS sin policies, así que `anon` no
puede leer ni escribir nada (ver docs/data-model.md).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from supabase import Client, create_client

PAGE_SIZE = 5000
MODEL_BUCKET = "models"


class PipelineDBError(RuntimeError):
    """El estado esperado no está en Supabase (p. ej. no hay modelo activo)."""


def get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


def _paginate(db: Client, table: str, *, select: str = "*", order: str,
              filters: list[tuple[str, str, Any]] | None = None) -> list[dict]:
    """Pagina con `.range()` avanzando por filas realmente recibidas, no por
    PAGE_SIZE: PostgREST puede aplicar su propio tope (`db-max-rows`, 1000 por
    defecto en Supabase) sin avisar, y ese tope puede ser menor al pedido."""
    rows: list[dict] = []
    offset = 0
    while True:
        query = db.table(table).select(select).order(order)
        for column, op, value in filters or []:
            query = getattr(query, op)(column, value)
        page = query.range(offset, offset + PAGE_SIZE - 1).execute().data
        if not page:
            return rows
        rows.extend(page)
        offset += len(page)


# ------------------------------------------------------------------- cursores
def get_cursor(db: Client, resource: str) -> str | None:
    rows = db.table("sync_state").select("cursor").eq("resource", resource).execute().data
    return rows[0]["cursor"] if rows else None


def set_cursor(db: Client, resource: str, cursor: str | None) -> None:
    db.table("sync_state").upsert(
        {"resource": resource, "cursor": cursor, "updated_at": datetime.now(timezone.utc).isoformat()}
    ).execute()


# --------------------------------------------------------------- datos crudos
def upsert_observations(db: Client, rows: list[dict]) -> None:
    if not rows:
        return
    payload = [{"observed_at": r["observed_at"], "station_id": r["station_id"], "demand": r["demand"]} for r in rows]
    db.table("observations").upsert(payload, on_conflict="observed_at,station_id").execute()


def upsert_context(db: Client, rows: list[dict]) -> None:
    if not rows:
        return
    cols = ["observed_at", "rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity"]
    payload = [{c: r[c] for c in cols} for r in rows]
    db.table("context").upsert(payload, on_conflict="observed_at").execute()


def fetch_stations(db: Client) -> pd.DataFrame:
    rows = db.table("stations").select("*").execute().data
    return pd.DataFrame(rows)


def upsert_stations(db: Client, rows: list[dict]) -> None:
    if not rows:
        return
    cols = ["station_id", "station_name", "corridor", "latitude", "longitude"]
    payload = [{c: r[c] for c in cols} for r in rows]
    db.table("stations").upsert(payload, on_conflict="station_id").execute()


def count_observations(db: Client) -> int:
    result = db.table("observations").select("station_id", count="exact").limit(1).execute()
    return result.count or 0


def fetch_history(db: Client) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Toda la historia disponible en Supabase, lista para `pulso_forecast.wide_from_frames`."""
    obs = pd.DataFrame(_paginate(db, "observations", order="observed_at"))
    ctx = pd.DataFrame(_paginate(db, "context", order="observed_at"))
    stations = fetch_stations(db)
    return obs, ctx, stations


# ------------------------------------------------------------------- modelos
def get_active_model(db: Client) -> dict[str, Any] | None:
    rows = db.table("model_versions").select("*").eq("is_active", True).execute().data
    return rows[0] if rows else None


def upload_model(db: Client, version: str, data: bytes) -> str:
    path = f"{version}/model.joblib"
    db.storage.from_(MODEL_BUCKET).upload(
        path, data, {"content-type": "application/octet-stream", "upsert": "true"}
    )
    return path


def download_model(db: Client, artifact_uri: str) -> bytes:
    return db.storage.from_(MODEL_BUCKET).download(artifact_uri)


def insert_model_version(db: Client, **fields: Any) -> str:
    row = db.table("model_versions").insert(fields).execute().data[0]
    return row["model_id"]


def promote_model(db: Client, model_id: str) -> None:
    db.table("model_versions").update({"is_active": False}).eq("is_active", True).execute()
    db.table("model_versions").update({"is_active": True}).eq("model_id", model_id).execute()


# --------------------------------------------------------------------- runs
def start_run(db: Client, git_commit: str) -> str:
    row = db.table("pipeline_runs").insert({"git_commit": git_commit, "status": "running"}).execute().data[0]
    return row["run_id"]


def finish_run(db: Client, run_id: str, *, status: str, failed_stage: str | None = None,
               error_message: str | None = None, data_cutoff: str | None = None,
               decision: str | None = None, decision_reason: str | None = None) -> None:
    db.table("pipeline_runs").update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "failed_stage": failed_stage,
        "error_message": error_message,
        "data_cutoff": data_cutoff,
        "decision": decision,
        "decision_reason": decision_reason,
    }).eq("run_id", run_id).execute()


# ------------------------------------------------------------------- recibos
def receipt_exists(db: Client, cycle_id: str, model_id: str) -> bool:
    rows = (
        db.table("submission_receipts")
        .select("receipt_id")
        .eq("cycle_id", cycle_id)
        .eq("model_id", model_id)
        .execute()
        .data
    )
    return bool(rows)


def save_receipt(db: Client, **fields: Any) -> None:
    db.table("submission_receipts").upsert(fields, on_conflict="cycle_id,model_id").execute()


def save_predictions(db: Client, rows: list[dict]) -> None:
    if rows:
        db.table("predictions").insert(rows).execute()


def save_metrics(db: Client, rows: list[dict]) -> None:
    if rows:
        db.table("model_metrics").insert(rows).execute()


def save_drift_signals(db: Client, rows: list[dict]) -> None:
    if rows:
        db.table("drift_signals").insert(rows).execute()

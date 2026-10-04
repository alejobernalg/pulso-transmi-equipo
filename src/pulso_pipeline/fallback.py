"""Envío de emergencia: si el camino normal falla, el ciclo igual se entrega.

Un ciclo sin enviar cuenta como error total en el leaderboard; una predicción
estacional simple cuesta unos pocos puntos. Por eso esta ruta no depende de
Supabase ni del modelo entrenado: lee la demanda directo del stream de la API y
predice `y(t-1d)`/`y(t-7d)` escalado al nivel de las últimas 2 h.

Si el camino normal se arregla en la siguiente corrida, su envío (con otra
`Idempotency-Key`) reemplaza a este.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pulso_transmi import PulsoTransmiClient

from . import db

FALLBACK_VERSION = "fallback-seasonal-1"
DAY = pd.Timedelta(days=1)
LEVEL_WINDOW = 8  # periodos (2 h) para estimar el nivel actual
RATIO_CLIP = (0.3, 3.0)


def stream_history(client: PulsoTransmiClient) -> pd.DataFrame:
    """Toda la demanda liberada en el stream, en formato ancho (sin Supabase)."""
    rows, cursor = [], None
    while True:
        page = client.stream_observations_page(cursor=cursor, limit=5000)
        for r in page["data"]:
            try:
                demand = db.observation_demand(r)
            except (KeyError, TypeError, ValueError):
                continue
            if demand is not None:
                rows.append((pd.Timestamp(r["observed_at"]), r["station_id"], float(demand)))
        cursor = page.get("next_cursor")
        if not cursor:
            break
    frame = pd.DataFrame(rows, columns=["observed_at", "station_id", "demand"])
    y = frame.pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last")
    return y.sort_index()


def seasonal_forecast(y: pd.DataFrame, targets: list[tuple[str, str]]) -> list[dict]:
    """`base(t) = media(y(t-1d), y(t-7d))`, escalada por real/base de las últimas 2 h."""
    def base(station: str, when: pd.Timestamp) -> float:
        vals = [y.at[when - lag, station] for lag in (DAY, 7 * DAY)
                if (when - lag) in y.index and np.isfinite(y.at[when - lag, station])]
        return float(np.mean(vals)) if vals else np.nan

    out = []
    for station, target_at in targets:
        series = y[station].dropna()
        last = float(series.iloc[-1]) if len(series) else 0.0
        recent = series.iloc[-LEVEL_WINDOW:]
        ref = np.array([base(station, t) for t in recent.index])
        ok = np.isfinite(ref)
        ratio = recent.to_numpy()[ok].sum() / ref[ok].sum() if ok.any() and ref[ok].sum() > 0 else 1.0
        ratio = float(np.clip(ratio, *RATIO_CLIP))
        b = base(station, pd.Timestamp(target_at))
        value = b * ratio if np.isfinite(b) else last
        out.append({"station_id": station, "target_at": target_at, "value": max(0.0, float(value))})
    return sorted(out, key=lambda p: (p["station_id"], p["target_at"]))


def emergency_submit(reason: str) -> bool:
    """Entrega el ciclo abierto con el pronóstico estacional. Devuelve si envió."""
    with PulsoTransmiClient() as client:
        cycle = client.current_cycle()
        if cycle is None:
            print("respaldo: no hay ciclo abierto")
            return False
        try:
            database = db.get_client()
            if database.table("submission_receipts").select("cycle_id").eq(
                    "cycle_id", cycle["cycle_id"]).limit(1).execute().data:
                print(f"respaldo: {cycle['cycle_id']} ya tiene recibo, no se reenvía")
                return False
        except Exception as exc:  # noqa: BLE001 - sin Supabase no se sabe; mejor enviar que perder el ciclo
            print(f"respaldo: Supabase no disponible ({exc}); se envía igual")

        y = stream_history(client)
        cutoff = pd.Timestamp(cycle["data_cutoff"])
        y = y.loc[:cutoff]
        targets = [(t["station_id"], t["target_at"]) for t in cycle["targets"]]
        predictions = seasonal_forecast(y, targets)
        digest = hashlib.sha256(json.dumps(predictions, sort_keys=True).encode()).hexdigest()
        receipt = client.submit(
            cycle_id=cycle["cycle_id"],
            client_run_id=str(uuid.uuid4()),
            data_cutoff=cycle["data_cutoff"],
            model={"version": FALLBACK_VERSION, "trained_at": None,
                   "training_data_end": y.index[-1].isoformat(), "git_commit": os.environ.get("GITHUB_SHA")},
            predictions=predictions,
            idempotency_key=hashlib.sha256(f"{cycle['cycle_id']}:fallback:{digest}".encode()).hexdigest(),
        )
        print(f"respaldo ENTREGADO: {cycle['cycle_id']} -> {receipt.get('submission_id')} (causa: {reason})")
        _mark_degraded(cycle["cycle_id"], reason)
        return True


def _mark_degraded(cycle_id: str, reason: str) -> None:
    """Deja rastro para que el workflow abra la alerta aunque el job termine en verde."""
    with open("degraded.txt", "w") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()} {cycle_id}: {reason}\n")

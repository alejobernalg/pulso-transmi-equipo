"""Workflow de entrenamiento y promoción, separado del de inferencia (guía
operativa v2.0, pág. 4 y 9). Se dispara manualmente (`workflow_dispatch`);
no reentrena en cada despertar del cron de predicción.

Uso:
    python -m pulso_pipeline.train_and_promote [--dry-run] [--tolerance 0.5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pulso_forecast import (
    HORIZONS,
    feature_columns,
    make_frame,
    run_from_frames,
    train_production_from_frames,
    wide_from_frames,
)
from pulso_transmi import PulsoTransmiClient

from . import db
from .submit_current_cycle import git_commit

MIN_BOOTSTRAP_ROWS = 40_000
DRIFT_THRESHOLD_POINTS = 5.0
DRIFT_MIN_SAMPLES = 20


def ensure_bootstrapped(client: PulsoTransmiClient, database) -> None:
    """Si Supabase todavía no tiene la historia completa, la carga desde la API.

    Cubre el caso de un repo de equipo nuevo sin la carga inicial hecha a mano.
    """
    if db.count_observations(database) >= MIN_BOOTSTRAP_ROWS:
        return
    print("Supabase tiene poca historia: bootstrap desde la API")
    stations = client.stations()
    db.upsert_stations(database, stations.to_dict("records"))
    obs = client.observations_dataframe()
    for start in range(0, len(obs), 5000):
        chunk = obs.iloc[start:start + 5000]
        db.upsert_observations(database, [
            {"observed_at": r.observed_at.isoformat(), "station_id": r.station_id, "demand": int(r.demand)}
            for r in chunk.itertuples()
        ])
    ctx = client.context_dataframe()
    for start in range(0, len(ctx), 5000):
        chunk = ctx.iloc[start:start + 5000]
        db.upsert_context(database, [
            {
                "observed_at": r.observed_at.isoformat(), "rain_mm": float(r.rain_mm),
                "rain_forecast": float(r.rain_forecast), "temperature_c": float(r.temperature_c),
                "temperature_forecast": float(r.temperature_forecast), "event_intensity": float(r.event_intensity),
            }
            for r in chunk.itertuples()
        ])
    print(f"bootstrap: {len(obs)} observaciones, {len(ctx)} periodos de contexto")


def champion_baseline_accuracy(database, model_id: str) -> float | None:
    rows = (
        database.table("model_metrics")
        .select("value")
        .eq("model_id", model_id)
        .eq("split", "validation")
        .eq("metric_name", "accuracy")
        .is_("station_id", "null")
        .execute()
        .data
    )
    if not rows:
        return None
    return float(pd.Series([r["value"] for r in rows]).mean())


def check_drift(database, active_model_id: str, baseline_accuracy: float | None, run_id: str) -> None:
    """Compara accuracy reciente (predicciones ya entregadas y ya observadas)
    contra la accuracy de validación del champion, y deja evidencia en
    `model_metrics` (split='live') aunque no dispare drift. Heurística simple
    (no PSI multivariado): un chequeo honesto de degradación, no un detector
    completo. La cobertura aquí es local (sobre las últimas 500 predicciones
    de este modelo), no la oficial de la plataforma (`/v1/leaderboard`).
    """
    preds = (
        database.table("predictions")
        .select("station_id,target_at,y_pred")
        .eq("model_id", active_model_id)
        .order("target_at", desc=True)
        .limit(500)
        .execute()
        .data
    )
    if not preds:
        return
    targets = sorted({p["target_at"] for p in preds})
    obs_rows = (
        database.table("observations")
        .select("station_id,observed_at,demand")
        .in_("observed_at", targets)
        .execute()
        .data
    )
    actual = {(r["station_id"], r["observed_at"]): r["demand"] for r in obs_rows}
    matched = [(p["y_pred"], actual[(p["station_id"], p["target_at"])])
               for p in preds if (p["station_id"], p["target_at"]) in actual]
    coverage = len(matched) / len(preds)
    now = datetime.now(timezone.utc).isoformat()
    metric_rows = [{
        "run_id": run_id, "model_id": active_model_id, "station_id": None, "split": "live",
        "metric_name": "coverage_local", "window_label": "last_500", "value": round(coverage, 4),
        "computed_at": now,
    }]
    if len(matched) >= DRIFT_MIN_SAMPLES:
        err = sum(abs(y - p) for p, y in matched)
        tot = sum(y for _, y in matched)
        recent_accuracy = 100 * max(0.0, 1 - err / tot) if tot else 0.0
        metric_rows.append({
            "run_id": run_id, "model_id": active_model_id, "station_id": None, "split": "live",
            "metric_name": "accuracy", "window_label": "recent", "value": round(recent_accuracy, 4),
            "computed_at": now,
        })
        print(f"live: accuracy reciente {recent_accuracy:.2f}, cobertura local {coverage:.0%} "
              f"({len(matched)}/{len(preds)})")
        if baseline_accuracy is not None:
            drop = baseline_accuracy - recent_accuracy
            db.save_drift_signals(database, [{
                "run_id": run_id,
                "station_id": None,
                "kind": "concept",
                "feature": "accuracy",
                "method": "wape_delta",
                "statistic": round(drop, 4),
                "threshold": DRIFT_THRESHOLD_POINTS,
                "triggered": drop > DRIFT_THRESHOLD_POINTS,
            }])
            print(f"drift: caída de {drop:.2f} pts vs baseline {baseline_accuracy:.2f} "
                  f"(umbral {DRIFT_THRESHOLD_POINTS})")
    else:
        print(f"live: cobertura local {coverage:.0%} ({len(matched)}/{len(preds)}), "
              f"insuficiente para accuracy/drift (mínimo {DRIFT_MIN_SAMPLES})")
    db.save_metrics(database, metric_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="evalúa y decide, no escribe en Supabase")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="puntos de accuracy que el candidato puede perder frente al champion sin bloquear la promoción")
    args = parser.parse_args(argv)

    database = db.get_client()
    commit = git_commit() or "unknown"
    run_id = db.start_run(database, commit)

    try:
        with PulsoTransmiClient() as client:
            ensure_bootstrapped(client, database)
            obs_df, ctx_df, stations_df = db.fetch_history(database)
            y, ctx, stations = wide_from_frames(obs_df, ctx_df, stations_df)

            with tempfile.TemporaryDirectory() as tmp:
                report = run_from_frames(y, ctx, stations, Path(tmp), horizons=HORIZONS)

            candidate_accuracy = float(pd.Series(
                [report["horizons"][f"h{h}"]["model"] for h in HORIZONS]
            ).mean())

            champion = db.get_active_model(database)
            champion_accuracy = champion_baseline_accuracy(database, champion["model_id"]) if champion else None

            if champion is None:
                promote, reason = True, "bootstrap: primer modelo activo"
            elif champion_accuracy is None:
                promote, reason = True, "champion activo sin métricas de validación registradas"
            elif candidate_accuracy >= champion_accuracy - args.tolerance:
                promote = True
                reason = f"candidato {candidate_accuracy:.2f} vs champion {champion_accuracy:.2f} (tolerancia {args.tolerance})"
            else:
                promote = False
                reason = f"candidato {candidate_accuracy:.2f} < champion {champion_accuracy:.2f} - tolerancia {args.tolerance}"

            print(f"candidato: accuracy media {candidate_accuracy:.2f} | decisión: "
                  f"{'promover' if promote else 'conservar champion'} ({reason})")

            if args.dry_run:
                db.finish_run(database, run_id, status="success", decision="keep" if not promote else "retrain",
                              decision_reason=f"dry_run: {reason}", data_cutoff=str(y.index[-1]))
                return 0

            if promote:
                params_by_h = {h: report["horizons"][f"h{h}"]["best_params"] for h in HORIZONS}
                with tempfile.TemporaryDirectory() as tmp:
                    train_production_from_frames(y, ctx, stations, Path(tmp), params_by_h, horizons=HORIZONS)
                    joblib_bytes = (Path(tmp) / "model.joblib").read_bytes()

                sample_features = sorted(feature_columns(make_frame(y, ctx, stations, HORIZONS[0])))
                features_hash = hashlib.sha256(json.dumps(sample_features).encode()).hexdigest()
                version = f"pulso-forecast-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                artifact_uri = db.upload_model(database, version, joblib_bytes)

                model_id = db.insert_model_version(
                    database,
                    version=version,
                    trained_in_run_id=run_id,
                    parent_model_id=champion["model_id"] if champion else None,
                    git_commit=commit,
                    algorithm="HistGradientBoostingRegressor",
                    params={str(h): p for h, p in params_by_h.items()},
                    feature_list=sample_features,
                    features_hash=features_hash,
                    train_cutoff=str(y.index[-1]),
                    artifact_uri=artifact_uri,
                    retrain_reason=reason,
                    is_active=False,
                )

                now = datetime.now(timezone.utc).isoformat()
                metric_rows = []
                for h in HORIZONS:
                    row = report["horizons"][f"h{h}"]
                    metric_rows.append({
                        "run_id": run_id, "model_id": model_id, "station_id": None, "split": "validation",
                        "metric_name": "accuracy", "window_label": f"h{h}", "value": row["model"], "computed_at": now,
                    })
                    for station_id, acc in row["por_estacion"].items():
                        metric_rows.append({
                            "run_id": run_id, "model_id": model_id, "station_id": station_id, "split": "validation",
                            "metric_name": "accuracy", "window_label": f"h{h}", "value": acc, "computed_at": now,
                        })
                db.save_metrics(database, metric_rows)
                db.promote_model(database, model_id)
                print(f"promovido: {version} ({model_id})")

            if champion is not None:
                check_drift(database, champion["model_id"], champion_accuracy, run_id)

            db.finish_run(database, run_id, status="success", decision="retrain" if promote else "keep",
                          decision_reason=reason, data_cutoff=str(y.index[-1]))
            return 0
    except Exception as exc:  # noqa: BLE001
        db.finish_run(database, run_id, status="failed", failed_stage="train", error_message=str(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())

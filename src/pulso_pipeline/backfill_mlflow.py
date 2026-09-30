"""Carga en MLflow los modelos que ya existían en Supabase (model_versions + bucket
`models`), en orden de creación, con sus métricas de validación. Idempotente: un
modelo ya registrado (tag `supabase_model_id`) se salta. El alias `champion`
queda en el modelo activo.

Uso:
    python -m pulso_pipeline.backfill_mlflow
"""
from __future__ import annotations

import sys
from collections import defaultdict

from . import db, tracking


def _report(database, model_id: str) -> dict:
    rows = (database.table("model_metrics").select("station_id,window_label,value")
            .eq("model_id", model_id).eq("split", "validation").eq("metric_name", "accuracy")
            .execute().data)
    horizons: dict = defaultdict(lambda: {"por_estacion": {}})
    for r in rows:
        if r["station_id"] is None:
            horizons[r["window_label"]]["model"] = r["value"]
        else:
            horizons[r["window_label"]]["por_estacion"][r["station_id"].strip()] = r["value"]
    return {"horizons": {k: v for k, v in horizons.items() if "model" in v}}


def main() -> int:
    if not tracking.enabled():
        print("MLFLOW_TRACKING_URI no está definido")
        return 1
    from mlflow import MlflowClient

    database = db.get_client()
    client = MlflowClient()
    try:
        done = {mv.tags.get("supabase_model_id") for mv in client.search_model_versions(f"name='{tracking.MODEL_NAME}'")}
    except Exception:  # noqa: BLE001 - el modelo registrado todavía no existe
        done = set()

    models = database.table("model_versions").select("*").order("created_at").execute().data
    for m in models:
        if m["model_id"] in done:
            print(f"ya registrado: {m['version']}")
            continue
        joblib_bytes = db.download_model(database, m["artifact_uri"])
        tracking.log_training(
            report=_report(database, m["model_id"]), params_by_h=m["params"], version=m["version"],
            decision="retrain", reason=m["retrain_reason"] or "", git_commit=m["git_commit"] or "unknown",
            data_cutoff=m["train_cutoff"], train_cutoff=m["train_cutoff"], supabase_model_id=m["model_id"],
            joblib_bytes=joblib_bytes, promote=m["is_active"], trained_at=m["created_at"], backfill=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

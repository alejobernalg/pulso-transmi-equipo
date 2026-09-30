"""Versionado de modelos en MLflow (servidor de DagsHub).

Cada corrida de entrenamiento queda como un run (parámetros, métricas de
validación, comparación contra el champion, commit y corte de datos). Los
modelos promovidos se registran en el Model Registry como `pulso-forecast`
y el alias `champion` apunta siempre al que está activo.

Supabase sigue siendo la fuente del pipeline de predicción; MLflow es un
espejo para trazabilidad, así que es opcional: sin `MLFLOW_TRACKING_URI` no
hace nada, y si falla solo avisa, nunca tumba el entrenamiento.

Variables: `MLFLOW_TRACKING_URI` (https://dagshub.com/<usuario>/<repo>.mlflow),
`MLFLOW_TRACKING_USERNAME` (usuario de DagsHub) y `MLFLOW_TRACKING_PASSWORD`
(token de DagsHub).
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_NAME = "pulso-forecast"
EXPERIMENT = "pulso-transmi-train"
CHAMPION_ALIAS = "champion"


def enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def _flat_params(params_by_h: dict) -> dict[str, Any]:
    """{"h1.hgb.learning_rate": 0.05, ...}: MLflow solo guarda parámetros planos."""
    out: dict[str, Any] = {}
    for h, p in params_by_h.items():
        for leg, value in p.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    out[f"h{h}.{leg}.{k}"] = v
            else:
                out[f"h{h}.{leg}"] = value
    return out


def _metrics(report: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    accs = []
    for key, row in report["horizons"].items():
        metrics[f"{key}.accuracy"] = row["model"]
        accs.append(row["model"])
        for name in ("naive_day", "naive_week"):
            if name in row:
                metrics[f"{key}.{name}"] = row[name]
        for sid, acc in row.get("por_estacion", {}).items():
            metrics[f"{key}.station_{sid}"] = acc
        if "same_block" in row:
            metrics[f"{key}.same_block.candidate"] = row["same_block"]["model"]
            metrics[f"{key}.same_block.champion"] = row["same_block"]["reference"]
    if accs:
        metrics["accuracy"] = sum(accs) / len(accs)
    return metrics


def log_training(*, report: dict | None, params_by_h: dict | None, version: str | None, decision: str,
                 reason: str, git_commit: str, data_cutoff: str, train_cutoff: str | None = None,
                 supabase_model_id: str | None = None, joblib_bytes: bytes | None = None,
                 promote: bool = False, trained_at: str | None = None, backfill: bool = False,
                 extra_metrics: dict[str, float] | None = None) -> str | None:
    """Registra una corrida de entrenamiento. Si trae `joblib_bytes`, sube el modelo y crea
    una versión en el registry; con `promote`, además le mueve el alias `champion`.
    Devuelve el número de versión registrada (o None)."""
    if not enabled():
        return None
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=version or f"sin-promover-{data_cutoff}") as run:
            mlflow.set_tags({
                "decision": decision, "reason": reason, "git_commit": git_commit,
                "data_cutoff": data_cutoff, "train_cutoff": train_cutoff or "",
                "model_version": version or "", "supabase_model_id": supabase_model_id or "",
                "promoted": str(promote).lower(), "backfill": str(backfill).lower(),
                "trained_at": trained_at or datetime.now(timezone.utc).isoformat(),
            })
            if params_by_h:
                mlflow.log_params(_flat_params(params_by_h))
            metrics = _metrics(report) if report else {}
            metrics.update(extra_metrics or {})
            if metrics:
                mlflow.log_metrics(metrics)
            if joblib_bytes is None:
                return None
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "model.joblib").write_bytes(joblib_bytes)
                mlflow.log_artifacts(tmp, artifact_path="model")

            client = MlflowClient()
            try:
                client.get_registered_model(MODEL_NAME)
            except Exception:  # noqa: BLE001 - primera vez: no existe todavía
                client.create_registered_model(
                    MODEL_NAME, description="Pronóstico de demanda Pulso TransMi (HGB+LightGBM por horizonte)")
            mv = client.create_model_version(
                MODEL_NAME, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id,
                tags={"model_version": version or "", "git_commit": git_commit,
                      "train_cutoff": train_cutoff or "", "supabase_model_id": supabase_model_id or ""},
                description=reason)
            if promote:
                client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)
            print(f"mlflow: run {run.info.run_id}, {MODEL_NAME} v{mv.version}"
                  f"{' (champion)' if promote else ''}")
            return mv.version
    except Exception as exc:  # noqa: BLE001 - MLflow es un espejo: nunca bloquea el entrenamiento
        print(f"aviso: no se pudo registrar en MLflow ({exc})")
        return None

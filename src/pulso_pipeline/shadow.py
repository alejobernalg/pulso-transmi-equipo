"""Reentrenamiento automático con evaluación en sombra.

1. Disparador (`python -m pulso_pipeline.shadow`, después de cada corrida de
   predicción): guarda una foto del leaderboard y, si nuestro puesto en
   `rolling_24h` está `RANK_GAP` o más por debajo del puesto acumulado,
   dispara `train.yml` en modo sombra.
2. Sombra: el modelo nuevo predice cada ciclo junto al champion, pero no se
   envía (`shadow_predictions`).
3. Decisión (al inicio de cada corrida de predicción): con `SHADOW_CYCLES`
   ciclos ya observados se comparan ambos en los mismos objetivos, con la
   métrica del reto. Si la sombra gana, se promueve y el champion anterior
   queda en sombra; si en sus `SHADOW_CYCLES` ciclos el anterior vuelve a
   ganar, se revierte. Nada de esto bloquea la entrega del ciclo.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO

import httpx
import joblib
import pandas as pd

from pulso_forecast import forecast_for_targets

from . import db

RANK_GAP = 3            # puestos por debajo (24 h vs acumulado) que disparan el reentrenamiento
SHADOW_CYCLES = 4       # ciclos observados para decidir
TRIGGER_COOLDOWN = timedelta(hours=2)


# ------------------------------------------------------------ decisiones puras
def should_trigger(rank_cumulative: int | None, rank_rolling: int | None, shadow_pending: bool,
                   last_trigger_at: datetime | None, now: datetime) -> tuple[bool, str]:
    if rank_cumulative is None or rank_rolling is None:
        return False, "sin posición en el leaderboard"
    gap = rank_rolling - rank_cumulative
    if gap < RANK_GAP:
        return False, f"24 h #{rank_rolling} vs acumulado #{rank_cumulative} (brecha {gap} < {RANK_GAP})"
    if shadow_pending:
        return False, f"brecha {gap}, pero ya hay un modelo en sombra"
    if last_trigger_at is not None and now - last_trigger_at < TRIGGER_COOLDOWN:
        return False, f"brecha {gap}, pero se disparó hace menos de {TRIGGER_COOLDOWN}"
    return True, f"24 h #{rank_rolling} vs acumulado #{rank_cumulative} (brecha {gap})"


def challenge_accuracy(frame: pd.DataFrame) -> tuple[float, float]:
    """(champion, sombra): 100*(1-WAPE) por estación y promedio, en los mismos objetivos.
    `frame` trae station_id, y, champion, shadow."""
    def acc(col: str) -> float:
        err = (frame["y"] - frame[col]).abs().groupby(frame["station_id"]).sum()
        tot = frame["y"].groupby(frame["station_id"]).sum()
        return float((100 * (1 - err / tot)).clip(lower=0).mean())
    return acc("champion"), acc("shadow")


def decide(champion_acc: float, shadow_acc: float, shadow_is_previous_champion: bool) -> tuple[str, str]:
    """(acción, etapa para el perdedor). Acciones: `promote` o `discard`."""
    if shadow_acc > champion_acc:
        # si gana el champion anterior es una reversión: el modelo que falló queda rechazado
        return "promote", "rejected" if shadow_is_previous_champion else "shadow"
    return "discard", "retired" if shadow_is_previous_champion else "rejected"


# ------------------------------------------------------------ evaluación
def _challenge_frame(database, champion: dict, shadow: dict) -> pd.DataFrame:
    sp = pd.DataFrame(database.table("shadow_predictions").select("cycle_id,station_id,target_at,y_pred")
                      .eq("model_id", shadow["model_id"]).gte("created_at", shadow["stage_changed_at"])
                      .execute().data)
    if sp.empty:
        return sp
    cp = pd.DataFrame(database.table("predictions").select("cycle_id,station_id,target_at,y_pred")
                      .eq("model_id", champion["model_id"]).in_("cycle_id", sorted(sp["cycle_id"].unique()))
                      .execute().data)
    if cp.empty:
        return cp
    for f in (sp, cp):
        f["station_id"] = f["station_id"].str.strip()
        f["target_at"] = pd.to_datetime(f["target_at"], utc=True)
    m = sp.rename(columns={"y_pred": "shadow"}).merge(
        cp.rename(columns={"y_pred": "champion"}), on=["cycle_id", "station_id", "target_at"])
    obs = pd.DataFrame(database.table("observations").select("station_id,observed_at,demand")
                       .in_("observed_at", sorted({t.isoformat() for t in m["target_at"]})).execute().data)
    if obs.empty:
        return obs
    obs["station_id"] = obs["station_id"].str.strip()
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    m = m.merge(obs.rename(columns={"observed_at": "target_at", "demand": "y"}), on=["station_id", "target_at"])
    # solo ciclos completos: todos sus objetivos ya observados
    expected = sp.groupby("cycle_id").size()
    got = m.groupby("cycle_id").size()
    complete = got[got == expected.reindex(got.index)].index
    return m[m["cycle_id"].isin(complete)]


def evaluate_shadow(database) -> str | None:
    """Promueve, revierte o descarta el modelo en sombra si ya hay suficientes ciclos."""
    shadow, champion = db.get_shadow_model(database), db.get_active_model(database)
    if shadow is None or champion is None:
        return None
    frame = _challenge_frame(database, champion, shadow)
    n_cycles = frame["cycle_id"].nunique() if not frame.empty else 0
    if n_cycles < SHADOW_CYCLES:
        print(f"sombra {shadow['version']}: {n_cycles}/{SHADOW_CYCLES} ciclos observados")
        return None
    champ_acc, shadow_acc = challenge_accuracy(frame)
    previous = pd.Timestamp(shadow["created_at"]) < pd.Timestamp(champion["created_at"])
    action, loser_stage = decide(champ_acc, shadow_acc, previous)
    summary = (f"sombra {shadow['version']} {shadow_acc:.2f} vs champion {champion['version']} "
               f"{champ_acc:.2f} en {n_cycles} ciclos")
    if action == "promote":
        db.promote_model(database, shadow["model_id"], previous_stage=loser_stage)
        print(f"{'reversión' if previous else 'promoción'}: {summary}")
    else:
        db.set_stage(database, shadow["model_id"], loser_stage)
        print(f"se conserva el champion: {summary}")
    return f"{action}: {summary}"


# ------------------------------------------------------------ predicción en sombra
def predict_shadow(database, cycle: dict, y, ctx, stations, targets: list, champion_id: str) -> None:
    shadow = db.get_shadow_model(database)
    if shadow is None or shadow["model_id"] == champion_id:
        return
    bundle = joblib.load(BytesIO(db.download_model(database, shadow["artifact_uri"])))
    preds = forecast_for_targets(bundle["models"], y, ctx, stations, cycle["data_cutoff"], targets,
                                 train_cutoff=bundle["meta"].get("data_cutoff"))
    db.save_shadow_predictions(database, [
        {"cycle_id": cycle["cycle_id"], "model_id": shadow["model_id"], "station_id": r.station_id,
         "target_at": r.target_at.isoformat(), "horizon_steps": int(r.horizon_steps), "y_pred": float(r.value),
         "issued_at": cycle["data_cutoff"]}
        for r in preds.itertuples()
    ])
    print(f"sombra {shadow['version']}: {len(preds)} predicciones guardadas (no enviadas)")


# ------------------------------------------------------------ disparador
def _my_rank(rows: list[dict], name: str) -> tuple[int | None, float | None]:
    me = next((r for r in rows if r.get("display_name") == name), None)
    return (me["rank"], me["accuracy"]) if me else (None, None)


def dispatch_training() -> None:
    repo, token = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"]
    r = httpx.post(f"https://api.github.com/repos/{repo}/actions/workflows/train.yml/dispatches",
                   headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                   json={"ref": "main", "inputs": {"mode": "shadow"}}, timeout=30)
    r.raise_for_status()


def main() -> int:
    from pulso_transmi import PulsoTransmiClient

    database = db.get_client()
    with PulsoTransmiClient() as client:
        name = client.me()["display_name"]
        cumulative, rolling = client.leaderboard("cumulative"), client.leaderboard("rolling_24h")
        clock = client.clock()
    rank_c, acc_c = _my_rank(cumulative, name)
    rank_r, acc_r = _my_rank(rolling, name)
    last = db.last_retrain_trigger_at(database)
    now = datetime.now(timezone.utc)
    trigger, reason = should_trigger(rank_c, rank_r, db.get_shadow_model(database) is not None,
                                     datetime.fromisoformat(last) if last else None, now)
    if trigger:
        dispatch_training()
    db.save_leaderboard_snapshot(database, {
        "virtual_now": clock.get("virtual_now"), "rank_cumulative": rank_c, "acc_cumulative": acc_c,
        "rank_rolling_24h": rank_r, "acc_rolling_24h": acc_r,
        "leader_cumulative": cumulative[0]["accuracy"] if cumulative else None,
        "leader_rolling_24h": rolling[0]["accuracy"] if rolling else None,
        "participants": len(cumulative), "triggered": trigger, "note": reason,
    })
    print(f"{'reentrenamiento en sombra disparado' if trigger else 'sin reentrenamiento'}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Selección automática de la política de corrección.

Cada predicción enviada guarda sus componentes (`prediction_components`): la
predicción base del modelo y las razones crudas de la corrección. Con ellos se
recalcula, para cada política de `pulso_forecast.model.POLICIES`, qué accuracy
habría tenido en los últimos `WINDOW_CYCLES` ciclos ya observados. El ciclo
siguiente usa la mejor solo si supera a la estándar por `MARGIN` puntos; si no,
o si faltan datos, se queda la estándar. Simulado en el replay: +0.37 en las
oleadas del 18-sep y neutral (±0.04) en días normales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pulso_forecast.model import DEFAULT_POLICY_NAME, POLICIES, policy_factor

WINDOW_CYCLES = 6
MARGIN = 0.5


def policy_scores(comp: pd.DataFrame) -> dict[str, float]:
    """Accuracy (métrica del reto) de cada política sobre componentes con demanda real `y`."""
    scores = {}
    for name, pol in POLICIES.items():
        pred = comp["base"].to_numpy() * policy_factor(comp["r_own"], comp["r_common"], comp["r_now"],
                                                        comp["horizon_steps"].to_numpy(), pol)
        err = pd.Series(np.abs(comp["y"].to_numpy() - pred)).groupby(comp["station_id"].to_numpy()).sum()
        tot = comp["y"].groupby(comp["station_id"].to_numpy()).sum()
        scores[name] = float((100 * (1 - err / tot)).clip(lower=0).mean())
    return scores


def choose(scores: dict[str, float], n_cycles: int) -> tuple[str, str]:
    if n_cycles < WINDOW_CYCLES:
        return DEFAULT_POLICY_NAME, f"{n_cycles}/{WINDOW_CYCLES} ciclos con componentes: política estándar"
    best = max(scores, key=scores.get)
    base = scores[DEFAULT_POLICY_NAME]
    summary = ", ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1]))
    if best != DEFAULT_POLICY_NAME and scores[best] > base + MARGIN:
        return best, f"{best} supera a la estándar por {scores[best] - base:.2f} en {n_cycles} ciclos ({summary})"
    return DEFAULT_POLICY_NAME, f"estándar ({summary})"


def select_policy(database, model_id: str) -> tuple[str, str]:
    comp = pd.DataFrame(database.table("prediction_components")
                        .select("cycle_id,station_id,target_at,horizon_steps,base,r_own,r_common,r_now")
                        .eq("model_id", model_id).order("target_at", desc=True)
                        .limit(48 * (WINDOW_CYCLES + 3)).execute().data)
    if comp.empty:
        return choose({}, 0)
    comp["station_id"] = comp["station_id"].str.strip()
    comp["target_at"] = pd.to_datetime(comp["target_at"], utc=True)
    obs = pd.DataFrame(database.table("observations").select("station_id,observed_at,demand")
                       .in_("observed_at", sorted({t.isoformat() for t in comp["target_at"]})).execute().data)
    if obs.empty:
        return choose({}, 0)
    obs["station_id"] = obs["station_id"].str.strip()
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    m = comp.merge(obs.rename(columns={"observed_at": "target_at", "demand": "y"}), on=["station_id", "target_at"])
    got = m.groupby("cycle_id").size()
    expected = comp.groupby("cycle_id").size().reindex(got.index)
    cycles = sorted(got[got == expected].index)[-WINDOW_CYCLES:]  # solo ciclos observados por completo
    m = m[m["cycle_id"].isin(cycles)]
    for c in ("r_own", "r_common", "r_now"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return choose(policy_scores(m) if len(cycles) else {}, len(cycles))

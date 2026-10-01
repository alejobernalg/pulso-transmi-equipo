"""Replay de la competencia en vivo: re-simula los ciclos horarios tal como los
ejecuta producción, para medir cambios antes de promoverlos.

En cada origen (un ciclo, minuto :00) se hace lo mismo que
`pulso_pipeline.submit_current_cycle`: se toma solo la historia hasta el
origen, se alinea (`align_history`), se arman las features y se predice cada
horizonte. En vez de aplicar ahí la corrección de nivel, se guardan la
predicción base y los residuos de los últimos objetivos ya observados, así
cualquier regla de corrección (`BiasRule`) se evalúa después sin volver a
predecir. `apply_production(preds, resid)` reproduce `forecast_for_targets`.

Uso típico (ver `python -m pulso_forecast.replay --help`):
    preds, resid = collect(models, y_raw, ctx, stations, origins, train_cutoff)
    scored = apply_production(preds, resid)
    summary(scored)
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from . import model as M

# Objetivos pasados que se guardan por (origen, estación, horizonte): cubre la
# ventana más larga que usa cualquier regla de corrección.
RESIDUAL_LOOKBACK = 32

BiasRule = Callable[[pd.DataFrame], pd.Series]


def collect(models: dict, y_raw: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame,
            origins, train_cutoff, align: Callable[[pd.DataFrame], pd.DataFrame] | None = None
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predicciones base y residuos recientes en cada origen.

    `preds`: origin, station_id, h, target_at, base, y (demanda real, sin alinear).
    `resid`: origin, station_id, h, age (1 = objetivo más reciente), y, p; solo
    objetivos posteriores a `train_cutoff` (igual que `recent_bias_factors`).
    """
    align = align or M.align_history
    train_cutoff = pd.Timestamp(train_cutoff)
    pred_rows, resid_rows = [], []
    for origin in origins:
        origin = pd.Timestamp(origin)
        y = align(y_raw.loc[:origin])
        last_t = len(y) - 1
        train_end_t = int(y.index.searchsorted(train_cutoff, side="right")) - 1
        for h, mdl in models.items():
            ext_idx = y.index.append(pd.date_range(origin + pd.Timedelta(minutes=M.STEP_MIN), periods=h,
                                                   freq=f"{M.STEP_MIN}min"))
            frame = M.make_frame(y.reindex(ext_idx), ctx.reindex(ext_idx), stations, h)
            cur = frame[frame["_t"] == last_t]
            base = M.predict(mdl, cur)
            target_at = origin + pd.Timedelta(minutes=M.STEP_MIN * h)
            actual = y_raw.loc[target_at] if target_at in y_raw.index else None
            for st_idx, b in zip(cur["station"].astype(int), base):
                sid = stations.index[st_idx]
                pred_rows.append((origin, sid, h, target_at, float(b),
                                  float(actual[sid]) if actual is not None else np.nan))

            tgt = frame["_t"] + h
            ok = ((tgt <= last_t) & (tgt > max(train_end_t, last_t - RESIDUAL_LOOKBACK))
                  & frame["_y"].notna() & frame["_scale"].notna())
            past = frame[ok]
            if past.empty:
                continue
            p = M.predict(mdl, past)
            ages = (last_t - (past["_t"] + h) + 1).to_numpy()
            for st_idx, age, yy, pp in zip(past["station"].astype(int), ages, past["_y"].to_numpy(), p):
                resid_rows.append((origin, stations.index[st_idx], h, int(age), float(yy), float(pp)))
    preds = pd.DataFrame(pred_rows, columns=["origin", "station_id", "h", "target_at", "base", "y"])
    resid = pd.DataFrame(resid_rows, columns=["origin", "station_id", "h", "age", "y", "p"])
    return preds, resid


# --------------------------------------------------------------- reglas de nivel
def production_rule(g: pd.DataFrame) -> float:
    """Parte por estación de `recent_bias_factors` para un (origen, estación, horizonte).
    El factor común entre estaciones lo agrega `apply_production`."""
    w = g[g["age"] <= M.BIAS_WINDOW]
    if len(w) != M.BIAS_WINDOW or w["p"].sum() <= 0:
        return 1.0
    return float(1 + M.BIAS_SHRINK * (np.clip(w["y"].sum() / w["p"].sum(), *M.BIAS_CLIP) - 1))


def apply_rule(preds: pd.DataFrame, resid: pd.DataFrame, rule: Callable[[pd.DataFrame], float]) -> pd.DataFrame:
    """Aplica una regla de corrección de nivel y devuelve `preds` con `factor` y `pred`."""
    factors = (resid.groupby(["origin", "station_id", "h"], sort=False)[["age", "y", "p"]]
               .apply(rule).rename("factor"))
    out = preds.join(factors, on=["origin", "station_id", "h"])
    out["factor"] = out["factor"].fillna(1.0)
    out["pred"] = out["base"] * out["factor"]
    return out


def common_factors(resid: pd.DataFrame) -> pd.Series:
    """Factor común de `recent_bias_factors` por (origen, horizonte): mediana entre estaciones
    de sum(y)/sum(p) en los últimos `COMMON_WINDOW` objetivos, aplicada a medias."""
    w = resid[resid["age"] <= M.COMMON_WINDOW]
    g = w.groupby(["origin", "h", "station_id"])
    sums = g[["y", "p"]].sum()[(g.size() == M.COMMON_WINDOW) & (g["p"].sum() > 0)]
    med = (sums["y"] / sums["p"]).groupby(level=["origin", "h"]).median()
    return (1 + M.COMMON_SHRINK * (med.clip(*M.COMMON_CLIP) - 1)).rename("common")


def apply_production(preds: pd.DataFrame, resid: pd.DataFrame) -> pd.DataFrame:
    """Réplica completa de `forecast_for_targets`: corrección por estación x factor común x nowcast."""
    out = apply_rule(preds, resid, production_rule).join(common_factors(resid), on=["origin", "h"])
    out["factor"] = out["factor"] * out["common"].fillna(1.0)
    x = resid[(resid["h"] == 1) & (resid["age"] <= M.NOWCAST_WINDOW)].groupby(["origin", "station_id"])[["y", "p"]].sum()
    now = (x["y"] / x["p"].where(x["p"] > 0)).clip(*M.NOWCAST_CLIP).rename("now")
    out = out.join(now, on=["origin", "station_id"])
    expo = out["h"].map(M.NOWCAST_DECAY).fillna(0.0) * M.NOWCAST_STRENGTH
    out["pred"] = out["base"] * out["factor"] * (out["now"] / out["factor"]).pow(expo).fillna(1.0)
    return out


def station_accuracy(scored: pd.DataFrame, col: str = "pred") -> pd.Series:
    """Accuracy 100*(1-WAPE) por estación (la métrica del reto, antes de promediar)."""
    s = scored.dropna(subset=["y"])
    err = (s["y"] - s[col]).abs().groupby(s["station_id"]).sum()
    tot = s["y"].groupby(s["station_id"]).sum()
    return (100 * (1 - err / tot)).clip(lower=0)


def summary(scored: pd.DataFrame, col: str = "pred") -> float:
    return float(station_accuracy(scored, col).mean())


# ----------------------------------------------------------------------- CLI
def _load_live(data_dir: str):
    from pathlib import Path
    d = Path(data_dir)
    obs = pd.read_csv(d / "observations.csv", dtype={"station_id": str})
    ctx = pd.read_csv(d / "context.csv")
    stations = pd.read_csv(d / "stations.csv", dtype={"station_id": str})
    y_raw, ctx, stations = M.raw_wide_from_frames(obs, ctx, stations)
    # el contexto no se publica durante la competencia: se extiende con NaN como en producción
    return y_raw, ctx.reindex(y_raw.index), stations


def main(argv: list[str] | None = None) -> int:
    import joblib
    parser = argparse.ArgumentParser(description="Replay de ciclos horarios sobre datos en vivo")
    parser.add_argument("--data", required=True, help="carpeta con observations/context/stations.csv en vivo")
    parser.add_argument("--model", required=True, help="model.joblib a usar")
    parser.add_argument("--start", required=True, help="primer origen (UTC)")
    parser.add_argument("--end", required=True, help="último origen (UTC)")
    parser.add_argument("--out", required=True, help="prefijo de salida (.preds.pkl / .resid.pkl)")
    args = parser.parse_args(argv)

    y_raw, ctx, stations = _load_live(args.data)
    bundle = joblib.load(args.model)
    origins = pd.date_range(args.start, args.end, freq="1h", tz="UTC")
    preds, resid = collect(bundle["models"], y_raw, ctx, stations, origins, bundle["meta"]["data_cutoff"])
    preds.to_pickle(f"{args.out}.preds.pkl")
    resid.to_pickle(f"{args.out}.resid.pkl")
    scored = apply_production(preds, resid)
    print(station_accuracy(scored).round(2).to_string())
    print(f"promedio: {summary(scored):.2f} (base sin corrección: {summary(scored, 'base'):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

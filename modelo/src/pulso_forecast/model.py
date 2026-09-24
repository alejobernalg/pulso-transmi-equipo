"""Pronóstico de demanda Pulso TransMi sin fuga de información.

Regla única: para un origen t y un horizonte h, toda feature usa solo datos con
marca de tiempo <= t. Las únicas columnas que miran a t+h son (a) el calendario,
que es determinista, y (b) los *pronósticos* de clima, que por definición se
emiten antes. La demanda y el clima real posteriores a t nunca entran.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path(os.environ.get("PULSO_DATA_DIR", "data"))  # carpeta con los CSV del reto
TZ = "America/Bogota"
STEP_MIN = 15
DAY, WEEK = 96, 672
HORIZONS = (1, 2, 3, 4)  # 15, 30, 45, 60 min: contrato oficial (guía operativa v2.0, 2026-09-21)
LAGS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 96)
SEASONAL = (96, 192, 672, 1344)
EVENT_THRESHOLD = 0.1  # event_intensity trae denormales ~1e-323: hay que umbralizar
ADAPTIVE_HALF_LIFE_DAYS = 14  # peso se reduce a la mitad cada 14 días del mismo tipo (laboral/finde)
ADAPTIVE_MAX_LOOKBACK_DAYS = 60  # 0.5**(60/14) ~ 0.03: suficiente para que la cola sea despreciable

FEATURES_NOTE = "ver make_frame: lags <= t, estacionalidad alineada al objetivo, calendario y pronósticos en t+h"


# --------------------------------------------------------------------------- datos
def load_data(data_dir: Path | str | None = None):
    data_dir = Path(data_dir) if data_dir else DATA
    obs = pd.read_csv(data_dir / "observations.csv", dtype={"station_id": str})
    ctx = pd.read_csv(data_dir / "context.csv")
    stations = pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str})
    return wide_from_frames(obs, ctx, stations)


def wide_from_frames(obs: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame):
    """Construye (y, ctx, stations) en formato ancho a partir de DataFrames ya
    cargados (de CSV o de Supabase).

    `context` puede ir más atrás que `observations` (en el pipeline en vivo,
    `/v1/stream/observations` libera periodos nuevos antes de que
    `/v1/context` los tenga): se rellena con NaN, que el modelo ya maneja de
    forma nativa. Lo que sí es un error real es que `context` tenga periodos
    que `observations` no tiene (huérfanos).
    """
    obs = obs.copy()
    ctx = ctx.copy()
    for frame in (obs, ctx):
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    y = obs.pivot(index="observed_at", columns="station_id", values="demand").sort_index().astype(float)
    ctx = ctx.set_index("observed_at").sort_index()
    steps = y.index.to_series().diff().dropna().unique()
    assert len(steps) == 1 and steps[0] == pd.Timedelta(minutes=STEP_MIN), "la serie debe ser regular"
    orphan_ctx = ctx.index.difference(y.index)
    assert orphan_ctx.empty, f"context tiene {len(orphan_ctx)} periodos sin observaciones"
    ctx = ctx.reindex(y.index)
    stations = stations.set_index("station_id").loc[y.columns]
    return y, ctx, stations


# ---------------------------------------------------------------------- features
def _adaptive_profile(y: pd.DataFrame, local, target, target_weekend, h: int, scale: pd.DataFrame) -> np.ndarray:
    """Generalización de `daytype14`: en vez de un promedio plano de los últimos
    14 días del mismo tipo, pondera exponencialmente TODO el historial disponible
    (hasta 60 días del mismo tipo), con vida media de `ADAPTIVE_HALF_LIFE_DAYS`
    días -- días más recientes pesan más, pero ninguno se descarta del todo.
    Mismo patrón `y.shift(k-h)` (k = DAY*d, d>=1) que ya usa daytype14, así que
    hereda la misma garantía de no ver el futuro."""
    vals, weights = [], []
    for d in range(1, ADAPTIVE_MAX_LOOKBACK_DAYS + 1):
        k = DAY * d
        if k < h:
            continue
        v = (y.shift(k - h) / scale).to_numpy().copy()
        past_weekend = np.asarray((local + pd.Timedelta(minutes=STEP_MIN * (h - k))).dayofweek >= 5)
        v[past_weekend != target_weekend] = np.nan
        vals.append(v)
        weights.append(0.5 ** (d / ADAPTIVE_HALF_LIFE_DAYS))
    stack = np.stack(vals)
    w = np.asarray(weights).reshape(-1, 1, 1)
    w_masked = np.where(np.isnan(stack), 0.0, w)
    weighted_sum = np.nansum(stack * w, axis=0)
    weight_total = w_masked.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight_total > 0, weighted_sum / weight_total, np.nan)


def make_frame(y: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame, h: int,
               use_ctx: bool = True, event_at_target: bool = False, use_adaptive_profile: bool = True) -> pd.DataFrame:
    """Una fila por (origen t, estación). Columnas `_*` son metadatos, no features."""
    n_t, n_s = y.shape
    scale = y.rolling(WEEK, min_periods=DAY).mean()  # nivel reciente de la estación, solo pasado
    cols: dict[str, np.ndarray] = {}

    def add(name: str, wide: pd.DataFrame | np.ndarray):
        cols[name] = np.asarray(wide, dtype=float).reshape(-1)

    for j in LAGS:
        add(f"lag{j}", y.shift(j) / scale)
    for w in (4, 16, 96):
        add(f"mean{w}", y.rolling(w).mean() / scale)
    add("std16", y.rolling(16).std() / scale)
    seasonal = []
    for k in SEASONAL:
        if k >= h:  # y_{t+h-k} ya ocurrió en t solo si k >= h
            s = y.shift(k - h) / scale
            add(f"seas{k}", s)
            if k in (WEEK, 2 * WEEK):
                seasonal.append(s.to_numpy())
    add("seas_week_mean", np.nanmean(np.stack(seasonal), axis=0) if seasonal else np.full((n_t, n_s), np.nan))
    add("scale_change", scale / scale.shift(DAY))  # deriva de nivel reciente

    local = y.index.tz_convert(TZ)
    target = local + pd.Timedelta(minutes=STEP_MIN * h)

    # Perfil del slot objetivo: la demanda es un perfil estable más ruido independiente, así que
    # promediar muchas observaciones pasadas del mismo slot reduce el ruido mejor que los lags cortos.
    smooth = [(y.shift(WEEK * m - o - h) / scale).to_numpy()
              for m in range(1, 5) for o in (-1, 0, 1) if WEEK * m - o >= h]
    add("wk4_smooth", np.nanmean(np.stack(smooth), axis=0))
    target_weekend = np.asarray(target.dayofweek >= 5)
    same_type = []
    for d in range(1, 15):
        k = DAY * d
        if k < h:
            continue
        v = (y.shift(k - h) / scale).to_numpy().copy()
        past_weekend = np.asarray((local + pd.Timedelta(minutes=STEP_MIN * (h - k))).dayofweek >= 5)
        v[past_weekend != target_weekend] = np.nan  # solo días del mismo tipo (laboral / fin de semana)
        same_type.append(v)
    stack = np.stack(same_type)
    add("daytype14", np.nanmean(stack, axis=0))
    add("daytype14_med", np.nanmedian(stack, axis=0))
    if use_adaptive_profile:
        add("adaptive_profile_hl14", _adaptive_profile(y, local, target, target_weekend, h, scale))
    slot = (target.hour * 60 + target.minute) // STEP_MIN
    dow = target.dayofweek
    for name, vec in {
        "slot": slot, "dow": dow, "weekend": (dow >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * slot / DAY), "hour_cos": np.cos(2 * np.pi * slot / DAY),
    }.items():
        add(name, np.repeat(np.asarray(vec, dtype=float)[:, None], n_s, axis=1))

    ev = ctx["event_intensity"].where(ctx["event_intensity"] > EVENT_THRESHOLD, 0.0)
    add("event_now", np.repeat(ev.to_numpy()[:, None], n_s, axis=1))
    if use_ctx:
        for c in ("rain_forecast", "temperature_forecast"):  # pronóstico emitido antes de t+h
            add(f"{c}_target", np.repeat(ctx[c].shift(-h).to_numpy()[:, None], n_s, axis=1))
    if event_at_target:  # cota superior: solo válida si la agenda de eventos se conoce de antemano
        add("event_target", np.repeat(ev.shift(-h).to_numpy()[:, None], n_s, axis=1))

    add("lat", np.tile(stations["latitude"].to_numpy(), (n_t, 1)))
    add("lon", np.tile(stations["longitude"].to_numpy(), (n_t, 1)))

    frame = pd.DataFrame(cols)
    frame["station"] = pd.Categorical(np.tile(np.arange(n_s), n_t), categories=range(n_s))
    frame["_t"] = np.repeat(np.arange(n_t), n_s)
    frame["_scale"] = scale.to_numpy().reshape(-1)
    frame["_y"] = y.shift(-h).to_numpy().reshape(-1)            # objetivo crudo (solo evaluación/entrenamiento)
    frame["_ratio"] = frame["_y"] / frame["_scale"]              # objetivo escalado
    for k in (96, WEEK):
        frame[f"_naive{k}"] = (y.shift(k - h)).to_numpy().reshape(-1)
    two = np.nanmean(np.stack([y.shift(WEEK - h).to_numpy(), y.shift(2 * WEEK - h).to_numpy()]), axis=0)
    frame["_naive_2wk"] = two.reshape(-1)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if not c.startswith("_")]


# --------------------------------------------------------------------- métricas
def accuracy(frame: pd.DataFrame, pred: np.ndarray) -> float:
    """100*(1-WAPE) por estación y luego promedio simple (la métrica del reto)."""
    err = pd.Series(np.abs(frame["_y"].to_numpy() - pred)).groupby(frame["station"].to_numpy()).sum()
    tot = frame["_y"].groupby(frame["station"].to_numpy()).sum()
    return float((100 * (1 - err / tot).clip(lower=0)).mean())


# ------------------------------------------------------------------ entrenamiento
def split_by_target(frame: pd.DataFrame, h: int, train_end: int, val: tuple[int, int] | None):
    """train: objetivos con t+h <= train_end (purga: ningún objetivo cruza el corte).
    val: objetivos en [val[0], val[1])."""
    tgt = frame["_t"] + h
    ok = frame["_ratio"].notna() & frame["_scale"].notna()
    train = frame[ok & (tgt <= train_end)]
    valid = frame[ok & (tgt >= val[0]) & (tgt < val[1])] if val else None
    return train, valid


def fit(train: pd.DataFrame, params: dict) -> HistGradientBoostingRegressor:
    """Se probó reemplazar esto por XGBoost (ganaba por 0.07 pts en un dataset
    más chico), pero al reevaluar en el dataset en vivo, más grande, HGB volvió
    a ganar por 1.43 pts en los 4 horizontes (86.92 vs 85.49) -- la ventaja de
    XGBoost no era real, era ruido de un split pequeño. Se mantiene HGB."""
    model = HistGradientBoostingRegressor(
        loss="absolute_error", categorical_features="from_dtype", random_state=0, **params)
    model.fit(train[feature_columns(train)], train["_ratio"])
    return model


def predict(model, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(frame[feature_columns(frame)]), 0, None) * frame["_scale"].to_numpy()


GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": it, "min_samples_leaf": msl, "l2_regularization": 1.0}
    for lr, leaves, it, msl in [(0.05, 15, 300, 40), (0.05, 31, 300, 40), (0.03, 15, 500, 80), (0.05, 7, 400, 80)]
]


def run(out_dir: Path, horizons=HORIZONS, test_days: int = 7, data_dir=None) -> dict:
    y, ctx, stations = load_data(data_dir)
    return run_from_frames(y, ctx, stations, out_dir, horizons=horizons, test_days=test_days)


def run_from_frames(y: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame, out_dir: Path,
                     horizons=HORIZONS, test_days: int = 7) -> dict:
    n_t = len(y)
    test_start = n_t - test_days * DAY
    folds = [(test_start - 2 * WEEK, test_start - WEEK), (test_start - WEEK, test_start)]
    report: dict = {"n_periods": n_t, "test_start": str(y.index[test_start]), "horizons": {}}

    for h in horizons:
        frame = make_frame(y, ctx, stations, h)
        # 1) selección de hiperparámetros solo con datos anteriores al test
        cv = []
        for params in GRID:
            scores = []
            for v0, v1 in folds:
                tr, va = split_by_target(frame, h, train_end=v0 - 1, val=(v0, v1))
                scores.append(accuracy(va, predict(fit(tr, params), va)))
            cv.append(float(np.mean(scores)))
        best = GRID[int(np.argmax(cv))]

        # 2) evaluación única en el bloque final, nunca visto
        tr, te = split_by_target(frame, h, train_end=test_start - 1, val=(test_start, n_t))
        model = fit(tr, best)
        row = {
            "best_params": best, "cv_accuracy": cv, "n_train": len(tr), "n_test": len(te),
            "model": accuracy(te, predict(model, te)),
            "naive_day": accuracy(te.dropna(subset=["_naive96"]), te.dropna(subset=["_naive96"])["_naive96"].to_numpy()),
            "naive_week": accuracy(te.dropna(subset=["_naive672"]), te.dropna(subset=["_naive672"])["_naive672"].to_numpy()),
        }
        # 3) ablaciones: ¿aporta el pronóstico de clima? ¿cuánto ganaría un evento conocido de antemano?
        for name, kw in {"sin_contexto": {"use_ctx": False}, "cota_evento_conocido": {"event_at_target": True}}.items():
            fa = make_frame(y, ctx, stations, h, **kw)
            tra, tea = split_by_target(fa, h, train_end=test_start - 1, val=(test_start, n_t))
            row[name] = accuracy(tea, predict(fit(tra, best), tea))
        # 4) accuracy por estación en test
        pred = predict(model, te)
        err = pd.Series(np.abs(te["_y"].to_numpy() - pred)).groupby(te["station"].to_numpy()).sum()
        tot = te["_y"].groupby(te["station"].to_numpy()).sum()
        row["por_estacion"] = {y.columns[int(i)]: round(float(100 * (1 - err[i] / tot[i])), 2) for i in err.index}
        report["horizons"][f"h{h}"] = row

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def train_production(out_dir: Path, params_by_h: dict[int, dict], horizons=HORIZONS, data_dir=None):
    """Reentrena con TODOS los datos disponibles y guarda un modelo por horizonte."""
    y, ctx, stations = load_data(data_dir)
    return train_production_from_frames(y, ctx, stations, out_dir, params_by_h, horizons=horizons)


def train_production_from_frames(y: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame, out_dir: Path,
                                  params_by_h: dict[int, dict], horizons=HORIZONS) -> dict:
    """Igual que `train_production` pero a partir de frames ya cargados (p. ej. desde Supabase)."""
    import joblib
    n_t = len(y)
    models = {}
    for h in horizons:
        frame = make_frame(y, ctx, stations, h)
        tr, _ = split_by_target(frame, h, train_end=n_t - 1, val=None)
        models[h] = fit(tr, params_by_h[h])
    import sklearn
    meta = {"sklearn_version": sklearn.__version__, "data_cutoff": str(y.index[-1]), "horizons": list(horizons), "features": FEATURES_NOTE,
            "trained_rows": {h: int(len(split_by_target(make_frame(y, ctx, stations, h), h, n_t - 1, None)[0]))
                             for h in horizons}}
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "meta": meta}, out_dir / "model.joblib")
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return models


def forecast_next(models: dict, horizons=None, data_dir=None) -> pd.DataFrame:
    """Predicciones desde el último origen disponible (una por estación y horizonte)."""
    horizons = horizons or sorted(models)
    y, ctx, stations = load_data(data_dir)
    origin = y.index[-1]
    rows = []
    for h in horizons:
        # el contexto futuro (pronósticos) no existe más allá del último dato: se extiende con NaN,
        # que HistGradientBoosting maneja de forma nativa.
        ext_idx = y.index.append(pd.date_range(origin + pd.Timedelta(minutes=STEP_MIN), periods=h,
                                               freq=f"{STEP_MIN}min"))
        y_ext = y.reindex(ext_idx)
        ctx_ext = ctx.reindex(ext_idx)
        frame = make_frame(y_ext, ctx_ext, stations, h)
        cur = frame[frame["_t"] == len(y) - 1]
        pred = predict(models[h], cur)
        for st, p in zip(stations.index[cur["station"].astype(int)], pred):
            rows.append({"station_id": st, "origin_at": origin, "horizon_steps": h,
                         "target_at": origin + pd.Timedelta(minutes=STEP_MIN * h), "value": round(float(p), 2)})
    return pd.DataFrame(rows)


def forecast_for_targets(models: dict, y: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame,
                          origin, targets) -> pd.DataFrame:
    """Predice exactamente los pares (station_id, target_at) que pide un ciclo.

    `origin` es el `data_cutoff` del ciclo (debe ser el último índice de `y`).
    `targets` es una lista de (station_id, target_at); el horizonte de cada
    fila se deriva de `target_at - origin` y debe caer en `models`. Devuelve
    las filas en el mismo orden que `targets`.
    """
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    if y.index[-1] != origin:
        raise ValueError(f"origin {origin} no coincide con el último dato disponible {y.index[-1]}")

    parsed = []
    for station_id, target_at in targets:
        target_at = pd.Timestamp(target_at)
        if target_at.tzinfo is None:
            target_at = target_at.tz_localize("UTC")
        delta_steps = (target_at - origin) / pd.Timedelta(minutes=STEP_MIN)
        h = round(delta_steps)
        if abs(delta_steps - h) > 1e-6 or h not in models:
            raise ValueError(f"target_at {target_at} no cae en un horizonte soportado ({sorted(models)} pasos)")
        parsed.append((station_id, target_at, h))

    by_h = pd.Series([h for _, _, h in parsed]).unique()
    preds_by_h: dict[int, pd.Series] = {}
    for h in by_h:
        ext_idx = y.index.append(pd.date_range(origin + pd.Timedelta(minutes=STEP_MIN), periods=h,
                                               freq=f"{STEP_MIN}min"))
        frame = make_frame(y.reindex(ext_idx), ctx.reindex(ext_idx), stations, h)
        cur = frame[frame["_t"] == len(y) - 1]
        pred = predict(models[h], cur)
        station_ids = stations.index[cur["station"].astype(int)]
        preds_by_h[h] = pd.Series(pred, index=station_ids)

    rows = []
    for station_id, target_at, h in parsed:
        if station_id not in preds_by_h[h].index:
            raise ValueError(f"estación desconocida en el modelo: {station_id}")
        rows.append({"station_id": station_id, "target_at": target_at, "horizon_steps": h,
                     "value": round(float(preds_by_h[h][station_id]), 2)})
    return pd.DataFrame(rows)

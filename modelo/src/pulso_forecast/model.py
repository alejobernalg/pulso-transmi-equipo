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
from lightgbm import LGBMRegressor
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
    cargados (de CSV o de Supabase), con la historia ya alineada al régimen
    actual (`align_history`).

    `context` puede ir más atrás que `observations` (en el pipeline en vivo,
    `/v1/stream/observations` libera periodos nuevos antes de que
    `/v1/context` los tenga): se rellena con NaN, que el modelo ya maneja de
    forma nativa. Lo que sí es un error real es que `context` tenga periodos
    que `observations` no tiene (huérfanos).
    """
    y, ctx, stations = raw_wide_from_frames(obs, ctx, stations)
    return align_history(y), ctx, stations


def raw_wide_from_frames(obs: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame):
    """Como `wide_from_frames` pero sin alinear la historia (demanda tal cual se observó)."""
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


def align_history(y: pd.DataFrame) -> pd.DataFrame:
    """Alinea la historia al régimen actual de cada estación: primero el horario del
    pico, luego el nivel. Solo usa datos de `y`, así que es seguro en inferencia."""
    return align_level_shifts(align_peak_shifts(y))


# ------------------------------------------------------------ corrimiento de picos
# El generador de la plataforma incluye drifts `peak_shift` (docs/pattern-generator.md del
# repo público del profesor): el pico diario de algunas estaciones se corre N minutos. Las
# features estacionales (lags de 1 día / 1 semana, perfiles) quedan desfasadas y el modelo
# predice el pico a la hora vieja. En vivo (11-sep) 4 estaciones corrieron su pico +45 min.
# Se detecta por estación comparando la forma diaria reciente contra el perfil previo a la
# competencia, y se corre el historial anterior al cambio para que quede alineado.
# Validado en dos ventanas posteriores al cambio: +0.96 a +2.31 pts (8/8), con las estaciones
# afectadas subiendo +1.3 a +7.0 pts y el resto sin cambio.
COMPETITION_START = pd.Timestamp("2026-09-09T05:00:00Z")  # fin de la historia inicial (/v1/meta)
SHIFT_MAX = 6            # se buscan corrimientos de hasta ±90 min
SHIFT_MIN = 2            # ±15 min es ruido día a día; el drift real fue de 3 periodos
SHIFT_ERR_RATIO = 0.8    # el perfil corrido debe reducir el error de forma al menos 20%
SHIFT_MIN_SLOTS = 48     # un día cuenta solo si tiene al menos 12 h observadas


def _day_shift(day: np.ndarray, base: np.ndarray) -> tuple[int, float]:
    """Mejor corrimiento k (en periodos) del perfil base para explicar la forma del día."""
    v = ~np.isnan(day)
    d = day[v] / day[v].sum()
    errs = {}
    for k in range(-SHIFT_MAX, SHIFT_MAX + 1):
        b = np.roll(base, k)[v]
        errs[k] = float(np.abs(d - b / b.sum()).sum())
    k = min(errs, key=errs.get)
    return k, errs[k] / errs[0] if errs[0] > 0 else 1.0


def detect_peak_shifts(y: pd.DataFrame) -> dict:
    """{station_id: (k, cambio)} para estaciones cuyo pico diario se corrió de forma sostenida.
    Solo usa datos de `y` (pasado), así que es seguro en inferencia."""
    if y.index[-1] <= COMPETITION_START:
        return {}
    local = y.index.tz_convert(TZ)
    slot = np.asarray(local.hour * 4 + local.minute // STEP_MIN)
    weekend = np.asarray(local.dayofweek >= 5)
    pre = np.asarray(y.index < COMPETITION_START)
    base = {we: y[pre & (weekend == we)].groupby(slot[pre & (weekend == we)]).mean().reindex(range(DAY))
            for we in (False, True)}
    days = pd.date_range(COMPETITION_START.tz_convert(TZ).normalize(), local[-1].normalize(), freq="D")
    shifts = {}
    for sid in y.columns:
        per_day = []
        for day in days:
            m = np.asarray((local >= day) & (local < day + pd.Timedelta(days=1)))
            if m.sum() < SHIFT_MIN_SLOTS:
                continue
            prof = pd.Series(y[sid].to_numpy()[m], index=slot[m]).groupby(level=0).mean().reindex(range(DAY)).to_numpy()
            k, ratio = _day_shift(prof, base[day.dayofweek >= 5][sid].to_numpy())
            per_day.append((day, k if abs(k) >= SHIFT_MIN and ratio < SHIFT_ERR_RATIO else 0))
        recent = [k for _, k in per_day[-3:]]
        nonzero = [k for k in recent if k != 0]
        if len(nonzero) < 2 or len({np.sign(k) for k in nonzero}) > 1:
            continue
        k = int(pd.Series(nonzero).mode().iloc[0])
        first = per_day[-1][0]
        for day, dk in reversed(per_day):  # inicio del tramo final de días corridos
            if dk == 0:
                break
            first = day
        shifts[sid] = (k, (first - pd.Timedelta(hours=12)).tz_convert("UTC"))
    return shifts


def align_peak_shifts(y: pd.DataFrame) -> pd.DataFrame:
    """Corre el historial previo a cada cambio de pico para alinearlo con el régimen actual."""
    shifts = detect_peak_shifts(y)
    if not shifts:
        return y
    y = y.copy()
    for sid, (k, change) in shifts.items():
        before = y.index < change
        y.loc[before, sid] = y[sid].shift(k)[before].fillna(y[sid][before])  # bordes: valor original
    return y


# ------------------------------------------------------------ cambios de nivel
# El generador también aplica cambios bruscos de nivel por estación durante la competencia
# (16-sep: Portal Américas x2.6, Calle 100 x2.4, Portal Suba x0.4; Banderas cayó por escalones
# hasta x0.2). El nivel de 7 días, los lags y los perfiles quedan días en el régimen viejo y
# la corrección online de sesgo no alcanza a compensarlo. Se detecta cada quiebre sobre el
# nivel horario (demanda / perfil previo a la competencia) y se reescala la historia anterior
# al nivel del tramo actual, como `align_peak_shifts` hace con el horario del pico.
# Replay en vivo (pulso_forecast.replay): +0.95 pts en la
# ventana con quiebres (14..18-sep: Banderas +8.4, Américas +1.5, Ricaurte +0.9, Suba +0.6)
# y -0.02 en la ventana sin quiebres grandes (11..14-sep), peor estación estable -0.15.
# Umbrales elegidos entre tres calibraciones: los más laxos (3 h, x1.3, z 4) confundían el
# corrimiento de picos del 11-sep con cambios de nivel (-0.07 en la ventana sin quiebres).
LEVEL_BLOCK = 4                  # periodos por bloque (1 h)
LEVEL_MIN_AFTER = 4              # bloques mínimos después del quiebre
LEVEL_MAX_BEFORE = 24            # bloques de referencia antes del quiebre
LEVEL_MIN_CHANGE = np.log(1.35)  # cambio mínimo de nivel (log)
LEVEL_MIN_Z = 5.0                # separación mínima (estadístico t de dos tramos)
LEVEL_LOOKBACK = pd.Timedelta(days=7)  # el nivel se mide desde 7 días antes de la competencia
# Un quiebre solo cuenta si el tramo nuevo tiene al menos LEVEL_MIN_AFTER horas diurnas. El
# 18-sep de madrugada hubo oleadas de x2-x10 sobre un perfil casi nulo que el detector tomaba
# por escalones (5 estaciones a las 23 h) y habría reescalado su historia. Los escalones
# reales (16-sep, 8-11 h) se detectan igual; uno nocturno se detectaría a la mañana siguiente.
# Replay: 84.93 / 87.83 en las ventanas 11..14 y 14..18-sep (igual que sin la regla) y
# ninguna estación marcada en falso durante las oleadas.
LEVEL_DAY_HOURS = (5, 22)


def _block_levels(y: pd.DataFrame) -> pd.DataFrame:
    """log(sum(y) / sum(perfil previo a la competencia)) por bloque horario y estación."""
    local = y.index.tz_convert(TZ)
    slot = np.asarray(local.hour * 4 + local.minute // STEP_MIN)
    weekend = np.asarray(local.dayofweek >= 5)
    pre = np.asarray(y.index < COMPETITION_START)
    base = np.full(y.shape, np.nan)
    for we in (False, True):
        m = weekend == we
        prof = y[pre & m].groupby(slot[pre & m]).mean()
        base[m] = prof.reindex(slot[m]).to_numpy()
    start = COMPETITION_START - LEVEL_LOOKBACK
    yy = y.loc[start:]
    bb = pd.DataFrame(base, index=y.index, columns=y.columns).loc[start:]
    blk = np.arange(len(yy)) // LEVEL_BLOCK
    num = yy.groupby(blk).sum(min_count=LEVEL_BLOCK)
    den = bb.groupby(blk).sum(min_count=LEVEL_BLOCK)
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.log((num / den).where((num > 0) & (den > 0)))
    lv.index = yy.index[::LEVEL_BLOCK][: len(lv)]
    return lv


def _latest_level_break(x: np.ndarray, day: np.ndarray) -> tuple[int, float, float] | None:
    """Quiebre más reciente en la serie de log-nivel `x` (`day`: bloque diurno): (índice,
    cambio, z) o None."""
    best = None
    for tau in range(len(x) - LEVEL_MIN_AFTER, 0, -1):
        after, before = x[tau:], x[max(0, tau - LEVEL_MAX_BEFORE):tau]
        if len(before) < LEVEL_MIN_AFTER:
            break
        if day[tau:].sum() < LEVEL_MIN_AFTER:
            continue
        diff = after.mean() - before.mean()
        s = np.sqrt(np.var(np.r_[after - after.mean(), before - before.mean()]) + 1e-6)
        z = abs(diff) / (s * np.sqrt(1 / len(after) + 1 / len(before)))
        if abs(diff) > LEVEL_MIN_CHANGE and z > LEVEL_MIN_Z and (best is None or z > best[2]):
            best = (tau, float(diff), float(z))
    return best


def detect_level_shifts(y: pd.DataFrame) -> dict:
    """{station_id: [inicio de cada tramo nuevo, ...]} para estaciones con quiebres de nivel
    desde la competencia. Busca el quiebre más reciente y repite hacia atrás."""
    if y.index[-1] <= COMPETITION_START:
        return {}
    lv = _block_levels(y)
    out = {}
    for sid in y.columns:
        s = lv[sid].dropna()
        x, idx = s.to_numpy(), s.index
        hour = idx.tz_convert(TZ).hour
        day = np.asarray((hour >= LEVEL_DAY_HOURS[0]) & (hour < LEVEL_DAY_HOURS[1]))
        breaks, end = [], len(x)
        while (b := _latest_level_break(x[:end], day[:end])) is not None:
            breaks.append(idx[b[0]])
            end = b[0]
        if breaks:
            out[sid] = sorted(breaks)
    return out


def align_level_shifts(y: pd.DataFrame) -> pd.DataFrame:
    """Reescala cada tramo anterior a un quiebre de nivel al nivel del tramo actual."""
    shifts = detect_level_shifts(y)
    if not shifts:
        return y
    lv = _block_levels(y)
    y = y.copy()
    for sid, breaks in shifts.items():
        edges = [y.index[0], *breaks, y.index[-1] + pd.Timedelta(minutes=STEP_MIN)]
        levels = [np.exp(lv[sid][(lv.index >= a) & (lv.index < b)].mean()) for a, b in zip(edges[:-1], edges[1:])]
        for a, b, lvl in zip(edges[:-2], edges[1:-1], levels[:-1]):
            seg = (y.index >= a) & (y.index < b)
            y.loc[seg, sid] = y.loc[seg, sid] * (levels[-1] / lvl)
    return y


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
               use_ctx: bool = True, event_at_target: bool = False, use_adaptive_profile: bool = True,
               scale_window: int = WEEK) -> pd.DataFrame:
    """Una fila por (origen t, estación). Columnas `_*` son metadatos, no features."""
    n_t, n_s = y.shape
    scale = y.rolling(scale_window, min_periods=min(DAY, scale_window)).mean()  # nivel reciente de la estación, solo pasado
    cols: dict[str, np.ndarray] = {}

    def add(name: str, wide: pd.DataFrame | np.ndarray):
        cols[name] = np.asarray(wide, dtype=float).reshape(-1)

    for j in LAGS:
        add(f"lag{j}", y.shift(j) / scale)
    for w in (4, 16, 96):
        add(f"mean{w}", y.rolling(w).mean() / scale)
    add("std16", y.rolling(16).std() / scale)
    # tendencia de corto plazo (última hora vs últimas 4 h): +0.08 pts medios, 7/8
    # combinaciones horizonte x ventana en vivo con la corrección de sesgo activa
    add("trend4_16", y.rolling(4).mean() / y.rolling(16).mean())
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


def _fit_hgb(train: pd.DataFrame, params: dict, sample_weight=None) -> HistGradientBoostingRegressor:
    """Se probó reemplazar esto por XGBoost (ganaba por 0.07 pts en un dataset
    más chico), pero al reevaluar en el dataset en vivo, más grande, HGB volvió
    a ganar por 1.43 pts en los 4 horizontes (86.92 vs 85.49) -- la ventaja de
    XGBoost no era real, era ruido de un split pequeño. Se mantiene HGB como
    una de las dos patas del blend (ver `fit`)."""
    model = HistGradientBoostingRegressor(
        loss="absolute_error", categorical_features="from_dtype", random_state=0, **params)
    model.fit(train[feature_columns(train)], train["_ratio"], sample_weight=sample_weight)
    return model


def _fit_lgb(train: pd.DataFrame, params: dict, sample_weight=None) -> LGBMRegressor:
    model = LGBMRegressor(objective="regression_l1", random_state=0, verbosity=-1, **params)
    # categorical_feature="auto" detecta "station"
    model.fit(train[feature_columns(train)], train["_ratio"], sample_weight=sample_weight)
    return model


class BlendedModel:
    """Promedio ponderado de HGB y LightGBM, ya clippeados y en escala de demanda
    (no de _ratio), tal como se validó en la comparación live: cada pata se
    clippea a >=0 por separado antes de mezclar, así el blend nunca puede dar
    negativo. `w_hgb` es el peso de HGB elegido por CV (ver run_from_frames)."""
    def __init__(self, hgb, lgb, w_hgb: float):
        self.hgb, self.lgb, self.w_hgb = hgb, lgb, w_hgb


def fit(train: pd.DataFrame, params: dict, sample_weight=None) -> BlendedModel:
    """Stacking simple HGB + LightGBM. `params` = {"hgb": {...}, "lgb": {...},
    "w_hgb": float}. Se probaron 9 alternativas de modelo único (XGBoost,
    RandomForest, ExtraTrees, GradientBoosting clásico, LightGBM solo,
    CatBoost, MLP, features de corredor) y ninguna superó a HGB de forma
    consistente. El blend HGB+LightGBM sí: +0.03 a +0.10 pts de accuracy en
    8/8 combinaciones horizonte x ventana probadas en datos en vivo (dos
    ventanas independientes) -- señal real, aunque modesta, no ruido."""
    return BlendedModel(_fit_hgb(train, params["hgb"], sample_weight), _fit_lgb(train, params["lgb"], sample_weight),
                        params["w_hgb"])


def predict(model, frame: pd.DataFrame) -> np.ndarray:
    # columnas con las que se entrenó el modelo (no las del frame actual): así un
    # modelo activo sigue prediciendo aunque el código agregue features nuevas
    trained = model.hgb if isinstance(model, BlendedModel) else model
    X = frame[list(getattr(trained, "feature_names_in_", feature_columns(frame)))]
    scale = frame["_scale"].to_numpy()
    if isinstance(model, BlendedModel):
        p_hgb = np.clip(model.hgb.predict(X), 0, None) * scale
        p_lgb = np.clip(model.lgb.predict(X), 0, None) * scale
        return model.w_hgb * p_hgb + (1 - model.w_hgb) * p_lgb
    return np.clip(model.predict(X), 0, None) * scale


HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": it, "min_samples_leaf": msl, "l2_regularization": 1.0}
    for lr, leaves, it, msl in [(0.05, 15, 300, 40), (0.05, 31, 300, 40), (0.03, 15, 500, 80), (0.05, 7, 400, 80)]
]
LGB_GRID = [
    {"learning_rate": lr, "num_leaves": leaves, "n_estimators": it, "min_child_samples": msl}
    for lr, leaves, it, msl in [(0.05, 31, 300, 40), (0.05, 63, 300, 40), (0.03, 31, 500, 80)]
]
BLEND_WEIGHTS = tuple(round(w, 1) for w in np.arange(0.0, 1.01, 0.1))


def run(out_dir: Path, horizons=HORIZONS, test_days: int = 7, data_dir=None) -> dict:
    y, ctx, stations = load_data(data_dir)
    return run_from_frames(y, ctx, stations, out_dir, horizons=horizons, test_days=test_days)


def run_from_frames(y: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame, out_dir: Path,
                     horizons=HORIZONS, test_days: int = 7, reference: dict | None = None) -> dict:
    """Validación temporal. Con `reference` = {"models": {h: modelo}, "train_cutoff": ts}
    (el champion), agrega por horizonte `same_block`: la receta candidata reentrenada con
    el mismo corte que el champion, y ambos medidos sobre los mismos objetivos posteriores
    a ese corte (ninguno los vio al entrenar)."""
    n_t = len(y)
    test_start = n_t - test_days * DAY
    folds = [(test_start - 2 * WEEK, test_start - WEEK), (test_start - WEEK, test_start)]
    report: dict = {"n_periods": n_t, "test_start": str(y.index[test_start]), "horizons": {}}

    for h in horizons:
        frame = make_frame(y, ctx, stations, h)
        # 1) hiperparámetros de HGB y de LightGBM elegidos por separado, solo con
        # datos anteriores al test (mismos folds para ambos, comparables)
        def _cv_best(grid, fit_fn):
            cv_scores = []
            for params in grid:
                scores = []
                for v0, v1 in folds:
                    tr, va = split_by_target(frame, h, train_end=v0 - 1, val=(v0, v1))
                    pred = np.clip(fit_fn(tr, params).predict(va[feature_columns(va)]), 0, None) * va["_scale"].to_numpy()
                    scores.append(accuracy(va, pred))
                cv_scores.append(float(np.mean(scores)))
            return grid[int(np.argmax(cv_scores))], cv_scores

        best_hgb, cv_hgb = _cv_best(HGB_GRID, _fit_hgb)
        best_lgb, cv_lgb = _cv_best(LGB_GRID, _fit_lgb)

        # 2) peso del blend optimizado en los mismos folds de CV (nunca en test):
        # se refitea una sola vez por fold con los mejores hiperparámetros y se
        # barren los pesos sobre esas predicciones ya calculadas
        fold_preds = []
        for v0, v1 in folds:
            tr, va = split_by_target(frame, h, train_end=v0 - 1, val=(v0, v1))
            p_hgb = np.clip(_fit_hgb(tr, best_hgb).predict(va[feature_columns(va)]), 0, None) * va["_scale"].to_numpy()
            p_lgb = np.clip(_fit_lgb(tr, best_lgb).predict(va[feature_columns(va)]), 0, None) * va["_scale"].to_numpy()
            fold_preds.append((va, p_hgb, p_lgb))
        blend_cv = [float(np.mean([accuracy(va, w * p_hgb + (1 - w) * p_lgb) for va, p_hgb, p_lgb in fold_preds]))
                    for w in BLEND_WEIGHTS]
        best_w = float(BLEND_WEIGHTS[int(np.argmax(blend_cv))])
        best = {"hgb": best_hgb, "lgb": best_lgb, "w_hgb": best_w}

        # 3) evaluación única en el bloque final, nunca visto
        tr, te = split_by_target(frame, h, train_end=test_start - 1, val=(test_start, n_t))
        model = fit(tr, best)
        row = {
            "best_params": best, "cv_accuracy": {"hgb": cv_hgb, "lgb": cv_lgb, "blend_by_weight": blend_cv},
            "n_train": len(tr), "n_test": len(te),
            "model": accuracy(te, predict(model, te)),
            "naive_day": accuracy(te.dropna(subset=["_naive96"]), te.dropna(subset=["_naive96"])["_naive96"].to_numpy()),
            "naive_week": accuracy(te.dropna(subset=["_naive672"]), te.dropna(subset=["_naive672"])["_naive672"].to_numpy()),
        }
        # 4) ablaciones: ¿aporta el pronóstico de clima? ¿cuánto ganaría un evento conocido de antemano?
        for name, kw in {"sin_contexto": {"use_ctx": False}, "cota_evento_conocido": {"event_at_target": True}}.items():
            fa = make_frame(y, ctx, stations, h, **kw)
            tra, tea = split_by_target(fa, h, train_end=test_start - 1, val=(test_start, n_t))
            row[name] = accuracy(tea, predict(fit(tra, best), tea))
        # 5) champion vs candidato con la misma información: el candidato se reentrena con el
        # corte del champion y ambos se miden en los objetivos posteriores (fuera de muestra)
        if reference is not None and h in reference["models"]:
            ref_end_t = int(y.index.searchsorted(pd.Timestamp(reference["train_cutoff"]), side="right")) - 1
            tr_ref, sub = split_by_target(frame, h, train_end=ref_end_t, val=(ref_end_t + 1, n_t))
            if not sub.empty:
                row["same_block"] = {"n_targets": int(len(sub)), "from": str(y.index[min(ref_end_t + 1, n_t - 1)]),
                                     "model": accuracy(sub, predict(fit(tr_ref, best), sub)),
                                     "reference": accuracy(sub, predict(reference["models"][h], sub))}
        # 6) accuracy por estación en test
        pred = predict(model, te)
        err = pd.Series(np.abs(te["_y"].to_numpy() - pred)).groupby(te["station"].to_numpy()).sum()
        tot = te["_y"].groupby(te["station"].to_numpy()).sum()
        row["por_estacion"] = {y.columns[int(i)]: round(float(100 * (1 - err[i] / tot[i])), 2) for i in err.index}
        report["horizons"][f"h{h}"] = row

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# Periodos finales que el modelo de producción NO ve al entrenar: así la corrección
# de sesgo (que solo usa objetivos posteriores al corte) tiene ventana completa desde
# el primer ciclo tras un reentrenamiento, en vez de quedar apagada 2-4 ciclos.
PRODUCTION_HOLDOUT = 20  # > BIAS_WINDOW + max(HORIZONS); se conserva el valor validado


def production_train_end(n_t: int) -> int:
    """Último índice de objetivo que entra al entrenamiento de producción."""
    return n_t - 1 - PRODUCTION_HOLDOUT


def train_production(out_dir: Path, params_by_h: dict[int, dict], horizons=HORIZONS, data_dir=None):
    """Reentrena con TODOS los datos disponibles y guarda un modelo por horizonte."""
    y, ctx, stations = load_data(data_dir)
    return train_production_from_frames(y, ctx, stations, out_dir, params_by_h, horizons=horizons)


# Recetas de entrenamiento que compiten en sombra cuando se reentrena por drift
# (pulso_pipeline.shadow): la base, una que pesa más lo reciente (vida media en días) y
# una que solo usa la ventana reciente. Decide su desempeño en vivo, no una validación.
RECIPES = {
    "base": {},
    "reciente": {"half_life_days": 7},
    "ventana14": {"window_days": 14},
}


def recipe_rows(train: pd.DataFrame, h: int, train_end: int, recipe: str = "base"):
    """(filas, pesos) de entrenamiento para una receta de `RECIPES`."""
    cfg = RECIPES[recipe]
    age = (train_end - (train["_t"] + h)).to_numpy()  # periodos entre el objetivo y el corte
    if "window_days" in cfg:
        keep = age < cfg["window_days"] * DAY
        return train[keep], None
    if "half_life_days" in cfg:
        return train, 0.5 ** (age / (cfg["half_life_days"] * DAY))
    return train, None


def train_production_from_frames(y: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame, out_dir: Path,
                                  params_by_h: dict[int, dict], horizons=HORIZONS, recipe: str = "base") -> dict:
    """Igual que `train_production` pero a partir de frames ya cargados (p. ej. desde Supabase)."""
    import joblib
    train_end = production_train_end(len(y))
    models = {}
    for h in horizons:
        frame = make_frame(y, ctx, stations, h)
        tr, _ = split_by_target(frame, h, train_end=train_end, val=None)
        tr, weights = recipe_rows(tr, h, train_end, recipe)
        models[h] = fit(tr, params_by_h[h], sample_weight=weights)
    import lightgbm
    import sklearn
    meta = {"sklearn_version": sklearn.__version__, "lightgbm_version": lightgbm.__version__,
            "data_cutoff": str(y.index[train_end]), "horizons": list(horizons), "features": FEATURES_NOTE,
            "recipe": recipe,
            "trained_rows": {h: int(len(split_by_target(make_frame(y, ctx, stations, h), h, train_end, None)[0]))
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


BIAS_WINDOW = 8     # últimos 8 objetivos ya observados (2 h) por estación
BIAS_SHRINK = 0.5   # se aplica la mitad del sesgo medido
BIAS_CLIP = (0.7, 1.3)
# Antes había un "modo quiebre" que aplicaba el sesgo completo (hasta x3) cuando las
# ventanas de 2 h y 4 h coincidían. Con `align_level_shifts` los escalones ya se corrigen
# en la historia, y ese modo solo amplificaba oleadas pasajeras: el 18-sep de madrugada
# multiplicaba la predicción justo cuando la oleada ya había bajado. Replay sin él: 35.6 ->
# 41.9 en esa madrugada, -0.04/-0.05 en las ventanas de escalones del 11..18-sep.
#
# Factor común del sistema: mediana entre estaciones de sum(real)/sum(predicho) en la última
# hora, aplicado a medias. Las oleadas del 18-sep saltaban de estación en estación (x0.6 a
# x11 por estación y hora), pero la mediana entre estaciones se mantenía en x2-x3: el nivel
# general sí es predecible aunque el lugar de cada ráfaga no. Replay: 42.3 -> 46.4 durante
# las oleadas, +0.006 / -0.019 en las ventanas 11..14 y 14..18-sep.
COMMON_WINDOW = 4
COMMON_SHRINK = 0.5
COMMON_CLIP = (0.3, 4.0)


def correction_ratios(model, frame: pd.DataFrame, h: int, last_t: int,
                      train_end_t: int | None = None) -> tuple[pd.Series, float]:
    """Razones crudas de la corrección online en el origen `last_t` (objetivos t+h <= last_t,
    sin fuga): (sum(real)/sum(predicho) por estación en los últimos `BIAS_WINDOW` objetivos,
    mediana entre estaciones de esa razón en los últimos `COMMON_WINDOW`). Cómo se convierten
    en factor lo decide la política (`policy_factor`). Solo se usan objetivos posteriores a
    `train_end_t` para no medir residuos in-sample."""
    tgt = frame["_t"] + h
    ok = (tgt <= last_t) & frame["_y"].notna() & frame["_scale"].notna()
    if train_end_t is not None:
        ok &= tgt > train_end_t
    past = frame[ok & (tgt > last_t - max(BIAS_WINDOW, COMMON_WINDOW))]
    if past.empty:
        return pd.Series(dtype=float), np.nan
    df = pd.DataFrame({"station": past["station"].astype(int).to_numpy(), "t": (past["_t"] + h).to_numpy(),
                       "y": past["_y"].to_numpy(), "p": predict(model, past)})

    def _ratio(window: int) -> pd.Series:
        g = df[df["t"] > last_t - window].groupby("station")
        sums = g[["y", "p"]].sum()[g.size() == window]
        return sums["y"] / sums["p"].where(sums["p"] > 0)

    pooled = _ratio(COMMON_WINDOW).dropna()
    return _ratio(BIAS_WINDOW).dropna(), float(pooled.median()) if len(pooled) else np.nan


def recent_bias_factors(model, frame: pd.DataFrame, h: int, last_t: int, train_end_t: int | None = None) -> pd.Series:
    """Factor de corrección por estación (propio x común, política por defecto, sin nowcast).

    Motivo: la plataforma cambia patrones de demanda durante la competencia y el
    modelo tarda en reaccionar hasta el siguiente reentrenamiento. Validado en
    dos ventanas en vivo independientes (2026-09-09..14): +0.25 a +1.71 pts en
    8/8 combinaciones horizonte x ventana (media +0.88)."""
    own, common = correction_ratios(model, frame, h, last_t, train_end_t)
    stations = np.arange(len(frame["station"].cat.categories))
    r_own = pd.Series(stations).map(own).to_numpy(dtype=float)
    f = policy_factor(r_own, np.full(len(stations), common), np.full(len(stations), np.nan),
                      np.full(len(stations), h))
    return pd.Series(f, index=stations)


# Nowcast por estación: durante las oleadas del 18-sep cada ráfaga duraba ~45 min en una
# estación (autocorrelación del exceso: 0.88 a 15 min, 0.64 a 30, ~0 a 60). La razón
# real/predicho de la última media hora a +15 min anticipa los horizontes cortos; se aplica
# sobre la corrección vigente, a media fuerza y con peso que decae hasta 0 a +60 min.
# Replay: +1.66 en la ventana de las oleadas, +0.05 / -0.01 en las ventanas 11..18-sep.
NOWCAST_WINDOW = 2
NOWCAST_STRENGTH = 0.5
NOWCAST_DECAY = {1: 0.88, 2: 0.64, 3: 0.3, 4: 0.0}
NOWCAST_CLIP = (0.33, 3.0)


# Política de corrección: cuánto de cada razón se aplica. La de por defecto es la validada;
# `pulso_pipeline.policy` puede elegir otra de `POLICIES` según el desempeño reciente.
DEFAULT_POLICY = {"bias_shrink": BIAS_SHRINK, "common_shrink": COMMON_SHRINK, "nowcast_strength": NOWCAST_STRENGTH}
# Lista fija y corta a propósito: un selector con muchas opciones persigue ruido. Simulación
# del selector en el replay (6 ciclos, margen 0.5): +0.37 en las oleadas del 18-sep,
# -0.04 / +0.01 en las ventanas 11..14 y 14..18-sep frente a dejar fija la estándar.
DEFAULT_POLICY_NAME = "estandar"
POLICIES = {
    "estandar": DEFAULT_POLICY,
    "reactiva": {"bias_shrink": 0.75, "common_shrink": 0.75, "nowcast_strength": 0.75},
    "tranquila": {"bias_shrink": 0.25, "common_shrink": 0.25, "nowcast_strength": 0.25},
    "nowcast": {"bias_shrink": 0.5, "common_shrink": 0.5, "nowcast_strength": 1.0},
}


def policy_factor(r_own, r_common, r_now, h, policy: dict | None = None) -> np.ndarray:
    """Factor final (vectorizado) a partir de las razones crudas y el horizonte."""
    p = policy or DEFAULT_POLICY
    r_own, r_common, r_now = (np.asarray(v, dtype=float) for v in (r_own, r_common, r_now))
    own = np.where(np.isnan(r_own), 1.0, 1 + p["bias_shrink"] * (np.clip(r_own, *BIAS_CLIP) - 1))
    common = np.where(np.isnan(r_common), 1.0, 1 + p["common_shrink"] * (np.clip(r_common, *COMMON_CLIP) - 1))
    f = own * common
    decay = np.array([NOWCAST_DECAY.get(int(x), 0.0) for x in np.asarray(h).ravel()]).reshape(np.shape(h))
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = np.where(np.isnan(r_now), 1.0, (r_now / f) ** (p["nowcast_strength"] * decay))
    return f * adj


def nowcast_ratios(models: dict, y: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame,
                   train_end_t: int | None) -> pd.Series:
    """Razón real/predicho (modelo +15 min) de los últimos `NOWCAST_WINDOW` objetivos ya
    observados, por estación (índice = posición de la estación). Solo usa datos <= origen."""
    last_t = len(y) - 1
    ext_idx = y.index.append(pd.date_range(y.index[-1] + pd.Timedelta(minutes=STEP_MIN), periods=1,
                                           freq=f"{STEP_MIN}min"))
    frame = make_frame(y.reindex(ext_idx), ctx.reindex(ext_idx), stations, 1)
    tgt = frame["_t"] + 1
    ok = (tgt <= last_t) & (tgt > last_t - NOWCAST_WINDOW) & frame["_y"].notna() & frame["_scale"].notna()
    if train_end_t is not None:
        ok &= tgt > train_end_t
    past = frame[ok]
    if past.empty:
        return pd.Series(dtype=float)
    g = pd.DataFrame({"station": past["station"].astype(int).to_numpy(), "y": past["_y"].to_numpy(),
                      "p": predict(models[1], past)}).groupby("station")[["y", "p"]].sum()
    return (g["y"] / g["p"].where(g["p"] > 0)).clip(*NOWCAST_CLIP).dropna()


# Selector de expertos. El drift de la continuación (revisiones 2 y 3 del docente) cambia la
# *forma* temporal de la demanda: el 18-sep ~05 h UTC pasó del perfil diario a una oscilación
# de 4 h por estación y el champion (perfiles diarios, entrenado con semanas del régimen viejo)
# cayó a ~40 % por ciclo. Reentrenar el champion no alcanza: sus datos siguen siendo casi todos
# del régimen viejo. En cada ciclo compiten tres expertos y se usa el que mejor predijo los dos
# ciclos anteriores ya observados (recalculados desde esos cortes, sin fuga):
#   - "champion": el modelo entrenado, con su corrección online (lo de siempre);
#   - "ciclica16" / "ciclica16xK": y[t+h-16], la demanda del mismo punto 4 h antes, o el
#     promedio de las últimas K oscilaciones (K <= `CYCLE_MAX_PERIODS`). Copiar una sola
#     oscilación arrastra su ruido; al madurar el régimen promediar más gana (18-sep 18 h:
#     93.1 con K=3 vs 91.1 con K=1) y el selector va subiendo K solo;
#   - "periodica" / "periodicaxK": lo mismo con el periodo dominante detectado en las últimas
#     4 h (`detect_period`), por si un cambio futuro trae otra oscilación. En el replay detecta
#     16 desde las 10 h del 18-sep sin saberlo de antemano y no cambia nada en días normales;
#   - "adaptativo" / "adaptativo6h": Extra Trees reentrenado en cada ciclo con las últimas 12 h
#     (`ONLINE_WINDOW`) o 6 h y solo rezagos recientes, así aprende cualquier forma nueva en
#     horas. El de 6 h se recupera antes tras un cambio (18-sep 10-11 h: 86-91 vs 71-72);
#   - "plantilla" / "plantillaxK": las 12 estaciones repiten la misma onda de 4 h, solo
#     desfasada (0/4/8/12 cuartos) y escalada por su nivel. Se estima una forma común con
#     las últimas K oscilaciones de todas las estaciones alineadas, y cada una la usa con su
#     fase y su nivel: 12 veces más datos que su propia historia. Replay del 18-19 sep:
#     régimen maduro 93.13 vs 92.59, y con una sola oscilación ya da 93.8 a las 10 h del 18.
# El champion se abandona solo si otro le gana por más de `EXPERT_MARGIN` puntos.
CYCLE_PERIOD = 16
CYCLE_MAX_PERIODS = 6
ONLINE_WINDOW = 48                 # 12 h de orígenes de entrenamiento
ONLINE_LAGS = (*range(24), 28, 32, 40, 48)
ONLINE_SEASONAL = (16, 32, 48, 96)
ONLINE_SCALE = DAY                 # nivel de la estación: media de las últimas 24 h
EXPERT_SCORE_CYCLES = 2            # ciclos anteriores (4 objetivos cada uno) para puntuar
# El ciclo más reciente pesa más: tras un cambio se suelta antes el experto que dejó de
# servir. Replay: +1.1 en la transición del 18-sep, días normales iguales (84.34).
EXPERT_SCORE_WEIGHTS = (0.7, 0.3)
ONLINE_FAST_WINDOW = 24
EXPERT_MARGIN = 3.0
PERIOD_RANGE = range(8, 49)        # 2 h a 12 h
PERIOD_WINDOW = 16                 # objetivos recientes con los que se elige el periodo


def _wape_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Métrica del reto sobre matrices (objetivos x estaciones)."""
    err, tot = np.nansum(np.abs(y_true - y_pred), axis=0), np.nansum(y_true, axis=0)
    ok = tot > 0
    return float(np.clip(100 * (1 - err[ok] / tot[ok]), 0, None).mean()) if ok.any() else np.nan


def detect_period(Y: np.ndarray, origin_t: int) -> int:
    """Periodo P (en cuartos de hora) para el que y[t-P] mejor explica los últimos
    `PERIOD_WINDOW` objetivos observados hasta `origin_t`."""
    win = np.arange(origin_t - PERIOD_WINDOW + 1, origin_t + 1)
    scores = {P: _wape_accuracy(Y[win], Y[win - P]) for P in PERIOD_RANGE}
    scores = {P: v for P, v in scores.items() if np.isfinite(v)}
    return max(scores, key=scores.get) if scores else CYCLE_PERIOD


def _online_features(Y: np.ndarray, origins, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Features del experto adaptativo en cada origen t (solo Y[<= t]), apiladas por estación."""
    n_s = Y.shape[1]
    X, S = [], []
    for t in origins:
        sc = np.nanmean(Y[t - ONLINE_SCALE + 1:t + 1], axis=0) + 1.0
        cols = [Y[t - j] / sc for j in ONLINE_LAGS]
        cols += [Y[t + h - k] / sc for k in ONLINE_SEASONAL]  # k >= 16 > h: ya observado
        cols += [np.nanmean(Y[t - 3:t + 1], axis=0) / sc, np.nanmean(Y[t - 15:t + 1], axis=0) / sc,
                 np.full(n_s, h), np.arange(n_s)]
        X.append(np.stack(cols, axis=1))
        S.append(sc)
    return np.nan_to_num(np.concatenate(X)), np.concatenate(S)


def adaptive_forecast(Y: np.ndarray, origin_t: int, horizons, window: int = ONLINE_WINDOW) -> dict[int, np.ndarray]:
    """{h: predicción por estación} del experto adaptativo, entrenado solo con objetivos <= origen."""
    from sklearn.ensemble import ExtraTreesRegressor
    out = {}
    for h in horizons:
        train_origins = np.arange(origin_t - window, origin_t - h + 1)
        X, sc = _online_features(Y, train_origins, h)
        target = np.concatenate([Y[t + h] for t in train_origins]) / sc
        ok = np.isfinite(target)
        mdl = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=0)
        mdl.fit(X[ok], target[ok])
        Xc, scc = _online_features(Y, [origin_t], h)
        out[h] = np.clip(mdl.predict(Xc), 0, None) * scc
    return out


def template_forecast(Y: np.ndarray, origin_t: int, horizons, k: int, period: int = CYCLE_PERIOD) -> dict[int, np.ndarray]:
    """{h: predicción por estación} con la forma de onda común a todas las estaciones."""
    prof = np.nanmean(Y[origin_t - period * k + 1:origin_t + 1].reshape(k, period, -1), axis=0)
    level = prof.mean(axis=0)
    shape = prof / np.where(level > 0, level, np.nan)
    ref = shape[:, np.nanargmax(level)]  # la estación más grande, la menos ruidosa
    shifts = [min(range(period), key=lambda s: np.nansum(np.abs(np.roll(shape[:, j], -s) - ref)))
              for j in range(shape.shape[1])]
    common = np.nanmean([np.roll(shape[:, j], -s) for j, s in enumerate(shifts)], axis=0)
    # la posición h-1 de la ventana es la fase del objetivo t+h (la ventana termina en t)
    return {h: np.array([common[(h - 1 - s) % period] for s in shifts]) * level for h in horizons}


def _champion_components(models, y, ctx, stations, horizons, train_end_t, policy) -> dict[int, pd.DataFrame]:
    """Predicción del champion (con corrección online) desde el último índice de `y`."""
    origin = y.index[-1]
    now = nowcast_ratios(models, y, ctx, stations, train_end_t) if 1 in models else pd.Series(dtype=float)
    out = {}
    for h in horizons:
        ext_idx = y.index.append(pd.date_range(origin + pd.Timedelta(minutes=STEP_MIN), periods=h,
                                               freq=f"{STEP_MIN}min"))
        frame = make_frame(y.reindex(ext_idx), ctx.reindex(ext_idx), stations, h)
        cur = frame[frame["_t"] == len(y) - 1]
        base = predict(models[h], cur)
        own, common = correction_ratios(models[h], frame, h, len(y) - 1, train_end_t)
        st_idx = cur["station"].astype(int)
        comp = pd.DataFrame({"base": base, "r_own": st_idx.map(own).to_numpy(dtype=float),
                             "r_common": common, "r_now": st_idx.map(now).to_numpy(dtype=float)},
                            index=stations.index[st_idx])
        comp["value"] = comp["base"] * policy_factor(comp["r_own"], comp["r_common"], comp["r_now"],
                                                     np.full(len(comp), h), policy)
        out[h] = comp.reindex(stations.index)
    return out


def _expert_values(name: str, models, y, ctx, stations, Y: np.ndarray, origin_t: int, horizons,
                   train_end_t, policy):
    """{h: valores por estación} de un experto desde el origen `origin_t` (índice en `y`)."""
    if name == "champion":
        comps = _champion_components(models, y.iloc[:origin_t + 1], ctx.iloc[:origin_t + 1], stations,
                                     horizons, train_end_t, policy)
        return {h: c["value"].to_numpy(dtype=float) for h, c in comps.items()}
    if name.startswith(("ciclica16", "periodica")):
        period = CYCLE_PERIOD if name.startswith("ciclica16") else detect_period(Y, origin_t)
        k = int(name.split("x")[1]) if "x" in name else 1
        return {h: np.mean([Y[origin_t + h - period * j] for j in range(1, k + 1)], axis=0) for h in horizons}
    if name.startswith("plantilla"):
        return template_forecast(Y, origin_t, horizons, int(name.split("x")[1]) if "x" in name else 1)
    if name == "adaptativo6h":
        return adaptive_forecast(Y, origin_t, horizons, window=ONLINE_FAST_WINDOW)
    return adaptive_forecast(Y, origin_t, horizons)


EXPERTS = ("champion", "adaptativo", "adaptativo6h",
           "ciclica16", *(f"ciclica16x{k}" for k in range(2, CYCLE_MAX_PERIODS + 1)),
           "periodica", *(f"periodicax{k}" for k in range(2, CYCLE_MAX_PERIODS + 1)),
           "plantilla", *(f"plantillax{k}" for k in range(2, CYCLE_MAX_PERIODS + 1)))


def select_expert(models, y, ctx, stations, Y: np.ndarray, train_end_t, policy) -> tuple[str, str, dict]:
    """Experto para el ciclo actual según la accuracy en los `EXPERT_SCORE_CYCLES` ciclos previos."""
    last_t = len(y) - 1
    hs = sorted(models)
    min_hist = max(ONLINE_SCALE + max(ONLINE_SEASONAL) + ONLINE_WINDOW, max(PERIOD_RANGE) * CYCLE_MAX_PERIODS
                   + PERIOD_WINDOW) \
        + 4 * EXPERT_SCORE_CYCLES
    if last_t < min_hist or not all(h in models for h in (1, 2, 3, 4)):
        return "champion", "historia insuficiente para puntuar expertos: champion", {}
    scores: dict[str, list[float]] = {e: [] for e in EXPERTS}
    for k in range(1, EXPERT_SCORE_CYCLES + 1):
        o = last_t - 4 * k
        truth = np.stack([Y[o + h] for h in hs])
        for e in EXPERTS:
            try:
                vals = _expert_values(e, models, y, ctx, stations, Y, o, hs, train_end_t, policy)
                scores[e].append(_wape_accuracy(truth, np.stack([vals[h] for h in hs])))
            except Exception:  # noqa: BLE001 - un experto que falla no compite
                scores[e].append(np.nan)
    w = np.asarray(EXPERT_SCORE_WEIGHTS[:EXPERT_SCORE_CYCLES])
    mean = {e: float(np.dot(w, v) / w.sum()) if np.isfinite(v).all() else np.nan for e, v in scores.items()}
    valid = {e: v for e, v in mean.items() if np.isfinite(v)}
    summary = f"periodo {detect_period(Y, last_t)}; " + ", ".join(f"{e} {v:.1f}" for e, v in sorted(valid.items(), key=lambda kv: -kv[1]))
    if not np.isfinite(mean["champion"]):
        best = max(valid, key=valid.get) if valid else "champion"
        return best, f"experto {best} (champion sin puntaje; {summary})", mean
    best = max(valid, key=valid.get)
    if best != "champion" and valid[best] > mean["champion"] + EXPERT_MARGIN:
        return best, f"experto {best} supera al champion por {valid[best] - mean['champion']:.1f} ({summary})", mean
    return "champion", f"experto champion ({summary})", mean


def forecast_for_targets(models: dict, y: pd.DataFrame, ctx: pd.DataFrame, stations: pd.DataFrame,
                          origin, targets, train_cutoff=None, policy: dict | None = None,
                          y_raw: pd.DataFrame | None = None) -> pd.DataFrame:
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
    train_end_t = None
    if train_cutoff is not None:
        train_end_t = int(y.index.searchsorted(pd.Timestamp(train_cutoff), side="right")) - 1
    hs = [int(h) for h in by_h]
    Y = (y if y_raw is None else y_raw).reindex(index=y.index, columns=stations.index).to_numpy(dtype=float)
    try:
        expert, reason, _ = select_expert(models, y, ctx, stations, Y, train_end_t, policy)
    except Exception as exc:  # noqa: BLE001 - ante cualquier duda, el champion
        expert, reason = "champion", f"champion (no se pudo puntuar expertos: {exc})"
    print(reason)
    if len(reason) > 400:  # el registro guarda el resumen; los peores expertos sobran
        reason = reason[:397] + "..."
    preds_by_h = _champion_components(models, y, ctx, stations, hs, train_end_t, policy)
    if expert != "champion":
        vals = _expert_values(expert, models, y, ctx, stations, Y, len(y) - 1, hs, train_end_t, policy)
        if all(np.isfinite(vals[h]).all() for h in hs):  # sin corrección: razones NaN -> factor 1
            nan = np.full(len(stations), np.nan)
            for h in hs:
                comp = pd.DataFrame({"base": np.clip(vals[h], 0, None), "r_own": nan, "r_common": nan,
                                     "r_now": nan}, index=stations.index)
                comp["value"] = comp["base"]
                preds_by_h[h] = comp

    rows = []
    for station_id, target_at, h in parsed:
        if station_id not in preds_by_h[h].index:
            raise ValueError(f"estación desconocida en el modelo: {station_id}")
        c = preds_by_h[h].loc[station_id]
        rows.append({"station_id": station_id, "target_at": target_at, "horizon_steps": h,
                     "value": round(float(c["value"]), 2), "base": float(c["base"]),
                     "r_own": float(c["r_own"]), "r_common": float(c["r_common"]), "r_now": float(c["r_now"])})
    out = pd.DataFrame(rows)
    out.attrs["expert"], out.attrs["expert_reason"] = expert, reason  # evidencia de la decisión
    return out

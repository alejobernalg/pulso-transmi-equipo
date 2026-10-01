"""Garantía anti-fuga: las features del origen t no dependen de nada posterior a t.

Se corrompe con ruido todo lo que ocurre después de t (demanda y clima/evento reales)
y se exige que la fila de features de t sea idéntica. Los pronósticos de clima
(`*_forecast`) se dejan intactos porque son información disponible antes de t+h.
"""
import numpy as np
import pandas as pd
import pytest

from pulso_forecast import model as pm

y, ctx, stations = pm.load_data()
rng = np.random.default_rng(0)
ORIGINS = [800, 1500, 2600, 3000, 4000]
ACTUAL_COLS = ["rain_mm", "temperature_c", "event_intensity"]


@pytest.mark.parametrize("h", pm.HORIZONS)
@pytest.mark.parametrize("t", ORIGINS)
def test_features_do_not_see_the_future(h, t):
    base = pm.make_frame(y, ctx, stations, h)
    y2, c2 = y.copy(), ctx.copy()
    y2.iloc[t + 1:] = rng.uniform(1e4, 1e5, y2.iloc[t + 1:].shape)
    for col in ACTUAL_COLS:
        c2.iloc[t + 1:, c2.columns.get_loc(col)] = rng.uniform(0.2, 1e3, len(c2) - t - 1)
    dirty = pm.make_frame(y2, c2, stations, h)

    cols = pm.feature_columns(base)
    a = base[base["_t"] == t][cols].reset_index(drop=True).astype(float)
    b = dirty[dirty["_t"] == t][cols].reset_index(drop=True).astype(float)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_no_future_information_in_target_alignment():
    # el objetivo de la fila t es exactamente y[t+h]; nada más
    h = 4
    f = pm.make_frame(y, ctx, stations, h)
    t = 1000
    row = f[(f["_t"] == t) & (f["station"] == 0)].iloc[0]
    assert row["_y"] == y.iloc[t + h, 0]


def test_seasonal_lags_only_use_the_past():
    # para h=96, seas96 = y[t+h-96] = y[t]: justo el límite permitido
    f = pm.make_frame(y, ctx, stations, 96)
    t = 2000
    row = f[(f["_t"] == t) & (f["station"] == 0)].iloc[0]
    assert row["seas96"] * row["_scale"] == pytest.approx(y.iloc[t, 0])


@pytest.mark.parametrize("h", [1, 4])
def test_bias_correction_only_uses_observed_targets(h):
    """El factor de corrección en el origen t solo puede usar objetivos t'+h <= t."""
    t = 3000
    frame = pm.make_frame(y, ctx, stations, h)
    tr, _ = pm.split_by_target(frame, h, train_end=t - 200, val=None)
    model = pm.fit(tr, {"hgb": {"max_iter": 20}, "lgb": {"n_estimators": 20}, "w_hgb": 0.5})
    base = pm.recent_bias_factors(model, frame, h, t, train_end_t=t - 200)
    y2 = y.copy()
    y2.iloc[t + 1:] = rng.uniform(1e4, 1e5, y2.iloc[t + 1:].shape)
    dirty = pm.recent_bias_factors(model, pm.make_frame(y2, ctx, stations, h), h, t, train_end_t=t - 200)
    assert len(base) == len(stations)
    pd.testing.assert_series_equal(base, dirty)
    lo, hi = (1 + pm.BIAS_SHRINK * (c - 1) for c in pm.BIAS_CLIP)
    clo, chi = (1 + pm.COMMON_SHRINK * (c - 1) for c in pm.COMMON_CLIP)
    assert base.between(lo * clo, hi * chi).all()


def test_peak_shift_detected_and_aligned(monkeypatch):
    """Un pico corrido +45 min en los últimos días se detecta y el historial previo se alinea."""
    start = y.index[-6 * pm.DAY]
    monkeypatch.setattr(pm, "COMPETITION_START", y.index[-8 * pm.DAY])
    y2 = y.copy()
    sid = y.columns[0]
    y2.loc[y2.index >= start, sid] = y[sid].shift(3)[y2.index >= start]
    shifts = pm.detect_peak_shifts(y2)
    assert set(shifts) == {sid} and shifts[sid][0] == 3
    assert pm.detect_peak_shifts(pm.align_peak_shifts(y2)) == {}
    assert pm.detect_peak_shifts(y) == {}


def test_level_shift_detected_and_aligned(monkeypatch):
    """Un salto de nivel x2.5 en una estación se detecta cerca de donde ocurrió y la historia
    previa se reescala al nivel nuevo; las demás estaciones no se tocan."""
    monkeypatch.setattr(pm, "COMPETITION_START", y.index[-8 * pm.DAY])
    start = y.index[-2 * pm.DAY]
    y2 = y.copy()
    sid = y.columns[0]
    y2.loc[y2.index >= start, sid] *= 2.5
    shifts = pm.detect_level_shifts(y2)
    assert set(shifts) == {sid}
    assert abs(shifts[sid][-1] - start) <= pd.Timedelta(hours=2)
    aligned = pm.align_level_shifts(y2)
    before = aligned.index < start - pd.Timedelta(hours=2)
    ratio = aligned.loc[before, sid].sum() / y.loc[before, sid].sum()
    assert ratio == pytest.approx(2.5, rel=0.1)
    pd.testing.assert_frame_equal(aligned.drop(columns=sid), y2.drop(columns=sid))
    assert pm.detect_level_shifts(y) == {}


def test_night_surge_is_not_a_level_shift(monkeypatch):
    """Una oleada de madrugada (x4 durante unas horas nocturnas) no se toma por un escalón."""
    monkeypatch.setattr(pm, "COMPETITION_START", y.index[-8 * pm.DAY])
    local = y.index.tz_convert(pm.TZ)
    night = (y.index > y.index[-pm.DAY]) & ((local.hour >= 23) | (local.hour < 4))
    y2 = y.copy()
    y2.loc[night, y.columns[0]] *= 4
    assert pm.detect_level_shifts(y2.loc[: y.index[night][-1]]) == {}


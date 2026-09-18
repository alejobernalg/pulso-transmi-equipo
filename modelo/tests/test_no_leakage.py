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

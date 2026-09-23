# pulso-forecast

Modelo de pronóstico de demanda (cada 15 min, 12 estaciones) para el reto Pulso TransMi.
Gradient boosting por horizonte (+15, +30, +45, +60 min, el contrato oficial de cada
ciclo) sobre features **solo del pasado**.

## Uso

```bash
pip install pulso_forecast-0.1.0-py3-none-any.whl
export PULSO_DATA_DIR=/ruta/a/data          # observations.csv, context.csv, stations.csv

pulso-forecast predict --model model.joblib --out salida/   # usa el modelo incluido
pulso-forecast train   --out artifacts/                     # reentrena con todos los datos
pulso-forecast evaluate --out artifacts/                    # solo validación temporal
```

`predict` escribe `next_forecast.csv` (station_id, origin_at, horizon_steps, target_at, value)
para el último instante de `observations.csv`.

## Métricas (últimos 7 días, nunca vistos)

| Horizonte | Accuracy | Baseline semanal |
|---|---:|---:|
| +15 min | 88,20 | 83,11 |
| +30 min | 88,17 | 83,11 |
| +45 min | 88,16 | 83,11 |
| +60 min | 88,08 | 83,11 |

## Sin fuga de datos

Toda feature del origen `t` usa datos con marca ≤ `t`; del futuro solo entran el calendario y
los pronósticos de clima. `tests/test_no_leakage.py` reemplaza por ruido todo lo posterior a `t`
y exige que las features no cambien (`pytest`).

## Limitaciones

- Entrenado con 45 días: hay que reentrenar cuando lleguen datos nuevos (deriva leve al alza).
- `model.joblib` requiere scikit-learn 1.x compatible con el usado al entrenar (ver `model_meta.json`).

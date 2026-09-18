# Pulso TransMi — documentación del proyecto

Estado a **2026-09-18**. Resume lo construido, cómo reproducirlo, las decisiones
tomadas y lo que queda pendiente. Detalle del esquema en
[data-model.md](data-model.md); guía del reto en [student-project.md](student-project.md).

## 1. Resumen

| Tema | Estado |
|---|---|
| Datos iniciales descargados y cargados en Supabase | Hecho: 68.172 filas |
| Modelo de demanda a 4 horizontes, validado en el tiempo | Hecho: 87,7–88,2 % de accuracy |
| Pruebas anti-fuga de información | Hecho: 22 pruebas verdes |
| Paquete instalable `pulso-forecast` con CLI | Hecho (`modelo/`) |
| Envío al ciclo de práctica de la API | Hecho: aceptado, 12/12 |
| Pipeline automático (GitHub Actions), monitoreo, reentrenamiento | **Pendiente** |
| Persistencia de corridas, métricas y predicciones en Supabase | **Pendiente** (tablas creadas, vacías) |

## 2. El reto y la API

Pronóstico de demanda sintética cada 15 minutos para 12 estaciones de TransMilenio.
La API pública (`PULSO_API_URL`, versión **0.4.1**) expone lectura de datos y, desde
0.4.1, el modo competencia:

| Ruta | Uso |
|---|---|
| `/v1/stations`, `/v1/observations`, `/v1/context` | Datos (paginados, cursor opaco) |
| `/v1/downloads/{archivo}` | CSV completos con SHA-256 en `/v1/meta` |
| `/v1/clock`, `/v1/forecast-cycles/current` | Reloj de la competencia y ciclo abierto |
| `POST /v1/submissions` | Envío de predicciones (API key + `Idempotency-Key`) |
| `/v1/leaderboard` | Posiciones; ventana `cumulative` o `rolling_24h` |
| `/v1/me`, portal web (`/`) | Identidad y generación de la API key |

Contrato de envío (`schema_version` 1.0): `cycle_id`, `client_run_id`, `data_cutoff`,
`model {version, trained_at, training_data_end, git_commit}` y de 1 a 100
`predictions {station_id, target_at, value}`.

**Lo que el profesor no ha publicado:** los horizontes definitivos de la ronda
competitiva. El ciclo de práctica pide solo 15 min (12 predicciones). Se asumieron
15 min, 1 h, 4 h y 24 h; se cambian en `HORIZONS` (`modelo/src/pulso_forecast/model.py`).

## 3. Datos y hallazgos del EDA

`data/` (ignorado por git) contiene `stations.csv` (12), `context.csv` (4.320) y
`observations.csv` (51.840): 45 días, del 2026-07-26 al 2026-09-08, hora de Bogotá.

Verificado: sin nulos, sin duplicados en claves, sin estaciones huérfanas, series
completas y `context` alineado con `observations`.

Hallazgos que condicionaron el modelo ([examples/03_eda.ipynb](../examples/03_eda.ipynb)):

- La demanda tiene doble pico (≈7 h y ≈17 h) y cae ~22 % en fin de semana; hora y tipo
  de día interactúan.
- El lag de **una semana** predice mejor que el de un día.
- La temperatura es el ciclo diario disfrazado (correlación −0,88 con la hora); la
  lluvia casi no explica la demanda.
- `event_intensity` trae valores denormales (~1e-323): `> 0` marca el 89,5 % de los
  periodos. Hay que **umbralizar** (se usa 0,1).
- Hay una deriva leve: las 12 estaciones suben entre 0,8 % y 4,6 % en la última semana.
- Los festivos colombianos (7 y 17 de agosto) **no** muestran caída de demanda; el
  generador no los modela, así que no hay feature de festivos.

## 4. Base de datos (Supabase)

Proyecto `pzflgohfvzjmxbexcwst`. Esquema aplicado con la migración `pulso_transmi_schema`:
tablas del reto (`stations`, `context`, `observations`) y de trazabilidad MLOps
(`pipeline_runs`, `model_versions`, `model_metrics`, `predictions`, `drift_signals`).
Diagrama ER, revisión del diseño y DDL en [data-model.md](data-model.md).

Estado verificado el 2026-09-18:

| Tabla | Filas | RLS | Policies |
|---|---:|---|---:|
| `stations` | 12 | sí | 0 |
| `context` | 4.320 | sí | 0 |
| `observations` | 51.840 | sí | 0 |
| tablas MLOps (5) | 0 | sí | 0 |

Sin policies, `anon` no lee nada; solo accede el pipeline con la service key.
Los datos se cargaron con una edge function temporal que descarga los CSV de la API,
verifica el SHA-256 y hace `upsert`. Comparado con los CSV: mismos conteos y misma suma
de demanda (18.482.146). Los hashes coinciden con `data/metadata.json`.

> **Pendiente:** la edge function `load-pulso-data` **sigue desplegada** (usa la service
> role y se puede invocar con la anon key, que es pública). Borrarla desde el dashboard
> de Supabase, en Edge Functions.

## 5. El modelo

Código en [`modelo/`](../modelo/), paquete `pulso-forecast`.

### 5.1 Diseño

Un modelo por horizonte (**directo**, sin encadenar predicciones): `HistGradientBoostingRegressor`
con pérdida L1. Se predice la razón `demanda(t+h) / nivel_reciente`, donde el nivel es la media
móvil de 7 días hasta `t`. Eso normaliza estaciones grandes y pequeñas (la métrica pesa igual
a todas) y absorbe la deriva de nivel.

Features, todas con información hasta `t`:

- lags de 0 a 96 periodos, medias y desviación móviles;
- estacionalidad **alineada al objetivo**: `y(t+h-96)`, `y(t+h-672)`, dos semanas, etc.;
- perfil del slot objetivo: promedio de 4 semanas con vecinos ±1 slot, y promedio/mediana de
  los últimos 14 días **del mismo tipo** (laboral o fin de semana);
- calendario del instante objetivo (slot, día, fin de semana, seno/coseno de la hora);
- pronósticos de lluvia y temperatura en `t+h`; `event_intensity` umbralizado en `t`;
- estación (categórica) y coordenadas.

### 5.2 Protocolo anti-fuga

| Riesgo | Control |
|---|---|
| Features que miran el futuro | Solo pasado. Del futuro únicamente calendario y pronósticos de clima, que existen antes de `t+h`. |
| Lags que cruzan el origen | La estacionalidad exige `k >= h`: `y(t+h-k)` ya ocurrió en `t`. |
| Objetivos que cruzan el corte de entrenamiento | Purga: se entrena solo con `t+h <= corte`. |
| Partición aleatoria | Prohibida; todo es temporal. |
| Selección de hiperparámetros con el test | Se eligen con 2 folds temporales previos al bloque final. |
| Clima/eventos reales | No se usan más allá de `t`. `event_intensity(t+h)` solo como ablación. |

[`modelo/tests/test_no_leakage.py`](../modelo/tests/test_no_leakage.py) reemplaza por ruido
toda la demanda y el clima real posteriores a `t` y exige que las features de `t` no cambien
(22 pruebas, 4 horizontes × 5 orígenes más chequeos de alineación).

### 5.3 Resultados

Bloque de test: últimos 7 días (2026-09-02 a 09-08), visto una sola vez. Métrica: accuracy
`100 × (1 − WAPE)` por estación y promedio.

| Horizonte | Modelo | Lag semanal | Lag diario |
|---|---:|---:|---:|
| 15 min | 88,20 | 83,11 | 77,89 |
| 1 h | 88,08 | 83,11 | 77,89 |
| 4 h | 88,01 | 83,11 | 77,89 |
| 24 h | 87,72 | 83,11 | 77,89 |

Por estación (15 min): de 86,5 (`02300`, la más débil) a 89,4 (`05100`).

Ablaciones en test:

- **Sin pronósticos de clima:** ±0,08 puntos; el clima casi no aporta.
- **Con `event_intensity(t+h)` (cota):** +0,02 a +0,25 puntos; no se usa porque no se sabe si la
  agenda de eventos existe de antemano.

### 5.4 Por qué el techo parece cercano

La precisión es casi igual a 15 min que a 24 h y los residuos no tienen autocorrelación
positiva (≈ −0,1). Es la firma de un perfil estable más ruido independiente de ~12 %. Por eso
lo que funcionó fue estimar mejor el perfil (+0,5 a +0,7 puntos en CV), no añadir lags; quitar
los lags no ayudó. Es una lectura de los datos, no una garantía: no se conoce el generador.

## 6. Uso del paquete

```bash
pip install modelo/dist/pulso_forecast-0.1.0-py3-none-any.whl   # o: pip install -e modelo
export PULSO_DATA_DIR=data
pulso-forecast evaluate --out artifacts/    # validación temporal
pulso-forecast train    --out artifacts/    # reentrena con todo y guarda model.joblib
pulso-forecast predict  --model artifacts/model.joblib --out salida/
pytest modelo/tests
```

`dist/` y `artifacts/` no se versionan; se regeneran con `python -m build --wheel modelo`. El
bundle (`pulso-forecast-0.1.0-bundle.zip`) incluye wheel, modelo, métricas y README.
`model.joblib` depende de scikit-learn; `model_meta.json` registra la versión (1.9.1).

Verificado en un entorno limpio: instalar el wheel, entrenar por CLI y predecir reproduce las
métricas y las predicciones con diferencia 0,0.

## 7. Envío al ciclo de práctica

| | |
|---|---|
| Ciclo | `cyc_practice_20260918` (cierra 2026-09-19 04:59 UTC) |
| Envío | `sub_2269e4fe09da4a4581fba2833797bfef`, `accepted`, intento 1, 12/12, oficial |
| Recibido | 2026-09-18 20:41 UTC |
| Contenido | Predicciones a 15 min, origen 2026-09-09 04:45Z, modelo `pulso-forecast-0.1.0` |
| `git_commit` | `null`: el código del modelo no estaba commiteado al enviar |

No hay puntaje todavía: el objetivo real aún no aparece en los datos de la API.

## 8. Credenciales

`PULSO_API_URL` y `PULSO_API_KEY` están en `.env` (ignorado por git, permisos 600). La key
apareció en una conversación con el asistente: **regenerarla desde el portal** cuando se quiera
cerrar esa exposición. En GitHub Actions va como secret `PULSO_API_KEY`; nunca en el código.

## 9. Estructura del repositorio

```text
docs/                 api.md, student-project.md, data-model.md, proyecto-mlops.md
examples/             01_download, 02_naive_baseline, 03_eda.ipynb
modelo/               paquete pulso-forecast (src/, tests/, pyproject.toml, README.md)
src/pulso_transmi/    cliente Python de la API (SDK del profesor)
templates/pipeline.yml  plantilla de GitHub Actions (manual)
data/, artifacts/, .env  ignorados por git
```

## 10. Decisiones y supuestos

1. Horizontes 15 min, 1 h, 4 h, 24 h: **supuesto** hasta que se publiquen.
2. Un modelo por horizonte, sin encadenar: evita acumular error.
3. Objetivo escalado por el nivel de 7 días: iguala estaciones y amortigua la deriva.
4. Sin feature de festivos ni de eventos futuros: no hay señal / no hay garantía de que se conozcan.
5. Sin FK física entre `observations` y `context` en la base: se unen por `observed_at`.
6. Entrenamiento final con todos los datos disponibles tras validar.

## 11. Pendientes y riesgos

- **Repo público:** `origin` es el repo del profesor, y `modelo/` es una solución completa. No hacer
  push ahí; crear un repo de equipo. Hay 3 commits locales sin subir.
- Borrar la edge function `load-pulso-data`.
- Construir el pipeline: ingesta incremental con cursor, monitoreo de error y deriva, regla de
  reentrenamiento, envío con versión y commit, y registro de cada corrida en Supabase.
- Confirmar los horizontes de la ronda competitiva y ajustar `HORIZONS`.
- Reentrenar cuando lleguen datos nuevos: el modelo se entrenó con 45 días y hay deriva al alza.
- Si un dashboard lee los datos del reto desde el navegador, agregar policies de solo lectura.

"""Interfaz de línea de comandos: evaluate | train | predict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from pulso_forecast import __version__, model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pulso-forecast", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("evaluate", "validación temporal y métricas en el bloque final"),
                        ("train", "evalúa y reentrena con todos los datos; guarda el modelo"),
                        ("predict", "predice desde el último dato con un modelo guardado")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--data-dir", default=None, help="carpeta con observations/context/stations.csv")
        p.add_argument("--out", default="artifacts", help="carpeta de salida")
        if name == "predict":
            p.add_argument("--model", default="model.joblib", help="archivo de modelo entrenado")
    args = parser.parse_args(argv)
    out = Path(args.out)

    if args.cmd == "predict":
        bundle = joblib.load(args.model)
        forecast = model.forecast_next(bundle["models"], data_dir=args.data_dir)
        out.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(out / "next_forecast.csv", index=False)
        print(forecast.to_string(index=False))
        return

    report = model.run(out, data_dir=args.data_dir)
    print(json.dumps({k: {m: round(v[m], 2) for m in ("model", "naive_day", "naive_week")}
                      for k, v in report["horizons"].items()}, indent=2))
    if args.cmd == "train":
        params = {int(k[1:]): v["best_params"] for k, v in report["horizons"].items()}
        model.train_production(out, params, data_dir=args.data_dir)
        print(f"modelo guardado en {out / 'model.joblib'}")


if __name__ == "__main__":
    main()

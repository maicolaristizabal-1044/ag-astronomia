"""
Punto de entrada del pipeline.

Ejecuta los tres subproblemas de optimización con Algoritmos Genéticos y
guarda métricas, gráficas y logs de convergencia en la carpeta de salida.

Uso:
    python main.py
    python main.py --data data/sdss_sample.csv --outputs outputs --seed 42
"""

import argparse
import json
import os
import sys
import time

from src.feature_selection import FeatureSelectionAG
from src.hyperparameter_tuning import HyperparameterAG
from src.clustering import ClusteringAG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de optimización con Algoritmos Genéticos "
                    "sobre datos astronómicos SDSS.")
    parser.add_argument("--data", default="data/sdss_sample.csv",
                        help="Ruta al dataset CSV.")
    parser.add_argument("--outputs", default="outputs",
                        help="Carpeta donde se guardan métricas y gráficas.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad.")
    return parser.parse_args()


def banner(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: no se encontró el dataset en '{args.data}'", file=sys.stderr)
        return 1

    os.makedirs(args.outputs, exist_ok=True)
    inicio = time.time()

    # ---------------------------------------------------------------
    banner("1. SELECCIÓN DE CARACTERÍSTICAS CON AG")
    fs = FeatureSelectionAG(args.data, random_state=args.seed)
    fs_res = fs.run(output_dir=args.outputs)
    print(f"Mejor cromosoma      : {fs_res['best_chromosome']}")
    print(f"Variables activas    : {', '.join(fs_res['selected_features'])}"
          f"  ({fs_res['n_features_selected']} de 6)")
    print(f"Accuracy del AG      : {fs_res['best_accuracy']:.4f}")
    print(f"Accuracy con las 6   : {fs_res['baseline_all_features_accuracy']:.4f}")

    # ---------------------------------------------------------------
    banner("2. OPTIMIZACIÓN DE HIPERPARÁMETROS CON AG")
    ht = HyperparameterAG(args.data, random_state=args.seed)
    ht_res = ht.run(output_dir=args.outputs)
    print(f"Mejor k encontrado   : {ht_res['best_k']}")
    print(f"MSE                  : {ht_res['mse']:.6f}")
    print(f"R²                   : {ht_res['r2_score']:.4f}")
    print(f"Línea base (k=5) MSE : {ht_res['baseline_k5']['mse']:.6f}"
          f"  |  R²: {ht_res['baseline_k5']['r2_score']:.4f}")

    # ---------------------------------------------------------------
    banner("3. OPTIMIZACIÓN DE AGRUPAMIENTO CON AG")
    cl = ClusteringAG(args.data, random_state=args.seed)
    cl_res = cl.run(output_dir=args.outputs)
    print(f"SSE del AG           : {cl_res['ag']['sse']:.2f}")
    print(f"SSE de KMeans        : {cl_res['kmeans']['sse']:.2f}")
    print(f"Diferencia           : {cl_res['sse_gap_pct']:+.2f}% frente a KMeans")
    print(f"ARI AG vs. clases    : {cl_res['ag']['adjusted_rand_index']:.4f}")
    print(f"ARI KMeans vs. clases: {cl_res['kmeans']['adjusted_rand_index']:.4f}")

    # ---------------------------------------------------------------
    metrics = {
        "dataset": args.data,
        "seed": args.seed,
        "execution_time_seconds": round(time.time() - inicio, 2),
        "feature_selection": fs_res,
        "hyperparameter_tuning": ht_res,
        "clustering": cl_res,
    }

    metrics_path = os.path.join(args.outputs, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=4, ensure_ascii=False)

    banner("EJECUCIÓN COMPLETADA")
    print(f"Tiempo total: {metrics['execution_time_seconds']} s")
    print(f"Resultados guardados en: {os.path.abspath(args.outputs)}/")
    for archivo in sorted(os.listdir(args.outputs)):
        print(f"  - {archivo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

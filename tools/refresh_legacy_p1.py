# -*- coding: utf-8 -*-
"""
Ré-exécute UNIQUEMENT la partie 1 du banc d'essai (métriques réellement affichées
par l'application) contre la version ORIGINALE du code (commit c5dde9b) et fusionne
le résultat dans reports/benchmark_results.json, sans rejouer les parties 2 à 7.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import benchmark as B  # noqa: E402
from make_synthetic_data import build_dataset  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "reports")
JSON_PATH = os.path.join(OUT, "benchmark_results.json")


def main():
    print("Version exécutée :", B.APP_PALU)
    assert "_baseline" in B.APP_PALU, "doit pointer sur la version originale"
    ds = build_dataset()
    p1 = B.partie1_legacy_metrics(ds)
    p1.to_csv(os.path.join(OUT, "benchmark_p1_legacy.csv"), index=False)
    print(p1.to_string(index=False))

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    results["p1_legacy"] = json.loads(p1.to_json(orient="records"))
    results["p1_source"] = os.path.relpath(B.APP_PALU, os.path.dirname(HERE))
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print("\n→ p1_legacy mis à jour dans", JSON_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())

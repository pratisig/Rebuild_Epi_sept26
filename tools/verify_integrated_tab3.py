"""
Vérifie le code RÉEL du bloc `with tab3:` de app_paludisme.py APRÈS intégration
du noyau `epimodel` (XGBoost, covariables statiques, validation temporelle).

Exécute le bloc via tools/legacy_harness.py avec un stub Streamlit, puis
contrôle les contrats attendus par les autres onglets de l'application :
carte des prédictions, validation rétrospective, exports.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from legacy_harness import run_tab3  # noqa: E402
from make_synthetic_data import build_dataset  # noqa: E402
from st_stub import StreamlitStub  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "app_paludisme.py")

FAILURES = []


def check(label, cond, detail=""):
    status = "✅" if cond else "❌"
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def main():
    ds = build_dataset()
    df_cases = ds.df_cases.copy()
    df_cases["health_area"] = df_cases["health_area"].astype(str).str.strip().str.lower()
    gdf_health = ds.gdf.merge(
        ds.df_population[["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]],
        on="health_area", how="left")

    stub = StreamlitStub(
        selectbox_values={"Algorithme": "XGBoost",
                          "Objectif de perte": "Poisson (comptages — recommandé)"},
        slider_values={"Semaines à prévoir": 4, "Seuil alerte": 75,
                       "Nombre de folds": 5, "Embargo": 4, "Clusters": 5, "Voisins": 5},
    )
    ss = stub.session_state
    for k in ["gdf_health", "df_cases", "temp_raster", "flood_raster", "rivers_gdf",
              "precipitation_raster", "humidity_raster", "elevation_raster",
              "model_results", "df_climate_aggregated", "df_gee_static",
              "df_env_static", "dfpopulation"]:
        ss[k] = None
    ss["gdf_health"] = gdf_health
    ss["df_cases"] = df_cases
    ss["dfpopulation"] = ds.df_population
    ss["df_env_static"] = ds.df_static      # covariables statiques (altitude…)

    print("=== EXÉCUTION DU BLOC `with tab3:` INTÉGRÉ (code réel) ===")
    res = run_tab3(APP, stub, {
        "df_cases": df_cases, "gdf_health": gdf_health,
        "years_selected": sorted(df_cases["year"].unique().tolist()),
        "iso3pays": "ner", "gee_ok": False}, tab_name="tab3")

    print("\n1) Exécution")
    check("aucune erreur", res["error"] is None,
          (res["error"][:1200] if res["error"] else ""))
    if res["error"]:
        for f in FAILURES:
            print("ECHEC:", f)
        return 1

    mr = ss["model_results"]

    print("\n2) Contrat `st.session_state.model_results`")
    for key in ["df_model", "df_future", "metrics", "pca_info", "feature_cols"]:
        check(f"clé '{key}' présente", key in mr)

    df_future = mr["df_future"]
    metrics = mr["metrics"]

    print("\n3) Contrat de sortie des prédictions (carte / exports)")
    for col in ["health_area", "week_num", "predicted_cases"]:
        check(f"colonne '{col}'", col in df_future.columns)
    check("prédictions non négatives", bool((df_future["predicted_cases"] >= 0).all()))
    check("période présente", "period" in df_future.columns,
          str(df_future["period"].iloc[0]) if "period" in df_future.columns else "")
    check("intervalles de prédiction", {"q10", "q90"}.issubset(df_future.columns))
    check("4 semaines x 72 aires", len(df_future) == 4 * 72, f"n={len(df_future)}")

    print("\n4) Métriques retournées")
    for key in ["mae", "rmse", "r2", "cv_r2_mean", "cv_r2_std", "cv_mae_mean",
                "n_train", "n_features", "algorithme", "objectif"]:
        check(f"metrics['{key}']", key in metrics)
    print(f"     algorithme={metrics['algorithme']} objectif={metrics['objectif']}")
    print(f"     R² in-sample={metrics['r2']:.3f} | R² CV temporel={metrics['cv_r2_mean']:.3f} "
          f"± {metrics['cv_r2_std']:.3f} | MAE CV={metrics['cv_mae_mean']:.1f}")
    print(f"     variables={metrics['n_features']} | lignes entraînement={metrics['n_train']}")
    check("R² CV < R² in-sample (pas d'optimisme masqué)",
          metrics["cv_r2_mean"] < metrics["r2"],
          f"{metrics['cv_r2_mean']:.3f} < {metrics['r2']:.3f}")

    print("\n5) Cohérence prédictions / historique")
    df_model = mr["df_model"]
    last8 = (df_model.sort_values(["health_area", "week_index"]).groupby("health_area")
             .tail(8).groupby("health_area")["cases"].mean())
    pred = df_future.groupby("health_area")["predicted_cases"].mean()
    ratio = (pred / last8.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    med = float(ratio.median())
    print(f"     ratio prédit/observé : médiane={med:.3f} "
          f"p10={ratio.quantile(.1):.3f} p90={ratio.quantile(.9):.3f}")
    check("pas de dérive systématique (0.5 < médiane < 2)", 0.5 < med < 2.0,
          f"médiane={med:.3f}")
    check("aucune aire à prédiction nulle",
          int((df_future.groupby("health_area")["predicted_cases"].sum() == 0).sum()) == 0)

    print("\n6) Variables utilisées (covariables statiques incluses)")
    cols = mr["feature_cols"]
    print(f"     {len(cols)} variables : {cols}")
    check("altitude présente", "Altitude_Moy" in cols)
    check("démographie présente", "Pop_Totale" in cols)
    check("saisonnalité présente", "sin_week_1" in cols)
    check("lag spatial présent", "spatial_lag_1" in cols)
    check("ordre déterministe (pas de set())", cols == list(dict.fromkeys(cols)))

    print("\n7) Importance des variables")
    imp = mr.get("importance")
    check("importance calculée", imp is not None and len(imp) > 0)
    if imp is not None and len(imp):
        print(imp.head(10)[["variable", "importance_pct"]].to_string(index=False))

    print("\n8) Folds de validation")
    folds = mr.get("cv_folds")
    check("folds présents", folds is not None and len(folds) == 5)
    if folds is not None and len(folds):
        print(folds[["fold", "train_weeks", "test_weeks", "mae", "r2"]]
              .to_string(index=False))
        # aucune semaine de test dans l'entraînement
        ok = all(float(a.split("-")[1]) < float(b.split("-")[0])
                 for a, b in zip(folds["train_weeks"], folds["test_weeks"]))
        check("chaque fold n'entraîne que sur le passé", ok)

    print("\n9) Erreurs/warnings Streamlit émis pendant l'exécution")
    check("aucun st.error", len(stub.errors) == 0, str(stub.errors[:3]))

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"❌ {len(FAILURES)} contrôle(s) en échec : {FAILURES}")
        return 1
    print("✅ Tous les contrôles passent")
    return 0


if __name__ == "__main__":
    sys.exit(main())

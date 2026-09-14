"""
Vérifie le code RÉEL du bloc `with tab3:` de app_rougeole.py APRÈS intégration
du noyau `epimodel` (XGBoost, validation temporelle, covariables statiques).

Les données d'entrée sont produites par les générateurs de démonstration de
l'application elle-même (`generate_dummy_linelists`, `generate_dummy_vaccination`),
extraits par AST — aucune réécriture de logique.
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from legacy_harness import load_module_context, run_tab3  # noqa: E402
from st_stub import StreamlitStub  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "app_rougeole.py")

FAILURES = []


def check(label, cond, detail=""):
    print(f"  {'✅' if cond else '❌'} {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def main():
    import geopandas as gpd
    gdf = gpd.read_file(os.path.join(REPO, "data", "ao_hlthArea.zip"))
    gdf = gdf[gdf["iso3"].astype(str).str.strip().str.lower() == "ner"].copy()
    gdf["health_area"] = gdf["health_are"].astype(str).str.strip()
    gdf = gdf[gdf.geometry.is_valid]
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf.reset_index(drop=True)

    stub = StreamlitStub(
        selectbox_values={"algorithme": "XGBoost",
                          "Objectif de perte": "Poisson (comptages — recommandé)"},
        radio_values={"Importance": "📊 Automatique (ML)"},
        buttons_true=["Lancer la Modélisation"],
    )
    ss = stub.session_state
    ss["prediction_rougeole_lancee"] = True
    ss["sa_gdf_cache"] = None
    ss["pays_precedent"] = None

    # fonctions réelles de l'application, extraites par AST
    ns, _, _ = load_module_context(APP, tab_name="tab3", stub=stub)
    df = ns["generate_dummy_linelists"](gdf, 500)
    vaccination_df = ns["generate_dummy_vaccination"](gdf)
    semaine_vers_date = ns["semaine_vers_date"]

    # enrichissement statique (même structure que sa_gdf_enrichi)
    rng = np.random.default_rng(7)
    proj = gdf.to_crs("ESRI:54009")
    sa_gdf_enrichi = gdf.copy()
    sa_gdf_enrichi["Superficie_km2"] = proj.geometry.area / 1e6
    sa_gdf_enrichi["Pop_Totale"] = (sa_gdf_enrichi["Superficie_km2"]
                                    * rng.lognormal(np.log(20), 0.6, len(gdf))).round()
    sa_gdf_enrichi["Pop_Enfants"] = (sa_gdf_enrichi["Pop_Totale"] * 0.48).round()
    sa_gdf_enrichi["Densite_Pop"] = (sa_gdf_enrichi["Pop_Totale"]
                                     / sa_gdf_enrichi["Superficie_km2"]).round(2)
    sa_gdf_enrichi["Densite_Enfants"] = (sa_gdf_enrichi["Pop_Enfants"]
                                         / sa_gdf_enrichi["Superficie_km2"]).round(2)
    sa_gdf_enrichi["Urbanisation"] = rng.choice(["Rural", "Semi-urbain", "Urbain"],
                                                size=len(gdf), p=[0.6, 0.3, 0.1])
    sa_gdf_enrichi["Temperature_Moy"] = rng.normal(30, 2.5, len(gdf)).round(1)
    sa_gdf_enrichi["Humidite_Moy"] = rng.uniform(15, 65, len(gdf)).round(1)
    sa_gdf_enrichi["Saison_Seche_Humidite"] = (sa_gdf_enrichi["Humidite_Moy"] * 0.7).round(1)
    sa_gdf_enrichi["Altitude_Moy"] = rng.uniform(200, 1200, len(gdf)).round(1)
    v = vaccination_df.copy()
    v["health_area"] = v["health_area"].astype(str).str.strip()
    sa_gdf_enrichi = sa_gdf_enrichi.drop(columns=["Taux_Vaccination"], errors="ignore")
    sa_gdf_enrichi = sa_gdf_enrichi.merge(v[["health_area", "Taux_Vaccination"]],
                                          on="health_area", how="left")

    weekly_cases = (df.groupby(["Annee", "Semaine_Epi"])
                      .agg(Cas=("ID_Cas", "count")).reset_index())
    weekly_cases["sort_key"] = weekly_cases["Annee"] * 100 + weekly_cases["Semaine_Epi"]
    weekly_cases["Semaine_Label"] = (weekly_cases["Annee"].astype(str) + "-S" +
                                     weekly_cases["Semaine_Epi"].astype(str).str.zfill(2))
    weekly_cases = weekly_cases.sort_values("sort_key").reset_index(drop=True)

    idx_last = weekly_cases["sort_key"].idxmax()
    derniere_semaine_epi = int(weekly_cases.loc[idx_last, "Semaine_Epi"])
    derniere_annee = int(weekly_cases.loc[idx_last, "Annee"])

    print("=== EXÉCUTION DU BLOC `with tab3:` ROUGEOLE INTÉGRÉ (code réel) ===")
    res = run_tab3(APP, stub, {
        "df": df,
        "sa_gdf": gdf,
        "sa_gdf_enrichi": sa_gdf_enrichi,
        "vaccination_df": vaccination_df,
        "weekly_cases": weekly_cases,
        "n_weeks_pred": 12,
        "pred_mois": 3,
        "seuil_hausse": 30.0,
        "seuil_baisse": 30.0,
        "modele_choisi": "XGBoost",
        "objectif_rougeole": "Poisson (comptages — recommandé)",
        "mode_importance": "📊 Automatique (ML)",
        "poids_normalises": {},
        "derniere_semaine_epi": derniere_semaine_epi,
        "derniere_annee": derniere_annee,
        "seuil_alerte_epidemique": 5,
        "gee_ok": False,
        "iso3_pays": "ner",
    }, tab_name="tab3")

    print("\n1) Exécution")
    check("aucune erreur", res["error"] is None,
          (res["error"][:2000] if res["error"] else ""))
    if res["error"]:
        print("ECHECS:", FAILURES)
        return 1

    mr = ss.get("rougeole_model")
    check("résultats stockés en session", mr is not None)
    if mr is None:
        return 1

    future_df = mr["future_df"]
    metrics = mr["metrics"]

    print("\n2) Contrat de sortie (courbe, synthèse des risques, heatmap, exports)")
    for col in ["Aire_Sante", "SemaineLabel", "SemaineEpi", "Annee", "sort_key", "CasPredits"]:
        check(f"colonne '{col}'", col in future_df.columns)
    check("prédictions non négatives", bool((future_df["CasPredits"] >= 0).all()))
    check("12 semaines prédites", future_df["SemaineLabel"].nunique() == 12,
          f"{future_df['SemaineLabel'].nunique()} semaines")
    check("intervalles de prédiction", {"Q10", "Q90"}.issubset(future_df.columns))

    print("\n3) Métriques")
    for k in ["mae", "rmse", "r2", "cv_r2_mean", "cv_r2_std", "cv_mae_mean",
              "n_train", "n_features", "algorithme", "objectif"]:
        check(f"metrics['{k}']", k in metrics)
    print(f"     algo={metrics['algorithme']} objectif={metrics['objectif']}")
    print(f"     R² in-sample={metrics['r2']:.3f} | R² CV temporel={metrics['cv_r2_mean']:.3f} "
          f"± {metrics['cv_r2_std']:.3f} | MAE CV={metrics['cv_mae_mean']:.2f}")
    check("R² CV < R² in-sample", metrics["cv_r2_mean"] < metrics["r2"])

    print("\n4) Variables (statiques incluses)")
    cols = mr["feature_cols"]
    print(f"     {len(cols)} variables : {cols}")
    check("couverture vaccinale présente", "Taux_Vaccination" in cols)
    check("population enfants présente", "Pop_Enfants" in cols)
    check("urbanisation présente", "UrbanEncoded" in cols)
    check("saisonnalité présente", "sin_week_1" in cols)
    check("retards présents", "cases_lag_1" in cols)

    print("\n5) Importance des variables")
    imp = mr.get("importance")
    check("importance calculée", imp is not None and len(imp) > 0)
    if imp is not None and len(imp):
        print(imp.head(10)[["variable", "importance_pct"]].to_string(index=False))

    print("\n6) Erreurs Streamlit émises")
    # L'application utilise st.error pour les ALERTES épidémiologiques (usage
    # légitime) : on n'écarte que les erreurs techniques.
    _alertes = [e for e in stub.errors if "FORTE HAUSSE" in e or "FORTE BAISSE" in e
                or "AREAS IN ALERT" in e or "AREAS UNDER WATCH" in e]
    _techniques = [e for e in stub.errors if e not in _alertes]
    print(f"     alertes épidémiologiques émises : {len(_alertes)} (usage légitime)")
    check("aucune erreur technique", len(_techniques) == 0, str(_techniques[:3]))

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"❌ {len(FAILURES)} contrôle(s) en échec : {FAILURES}")
        return 1
    print("✅ Tous les contrôles passent")
    return 0


if __name__ == "__main__":
    sys.exit(main())

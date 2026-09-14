"""
Pont entre les interfaces Streamlit existantes et le noyau `epimodel`.

Rôle : conserver **exactement** les contrats de données de l'application
(noms de colonnes, clés de ``st.session_state``, formats de sortie) tout en
remplaçant le cœur de calcul par le pipeline corrigé.

Ce module est volontairement sans logique métier propre : il prépare les
entrées, appelle `epimodel`, et remet les sorties au format attendu par les
onglets existants (carte des prédictions, validation, exports, rapports).
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import epimodel as em
from epimodel.features import build_neighbour_weights
from epimodel.validation import metrics_summary, temporal_cv_splits

# ----------------------------------------------------------------------
# Colonnes historiques de l'application (contrat à ne pas casser)
# ----------------------------------------------------------------------
PALU_STATIC_POP = ["Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]
PALU_ENV_RASTER = ["flood_mean", "elevation_mean", "dist_river", "flood_risk",
                   "climate_index"]
PALU_CLIMATE = ["temp_api", "precip_api", "humidity_api"]

ROUGEOLE_STATIC_EXTRA = [
    "Pop_Enfants", "Densite_Pop", "Densite_Enfants", "UrbanEncoded",
    "Temperature_Moy", "Humidite_Moy", "Taux_Vaccination",
    "Gap_Immunite_Collective", "Susceptibles_Estimes", "Zone_Sous_Seuil",
    "Part_NonVaccines_Cas",
]

# Covariables statiques « prêtes à l'emploi » si extraites depuis GEE
GEE_STATIC_COLS = ["Altitude_Moy", "Altitude_Max", "Altitude_Min",
                   "Altitude_EcartType", "Pente_Moy", "NDVI_Moy",
                   "NDVI_Saisonnalite", "LST_Jour_Moy", "Eau_Pct",
                   "Eau_Saison_Pct", "Dist_Eau_km", "Arbre_Pct",
                   "Culture_Pct", "Urbain_Pct", "Superficie_km2"]


# ======================================================================
# Assemblage des covariables statiques
# ======================================================================
def build_static_table(dfpopulation: Optional[pd.DataFrame] = None,
                       df_gee_static: Optional[pd.DataFrame] = None,
                       df_env: Optional[pd.DataFrame] = None,
                       df_extra: Optional[pd.DataFrame] = None,
                       area_col: str = "health_area") -> pd.DataFrame:
    """
    Fusionne toutes les sources statiques en une table unique par aire.

    Ordre de priorité : ``dfpopulation`` → ``df_gee_static`` → ``df_env`` →
    ``df_extra`` (les valeurs déjà renseignées ne sont pas écrasées).
    """
    from epimodel.static_covariates import merge_static_tables
    return merge_static_tables(dfpopulation, df_gee_static, df_env, df_extra,
                               area_col=area_col)


def make_design_builder(gdf,
                        static_df: Optional[pd.DataFrame],
                        climate_df: Optional[pd.DataFrame] = None,
                        use_spatial: bool = True,
                        neighbour_k: int = 5,
                        n_clusters: int = 5,
                        groups: Optional[Sequence[str]] = None,
                        lags: Optional[Sequence[int]] = None,
                        extra_static: Optional[Sequence[str]] = None) -> Callable:
    """
    Fabrique la fonction unique d'ingénierie de variables, utilisée à
    l'entraînement comme à la prévision.
    """
    weights = build_neighbour_weights(gdf, k=neighbour_k) \
        if (use_spatial and gdf is not None and len(gdf) > 1) else None
    kwargs: Dict[str, Any] = dict(
        static_df=static_df,
        gdf=gdf,
        climate_df=climate_df,
        neighbour_weights=weights,
        n_clusters=n_clusters if use_spatial else 0,
        groups=list(groups) if groups is not None else list(em.features.DEFAULT_GROUP_ORDER),
        extra_static=extra_static,
    )
    if lags is not None:
        kwargs["lags"] = tuple(lags)
    return partial(em.build_design_matrix, **kwargs)


# ======================================================================
# Validation temporelle honnête (h = 1)
# ======================================================================
def temporal_validation(panel: pd.DataFrame,
                        design_builder: Callable,
                        model_factory: Callable[[], Any],
                        n_splits: int = 5,
                        embargo: int = 4,
                        feature_cols: Optional[Sequence[str]] = None
                        ) -> Dict[str, Any]:
    """
    Validation croisée **bloquée par semaine** avec embargo, à horizon 1.

    Contrairement au pipeline historique (qui découpait par aire de santé),
    chaque fold entraîne uniquement sur des semaines strictement antérieures
    aux semaines testées.
    """
    df, cols = design_builder(panel)
    if feature_cols is not None:
        cols = [c for c in feature_cols if c in df.columns]

    d = df[df["cases_lag_1"].notna()].reset_index(drop=True) if "cases_lag_1" in df.columns else df
    if len(d) < 50:
        raise ValueError("Historique insuffisant pour la validation temporelle.")

    X = (d[cols].apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan))
    y = pd.to_numeric(d["cases"], errors="coerce").to_numpy(dtype=float)

    from epimodel.forecast import _make_imputer
    splits = temporal_cv_splits(d["week_index"].to_numpy(), n_splits=n_splits,
                                embargo=embargo)

    folds, y_all, p_all = [], [], []
    for i, (tr, te) in enumerate(splits, start=1):
        imp = _make_imputer().fit(X.iloc[tr])
        Xtr = pd.DataFrame(imp.transform(X.iloc[tr]), columns=cols)
        Xte = pd.DataFrame(imp.transform(X.iloc[te]), columns=cols)
        mdl = model_factory().fit(Xtr, y[tr])
        pred = np.clip(np.nan_to_num(mdl.predict(Xte), nan=0.0), 0, None)
        m = metrics_summary(y[te], pred)
        m["fold"] = i
        m["train_weeks"] = f"{d.iloc[tr]['week_index'].min():.0f}-" \
                           f"{d.iloc[tr]['week_index'].max():.0f}"
        m["test_weeks"] = f"{d.iloc[te]['week_index'].min():.0f}-" \
                          f"{d.iloc[te]['week_index'].max():.0f}"
        folds.append(m)
        y_all.extend(y[te].tolist())
        p_all.extend(pred.tolist())

    fdf = pd.DataFrame(folds)
    pooled = metrics_summary(np.asarray(y_all), np.asarray(p_all))
    return {
        "folds": fdf,
        "cv_r2_mean": float(np.nanmean(fdf["r2"].to_numpy(dtype=float))) if len(fdf) else float("nan"),
        "cv_r2_std": float(np.nanstd(fdf["r2"].to_numpy(dtype=float))) if len(fdf) else float("nan"),
        "cv_mae_mean": float(fdf["mae"].mean()) if len(fdf) else float("nan"),
        "pooled": pooled,
        "n_splits": len(folds),
        "leakage_check": "aucune semaine de test présente à l'entraînement "
                         "(découpage bloqué par semaine + embargo)",
    }


# ======================================================================
# Modélisation PALUDISME
# ======================================================================
def prepare_palu(df_cases: pd.DataFrame,
                 gdf_health,
                 dfpopulation: Optional[pd.DataFrame] = None,
                 df_gee_static: Optional[pd.DataFrame] = None,
                 df_env: Optional[pd.DataFrame] = None,
                 df_climate: Optional[pd.DataFrame] = None,
                 years: Optional[Sequence[int]] = None,
                 use_spatial: bool = True,
                 neighbour_k: int = 5,
                 n_clusters: int = 5,
                 use_climate: bool = True,
                 lags: Optional[Sequence[int]] = None):
    """Prépare panneau + constructeur de variables pour le module paludisme."""
    panel = em.build_panel(df_cases, years=years)
    static = build_static_table(dfpopulation, df_gee_static, df_env)
    climate = df_climate if (use_climate and df_climate is not None
                             and not df_climate.empty) else None
    db = make_design_builder(gdf_health, static, climate_df=climate,
                             use_spatial=use_spatial, neighbour_k=neighbour_k,
                             n_clusters=n_clusters, lags=lags)
    info = {
        "n_areas": int(panel["health_area"].nunique()),
        "n_weeks": int(panel["week_index"].nunique()),
        "n_rows": int(len(panel)),
        "static_cols": [c for c in static.columns if c != "health_area"]
        if static is not None else [],
        "climate": climate is not None,
    }
    return panel, db, info


def run_modelling_palu(panel: pd.DataFrame,
                       design_builder: Callable,
                       algo: str = "XGBoost",
                       n_future_weeks: int = 4,
                       objective: str = "squared_error",
                       n_splits: int = 5,
                       embargo: int = 4,
                       quantile_levels: Sequence[float] = (0.1, 0.9),
                       with_backtest: bool = False,
                       backtest_origins: int = 6,
                       progress: Optional[Callable[[str, int], None]] = None
                       ) -> Dict[str, Any]:
    """
    Exécute la modélisation et renvoie un dictionnaire **compatible** avec
    ``st.session_state.model_results`` de l'application :

    ``{'df_model', 'df_future', 'metrics', 'pca_info', 'feature_cols', ...}``

    ``df_future`` conserve les colonnes ``health_area``, ``week_num``,
    ``predicted_cases`` attendues par la carte des prédictions et les exports,
    et ajoute ``period``, ``horizon``, ``q10`` et ``q90``.
    """
    def _p(msg, pct):
        if progress is not None:
            progress(msg, pct)

    _p("Entraînement…", 20)
    log_target = False
    factory = lambda: em.make_model(algo, objective=objective)  # noqa: E731
    if objective == "poisson" and em.models.resolve_objective(algo, objective) != "poisson":
        # le modèle choisi ne gère pas Poisson : on passe par log1p
        log_target = True
        factory = lambda: em.make_model(algo, objective="squared_error")  # noqa: E731

    fitted = em.fit_pipeline(panel, design_builder=design_builder,
                             model_factory=factory, objective=objective,
                             log_target=log_target)

    _p("Validation temporelle…", 45)
    try:
        val = temporal_validation(panel, design_builder, factory,
                                  n_splits=n_splits, embargo=embargo,
                                  feature_cols=fitted["feature_cols"])
    except Exception:
        val = {"cv_r2_mean": float("nan"), "cv_r2_std": float("nan"),
               "cv_mae_mean": float("nan"), "folds": pd.DataFrame(),
               "pooled": {}, "n_splits": 0}

    _p("Prévision récursive…", 70)
    fc = em.recursive_forecast(fitted, panel=panel.copy(),
                               horizons=int(n_future_weeks),
                               design_builder=design_builder)
    if len(fc) and quantile_levels:
        fc = em.predict_quantiles(fitted, fc, levels=tuple(quantile_levels))

    _p("Métriques…", 88)
    df_model, feature_cols = design_builder(panel)
    insample = fitted["train_metrics"]

    metrics = {
        # clés historiques (lues par l'interface existante)
        "mae": float(insample.get("mae", float("nan"))),
        "rmse": float(insample.get("rmse", float("nan"))),
        "r2": float(insample.get("r2", float("nan"))),
        "cv_r2_mean": float(val.get("cv_r2_mean", float("nan"))),
        "cv_r2_std": float(val.get("cv_r2_std", float("nan"))),
        # clés ajoutées (transparence méthodologique)
        "cv_mae_mean": float(val.get("cv_mae_mean", float("nan"))),
        "cv_r2_pool": float((val.get("pooled") or {}).get("r2", float("nan"))),
        "cv_mae_pool": float((val.get("pooled") or {}).get("mae", float("nan"))),
        "n_train": int(fitted["n_train"]),
        "n_features": len(feature_cols),
        "algorithme": algo,
        "objectif": ("log1p+squared_error" if log_target else objective),
        "horizon_semaines": int(n_future_weeks),
        "protocole": "validation temporelle bloquée par semaine + embargo "
                     f"{embargo} semaines",
    }

    result: Dict[str, Any] = {
        "df_model": df_model,
        "df_future": fc,
        "metrics": metrics,
        "pca_info": None,          # clé conservée pour compatibilité
        "feature_cols": feature_cols,
        "cv_folds": val.get("folds"),
        "importance": em.forecast.feature_importance(fitted, top_n=30),
        "fitted": {k: v for k, v in fitted.items() if k != "model"},
        "_model": fitted["model"],
    }

    if with_backtest:
        _p("Backtest multi-horizons…", 94)
        from epimodel.validation import backtest
        horizons = sorted({h for h in (1, 2, 4, 8) if h <= int(n_future_weeks)} or {1})
        try:
            result["backtest"] = backtest(
                panel, design_builder, factory, horizons=horizons,
                n_origins=backtest_origins,
                min_train_weeks=min(52, max(20, int(panel["week_index"].nunique() // 2))))
        except Exception as e:  # noqa: BLE001
            result["backtest_error"] = str(e)

    _p("Terminé", 100)
    return result


# ======================================================================
# Modélisation ROUGEOLE
# ======================================================================
def prepare_rougeole(df_cases_weekly: pd.DataFrame,
                     sa_gdf_enrichi,
                     df_vaccination: Optional[pd.DataFrame] = None,
                     df_linelist: Optional[pd.DataFrame] = None,
                     df_gee_static: Optional[pd.DataFrame] = None,
                     use_spatial: bool = True,
                     neighbour_k: int = 5,
                     n_clusters: int = 5,
                     area_col: str = "Aire_Sante"):
    """
    Prépare panneau + constructeur de variables pour le module rougeole.

    ``df_cases_weekly`` doit contenir ``Aire_Sante``, ``Annee``,
    ``Semaine_Epi`` et un nombre de cas (``CasObserves`` ou ``Cas_Total``).
    """
    df = df_cases_weekly.copy()
    case_col = None
    for c in ["CasObserves", "Cas_Total", "cas", "cases", "Cas"]:
        if c in df.columns:
            case_col = c
            break
    if case_col is None:
        raise ValueError("Colonne de cas introuvable dans les données hebdomadaires.")

    cases = pd.DataFrame({
        "health_area": df[area_col].astype(str).str.strip().str.lower(),
        "year": pd.to_numeric(df["Annee"], errors="coerce"),
        "week_": pd.to_numeric(df["Semaine_Epi"], errors="coerce"),
        "cases": pd.to_numeric(df[case_col], errors="coerce").fillna(0),
    }).dropna(subset=["year", "week_"])
    cases["year"] = cases["year"].astype(int)
    cases["week_"] = cases["week_"].astype(int)

    panel = em.build_panel(cases)

    # ── table statique rougeole ─────────────────────────────────────────
    static_frames = []
    if sa_gdf_enrichi is not None and len(sa_gdf_enrichi) > 0:
        g = sa_gdf_enrichi.copy()
        g["health_area"] = g["health_area"].astype(str).str.strip().str.lower()
        cols = [c for c in ["health_area", "Pop_Totale", "Pop_Enfants", "Densite_Pop",
                            "Densite_Enfants", "Urbanisation", "Temperature_Moy",
                            "Humidite_Moy", "Taux_Vaccination", "Superficie_km2",
                            "Altitude_Moy", "Pente_Moy", "NDVI_Moy", "Urbain_Pct"]
                if c in g.columns]
        static_frames.append(g[cols].drop_duplicates(subset=["health_area"]))

    if df_vaccination is not None and "Taux_Vaccination" in df_vaccination.columns:
        v = df_vaccination.copy()
        if "health_area" not in v.columns and area_col in v.columns:
            v["health_area"] = v[area_col]
        v["health_area"] = v["health_area"].astype(str).str.strip().str.lower()
        v["Taux_Vaccination"] = pd.to_numeric(v["Taux_Vaccination"], errors="coerce")
        static_frames.append(v[["health_area", "Taux_Vaccination"]]
                             .drop_duplicates(subset=["health_area"]))

    if df_gee_static is not None and not df_gee_static.empty:
        static_frames.append(df_gee_static)

    static = build_static_table(*static_frames)

    # encodage déterministe de l'urbanisation (le LabelEncoder de l'ancien code
    # dépendait de l'ordre d'apparition et produisait un code 0 par défaut
    # en cas de valeur inconnue à la prévision)
    urb_map = {"rural": 0.0, "semi-urbain": 1.0, "semi urbain": 1.0, "urbain": 2.0}
    if "Urbanisation" in static.columns:
        static["UrbanEncoded"] = (static["Urbanisation"].astype(str).str.strip()
                                  .str.lower().map(urb_map))
    else:
        static["UrbanEncoded"] = np.nan

    # couverture vaccinale : UNE seule règle d'imputation (l'ancien code
    # traitait une couverture manquante comme 100 % pour les non-vaccinés
    # estimés et comme 0 % pour le test de seuil)
    if "Taux_Vaccination" in static.columns:
        cov = pd.to_numeric(static["Taux_Vaccination"], errors="coerce")
        static["Taux_Vaccination"] = cov
        static["Gap_Immunite_Collective"] = (95.0 - cov).clip(lower=0)
        static["Zone_Sous_Seuil"] = np.where(cov.notna(), (cov < 95).astype(float), np.nan)
        pop_enf = pd.to_numeric(static.get("Pop_Enfants", pd.Series(np.nan, index=static.index)),
                                errors="coerce")
        static["Susceptibles_Estimes"] = (1.0 - cov / 100.0) * pop_enf
    else:
        for c in ["Gap_Immunite_Collective", "Zone_Sous_Seuil", "Susceptibles_Estimes"]:
            static[c] = np.nan

    # part de cas non vaccinés : variable d'aire (statistique descriptive),
    # et NON une variable hebdomadaire calculée sur les cas de la semaine
    # cible — c'était une fuite de la cible dans l'ancien code.
    if df_linelist is not None and "Statut_Vaccinal" in df_linelist.columns:
        ll = df_linelist.copy()
        key = "Aire_Sante" if "Aire_Sante" in ll.columns else area_col
        if key in ll.columns:
            ll["_aire"] = ll[key].astype(str).str.strip().str.lower()
            known = ll[ll["Statut_Vaccinal"].isin(["Oui", "Non"])]
            if len(known):
                part = known.groupby("_aire")["Statut_Vaccinal"] \
                            .apply(lambda s: (s == "Non").mean() * 100.0)
                static["Part_NonVaccines_Cas"] = static["health_area"].map(part)
    if "Part_NonVaccines_Cas" not in static.columns:
        static["Part_NonVaccines_Cas"] = np.nan

    # `Pop_Enfants` (0-14 ans) alimente aussi les taux dérivés génériques
    if "Pop_Enfants" in static.columns and "Pop_Enfants_0_14" not in static.columns:
        static["Pop_Enfants_0_14"] = static["Pop_Enfants"]

    db = make_design_builder(sa_gdf_enrichi, static, climate_df=None,
                             use_spatial=use_spatial, neighbour_k=neighbour_k,
                             n_clusters=n_clusters, extra_static=ROUGEOLE_STATIC_EXTRA)
    info = {
        "n_areas": int(panel["health_area"].nunique()),
        "n_weeks": int(panel["week_index"].nunique()),
        "n_rows": int(len(panel)),
        "static_cols": [c for c in static.columns if c != "health_area"],
    }
    return panel, db, static, info


def run_modelling_rougeole(panel: pd.DataFrame,
                           design_builder: Callable,
                           algo: str = "XGBoost",
                           n_weeks_pred: int = 12,
                           derniere_annee: Optional[int] = None,
                           derniere_semaine: Optional[int] = None,
                           objective: str = "squared_error",
                           n_splits: int = 5,
                           embargo: int = 4,
                           progress: Optional[Callable[[str, int], None]] = None
                           ) -> Dict[str, Any]:
    """
    Modélisation rougeole. Retourne ``future_df`` au format historique
    (``Aire_Sante``, ``SemaineLabel``, ``SemaineEpi``, ``Annee``, ``sort_key``,
    ``CasPredits``) plus les métriques et l'importance des variables.
    """
    def _p(msg, pct):
        if progress is not None:
            progress(msg, pct)

    _p("Entraînement…", 20)
    log_target = False
    factory = lambda: em.make_model(algo, objective=objective)  # noqa: E731
    if objective == "poisson" and em.models.resolve_objective(algo, objective) != "poisson":
        log_target = True
        factory = lambda: em.make_model(algo, objective="squared_error")  # noqa: E731

    fitted = em.fit_pipeline(panel, design_builder=design_builder,
                             model_factory=factory, objective=objective,
                             log_target=log_target)

    _p("Validation temporelle…", 45)
    try:
        val = temporal_validation(panel, design_builder, factory,
                                  n_splits=n_splits, embargo=embargo,
                                  feature_cols=fitted["feature_cols"])
    except Exception:
        val = {"cv_r2_mean": float("nan"), "cv_r2_std": float("nan"),
               "cv_mae_mean": float("nan"), "folds": pd.DataFrame(), "pooled": {}}

    _p("Prévision…", 70)
    fc = em.recursive_forecast(fitted, panel=panel.copy(),
                               horizons=int(n_weeks_pred),
                               design_builder=design_builder)
    if len(fc):
        fc = em.predict_quantiles(fitted, fc, levels=(0.1, 0.9))
        future_df = pd.DataFrame({
            "Aire_Sante": fc["health_area"],
            "SemaineLabel": fc["period"],
            "SemaineEpi": fc["week_"].astype(int),
            "Annee": fc["year"].astype(int),
            "sort_key": fc["year"].astype(int) * 100 + fc["week_"].astype(int),
            "CasPredits": fc["predicted_cases"].round(1),
            "Q10": fc.get("q10", np.nan),
            "Q90": fc.get("q90", np.nan),
            "Horizon": fc["horizon"].astype(int),
        })
    else:
        future_df = pd.DataFrame(columns=["Aire_Sante", "SemaineLabel", "SemaineEpi",
                                          "Annee", "sort_key", "CasPredits"])

    insample = fitted["train_metrics"]
    metrics = {
        "mae": float(insample.get("mae", float("nan"))),
        "rmse": float(insample.get("rmse", float("nan"))),
        "r2": float(insample.get("r2", float("nan"))),
        "cv_r2_mean": float(val.get("cv_r2_mean", float("nan"))),
        "cv_r2_std": float(val.get("cv_r2_std", float("nan"))),
        "cv_mae_mean": float(val.get("cv_mae_mean", float("nan"))),
        "n_train": int(fitted["n_train"]),
        "n_features": len(fitted["feature_cols"]),
        "algorithme": algo,
        "objectif": ("log1p+squared_error" if log_target else objective),
    }
    _p("Terminé", 100)
    return {
        "future_df": future_df,
        "forecast": fc,
        "metrics": metrics,
        "cv_folds": val.get("folds"),
        "importance": em.forecast.feature_importance(fitted, top_n=30),
        "feature_cols": fitted["feature_cols"],
        "fitted": {k: v for k, v in fitted.items() if k != "model"},
        "_model": fitted["model"],
    }


# ======================================================================
# WorldPop : sélection d'un millésime unique
# ======================================================================
WORLDPOP_COLLECTION = "WorldPop/GP/100m/pop_age_sex"


def worldpop_mosaic(ee_module, year: Optional[int] = None, verbose: bool = True):
    """
    Mosaïque WorldPop restreinte à **un seul millésime**.

    **Correction (action D3 de l'audit)** : les deux applications appelaient
    ``dataset.mosaic()`` sans filtre temporel. Or cette collection contient une
    image par pays **et par année**, couvrant la même emprise : la mosaïque
    superposait donc tous les millésimes et retenait, pixel par pixel, celui qui
    arrivait en dernier — ordre non garanti. Une même aire de santé pouvait se
    retrouver avec des pixels de 2015 et d'autres de 2020, et les taux
    d'incidence reposaient sur des dénominateurs d'années différentes.

    Comportement :
      * ``year`` fourni  -> filtre sur l'année civile demandée ;
      * ``year`` absent  -> dernier millésime présent dans la collection ;
      * échec de détermination -> repli sur la mosaïque non filtrée, avec un
        avertissement : mieux vaut une donnée approximative mais présente
        qu'une interruption de la chaîne.

    Retourne ``(image, millésime_utilisé)``.
    """
    ee = ee_module
    dataset = ee.ImageCollection(WORLDPOP_COLLECTION)

    if year is None:
        try:
            # date de début la plus récente -> millésime disponible le plus récent
            derniere = dataset.aggregate_max("system:time_start")
            if derniere is not None:
                import datetime as _dt
                year = _dt.datetime.utcfromtimestamp(int(derniere) / 1000).year
        except Exception:
            year = None

    if year is None:
        if verbose:
            import warnings as _w
            _w.warn("WorldPop : millésime indéterminé, mosaïque non filtrée "
                    "(plusieurs années mélangées).", RuntimeWarning, stacklevel=2)
        return dataset.mosaic(), None

    debut, fin = f"{year}-01-01", f"{year + 1}-01-01"
    filtre = dataset.filterDate(debut, fin)
    try:
        if filtre.size().getInfo() == 0:
            if verbose:
                import warnings as _w
                _w.warn(f"WorldPop : aucune image pour {year}, repli sur la "
                        f"mosaïque non filtrée.", RuntimeWarning, stacklevel=2)
            return dataset.mosaic(), None
    except Exception:
        # size().getInfo() suppose un accès réseau : en cas d'échec on ne bloque
        # pas la chaîne, le filtre reste appliqué.
        pass
    return filtre.mosaic(), year

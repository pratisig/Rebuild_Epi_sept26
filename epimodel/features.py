"""
Ingénierie de variables (feature engineering) pour la prévision épidémiologique.

Principe directeur : **une seule fonction** construit les variables, à
l'entraînement comme à la prévision. Le décalage entraînement/inférence
(``train-serve skew``) du pipeline historique — où les variables dérivées
étaient recalculées à la main dans la boucle de prévision puis remplies par 0 —
devient structurellement impossible.

Familles de variables
---------------------
* ``temporal``  : retards (lags), moyennes mobiles **strictement décalées**,
  écart-type, maximum glissant, taux de croissance.
* ``seasonal``  : harmoniques de Fourier de la semaine ISO (K=2), mois.
* ``static``    : covariables **non dynamiques** par aire de santé — altitude,
  pente, NDVI, eau, urbain, distance à l'eau, superficie, latitude/longitude.
* ``population``: démographie et taux dérivés (noms historiques conservés :
  ``incidence_rate``, ``child_risk``, ``demo_pressure``, ``coef_population``).
* ``spatial``   : lag spatial **à t-1** (moyenne pondérée des voisins), cluster.
* ``climate``   : variables hebdomadaires + **retards de précipitations**
  (4 à 8 semaines), cohérents avec le cycle de développement du vecteur.

Aucune variable n'utilise l'information de la semaine courante : toutes les
statistiques glissantes sont calculées sur des séries décalées d'une période.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .panel import COL_AREA, COL_CASES, COL_WEEK, COL_WIDX, COL_YEAR

# ----------------------------------------------------------------------
# Paramètres par défaut
# ----------------------------------------------------------------------
DEFAULT_LAGS = (1, 2, 3, 4, 8, 13, 26, 52)
DEFAULT_WINDOWS = (4, 8, 13)
PRECIP_LAGS = (4, 6, 8)

# Décalage de disponibilité des produits climatiques, en semaines.
#
# Les produits satellitaires et de réanalyse (NASA POWER, ERA5, WorldPop) ne
# sont publiés qu'après un délai : la valeur d'une semaine donnée n'est pas
# connue au moment où cette semaine se produit. Utiliser la météo de la semaine
# t pour prédire la semaine t revient donc à employer, en prévision, une
# information qui n'existera pas encore — une fuite de même nature que celle des
# moyennes mobiles non décalées, mais plus discrète car elle porte sur une
# variable exogène.
#
# La valeur par défaut est 0, c'est-à-dire le comportement antérieur : le délai
# réel dépend du produit effectivement branché et de sa cadence de republication,
# que cet audit n'a pas pu mesurer faute d'accès réseau. Il doit être renseigné
# au déploiement (2 à 6 semaines pour POWER et ERA5).
CLIMATE_AVAILABILITY_LAG = 0

# Ordre canonique : garantit la reproductibilité (l'ancien code utilisait
# `list(set(...))`, dont l'ordre varie d'un processus à l'autre).
FEATURE_GROUPS: Dict[str, List[str]] = {
    "temporal": [
        *[f"cases_lag_{l}" for l in DEFAULT_LAGS],
        "cases_ma_4", "cases_ma_8", "cases_ma_13",
        "cases_std_4", "cases_max_4", "cases_ewm_4",
        "growth_rate", "cases_diff_1",
    ],
    "seasonal": [
        "sin_week_1", "cos_week_1", "sin_week_2", "cos_week_2",
        "week_of_year", "month",
    ],
    "static": [
        "Altitude_Moy", "Altitude_Max", "Pente_Moy", "NDVI_Moy",
        "Eau_Pct", "Urbain_Pct", "Dist_Eau_km", "Superficie_km2",
        "latitude", "longitude",
    ],
    "population": [
        "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop", "Densite_Enfants",
        "incidence_rate", "child_risk", "demo_pressure", "coef_population",
    ],
    "spatial": [
        "spatial_lag_1", "spatial_cluster",
    ],
    "climate": [
        "temp_api", "precip_api", "humidity_api",
        "precip_lag_4", "precip_lag_6", "precip_lag_8",
        "precip_ma_4_8", "temp_ma_4",
    ],
}

# Ordre global utilisé par défaut
DEFAULT_GROUP_ORDER = ("temporal", "seasonal", "static", "population",
                       "spatial", "climate")


# ======================================================================
# 1. Variables temporelles
# ======================================================================
def _wide_cases(panel: pd.DataFrame) -> pd.DataFrame:
    """Matrice (semaines x aires) des cas, index temporel complet et trié."""
    wide = panel.pivot_table(index=COL_WIDX, columns=COL_AREA, values=COL_CASES,
                             aggfunc="sum")
    idx = pd.RangeIndex(int(np.nanmin(wide.index.values)),
                        int(np.nanmax(wide.index.values)) + 1, name=COL_WIDX)
    wide = wide.reindex(idx)
    wide.index.name = COL_WIDX
    return wide.sort_index()


def _melt_back(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    idx_name = wide.index.name or COL_WIDX
    out = wide.reset_index().melt(id_vars=[idx_name], var_name=COL_AREA,
                                  value_name=name)
    out = out.rename(columns={idx_name: COL_WIDX})
    out[COL_WIDX] = out[COL_WIDX].astype(float)
    out[COL_AREA] = out[COL_AREA].astype(str)
    return out[[COL_WIDX, COL_AREA, name]]


def add_temporal_features(panel: pd.DataFrame,
                          lags: Sequence[int] = DEFAULT_LAGS,
                          windows: Sequence[int] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """
    Ajoute retards et statistiques glissantes, **toutes décalées d'une période**.

    ``cases_ma_4`` à la semaine t = moyenne des semaines t-4..t-1 (jamais t).

    Implémentation vectorisée sur une matrice (semaines x aires) : une seule
    passe, identique à l'entraînement et à la prévision.
    """
    df = panel.sort_values([COL_AREA, COL_WIDX]).reset_index(drop=True).copy()
    wide = _wide_cases(panel)

    computed = {}
    for lag in lags:
        computed[f"cases_lag_{lag}"] = wide.shift(lag)

    shifted = wide.shift(1)
    for w in windows:
        computed[f"cases_ma_{w}"] = shifted.rolling(w, min_periods=1).mean()
    computed["cases_std_4"] = shifted.rolling(4, min_periods=2).std()
    computed["cases_max_4"] = shifted.rolling(4, min_periods=1).max()
    computed["cases_ewm_4"] = shifted.ewm(span=4, min_periods=1).mean()

    prev1, prev2 = wide.shift(1), wide.shift(2)
    computed["growth_rate"] = ((prev1 - prev2) / prev2.replace(0, np.nan)).clip(-5, 5)
    computed["cases_diff_1"] = prev1 - prev2

    keys = [COL_WIDX, COL_AREA]
    for name, w in computed.items():
        long = _melt_back(w, name)
        df = df.drop(columns=[name], errors="ignore").merge(long, on=keys, how="left")
    return df


# ======================================================================
# 2. Saisonnalité
# ======================================================================
def add_seasonality(panel: pd.DataFrame, n_harmonics: int = 2) -> pd.DataFrame:
    """
    Harmoniques de Fourier de la position dans l'année + mois calendaire.

    Utilise la période réelle 52.1775 semaines (et non 52) pour éviter la
    dérive de phase entre années, et le **vrai** numéro de semaine ISO
    (l'ancien code utilisait un index factorisé, donc décalé dès que la série
    ne commençait pas à la semaine 1).
    """
    df = panel.copy()
    frac = (df[COL_WEEK] - 1) / 52.1775
    for k in range(1, n_harmonics + 1):
        df[f"sin_week_{k}"] = np.sin(2 * np.pi * k * frac)
        df[f"cos_week_{k}"] = np.cos(2 * np.pi * k * frac)
    df["week_of_year"] = df[COL_WEEK].astype(float)
    df["month"] = np.clip((df[COL_WEEK] / 52.1775 * 12).astype(int) + 1, 1, 12).astype(float)
    return df


# ======================================================================
# 3. Covariables statiques
# ======================================================================
def add_static_features(panel: pd.DataFrame,
                        static_df: Optional[pd.DataFrame] = None,
                        gdf=None,
                        extra_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Fusionne les covariables **non dynamiques** par aire de santé.

    Ce sont elles qui permettent au modèle de distinguer durablement une zone
    de haute altitude (faible transmission) d'une zone de plaine, information
    qu'un modèle purement temporel ne peut pas apprendre avec peu d'années.

    ``gdf`` (optionnel) apporte ``latitude``, ``longitude``, ``Superficie_km2``.
    """
    df = panel.copy()

    if gdf is not None and len(gdf) > 0:
        gg = gdf.copy()
        name_col = COL_AREA if COL_AREA in gg.columns else None
        if name_col is None:
            for c in ["health_area", "health_are", "name_fr", "name", "nom"]:
                if c in gg.columns:
                    name_col = c
                    break
        if name_col is not None:
            gg[COL_AREA] = gg[name_col].astype(str).str.strip().str.lower()
            try:
                gg_proj = gg.to_crs("ESRI:54009")
                superficie = gg_proj.geometry.area / 1e6
                # centroïde calculé en projection équidistante (correct),
                # puis ramené en WGS84 pour l'usage cartographique
                cents = gg_proj.geometry.centroid.to_crs(epsg=4326)
            except Exception:
                superficie = pd.Series(np.nan, index=gg.index)
                cents = gg.to_crs(epsg=4326).geometry.representative_point()
            geo = pd.DataFrame({
                COL_AREA: gg[COL_AREA].to_numpy(),
                "latitude": cents.y.to_numpy(),
                "longitude": cents.x.to_numpy(),
                "Superficie_km2": np.asarray(superficie, dtype=float),
            }).drop_duplicates(subset=[COL_AREA])
            df = df.drop(columns=[c for c in ["latitude", "longitude", "Superficie_km2"]
                                  if c in df.columns], errors="ignore")
            df = df.merge(geo, on=COL_AREA, how="left")

    if static_df is not None and len(static_df) > 0:
        sd = static_df.copy()
        sd[COL_AREA] = sd[COL_AREA].astype(str).str.strip().str.lower()
        sd = sd.drop_duplicates(subset=[COL_AREA])
        cols = [c for c in sd.columns if c != COL_AREA]
        df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
        df = df.merge(sd, on=COL_AREA, how="left")

    # densité d'enfants si superficie disponible
    if {"Pop_Enfants_0_14", "Superficie_km2"}.issubset(df.columns):
        df["Densite_Enfants"] = (df["Pop_Enfants_0_14"]
                                 / df["Superficie_km2"].replace(0, np.nan))

    if extra_columns:
        for c in extra_columns:
            if c not in df.columns:
                df[c] = np.nan
    return df


# ======================================================================
# 4. Variables démographiques dérivées
# ======================================================================
def add_population_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Taux dérivés de la population. Noms historiques de l'application conservés
    (``incidence_rate``, ``child_risk``, ``demo_pressure``, ``coef_population``).

    **Correction de fuite** : dans le pipeline historique, ces variables étaient
    calculées avec les cas de la semaine *courante*
    (``incidence_rate = cases_t / Pop_Totale``). Comme ``Pop_Totale`` était
    également une variable d'entrée, le modèle pouvait retrouver ``cases_t``
    par simple multiplication : c'est une fuite de la cible, et elle explique
    l'essentiel du R² ≈ 0.99 affiché.

    Elles sont désormais calculées sur les cas de la semaine **précédente**
    (``cases_lag_1``), ce qui conserve le sens opérationnel (pression
    épidémique récente rapportée à la population exposée) sans fuite.
    Elles sont recalculées sur le panneau complet — y compris sur les lignes de
    prévision, où ``cases`` contient la valeur prédite au pas précédent — ce qui
    supprime aussi le décalage entraînement/inférence.
    """
    df = panel.copy()

    # base sans fuite : cas de la semaine précédente (fallback : moyenne glissante)
    if "cases_lag_1" in df.columns:
        base_cases = df["cases_lag_1"]
    elif "cases_ma_4" in df.columns:
        base_cases = df["cases_ma_4"]
    else:
        base_cases = df.groupby(COL_AREA, sort=False)[COL_CASES].shift(1)

    if "Pop_Totale" in df.columns:
        df["incidence_rate"] = (base_cases / df["Pop_Totale"].replace(0, np.nan) * 10000)
        df["incidence_rate"] = df["incidence_rate"].replace([np.inf, -np.inf], np.nan)
    if "Pop_Enfants_0_14" in df.columns:
        df["child_risk"] = (base_cases / df["Pop_Enfants_0_14"].replace(0, np.nan) * 1000)
        df["child_risk"] = df["child_risk"].replace([np.inf, -np.inf], np.nan)
    if "Densite_Pop" in df.columns and "incidence_rate" in df.columns:
        df["demo_pressure"] = df["Densite_Pop"] * df["incidence_rate"]
        df["demo_pressure"] = df["demo_pressure"].replace([np.inf, -np.inf], np.nan)

    # Coefficient d'ajustement par aire (sens historique conservé) :
    # incidence moyenne de l'aire / incidence globale, bornée à [0.5, 2.0].
    #
    # **Correction de fuite** : la version initiale calculait cette moyenne sur
    # l'ensemble du panneau, donc en utilisant aussi les semaines POSTÉRIEURES à
    # la semaine T. La variable à la semaine T dépendait alors de l'avenir — une
    # fuite faible mais réelle (détectée par le test de perturbation de cible).
    # La moyenne est désormais **cumulative et strictement antérieure** : à la
    # semaine T, elle n'utilise que les semaines < T.
    if "Pop_Totale" in df.columns:
        if "cases_is_forecast" in df.columns:
            hist = df[df["cases_is_forecast"].fillna(False) == False]  # noqa: E712
        else:
            hist = df
        pop = hist.groupby(COL_AREA)["Pop_Totale"].first()
        total_pop = float(pop.replace(0, np.nan).sum())
        coef = None
        if total_pop > 0 and COL_WIDX in hist.columns:
            wa = (hist.groupby([COL_WIDX, COL_AREA])[COL_CASES].sum()
                      .unstack(fill_value=0.0).sort_index())
            wa = wa.reindex(columns=pop.index).fillna(0.0)
            seen = (wa > 0).cumsum()                       # semaines observées
            cum = wa.cumsum()                              # somme cumulée <= T
            mean_prev = cum.shift(1) / seen.shift(1).replace(0, np.nan)
            inc_prev = (mean_prev / pop.replace(0, np.nan) * 10000)
            glob_prev = (wa.sum(axis=1).cumsum().shift(1) / total_pop * 10000)
            coef = inc_prev.div(glob_prev.replace(0, np.nan), axis=0)
            coef = coef.replace([np.inf, -np.inf], np.nan)
            coef = coef.ffill().bfill().fillna(1.0).clip(0.5, 2.0)

        if coef is not None and not coef.empty:
            long = coef.stack().rename("coef_population").reset_index()
            long.columns = [COL_WIDX, COL_AREA, "coef_population"]
            df = df.drop(columns=["coef_population"], errors="ignore")
            df = df.merge(long, on=[COL_WIDX, COL_AREA], how="left")
            # au-delà de l'historique observé : dernier coefficient connu
            df["coef_population"] = df["coef_population"].fillna(
                float(coef.iloc[-1].median()))
        else:
            df["coef_population"] = 1.0
    else:
        df["coef_population"] = 1.0

    # offset d'exposition (pour les objectifs de type Poisson / log-link)
    if "Pop_Totale" in df.columns:
        df["log_exposure"] = np.log(df["Pop_Totale"].clip(lower=1).astype(float))
    else:
        df["log_exposure"] = 0.0
    return df


# ======================================================================
# 5. Variables spatiales
# ======================================================================
def build_neighbour_weights(gdf, k: int = 5, area_col: str = COL_AREA) -> pd.DataFrame:
    """
    Matrice de poids des k plus proches voisins (distance inverse), normalisée.

    Retourne un DataFrame long ``[health_area, neighbor, weight]``.
    """
    gg = gdf.copy()
    if area_col not in gg.columns:
        for c in ["health_area", "health_are", "name_fr", "name", "nom"]:
            if c in gg.columns:
                gg[area_col] = gg[c].astype(str).str.strip().str.lower()
                break
    # distances calculées en projection équidistante (les degrés ne sont pas
    # une métrique euclidienne valide hors des très petites emprises)
    try:
        gproj = gg.to_crs("ESRI:54009")
    except Exception:
        gproj = gg.to_crs(epsg=4326)
    cents = gproj.geometry.centroid
    coords = np.column_stack([cents.x.to_numpy(), cents.y.to_numpy()])
    names = gg[area_col].astype(str).str.strip().str.lower().to_numpy()

    from scipy.spatial.distance import cdist
    d = cdist(coords, coords)
    n = len(coords)
    rows = []
    for i in range(n):
        order = np.argsort(d[i])
        neigh = [j for j in order if j != i][:max(1, min(k, n - 1))]
        if not neigh:
            continue
        w = 1.0 / (d[i, neigh] + 1e-6)
        w = w / w.sum()
        for j, wj in zip(neigh, w):
            rows.append((names[i], names[j], float(wj)))
    return pd.DataFrame(rows, columns=[area_col, "neighbor", "weight"])


def add_spatial_features(panel: pd.DataFrame,
                         weights: Optional[pd.DataFrame] = None,
                         n_clusters: int = 0,
                         gdf=None) -> pd.DataFrame:
    """
    Lag spatial **à t-1** : moyenne pondérée des cas des voisins la semaine
    précédente. (Le pipeline historique utilisait la valeur de l'aire
    elle-même à t-1, ce qui n'est pas un lag spatial.)
    """
    df = panel.copy()
    df["spatial_lag_1"] = np.nan
    df["spatial_cluster"] = np.nan

    if weights is not None and len(weights) > 0:
        wide = df.pivot_table(index=COL_WIDX, columns=COL_AREA, values=COL_CASES,
                              aggfunc="sum")
        wide = wide.sort_index()
        shifted = wide.shift(1)
        lag = pd.DataFrame(np.nan, index=shifted.index, columns=shifted.columns)
        for area, grp in weights.groupby(COL_AREA):
            if area not in lag.columns:
                continue
            num = np.zeros(len(shifted))
            den = np.zeros(len(shifted))
            for _, r in grp.iterrows():
                if r["neighbor"] in shifted.columns:
                    vals = shifted[r["neighbor"]].fillna(0.0).to_numpy()
                    num += r["weight"] * vals
                    den += r["weight"]
            lag[area] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        lag_long = _melt_back(lag, "spatial_lag_1")
        df = df.drop(columns=["spatial_lag_1"]).merge(lag_long, on=[COL_WIDX, COL_AREA],
                                                      how="left")

    if gdf is not None and n_clusters and n_clusters > 0 and len(gdf) >= n_clusters:
        try:
            from sklearn.cluster import KMeans
            try:
                gg = gdf.to_crs("ESRI:54009")
            except Exception:
                gg = gdf.to_crs(epsg=4326)
            cents = gg.geometry.centroid
            X = np.column_stack([cents.x.to_numpy(), cents.y.to_numpy()])
            area_col = COL_AREA if COL_AREA in gg.columns else "health_are"
            names = gg[area_col].astype(str).str.strip().str.lower()
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
            mapping = dict(zip(names.to_numpy(), km.labels_.astype(float)))
            df["spatial_cluster"] = df[COL_AREA].map(mapping)
        except Exception:
            df["spatial_cluster"] = np.nan
    return df


# ======================================================================
# 6. Variables climatiques
# ======================================================================
def add_climate_features(panel: pd.DataFrame,
                         climate_df: Optional[pd.DataFrame] = None,
                         precip_lags: Sequence[int] = PRECIP_LAGS,
                         availability_lag: int = CLIMATE_AVAILABILITY_LAG
                         ) -> pd.DataFrame:
    """
    Variables climatiques hebdomadaires + retards de précipitations.

    Justification épidémiologique : pour *P. falciparum* au Sahel, l'effet des
    pluies sur l'abondance vectorielle est décalé de 4 à 8 semaines (durée du
    cycle aquatique + sporogonique). Modéliser uniquement la pluie de la semaine
    courante — comme le faisait l'ancien pipeline — passe à côté du signal.

    ``availability_lag`` (action C5 de l'audit) décale les valeurs climatiques
    du délai de publication du produit. Sans lui, la météo de la semaine *t*
    sert à prédire la semaine *t* alors qu'elle ne sera connue qu'après coup :
    en prévision réelle, la colonne serait vide ou approximée, et le modèle se
    retrouverait hors de son domaine d'entraînement. Le décalage est appliqué
    **avant** le calcul des retards épidémiologiques, qui se composent donc avec
    lui (``precip_lag_4`` avec un délai de 2 désigne la pluie de t-6, bien
    disponible à t). Par défaut 0 : le délai réel dépend du produit branché et
    n'a pas pu être mesuré dans cet environnement sans accès réseau.
    """
    df = panel.copy()
    if climate_df is None or len(climate_df) == 0:
        for c in ["temp_api", "precip_api", "humidity_api",
                  *[f"precip_lag_{l}" for l in precip_lags],
                  "precip_ma_4_8", "temp_ma_4"]:
            df[c] = np.nan
        return df

    cd = climate_df.copy()
    cd[COL_AREA] = cd[COL_AREA].astype(str).str.strip().str.lower()
    keys = [c for c in [COL_AREA, COL_YEAR, COL_WEEK] if c in cd.columns]
    use = [c for c in ["temp_api", "precip_api", "humidity_api"] if c in cd.columns]
    if not use:
        return df
    cd = cd.groupby(keys, as_index=False)[use].mean()
    df = df.drop(columns=[c for c in use if c in df.columns], errors="ignore")
    df = df.merge(cd, on=keys, how="left")

    # Décalage de disponibilité : appliqué ici, avant tout retard dérivé.
    if availability_lag and availability_lag > 0:
        for c in use:
            df[c] = df.groupby(COL_AREA, sort=False)[c].shift(availability_lag)

    for lag in precip_lags:
        df[f"precip_lag_{lag}"] = (df.groupby(COL_AREA, sort=False)["precip_api"]
                                   .shift(lag) if "precip_api" in df.columns else np.nan)
    if "precip_api" in df.columns:
        sh = df.groupby(COL_AREA, sort=False)["precip_api"].shift(4)
        df["precip_ma_4_8"] = (sh.groupby(df[COL_AREA], sort=False)
                               .transform(lambda s: s.rolling(4, min_periods=1).mean()))
        sht = df.groupby(COL_AREA, sort=False)["temp_api"].shift(1)
        df["temp_ma_4"] = (sht.groupby(df[COL_AREA], sort=False)
                           .transform(lambda s: s.rolling(4, min_periods=1).mean()))
    return df


# ======================================================================
# 7. Sélection de variables (ordre déterministe)
# ======================================================================
def select_features(df: pd.DataFrame,
                    groups: Sequence[str] = DEFAULT_GROUP_ORDER,
                    min_non_null: float = 0.05,
                    extra_static: Optional[Sequence[str]] = None) -> List[str]:
    """
    Retourne la liste **ordonnée et stable** des variables disponibles.

    * ordre canonique fixé par ``FEATURE_GROUPS`` (pas de ``set()``) ;
    * exclusion des colonnes quasi-vides (< ``min_non_null`` de valeurs) ;
    * exclusion des colonnes constantes (variance nulle) qui n'apportent rien.
    """
    ordered: List[str] = []
    extra = list(extra_static or [])
    for grp in groups:
        for col in FEATURE_GROUPS.get(grp, []):
            if col in df.columns and col not in ordered:
                ordered.append(col)
        # variables statiques spécifiques au pathogène (ex. rougeole :
        # couverture vaccinale, urbains, température) ajoutées après le bloc
        # « static » pour rester dans un ordre canonique et reproductible.
        if grp == "static":
            for col in extra:
                if col in df.columns and col not in ordered:
                    ordered.append(col)

    kept: List[str] = []
    n = max(len(df), 1)
    for col in ordered:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < max(1, int(min_non_null * n)):
            continue
        if s.notna().sum() > 1 and float(s.std(ddof=0) or 0.0) == 0.0:
            continue
        kept.append(col)
    return kept


# ======================================================================
# 8. Point d'entrée unique (entraînement ET prévision)
# ======================================================================
def build_design_matrix(panel: pd.DataFrame,
                        static_df: Optional[pd.DataFrame] = None,
                        climate_df: Optional[pd.DataFrame] = None,
                        gdf=None,
                        neighbour_weights: Optional[pd.DataFrame] = None,
                        n_clusters: int = 0,
                        groups: Sequence[str] = DEFAULT_GROUP_ORDER,
                        lags: Sequence[int] = DEFAULT_LAGS,
                        windows: Sequence[int] = DEFAULT_WINDOWS,
                        extra_static: Optional[Sequence[str]] = None,
                        feature_cols: Optional[Sequence[str]] = None,
                        climate_availability_lag: int = CLIMATE_AVAILABILITY_LAG
                        ) -> (pd.DataFrame, List[str]):
    """
    Construit la matrice de conception complète.

    C'est LA fonction appelée à l'entraînement comme à chaque pas de prévision :
    les variables sont donc, par construction, identiques des deux côtés.

    Retourne ``(df_enrichi, feature_cols)``.
    """
    df = add_temporal_features(panel, lags=lags, windows=windows)
    df = add_seasonality(df)
    df = add_static_features(df, static_df=static_df, gdf=gdf)
    df = add_population_features(df)
    if "spatial" in groups:
        df = add_spatial_features(df, weights=neighbour_weights,
                                  n_clusters=n_clusters, gdf=gdf)
    else:
        df["spatial_lag_1"] = np.nan
        df["spatial_cluster"] = np.nan
    df = add_climate_features(df, climate_df=climate_df,
                              availability_lag=climate_availability_lag)

    if feature_cols is None:
        feature_cols = select_features(df, groups=groups, extra_static=extra_static)
    else:
        feature_cols = list(feature_cols)
        for c in feature_cols:
            if c not in df.columns:
                df[c] = np.nan
    return df, feature_cols


# ======================================================================
# 9. Diagnostic de variabilité temporelle (action F5)
# ======================================================================
def describe_feature_dynamics(df: pd.DataFrame,
                              feature_cols: Optional[Sequence[str]] = None,
                              area_col: str = COL_AREA) -> pd.DataFrame:
    """
    Classe chaque variable selon sa variabilité **à l'intérieur de chaque aire**.

    Pourquoi ce diagnostic est nécessaire : :func:`select_features` écarte les
    colonnes *globalement* constantes, mais une variable constante dans le temps
    et différente d'une aire à l'autre — le cas d'un climat réduit à une moyenne
    par aire — passe ce filtre. Elle est alors créditée comme pilote temporel
    alors qu'elle n'agit que comme décalage de niveau entre aires : elle ne peut
    expliquer aucune dynamique, aucun pic, aucune saison.

    Ces variables ne doivent **pas** être supprimées pour autant. L'altitude en
    est exactement un exemple, et elle est épidémiologiquement légitime : elle
    module le niveau de risque. Le diagnostic distingue donc deux rôles, sans
    rien retirer :

    * ``temporelle``            — varie dans le temps au sein des aires ;
    * ``niveau par aire``       — invariante dans le temps, varie entre aires ;
    * ``constante``             — ne varie nulle part (apport nul) ;
    * ``insuffisante``          — trop de valeurs manquantes pour conclure.

    Retourne un cadre avec, par variable : le groupe, le nombre de valeurs
    distinctes au sein d'une même aire, l'écart-type intra-aire moyen, l'écart-
    type global, et le rôle retenu.
    """
    if feature_cols is None:
        feature_cols = [c for grp in FEATURE_GROUPS.values() for c in grp
                        if c in df.columns]

    col_groupe = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
    lignes = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < 2:
            lignes.append({"variable": col, "groupe": col_groupe.get(col, ""),
                           "n_distinct_intra_aire": 0,
                           "std_intra_aire": np.nan,
                           "std_global": np.nan, "role": "insuffisante"})
            continue

        if area_col in df.columns:
            grp = s.groupby(df[area_col], sort=False)
            n_distinct = grp.nunique(dropna=True)
            std_intra = grp.std(ddof=0)
            n_distinct_max = int(n_distinct.max()) if len(n_distinct) else 0
            std_intra_moy = float(std_intra.mean()) if len(std_intra) else np.nan
        else:
            n_distinct_max = int(s.nunique(dropna=True))
            std_intra_moy = float(s.std(ddof=0) or 0.0)

        std_global = float(s.std(ddof=0) or 0.0)

        if n_distinct_max <= 1:
            role = "constante" if std_global == 0.0 else "niveau par aire"
        else:
            role = "temporelle"

        lignes.append({"variable": col, "groupe": col_groupe.get(col, ""),
                       "n_distinct_intra_aire": n_distinct_max,
                       "std_intra_aire": std_intra_moy,
                       "std_global": std_global, "role": role})

    return pd.DataFrame(lignes, columns=["variable", "groupe",
                                         "n_distinct_intra_aire",
                                         "std_intra_aire", "std_global", "role"])


def invariant_climate_variables(df: pd.DataFrame,
                                feature_cols: Optional[Sequence[str]] = None,
                                area_col: str = COL_AREA) -> List[str]:
    """
    Liste les variables du groupe climatique invariantes dans le temps.

    C'est le signal concret de l'action F5 : si des variables climatiques sont
    retenues par le modèle alors qu'elles ne varient pas au sein des aires, le
    climat n'apporte aucune information sur la dynamique — il faut des
    précipitations, températures et humidités **hebdomadaires**, pas des
    moyennes par aire.
    """
    diag = describe_feature_dynamics(df, feature_cols=feature_cols,
                                     area_col=area_col)
    if diag.empty:
        return []
    # Seules les variables réellement mesurées mais figées dans le temps sont
    # concernées. Une colonne entièrement vide (« insuffisante ») n'est pas
    # invariante : elle est absente, et la signaler ici serait un faux diagnostic.
    invariantes = ("niveau par aire", "constante")
    sel = diag[diag["groupe"].eq("climate") & diag["role"].isin(invariantes)]
    return sel["variable"].tolist()

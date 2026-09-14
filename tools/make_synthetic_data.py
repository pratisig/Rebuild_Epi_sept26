"""
Génération d'un jeu de données SYNTHÉTIQUE mais structurellement identique
aux tables réellement utilisées par EpiPrediction.

Pourquoi synthétique ?
----------------------
Les API externes de l'application (Google Earth Engine, NASA POWER, WorldPop)
ne sont pas joignables depuis l'environnement de test. On reconstruit donc un
panneau `health_area x semaine` dont :
  * la géométrie est RÉELLE (ao_hlthArea.zip, iso3 = ner, 72 aires de santé) ;
  * les noms de colonnes sont EXACTEMENT ceux de l'application ;
  * la dynamique reproduit les propriétés connues du paludisme sahélien
    (saisonnalité marquée août-octobre, surdispersion, effet altitude).

Deux variantes du générateur (DGP) sont disponibles afin de tester honnêtement
l'apport des covariables statiques :
  * altitude_effect=True  -> l'altitude influence réellement la transmission ;
  * altitude_effect=False -> l'altitude n'a AUCUN effet (contrôle négatif).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHP_ZIP = os.path.join(REPO, "data", "ao_hlthArea.zip")

# Colonnes EXACTES de l'application (à conserver telles quelles)
CASES_COLUMNS = ["health_area", "week_", "cases", "deaths", "year", "period"]
POP_COLUMNS = ["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop",
               "Superficie_km2"]
STATIC_COLUMNS = ["health_area", "Altitude_Moy", "Altitude_Max", "Pente_Moy",
                  "NDVI_Moy", "Eau_Pct", "Urbain_Pct", "Dist_Eau_km"]


@dataclass
class SyntheticDataset:
    df_cases: pd.DataFrame          # format CSV de l'application
    df_population: pd.DataFrame     # colonnes WorldPop de l'application
    df_static: pd.DataFrame         # covariables statiques (altitude, ...)
    gdf: "object"                   # GeoDataFrame des aires de santé (EPSG:4326)
    meta: Dict[str, object]


# ----------------------------------------------------------------------
def load_health_areas(iso3: str = "ner"):
    import geopandas as gpd
    gdf = gpd.read_file(SHP_ZIP)
    gdf = gdf[gdf["iso3"].astype(str).str.strip().str.lower() == iso3].copy()
    # même normalisation que app_paludisme.load_health_areas_from_zip
    name_col = None
    for col in ["health_area", "health_are", "name_fr", "name", "NAME", "nom", "NOM"]:
        if col in gdf.columns:
            name_col = col
            break
    gdf["health_area"] = (gdf[name_col].astype(str).str.strip().str.lower()
                          if name_col else [f"aire_{i+1}" for i in range(len(gdf))])
    gdf = gdf[gdf.geometry.is_valid]
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf.reset_index(drop=True)


# ----------------------------------------------------------------------
def altitude_field(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Champ d'altitude plausible pour le Niger (m).
    Plaine sud ~200-300 m, plateau ~400-600 m, massif de l'Aïr au nord-est.
    (Synthétique : aucune API DEM accessible depuis l'environnement de test.)
    """
    base = 220.0 + 260.0 * np.clip((lat - 11.5) / 9.0, 0, 1)
    # Massif de l'Aïr (centre ~ 18.3 N, 8.3 E)
    d_air = np.sqrt(((lat - 18.3) / 1.6) ** 2 + ((lon - 8.3) / 1.9) ** 2)
    air = 900.0 * np.exp(-0.5 * d_air ** 2)
    # Falaises de Bandiagara / Adar Doutchi (sud-ouest)
    d_adar = np.sqrt(((lat - 14.2) / 1.2) ** 2 + ((lon - 5.6) / 1.4) ** 2)
    adar = 320.0 * np.exp(-0.5 * d_adar ** 2)
    rng = np.random.default_rng(1234)
    noise = rng.normal(0, 35, size=lat.shape)
    return np.clip(base + air + adar + noise, 150, 1900)


# ----------------------------------------------------------------------
def build_dataset(iso3: str = "ner",
                  years: Tuple[int, ...] = (2022, 2023, 2024),
                  seed: int = 20240914,
                  altitude_effect: bool = True,
                  drop_zero_rows: float = 0.55,
                  n_weeks: int = 52) -> SyntheticDataset:
    """Construit le panneau synthétique complet."""
    rng = np.random.default_rng(seed)

    gdf = load_health_areas(iso3)
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y

    # Surface (km²) — même méthode que l'application (ESRI:54009)
    gdf_m = gdf.to_crs("ESRI:54009")
    superficie = gdf_m.geometry.area / 1e6

    # ── Covariables statiques ──────────────────────────────────
    alt = altitude_field(gdf["lat"].to_numpy(), gdf["lon"].to_numpy())
    # pente approximée par gradient local du champ d'altitude (proxy)
    pente = np.abs(np.gradient(np.sort(alt)))[: len(alt)] * 1.0
    ndvi = np.clip(0.45 - 0.018 * (alt - 250) / 100 + rng.normal(0, 0.05, len(alt)), 0.05, 0.75)
    eau = np.clip(rng.gamma(2.0, 0.9, len(alt)) - 0.02 * (alt - 250) / 100, 0, 25)
    urbain = np.clip(rng.gamma(1.4, 1.1, len(alt)), 0, 30)
    dist_eau = np.clip(rng.gamma(2.2, 1.6, len(alt)), 0.1, 40)

    df_static = pd.DataFrame({
        "health_area": gdf["health_area"],
        "Altitude_Moy": np.round(alt, 1),
        "Altitude_Max": np.round(alt * 1.25 + 60, 1),
        "Pente_Moy": np.round(pente, 3),
        "NDVI_Moy": np.round(ndvi, 4),
        "Eau_Pct": np.round(eau, 3),
        "Urbain_Pct": np.round(urbain, 3),
        "Dist_Eau_km": np.round(dist_eau, 2),
    })

    # ── Population (structure WorldPop de l'application) ────────
    densite = np.clip(rng.lognormal(mean=np.log(22.0), sigma=0.65, size=len(gdf)), 2.0, 900.0)
    pop_tot = superficie.to_numpy() * densite
    pop_enf = pop_tot * 0.48  # ~48 % de 0-14 ans au Niger
    df_population = pd.DataFrame({
        "health_area": gdf["health_area"],
        "Pop_Totale": np.round(pop_tot).astype("int64"),
        "Pop_Enfants_0_14": np.round(pop_enf).astype("int64"),
        "Densite_Pop": np.round(densite, 2),
        "Superficie_km2": np.round(superficie.to_numpy(), 2),
    })

    # ── Génération des cas hebdomadaires ───────────────────────
    # incidence annuelle de base (cas pour 1000 hab) hétérogène entre aires
    inc_base = rng.lognormal(mean=np.log(180.0), sigma=0.55, size=len(gdf))

    # effet altitude : chute de transmission au-dessus de ~900-1000 m (littérature)
    if altitude_effect:
        alt_factor = 1.0 / (1.0 + np.exp((alt - 950.0) / 180.0))
        alt_factor = 0.15 + 0.85 * alt_factor
    else:
        alt_factor = np.ones_like(alt)

    # effet urbain léger (réduction) + effet NDVI (augmentation)
    urbain_factor = 1.0 - 0.006 * urbain
    ndvi_factor = 0.6 + 0.9 * ndvi

    weeks = np.arange(1, n_weeks + 1)
    # saisonnalité sahélienne : pic semaines 32-37
    saison = np.exp(-0.5 * ((weeks - 34.5) / 7.0) ** 2)
    saison = 0.18 + 1.55 * saison

    rows = []
    for yi, year in enumerate(years):
        # variabilité interannuelle (pluviométrie)
        year_factor = float(np.exp(rng.normal(0.0, 0.22)))
        # épidémie localisée sur quelques aires (année du milieu)
        outbreak_areas = set(rng.choice(len(gdf), size=max(3, len(gdf) // 12),
                                        replace=False).tolist()) if yi == 1 else set()
        for ai, area in enumerate(gdf["health_area"]):
            lam_area = (pop_tot[ai] / 1000.0) * inc_base[ai] * alt_factor[ai] \
                * urbain_factor[ai] * ndvi_factor[ai] * year_factor / 52.0
            for w in weeks:
                lam = lam_area * saison[w - 1]
                if ai in outbreak_areas and 30 <= w <= 40:
                    lam *= 3.4
                # surdispersion : loi binomiale négative
                r = 6.0
                p = r / (r + lam) if lam > 0 else 1.0
                cas = int(rng.negative_binomial(r, p)) if lam > 0 else 0
                deces = int(rng.binomial(cas, 0.0022)) if cas > 0 else 0
                # les semaines à 0 cas sont fréquemment ABSENTES des linelists réels
                if cas == 0 and rng.random() < drop_zero_rows:
                    continue
                rows.append({
                    "health_area": area,
                    "week_": int(w),
                    "cases": cas,
                    "deaths": deces,
                    "year": int(year),
                })

    df_cases = pd.DataFrame(rows, columns=["health_area", "week_", "cases", "deaths", "year"])
    df_cases["period"] = (df_cases["year"].astype(str) + "-S"
                          + df_cases["week_"].astype(str).str.zfill(2))
    df_cases = df_cases.sort_values(["health_area", "year", "week_"]).reset_index(drop=True)

    meta = {
        "iso3": iso3,
        "n_areas": int(gdf["health_area"].nunique()),
        "years": list(years),
        "n_rows": int(len(df_cases)),
        "total_cases": int(df_cases["cases"].sum()),
        "altitude_effect": altitude_effect,
        "drop_zero_rows": drop_zero_rows,
        "seed": seed,
        "altitude_min": float(df_static["Altitude_Moy"].min()),
        "altitude_max": float(df_static["Altitude_Moy"].max()),
    }
    return SyntheticDataset(df_cases=df_cases, df_population=df_population,
                            df_static=df_static, gdf=gdf, meta=meta)


if __name__ == "__main__":
    ds = build_dataset()
    print("META:", ds.meta)
    print(ds.df_cases.head())
    print(ds.df_population.head())
    print(ds.df_static.head())

"""
Extraction des **covariables statiques** (non dynamiques) par aire de santé.

Pourquoi c'est déterminant
--------------------------
Un modèle purement temporel ne peut pas expliquer pourquoi deux aires de santé
ayant la même dynamique passée n'ont pas le même risque futur. Les déterminants
structurels — **altitude**, pente, couvert végétal, présence d'eau, urbanisation,
population exposée — varient peu dans le temps mais conditionnent fortement la
transmission :

* paludisme : l'altitude est le premier déterminant de la présence d'*Anopheles*
  (transmission quasi nulle au-delà de ~1 500-2 000 m, forte réduction dès
  ~1 000 m) ; la pente conditionne la rétention des gîtes larvaires ; le NDVI
  et l'humidité proxy-fient la disponibilité des gîtes ;
* rougeole : la densité d'enfants susceptibles et l'urbanisation conditionnent
  la vitesse de propagation et l'amplitude des flambées.

Ce module fournit :
* :func:`extract_static_covariates_gee` — extraction réelle via Google Earth
  Engine (SRTM, MODIS, JRC GSW, ESA WorldCover, GHSL) ;
* :func:`static_covariates_schema` — contrat de noms de colonnes ;
* :func:`merge_static_tables` — fusion défensive de plusieurs sources.

Aucune dépendance à Streamlit : l'avancement est remonté par callback.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Contrat de noms de colonnes (stable, réutilisé par l'interface et les exports)
# ----------------------------------------------------------------------
STATIC_SCHEMA: Dict[str, Dict[str, str]] = {
    "Altitude_Moy":    {"unite": "m",     "source": "USGS/SRTMGL1_003",
                        "description": "Altitude moyenne de l'aire de santé"},
    "Altitude_Max":    {"unite": "m",     "source": "USGS/SRTMGL1_003",
                        "description": "Altitude maximale (relief)"},
    "Altitude_Min":    {"unite": "m",     "source": "USGS/SRTMGL1_003",
                        "description": "Altitude minimale"},
    "Altitude_EcartType": {"unite": "m",  "source": "USGS/SRTMGL1_003",
                           "description": "Hétérogénéité du relief"},
    "Pente_Moy":       {"unite": "°",     "source": "USGS/SRTMGL1_003 + ee.Terrain",
                        "description": "Pente moyenne : drainage des gîtes larvaires"},
    "NDVI_Moy":        {"unite": "-",     "source": "MODIS/061/MOD13A2",
                        "description": "Vigueur végétale moyenne annuelle"},
    "NDVI_Saisonnalite": {"unite": "-",   "source": "MODIS/061/MOD13A2",
                          "description": "Amplitude saisonnière du NDVI (max - min)"},
    "LST_Jour_Moy":    {"unite": "°C",    "source": "MODIS/061/MOD11A2",
                        "description": "Température de surface diurne moyenne"},
    "Eau_Pct":         {"unite": "%",     "source": "JRC/GSW1_4/GlobalSurfaceWater",
                        "description": "Part de surface en eau permanente"},
    "Eau_Saison_Pct":  {"unite": "%",     "source": "JRC/GSW1_4/GlobalSurfaceWater",
                        "description": "Part de surface en eau saisonnière"},
    "Dist_Eau_km":     {"unite": "km",    "source": "JRC/GSW1_4 (distance transform)",
                        "description": "Distance moyenne au plan d'eau le plus"},
    "Arbre_Pct":       {"unite": "%",     "source": "ESA/WorldCover/v200",
                        "description": "Couvert arboré"},
    "Culture_Pct":     {"unite": "%",     "source": "ESA/WorldCover/v200",
                        "description": "Terres cultivées (irrigation, gîtes)"},
    "Urbain_Pct":      {"unite": "%",     "source": "ESA/WorldCover/v200",
                        "description": "Zones bâties"},
    "Superficie_km2":  {"unite": "km²",   "source": "géométrie (ESRI:54009)",
                        "description": "Superficie de l'aire de santé"},
}

STATIC_COLUMNS: List[str] = list(STATIC_SCHEMA.keys())


def static_covariates_schema() -> pd.DataFrame:
    """Table descriptive des covariables statiques (documentation / export)."""
    rows = [{"variable": k, **v} for k, v in STATIC_SCHEMA.items()]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Extraction Google Earth Engine
# ----------------------------------------------------------------------
def _to_feature_collection(gdf, area_col: str = "health_area"):
    import ee
    gg = gdf.to_crs(epsg=4326)
    feats = []
    for _, row in gg.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        name = str(row[area_col]) if area_col in gg.columns else ""
        if geom.geom_type == "Polygon":
            coords = [[list(c) for c in geom.exterior.coords]]
            for interior in geom.interiors:
                coords.append([list(c) for c in interior.coords])
            ee_geom = ee.Geometry.Polygon(coords)
        elif geom.geom_type == "MultiPolygon":
            coords = []
            for poly in geom.geoms:
                rings = [[list(c) for c in poly.exterior.coords]]
                for interior in poly.interiors:
                    rings.append([list(c) for c in interior.coords])
                coords.append(rings)
            ee_geom = ee.Geometry.MultiPolygon(coords)
        else:
            continue
        feats.append(ee.Feature(ee_geom, {area_col: name}))
    return ee.FeatureCollection(feats)


def _reduce(fc, image, reducer, scale, name_map):
    """reduceRegions + extraction en dictionnaire {area: {col: val}}."""
    out = fc.map(lambda f: f.set(image.reduceRegion(
        reducer=reducer, geometry=f.geometry(), scale=scale,
        maxPixels=1e10, bestEffort=True)))
    info = out.getInfo()
    res: Dict[str, Dict[str, float]] = {}
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        area = props.get(list(name_map.values())[0]) if name_map else None
        rec = {}
        for band, col in name_map.items():
            v = props.get(band)
            rec[col] = float(v) if v is not None else np.nan
        if area is not None:
            res[str(area)] = rec
    return res


def extract_static_covariates_gee(gdf,
                                  area_col: str = "health_area",
                                  progress: Optional[Callable[[str, float], None]] = None,
                                  include: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extrait les covariables statiques par aire de santé via Google Earth Engine.

    Chaque famille de variables est extraite indépendamment : l'échec d'une
    source (quota, indisponibilité) ne bloque pas les autres, la colonne
    correspondante reste à ``NaN``.

    Paramètres
    ----------
    gdf : GeoDataFrame des aires de santé avec la colonne ``area_col``.
    progress : callback ``(message, fraction)`` pour l'affichage d'avancement.
    include : restreindre la liste des variables extraites.

    Retour
    ------
    DataFrame indexé par ``area_col`` avec les colonnes de :data:`STATIC_SCHEMA`
    réellement obtenues.
    """
    def _step(msg, frac):
        if progress is not None:
            progress(msg, frac)

    try:
        import ee
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"earthengine-api indisponible : {e}") from e

    wanted = set(include) if include else set(STATIC_COLUMNS)
    fc = _to_feature_collection(gdf, area_col=area_col)
    n = len(gdf)
    results: Dict[str, Dict[str, float]] = {str(a): {} for a in gdf[area_col].astype(str)}

    # ── 1. Altitude + pente (SRTM 30 m) ──────────────────────────────────
    if wanted & {"Altitude_Moy", "Altitude_Max", "Altitude_Min",
                 "Altitude_EcartType", "Pente_Moy"}:
        _step("Altitude / pente (SRTM)…", 0.10)
        try:
            dem = ee.Image("USGS/SRTMGL1_003")
            terr = ee.Terrain.products(dem)
            img = dem.rename("elev").addBands(terr.select("slope"))
            red = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True) \
                                     .combine(ee.Reducer.min(), sharedInputs=True) \
                                     .combine(ee.Reducer.stdDev(), sharedInputs=True)
            # reducer combiné -> bandes suffixées (_mean, _max, _min, _stdDev)
            info = fc.map(lambda f: f.set(img.reduceRegion(
                reducer=red, geometry=f.geometry(), scale=90,
                maxPixels=1e10, bestEffort=True))).getInfo()
            for feat in info.get("features", []):
                p = feat.get("properties", {})
                area = str(p.get(area_col, ""))
                if area not in results:
                    continue
                results[area].update({
                    "Altitude_Moy": p.get("elev_mean"),
                    "Altitude_Max": p.get("elev_max"),
                    "Altitude_Min": p.get("elev_min"),
                    "Altitude_EcartType": p.get("elev_stdDev"),
                    "Pente_Moy": p.get("slope_mean"),
                })
            del raw
        except Exception as e:  # noqa: BLE001
            _step(f"SRTM indisponible : {str(e)[:80]}", 0.15)

    # ── 2. NDVI (MOD13A2) ────────────────────────────────────────────────
    if wanted & {"NDVI_Moy", "NDVI_Saisonnalite"}:
        _step("NDVI (MODIS)…", 0.30)
        try:
            ndvi = (ee.ImageCollection("MODIS/061/MOD13A2")
                    .filterDate("2023-01-01", "2024-01-01")
                    .select("NDVI").multiply(0.0001))
            img = ndvi.mean().rename("ndvi_mean").addBands(
                ndvi.max().subtract(ndvi.min()).rename("ndvi_amp"))
            info = fc.map(lambda f: f.set(img.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=f.geometry(), scale=1000,
                maxPixels=1e10, bestEffort=True))).getInfo()
            for feat in info.get("features", []):
                p = feat.get("properties", {})
                area = str(p.get(area_col, ""))
                if area in results:
                    results[area].update({
                        "NDVI_Moy": p.get("ndvi_mean"),
                        "NDVI_Saisonnalite": p.get("ndvi_amp"),
                    })
        except Exception as e:  # noqa: BLE001
            _step(f"MODIS NDVI indisponible : {str(e)[:80]}", 0.35)

    # ── 3. Température de surface (MOD11A2) ──────────────────────────────
    if "LST_Jour_Moy" in wanted:
        _step("Température de surface (MODIS LST)…", 0.45)
        try:
            lst = (ee.ImageCollection("MODIS/061/MOD11A2")
                   .filterDate("2023-01-01", "2024-01-01")
                   .select("LST_Day_1km").multiply(0.02).subtract(273.15).mean()
                   .rename("lst"))
            info = fc.map(lambda f: f.set(lst.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=f.geometry(), scale=1000,
                maxPixels=1e10, bestEffort=True))).getInfo()
            for feat in info.get("features", []):
                p = feat.get("properties", {})
                area = str(p.get(area_col, ""))
                if area in results:
                    results[area]["LST_Jour_Moy"] = p.get("lst")
        except Exception as e:  # noqa: BLE001
            _step(f"MODIS LST indisponible : {str(e)[:80]}", 0.50)

    # ── 4. Eau (JRC Global Surface Water) ────────────────────────────────
    if wanted & {"Eau_Pct", "Eau_Saison_Pct", "Dist_Eau_km"}:
        _step("Surfaces en eau (JRC GSW)…", 0.60)
        try:
            gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
            occ = gsw.select("occurrence").divide(100.0).rename("occ")
            sea = gsw.select("seasonality").divide(12.0).rename("sea")
            img = occ.addBands(sea)
            info = fc.map(lambda f: f.set(img.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=f.geometry(), scale=30,
                maxPixels=1e11, bestEffort=True))).getInfo()
            for feat in info.get("features", []):
                p = feat.get("properties", {})
                area = str(p.get(area_col, ""))
                if area in results:
                    results[area].update({
                        "Eau_Pct": (p.get("occ") or 0) * 100.0 if p.get("occ") is not None else np.nan,
                        "Eau_Saison_Pct": (p.get("sea") or 0) * 100.0 if p.get("sea") is not None else np.nan,
                    })
            # distance à l'eau : transformée de distance sur l'extension max
            max_extent = gsw.select("max_extent")
            dist = (max_extent.Not()
                    .fastDistanceTransform(neighborhoodSize=256)
                    .sqrt().multiply(ee.Image.pixelArea().sqrt()).divide(1000.0)
                    .rename("dist"))
            info = fc.map(lambda f: f.set(dist.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=f.geometry(), scale=1000,
                maxPixels=1e11, bestEffort=True))).getInfo()
            for feat in info.get("features", []):
                p = feat.get("properties", {})
                area = str(p.get(area_col, ""))
                if area in results:
                    results[area]["Dist_Eau_km"] = p.get("dist")
        except Exception as e:  # noqa: BLE001
            _step(f"JRC GSW indisponible : {str(e)[:80]}", 0.65)

    # ── 5. Occupation du sol (ESA WorldCover) ────────────────────────────
    if wanted & {"Arbre_Pct", "Culture_Pct", "Urbain_Pct"}:
        _step("Occupation du sol (ESA WorldCover)…", 0.80)
        try:
            lc = ee.Image("ESA/WorldCover/v200")
            total = lc.neq(0).rename("tot")
            tree = lc.eq(10).rename("tree")
            crop = lc.eq(40).rename("crop")
            built = lc.eq(50).rename("built")
            img = total.addBands(tree).addBands(crop).addBands(built)
            # moyenne des masques binaires = proportion de la classe
            info = fc.map(lambda f: f.set(img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=f.geometry(), scale=100, maxPixels=1e11,
                bestEffort=True))).getInfo()
            for feat in info.get("features", []):
                p = feat.get("properties", {})
                area = str(p.get(area_col, ""))
                if area in results:
                    results[area].update({
                        "Arbre_Pct": (p.get("tree") or 0) * 100.0 if p.get("tree") is not None else np.nan,
                        "Culture_Pct": (p.get("crop") or 0) * 100.0 if p.get("crop") is not None else np.nan,
                        "Urbain_Pct": (p.get("built") or 0) * 100.0 if p.get("built") is not None else np.nan,
                    })
        except Exception as e:  # noqa: BLE001
            _step(f"ESA WorldCover indisponible : {str(e)[:80]}", 0.85)

    # ── 6. Superficie (toujours disponible, sans GEE) ────────────────────
    _step("Superficie…", 0.95)
    try:
        proj = gdf.to_crs("ESRI:54009")
        surf = (proj.geometry.area / 1e6).to_numpy(dtype=float)
    except Exception:
        surf = np.full(len(gdf), np.nan)
    for area, v in zip(gdf[area_col].astype(str), surf):
        results.setdefault(area, {})["Superficie_km2"] = float(v) if v == v else np.nan

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = area_col
    df = df.reset_index()
    keep = [area_col] + [c for c in STATIC_COLUMNS if c in df.columns]
    df = df[keep]
    _step("Terminé", 1.0)
    return df


# ----------------------------------------------------------------------
# Fusion de sources
# ----------------------------------------------------------------------
def merge_static_tables(*tables, area_col: str = "health_area") -> pd.DataFrame:
    """
    Fusionne plusieurs tables de covariables en priorisant les valeurs non
    nulles (la première table fournie a la priorité).
    """
    frames = []
    for t in tables:
        if t is None:
            continue
        if not isinstance(t, pd.DataFrame) or t.empty:
            continue
        tt = t.copy()
        if area_col not in tt.columns:
            continue
        tt[area_col] = tt[area_col].astype(str).str.strip().str.lower()
        tt = tt.drop_duplicates(subset=[area_col])
        frames.append(tt.set_index(area_col))
    if not frames:
        return pd.DataFrame(columns=[area_col])

    out = frames[0]
    for f in frames[1:]:
        for col in f.columns:
            if col in out.columns:
                out[col] = out[col].where(out[col].notna(), f[col])
            else:
                out[col] = f[col]
    return out.reset_index().rename(columns={"index": area_col})


def describe_coverage(static_df: pd.DataFrame, area_col: str = "health_area") -> pd.DataFrame:
    """Taux de renseignement par covariable — utile au contrôle qualité."""
    if static_df is None or static_df.empty:
        return pd.DataFrame(columns=["variable", "n_renseigne", "pct", "unite", "source"])
    rows = []
    n = len(static_df)
    for col in static_df.columns:
        if col == area_col:
            continue
        s = pd.to_numeric(static_df[col], errors="coerce")
        meta = STATIC_SCHEMA.get(col, {"unite": "", "source": "", "description": ""})
        rows.append({
            "variable": col,
            "n_renseigne": int(s.notna().sum()),
            "pct": round(100.0 * s.notna().sum() / max(n, 1), 1),
            "min": round(float(s.min()), 3) if s.notna().any() else None,
            "max": round(float(s.max()), 3) if s.notna().any() else None,
            "unite": meta.get("unite", ""),
            "source": meta.get("source", ""),
        })
    return pd.DataFrame(rows)

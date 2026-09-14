# ============================================================
# VERSION 3.0 - SOURCES CLIMATIQUES MULTIPLES
# 1. NASA POWER API (simple, fiable, gratuit)
# 2. Open-Meteo API (excellent, sans clé)
# 3. CDS API (optionnel, nécessite compte)
# ============================================================
# -*- coding: utf-8 -*-

# Imports principaux (obligatoires)
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium import Popup, Tooltip, CircleMarker, GeoJson, LayerControl, DivIcon
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from prediction_map_tab import create_prediction_map_tab
from validation_tab import create_validation_tab
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import warnings
import requests
import json
from shapely.geometry import Point
import tempfile
import zipfile
import os
import base64
from io import BytesIO
from branca.colormap import linear
import difflib
import unicodedata
import re

# ============================================================
# CONFIG STREAMLIT
# ============================================================
#st.set_page_config(
   # layout="wide", 
   # page_title="🦟 Surveillance Paludisme", 
  #  page_icon="🦟",
   # initial_sidebar_state="expanded"
#)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🦟 Surveillance et Modélisation Épidémiologique du Paludisme</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
📊 <b>Plateforme d'analyse avancée</b> intégrant données épidémiologiques, environnementales et climatiques<br>
🎯 Modélisation prédictive multi-factorielle avec Machine Learning et validation croisée temporelle
</div>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
for key in ["gdf_health", "df_cases", "temp_raster", "flood_raster", "rivers_gdf",
            "precipitation_raster", "humidity_raster", "elevation_raster", "model_results",
            "df_climate_aggregated", "enrichi_bfa", "enrichi_mli", "enrichi_ner", "enrichi_mrt", "enrichi_upload"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "pays_precedent"  not in st.session_state: st.session_state["pays_precedent"]  = None
if "sa_gdf_cache"    not in st.session_state: st.session_state["sa_gdf_cache"]    = None
if "iso3pays_cache"  not in st.session_state: st.session_state["iso3pays_cache"]  = None

if "pays_precedent" not in st.session_state: st.session_state["pays_precedent"] = None
if "sa_gdf_cache"   not in st.session_state: st.session_state["sa_gdf_cache"]   = None

# ============================================================
# FONCTIONS API CLIMATIQUES
# ============================================================

def week_to_date_range(week_num, year=2024):
    """Convertit un numéro de semaine en plage de dates"""
    week_num = int(week_num)
    year = int(year)
    jan_first = datetime(year, 1, 1)
    week_start = jan_first + timedelta(weeks=week_num - 1)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end

# ============================================================
# NASA POWER API - SIMPLE ET FIABLE
# ============================================================

@st.cache_data(ttl=86400)
def fetch_climate_nasa_power(lat, lon, start_date, end_date):
    """
    Récupère données climatiques depuis NASA POWER API
    Variables: température (T2M), précipitations (PRECTOTCORR), humidité (RH2M)
    """
    try:
        # Format dates YYYYMMDD
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        # URL API NASA POWER
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        params = {
            "parameters": "T2M,PRECTOTCORR,RH2M",  # Temp, Précip, Humidité
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_str,
            "end": end_str,
            "format": "JSON"
        }
        
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            if "properties" in data and "parameter" in data["properties"]:
                params_data = data["properties"]["parameter"]
                
                # Convertir en DataFrame
                dates = list(params_data.get("T2M", {}).keys())
                
                df = pd.DataFrame({
                    'date': pd.to_datetime(dates, format='%Y%m%d'),
                    'temp': [params_data.get("T2M", {}).get(d, np.nan) for d in dates],
                    'precip': [params_data.get("PRECTOTCORR", {}).get(d, np.nan) for d in dates],
                    'humidity': [params_data.get("RH2M", {}).get(d, np.nan) for d in dates]
                })
                
                return df
        
        return None
        
    except Exception as e:
        st.warning(f"⚠️ NASA POWER API erreur: {str(e)}")
        return None

# ============================================================
# OPEN-METEO API - EXCELLENT ET GRATUIT
# ============================================================

@st.cache_data(ttl=86400)
def fetch_climate_open_meteo(lat, lon, start_date, end_date):
    """
    Récupère données climatiques depuis Open-Meteo Archive API
    Variables: température, précipitations, humidité
    """
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            if "daily" in data:
                daily = data["daily"]
                
                df = pd.DataFrame({
                    'date': pd.to_datetime(daily["time"]),
                    'temp': daily.get("temperature_2m_mean", []),
                    'precip': daily.get("precipitation_sum", []),
                    'humidity': daily.get("relative_humidity_2m_mean", [])
                })
                
                return df
        
        return None
        
    except Exception as e:
        st.warning(f"⚠️ Open-Meteo API erreur: {str(e)}")
        return None
# -------------------------
# Initialisation Google Earth Engine (Streamlit Cloud)
# -------------------------
@st.cache_resource
def init_gee():
    """Initialise Google Earth Engine"""
    try:
        import ee
        
        # Essayer avec service account
        try:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"],
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
            st.sidebar.success("✅ GEE initialisé (Service Account)")
            return True
        except Exception as e:
            st.sidebar.warning(f"⚠️ Service Account échec : {str(e)[:100]}")
        
        # Essayer authentification par défaut
        try:
            ee.Initialize()
            st.sidebar.success("✅ GEE initialisé (Défaut)")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ GEE échec total : {str(e)[:100]}")
            return False
    
    except ImportError:
        st.sidebar.error("❌ Package 'earthengine-api' non installé")
        return False

gee_ok = init_gee()
use_gee = gee_ok  # ✅ Utiliser le résultat de init_gee()

# -------------------------
# Fonction WorldPop UNIQUE
# -------------------------

def worldpop_malaria_stats(_sa_gdf, use_gee):
    """Calqué sur worldpop_children_stats (rougeole)."""
    if not use_gee:
        st.sidebar.warning("⚠️ WorldPop / GEE indisponible")
        return pd.DataFrame({
            "health_area":      _sa_gdf["health_area"].tolist(),
            "Pop_Totale":       [np.nan] * len(_sa_gdf),
            "Pop_Enfants_0_14": [np.nan] * len(_sa_gdf),
            "Densite_Pop":      [np.nan] * len(_sa_gdf),
        })
    try:
        import ee
        progress_bar = st.sidebar.progress(0)
        status_text  = st.sidebar.empty()
        status_text.text("Chargement WorldPop...")

        dataset      = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
        pop_img      = dataset.mosaic()

        male_bands_all   = ["M_0", "M_1", "M_5", "M_10", "M_15", "M_20", "M_25", "M_30"]
        female_bands_all = ["F_0", "F_1", "F_5", "F_10", "F_15", "F_20", "F_25", "F_30"]

        # Bandes UNIQUEMENT pour 0–14 ans (M_0=0-1an, M_1=1-4ans, M_5=5-9ans, M_10=10-14ans)
        male_bands_0_14   = ["M_0", "M_1", "M_5", "M_10"]
        female_bands_0_14 = ["F_0", "F_1", "F_5", "F_10"]

        selected_males_all   = pop_img.select(male_bands_all)
        selected_females_all = pop_img.select(female_bands_all)
        selected_males_0_14   = pop_img.select(male_bands_0_14)
        selected_females_0_14 = pop_img.select(female_bands_0_14)
        total_pop             = pop_img.select("population")

        # Calcul correct enfants 0–14 ans uniquement
        males_0_14_sum   = selected_males_0_14.reduce(ee.Reducer.sum()).rename("garcons_0_14")
        females_0_14_sum = selected_females_0_14.reduce(ee.Reducer.sum()).rename("filles_0_14")
        enfants_0_14     = males_0_14_sum.add(females_0_14_sum).rename("enfants_0_14")

        final_mosaic = (total_pop
                        .addBands(selected_males_all)
                        .addBands(selected_females_all)
                        .addBands(enfants_0_14))

        pixel_area         = ee.Image.pixelArea().divide(10000)
        final_mosaic_count = final_mosaic.multiply(pixel_area)

        # Conversion géométries — pattern rougeole (Polygon/MultiPolygon explicite)
        status_text.text("Conversion géométries...")
        features = []
        for _, row in _sa_gdf.iterrows():
            geom  = row.geometry
            props = {"health_area": row["health_area"]}
            if geom.geom_type == "Polygon":
                coords  = [[x, y] for x, y in geom.exterior.coords]
                ee_geom = ee.Geometry.Polygon(coords)
            elif geom.geom_type == "MultiPolygon":
                coords  = [[[x, y] for x, y in poly.exterior.coords] for poly in geom.geoms]
                ee_geom = ee.Geometry.MultiPolygon(coords)
            else:
                continue
            features.append(ee.Feature(ee_geom, props))

        fc = ee.FeatureCollection(features)

        status_text.text("Calcul statistiques zonales...")
        stats      = final_mosaic_count.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.sum(),
            scale=100,
            crs="EPSG:4326"
        )
        status_text.text("Extraction résultats...")
        stats_info  = stats.getInfo()
        data_list   = []
        total_aires = len(stats_info["features"])

        for i, feat in enumerate(stats_info["features"]):
            props     = feat["properties"]
            pop_tot   = props.get("population", 0)
            enfants_t = props.get("enfants_0_14",    0)
            data_list.append({
                "health_area":      props.get("health_area", ""),
                "Pop_Totale":       int(pop_tot)   if pop_tot   else np.nan,
                "Pop_Enfants_0_14": int(enfants_t) if enfants_t else np.nan,
                # ── Hommes ──────────────────────────────────────────────
                "Pop_MALE_0_4":     int(props.get("M_0", 0) + props.get("M_1", 0)),
                "Pop_MALE_5_9":     int(props.get("M_5",  0)),
                "Pop_MALE_10_14":   int(props.get("M_10", 0)),
                "Pop_MALE_15_19":   int(props.get("M_15", 0)),   # ← AJOUTÉ
                "Pop_MALE_20_24":   int(props.get("M_20", 0)),   # ← AJOUTÉ
                "Pop_MALE_25_29":   int(props.get("M_25", 0)),   # ← AJOUTÉ
                "Pop_MALE_30_34":   int(props.get("M_30", 0)),   # ← AJOUTÉ
                # ── Femmes ──────────────────────────────────────────────
                "Pop_FEMALE_0_4":   int(props.get("F_0", 0) + props.get("F_1", 0)),
                "Pop_FEMALE_5_9":   int(props.get("F_5",  0)),
                "Pop_FEMALE_10_14": int(props.get("F_10", 0)),
                "Pop_FEMALE_15_19": int(props.get("F_15", 0)),   # ← AJOUTÉ
                "Pop_FEMALE_20_24": int(props.get("F_20", 0)),   # ← AJOUTÉ
                "Pop_FEMALE_25_29": int(props.get("F_25", 0)),   # ← AJOUTÉ
                "Pop_FEMALE_30_34": int(props.get("F_30", 0)),   # ← AJOUTÉ
                "Densite_Pop":      np.nan,
            })

            progress_bar.progress(min((i + 1) / total_aires, 1.0))

        progress_bar.empty()
        status_text.text("WorldPop terminé ✅")

        if not data_list:
            raise ValueError("GEE a retourné 0 features")

        return pd.DataFrame(data_list)

    except Exception as e:
        st.sidebar.error(f"❌ WorldPop : {e}")
        try: progress_bar.empty()
        except: pass
        try: status_text.empty()
        except: pass
        return pd.DataFrame({
            "health_area":      _sa_gdf["health_area"].tolist(),
            "Pop_Totale":       [np.nan] * len(_sa_gdf),
            "Pop_Enfants_0_14": [np.nan] * len(_sa_gdf),
            "Densite_Pop":      [np.nan] * len(_sa_gdf),
        })

# === DEBUG NOM D'AIRES ===
if st.session_state.gdf_health is not None and st.session_state.df_cases is not None:
    gdf_health = st.session_state.gdf_health
    df_cases   = st.session_state.df_cases

    geo_names = set(gdf_health["health_area"])
    csv_names = set(df_cases["health_area"])

    missing_in_geo = sorted(csv_names - geo_names)

    if missing_in_geo:
        with st.sidebar.expander("⚠️ Aires du CSV sans géométrie", expanded=False):
            st.write(missing_in_geo)
# Propositions de correspondance fuzzy
    import difflib

    suggested_mapping = {}
    for name in missing_in_geo:
        matches = difflib.get_close_matches(
            name,
            list(geo_names),
            n=1,
            cutoff=0.7  # à ajuster si besoin
        )
        if matches:
            suggested_mapping[name] = matches[0]

    if suggested_mapping:
        with st.sidebar.expander("🧠 Propositions de corrections de noms", expanded=False):
            for old, new in suggested_mapping.items():
                st.write(f"{old!r} → {new!r}")

# ============================================================
# AGRÉGATION PAR AIRE ET SEMAINE
# ============================================================

def aggregate_climate_by_week_and_area(gdf_health, df_cases, year, api_choice="NASA POWER"):
    """
    Télécharge et agrège données climatiques par aire de santé et semaine
    """
    records = []
    
    # Déterminer les semaines uniques
    weeks = sorted([int(w) for w in df_cases['week_'].unique()])
    
    st.info(f"📅 Traitement de {len(weeks)} semaines pour {len(gdf_health)} aires de santé")
    
    # Progress bar
    progress_bar = st.progress(0)
    total_ops = len(gdf_health) * len(weeks)
    processed = 0
    
    # Cache pour éviter requêtes multiples pour même point
    cache_data = {}
    
    for idx, row in gdf_health.iterrows():
        area = row['health_area']
        lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
        
        # Clé cache basée sur coordonnées arrondies
        cache_key = f"{round(lat, 2)}_{round(lon, 2)}"
        
        # Télécharger données pour ce point si pas en cache
        if cache_key not in cache_data:
            # Déterminer plage complète de dates
            first_week = min(weeks)
            last_week = max(weeks)
            
            start_date, _ = week_to_date_range(first_week, year)
            _, end_date = week_to_date_range(last_week, year)
            
            # Télécharger selon API choisie
            if api_choice == "NASA POWER":
                df_climate_point = fetch_climate_nasa_power(lat, lon, start_date, end_date)
            elif api_choice == "Open-Meteo":
                df_climate_point = fetch_climate_open_meteo(lat, lon, start_date, end_date)
            else:
                df_climate_point = None
            
            if df_climate_point is not None and not df_climate_point.empty:
                cache_data[cache_key] = df_climate_point
            else:
                cache_data[cache_key] = None
        
        df_point = cache_data.get(cache_key)
        
        if df_point is not None:
            # Pour chaque semaine de cette aire
            for week in weeks:
                try:
                    week_start, week_end = week_to_date_range(week, year)
                    
                    # Filtrer données de la semaine
                    mask = (df_point['date'] >= week_start) & (df_point['date'] <= week_end)
                    df_week = df_point[mask]
                    
                    if not df_week.empty:
                        record = {
                            'health_area': area,
                            'week_': week,
                            'nb_days': len(df_week),
                            "year": year
                        }
                        
                        # Température (moyenne hebdomadaire)
                        if 'temp' in df_week.columns:
                            temp_values = df_week['temp'].dropna()
                            if len(temp_values) > 0:
                                record['temp_api'] = round(float(temp_values.mean()), 2)
                                record['temp_api_min'] = round(float(temp_values.min()), 2)
                                record['temp_api_max'] = round(float(temp_values.max()), 2)
                        
                        # Précipitations (somme hebdomadaire)
                        if 'precip' in df_week.columns:
                            precip_values = df_week['precip'].dropna()
                            if len(precip_values) > 0:
                                record['precip_api'] = round(float(precip_values.sum()), 2)
                                record['precip_api_max'] = round(float(precip_values.max()), 2)
                        
                        # Humidité (moyenne hebdomadaire)
                        if 'humidity' in df_week.columns:
                            humidity_values = df_week['humidity'].dropna()
                            if len(humidity_values) > 0:
                                record['humidity_api'] = round(float(humidity_values.mean()), 2)
                                record['humidity_api_min'] = round(float(humidity_values.min()), 2)
                                record['humidity_api_max'] = round(float(humidity_values.max()), 2)
                        
                        if len(record) > 3:  # Au moins une variable climatique
                            records.append(record)
                
                except Exception as e:
                    continue
                
                processed += 1
                progress_bar.progress(processed / total_ops)
    
    progress_bar.empty()
    
    df_result = pd.DataFrame(records)
    
    if not df_result.empty:
        st.success(f"✅ {len(df_result)} enregistrements climatiques extraits")
        
        # Statistiques
        st.markdown("### 📊 Statistiques Climatiques")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'temp_api' in df_result.columns:
                st.markdown("#### 🌡️ Température (°C)")
                st.write(df_result['temp_api'].describe())
        
        with col2:
            if 'precip_api' in df_result.columns:
                st.markdown("#### 🌧️ Précipitations (mm)")
                st.write(df_result['precip_api'].describe())
        
        with col3:
            if 'humidity_api' in df_result.columns:
                st.markdown("#### 💧 Humidité (%)")
                st.write(df_result['humidity_api'].describe())
        
        # Visualisations
        st.markdown("### 📈 Visualisations")
        
        fig = go.Figure()
        
        df_week_avg = df_result.groupby('week_').agg({
            col: 'mean' for col in df_result.columns 
            if col.endswith('_api') and not any(x in col for x in ['_min', '_max'])
        }).reset_index()
        
        if 'temp_api' in df_week_avg.columns:
            fig.add_trace(go.Scatter(
                x=df_week_avg['week_'],
                y=df_week_avg['temp_api'],
                mode='lines+markers',
                name='Température (°C)',
                yaxis='y'
            ))
        
        if 'precip_api' in df_week_avg.columns:
            fig.add_trace(go.Scatter(
                x=df_week_avg['week_'],
                y=df_week_avg['precip_api'],
                mode='lines+markers',
                name='Précipitations (mm)',
                yaxis='y2'
            ))
        
        if 'humidity_api' in df_week_avg.columns:
            fig.add_trace(go.Scatter(
                x=df_week_avg['week_'],
                y=df_week_avg['humidity_api'],
                mode='lines+markers',
                name='Humidité (%)',
                yaxis='y3'
            ))
        
        fig.update_layout(
            title="Évolution Climatique Hebdomadaire",
            xaxis_title="Semaine",
            yaxis=dict(title="Température (°C)"),
            yaxis2=dict(title="Précipitations (mm)", overlaying='y', side='right'),
            yaxis3=dict(title="Humidité (%)", overlaying='y', side='right', anchor='free', position=0.95),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Couverture
        coverage_areas = df_result['health_area'].nunique()
        coverage_weeks = df_result['week_'].nunique()
        
        st.info(f"📍 Couverture: {coverage_areas}/{len(gdf_health)} aires ({coverage_areas/len(gdf_health)*100:.1f}%)")
        st.info(f"📅 Couverture: {coverage_weeks}/{len(weeks)} semaines ({coverage_weeks/len(weeks)*100:.1f}%)")
    else:
        st.error(" Aucune donnée climatique extraite")
    
    return df_result

# ============================================================
# FONCTIONS UTILITAIRES (INCHANGÉES)
# ============================================================

def safe_int(value):
    if pd.isna(value) or value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def safe_float(value, default=0.0):
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def ensure_wgs84(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def extract_raster_statistics(gdf, raster, stat='mean'):
    stats = []
    for geom in gdf.geometry:
        try:
            out_img, _ = mask(raster, [geom], crop=True)
            data = out_img[0].astype(float)
            nodata = raster.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            
            if stat == 'mean':
                value = np.nanmean(data)
            elif stat == 'max':
                value = np.nanmax(data)
            elif stat == 'min':
                value = np.nanmin(data)
            elif stat == 'std':
                value = np.nanstd(data)
            else:
                value = np.nanmean(data)
            
            if np.isinf(value) or np.isnan(value):
                value = np.nan
            
            stats.append(value)
        except Exception:
            stats.append(np.nan)
    return stats

def distance_to_nearest_line(point, lines_gdf):
    if lines_gdf.empty:
        return np.nan
    return lines_gdf.geometry.apply(lambda x: point.distance(x)).min() * 111

def create_advanced_features(df):
    df = df.sort_values(['health_area', 'week_num'])
    
    for lag in [1, 2, 4]:
        df[f'cases_lag_{lag}'] = df.groupby('health_area')['cases'].shift(lag)
    
    for window in [2, 4]:
        df[f'cases_ma_{window}'] = df.groupby('health_area')['cases'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    df['growth_rate'] = df.groupby('health_area')['cases'].pct_change().fillna(0)
    
    df['week_of_year'] = df['week_num'] % 52
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    return df

def create_environmental_features(gdf_map):
    if 'flood_mean' in gdf_map.columns and 'dist_river' in gdf_map.columns:
        gdf_map['flood_risk'] = gdf_map['flood_mean'] / (gdf_map['dist_river'] + 0.1)
    
    if 'temp_mean' in gdf_map.columns and 'humidity_mean' in gdf_map.columns:
        gdf_map['climate_index'] = (
            np.exp(-((gdf_map['temp_mean'] - 27.5)**2) / 50) * 
            (gdf_map['humidity_mean'] / 100)
        )
    
    if 'temp_mean' in gdf_map.columns and 'precipitation_mean' in gdf_map.columns:
        gdf_map['temp_precip_interaction'] = (
            gdf_map['temp_mean'] * gdf_map['precipitation_mean']
        )
    
    return gdf_map
def create_population_features(df):
    """
    Crée des features dérivées de population pour la modélisation
    """
    df = df.copy()

    # Taux d'incidence (cas pour 10 000 hab)
    if "Pop_Totale" in df.columns:
        df["incidence_rate"] = (df["cases"] / df["Pop_Totale"] * 10000)
        df["incidence_rate"] = df["incidence_rate"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Risque enfants (cas pour 1 000 enfants 0-14 ans)
    if "Pop_Enfants_0_14" in df.columns:
        df["child_risk"] = (df["cases"] / df["Pop_Enfants_0_14"] * 1000)
        df["child_risk"] = df["child_risk"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Pression démographique (densité × incidence)
    if "Densite_Pop" in df.columns and "incidence_rate" in df.columns:
        df["demo_pressure"] = df["Densite_Pop"] * df["incidence_rate"]
        df["demo_pressure"] = df["demo_pressure"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

def generate_alerts(df_future, threshold_percentile=75):
    if df_future.empty:
        return pd.DataFrame()
    
    threshold = df_future['predicted_cases'].quantile(threshold_percentile / 100)
    alerts = df_future[df_future['predicted_cases'] > threshold]
    
    return alerts.sort_values('predicted_cases', ascending=False)

def validate_numeric_features(df, feature_cols):
    non_numeric = []
    for col in feature_cols:
        if col not in df.columns:
            st.warning(f"⚠️ Colonne manquante : {col}")
        elif df[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
            non_numeric.append((col, df[col].dtype))
    
    if non_numeric:
        st.error(f" Colonnes non-numériques détectées : {non_numeric}")
        return False
    return True

def normalize_week_format(week_series):
    unique_weeks = week_series.unique()
    week_mapping = {}
    
    for i, week in enumerate(sorted(unique_weeks), start=1):
        week_str = str(week)
        if 'W' in week_str or 'w' in week_str:
            num = ''.join(filter(str.isdigit, week_str.split('-')[-1]))
        elif 'S' in week_str or 's' in week_str:
            num = ''.join(filter(str.isdigit, week_str))
        else:
            num = ''.join(filter(str.isdigit, week_str.split('-')[-1]))
        
        week_mapping[week] = int(num) if num else i
    
    return week_series.map(week_mapping)
def detect_and_rename_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rend l'import CSV robuste aux variations de noms de colonnes
    en mappant vers : health_area, week_, cases, deaths, year.
    """
    cols = list(df.columns)

    def find_col(candidates):
        # 1) match exact
        for c in candidates:
            if c in cols:
                return c
        # 2) match "contient"
        for col in cols:
            for c in candidates:
                if c in col:
                    return col
        return None

    health_candidates = [
        "health_area", "aire_de_sante", "aire_sante", "aire",
        "aire_de_santé", "aire_sanitaire", "zone_sanitaire", "district"
    ]
    week_candidates = [
        "week_", "week", "semaine", "semaine_epi", "epiweek",
        "epi_week", "semaine_epidemio", "semaine_epid", "semaine_epidemiologique"
    ]
    cases_candidates = [
        "cases", "nb_cas", "nombre_de_cas", "n_cas",
        "cas", "cas_palu", "cases_palu", "cas_paludisme"
    ]
    deaths_candidates = [
        "deaths", "death", "deces", "décès", "nb_deces",
        "nombre_de_deces", "nb_deces_palu", "deaths_palu"
    ]
    year_candidates = [
        "year", "annee", "année", "annee_epi", "annee_epid",
        "year_epi", "epi_year"
    ]

    col_health = find_col(health_candidates)
    col_week   = find_col(week_candidates)
    col_cases  = find_col(cases_candidates)
    col_deaths = find_col(deaths_candidates)
    col_year   = find_col(year_candidates)

    mapping = {}
    if col_health is not None and col_health != "health_area":
        mapping[col_health] = "health_area"
    if col_week is not None and col_week != "week_":
        mapping[col_week] = "week_"
    if col_cases is not None and col_cases != "cases":
        mapping[col_cases] = "cases"
    if col_deaths is not None and col_deaths != "deaths":
        mapping[col_deaths] = "deaths"
    if col_year is not None and col_year != "year":
        mapping[col_year] = "year"

    if mapping:
        df = df.rename(columns=mapping)

    return df

def normalize_health_name_for_match(name: str) -> str:
    """Normalisation forte des noms d'aires pour fuzzy-match."""
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    # enlever accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # remplacer ponctuation par espace
    s = re.sub(r"[\\-_/']", " ", s)
    # compresser les espaces
    s = re.sub(r"\\s+", " ", s)
    return s


def build_auto_health_area_mapping(gdf_health: gpd.GeoDataFrame,
                                   df_cases: pd.DataFrame,
                                   cutoff: float = 0.90) -> dict:
    """
    Mapping automatique très conservateur {nom_csv -> nom_geo}.

    - On ne matche que quand le score est élevé (cutoff ≥ 0.90).
    - On exige un préfixe commun (au moins 3 lettres identiques au début).
    - On exige une longueur proche (|len(a) - len(b)| ≤ 2).
    - Si plusieurs géométries sont en concurrence, on ne décide pas.
    """
    if gdf_health is None or df_cases is None:
        return {}

    geo_names = set(gdf_health["health_area"])
    csv_names = set(df_cases["health_area"])

    missing = sorted(csv_names - geo_names)
    if not missing:
        return {}

    # Préparer une version normalisée des noms géo pour la recherche
    geo_norm_to_real = {}
    for n in geo_names:
        norm = normalize_health_name_for_match(n)
        geo_norm_to_real.setdefault(norm, set()).add(n)
    geo_norm_list = list(geo_norm_to_real.keys())

    mapping = {}

    for name in missing:
        norm = normalize_health_name_for_match(name)
        if not norm:
            continue

        # Candidats sur les formes normalisées
        candidates = difflib.get_close_matches(norm, geo_norm_list, n=5, cutoff=cutoff)
        if not candidates:
            continue

        # Recalculer les scores + vérifier contraintes de préfixe / longueur
        best_real = None
        best_score = 0.0
        second_score = 0.0

        for cand_norm in candidates:
            for real_name in geo_norm_to_real[cand_norm]:
                # Score calculé sur les chaînes originales (pas seulement normalisées)
                s1 = name
                s2 = real_name
                score = difflib.SequenceMatcher(None, s1, s2).ratio()

                # Filtre préfixe : au moins 3 lettres identiques au début
                prefix_len = min(len(s1), len(s2), 4)
                same_prefix = (s1[:prefix_len] == s2[:prefix_len])

                # Filtre longueur : différence max de 2 caractères
                length_close = abs(len(s1) - len(s2)) <= 2

                if not (score >= cutoff and same_prefix and length_close):
                    continue

                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_real = real_name
                elif score > second_score:
                    second_score = score

        # Si rien ne passe les filtres stricts → pas de mapping auto
        if best_real is None:
            continue

        # On exige aussi que le meilleur soit nettement devant les autres
        if (best_score - second_score) < 0.05:
            continue

        mapping[name] = best_real

    return mapping

def add_raster_to_map(m, raster, name):
    bounds = raster.bounds
    
    data = raster.read(1).astype(float)
    if raster.nodata is not None:
        data[data == raster.nodata] = np.nan
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    if "Inondation" in name or "flood" in name.lower():
        cmap = linear.Blues_09.scale(vmin, vmax)
    elif "Température" in name or "temp" in name.lower():
        cmap = linear.YlOrRd_09.scale(vmin, vmax)
    elif "Précipitation" in name or "precip" in name.lower():
        cmap = linear.BuPu_09.scale(vmin, vmax)
    elif "Humidité" in name or "humid" in name.lower():
        cmap = linear.GnBu_09.scale(vmin, vmax)
    else:
        cmap = linear.Viridis_09.scale(vmin, vmax)
    
    for i in range(h):
        for j in range(w):
            if np.isnan(data[i, j]):
                rgba[i, j] = [0,0,0,0]
            else:
                hex_color = cmap(data[i,j]).lstrip("#")
                r = int(hex_color[0:2],16)
                g = int(hex_color[2:4],16)
                b = int(hex_color[4:6],16)
                rgba[i,j] = [r,g,b,180]
    
    img = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    img_url = f"data:image/png;base64,{encoded}"
    
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[bounds.bottom, bounds.left],[bounds.top,bounds.right]],
        name=name,
        opacity=0.7,
        interactive=True,
        zindex=1
    ).add_to(m)
    
    colormap = cmap
    colormap.caption = name
    colormap.add_to(m)
# ============================================================
# FONCTIONS AVANCÉES POUR MODÉLISATION
# ============================================================

def create_spatial_clusters(gdf, n_clusters=5):
    """
    Clustering spatial des zones pour capturer hétérogénéité géographique
    """
    # Extraire centroides
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(centroids)
    
    return clusters, kmeans

def calculate_spatial_lag(gdf, values, k_neighbors=5):
    """Calcule le lag spatial — influence des zones voisines"""
    from scipy.spatial.distance import cdist

    # Réinitialiser l'index pour garantir un alignement positionnel
    gdf    = gdf.reset_index(drop=True)
    values = values.reset_index(drop=True)

    centroids   = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
    dist_matrix = cdist(centroids, centroids, metric='euclidean')

    n = len(gdf)
    k = min(k_neighbors, n - 1)   # ← sécurité si peu d'aires

    spatial_lag = []
    for i in range(n):
        neighbors_idx = np.argsort(dist_matrix[i])[1:k+1]
        # Vérification que les indices sont dans les bornes
        neighbors_idx = neighbors_idx[neighbors_idx < n]
        if len(neighbors_idx) == 0:
            spatial_lag.append(0.0)
            continue
        weights = 1 / (dist_matrix[i, neighbors_idx] + 1e-6)
        weights = weights / weights.sum()
        lag_value = np.sum(values.iloc[neighbors_idx].values * weights)  # ← .values évite tout pb d'index
        spatial_lag.append(lag_value)

    return np.array(spatial_lag)

def perform_pca_analysis(df, feature_cols, explained_variance_threshold=0.95):
    """
    Version robuste avec gestion d'erreur complète
    """
    try:
        # Validation des entrées
        if not isinstance(explained_variance_threshold, (int, float)):
            raise ValueError(f"explained_variance_threshold doit être un nombre, reçu: {type(explained_variance_threshold)}")
        
        if not 0 < explained_variance_threshold <= 1:
            raise ValueError(f"explained_variance_threshold doit être entre 0 et 1, reçu: {explained_variance_threshold}")
        
        if len(feature_cols) < 2:
            raise ValueError(f"Au moins 2 features nécessaires pour ACP, reçu: {len(feature_cols)}")
        
        # Préparer données
        X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
        
        # Vérifier s'il y a des données
        if X.isnull().all().all():
            raise ValueError("Toutes les valeurs sont NaN après nettoyage")
        
        # Imputation
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # ACP complète pour analyse
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Calculer variance cumulée
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Trouver nombre de composantes
        n_components_idx = np.argmax(cumsum_variance >= explained_variance_threshold)
        n_components = int(n_components_idx + 1)
        
        # Contraintes de sécurité
        n_components = max(1, min(n_components, len(feature_cols), X_scaled.shape[0] - 1))
        
        # ACP finale
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # DataFrame résultat
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        # Informations
        pca_info = {
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'n_components': n_components,
            'feature_names': feature_cols,
            'total_variance_explained': float(cumsum_variance[n_components - 1])
        }
        
        return df_pca, pca, scaler, imputer, pca_info
    
    except Exception as e:
        # En cas d'erreur, retourner les données originales sans ACP
        import warnings
        warnings.warn(f"Erreur lors de l'ACP: {str(e)}. Retour données originales.")
        
        # Retour fallback
        X_fallback = df[feature_cols].copy().fillna(0)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_fallback)
        
        pca_info = {
            'explained_variance': np.array([1.0]),
            'cumulative_variance': np.array([1.0]),
            'components': np.eye(len(feature_cols)),
            'n_components': len(feature_cols),
            'feature_names': feature_cols,
            'error': str(e)
        }
        
        return pd.DataFrame(X_imputed, columns=feature_cols, index=df.index), None, None, imputer, pca_info
# ============================================================
# FONCTIONS CHARGEMENT GÉOGRAPHIQUE (pattern rougeole)
# ============================================================

@st.cache_data
def load_health_areas_from_zip(zip_path, iso3filter):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files:
                raise ValueError("Aucun fichier .shp trouvé dans le ZIP")
            gdf_full = gpd.read_file(os.path.join(tmpdir, shp_files[0]))

        iso3_col = None
        for col in ["iso3", "ISO3", "isocode", "ISOCODE"]:
            if col in gdf_full.columns:
                iso3_col = col
                break
        if iso3_col is None:
            st.warning(f"⚠️ Colonne ISO3 non trouvée. Colonnes : {list(gdf_full.columns)}")
            return gpd.GeoDataFrame()

        gdf = gdf_full[gdf_full[iso3_col].astype(str).str.strip().str.lower() == iso3filter].copy()
        if gdf.empty:
            st.warning(f"⚠️ Aucune aire pour '{iso3filter}'. Valeurs dispo : {gdf_full[iso3_col].unique().tolist()}")
            return gpd.GeoDataFrame()

        name_col = None
        for col in ["health_area", "health_are", "name_fr", "name", "NAME", "nom", "NOM"]:
            if col in gdf.columns:
                name_col = col
                break
        gdf["health_area"] = gdf[name_col].astype(str).str.strip().str.lower() if name_col else [f"aire_{i+1}" for i in range(len(gdf))]

        gdf = gdf[gdf.geometry.is_valid]
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf

    except Exception as e:
        st.error(f"❌ Erreur ZIP : {e}")
        return gpd.GeoDataFrame()


def load_shapefile_from_upload(upload_file):
    try:
        if upload_file.name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(upload_file.getvalue())
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(tmpdir)
                shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
                if shp_files:
                    gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                else:
                    raise ValueError("Aucun .shp trouvé")
        else:
            gdf = gpd.read_file(upload_file)

        if "health_area" not in gdf.columns:
            for col in ["health_area", "health_are", "name_fr", "name", "NAME", "nom", "NOM"]:
                if col in gdf.columns:
                    gdf["health_area"] = gdf[col].astype(str).str.strip().str.lower()
                    break
            else:
                gdf["health_area"] = [f"aire_{i}" for i in range(len(gdf))]
        else:
            gdf["health_area"] = gdf["health_area"].astype(str).str.strip().str.lower()

        gdf = gdf[gdf.geometry.is_valid]
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf

    except Exception as e:
        st.error(f"❌ Erreur lecture : {e}")
        return gpd.GeoDataFrame()

# ============================================================
# SIDEBAR – CHARGEMENT DES DONNÉES
# ============================================================

st.sidebar.header("📁 Chargement des Données")
PAYS_ISO3_MAP = {
    "Burkina Faso": "bfa",
    "Mali":         "mli",
    "Niger":        "ner",
    "Mauritanie":   "mrt"
}

with st.sidebar.expander("📍 Données Obligatoires", expanded=True):

    source_geo = st.radio(
        "Source des Aires de Santé",
        ["Fichier local (ao_hlthArea.zip)", "Charger un fichier (GeoJSON/SHP/ZIP)"],
        key="source_geo_palu"
    )

    iso3pays    = None
    upload_file = None

    # ── Sélection pays — toujours visible (pattern rougeole) ──
    if source_geo == "Fichier local (ao_hlthArea.zip)":
        pays_selectionne = st.selectbox(
            "🌍 Sélectionner le pays",
            list(PAYS_ISO3_MAP.keys()),
            key="pays_select"
        )
        iso3pays = PAYS_ISO3_MAP[pays_selectionne]
        st.session_state["iso3pays_courant"] = iso3pays
        if st.session_state["pays_precedent"] != pays_selectionne:
            st.session_state["pays_precedent"]  = pays_selectionne
            st.session_state["sa_gdf_cache"]    = None
            st.session_state["iso3pays_cache"]  = None   # ← ajout
            for k in [f"enrichi_{v}" for v in PAYS_ISO3_MAP.values()]:
                st.session_state[k] = None
    else:
        upload_file = st.file_uploader(
            "Aires de santé (GeoJSON/SHP/ZIP)",
            type=["geojson", "shp", "zip"],
            key="health_upload"
        )

    # ── Chargement avec cache 
    _cache_valide = (
        st.session_state["sa_gdf_cache"] is not None
        and source_geo == "Fichier local (ao_hlthArea.zip)"
        and st.session_state.get("iso3pays_cache") == iso3pays   # ← vérifie que le cache = bon pays
    )
    if _cache_valide:
        gdf_health = st.session_state["sa_gdf_cache"]
        st.success(f"✅ {len(gdf_health)} aires chargées (cache - {iso3pays})")
    else:
        if source_geo == "Fichier local (ao_hlthArea.zip)":
            zip_path = os.path.join("data", "ao_hlthArea.zip")
            if not os.path.exists(zip_path):
                st.error(f"❌ Fichier non trouvé : {zip_path}")
                st.info("📁 Placez ao_hlthArea.zip dans le dossier data/")
                st.stop()
            gdf_health = load_health_areas_from_zip(zip_path, iso3pays)
            if gdf_health.empty:
                st.error(f"❌ Impossible de charger {pays_selectionne} ({iso3pays})")
                st.stop()
            st.sidebar.success(f"✅ {len(gdf_health)} aires chargées ({iso3pays})")
            st.session_state["sa_gdf_cache"] = gdf_health
            st.session_state["sa_gdf_cache"] = gdf_health
            st.session_state["iso3pays_cache"] = iso3pays 
        else:
            if upload_file is None:
                st.warning("⚠️ Veuillez uploader un fichier")
                st.stop()
            gdf_health = load_shapefile_from_upload(upload_file)
            if gdf_health.empty:
                st.error("❌ Fichier invalide")
                st.stop()
            st.sidebar.success(f"✅ {len(gdf_health)} aires chargées")
            st.session_state["sa_gdf_cache"] = gdf_health

    # Synchronisation session_state.gdf_health
    st.session_state.gdf_health = gdf_health

    # ── Chargement CSV cas hebdomadaires (TOUJOURS VISIBLE) ────
    st.markdown("---")
    cases_file = st.file_uploader(
        "📊 Cas hebdomadaires (CSV)",
        type=["csv", "txt", "tsv"],
        key="cases"
    )

    df = None  # IMPORTANT : initialiser df

    if cases_file:
        try:
            cases_file.seek(0)
            first_line = cases_file.readline().decode("utf-8").strip()
            cases_file.seek(0)

            if "\t" in first_line:
                separator = "\t"
            elif ";" in first_line:
                separator = ";"
            elif "," in first_line:
                separator = ","
            else:
                separator = None

            if separator:
                df = pd.read_csv(cases_file, sep=separator, encoding="utf-8")
            else:
                df = pd.read_csv(cases_file, sep=None, engine="python", encoding="utf-8")

        except UnicodeDecodeError:
            cases_file.seek(0)
            try:
                if separator:
                    df = pd.read_csv(cases_file, sep=separator, encoding="latin-1")
                else:
                    df = pd.read_csv(cases_file, sep=None, engine="python", encoding="latin-1")
                st.warning("⚠️ Encodage Latin-1 utilisé")
            except Exception as e:
                st.error(f"❌ Erreur de lecture : {str(e)}")
                df = None
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
            df = None

        if df is not None:
            # Nettoyage des noms de colonnes
            df.columns = (
                df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
            )

            # Harmonisation automatique des noms (health_area, week_, cases, deaths, year)
            df = detect_and_rename_core_columns(df)

            required = {"health_area", "week_", "cases"}

            if required.issubset(set(df.columns)):
                st.success("✅ Colonnes requises détectées (y compris variations de noms)")

                # Standardiser les valeurs d’aire de santé
                df["health_area"] = df["health_area"].astype(str).str.strip().str.lower()
               
               # Mapping automatique CSV ->géométries via fuzzy-match
                if st.session_state.gdf_health is not None:
                    auto_map = build_auto_health_area_mapping(
                        st.session_state.gdf_health,
                        df
                    )
                    if auto_map:
                        df["health_area"] = df["health_area"].replace(auto_map)
                        with st.sidebar.expander("ℹ️ Corrections automatiques d'aires", expanded=False):
                            st.write(auto_map)

                # Colonne deaths optionnelle
                if "deaths" not in df.columns:
                    df["deaths"] = 0
                    st.info("ℹ️ Colonne 'deaths' absente → créée avec valeur 0")

                # Normalisation des types
                df["week_"]  = normalize_week_format(df["week_"])
                df["cases"]  = pd.to_numeric(df["cases"],  errors="coerce").fillna(0).astype(int)
                df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").fillna(0).astype(int)

                # 🔹 Année (si présente dans le CSV)
                if "year" in df.columns:
                    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

                # Sécurité supplémentaire sur week_
                df["week_"] = pd.to_numeric(df["week_"], errors="coerce").fillna(1).astype(int)

                # Filtrer valeurs aberrantes
                df = df[(df["cases"] >= 0) & (df["week_"] > 0)]

                # Stockage en session
                st.session_state.df_cases = df
                if "year" in df.columns:
                    df["period"] = (
                        df["year"].astype(int).astype(str) + "-S" +
                        df["week_"].astype(str).str.zfill(2)
                    )
                # ── Statistiques série (filtrées si années déjà sélectionnées) ──
                _years_sel = st.session_state.get("year_filter", None)
                if _years_sel and "year" in df.columns:
                    df_display = df[df["year"].isin(_years_sel)]
                else:
                    df_display = df   # toutes les années par défaut

                periode_min = df_display["period"].min()
                periode_max = df_display["period"].max()
                nb_periodes = df_display["period"].nunique()
                nb_annees   = df_display["year"].nunique() if "year" in df_display.columns else "?"

                st.success(f"✅ {len(df_display)} enregistrements chargés")
                st.sidebar.info(
                    f"📅 **Série de données**\n\n"
                    f"- De : `{periode_min}`\n"
                    f"- À  : `{periode_max}`\n"
                    f"- **{nb_periodes} semaines · {nb_annees} an(s)**"
                )
                # Aperçu
                with st.expander("👁️ Aperçu"):
                    st.dataframe(df.head())

                # Statistiques rapides
                with st.expander("📊 Statistiques"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total cas",     int(df["cases"].sum()))
                    col2.metric("Total décès",   int(df["deaths"].sum()))
                    col3.metric("Aires uniques", df["health_area"].nunique())

            else:
                missing = required - set(df.columns)
                st.error(f"❌ Colonnes requises non trouvées après normalisation : {missing}")
                st.error(f"📋 Colonnes disponibles : {list(df.columns)}")
        else:
            st.error("❌ Impossible de lire le fichier de cas (df est vide).")
    # ── WorldPop — cache par pays (pattern rougeole) ──────────
    cache_key = f"enrichi_{iso3pays}" if iso3pays else "enrichi_upload"
    if cache_key not in st.session_state or st.session_state[cache_key] is None:
        with st.spinner("📥 Enrichissement WorldPop..."):
            dfpopulation = worldpop_malaria_stats(gdf_health, gee_ok)
            gdf_m = gdf_health.to_crs("ESRI:54009")
            surf  = gdf_m.geometry.area / 1e6
            gdf_health_tmp = gdf_health.copy()
            gdf_health_tmp["Superficie_km2"] = surf.values
            dfpopulation = dfpopulation.merge(
                gdf_health_tmp[["health_area", "Superficie_km2"]], on="health_area", how="left"
            )
            dfpopulation["Densite_Pop"] = (
                dfpopulation["Pop_Totale"] / dfpopulation["Superficie_km2"].replace(0, np.nan)
            )
            st.session_state[cache_key] = dfpopulation
    else:
        dfpopulation = st.session_state[cache_key]

    # Merge population → gdf_health
    cols_pop = [c for c in ["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]
                if c in dfpopulation.columns]
    if len(cols_pop) > 1:
        gdf_health = gdf_health.merge(dfpopulation[cols_pop], on="health_area", how="left")
        st.session_state.gdf_health = gdf_health

    # Info WorldPop
    if "Pop_Totale" not in dfpopulation.columns or not dfpopulation["Pop_Totale"].notna().any():
        st.sidebar.warning("⚠️ WorldPop non disponible (GEE requis)")

# ── Récupération iso3pays hors expander (pour les tabs) ───────────────────
iso3pays = st.session_state.get("iso3pays_courant", None)

# ============================================================
# === API CLIMAT - MULTIPLE SOURCES ===
# ============================================================
with st.sidebar.expander("🌦️ API Climat (Optionnel)", expanded=False):
    use_climate_api = st.checkbox("Activer API Climat", value=False, key="use_climate_toggle")

    if use_climate_api:
        st.markdown("### 📡 Choix de la Source")

        api_choice = st.radio(
            "Source de données climatiques",
            ["NASA POWER", "Open-Meteo"],
            help="""
            **NASA POWER**: Fiable, données historiques complètes
            **Open-Meteo**: Excellent, gratuit, sans inscription
            """
        )

        if api_choice == "NASA POWER":
            st.info("""
            📡 **NASA POWER**
            - ✅ Gratuit, sans clé API
            - ✅ Température, précipitations, humidité
            - ✅ Données depuis 1981
            - ⏱️ Temps de réponse : ~1-2 min
            """)

        elif api_choice == "Open-Meteo":
            st.info("""
            📡 **Open-Meteo Archive**
            - ✅ Gratuit, sans clé API
            - ✅ Température, précipitations, humidité
            - ✅ Données depuis 1940
            - ⚡ Très rapide (~30 sec)
            """)

        if st.session_state.gdf_health is not None and st.session_state.df_cases is not None:


            years_for_climat = st.session_state.get("year_filter", [])
            if not years_for_climat:
                st.warning("⚠️ Sélectionnez au moins une année dans les Filtres.")
                year_input = None
            else:
                year_input = int(sorted(years_for_climat)[0])   # 1re année pour référence API
                st.info(
                    f"📅 Année de référence API climat : **{year_input}** "
                    f"(sélectionnées : {sorted(years_for_climat)})"
                )

            if st.session_state.df_climate_aggregated is not None:
                nb_records = len(st.session_state.df_climate_aggregated)
                st.success(f"✅ {nb_records} enregistrements climat en mémoire")

                df_clim = st.session_state.df_climate_aggregated
                col1, col2, col3 = st.columns(3)

                if 'temp_api' in df_clim.columns:
                    col1.metric("🌡️ Temp. moy", f"{df_clim['temp_api'].mean():.1f}°C")
                if 'precip_api' in df_clim.columns:
                    col2.metric("🌧️ Précip. moy", f"{df_clim['precip_api'].mean():.1f}mm")
                if 'humidity_api' in df_clim.columns:
                    col3.metric("💧 Humid. moy", f"{df_clim['humidity_api'].mean():.1f}%")

                if st.button("🔄 Réinitialiser données climat", key="reset_climate"):
                    st.session_state.df_climate_aggregated = None
                    st.success("✅ Données climat effacées")
                    st.rerun()
            else:
                st.info("ℹ️ Aucune donnée climat chargée")

            if st.button("🚀 Télécharger Données Climatiques", key="download_climate", type="primary"):
                if year_input is None:
                    st.error("❌ Aucune année sélectionnée. Choisissez une année dans les filtres en haut de la sidebar.")
                else:
                    with st.spinner(f"⏳ Téléchargement depuis {api_choice} pour l'année {year_input}..."):
                        df_climate_agg = aggregate_climate_by_week_and_area(
                           gdf_health, st.session_state.df_cases, year_input, api_choice
                           )

                    if not df_climate_agg.empty:
                        st.session_state.df_climate_aggregated = df_climate_agg
                        st.success("🎉 Données climatiques intégrées avec succès !")
                    else:
                        st.error("❌ Aucune donnée climatique récupérée")
        else:
            st.warning("⚠️ Chargez d'abord les aires de santé et les cas")

with st.sidebar.expander("🌍 Données Environnementales", expanded=False):
    flood_file = st.file_uploader("🌊 Raster inondation (TIF)", type=["tif"], key="flood")
    if flood_file:
        st.session_state.flood_raster = rasterio.open(flood_file)
        st.success("✅ Inondation chargée")
    
    elev_file = st.file_uploader("⛰️ Raster élévation (TIF)", type=["tif"], key="elevation")
    if elev_file:
        st.session_state.elevation_raster = rasterio.open(elev_file)
        st.success("✅ Élévation chargée")
    
    river_file = st.file_uploader("🏞️ Rivières (GeoJSON/SHP/ZIP)", type=["geojson","shp","zip"], key="rivers")
    if river_file:
        rivers_gdf = gpd.read_file(river_file)
        rivers_gdf = ensure_wgs84(rivers_gdf)
        st.session_state.rivers_gdf = rivers_gdf
        st.success(f"✅ {len(rivers_gdf)} cours d'eau")

# ============================================================
# FILTRES
# ============================================================
st.sidebar.header("🔍 Filtres")
gdf_health = st.session_state.gdf_health
df_cases = st.session_state.df_cases  # None si CSV pas encore chargé

year_selected = None
week_selected = None
area_selected = None

if df_cases is not None:
    # 1) Filtre année (si colonne disponible)
    if "year" in df_cases.columns and df_cases["year"].notna().any():
        years_available = sorted(df_cases["year"].dropna().unique())
        years_selected = st.sidebar.multiselect(
            "📅 Années",
            years_available,
            default=years_available,          # toutes cochées par défaut
            key="year_filter",
            help="Sélectionnez une ou plusieurs années. Le modèle ML nécessite ≥ 2 ans."
        )
        year_selected = years_selected        # alias pour compatibilité avec le reste du code
        
        if len(years_selected) == 1:
            st.sidebar.warning(
                "⚠️ 1 seule année sélectionnée. Le modèle ML nécessite ≥ 2 ans "
                "pour capter la saisonnalité annuelle."
            )
        elif len(years_selected) == 0:
            st.sidebar.error("❌ Aucune année sélectionnée.")
        
        # Les listes semaines/aires affichent TOUTES les valeurs (pas de filtre année ici)
        df_for_filters = df_cases.copy()
        week_selected = st.sidebar.multiselect(
            "📆 Semaines",
            sorted(df_for_filters["week_"].unique()),
            key="week_filter"
        )
        area_selected = st.sidebar.multiselect(
            "🏥 Aires de santé",
            sorted(df_for_filters["health_area"].unique()),
            key="area_filter"
        )
# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard",
    "🗺️ Cartographie",
    "🤖 Modélisation",
    "🗺️ Carte Prédictions",
    "📈 Analyse Avancée",
    "📥 Export",
    "🔬 Validation",
])

# ============================================================
# TAB 1 – DASHBOARD
# ============================================================
with tab1:
    if df_cases is not None:
        df_w = df_cases.copy()

        # Filtre par année sélectionnée
        year_selected = st.session_state.get("year_filter", None)
        if years_selected and "year" in df_w.columns:
            df_w = df_w[df_w["year"].isin(years_selected)]

        # Puis filtres semaine + aire
        if week_selected:
            df_w = df_w[df_w["week_"].isin(week_selected)]
        if area_selected:
            df_w = df_w[df_w["health_area"].isin(area_selected)]

        st.markdown("### 📊 Indicateurs Clés")
        st.markdown("""
        <style>
        [data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(232, 0, 28, 0.18);
            border-left: 3px solid #E8001C;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        }
        [data-testid="stMetricLabel"] {
            color: #555555 !important;
            font-size: 0.80rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        [data-testid="stMetricValue"] {
            color: #1a1a1a !important;
            font-size: 1.35rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            total_cases = safe_int(df_w["cases"].sum())
            st.metric("🦟 Cas Totaux", f"{total_cases:,}")
        with col2:
            total_deaths = safe_int(df_w["deaths"].sum())
            st.metric("💀 Décès", f"{total_deaths:,}")
        with col3:
            st.metric("🏥 Aires", df_w["health_area"].nunique())
        with col4:
            st.metric("📅 Semaines", df_display["period"].nunique())
        with col5:
            cfr = (total_deaths / total_cases * 100) if total_cases > 0 else 0
            st.metric("⚠️ Létalité", f"{cfr:.1f}%")

        # ── Incidence & Taux d'Attaque (si population disponible) ──────
        iso3pays = st.session_state.get("iso3pays_courant", None)
        cache_key_pop = f"enrichi_{iso3pays}" if iso3pays else "enrichi_upload"
        _df_pop_kpi = st.session_state.get(cache_key_pop)
        if _df_pop_kpi is not None and not _df_pop_kpi.empty and "Pop_Totale" in _df_pop_kpi.columns:
            _pop_filtre = _df_pop_kpi.copy()
            if area_selected:
                _pop_filtre = _pop_filtre[_pop_filtre["health_area"].isin(area_selected)]
            _pop_tot_kpi    = safe_float(_pop_filtre["Pop_Totale"].sum())
            _pop_enf_kpi    = safe_float(_pop_filtre.get("Pop_Enfants_0_14", pd.Series([0])).sum())
            _incidence      = (total_cases / _pop_tot_kpi * 10000) if _pop_tot_kpi > 0 else 0
            _ta_enfants_kpi = (total_cases / _pop_enf_kpi * 1000)  if _pop_enf_kpi > 0 else 0
            st.markdown("<br>", unsafe_allow_html=True)
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.metric("📈 Incidence (p.10 000 hab)", f"{_incidence:.1f}")
            with col_i2:
                st.metric("🎯 Taux d'Attaque Enfants (p.1 000)", f"{_ta_enfants_kpi:.1f}")
            with col_i3:
                _nb_aires_kpi = df_w["health_area"].nunique()
                _incidence_moy_aire = (_incidence / _nb_aires_kpi) if _nb_aires_kpi > 0 else 0
                st.metric("🏥 Incidence Moy./Aire", f"{_incidence_moy_aire:.2f}")

        # KPI Population
        iso3pays = st.session_state.get("iso3pays_courant", None)
        cache_key_pop = f"enrichi_{iso3pays}" if iso3pays else "enrichi_upload"
        if st.session_state.get(cache_key_pop) is not None and not st.session_state[cache_key_pop].empty:
            df_pop = st.session_state[cache_key_pop].copy()
            if area_selected:
                df_pop = df_pop[df_pop["health_area"].isin(area_selected)]
            st.markdown("<br>", unsafe_allow_html=True)
            colp1, colp2, colp3 = st.columns(3)
            with colp1:
                st.metric("👥 Population totale",
                          f"{int(df_pop['Pop_Totale'].sum()):,}".replace(",", " ")
                          if "Pop_Totale" in df_pop.columns else "N/A")
            with colp2:
                st.metric("👶 Enfants 0–14 ans",
                          f"{int(df_pop['Pop_Enfants_0_14'].sum()):,}".replace(",", " ")
                          if "Pop_Enfants_0_14" in df_pop.columns else "N/A")
            with colp3:
                st.metric("📏 Densité moyenne",
                          f"{df_pop['Densite_Pop'].mean():.1f} hab/km²"
                          if "Densite_Pop" in df_pop.columns else "N/A")
       # Pyramide des âges - VERSION ROBUSTE
            cache_key_pop = f"enrichi_{iso3pays}" if iso3pays else "enrichi_upload"
            if st.session_state.get(cache_key_pop) is not None and not st.session_state[cache_key_pop].empty:
                df_pop = st.session_state[cache_key_pop]
                if area_selected:
                    df_pop = df_pop[df_pop["health_area"].isin(area_selected)]
            
                total_pop = df_pop["Pop_Totale"].sum()
                enfants_0_14 = df_pop.get("Pop_Enfants_0_14", pd.Series([0]*len(df_pop))).sum()
            
                st.markdown("---")
                st.subheader("👥 Pyramide des âges")
            
                # ✅ VÉRIFIER si données détaillées disponibles (CORRECTION NOMS)
                detailed_ages = ['0_4', '5_9', '10_14', '15_19', '20_24', '25_29', '30_34']
                has_detailed = all(f"Pop_MALE_{age}" in df_pop.columns for age in detailed_ages)  # ✅ MALE en majuscules
            
                if has_detailed:
                    # Pyramide DÉTAILLÉE <35 ans
                    pop_data = {}
                    for group in detailed_ages:
                        pop_data[group] = {
                            'male': df_pop[f'Pop_MALE_{group}'].sum(),      # ✅ MALE en majuscules
                            'female': df_pop[f'Pop_FEMALE_{group}'].sum()   # ✅ FEMALE en majuscules
                        }
                    
                    fig_pyr = go.Figure()
                    
                    # Hommes
                    fig_pyr.add_trace(go.Bar(
                        y=[f"{int(g.split('_')[0])}-{int(g.split('_')[1])}" for g in detailed_ages],
                        x=[-pop_data[g]['male'] for g in detailed_ages],
                        name="Hommes", orientation="h", marker_color="#1f77b4", opacity=0.85,
                        text=[f"{int(pop_data[g]['male']):,}" for g in detailed_ages], 
                        textposition="inside", insidetextanchor="end"
                    ))
                    
                    # Femmes  
                    fig_pyr.add_trace(go.Bar(
                        y=[f"{int(g.split('_')[0])}-{int(g.split('_')[1])}" for g in detailed_ages],
                        x=[pop_data[g]['female'] for g in detailed_ages],
                        name="Femmes", orientation="h", marker_color="#ff7f0e", opacity=0.85,
                        text=[f"{int(pop_data[g]['female']):,}" for g in detailed_ages], 
                        textposition="inside"
                    ))
                    
                    total_under_35 = sum(pop_data[g]['male'] + pop_data[g]['female'] for g in detailed_ages)
                    _denom = total_pop if total_pop > 0 else 1
                    subtitle = f"Détaillée <35 ans | Total: {int(total_pop):,} | <35 ans: {int(total_under_35):,} ({total_under_35/_denom*100:.1f}%)"
                    
                else:
                    # Pyramide SIMPLIFIÉE (0-14 vs 15+)
                    st.info("ℹ️ Données détaillées indisponibles - Vue simplifiée")
                    
                    adultes_15p = max(total_pop - enfants_0_14, 0)
                    enfants_g, enfants_f = enfants_0_14 * 0.51, enfants_0_14 * 0.49
                    adultes_g, adultes_f = adultes_15p * 0.48, adultes_15p * 0.52
                    
                    fig_pyr = go.Figure()
                    fig_pyr.add_trace(go.Bar(
                        y=["0-14 ans", "15+ ans"],
                        x=[-enfants_g, -adultes_g], name="Hommes", orientation="h", 
                        marker_color="#1f77b4", opacity=0.8,
                        text=[f"{int(enfants_g):,}", f"{int(adultes_g):,}"],
                        textposition="inside", insidetextanchor="end"
                    ))
                    fig_pyr.add_trace(go.Bar(
                        y=["0-14 ans", "15+ ans"],
                        x=[enfants_f, adultes_f], name="Femmes", orientation="h", 
                        marker_color="#ff7f0e", opacity=0.8,
                        text=[f"{int(enfants_f):,}", f"{int(adultes_f):,}"],
                        textposition="inside"
                    ))
                    
                    _denom = total_pop if total_pop > 0 else 1
                    subtitle = f"Simplifiée | Total: {int(total_pop):,} | Enfants 0-14: {int(enfants_0_14):,} ({enfants_0_14/_denom*100:.1f}%)"
            
                # Layout commun
                fig_pyr.update_layout(
                    barmode="relative",
                    title={"text": f"Structure démographique (WorldPop 100m)<br><sub>{subtitle}</sub>", 
                           "x": 0.5, "font": {"size": 16}},
                    xaxis={"title": "Population", "tickformat": ",", "zeroline": True},
                    yaxis={"title": "Âge"},
                    height=500, 
                    legend={"x": 0.02, "y": 1.02}
                )
                
                st.plotly_chart(fig_pyr, use_container_width=True) 
       # Section climat
        if st.session_state.df_climate_aggregated is not None:
            df_clim = st.session_state.df_climate_aggregated  # ✅ FIX NameError
            st.markdown("""
            <style>
            div[data-testid="stMetric"].clim-card {
                background: rgba(219, 234, 254, 0.80) !important;
                border: 1px solid rgba(59, 130, 246, 0.25) !important;
                border-left: 3px solid #3B82F6 !important;
                border-radius: 8px;
                padding: 8px 14px !important;
            }
            </style>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'temp_api' in df_clim.columns:
                    st.markdown('<div class="clim-card">', unsafe_allow_html=True)
                    st.metric("🌡️ Temp. moy.", f"{df_clim['temp_api'].mean():.1f} °C")
                    st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                if 'precip_api' in df_clim.columns:
                    st.markdown('<div class="clim-card">', unsafe_allow_html=True)
                    st.metric("🌧️ Précip. tot.", f"{df_clim['precip_api'].sum():.1f} mm")
                    st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                if 'humidity_api' in df_clim.columns:
                    st.markdown('<div class="clim-card">', unsafe_allow_html=True)
                    st.metric("💧 Humid. moy.", f"{df_clim['humidity_api'].mean():.1f} %")
                    st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="clim-card">', unsafe_allow_html=True)
                st.metric("📅 Semaines", df_clim['week_'].nunique())
                st.markdown('</div>', unsafe_allow_html=True)


            # CORRECTION: Graphiques séparés
            st.markdown("### 📈 Évolution Hebdomadaire")

        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            # Graphique CAS et DÉCÈS
            x_col = "period" if "period" in df_w.columns else "week_"
            df_week_cases = df_w.groupby(x_col).agg({"cases": "sum", "deaths": "sum"}).reset_index()
            
            fig_cases = go.Figure()
            fig_cases.add_trace(go.Bar(
                x=df_week_cases[x_col], y=df_week_cases["cases"],
                name='Cas', marker_color='#FF6B6B'
            ))
            fig_cases.add_trace(go.Scatter(
                x=df_week_cases[x_col], y=df_week_cases["deaths"],
                mode='lines+markers', name='Décès',
                line=dict(color='#4ECDC4', width=3), yaxis='y2'
            ))
            fig_cases.update_layout(
                title="Cas et Décès par Semaine",
                xaxis_title="Semaine (Année-SXX)",
                yaxis=dict(title="Nombre de Cas"),
                yaxis2=dict(title="Nombre de Décès", overlaying='y', side='right'),
                height=400, hovermode='x unified'
            )
            st.plotly_chart(fig_cases, use_container_width=True)

        with col_graph2:
            # Graphique CLIMAT — uniquement si données disponibles
            if st.session_state.df_climate_aggregated is not None:
                _df_clim_g = st.session_state.df_climate_aggregated.copy()
                if week_selected:
                    _df_clim_g = _df_clim_g[_df_clim_g["week_"].isin(week_selected)]
                if area_selected:
                    _df_clim_g = _df_clim_g[_df_clim_g["health_area"].isin(area_selected)]

                df_week_climate = _df_clim_g.groupby("week_").agg({
                    col: 'mean' for col in _df_clim_g.columns
                    if col.endswith('_api') and '_min' not in col and '_max' not in col
                }).reset_index()

                fig_climate = go.Figure()
                if 'temp_api' in df_week_climate.columns:
                    fig_climate.add_trace(go.Scatter(
                        x=df_week_climate["week_"], y=df_week_climate["temp_api"],
                        mode='lines+markers', name='Température (°C)',
                        line=dict(color='orange', width=3)
                    ))
                if 'precip_api' in df_week_climate.columns:
                    fig_climate.add_trace(go.Scatter(
                        x=df_week_climate["week_"], y=df_week_climate["precip_api"],
                        mode='lines+markers', name='Précipitations (mm)',
                        line=dict(color='blue', width=2), yaxis='y2'
                    ))
                if 'humidity_api' in df_week_climate.columns:
                    fig_climate.add_trace(go.Scatter(
                        x=df_week_climate["week_"], y=df_week_climate["humidity_api"],
                        mode='lines+markers', name='Humidité (%)',
                        line=dict(color='green', width=2, dash='dash'), yaxis='y3'
                    ))
                fig_climate.update_layout(
                    title="Données Climatiques par Semaine",
                    xaxis_title="Semaine",
                    yaxis=dict(title="Temp °C"),
                    yaxis2=dict(title="Précip mm", overlaying='y', side='right'),
                    yaxis3=dict(title="Humid %", overlaying='y', side='right',
                                anchor='free', position=0.95),
                    height=400, hovermode='x unified'
                )
                st.plotly_chart(fig_climate, use_container_width=True)
            else:
                st.info("ℹ️ Activez l'API Climat pour afficher les données climatiques ici.")

        st.subheader("📈 Évolution Temporelle")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            x_col = "period" if "period" in df_w.columns else "week_"
            df_week = df_w.groupby(x_col).agg({"cases": "sum", "deaths": "sum"}).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_week[x_col],
                y=df_week["cases"],
                mode='lines+markers',
                name='Cas',
                line=dict(color='#FF6B6B', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=df_week[x_col], 
                y=df_week["deaths"],
                mode='lines+markers',
                name='Décès',
                line=dict(color='#4ECDC4', width=3),
                yaxis='y2'
            ))
            fig.update_layout(
                title="Évolution cas et décès",
                xaxis_title="Semaine",
                yaxis_title="Cas",
                yaxis2=dict(title="Décès", overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            df_top = df_w.groupby("health_area")["cases"].sum().sort_values(ascending=False).head(10)
            fig2 = px.bar(
                x=df_top.values,
                y=df_top.index,
                orientation='h',
                title="Top 10 Aires",
                color=df_top.values,
                color_continuous_scale='Reds'
            )
            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TAB 2 – CARTOGRAPHIE (VERSION FINALE CORRIGÉE)
# ============================================================

with tab2:
    if gdf_health is not None and df_cases is not None:
        st.subheader("🗺️ Cartographie Interactive")
        
        # =====================================================
        # SECTION 1 : VISUALISATION CLIMAT (optionnelle)
        # =====================================================
        if st.session_state.df_climate_aggregated is not None:
            st.markdown("---")
            st.markdown("## 🌡️ VISUALISATION DONNÉES CLIMATIQUES")
            
            df_climate = st.session_state.df_climate_aggregated
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                weeks_available = sorted(df_climate['week_'].unique())
                selected_week_climate = st.selectbox(
                    "📅 Semaine",
                    weeks_available,
                    key="week_climate_map"
                )
            
            with col2:
                climate_vars = [c for c in df_climate.columns if c.endswith('_api') and '_min' not in c and '_max' not in c]
                
                var_labels = {
                    'temp_api': '🌡️ Température',
                    'precip_api': '🌧️ Précipitations',
                    'humidity_api': '💧 Humidité'
                }
                
                climate_var = st.selectbox(
                    "Variable",
                    climate_vars,
                    format_func=lambda x: var_labels.get(x, x),
                    key="climate_var_map"
                )
            
            with col3:
                pass  
            
            df_week_climate = df_climate[df_climate['week_'] == selected_week_climate]
            
            if not df_week_climate.empty:
                gdf_climate = gdf_health.merge(
                    df_week_climate[['health_area', climate_var]],
                    on='health_area',
                    how='left'
                )
                
                if not gdf_climate[climate_var].isna().all():
                    center = gdf_climate.geometry.union_all().centroid
                    
                    m_climate = folium.Map(
                        location=[center.y, center.x],
                        zoom_start=8,
                        tiles="CartoDB positron"
                    )
                    
                    folium.Choropleth(
                        geo_data=gdf_climate,
                        data=gdf_climate,
                        columns=['health_area', climate_var],
                        key_on='feature.properties.health_area',
                        fill_color='RdYlBu_r' if 'temp' in climate_var else 'Blues',
                        fill_opacity=0.7,
                        line_opacity=0.8,
                        legend_name=f"{var_labels.get(climate_var, climate_var)} - S{selected_week_climate}"
                    ).add_to(m_climate)
                    
                    feature_group_labels_climate = folium.FeatureGroup(name="Étiquettes Climat", show=True)
                    
                    for idx, row in gdf_climate.iterrows():
                        if not pd.isna(row[climate_var]):
                            centroid = row.geometry.centroid
                            
                            folium.Marker(
                                location=[centroid.y, centroid.x],
                                icon=DivIcon(html=f"""
                                    <div style="font-size: 9pt; font-weight: 600; color: #1a1a1a; 
                                    text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 0px 2px 3px rgba(0,0,0,0.3); white-space: nowrap;">
                                        {row['health_area']}<br>
                                        <span style="color: #d32f2f; font-weight: 700;">{row[climate_var]:.1f}</span>
                                    </div>
                                """)
                            ).add_to(feature_group_labels_climate)
                    
                    feature_group_labels_climate.add_to(m_climate)
                    LayerControl().add_to(m_climate)
                    
                    st_folium(m_climate, width=1200, height=500, key="map_climate")
                    
                    values = gdf_climate[climate_var].dropna()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Min",       f"{values.min():.2f}")
                    col2.metric("Moy",       f"{values.mean():.2f}")
                    col3.metric("Max",       f"{values.max():.2f}")
                    col4.metric("Écart-type",f"{values.std():.2f}")
                    st.caption(f"📊 {len(df_week_climate)} enregistrements · Semaine {selected_week_climate} · {var_labels.get(climate_var, climate_var)}")
            
            st.markdown("---")
        
        # =====================================================
        # SECTION 2 : CARTE ÉPIDÉMIOLOGIQUE
        # =====================================================
        st.markdown("## 📍 RÉPARTITION DES CAS")
        
        # Filtrer données cas
        df_plot = df_cases.copy()
        if week_selected:
            df_plot = df_plot[df_plot["week_"].isin(week_selected)]
        if area_selected:
            df_plot = df_plot[df_plot["health_area"].isin(area_selected)]

        # Agréger par aire de santé
        df_agg = df_plot.groupby("health_area", as_index=False).agg({"cases":"sum","deaths":"sum"})
        
        # ✅ CRÉER gdf_map (base épidémiologique)
        gdf_map = gdf_health.merge(df_agg, on="health_area", how="left")
        gdf_map[["cases","deaths"]] = gdf_map[["cases","deaths"]].fillna(0)
        
       # ✅ MERGER POPULATION dans gdf_map
        _pop_cols_attendus = ['Pop_Totale', 'Pop_Enfants_0_14', 'Densite_Pop']
        _pop_cols_manquants = [c for c in _pop_cols_attendus if c not in gdf_map.columns]
        if _pop_cols_manquants and 'dfpopulation' in st.session_state and st.session_state.dfpopulation is not None:
            _df_pop_raw = st.session_state.dfpopulation
            _cols_pop = ['health_area'] + [c for c in _pop_cols_manquants if c in _df_pop_raw.columns]
            if 'Densite_Pop' in _pop_cols_manquants and 'Densite_Pop' not in _df_pop_raw.columns:
                if 'Densite_Pop' in st.session_state.gdf_health.columns:
                    _df_pop_raw = _df_pop_raw.merge(
                        st.session_state.gdf_health[['health_area', 'Densite_Pop']],
                        on='health_area', how='left'
                    )
                    _cols_pop = ['health_area'] + [c for c in _pop_cols_manquants if c in _df_pop_raw.columns]
            if len(_cols_pop) > 1:
                gdf_map = gdf_map.merge(_df_pop_raw[_cols_pop], on='health_area', how='left')
        for _c in _pop_cols_attendus:
            if _c not in gdf_map.columns:
                gdf_map[_c] = np.nan
        _pop_ok = gdf_map['Pop_Totale'].notna().sum()
        if _pop_ok > 0:
            st.info(f"✅ Population disponible : {_pop_ok}/{len(gdf_map)} aires")
        
        # Ajouter moyennes climatiques
        if st.session_state.df_climate_aggregated is not None:
            df_climate_avg = st.session_state.df_climate_aggregated.groupby('health_area').agg({
                col: 'mean' for col in st.session_state.df_climate_aggregated.columns
                if col.endswith('_api') and '_min' not in col and '_max' not in col
            }).reset_index()
            gdf_map = gdf_map.merge(df_climate_avg, on='health_area', how='left')

        # Données environnementales
        with st.spinner("📊 Extraction données environnementales..."):
            if st.session_state.flood_raster is not None:
                gdf_map["flood_mean"] = extract_raster_statistics(gdf_map, st.session_state.flood_raster, 'mean')
            
            if st.session_state.elevation_raster is not None:
                gdf_map["elevation_mean"] = extract_raster_statistics(gdf_map, st.session_state.elevation_raster, 'mean')
            
            if st.session_state.rivers_gdf is not None and not st.session_state.rivers_gdf.empty:
                gdf_map["dist_river"] = gdf_map.centroid.apply(
                    lambda x: distance_to_nearest_line(x, st.session_state.rivers_gdf)
                )
            
            gdf_map = create_environmental_features(gdf_map)

        # Contrôles carte
        col1, col2, col3 = st.columns(3)
        with col1:
            map_viz = st.selectbox("Type", ["Choroplèthe", "Cercles", "Heatmap"])
        with col2:
            show_rasters = st.multiselect("Rasters", ["Inondation", "Élévation"])
        with col3:
            show_rivers = st.checkbox("Rivières", value=False)

        # Initialiser carte
        center = gdf_map.geometry.union_all().centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=8, tiles="CartoDB positron")

        # Ajouter rasters
        if "Inondation" in show_rasters and st.session_state.flood_raster is not None:
            add_raster_to_map(m, st.session_state.flood_raster, "Inondation")
        if "Élévation" in show_rasters and st.session_state.elevation_raster is not None:
            add_raster_to_map(m, st.session_state.elevation_raster, "Élévation")

        # Couches selon type de visualisation
        if map_viz == "Choroplèthe":
            folium.Choropleth(
                geo_data=gdf_map,
                data=gdf_map,
                columns=["health_area", "cases"],
                key_on="feature.properties.health_area",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                name="Aire de santé (cas)",
                legend_name="Cas"
            ).add_to(m)
        
        elif map_viz == "Cercles":
            feature_group_boundaries = folium.FeatureGroup(name="Aires de Santé (limites)", show=True)
            for idx, row in gdf_map.iterrows():
                folium.GeoJson(
                    row['geometry'],
                    style_function=lambda x: {
                        'fillColor': 'transparent',
                        'color': '#2E86AB',
                        'weight': 2,
                        'fillOpacity': 0,
                        'opacity': 0.6
                    }
                ).add_to(feature_group_boundaries)
            feature_group_boundaries.add_to(m)
            
            feature_group_circles = folium.FeatureGroup(name="Cercles Proportionnels (Cas)", show=True)
            max_cases = max(gdf_map["cases"].max(), 1)
            for idx, row in gdf_map.iterrows():
                radius = 10 + 40 * safe_float(row["cases"]) / max_cases
                CircleMarker(
                    location=[row.geometry.centroid.y, row.geometry.centroid.x],
                    radius=radius,
                    color='#FF4444',
                    fill=True,
                    fillOpacity=0.7,
                    popup=f"<b>{row['health_area']}</b><br>Cas: {safe_int(row['cases'])}"
                ).add_to(feature_group_circles)
            feature_group_circles.add_to(m)
        
        elif map_viz == "Heatmap":
            heat_data = [
                [row.geometry.centroid.y, row.geometry.centroid.x, safe_float(row["cases"])]
                for _, row in gdf_map.iterrows() if safe_float(row["cases"]) > 0
            ]
            if heat_data:
                HeatMap(heat_data, radius=15, blur=25, name="Heatmap Cas").add_to(m)

        # Rivières
        if show_rivers and st.session_state.rivers_gdf is not None and not st.session_state.rivers_gdf.empty:
            GeoJson(
                st.session_state.rivers_gdf,
                name="Rivières",
                style_function=lambda x: {"color":"#0066CC", "weight": 2}
            ).add_to(m)
        
        # =====================================================
        # ÉTIQUETTES AIRES (toujours visibles)
        # =====================================================
        feature_group_labels = folium.FeatureGroup(name="Étiquettes Aires", show=True)
        for idx, row in gdf_map.iterrows():
            centroid = row.geometry.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=DivIcon(html=f"""
                    <div style="font-size: 9pt; font-weight: 600; color: #1a1a1a;
                    text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 0px 2px 3px rgba(0,0,0,0.3); white-space: nowrap;">
                        {row['health_area']}
                    </div>
                """)
            ).add_to(feature_group_labels)
        feature_group_labels.add_to(m)
        
        # =====================================================
        # POPUPS DÉTAILLÉS (avec population + climat + env)
        # =====================================================
        feature_group_popups = folium.FeatureGroup(name="Popups Détails", show=False)
        
        for idx, row in gdf_map.iterrows():
            # Construction du HTML popup
            # Calcul taux d'attaque pour ce popup
            _cases_row   = safe_int(row['cases'])
            _pop_row     = safe_float(row.get('Pop_Totale', 0))
            _enfants_row = safe_float(row.get('Pop_Enfants_0_14', 0))
            _ta_total    = (_cases_row / _pop_row * 10000)    if _pop_row > 0    else None
            _ta_enfants  = (_cases_row / _enfants_row * 10000) if _enfants_row > 0 else None

            popup_html = f"""
            <div style="width:360px; font-family:Arial; font-size:12px;">
                <h4 style="color:#2E86AB; margin:0; text-transform:capitalize;">
                    {row['health_area']}
                </h4>
                <hr style="margin:5px 0; border-color:#2E86AB;">

                <p style="margin:4px 0; font-weight:700; color:#555;">📊 Épidémiologie</p>
                <table style="width:100%; border-collapse:collapse;">
                    <tr style="background:#FFF3E0;">
                        <td style="padding:3px 6px;"><b>🦟 Cas</b></td>
                        <td style="padding:3px 6px; text-align:right;">{_cases_row:,}</td>
                    </tr>
                    <tr>
                        <td style="padding:3px 6px;"><b>💀 Décès</b></td>
                        <td style="padding:3px 6px; text-align:right;">{safe_int(row['deaths']):,}</td>
                    </tr>
            """
            if _ta_total is not None:
                popup_html += f"""
                    <tr style="background:#FCE4EC;">
                        <td style="padding:3px 6px;"><b>📈 TA (p.10 000 hab)</b></td>
                        <td style="padding:3px 6px; text-align:right; font-weight:700; color:#c62828;">{_ta_total:.1f}</td>
                    </tr>"""
            if _ta_enfants is not None:
                popup_html += f"""
                    <tr style="background:#F3E5F5;">
                        <td style="padding:3px 6px;"><b>👶 TA enfants (p.10 000)</b></td>
                        <td style="padding:3px 6px; text-align:right; font-weight:700; color:#6a1b9a;">{_ta_enfants:.1f}</td>
                    </tr>"""

            # Bloc population
            if 'Pop_Totale' in gdf_map.columns or 'Pop_Enfants_0_14' in gdf_map.columns:
                popup_html += """<tr><td colspan="2" style="padding:6px 6px 2px;">
                    <b style="color:#555;">👥 Démographie</b></td></tr>"""
            if 'Pop_Totale' in gdf_map.columns and pd.notna(row.get('Pop_Totale')):
                popup_html += f"""
                    <tr style="background:#E8F5E9;">
                        <td style="padding:3px 6px;">Population totale</td>
                        <td style="padding:3px 6px; text-align:right;">{int(row['Pop_Totale']):,}</td>
                    </tr>"""
            if 'Pop_Enfants_0_14' in gdf_map.columns and pd.notna(row.get('Pop_Enfants_0_14')):
                popup_html += f"""
                    <tr style="background:#E3F2FD;">
                        <td style="padding:3px 6px;">Enfants 0–14 ans</td>
                        <td style="padding:3px 6px; text-align:right;">{int(row['Pop_Enfants_0_14']):,}</td>
                    </tr>"""
            if 'Densite_Pop' in gdf_map.columns and pd.notna(row.get('Densite_Pop')):
                popup_html += f"""
                    <tr>
                        <td style="padding:3px 6px;">Densité</td>
                        <td style="padding:3px 6px; text-align:right;">{safe_float(row['Densite_Pop']):.1f} hab/km²</td>
                    </tr>"""
            # Climat (si disponible)
            if 'temp_api' in gdf_map.columns and pd.notna(row.get('temp_api')):
                popup_html += f"<tr style='background:#FFF3E0;'><td><b>🌡️ Température:</b></td><td>{safe_float(row['temp_api']):.1f}°C</td></tr>"
            if 'precip_api' in gdf_map.columns and pd.notna(row.get('precip_api')):
                popup_html += f"<tr style='background:#E1F5FE;'><td><b>🌧️ Précipitations:</b></td><td>{safe_float(row['precip_api']):.1f}mm</td></tr>"
            if 'humidity_api' in gdf_map.columns and pd.notna(row.get('humidity_api')):
                popup_html += f"<tr style='background:#E8F5E9;'><td><b>💧 Humidité:</b></td><td>{safe_float(row['humidity_api']):.1f}%</td></tr>"
            
            # Environnement (si disponible)
            if 'flood_mean' in gdf_map.columns and pd.notna(row.get('flood_mean')):
                popup_html += f"<tr><td><b>🌊 Inondation:</b></td><td>{safe_float(row['flood_mean']):.2f}</td></tr>"
            if 'elevation_mean' in gdf_map.columns and pd.notna(row.get('elevation_mean')):
                popup_html += f"<tr><td><b>⛰️ Élévation:</b></td><td>{safe_float(row['elevation_mean']):.0f}m</td></tr>"
            if 'dist_river' in gdf_map.columns and pd.notna(row.get('dist_river')):
                popup_html += f"<tr><td><b>🏞️ Dist. rivière:</b></td><td>{safe_float(row['dist_river']):.2f}km</td></tr>"
            
            # Fermeture table et div
            popup_html += "</table></div>"
            
            # Ajouter popup au layer
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x: {'fillOpacity': 0, 'color': 'transparent'},
                popup=folium.Popup(popup_html, max_width=360)
            ).add_to(feature_group_popups)
        
        feature_group_popups.add_to(m)
        
        # Affichage final
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=1200, height=700, key="main_map")
    
    else:
        st.info("ℹ️ Chargez d'abord les aires de santé et les cas dans la sidebar")




        
# ============================================================
# TAB 3 – MODÉLISATION SIMPLIFIÉE
# ============================================================

with tab3:
    if df_cases is not None and gdf_health is not None:
        st.subheader("🤖 Modélisation Prédictive du Paludisme")
        
        st.markdown("""
        <div class="info-box">
        🎯 <b>Objectif</b> : Prévoir les cas de paludisme pour anticiper les besoins en ressources
        </div>
        """, unsafe_allow_html=True)
        
        # ========================================
        # CONFIGURATION SIMPLIFIÉE
        # ========================================
        st.markdown("### ⚙️ Configuration")
        
        col_conf1, col_conf2 = st.columns([2, 1])
        
        with col_conf1:
            # Paramètres principaux
            subcol1, subcol2 = st.columns(2)
            
            with subcol1:
                model_choice = st.selectbox(
                    "🤖 Algorithme",
                    ["RandomForest", "GradientBoosting", "ExtraTrees"],
                    help="**RandomForest** : Équilibré (recommandé)\n**GradientBoosting** : Plus précis\n**ExtraTrees** : Rapide"
                )
            
            with subcol2:
                n_future_weeks = st.slider(
                    "📅 Semaines à prévoir", 
                    1, 12, 4,
                    help="1-4 semaines : fiable | 5-12 semaines : indicatif"
                )
                
                # ── Détecter et afficher la dernière semaine de la série ──────
                df_mod_preview = df_cases.copy()
                if years_selected and "year" in df_mod_preview.columns:
                    df_mod_preview = df_mod_preview[df_mod_preview["year"].isin(years_selected)]
                
                if "period" in df_mod_preview.columns:
                    last_period  = df_mod_preview["period"].max()
                    first_period = df_mod_preview["period"].min()
                    nb_total_sem = df_mod_preview["period"].nunique()
                    st.info(
                        f"📌 **Série sélectionnée** : `{first_period}` → `{last_period}`  \n"
                        f"📊 **{nb_total_sem} semaines** · Les prédictions démarrent après `{last_period}`"
                    )
            
            # Mode
            mode = st.radio(
                "🎚️ Mode",
                ["🟢 Simple", "🔵 Expert"],
                horizontal=True,
                help="Simple : Optimisé auto | Expert : Contrôle total"
            )
            # Avec les autres sliders
            alert_threshold = st.slider("🚨 Seuil alerte (%)", 50, 95, 75, 
                                        help="Top X% des prédictions considérées à risque")
        with col_conf2:
            st.markdown("#### 📊 État")
            st.info(f"""
            **Algo** : {model_choice}  
            **Horizon** : {n_future_weeks}W  
            **Mode** : {mode.split()[1]}
            """)
            
            # Score qualité
            nb_weeks = df_cases['week_'].nunique()
            has_climate = st.session_state.df_climate_aggregated is not None
            quality = min(100, nb_weeks*1.5 + (40 if has_climate else 0))
            
            st.metric("🎯 Qualité Données", f"{quality:.0f}/100")
        
        # Paramètres avancés (mode expert)
        if "Expert" in mode:
            with st.expander("🔧 Paramètres Avancés"):
                col1, col2 = st.columns(2)
                
                with col1:
                    use_pca = st.checkbox("📐 ACP", True, help="Réduction dimensionnalité")
                    if use_pca:
                        variance_threshold = st.slider("% Variance", 80, 99, 95) / 100
                
                with col2:
                    use_spatial = st.checkbox("🗺️ Spatial", True, help="Clustering + lag")
                    if use_spatial:
                        c1, c2 = st.columns(2)
                        with c1:
                            n_clusters = st.slider("Clusters", 3, 10, 5)
                        with c2:
                            k_neighbors = st.slider("Voisins", 3, 10, 5)
        else:
            use_pca, variance_threshold = True, 0.95
            use_spatial, n_clusters, k_neighbors = True, 5, 5
        
        st.markdown("---")
        
        # ========================================
        # BOUTON LANCEMENT
        # ========================================
        if st.button("🚀 LANCER MODÉLISATION", type="primary", use_container_width=True):
            with st.spinner("⏳ Traitement en cours..."):
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    # ÉTAPE 1 : Données de base (0-20%)
                    status.text("📊 1/6 : Préparation données...")
                    df_mod_src = df_cases.copy()
                    if years_selected and "year" in df_mod_src.columns:
                        df_mod_src = df_mod_src[df_mod_src["year"].isin(years_selected)]
                    df_model = df_mod_src.groupby(["health_area", "week_"], as_index=False).agg({"cases": "sum"})
                    df_model["week_num"] = pd.factorize(df_model["week_"])[0]
                    progress_bar.progress(10)
                    
                    # Intégration climat
                    climate_features = []
                    if st.session_state.df_climate_aggregated is not None:
                        df_climate = st.session_state.df_climate_aggregated
                        df_model = df_model.merge(
                            df_climate[['health_area', 'week_', 'temp_api', 'precip_api', 'humidity_api']],
                            on=['health_area', 'week_'], how='left'
                        )
                        climate_features = [c for c in ['temp_api', 'precip_api', 'humidity_api'] 
                                          if c in df_model.columns and df_model[c].notna().sum() > 0]
                    progress_bar.progress(20)
                    
                    # ÉTAPE 2 : Features temporelles (20-40%)
                    status.text("⏰ 2/6 : Features temporelles...")
                    df_model = create_advanced_features(df_model)
                    progress_bar.progress(40)

                    # 🎯 Features population (après features temporelles)
                    df_model = create_population_features(df_model)

                    # ÉTAPE 3 : Environnement (40-50%)
                    status.text("🌍 3/6 : Données environnementales...")
                    gdf_env = gdf_health.copy()
                    static_env_cols = []
                    
                    if st.session_state.flood_raster:
                        gdf_env["flood_mean"] = extract_raster_statistics(gdf_env, st.session_state.flood_raster, 'mean')
                        static_env_cols.append("flood_mean")
                    
                    if st.session_state.elevation_raster:
                        gdf_env["elevation_mean"] = extract_raster_statistics(gdf_env, st.session_state.elevation_raster, 'mean')
                        static_env_cols.append("elevation_mean")
                    
                    if st.session_state.rivers_gdf is not None and not st.session_state.rivers_gdf.empty:
                        gdf_env["dist_river"] = gdf_env.centroid.apply(
                            lambda x: distance_to_nearest_line(x, st.session_state.rivers_gdf)
                        )
                        static_env_cols.append("dist_river")
                    
                    gdf_env = create_environmental_features(gdf_env)
                    if 'flood_risk' in gdf_env.columns:
                        static_env_cols.append('flood_risk')
                    # Intégration population dans gdf_env
                    if 'dfpopulation' in st.session_state and st.session_state.dfpopulation is not None and not st.session_state.dfpopulation.empty:  # ✅

                        gdf_env = gdf_env.merge(
                           st.session_state.dfpopulation[["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]],
                            on="health_area",
                            how="left"
                        )
                        static_env_cols.extend(
                            [c for c in ["Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"] if c in gdf_env.columns]
                        )

                    static_env_cols = [c for c in static_env_cols if c in gdf_env.columns]
                    if static_env_cols:
                        df_model = df_model.merge(gdf_env[['health_area'] + static_env_cols], on="health_area", how="left")
                    progress_bar.progress(50)
                    
                    # ÉTAPE 4 : Spatial (50-60%)
                    if use_spatial:
                        status.text("🗺️ 4/6 : Analyse spatiale...")
                        clusters, _ = create_spatial_clusters(gdf_env, n_clusters)
                        gdf_env['spatial_cluster'] = clusters
                        df_model = df_model.merge(gdf_env[['health_area', 'spatial_cluster']], on='health_area', how='left')
                        
                        cluster_dummies = pd.get_dummies(df_model['spatial_cluster'], prefix='cluster')
                        df_model = pd.concat([df_model, cluster_dummies], axis=1)
                        
                        # Lag spatial robuste aux aires manquantes
                        spatial_lag_values = []
                        all_geo_areas = set(gdf_env['health_area'])
                        
                        for week in df_model['week_num'].unique():
                          df_week = (df_model[df_model['week_num'] == week]
                                     .sort_values('health_area')
                                     .reset_index(drop=True))
                          cases_week   = df_week['cases'].reset_index(drop=True)
                          common_areas = df_week['health_area'].tolist()
                        
                          # Sous‑ensemble pour lequel on a vraiment une géométrie
                          common_areas_ok = [a for a in common_areas if a in all_geo_areas]
                        
                          # Si aucune aire de cette semaine n’existe dans le shapefile → tout à 0
                          if not common_areas_ok:
                              spatial_lag_values.extend([0.0] * len(df_week))
                              continue
                        
                          gdf_week = gdf_env[gdf_env['health_area'].isin(common_areas_ok)].copy()
                        
                          # Forcer le même ordre que df_week sur les aires valides
                          gdf_aligned = (
                              gdf_week.set_index('health_area')
                                      .loc[common_areas_ok]
                                      .reset_index()
                          )
                        
                          # Cas “propre” : même longueur → comme avant
                          if len(gdf_aligned) == len(cases_week):
                              lag_values = calculate_spatial_lag(gdf_aligned, cases_week, k_neighbors)
                              spatial_lag_values.extend(lag_values.tolist())
                              continue
                        
                          # Cas où certaines aires du CSV n’ont pas de géométrie :
                          # on calcule le lag uniquement pour les aires valides puis on remplit 0 pour les autres.
                          cases_sub = (
                              df_week[df_week['health_area'].isin(common_areas_ok)]['cases']
                              .reset_index(drop=True)
                          )
                          if len(cases_sub) != len(gdf_aligned):
                              # Sécurité : si toujours incohérent → tout à 0 pour cette semaine
                              spatial_lag_values.extend([0.0] * len(df_week))
                              continue
                        
                          lag_sub = calculate_spatial_lag(gdf_aligned, cases_sub, k_neighbors)
                        
                          # Réinjection dans l’ordre original (0 pour les aires sans géométrie)
                          idx_sub = 0
                          for ha in df_week['health_area']:
                              if ha in common_areas_ok:
                                  spatial_lag_values.append(float(lag_sub[idx_sub]))
                                  idx_sub += 1
                              else:
                                  spatial_lag_values.append(0.0)
                        
                        df_model['spatial_lag'] = spatial_lag_values
                    progress_bar.progress(60)
                    
                    # 🧮 Coefficient d'ajustement population par aire
                    if "Pop_Totale" in df_model.columns and df_model["Pop_Totale"].notna().any():
                        mean_cases_by_area = df_model.groupby("health_area")["cases"].mean()
                        pop_by_area = df_model.groupby("health_area")["Pop_Totale"].first()
                    
                        incidence_by_area = (mean_cases_by_area / pop_by_area * 10000)
                        incidence_by_area = incidence_by_area.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                        if pop_by_area.sum() > 0:
                            global_incidence = mean_cases_by_area.sum() / pop_by_area.sum() * 10000
                            coef_ajustement = (incidence_by_area / global_incidence)
                            coef_ajustement = coef_ajustement.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                            coef_ajustement = coef_ajustement.clip(0.5, 2.0)
                        else:
                            coef_ajustement = pd.Series(1.0, index=incidence_by_area.index)
                    
                        df_model["coef_population"] = df_model["health_area"].map(coef_ajustement).fillna(1.0)
                    else:
                        df_model["coef_population"] = 1.0

                    # ÉTAPE 5 : Sélection features (60-70%)
                    status.text("🔧 5/6 : Sélection features...")
                    feature_cols = ['week_num']
                    feature_cols.extend([c for c in df_model.columns if 'sin_' in c or 'cos_' in c])
                    temporal = [c for c in df_model.columns if any(x in c for x in ['lag', 'ma_', 'std_', 'growth'])]
                    feature_cols.extend([c for c in temporal if df_model[c].dtype in ['int64', 'float64']])
                    feature_cols.extend(climate_features)
                    feature_cols.extend(static_env_cols)
                    
                    # ✅ CORRECTION : Ajouter spatial AVANT ACP, pas après
                    if use_spatial:
                        if 'spatial_lag' in df_model.columns:
                            feature_cols.append('spatial_lag')
                        feature_cols.extend([c for c in df_model.columns if c.startswith('cluster_')])
                    
                    # 🧮 Features population
                    pop_features = [
                        "Pop_Totale",
                        "Pop_Enfants_0_14",
                        "Densite_Pop",
                        "incidence_rate",
                        "child_risk",
                        "demo_pressure",
                        "coef_population",
                    ]
                    pop_features = [c for c in pop_features if c in df_model.columns]
                    feature_cols.extend(pop_features)
                    
                    # Nettoyage final
                    feature_cols = list(set([c for c in feature_cols if c in df_model.columns]))
                    
                    X = df_model[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
                    y = df_model["cases"].copy()
                    
                    # ACP
                    pca_info = None
                    if use_pca and len(feature_cols) > 10:
                        df_pca, pca_model, pca_scaler, pca_imputer, pca_info = perform_pca_analysis(
                            df_model, feature_cols, variance_threshold
                        )
                        X = df_pca
                        feature_cols = df_pca.columns.tolist()  # ✅ Mettre à jour feature_cols après ACP
                    progress_bar.progress(70)
                    
                    # ÉTAPE 6 : Entraînement (70-100%)
                    status.text("🎯 6/6 : Entraînement...")
                    from sklearn.ensemble import ExtraTreesRegressor
                    from sklearn.preprocessing import RobustScaler
                    
                    if model_choice == "GradientBoosting":
                        if use_pca:
                            model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
                        else:
                            model = Pipeline([
                                ("imputer", SimpleImputer(strategy="mean")),
                                ("scaler", RobustScaler()),
                                ("regressor", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42))
                            ])
                    elif model_choice == "ExtraTrees":
                        if use_pca:
                            model = ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_split=5, random_state=42, n_jobs=-1)
                        else:
                            model = Pipeline([
                                ("imputer", SimpleImputer(strategy="mean")),
                                ("regressor", ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_split=5, random_state=42, n_jobs=-1))
                            ])
                    else:  # RandomForest
                        if use_pca:
                            model = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=5, random_state=42, n_jobs=-1)
                        else:
                            model = Pipeline([
                                ("imputer", SimpleImputer(strategy="mean")),
                                ("regressor", RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=5, random_state=42, n_jobs=-1))
                            ])
                    
                    progress_bar.progress(75)
                    
                    # Validation croisée
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2', n_jobs=-1)
                    cv_mae = -cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
                    progress_bar.progress(85)
                    
                    # Entraînement final
                    model.fit(X, y)
                    df_model["predicted_cases"] = model.predict(X).clip(0).round().astype(int)
                    mae = mean_absolute_error(y, df_model["predicted_cases"])
                    rmse = np.sqrt(mean_squared_error(y, df_model["predicted_cases"]))
                    from sklearn.metrics import r2_score
                    r2 = r2_score(y, df_model["predicted_cases"])
                    progress_bar.progress(90)
                    
                    # Prédictions futures
                    status.text("🔮 Prédictions futures...")
                    max_week = df_model["week_num"].max()
                    future_rows = []
                    
                    for ha in df_model["health_area"].unique():
                        df_ha = df_model[df_model["health_area"] == ha].sort_values("week_num")
                        history = df_ha['cases'].tail(8).tolist()
                        
                        # Valeurs statiques
                        static_vals = {
                            col: df_ha.iloc[-1][col] 
                            for col in static_env_cols + (['spatial_cluster'] if use_spatial else [])
                            if col in df_ha.columns and not pd.isna(df_ha.iloc[-1][col])
                        }
                        
                        # Valeurs climat
                        climate_vals = {
                            col: df_ha.iloc[-1][col] 
                            for col in climate_features
                            if col in df_ha.columns and not pd.isna(df_ha.iloc[-1][col])
                        }
                        
                        for step in range(1, n_future_weeks + 1):
                            future_week = max_week + step
                            row = {"health_area": ha, "week_num": future_week}
                            
                            # ===== Saisonnalité =====
                            week_of_year = future_week % 52
                            row['sin_week'] = np.sin(2 * np.pi * week_of_year / 52)
                            row['cos_week'] = np.cos(2 * np.pi * week_of_year / 52)
                            
                            # ===== Features temporelles =====
                            if history:
                                row['cases_lag_1'] = history[-1]
                                if len(history) >= 2:
                                    row['cases_lag_2'] = history[-2]
                                    row['cases_ma_2'] = np.mean(history[-2:])
                                    row['growth_rate'] = (history[-1] - history[-2]) / (history[-2] + 1)
                                if len(history) >= 4:
                                    row['cases_lag_4'] = history[-4]
                                    row['cases_ma_4'] = np.mean(history[-4:])
                            
                            # ===== Features statiques =====
                            row.update(static_vals)
                            
                            # ===== Features climatiques =====
                            if climate_vals:
                                seasonal_factor = 1 + 0.15 * row['sin_week']
                                for var, val in climate_vals.items():
                                    row[var] = val * seasonal_factor
                            
                            # ===== ✅ CORRECTION : Features spatiales AVANT complétion =====
                            if use_spatial:
                                # Spatial lag
                                row['spatial_lag'] = history[-1] if history else 0
                                
                                # Cluster dummies
                                if 'spatial_cluster' in static_vals:
                                    for i in range(n_clusters):
                                        row[f'cluster_{i}'] = 1 if static_vals['spatial_cluster'] == i else 0
                            
                            # ===== ✅ CORRECTION : Préparer input SELON mode ACP =====
                            if use_pca and pca_info:
                                # AVEC ACP : Utiliser feature_names ORIGINALES (avant ACP)
                                feature_names_original = pca_info['feature_names']
                                
                                # Compléter features manquantes avec 0
                                for col in feature_names_original:
                                    if col not in row:
                                        row[col] = 0
                                
                                # Transformer
                                try:
                                    row_df = pd.DataFrame([row])[feature_names_original]
                                    row_imputed = pca_imputer.transform(row_df)
                                    row_scaled = pca_scaler.transform(row_imputed)
                                    row_pca = pca_model.transform(row_scaled)
                                    X_step = pd.DataFrame(
                                        row_pca, 
                                        columns=[f'PC{i+1}' for i in range(pca_info['n_components'])]
                                    )
                                except KeyError as e:
                                    missing_cols = [c for c in feature_names_original if c not in row]
                                    st.error(f"❌ Colonnes manquantes pour ACP : {missing_cols}")
                                    st.error(f"📋 Colonnes disponibles : {list(row.keys())}")
                                    raise
                            else:
                                # SANS ACP : Utiliser feature_cols directement
                                # Compléter features manquantes avec 0
                                for col in feature_cols:
                                    if col not in row:
                                        row[col] = 0
                                
                                try:
                                    X_step = pd.DataFrame([row])[feature_cols]
                                except KeyError as e:
                                    missing_cols = [c for c in feature_cols if c not in row]
                                    st.error(f"❌ Colonnes manquantes : {missing_cols}")
                                    st.error(f"📋 Colonnes disponibles : {list(row.keys())}")
                                    raise
                            
                            # ===== Prédiction =====
                            pred = max(0, round(model.predict(X_step)[0]))
                            
                            # Mise à jour historique
                            history.append(pred)
                            if len(history) > 8:
                                history.pop(0)
                            
                            row['predicted_cases'] = pred
                            future_rows.append(row)
                    
                    df_future = pd.DataFrame(future_rows)
                    progress_bar.progress(100)
                    status.text("✅ Terminé !")
                    
                    # Sauvegarder
                    st.session_state.model_results = {
                        'df_model': df_model, 'df_future': df_future,
                        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2, 'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std()},
                        'pca_info': pca_info, 'feature_cols': feature_cols
                    }
                    
                except Exception as e:
                    st.error(f" Erreur : {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # ========================================
            # AFFICHAGE RÉSULTATS
            # ========================================
            if st.session_state.model_results:
                st.markdown("---")
                st.markdown("## 📊 Résultats")
                
                metrics = st.session_state.model_results['metrics']
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📉 MAE", f"{metrics['mae']:.2f}")
                col2.metric("📊 RMSE", f"{metrics['rmse']:.2f}")
                col3.metric("🎯 r2", f"{metrics['r2']:.3f}")
                col4.metric("✅ r2 CV", f"{metrics['cv_r2_mean']:.3f}")
                
                # Interprétation
                r2, cv_r2 = metrics['r2'], metrics['cv_r2_mean']
                if r2 > 0.85 and cv_r2> 0.80:
                    st.success(f"✅ **Excellent** : r2={r2:.3f}, CV={cv_r2:.3f} - Fiable pour décisions stratégiques")
                elif r2 > 0.70 and cv_r2> 0.65:
                    st.info(f"🟡 **Bon** : r2={r2:.3f}, CV={cv_r2:.3f} - OK pour alertes précoces")
                else:
                    st.warning(f"⚠️ **Moyen** : r2={r2:.3f}, CV={cv_r2:.3f} - Activer climat / vérifier données")
                
                # Prédictions
                st.markdown("### 🔮 Prédictions")
                df_future = st.session_state.model_results['df_future']
                df_display = df_future[['health_area', 'week_num', 'predicted_cases']].copy()
                if "period" in df_cases.columns and "year" in df_cases.columns:
                    max_week_num = df_cases.groupby(
                        df_cases["period"]
                    ).ngroups
                    last_year  = int(df_cases.sort_values("period").iloc[-1]["year"])
                    last_week  = int(df_cases.sort_values("period").iloc[-1]["week_"])
                    base_num   = df_cases.sort_values("period")["week_"].max()
                
                    def week_num_to_label(wn):
                        offset = wn - df_cases["week_"].max()  # pas depuis 0
                        y, w = last_year, last_week
                        for _ in range(max(0, offset)):
                            w += 1
                            if w > 52:
                                w = 1
                                y += 1
                        return f"{y}-S{str(w).zfill(2)}"
                
                    df_display['week_num'] = df_display['week_num'].apply(week_num_to_label)
                else:
                    df_display['week_num'] = df_display['week_num'].apply(lambda x: f"S{x}")
                
                df_display.columns = ['Aire', 'Semaine', 'Cas Prédits']
                st.dataframe(df_display.sort_values('Cas Prédits', ascending=False).head(30))
                
                # Heatmap
                top15 = df_future.groupby('health_area')['predicted_cases'].sum().sort_values(ascending=False).head(15).index
                pivot = df_future[df_future['health_area'].isin(top15)].pivot_table(
                    index='health_area', columns='week_num', values='predicted_cases', aggfunc='sum'
                )
                
                fig = px.imshow(
                    pivot, labels=dict(x="Semaine", y="Aire", color="Cas"),
                    x=[f"S{c}" for c in pivot.columns], y=pivot.index,
                    color_continuous_scale='Reds', title="Top 15 Aires à Risque"
                )
                fig.update_layout(height=700, width=None)
                st.plotly_chart(fig, width='stretch')
                # Après la heatmap (ligne ~1580)
                st.markdown("---")
                st.markdown("### 🚨 Zones à Risque Élevé")
                
                # Calculer seuil
                alert_threshold = 75  # Vous pouvez le remettre en slider si besoin
                threshold_value = df_future['predicted_cases'].quantile(alert_threshold / 100)
                
                # Filtrer zones à risque
                df_alerts = df_future[df_future['predicted_cases'] > threshold_value].copy()
                df_alerts = df_alerts.groupby('health_area')['predicted_cases'].sum().sort_values(ascending=False)
                
                if not df_alerts.empty:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(
                            df_alerts.reset_index().rename(columns={'health_area': 'Aire de Santé', 'predicted_cases': 'Cas Totaux Prévus'}),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.metric("🚨 Zones à Risque", len(df_alerts))
                        st.metric("📊 Seuil", f"{threshold_value:.0f} cas")
                        st.info(f"Alertes pour le top {100 - alert_threshold}% des prédictions (au-dessus du {alert_threshold}e percentile)")
                else:
                    st.success("✅ Aucune zone au-dessus du seuil d'alerte")               
# ============================================================
# TAB 4 : CARTOGRAPHIE DES PRÉDICTIONS
# ============================================================

with tab4:  # Carte Prédictions
    create_prediction_map_tab(
        gdf_health=gdf_health, 
        model_results=st.session_state.get('model_results')
    )
# ============================================================
# TAB 4 – ANALYSE AVANCÉE (VERSION CORRIGÉE)
# ============================================================

with tab5:
    st.subheader("📈 Analyse de Corrélation")
    
    if df_cases is not None and gdf_health is not None:
        # ✅ CORRECTION 1 : Agrégation de base TOUJOURS disponible
        df_agg = df_cases.groupby('health_area', as_index=False).agg({'cases': 'sum', 'deaths': 'sum'})
        df_corr = df_agg.copy()
        
        # ✅ NOUVEAU : Ajouter population à df_corr
        cache_key_pop = f"enrichi_{iso3pays}" if iso3pays else "enrichi_upload"
        if st.session_state.get(cache_key_pop) is not None and not st.session_state[cache_key_pop].empty:
            df_pop = st.session_state[cache_key_pop]
            df_corr = df_corr.merge(df_pop, on='health_area', how='left')
            st.info("✅ Données population mergées dans l'analyse")
        else:
            st.warning("⚠️ Population non disponible pour cette analyse")
        
        numeric_cols = ['cases', 'deaths']

        
        # ✅ CORRECTION 2 : Ajouter climat SI DISPONIBLE
        if st.session_state.df_climate_aggregated is not None:
            st.info("🌡️ Intégration données climatiques...")
            
            df_clim_avg = st.session_state.df_climate_aggregated.groupby('health_area').agg({
                col: 'mean' for col in st.session_state.df_climate_aggregated.columns
                if col.endswith('_api') and '_min' not in col and '_max' not in col
            }).reset_index()
            
            df_corr = df_corr.merge(df_clim_avg, on='health_area', how='left')
            
            # Ajouter colonnes climat à numeric_cols
            for col in df_clim_avg.columns:
                if col != 'health_area' and col in df_corr.columns:
                    numeric_cols.append(col)
            
            st.success(f"✅ {len([c for c in numeric_cols if c.endswith('_api')])} variables climat ajoutées")
        else:
            st.info("ℹ️ Pas de données climat - Analyse épidémiologique uniquement")
        
        # ✅ CORRECTION 3 : Ajouter environnement SI DISPONIBLE (OPTIONNEL)
        env_cols_added = []
        
        if st.session_state.flood_raster is not None or \
           st.session_state.elevation_raster is not None or \
           (st.session_state.rivers_gdf is not None and not st.session_state.rivers_gdf.empty):
            
            st.info("🌍 Intégration données environnementales...")
            
            gdf_env = gdf_health.copy()
            
            if st.session_state.flood_raster is not None:
                gdf_env["flood_mean"] = extract_raster_statistics(gdf_env, st.session_state.flood_raster, 'mean')
                env_cols_added.append('flood_mean')
            
            if st.session_state.elevation_raster is not None:
                gdf_env["elevation_mean"] = extract_raster_statistics(gdf_env, st.session_state.elevation_raster, 'mean')
                env_cols_added.append('elevation_mean')
            
            if st.session_state.rivers_gdf is not None and not st.session_state.rivers_gdf.empty:
                gdf_env["dist_river"] = gdf_env.centroid.apply(
                    lambda x: distance_to_nearest_line(x, st.session_state.rivers_gdf)
                )
                env_cols_added.append('dist_river')
            
            # Filtrer colonnes existantes
            env_cols_valid = [c for c in env_cols_added if c in gdf_env.columns]
            
            if env_cols_valid:
                df_corr = df_corr.merge(gdf_env[['health_area'] + env_cols_valid], on='health_area', how='left')
                numeric_cols.extend(env_cols_valid)
                st.success(f"✅ {len(env_cols_valid)} variables environnement ajoutées")
        
        # ✅ CORRECTION 4 : Vérifier qu'on a au moins 3 colonnes numériques
        # Filtrer colonnes réellement numériques et présentes
        numeric_cols = list(set([
            c for c in numeric_cols 
            if c in df_corr.columns and df_corr[c].dtype in ['int64', 'float64', 'int32', 'float32']
        ]))
        
        st.markdown(f"### 📊 Variables disponibles pour l'analyse : **{len(numeric_cols)}**")
        
        with st.expander("📋 Liste des variables"):
            var_categories = {
                'Épidémiologiques': [c for c in numeric_cols if c in ['cases', 'deaths']],
                'Climat API': [c for c in numeric_cols if c.endswith('_api')],
                'Environnement': [c for c in numeric_cols if c in env_cols_added]
            }
            
            for cat, cols in var_categories.items():
                if cols:
                    st.write(f"**{cat}** ({len(cols)}) : {', '.join(cols)}")
        
        # ✅ CORRECTION 5 : Afficher matrice SI au moins 2 variables
        if len(numeric_cols) >= 2:
            st.markdown("---")
            st.markdown("### 🔥 Matrice de Corrélation")
            
            corr_matrix = df_corr[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                ax=ax,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title("Matrice de Corrélation Complète", fontsize=16, fontweight='bold')
            st.pyplot(fig)
            
            # ✅ CORRECTION 6 : Analyse corrélations avec cas
            if 'cases' in corr_matrix.columns and len(corr_matrix) > 1:
                st.markdown("---")
                st.markdown("### 🔍 Corrélations Significatives avec les Cas")
                
                corr_with_cases = corr_matrix['cases'].drop('cases').sort_values(ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Corrélations Positives")
                    positive_corr = corr_with_cases[corr_with_cases > 0.2]
                    
                    if not positive_corr.empty:
                        for var, corr_val in positive_corr.items():
                            # Emoji selon type
                            if '_api' in var:
                                emoji = "🌡️" if 'temp' in var else "🌧️" if 'precip' in var else "💧"
                            else:
                                emoji = "🌍"
                            
                            st.write(f"{emoji} **{var}** : {corr_val:.3f}")
                            st.progress(float(abs(corr_val)))
                    else:
                        st.info("Aucune corrélation positive forte (> 0.2)")
                
                with col2:
                    st.markdown("#### 📉 Corrélations Négatives")
                    negative_corr = corr_with_cases[corr_with_cases < -0.2]
                    
                    if not negative_corr.empty:
                        for var, corr_val in negative_corr.items():
                            if '_api' in var:
                                emoji = "🌡️" if 'temp' in var else "🌧️" if 'precip' in var else "💧"
                            else:
                                emoji = "🌍"
                            
                            st.write(f"{emoji} **{var}** : {corr_val:.3f}")
                            st.progress(float(abs(corr_val)))
                    else:
                        st.info("Aucune corrélation négative forte (< -0.2)")
                
                # ✅ CORRECTION 7 : Scatter plots si corrélations fortes
                st.markdown("---")
                st.markdown("### 📊 Graphiques de Corrélation")
                
                strong_corr = corr_with_cases[abs(corr_with_cases) > 0.3]
                
                if not strong_corr.empty:
                    # Limiter à 6 graphiques max
                    n_plots = min(6, len(strong_corr))
                    cols = st.columns(min(3, n_plots))
                    
                    for i, (var, corr_val) in enumerate(strong_corr.head(n_plots).items()):
                        with cols[i % 3]:
                            # Nettoyer données
                            df_plot = df_corr[[var, 'cases']].dropna()
                            
                            if len(df_plot) > 3:
                                fig_scatter = px.scatter(
                                    df_plot,
                                    x=var,
                                    y='cases',
                                    trendline='ols',
                                    title=f"{var} vs Cas<br>(r={corr_val:.3f})",
                                    labels={var: var, 'cases': 'Cas'}
                                )
                                fig_scatter.update_layout(height=300)
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            else:
                                st.warning(f"⚠️ Pas assez de données pour {var}")
                else:
                    st.info("ℹ️ Aucune corrélation forte (|r| > 0.3) détectée")
                
                # ✅ CORRECTION 8 : Résumé interprétatif
                st.markdown("---")
                st.markdown("### 💡 Interprétation")
                
                # Identifier variable la plus corrélée (hors deaths)
                corr_wo_deaths = corr_with_cases.drop('deaths', errors='ignore')
                
                if not corr_wo_deaths.empty:
                    max_corr_var = corr_wo_deaths.abs().idxmax()
                    max_corr_val = corr_wo_deaths[max_corr_var]
                    
                    if abs(max_corr_val) > 0.3:
                        direction = "positive" if max_corr_val > 0 else "négative"
                        
                        st.markdown(f"""
                        <div class="info-card">
                        <h4>🎯 Variable Clé</h4>
                        <p><b>{max_corr_var}</b> présente la corrélation la plus forte avec les cas de paludisme 
                        (r = {max_corr_val:.3f}, corrélation {direction}).</p>
                        
                        <p><b>Implication :</b> 
                        {'Une augmentation' if max_corr_val > 0 else 'Une diminution'} de cette variable 
                        est {'associée à plus' if max_corr_val > 0 else 'associée à moins'} de cas.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ Aucune variable ne présente de corrélation forte avec les cas")
                # 🧮 Coefficient d'ajustement population (par aire)
                if "Pop_Totale" in df_corr.columns and df_corr["Pop_Totale"].notna().any():  # ✅
                    mean_cases_by_area = df_corr.groupby("health_area")["cases"].mean()
                    pop_by_area = df_corr.groupby("health_area")["Pop_Totale"].first()
                
                    incidence_by_area = (mean_cases_by_area / pop_by_area * 10000).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                    global_incidence = (mean_cases_by_area.sum() / pop_by_area.sum() * 10000) if pop_by_area.sum() > 0 else 0
                    if global_incidence > 0:
                        coef_ajustement = (incidence_by_area / global_incidence).replace([np.inf, -np.inf], np.nan).fillna(1.0)
                        coef_ajustement = coef_ajustement.clip(0.5, 2.0)
                    else:
                        coef_ajustement = pd.Series(1.0, index=incidence_by_area.index)
                
                    df_corr["coef_population"] = df_corr["health_area"].map(coef_ajustement).fillna(1.0)  # ✅
                    
                    st.markdown("---")
                    st.markdown("### 👥 Coefficient d'Ajustement Population")
                    
                    st.info(f"""
                    ✅ **Coefficient calculé pour {len(coef_ajustement)} aires de santé**
                    
                    Ce coefficient ajuste les prédictions en fonction du risque démographique relatif de chaque aire.
                    - Valeur > 1 : Zone à risque plus élevé que la moyenne
                    - Valeur < 1 : Zone à risque plus faible que la moyenne
                    """)
                    # Afficher tableau des coefficients
                    coef_df = pd.DataFrame({
                        'Aire de santé': coef_ajustement.index,
                        'Coefficient': coef_ajustement.values,
                        'Interprétation': ['Risque élevé' if c > 1.2 else 'Risque faible' if c < 0.8 else 'Risque moyen' 
                                           for c in coef_ajustement.values]
                    }).sort_values('Coefficient', ascending=False)
                    
                    st.dataframe(coef_df, use_container_width=True)
                    
                else:
                    df_corr["coef_population"] = 1.0  # ✅
                    st.info("ℹ️ Coefficient population non calculé (données Pop_Totale manquantes)")    


                # Conseils données manquantes
                if st.session_state.df_climate_aggregated is None:
                    st.markdown("""
                    <div class="warning-box">
                    <h4>💡 Conseil</h4>
                    <p>Activez l'<b>API Climat</b> (gratuit) pour enrichir l'analyse avec température, 
                    précipitations et humidité. Cela peut révéler des corrélations importantes !</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.warning("⚠️ Pas assez de variables pour analyser les corrélations avec 'cases'")
        
        else:
            st.warning("""
            ⚠️ **Pas assez de données pour l'analyse de corrélation**
            
            Il faut au minimum 2 variables numériques. Actuellement : **{} variable(s)**.
            
            💡 **Solutions** :
            - Activez l'API Climat (gratuit) pour ajouter température, précipitations, humidité
            - Ajoutez des rasters environnementaux (inondation, élévation)
            - Vérifiez que vos données contiennent bien les colonnes 'cases' et 'deaths'
            """.format(len(numeric_cols)))
    
    else:
        st.info("ℹ️ Chargez d'abord les aires de santé et les cas pour l'analyse de corrélation")
# ============================================================
# TAB 7 – VALIDATION RÉTROSPECTIVE
# ============================================================
with tab7:
    create_validation_tab(
        df_cases=st.session_state.df_cases,
        gdf_health=st.session_state.gdf_health,
        model_results=st.session_state.get("model_results"),
    )       
# ============================================================
# TAB 6 – EXPORT
# ============================================================
with tab6:
    st.subheader("📥 Export des Données")
    
    st.markdown('''
    <div class="info-box">
    📦 <b>Exportez toutes vos données</b> : aires de santé, cas, climat, environnement, prédictions
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Données Géographiques")
        
        # 1. Aires de santé (GeoJSON)
        if st.session_state.gdf_health is not None:
            gdf_export = st.session_state.gdf_health.copy()
            geojson_str = gdf_export.to_json()
            st.download_button(
                "🗺️ Aires de Santé (GeoJSON)",
                geojson_str,
                "aires_sante.geojson",
                "application/json",
                help="Carte des zones géographiques"
            )
        
        # 2. Rivières (si disponible)
        if st.session_state.rivers_gdf is not None:
            rivers_json = st.session_state.rivers_gdf.to_json()
            st.download_button(
                "🏞️ Rivières (GeoJSON)",
                rivers_json,
                "rivieres.geojson",
                "application/json"
            )
    
    with col2:
        st.markdown("### 📊 Données Épidémiologiques")
        
        # 3. Cas hebdomadaires
        if st.session_state.df_cases is not None:
            csv_cases = st.session_state.df_cases.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📋 Cas Hebdomadaires (CSV)",
                csv_cases,
                "cas_hebdomadaires.csv",
                "text/csv",
                help="Nombre de cas et décès par zone et semaine"
            )
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 🌡️ Données Climatiques")
        
        # 4. Données climat API
        if st.session_state.df_climate_aggregated is not None:
            csv_climate = st.session_state.df_climate_aggregated.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "🌦️ Données Climat (CSV)",
                csv_climate,
                "donnees_climat.csv",
                "text/csv",
                help="Température, précipitations, humidité par zone et semaine"
            )
            
            st.info(f"✅ {len(st.session_state.df_climate_aggregated)} enregistrements")
    
    with col4:
        st.markdown("### 🤖 Résultats Modélisation")
        
        # 5. Prédictions
        if st.session_state.model_results is not None:
            df_future = st.session_state.model_results.get('df_future')
            
            if df_future is not None:
                csv_pred = df_future.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "🔮 Prédictions Futures (CSV)",
                    csv_pred,
                    "predictions.csv",
                    "text/csv",
                    help="Cas prévus par zone et semaine"
                )
                
                st.info(f"✅ {len(df_future)} prédictions")
    
    st.markdown("---")
    
    # 6. Export combiné
    st.markdown("### 📦 Export Complet")
    
    if st.button("📥 Générer Export Complet (ZIP)", type="primary"):
        import zipfile
        from io import BytesIO
        
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # Aires de santé
            if st.session_state.gdf_health is not None:
                zip_file.writestr("aires_sante.geojson", st.session_state.gdf_health.to_json())
            
            # Cas
            if st.session_state.df_cases is not None:
                zip_file.writestr("cas_hebdomadaires.csv", 
                                st.session_state.df_cases.to_csv(index=False, encoding='utf-8-sig'))
            
            # Climat
            if st.session_state.df_climate_aggregated is not None:
                zip_file.writestr("donnees_climat.csv",
                                st.session_state.df_climate_aggregated.to_csv(index=False, encoding='utf-8-sig'))
            
            # Prédictions
            if st.session_state.model_results is not None:
                df_future = st.session_state.model_results.get('df_future')
                if df_future is not None:
                    zip_file.writestr("predictions.csv",
                                    df_future.to_csv(index=False, encoding='utf-8-sig'))
            
            # Rivières
            if st.session_state.rivers_gdf is not None:
                zip_file.writestr("rivieres.geojson", st.session_state.rivers_gdf.to_json())
        
        zip_buffer.seek(0)
        
        st.download_button(
            "💾 Télécharger Export Complet (ZIP)",
            zip_buffer,
            "epipalu_export_complet.zip",
            "application/zip"
        )
    
    # ========================================
    # SECTION 7 : SUPPORT ET CONTACT
    # ========================================
    st.markdown("""
    <div class="section-card">
        <h2 style="margin:0; color:white; text-align:center;">📞 Besoin d'aide ?</h2>
        <br>
        <div style="text-align:center; font-size:1.1rem;">
            <p><b>Contact Support Technique</b></p>
            <p>📧 Email : <a href="mailto:youssoupha.mbodji@example.com" style="color:#FFD700;">youssoupha.mbodji@example.com</a></p>
            <p>💬 Questions fréquentes : <a href="#" style="color:#FFD700;">FAQ (à venir)</a></p>
            <p>📖 Documentation complète : <a href="#" style="color:#FFD700;">Manuel utilisateur</a></p>
        </div>
        <br>
        <p style="text-align:center; font-size:0.9rem; opacity:0.9;">
            Version 3.0 | Développé par <b>Youssoupha MBODJI</b><br>
            © 2025 - Licence Open Source MIT
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bouton retour rapide
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🏠 Retour au Tableau de Bord", use_container_width=True, type="primary"):
            st.info("💡 Cliquez sur l'onglet 'Dashboard' en haut de la page")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><b>🦟 Application de Surveillance du Paludisme</b></p>
    <p>Version 1.0 | Développé avec | Python • Streamlit • GeoPandas • Scikit-learn par Youssoupha MBODJI</p>
</div>
""", unsafe_allow_html=True)































































































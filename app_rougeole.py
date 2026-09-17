# ============================================================
# APP SURVEILLANCE & PRÉDICTION ROUGEOLE - VERSION 3.0
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import epimodel as em
import epi_app_bridge
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
import ee
import json
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium, folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import zipfile
import tempfile
import os
from shapely.geometry import shape
from validation_tab_rougeole import create_validation_tab_rougeole
import warnings
warnings.filterwarnings('ignore')

# CSS personnalisé
st.markdown("""
<style>
.model-hint { background:#f0f7ff; border-left:4px solid #1976d2; padding:.8rem 1rem; border-radius:4px; margin:.5rem 0; font-size:.9rem; }
.weight-box { background:#fff8e1; border-left:4px solid #f9a825; padding:.8rem 1rem; border-radius:4px; margin:.5rem 0; }
.info-box   { background:#fef9f9; border-left:4px solid #e74c3c; padding:1rem; border-radius:6px; margin:.5rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🦠 Plateforme de Surveillance et Prédiction - Rougeole")
st.markdown("### Analyse épidémiologique et modélisation prédictive par semaines épidémiologiques")

# Mapping pays ISO3
PAYS_ISO3_MAP = {
    "Niger": "ner",
    "Burkina Faso": "bfa",
    "Mali": "mli",
    "Mauritanie": "mrt"
}

# ── GEE ───────────────────────────────────────────────────────
@st.cache_resource
def init_gee():
    """Initialise Google Earth Engine.

    Les erreurs sont affichées au lieu d'être avalées : un échec silencieux
    laisse croire que GEE fonctionne alors que les covariables sont absentes.
    """
    try:
        key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key_dict["client_email"], key_data=json.dumps(key_dict))
        ee.Initialize(credentials)
        st.sidebar.success("✅ GEE initialisé (Service Account)")
        return True
    except Exception as e:
        st.sidebar.warning(f"⚠️ Service Account échec : {str(e)[:100]}")

    try:
        ee.Initialize()
        st.sidebar.success("✅ GEE initialisé (Défaut)")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ GEE échec total : {str(e)[:100]}")
        return False

gee_ok = init_gee()
    st.sidebar.success("✓ GEE connecté")

# ── Session state ─────────────────────────────────────────────
if 'pays_precedent' not in st.session_state:
    st.session_state.pays_precedent = None
if 'sa_gdf_cache' not in st.session_state:
    st.session_state.sa_gdf_cache = None
if 'prediction_rougeole_lancee' not in st.session_state:
    st.session_state.prediction_rougeole_lancee = False
if 'enrichi_ner' not in st.session_state:
    st.session_state['enrichi_ner'] = None
if 'enrichi_bfa' not in st.session_state:
    st.session_state['enrichi_bfa'] = None
if 'enrichi_mli' not in st.session_state:
    st.session_state['enrichi_mli'] = None
if 'enrichi_mrt' not in st.session_state:
    st.session_state['enrichi_mrt'] = None
if 'enrichi_upload' not in st.session_state:
    st.session_state['enrichi_upload'] = None
if 'upload_file_name_cache' not in st.session_state:
    st.session_state['upload_file_name_cache'] = None
# ============================================================
# CORRECTION 1 & 2 : MAPPING COLONNES ROBUSTE + SÉPARATEUR CSV
# ============================================================

COLONNES_MAPPING = {
    "Aire_Sante": [
        "Aire_Sante","aire_sante","health_area","HEALTH_AREA","name_fr","NAME",
        "nom","NOM","aire de sante","aire_de_sante","zone_sante","district",
        "area","localite","locality","nom_aire","nom aire","aire","fosa",
        "nom_fosa","fosa_name","Aire_de_Sante","AIRE_SANTE"
    ],
    "Date_Debut_Eruption": [
        "Date_Debut_Eruption","date_debut_eruption","Date_Debut","date_onset",
        "Date_Onset","symptom_onset","date_eruption","Date_Eruption",
        "date_debut","DateDebut","DATE_DEBUT"
    ],
    "Date_Notification": [
        "Date_Notification","date_notification","Date_Notif","date_notif",
        "notification_date","DateNotif","DATE_NOTIFICATION"
    ],
    "ID_Cas": [
        "ID_Cas","id_cas","ID","id","Case_ID","case_id","ID_cas",
        "identifiant","Identifiant","IDCAS","id_case"
    ],
    "Age_Mois": [
        "Age_Mois","age_mois","Age","age","AGE","Age_Months","age_months",
        "age_en_mois","Age_En_Mois","AGE_MOIS"
    ],
    "Statut_Vaccinal": [
        "Statut_Vaccinal","statut_vaccinal","Vaccin","vaccin",
        "Vaccination_Status","vaccination_status","Vacc_Statut",
        "statut_vaccination","Statut_Vaccination","STATUT_VACCINAL",
        "vaccinated","Vaccinated","non_vaccine","Non_Vaccine"
    ],
    "Sexe": ["Sexe","sexe","Sex","sex","Gender","gender","SEXE","SEX"],
    "Issue": [
        "Issue","issue","Outcome","outcome","OUTCOME","issue_cas",
        "Issue_Cas","resultat","Resultat"
    ],
    "Semaine_Epi": [
        "Semaine_Epi","semaine_epi","Semaine_epi","SEMAINE_EPI",
        "semaine","Semaine","SEMAINE","week","Week","WEEK",
        "epi_week","Epi_Week","EPI_WEEK","epiweek","Epiweek",
        "se","SE","s_epi","S_epi","sem_epi","Sem_Epi",
        "semaine_epidemiologique","Semaine_Epidemiologique",
        "epi week","Epi Week","epid_week","Epid_Week",
        "week_number","Week_Number","no_semaine","No_Semaine",
        "numero_semaine","Numero_Semaine","n_semaine","N_Semaine",
        "wk","WK","sw","SW"
    ],
    "Annee": [
        "Annee","annee","ANNEE","année","Année",
        "year","Year","YEAR","an","An","AN",
        "yr","Yr","annee_epi","Annee_Epi","epi_year","Epi_Year",
        "epiyear","EpiYear","annee_epidemiologique","Annee_Epidemiologique"
    ],
    "Cas_Total": [
        "Cas_Total","cas_total","CAS_TOTAL","Cas","cas","CAS",
        "cases","Cases","CASES","nb_cas","Nb_Cas","NB_CAS",
        "nombre_cas","Nombre_Cas","NOMBRE_CAS","nbcas","nbre_cas",
        "count","Count","total_cas","Total_Cas","confirmed","Confirmed"
    ],
    "Deces": [
        "Deces","deces","DECES","décès","Décès",
        "deaths","Deaths","DEATHS","nb_deces","Nb_Deces",
        "mort","morts","dead","Dead","nb_morts"
    ],
}

SEP_CANDIDATES = [",", ";", "\t", "|"]

def detect_separator(uploaded_file) -> str:
    raw = uploaded_file.read(4096).decode("utf-8", errors="ignore")
    uploaded_file.seek(0)
    lines = [l for l in raw.split("\n")[:6] if l.strip()]
    scores = {}
    for sep in SEP_CANDIDATES:
        counts = [line.count(sep) for line in lines]
        scores[sep] = sum(counts) / max(len(counts), 1)
    best = max(scores, key=scores.get)
    return best if scores.get(best, 0) > 0 else ","

def normaliser_colonnes(dataframe, mapping):
    import unicodedata
    def _norm(s):
        s = str(s).strip().lower()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.replace(" ", "_").replace("-", "_")
    norm_existing = {_norm(c): c for c in dataframe.columns}
    rename_dict = {}
    for col_standard, col_possibles in mapping.items():
        if col_standard in dataframe.columns:
            continue
        for col_possible in col_possibles:
            nc = _norm(col_possible)
            if nc in norm_existing and norm_existing[nc] not in rename_dict.values():
                rename_dict[norm_existing[nc]] = col_standard
                break
    if rename_dict:
        dataframe = dataframe.rename(columns=rename_dict)
    return dataframe

def semaine_vers_date(annee: int, semaine: int) -> datetime:
    try:
        semaine = int(max(1, min(52, semaine)))
        annee = int(annee)
        return datetime.strptime(f"{annee}-W{semaine:02d}-1", "%G-W%V-%u")
    except Exception:
        try:
            return datetime(int(annee), 1, 1) + timedelta(weeks=int(semaine) - 1)
        except Exception:
            return datetime(2020, 1, 1)
def fuzzy_match_aire(nom_csv: str, liste_aires: list, seuil: int = 70) -> str | None:
    """
    Retourne le nom d'aire le plus proche dans liste_aires,
    ou None si le score est inférieur à seuil.
    """
    from difflib import SequenceMatcher
    import unicodedata

    def _norm(s):
        s = str(s).strip().lower()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.replace("-", " ").replace("_", " ")

    nom_n = _norm(nom_csv)
    best_score = 0
    best_match = None

    for aire in liste_aires:
        aire_n = _norm(aire)
        score = SequenceMatcher(None, nom_n, aire_n).ratio() * 100
        if score > best_score:
            best_score = score
            best_match = aire

    return best_match if best_score >= seuil else None
def mapper_aires_csv(df: pd.DataFrame, col_aire: str, liste_aires: list, seuil: int = 70) -> pd.DataFrame:
    mapping_cache = {}
    for nom in df[col_aire].unique():
        if nom not in mapping_cache:
            match = fuzzy_match_aire(str(nom), liste_aires, seuil=seuil)
            mapping_cache[nom] = match

    df = df.copy()
    df["health_area"] = df[col_aire].map(mapping_cache)

    total = len(mapping_cache)
    matches = sum(1 for v in mapping_cache.values() if v is not None)
    non_matches = [k for k, v in mapping_cache.items() if v is None]

    # ✅ toast() → disparaît automatiquement (pas permanent)
    if matches == total:
        st.toast(f"✅ Correspondance CSV → aires : {matches}/{total} noms reconnus", icon="✅")
    elif matches > 0:
        st.toast(
            f"⚠️ Correspondance partielle : {matches}/{total} reconnus. "
            f"Non reconnus : {non_matches[:3]}{'...' if len(non_matches) > 3 else ''}",
            icon="⚠️"
        )
    else:
        st.toast(
            f"❌ Aucune correspondance ({total} noms testés). Vérifiez la colonne aire.",
            icon="❌"
        )

    return df    
# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("📂 Configuration de l'Analyse")

# Mode démo
st.sidebar.subheader("🎯 Mode d'utilisation")
mode_demo = st.sidebar.radio(
    "Choisissez votre mode",
    ["📊 Données réelles", "🧪 Mode démo (données simulées)"],
    help="Mode démo : génère automatiquement des données fictives pour tester l'application"
)

# Aires de santé
st.sidebar.subheader("🗺️ Aires de Santé")
option_aire = st.sidebar.radio(
    "Source des données géographiques",
    ["Fichier local (ao_hlthArea.zip)", "Upload personnalisé"],
    key='option_aire'
)

pays_selectionne = None
iso3_pays = None
if option_aire == "Fichier local (ao_hlthArea.zip)":
    pays_selectionne = st.sidebar.selectbox(
        "🌍 Sélectionner le pays",
        list(PAYS_ISO3_MAP.keys()),
        key='pays_select'
    )
    if option_aire == "Fichier local (ao_hlthArea.zip)" and pays_selectionne:
        iso3_pays = PAYS_ISO3_MAP[pays_selectionne]
    if st.session_state.pays_precedent != pays_selectionne:
        st.session_state.pays_precedent = pays_selectionne
        st.session_state.sa_gdf_cache = None
        for _k in ['enrichi_ner', 'enrichi_bfa', 'enrichi_mli', 'enrichi_mrt', 'enrichi_upload']:
            st.session_state[_k] = None

upload_file = None
if option_aire == "Upload personnalisé":
    upload_file = st.sidebar.file_uploader(
        "Charger un fichier géographique",
        type=["shp", "geojson", "zip"],
        help="Format : Shapefile ou GeoJSON avec colonnes 'iso3' et 'health_area'"
    )

# Données épidémiologiques
st.sidebar.subheader("📊 Données Épidémiologiques")
if mode_demo == "🧪 Mode démo (données simulées)":
    linelist_file = None
    vaccination_file = None
    st.sidebar.info("📊 Mode démo activé - Données simulées")
else:
    linelist_file = st.sidebar.file_uploader(
        "📋 Linelists rougeole (CSV)",
        type=["csv"],
        help="Format : health_area, Semaine_Epi, Annee, Cas_Total OU Date_Debut_Eruption, Aire_Sante..."
    )
    vaccination_file = st.sidebar.file_uploader(
        "💉 Couverture vaccinale (CSV - optionnel)",
        type=["csv"],
        help="Format : health_area, Taux_Vaccination (en %)"
    )

# ── CORRECTION 3 : Filtres temporels dynamiques ───────────────
# (les multiselect sont construits APRÈS chargement des données,
#  voir Partie 2 — ici on initialise juste les valeurs par défaut)

# Paramètres de prédiction
st.sidebar.subheader("🔮 Paramètres de Prédiction")
pred_mois = st.sidebar.slider(
    "Période de prédiction (mois)",
    min_value=1, max_value=12, value=3,
    help="Nombre de mois à prédire après la dernière semaine de données"
)
n_weeks_pred = pred_mois * 4
st.sidebar.info(f"📆 Prédiction sur **{n_weeks_pred} semaines épidémiologiques** (~{pred_mois} mois)")

# Choix du modèle
st.sidebar.subheader("🤖 Modèle de Prédiction")
_dispo_rougeole = em.available_models()
_labels_rougeole = {m: em.MODEL_REGISTRY[m]["label"] for m in _dispo_rougeole}
modele_choisi = st.sidebar.selectbox(
    "Choisissez votre algorithme",
    _dispo_rougeole,
    index=_dispo_rougeole.index("XGBoost") if "XGBoost" in _dispo_rougeole else 0,
    format_func=lambda m: _labels_rougeole.get(m, m),
    help="Sélectionnez l'algorithme de machine learning pour la prédiction",
    key="rougeole_algo"
)
st.sidebar.markdown(
    f'<div class="model-hint">{em.describe_model(modele_choisi)}</div>',
    unsafe_allow_html=True)

objectif_rougeole = st.sidebar.selectbox(
    "Objectif de perte",
    ["Poisson (comptages — recommandé)", "Erreur quadratique (historique)"],
    help="Les cas de rougeole sont des comptages : leur variance croît avec la "
         "moyenne. L'objectif Poisson est mieux spécifié.",
    key="rougeole_objectif"
)

# ── MODE EXPERT — Importance des variables (RESTAURÉ) ─────────
st.sidebar.subheader("⚖️ Importance des Variables")

mode_importance = st.sidebar.radio(
    "Mode de pondération",
    ["🤖 Automatique (ML)", "👨‍⚕️ Manuel (Expert)"],
    help="Automatique : calculé par le modèle ML | Manuel : poids définis par expertise épidémiologique"
)

poids_manuels = {}
poids_normalises = {}

if mode_importance == "👨‍⚕️ Manuel (Expert)":
    with st.sidebar.expander("⚙️ Configurer les poids", expanded=True):
        st.markdown("**Définissez l'importance de chaque groupe de variables**")
        st.caption("Les poids seront automatiquement normalisés pour totaliser 100%")

        poids_manuels["Historique_Cas"] = st.slider(
            "📈 Historique des cas (lags)", 0, 100, 40, step=5,
            help="Importance des cas passés (4 dernières semaines)"
        )
        poids_manuels["Vaccination"] = st.slider(
            "💉 Couverture vaccinale", 0, 100, 35, step=5,
            help="Importance du taux de vaccination et non-vaccinés"
        )
        poids_manuels["Demographie"] = st.slider(
            "👥 Démographie", 0, 100, 15, step=5,
            help="Importance de la population et densité"
        )
        poids_manuels["Urbanisation"] = st.slider(
            "🏙️ Urbanisation", 0, 100, 8, step=2,
            help="Importance du type d'habitat (urbain/rural)"
        )
        poids_manuels["Climat"] = st.slider(
            "🌡️ Facteurs climatiques", 0, 100, 2, step=1,
            help="Importance de la température, humidité, saison"
        )

        total_poids = sum(poids_manuels.values())
        if total_poids > 0:
            for key in poids_manuels:
                poids_normalises[key] = poids_manuels[key] / total_poids

        st.markdown("---")
        st.markdown("**📊 Répartition normalisée :**")
        for key, value in poids_normalises.items():
            st.markdown(f"• {key} : **{value*100:.1f}%**")

        if abs(total_poids - 100) > 5:
            st.info(f"ℹ️ Total brut : {total_poids}% → Normalisé à 100%")
else:
    st.sidebar.info("Le modèle ML calculera automatiquement l'importance optimale de chaque variable")

# ── Seuils d'alerte (RESTAURÉ) ────────────────────────────────
st.sidebar.subheader("⚙️ Seuils d'Alerte")
with st.sidebar.expander("Configurer les seuils", expanded=False):
    seuil_baisse = st.slider(
        "Seuil de baisse significative (%)",
        min_value=10, max_value=90, value=75, step=5,
        help="Afficher les aires avec baisse ≥ X% par rapport à la moyenne"
    )
    seuil_hausse = st.slider(
        "Seuil de hausse significative (%)",
        min_value=10, max_value=200, value=50, step=10,
        help="Afficher les aires avec hausse ≥ X% par rapport à la moyenne"
    )
    seuil_alerte_epidemique = st.number_input(
        "Seuil d'alerte épidémique (cas/semaine)",
        min_value=1, max_value=100, value=5,
        help="Nombre de cas par semaine déclenchant une alerte"
    )

# ── Fonctions chargement géographique (inchangées) ────────────
@st.cache_data
def load_health_areas_from_zip(zip_path, iso3_filter):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if not shp_files:
                raise ValueError("Aucun fichier .shp trouvé dans le ZIP")
            shp_path = os.path.join(tmpdir, shp_files[0])
            gdf_full = gpd.read_file(shp_path)
            iso3_col = None
            for col in ['iso3','ISO3','iso_code','ISO_CODE','country_iso','COUNTRY_ISO']:
                if col in gdf_full.columns:
                    iso3_col = col
                    break
            if iso3_col is None:
                st.warning(f"⚠️ Colonne ISO3 non trouvée. Colonnes : {list(gdf_full.columns)}")
                return gpd.GeoDataFrame()
            gdf = gdf_full[gdf_full[iso3_col] == iso3_filter].copy()
            if gdf.empty:
                st.warning(f"⚠️ Aucune aire de santé pour {iso3_filter}")
                return gpd.GeoDataFrame()
            name_col = None
            for col in ['health_area','HEALTH_AREA','name_fr','name','NAME','nom','NOM','aire_sante']:
                if col in gdf.columns:
                    name_col = col
                    break
            if name_col:
                gdf['health_area'] = gdf[name_col]
            else:
                gdf['health_area'] = [f"Aire_{i+1}" for i in range(len(gdf))]

            # CORRECTION : supprimer lignes NaN/vides et géométries invalides
            gdf['health_area'] = gdf['health_area'].astype(str).str.strip()
            gdf = gdf[
                gdf['health_area'].notna() &
                (gdf['health_area'] != '') &
                (gdf['health_area'] != 'nan') &
                (gdf['health_area'] != 'None')
            ].copy()
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
    except Exception as e:
        st.error(f"❌ Erreur ZIP : {e}")
        return gpd.GeoDataFrame()

def load_shapefile_from_upload(upload_file):
    try:
        if upload_file.name.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, 'upload.zip')
                with open(zip_path, 'wb') as f:
                    f.write(upload_file.getvalue())
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(tmpdir)
                    shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                    if shp_files:
                        gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                    else:
                        raise ValueError("Aucun .shp trouvé")
        else:
            gdf = gpd.read_file(upload_file)
        if "health_area" not in gdf.columns:
            for col in ["health_area","HEALTH_AREA","name_fr","name","NAME","nom","NOM"]:
                if col in gdf.columns:
                    gdf["health_area"] = gdf[col]
                    break
            else:
                gdf["health_area"] = [f"Aire_{i}" for i in range(len(gdf))]
        gdf = gdf[gdf.geometry.is_valid]
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"❌ Erreur lecture : {e}")
        return gpd.GeoDataFrame()
# ============================================================
# PARTIE 2/5 - CHARGEMENT AIRES DE SANTÉ ET DONNÉES DE CAS
# ============================================================

# Chargement des aires de santé
_current_upload_name = upload_file.name if upload_file is not None else None
_cache_valide = (
    st.session_state.sa_gdf_cache is not None and (
        option_aire == "Fichier local (ao_hlthArea.zip)" or
        (option_aire == "Upload personnalisé" and
         st.session_state.get("upload_file_name_cache") == _current_upload_name)
    )
)
if _cache_valide:
    sa_gdf = st.session_state.sa_gdf_cache
    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées (cache)")
else:
    with st.spinner("🔄 Chargement des aires de santé..."):
        if option_aire == "Fichier local (ao_hlthArea.zip)":
            zip_path = os.path.join("data", "ao_hlthArea.zip")
            if not os.path.exists(zip_path):
                st.error(f"❌ Fichier non trouvé : {zip_path}")
                st.info("📁 Placez 'ao_hlthArea.zip' dans le dossier 'data/'")
                st.stop()
            sa_gdf = load_health_areas_from_zip(zip_path, iso3_pays)
            if sa_gdf.empty:
                st.error(f"❌ Impossible de charger {pays_selectionne} ({iso3_pays})")
                st.stop()
            else:
                st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées ({iso3_pays})")
                st.session_state.sa_gdf_cache = sa_gdf
        elif option_aire == "Upload personnalisé":
            if upload_file is None:
                st.warning("⚠️ Veuillez uploader un fichier")
                st.stop()
            else:
                sa_gdf = load_shapefile_from_upload(upload_file)
                if sa_gdf.empty:
                    st.error("❌ Fichier invalide")
                    st.stop()
                else:
                    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées")
                    st.session_state.sa_gdf_cache = sa_gdf
                    st.session_state["upload_file_name_cache"] = upload_file.name

if sa_gdf is None or sa_gdf.empty:
    st.error("❌ Aucune aire chargée")
    st.stop()

# ── Données fictives mode démo ────────────────────────────────
@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500):
    np.random.seed(42)
    aires = _sa_gdf["health_area"].unique()
    rows = []
    for annee in [2024, 2025]:
        for semaine in range(1, 53):
            n_cas = max(0, int(np.random.poisson(
                lam=max(1, 15 * np.sin(semaine * np.pi / 26) + 3))))
            for aire in np.random.choice(aires, size=min(n_cas, len(aires)), replace=False):
                rows.append({
                    "ID_Cas": len(rows) + 1,
                    "Semaine_Epi": semaine,
                    "Annee": annee,
                    "Aire_Sante": aire,
                    "Age_Mois": int(np.random.gamma(shape=2, scale=30, size=1)[0].clip(6, 180)),
                    "Statut_Vaccinal": np.random.choice(["Oui", "Non"], p=[0.55, 0.45]),
                    "Sexe": np.random.choice(["M", "F"]),
                    "Issue": np.random.choice(["Guéri", "Décédé", "Inconnu"], p=[0.92, 0.03, 0.05])
                })
    df_demo = pd.DataFrame(rows)
    df_demo["Date_Debut_Eruption"] = df_demo.apply(
        lambda r: semaine_vers_date(r["Annee"], r["Semaine_Epi"])
                  + timedelta(days=int(np.random.randint(0, 7))), axis=1)
    df_demo["Date_Notification"] = df_demo["Date_Debut_Eruption"] + timedelta(days=3)
    return df_demo

@st.cache_data
def generate_dummy_vaccination(_sa_gdf):
    np.random.seed(42)
    return pd.DataFrame({
        "health_area": _sa_gdf["health_area"],
        "Taux_Vaccination": np.random.beta(a=8, b=2, size=len(_sa_gdf)) * 100
    })

# Chargement des données de cas
with st.spinner("📥 Chargement données de cas..."):
    if mode_demo == "🧪 Mode démo (données simulées)":
        df = generate_dummy_linelists(sa_gdf)
        vaccination_df = generate_dummy_vaccination(sa_gdf)
        st.sidebar.info(f"📊 {len(df)} cas simulés générés")

    else:
        vaccination_df = None
        if linelist_file is None:
            st.error("❌ Veuillez uploader un fichier CSV de lineliste")
            st.stop()

        try:
            # ── Détection séparateur ──────────────────────────
            sep = detect_separator(linelist_file)
            try:
                df_raw = pd.read_csv(linelist_file, sep=sep, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                linelist_file.seek(0)
                df_raw = pd.read_csv(linelist_file, sep=sep, encoding="latin-1", low_memory=False)

            # ── Normalisation des noms de colonnes ────────────
            df_raw = normaliser_colonnes(df_raw, COLONNES_MAPPING)
            # ── Fuzzy match : lier le CSV aux aires du shapefile ──
            col_aire_csv = None
            for candidate in ["Aire_Sante", "health_area"]:
                if candidate in df_raw.columns:
                    col_aire_csv = candidate
                    break

            if col_aire_csv is not None and sa_gdf is not None and not sa_gdf.empty:
                liste_aires_shp = list(sa_gdf["health_area"].dropna().unique())
                df_raw = mapper_aires_csv(         
                    df_raw,
                    col_aire=col_aire_csv,
                    liste_aires=liste_aires_shp,
                    seuil=70
                )
                df_raw["Aire_Sante"] = df_raw["health_area"].fillna(df_raw[col_aire_csv])
            else:
                st.warning("⚠️ Colonne aire de santé non détectée dans le CSV.")
            # ── Détection colonne cas ─────────────────────────
            cas_col = None
            for _c in ["Cas_Total", "Cas", "cases", "Cases", "nb_cas", "nombre_cas"]:
                if _c in df_raw.columns:
                    cas_col = _c
                    break

            # ── CAS A : données agrégées Semaine_Epi + Cas_Total ──
            if "Semaine_Epi" in df_raw.columns and cas_col:

                # CORRECTION : nettoyage COMPLET avant toute conversion int()
                df_raw["Semaine_Epi"] = pd.to_numeric(df_raw["Semaine_Epi"], errors="coerce")
                df_raw = df_raw[df_raw["Semaine_Epi"].between(1, 52, inclusive="both")].copy()
                df_raw["Semaine_Epi"] = df_raw["Semaine_Epi"].fillna(1).astype(int)

                if "Annee" in df_raw.columns:
                    df_raw["Annee"] = pd.to_numeric(df_raw["Annee"], errors="coerce")
                    df_raw = df_raw[df_raw["Annee"].notna()].copy()
                    df_raw["Annee"] = df_raw["Annee"].astype(int)
                else:
                    df_raw["Annee"] = datetime.now().year

                if "Aire_Sante" not in df_raw.columns:
                    for _ac in ["health_area","name_fr","aire","zone_sante","district","localite"]:
                        if _ac in df_raw.columns:
                            df_raw["Aire_Sante"] = df_raw[_ac]
                            break
                    else:
                        df_raw["Aire_Sante"] = sa_gdf["health_area"].iloc[0]

                expanded_rows = []
                for _, row in df_raw.iterrows():
                    aire    = row.get("Aire_Sante", "Inconnu")
                    semaine = int(row["Semaine_Epi"])
                    annee   = int(row["Annee"])

                    # CORRECTION : fillna + coerce avant int() sur cas/décès
                    cas_total = int(max(0, pd.to_numeric(row.get(cas_col, 0), errors="coerce") or 0))
                    deces     = int(max(0, pd.to_numeric(row.get("Deces",     0), errors="coerce") or 0))

                    base_date = semaine_vers_date(annee, semaine)

                    for i in range(cas_total):
                        issue = "Décédé" if i < deces else "Guéri"
                        expanded_rows.append({
                            "ID_Cas":              len(expanded_rows) + 1,
                            "Semaine_Epi":         semaine,
                            "Annee":               annee,
                            "Date_Debut_Eruption": base_date + timedelta(days=int(np.random.randint(0, 7))),
                            "Date_Notification":   base_date + timedelta(days=int(np.random.randint(0, 10))),
                            "Aire_Sante":          aire,
                            "Age_Mois":            np.nan,
                            "Statut_Vaccinal":     "Inconnu",
                            "Sexe":                "Inconnu",
                            "Issue":               issue
                        })
                df = pd.DataFrame(expanded_rows)

            # ── CAS B : linelist individuelle avec Date_Debut_Eruption ──
            elif "Date_Debut_Eruption" in df_raw.columns:
                df = df_raw.copy()
                for col in ["Date_Debut_Eruption", "Date_Notification"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

            # ── CAS C : tentative détection automatique date ──
            else:
                st.warning("⚠️ Format CSV non standard — tentative de détection automatique...")
                df = df_raw.copy()
                for col in df.columns:
                    try:
                        test_dates = pd.to_datetime(df[col], errors='coerce')
                        if test_dates.notna().sum() > len(df) * 0.5:
                            df["Date_Debut_Eruption"] = test_dates
                            break
                    except Exception:
                        continue
                if "Date_Debut_Eruption" not in df.columns:
                    st.error("❌ Impossible de détecter une colonne date ou semaine valide dans ce fichier.")
                    st.stop()

            st.sidebar.success(f"✓ {len(df)} cas chargés")

        except Exception as e:
            st.error(f"❌ Erreur CSV : {e}")
            st.stop()

        # ── Vaccination ──────────────────────────────────────────────
        if vaccination_file is not None:
            try:
                sep_vax = detect_separator(vaccination_file)
                vaccination_df = pd.read_csv(vaccination_file, sep=sep_vax, encoding="utf-8-sig")
                if len(vaccination_df.columns) == 1:
                    vaccination_file.seek(0)
                    vaccination_df = pd.read_csv(vaccination_file, sep=";", encoding="utf-8-sig")
        
                vaccination_df = normaliser_colonnes(vaccination_df, COLONNES_MAPPING)
        
                # Créer la clé de jointure "health_area" depuis "Aire_Sante" si nécessaire
                if "health_area" not in vaccination_df.columns and "Aire_Sante" in vaccination_df.columns:
                    vaccination_df["health_area"] = vaccination_df["Aire_Sante"]
                elif "health_area" not in vaccination_df.columns:
                    st.sidebar.warning("⚠️ Colonne 'health_area' ou 'Aire_Sante' introuvable dans le CSV vaccination")
                    vaccination_df = None
                    
                if vaccination_df is not None and "health_area" in vaccination_df.columns and sa_gdf is not None:
                    liste_aires_shp = list(sa_gdf["health_area"].dropna().unique())
                    
                    # ✅ CORRECTION : renommer la colonne SOURCE avant le fuzzy match
                    # pour éviter le conflit "health_area" → "health_area"
                    vaccination_df = vaccination_df.rename(columns={"health_area": "health_area_orig"})
                    vaccination_df = mapper_aires_csv(
                        vaccination_df,
                        col_aire="health_area_orig",   # ← col source renommée
                        liste_aires=liste_aires_shp,
                        seuil=70
                    )
                    # mapper_aires_csv crée "health_area" = nom shapefile → c'est la bonne clé
        
                # Conversion taux (virgule → point)
                if vaccination_df is not None and "Taux_Vaccination" in vaccination_df.columns:
                    vaccination_df["Taux_Vaccination"] = (
                        vaccination_df["Taux_Vaccination"]
                        .astype(str)
                        .str.replace(",", ".", regex=False)
                        .str.strip()
                    )
                    vaccination_df["Taux_Vaccination"] = pd.to_numeric(
                        vaccination_df["Taux_Vaccination"], errors="coerce"
                    )
        
                # Garder uniquement les colonnes utiles
                if vaccination_df is not None:
                    cols_vax_keep = [c for c in ["health_area", "Taux_Vaccination"] if c in vaccination_df.columns]
                    vaccination_df = vaccination_df[cols_vax_keep].drop_duplicates(subset=["health_area"])
        
                    n_ok = vaccination_df["Taux_Vaccination"].notna().sum() if "Taux_Vaccination" in vaccination_df.columns else 0
                    st.sidebar.success(
                        f"✓ Couverture vaccinale chargée : {n_ok}/{len(vaccination_df)} "
                        f"aires avec valeurs valides"
                    )
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erreur vaccination CSV : {e}")
                vaccination_df = None

# ── Normalisation colonnes ─────────────────────────────────────
df = normaliser_colonnes(df, COLONNES_MAPPING)

if "ID_Cas" not in df.columns:
    df["ID_Cas"] = range(1, len(df) + 1)

if "Aire_Sante" not in df.columns:
    for col in df.columns:
        if df[col].dtype == object:
            sample_values = set(df[col].dropna().unique())
            sa_values = set(sa_gdf["health_area"].unique())
            if len(sample_values.intersection(sa_values)) > 0:
                df["Aire_Sante"] = df[col]
                st.sidebar.info(f"ℹ️ Colonne 'Aire_Sante' créée depuis '{col}'")
                break
    else:
        df["Aire_Sante"] = sa_gdf["health_area"].iloc[0]
        st.sidebar.warning("⚠️ Aucune colonne aire trouvée, valeur par défaut assignée")

if "Date_Debut_Eruption" not in df.columns:
    if "Semaine_Epi" in df.columns and "Annee" in df.columns:
        df["Date_Debut_Eruption"] = df.apply(
            lambda r: semaine_vers_date(r["Annee"], r["Semaine_Epi"])
            if pd.notna(r.get("Annee")) and pd.notna(r.get("Semaine_Epi")) else pd.NaT, axis=1)
    else:
        df["Date_Debut_Eruption"] = pd.to_datetime(datetime.now())
else:
    df["Date_Debut_Eruption"] = pd.to_datetime(df["Date_Debut_Eruption"], errors='coerce')

if "Date_Notification" not in df.columns:
    df["Date_Notification"] = df["Date_Debut_Eruption"] + pd.to_timedelta(3, unit="D")

if "Age_Mois" not in df.columns:
    df["Age_Mois"] = np.nan
if "Statut_Vaccinal" not in df.columns:
    df["Statut_Vaccinal"] = "Inconnu"
if "Sexe" not in df.columns:
    df["Sexe"] = "Inconnu"
if "Issue" not in df.columns:
    df["Issue"] = "Inconnu"

# ── Calcul semaine épidémiologique ─────────────────────────────
if "Semaine_Epi" not in df.columns:
    df["Semaine_Epi"] = df["Date_Debut_Eruption"].apply(
        lambda d: int(d.isocalendar()[1]) if pd.notna(d) else np.nan)
if "Annee" not in df.columns:
    df["Annee"] = df["Date_Debut_Eruption"].dt.year

df["Semaine_Epi"] = pd.to_numeric(df["Semaine_Epi"], errors="coerce")
df = df[df["Semaine_Epi"].between(1, 52, inclusive="both")].copy()
df["Semaine_Epi"] = df["Semaine_Epi"].astype(int)
df["Annee"] = pd.to_numeric(df["Annee"], errors="coerce").fillna(datetime.now().year).astype(int)
df["Semaine_Annee"] = df["Annee"].astype(str) + "-S" + df["Semaine_Epi"].astype(str).str.zfill(2)
df["sort_key"] = df["Annee"] * 100 + df["Semaine_Epi"]

# ── Détection dernière semaine ────────────────────────────────
idx_last = df["sort_key"].idxmax()
derniere_semaine_epi = int(df.loc[idx_last, "Semaine_Epi"])
derniere_annee       = int(df.loc[idx_last, "Annee"])
n_semaines_uniques   = df["Semaine_Annee"].nunique()

st.sidebar.info(
    f"📅 Dernière semaine : **S{derniere_semaine_epi:02d} {derniere_annee}** | "
    f"**{n_semaines_uniques}** semaines au total"
)

st.sidebar.subheader("📅 Filtres Temporels & Géographiques")

annees_dispo   = sorted(df["Annee"].dropna().unique().astype(int).tolist())
semaines_dispo = sorted(df["Semaine_Epi"].dropna().unique().astype(int).tolist())
aires_dispo    = sorted(df["Aire_Sante"].dropna().unique().tolist()) if "Aire_Sante" in df.columns else []

filtre_annees   = st.sidebar.multiselect("📅 Années", options=annees_dispo, default=annees_dispo)
filtre_semaines = st.sidebar.multiselect("🗓️ Semaines", options=semaines_dispo, default=semaines_dispo,
                                          format_func=lambda s: f"S{s:02d}")
filtre_aires    = st.sidebar.multiselect("🏥 Aires de santé", options=aires_dispo, default=aires_dispo)

df_filtre = df.copy()
if filtre_annees:
    df_filtre = df_filtre[df_filtre["Annee"].isin([int(a) for a in filtre_annees])]
if filtre_semaines:
    df_filtre = df_filtre[df_filtre["Semaine_Epi"].isin([int(s) for s in filtre_semaines])]
if filtre_aires:
    df_filtre = df_filtre[df_filtre["Aire_Sante"].isin(filtre_aires)]

df = df_filtre.copy()

if len(df) == 0:
    st.warning("⚠️ Aucun cas dans la sélection. Ajustez les filtres.")
    st.stop()
# ============================================================
# PARTIE 3/5 - ENRICHISSEMENT AVEC DONNÉES EXTERNES
# WorldPop, NASA POWER, GHSL
# ============================================================

# WorldPop - Données démographiques
@st.cache_data
def worldpop_children_stats(_sa_gdf, use_gee, cache_key=""):
    if not use_gee:
        st.sidebar.warning("⚠️ WorldPop : GEE indisponible")
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Garcons": [np.nan] * len(_sa_gdf),
            "Pop_Filles": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf),
            "Pop_M_0": [np.nan] * len(_sa_gdf),
            "Pop_M_1": [np.nan] * len(_sa_gdf),
            "Pop_M_5": [np.nan] * len(_sa_gdf),
            "Pop_M_10": [np.nan] * len(_sa_gdf),
            "Pop_F_0": [np.nan] * len(_sa_gdf),
            "Pop_F_1": [np.nan] * len(_sa_gdf),
            "Pop_F_5": [np.nan] * len(_sa_gdf),
            "Pop_F_10": [np.nan] * len(_sa_gdf)
        })

    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        status_text.text("📥 Chargement WorldPop...")
        # Correction D3 : mosaïque restreinte à un millésime unique (voir
        # epi_app_bridge.worldpop_mosaic pour le détail du défaut).
        pop_img, _wp_annee = epi_app_bridge.worldpop_mosaic(ee)
        if _wp_annee:
            st.sidebar.caption(f"WorldPop : millésime {_wp_annee}")

        male_bands = ["M_0", "M_1", "M_5", "M_10"]
        female_bands = ["F_0", "F_1", "F_5", "F_10"]

        selected_males = pop_img.select(male_bands)
        selected_females = pop_img.select(female_bands)
        total_pop = pop_img.select(['population'])

        males_sum = selected_males.reduce(ee.Reducer.sum()).rename('garcons')
        females_sum = selected_females.reduce(ee.Reducer.sum()).rename('filles')
        enfants = males_sum.add(females_sum).rename('enfants')

        final_mosaic = (total_pop
                       .addBands(selected_males)
                       .addBands(selected_females)
                       .addBands(males_sum)
                       .addBands(females_sum)
                       .addBands(enfants))

        pixel_area = ee.Image.pixelArea().divide(10000)
        final_mosaic_count = final_mosaic.multiply(pixel_area)

        status_text.text("🗺️ Conversion géométries...")
        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"health_area": row["health_area"]}

            if geom.geom_type == 'Polygon':
                coords = [[[x, y] for x, y in geom.exterior.coords]]
                ee_geom = ee.Geometry.Polygon(coords)
            elif geom.geom_type == 'MultiPolygon':
                coords = []
                for poly in geom.geoms:
                    coords.append([[[x, y] for x, y in poly.exterior.coords]])
                ee_geom = ee.Geometry.MultiPolygon(coords)
            else:
                continue

            features.append(ee.Feature(ee_geom, props))

        fc = ee.FeatureCollection(features)

        status_text.text("🔢 Calcul statistiques zonales...")
        stats = final_mosaic_count.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.sum(),
            scale=100,
            crs='EPSG:4326'
        )

        status_text.text("📊 Extraction résultats...")
        stats_info = stats.getInfo()

        data_list = []
        total_aires = len(stats_info['features'])

        for i, feat in enumerate(stats_info['features']):
            props = feat['properties']

            pop_totale = props.get("population", 0)
            garcons = props.get("garcons", 0)
            filles = props.get("filles", 0)
            enfants_total = props.get("enfants", 0)

            m_0 = props.get("M_0", 0)
            m_1 = props.get("M_1", 0)
            m_5 = props.get("M_5", 0)
            m_10 = props.get("M_10", 0)

            f_0 = props.get("F_0", 0)
            f_1 = props.get("F_1", 0)
            f_5 = props.get("F_5", 0)
            f_10 = props.get("F_10", 0)

            data_list.append({
                "health_area": props.get("health_area", ""),
                "Pop_Totale": int(pop_totale) if pop_totale > 0 else np.nan,
                "Pop_Garcons": int(garcons),
                "Pop_Filles": int(filles),
                "Pop_Enfants": int(enfants_total),
                "Pop_M_0": int(m_0),
                "Pop_M_1": int(m_1),
                "Pop_M_5": int(m_5),
                "Pop_M_10": int(m_10),
                "Pop_F_0": int(f_0),
                "Pop_F_1": int(f_1),
                "Pop_F_5": int(f_5),
                "Pop_F_10": int(f_10)
            })

            progress_value = min((i + 1) / total_aires, 1.0)
            progress_bar.progress(progress_value)

        progress_bar.empty()
        status_text.text("✅ WorldPop terminé")

        return pd.DataFrame(data_list)

    except Exception as e:
        st.sidebar.error(f"❌ WorldPop : {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Garcons": [np.nan] * len(_sa_gdf),
            "Pop_Filles": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf),
            "Pop_M_0": [np.nan] * len(_sa_gdf),
            "Pop_M_1": [np.nan] * len(_sa_gdf),
            "Pop_M_5": [np.nan] * len(_sa_gdf),
            "Pop_M_10": [np.nan] * len(_sa_gdf),
            "Pop_F_0": [np.nan] * len(_sa_gdf),
            "Pop_F_1": [np.nan] * len(_sa_gdf),
            "Pop_F_5": [np.nan] * len(_sa_gdf),
            "Pop_F_10": [np.nan] * len(_sa_gdf)
        })


# GHSL - Classification urbaine
@st.cache_data
def urban_classification(_sa_gdf, use_gee, cache_key=""):
    if not use_gee:
        st.sidebar.warning("⚠️ GHSL : GEE indisponible")
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Urbanisation": [np.nan] * len(_sa_gdf)
        })

    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        status_text.text("🏙️ Classification urbaine...")

        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"health_area": row["health_area"]}

            if geom.geom_type == 'Polygon':
                coords = [[[x, y] for x, y in geom.exterior.coords]]
                ee_geom = ee.Geometry.Polygon(coords)
            elif geom.geom_type == 'MultiPolygon':
                coords = []
                for poly in geom.geoms:
                    coords.append([[[x, y] for x, y in poly.exterior.coords]])
                ee_geom = ee.Geometry.MultiPolygon(coords)
            else:
                continue

            features.append(ee.Feature(ee_geom, props))

        fc = ee.FeatureCollection(features)
        smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD/2020")

        def classify(feature):
            stats = smod.reduceRegion(
                ee.Reducer.mode(),
                feature.geometry(),
                scale=1000,
                maxPixels=1e9
            )
            smod_value = ee.Number(stats.get("smod_code")).toInt()
            urbanisation = ee.Algorithms.If(
                smod_value.gte(30),
                "Urbain",
                ee.Algorithms.If(smod_value.eq(23), "Semi-urbain", "Rural")
            )
            return feature.set({"Urbanisation": urbanisation})

        urban_fc = fc.map(classify)
        urban_info = urban_fc.getInfo()

        data_list = []
        total_aires = len(urban_info['features'])

        for i, feat in enumerate(urban_info['features']):
            props = feat['properties']
            data_list.append({
                "health_area": props.get("health_area", ""),
                "Urbanisation": props.get("Urbanisation", "Rural")
            })
            progress_value = min((i + 1) / total_aires, 1.0)
            progress_bar.progress(progress_value)

        progress_bar.empty()
        status_text.text("✅ GHSL terminé")

        return pd.DataFrame(data_list)

    except Exception as e:
        st.sidebar.error(f"❌ GHSL : {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Urbanisation": [np.nan] * len(_sa_gdf)
        })


# NASA POWER - Données climatiques
@st.cache_data(ttl=86400)
def fetch_climate_nasa_power(_sa_gdf, start_date, end_date, cache_key=""):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    data_list = []
    total_aires = len(_sa_gdf)

    for idx, row in _sa_gdf.iterrows():
        status_text.text(f"🌡️ Climat {idx+1}/{total_aires}...")

        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x

        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,PRECTOTCORR,RH2M",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "format": "JSON"
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            j = r.json()

            if "properties" in j and "parameter" in j["properties"]:
                p = j["properties"]["parameter"]

                temp_values = list(p.get("T2M", {}).values())
                rh_values = list(p.get("RH2M", {}).values())

                temp_mean = np.nanmean(temp_values) if temp_values else np.nan
                rh_mean = np.nanmean(rh_values) if rh_values else np.nan

                saison_seche_hum = rh_mean * 0.7 if not np.isnan(rh_mean) else np.nan

                data_list.append({
                    "health_area": row["health_area"],
                    "Temperature_Moy": temp_mean,
                    "Humidite_Moy": rh_mean,
                    "Saison_Seche_Humidite": saison_seche_hum
                })
            else:
                data_list.append({
                    "health_area": row["health_area"],
                    "Temperature_Moy": np.nan,
                    "Humidite_Moy": np.nan,
                    "Saison_Seche_Humidite": np.nan
                })
        except:
            data_list.append({
                "health_area": row["health_area"],
                "Temperature_Moy": np.nan,
                "Humidite_Moy": np.nan,
                "Saison_Seche_Humidite": np.nan
            })

        progress_value = min((idx + 1) / total_aires, 1.0)
        progress_bar.progress(progress_value)

    progress_bar.empty()
    status_text.text("✅ Climat terminé")

    return pd.DataFrame(data_list)


# ── Enrichissement du GeoDataFrame ────────────────────────────
if len(df) > 0:
    _date_min = df["Date_Debut_Eruption"].min()
    _date_max = df["Date_Debut_Eruption"].max()
    if pd.isna(_date_min):
        _date_min = datetime(datetime.now().year, 1, 1)
    if pd.isna(_date_max):
        _date_max = datetime.now()
    climat_start = _date_min.to_pydatetime() if hasattr(_date_min, "to_pydatetime") else _date_min
    climat_end   = _date_max.to_pydatetime() if hasattr(_date_max, "to_pydatetime") else _date_max
else:
    climat_start = datetime(datetime.now().year, 1, 1)
    climat_end   = datetime.now()

_cache_key = f"enrichi_{iso3_pays if iso3_pays else (upload_file.name if upload_file else 'upload')}"

# CORRECTION : vérifier que le cache contient bien les 3 clés attendues
_cache_valide_enrichi = (
    _cache_key in st.session_state
    and st.session_state[_cache_key] is not None
    and isinstance(st.session_state[_cache_key], dict)
    and all(k in st.session_state[_cache_key] for k in ["pop", "urban", "climate"])
)

if not _cache_valide_enrichi:
    with st.spinner("🔄 Enrichissement des données..."):
        # cache_key force @st.cache_data à distinguer chaque pays/fichier
        _ck = iso3_pays or (upload_file.name if upload_file else "upload")
        pop_df     = worldpop_children_stats(sa_gdf, gee_ok, cache_key=_ck)
        urban_df   = urban_classification(sa_gdf, gee_ok, cache_key=_ck)
        climate_df = fetch_climate_nasa_power(sa_gdf, climat_start, climat_end, cache_key=_ck)
        st.session_state[_cache_key] = {
            "pop": pop_df, "urban": urban_df, "climate": climate_df
        }
else:
    pop_df     = st.session_state[_cache_key]["pop"]
    urban_df   = st.session_state[_cache_key]["urban"]
    climate_df = st.session_state[_cache_key]["climate"]

sa_gdf_enrichi = sa_gdf.copy()
sa_gdf_enrichi = sa_gdf_enrichi.merge(pop_df,     on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(urban_df,   on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(climate_df, on="health_area", how="left")

if vaccination_df is not None and "health_area" in vaccination_df.columns:
    # Supprimer la colonne si elle existe déjà (évite _x/_y en cas de double merge)
    if "Taux_Vaccination" in sa_gdf_enrichi.columns:
        sa_gdf_enrichi = sa_gdf_enrichi.drop(columns=["Taux_Vaccination"])
    sa_gdf_enrichi = sa_gdf_enrichi.merge(
        vaccination_df[["health_area", "Taux_Vaccination"]],
        on="health_area",
        how="left"
    )
    # ✅ Variables dérivées calculées ICI, après merge, quand Pop_Enfants est disponible
    if "Taux_Vaccination" in sa_gdf_enrichi.columns:
        sa_gdf_enrichi["Couverture_Insuffisante"] = (
            sa_gdf_enrichi["Taux_Vaccination"] < 95
        ).astype(int)
        sa_gdf_enrichi["Gap_Immunite_Collective"] = (
            95 - sa_gdf_enrichi["Taux_Vaccination"]
        ).clip(lower=0)
        if "Pop_Enfants" in sa_gdf_enrichi.columns:
            sa_gdf_enrichi["Non_Vaccines_Estimes"] = (
                (1 - sa_gdf_enrichi["Taux_Vaccination"].fillna(100) / 100)
                * sa_gdf_enrichi["Pop_Enfants"].fillna(0)
            )
else:
    sa_gdf_enrichi["Taux_Vaccination"] = np.nan
    sa_gdf_enrichi["Couverture_Insuffisante"] = np.nan
    sa_gdf_enrichi["Gap_Immunite_Collective"] = np.nan
    sa_gdf_enrichi["Non_Vaccines_Estimes"] = np.nan

sa_gdf_m = sa_gdf_enrichi.to_crs("ESRI:54009")
sa_gdf_enrichi["Superficie_km2"] = sa_gdf_m.geometry.area / 1e6

sa_gdf_enrichi["Densite_Pop"] = (
    sa_gdf_enrichi["Pop_Totale"] /
    sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

sa_gdf_enrichi["Densite_Enfants"] = (
    sa_gdf_enrichi["Pop_Enfants"] /
    sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

sa_gdf_enrichi = sa_gdf_enrichi.replace([np.inf, -np.inf], np.nan)

st.sidebar.success("✓ Enrichissement terminé")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Données disponibles")

donnees_dispo = {
    "Population":   not sa_gdf_enrichi["Pop_Totale"].isna().all(),
    "Urbanisation": not sa_gdf_enrichi["Urbanisation"].isna().all(),
    "Climat":       not sa_gdf_enrichi["Humidite_Moy"].isna().all(),
    "Vaccination":  not sa_gdf_enrichi["Taux_Vaccination"].isna().all()
}

for nom, dispo in donnees_dispo.items():
    icone = "✅" if dispo else "❌"
    st.sidebar.text(f"{icone} {nom}")

# ============================================================
# CORRECTION 4 : FLAGS DE DISPONIBILITÉ DES DONNÉES RÉELLES
# ============================================================

has_age_reel = (
    "Age_Mois" in df.columns
    and df["Age_Mois"].notna().sum() > 0
    and (df["Age_Mois"] > 0).sum() > 0
)

has_vaccination_reel = (
    "Statut_Vaccinal" in df.columns
    and df["Statut_Vaccinal"].notna().sum() > 0
    and (df["Statut_Vaccinal"] != "Inconnu").sum() > 0
)

if mode_demo == "🧪 Mode démo (données simulées)":
    has_age_reel = True
    has_vaccination_reel = True

age_median_worldpop = None
if not has_age_reel and donnees_dispo["Population"]:
    tranches = [
        (0,  12, "Pop_M_0",  "Pop_F_0"),
        (12, 48, "Pop_M_1",  "Pop_F_1"),
        (60, 48, "Pop_M_5",  "Pop_F_5"),
        (120,60, "Pop_M_10", "Pop_F_10"),
    ]
    totaux = []
    for age_debut_mois, duree_mois, col_m, col_f in tranches:
        t = 0
        if col_m in sa_gdf_enrichi.columns:
            t += pd.to_numeric(sa_gdf_enrichi[col_m], errors="coerce").fillna(0).sum()
        if col_f in sa_gdf_enrichi.columns:
            t += pd.to_numeric(sa_gdf_enrichi[col_f], errors="coerce").fillna(0).sum()
        totaux.append((age_debut_mois, duree_mois, t))

    total_pop_enfants = sum(t for _, _, t in totaux)
    if total_pop_enfants > 0:
        cumul = 0
        for age_debut_mois, duree_mois, t in totaux:
            cumul += t
            if cumul >= total_pop_enfants / 2:
                cumul_avant = cumul - t
                frac = (total_pop_enfants / 2 - cumul_avant) / t if t > 0 else 0.5
                age_median_worldpop = age_debut_mois + frac * duree_mois
                break

# ── Agrégation par aire ────────────────────────────────────────
agg_dict = {"ID_Cas": "count"}

if has_age_reel:
    agg_dict["Age_Mois"] = "mean"

if has_vaccination_reel:
    agg_dict["Statut_Vaccinal"] = lambda x: (x == "Non").mean() * 100

cases_by_area = df.groupby("Aire_Sante").agg(agg_dict).reset_index()

rename_map = {"ID_Cas": "Cas_Observes"}
if has_age_reel and "Age_Mois" in cases_by_area.columns:
    rename_map["Age_Mois"] = "Age_Moyen"
if has_vaccination_reel and "Statut_Vaccinal" in cases_by_area.columns:
    rename_map["Statut_Vaccinal"] = "Taux_Non_Vaccines"

cases_by_area = cases_by_area.rename(columns=rename_map)

if "Taux_Non_Vaccines" not in cases_by_area.columns:
    cases_by_area["Taux_Non_Vaccines"] = np.nan
if "Age_Moyen" not in cases_by_area.columns:
    cases_by_area["Age_Moyen"] = np.nan

_merge_key = "health_area" if "health_area" in cases_by_area.columns else "Aire_Sante"
_cases_temp = cases_by_area.copy()
if _merge_key != "health_area":
    _cases_temp = _cases_temp.rename(columns={_merge_key: "health_area"})

sa_gdf_with_cases = sa_gdf_enrichi.copy()
sa_gdf_with_cases = sa_gdf_with_cases.merge(
    _cases_temp,
    on="health_area",
    how="left"
)

# CORRECTION CRITIQUE : merge() peut dégrader un GeoDataFrame en DataFrame ordinaire
if not isinstance(sa_gdf_with_cases, gpd.GeoDataFrame):
    sa_gdf_with_cases = gpd.GeoDataFrame(
        sa_gdf_with_cases,
        geometry="geometry",
        crs="EPSG:4326"
    )
elif sa_gdf_with_cases.crs is None:
    sa_gdf_with_cases = sa_gdf_with_cases.set_crs("EPSG:4326")
elif sa_gdf_with_cases.crs.to_epsg() != 4326:
    sa_gdf_with_cases = sa_gdf_with_cases.to_crs("EPSG:4326")

sa_gdf_with_cases["Cas_Observes"] = sa_gdf_with_cases["Cas_Observes"].fillna(0)

sa_gdf_with_cases["Taux_Attaque_10000"] = (
    sa_gdf_with_cases["Cas_Observes"] /
    sa_gdf_with_cases["Pop_Enfants"].replace(0, np.nan) * 10000
).replace([np.inf, -np.inf], np.nan)
# ============================================================
# PARTIE 4/5 - ONGLETS TAB1 (DASHBOARD) ET TAB2 (CARTOGRAPHIE)
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard & Analyse",
    "🗺️ Cartographie",
    "🔮 Modélisation & Prédiction",
    "🔬 Validation Rétrospective"
])

# ============================================================
# TAB 1 — DASHBOARD & ANALYSE
# ============================================================
with tab1:

    st.header("📊 Indicateurs Clés de Performance")
    ann_str = ", ".join(str(a) for a in sorted(set(df["Annee"].dropna().astype(int))))
    st.caption(
        f"📌 Analyse : Années **{ann_str}** | "
        f"**{df['Aire_Sante'].nunique()}** aires | "
        f"**{df['Semaine_Annee'].nunique()}** semaines épidémiologiques | "
        f"Dernière semaine : **S{derniere_semaine_epi:02d} {derniere_annee}**"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📈 Cas totaux", f"{len(df):,}")
    with col2:
        if has_vaccination_reel:
            taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
            st.metric("💉 Non vaccinés", f"{taux_non_vac:.1f}%", delta=f"{taux_non_vac-45:+.1f}%")
        else:
            st.metric("💉 Non vaccinés", "N/A")
    with col3:
        if donnees_dispo["Population"] and age_median_worldpop is not None:
            st.metric("👶 Âge médian (WorldPop)", f"{int(age_median_worldpop)} mois")
        elif has_age_reel:
            st.metric("👶 Âge médian", f"{int(df['Age_Mois'].median())} mois")
        else:
            st.metric("👶 Âge médian", "N/A")
    with col4:
        taux_letalite = ((df["Issue"] == "Décédé").mean() * 100) if "Issue" in df.columns else 0
        st.metric("☠️ Létalité", f"{taux_letalite:.2f}%" if taux_letalite > 0 else "N/A")
    with col5:
        n_aires_touchees = df["Aire_Sante"].nunique()
        pct_aires = (n_aires_touchees / len(sa_gdf)) * 100
        st.metric("🗺️ Aires touchées", f"{n_aires_touchees}/{len(sa_gdf)}", delta=f"{pct_aires:.0f}%")

    # ── Courbe épidémique ─────────────────────────────────────
    st.header("📈 Analyse Temporelle par Semaines Épidémiologiques")

    weekly_cases = (
        df.groupby(["Annee", "Semaine_Epi"])
        .size()
        .reset_index(name="Cas")
    )
    weekly_cases["sort_key"] = weekly_cases["Annee"] * 100 + weekly_cases["Semaine_Epi"]
    weekly_cases["Semaine_Label"] = (
        weekly_cases["Annee"].astype(str) + "-S" +
        weekly_cases["Semaine_Epi"].astype(str).str.zfill(2)
    )
    weekly_cases = weekly_cases.sort_values("sort_key").reset_index(drop=True)

    fig_epi = go.Figure()
    fig_epi.add_trace(go.Scatter(
        x=weekly_cases["Semaine_Label"], y=weekly_cases["Cas"],
        mode="lines+markers", name="Cas observés",
        line=dict(color="#d32f2f", width=3), marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Cas : %{y}<extra></extra>"
    ))
    try:
        from scipy.signal import savgol_filter
        if len(weekly_cases) > 5:
            wl = min(7, len(weekly_cases) if len(weekly_cases) % 2 == 1 else len(weekly_cases) - 1)
            tendance = savgol_filter(weekly_cases["Cas"].values, window_length=wl, polyorder=2)
            fig_epi.add_trace(go.Scatter(
                x=weekly_cases["Semaine_Label"], y=tendance,
                mode="lines", name="Tendance",
                line=dict(color="#1976d2", width=2, dash="dash")
            ))
    except Exception:
        pass

    fig_epi.add_hline(
        y=float(seuil_alerte_epidemique),
        line_dash="dot", line_color="orange",
        annotation_text=f"Seuil d'alerte ({seuil_alerte_epidemique} cas/sem)",
        annotation_position="right"
    )
    fig_epi.update_layout(
        title="Courbe épidémique par semaines épidémiologiques",
        xaxis_title="Semaine épidémiologique", yaxis_title="Nombre de cas",
        hovermode="x unified", height=400,
        xaxis=dict(tickangle=-45, nticks=20), template="plotly_white"
    )
    st.plotly_chart(fig_epi, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        semaine_max = weekly_cases.loc[weekly_cases["Cas"].idxmax()]
        st.metric("🔴 Semaine pic", semaine_max["Semaine_Label"], f"{int(semaine_max['Cas'])} cas")
    with col2:
        st.metric("📊 Moyenne hebdo", f"{weekly_cases['Cas'].mean():.1f} cas")
    with col3:
        if len(weekly_cases) >= 2:
            variation = weekly_cases.iloc[-1]["Cas"] - weekly_cases.iloc[-2]["Cas"]
            cas_prec  = weekly_cases.iloc[-2]["Cas"]
            pct_var   = (variation / cas_prec * 100) if cas_prec > 0 else 0
            st.metric("📉 Variation dernière semaine", f"{int(variation):+d} cas", f"{pct_var:+.1f}%")
        else:
            st.metric("📉 Variation dernière semaine", "N/A")

    # ── Distribution par âge ──────────────────────────────────
    st.subheader("👶 Distribution par Tranches d'Âge")
    if has_age_reel:
        df["Tranche_Age"] = pd.cut(df["Age_Mois"],
            bins=[0,12,60,120,180],
            labels=["0-1 an","1-5 ans","5-10 ans","10-15 ans"])
        agg_dict_age = {"ID_Cas": "count"}
        if has_vaccination_reel:
            agg_dict_age["Statut_Vaccinal"] = lambda x: (x == "Non").mean() * 100
        age_stats = df.groupby("Tranche_Age").agg(agg_dict_age).reset_index()
        rename_age = {"ID_Cas": "Nombre_Cas"}
        if has_vaccination_reel and "Statut_Vaccinal" in age_stats.columns:
            rename_age["Statut_Vaccinal"] = "Pct_Non_Vaccines"
        age_stats = age_stats.rename(columns=rename_age)
        col1, col2 = st.columns(2)
        with col1:
            fig_age = px.bar(age_stats, x="Tranche_Age", y="Nombre_Cas",
                title="Cas par tranche d'âge", color="Nombre_Cas",
                color_continuous_scale="Reds", text="Nombre_Cas")
            fig_age.update_traces(textposition="outside")
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            if has_vaccination_reel and "Pct_Non_Vaccines" in age_stats.columns:
                fig_vacc_age = px.bar(age_stats, x="Tranche_Age", y="Pct_Non_Vaccines",
                    title="% non vaccinés par âge", color="Pct_Non_Vaccines",
                    color_continuous_scale="Oranges")
                st.plotly_chart(fig_vacc_age, use_container_width=True)
            else:
                st.info("ℹ️ Données vaccination par âge non disponibles")
    else:
        st.info("ℹ️ Données d'âge non disponibles dans ce fichier")

    # ── Top 10 ────────────────────────────────────────────────
    # ✅ CORRIGÉ : 4 espaces = directement dans with tab1:, PAS dans le else ci-dessus
    st.header("🏆 10 aires de santés avec le taux d'attaque le plus élevé")
    top_data = sa_gdf_with_cases[["health_area", "Cas_Observes", "Taux_Attaque_10000"]].copy()
    top_data = top_data[top_data["Cas_Observes"] > 0]

    top_data_agg = (
        top_data
        .groupby("health_area", as_index=False)
        .agg(
            Cas_Observes=("Cas_Observes", "sum"),
            Taux_Attaque_10000=("Taux_Attaque_10000", "mean")
        )
    )

    has_taux = (
        "Taux_Attaque_10000" in top_data_agg.columns
        and top_data_agg["Taux_Attaque_10000"].notna().sum() > 0
    )

    if has_taux:
        tab_ta, tab_cas = st.tabs(["📊 Taux d'attaque", "📊 Nombre de cas"])

        with tab_ta:
            top10 = (
                top_data_agg
                .nlargest(10, "Taux_Attaque_10000")
                .sort_values("Taux_Attaque_10000", ascending=True)
                .reset_index(drop=True)
            )
            fig_ta = px.bar(
                top10, x="Taux_Attaque_10000", y="health_area",
                orientation="h", color="Taux_Attaque_10000",
                color_continuous_scale="Reds", text="Taux_Attaque_10000",
                labels={
                    "Taux_Attaque_10000": "Taux d'attaque (/10 000 enf.)",
                    "health_area": "Aire de santé"
                },
                title="Top 10 — Taux d'attaque pour 10 000 enfants",
            )
            fig_ta.update_traces(
                texttemplate="%{x:.1f} /10 000", textposition="outside",
                hovertemplate="<b>%{y}</b><br>Taux d'attaque : %{x:.1f} /10 000 enf.<extra></extra>"
            )
            fig_ta.update_layout(
                height=max(400, len(top10) * 50), template="plotly_white",
                margin=dict(l=10, r=140, t=50, b=30),
                coloraxis_showscale=False,
                xaxis_title="Taux d'attaque (/10 000 enfants)",
                yaxis_title="Aire de santé",
                yaxis=dict(categoryorder="array", categoryarray=top10["health_area"].tolist())
            )
            st.plotly_chart(fig_ta, use_container_width=True)

        with tab_cas:
            top10c = (
                top_data_agg
                .nlargest(10, "Cas_Observes")
                .sort_values("Cas_Observes", ascending=True)
                .reset_index(drop=True)
            )
            fig_cas = px.bar(
                top10c, x="Cas_Observes", y="health_area",
                orientation="h", color="Cas_Observes",
                color_continuous_scale="Reds", text="Cas_Observes",
                labels={"Cas_Observes": "Nombre de cas", "health_area": "Aire de santé"},
                title="Top 10 — Nombre de cas observés",
            )
            fig_cas.update_traces(
                texttemplate="%{x} cas", textposition="outside",
                hovertemplate="<b>%{y}</b><br>Cas observés : %{x}<extra></extra>"
            )
            fig_cas.update_layout(
                height=max(400, len(top10c) * 50), template="plotly_white",
                margin=dict(l=10, r=100, t=50, b=30),
                coloraxis_showscale=False,
                xaxis_title="Nombre de cas", yaxis_title="Aire de santé"
            )
            st.plotly_chart(fig_cas, use_container_width=True)

    else:
        top10c = (
            top_data_agg
            .nlargest(10, "Cas_Observes")
            .sort_values("Cas_Observes", ascending=True)
            .reset_index(drop=True)
        )
        fig_cas = px.bar(
            top10c, x="Cas_Observes", y="health_area",
            orientation="h", color="Cas_Observes",
            color_continuous_scale="Reds", text="Cas_Observes",
            labels={"Cas_Observes": "Nombre de cas", "health_area": "Aire de santé"},
            title="Top 10 — Nombre de cas (données pop. indisponibles)",
        )
        fig_cas.update_traces(
            texttemplate="%{x} cas", textposition="outside",
            hovertemplate="<b>%{y}</b><br>Cas observés : %{x}<extra></extra>"
        )
        fig_cas.update_layout(
            height=max(400, len(top10c) * 45), template="plotly_white",
            margin=dict(l=10, r=100, t=50, b=30),
            coloraxis_showscale=False,
            xaxis_title="Nombre de cas", yaxis_title="Aire de santé"
        )
        st.plotly_chart(fig_cas, use_container_width=True)
        st.info("ℹ️ Taux d'attaque indisponible — données de population WorldPop non chargées.")

    # ── Pyramide des âges WorldPop ────────────────────────────
    st.header("📊 Pyramide des Âges — Population Enfantine (WorldPop)")
    if donnees_dispo["Population"]:
        tG04   = sa_gdf_enrichi.get("Pop_M_0", pd.Series(0)).fillna(0).sum() + \
                 sa_gdf_enrichi.get("Pop_M_1", pd.Series(0)).fillna(0).sum()
        tG59   = sa_gdf_enrichi.get("Pop_M_5",  pd.Series(0)).fillna(0).sum()
        tG1014 = sa_gdf_enrichi.get("Pop_M_10", pd.Series(0)).fillna(0).sum()
        tF04   = sa_gdf_enrichi.get("Pop_F_0", pd.Series(0)).fillna(0).sum() + \
                 sa_gdf_enrichi.get("Pop_F_1", pd.Series(0)).fillna(0).sum()
        tF59   = sa_gdf_enrichi.get("Pop_F_5",  pd.Series(0)).fillna(0).sum()
        tF1014 = sa_gdf_enrichi.get("Pop_F_10", pd.Series(0)).fillna(0).sum()

        pyr_df = pd.DataFrame({
            "Age":    ["0-4 ans","5-9 ans","10-14 ans"],
            "Garçons": [-float(tG04), -float(tG59), -float(tG1014)],
            "Filles":  [ float(tF04),  float(tF59),  float(tF1014)]
        })
        max_v = max(abs(pyr_df["Garçons"].min()), pyr_df["Filles"].max(), 1)

        fig_pyr = go.Figure()
        fig_pyr.add_trace(go.Bar(
            y=pyr_df["Age"], x=pyr_df["Garçons"], name="Garçons",
            orientation="h", marker_color="#42a5f5",
            text=[f"{abs(int(x)):,}" for x in pyr_df["Garçons"]],
            textposition="inside"
        ))
        fig_pyr.add_trace(go.Bar(
            y=pyr_df["Age"], x=pyr_df["Filles"], name="Filles",
            orientation="h", marker_color="#ec407a",
            text=[f"{int(x):,}" for x in pyr_df["Filles"]],
            textposition="inside"
        ))
        fig_pyr.update_layout(
            title="Pyramide des Âges — Population Enfantine (0-14 ans) — Source : WorldPop",
            xaxis=dict(
                title="Population",
                tickvals=[-max_v, -max_v/2, 0, max_v/2, max_v],
                ticktext=[f"{int(max_v):,}", f"{int(max_v/2):,}", "0",
                          f"{int(max_v/2):,}", f"{int(max_v):,}"],
                range=[-max_v * 1.1, max_v * 1.1]
            ),
            yaxis_title="Tranche d'âge",
            barmode="overlay", height=400, bargap=0.1,
            template="plotly_white", hovermode="y unified"
        )
        st.plotly_chart(fig_pyr, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👦 Garçons (0-14 ans)", f"{int(tG04+tG59+tG1014):,}")
        with col2:
            st.metric("👧 Filles (0-14 ans)", f"{int(tF04+tF59+tF1014):,}")
        with col3:
            tG = tG04 + tG59 + tG1014
            tF = tF04 + tF59 + tF1014
            ratio = (tG / tF * 100) if tF > 0 else 0
            st.metric("⚖️ Ratio G/F", f"{ratio:.1f}%")
    else:
        st.info("📊 Données WorldPop non disponibles")
# ============================================================
# TAB 2 — CARTOGRAPHIE (TOUS BUGS CORRIGÉS)
# ============================================================
with tab2:
    st.header("🗺️ Cartographie de la Situation Actuelle")

    # CORRECTION : garantir géométries valides et CRS correct avant Folium
    sa_gdf_with_cases = sa_gdf_with_cases[
        sa_gdf_with_cases.geometry.notna() &
        ~sa_gdf_with_cases.geometry.is_empty &
        sa_gdf_with_cases.geometry.is_valid
    ].copy()
    if sa_gdf_with_cases.crs is None:
        sa_gdf_with_cases = sa_gdf_with_cases.set_crs("EPSG:4326")
    elif sa_gdf_with_cases.crs.to_epsg() != 4326:
        sa_gdf_with_cases = sa_gdf_with_cases.to_crs("EPSG:4326")

    # ── Fonctions utilitaires ────────────────────────────────
    def safe_float(val):
        try:
            f = float(val)
            return np.nan if np.isinf(f) else f
        except (TypeError, ValueError):
            return np.nan

    def safe_int(val, default=0):
        try:
            f = float(val)
            return default if (np.isnan(f) or np.isinf(f)) else int(f)
        except (TypeError, ValueError):
            return default

    def fmt_val(val, fmt="{:.1f}", suffix="", fallback="N/A"):
        f = safe_float(val)
        return fallback if np.isnan(f) else fmt.format(f) + suffix

    # ── Centrage de la carte ─────────────────────────────────
    try:
        center_lat = float(sa_gdf_with_cases.geometry.centroid.y.mean())
        center_lon = float(sa_gdf_with_cases.geometry.centroid.x.mean())
        if np.isnan(center_lat) or np.isnan(center_lon):
            center_lat, center_lon = 15.0, 2.0
    except Exception:
        center_lat, center_lon = 15.0, 2.0

    # ── Construction de la carte folium ──────────────────────
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True
    )

    # BUG 8 CORRIGÉ : import branca ici (idéalement à mettre en tête de fichier)
    import branca.colormap as cm

    max_cases = safe_float(sa_gdf_with_cases["Cas_Observes"].max())
    max_cases = 1.0 if (np.isnan(max_cases) or max_cases <= 0) else float(max_cases)

    colormap = cm.LinearColormap(
        colors=["#e8f5e9", "#81c784", "#ffeb3b", "#ff9800", "#f44336", "#b71c1c"],
        vmin=0,
        vmax=max_cases,
        caption="Nombre de cas observés"
    )
    colormap.add_to(m)

    # ── Compteurs pour diagnostic ────────────────────────────
    n_poly_ok  = 0
    n_poly_err = 0

    # ── Ajout des polygones row-by-row ───────────────────────
    for _, row in sa_gdf_with_cases.iterrows():
        aire_name    = str(row.get("health_area", "N/A"))
        cas_obs      = safe_int(row.get("Cas_Observes"), 0)
        pop_enfants  = safe_float(row.get("Pop_Enfants", np.nan))
        pop_totale   = safe_float(row.get("Pop_Totale", np.nan))
        taux_attaque = safe_float(row.get("Taux_Attaque_10000", np.nan))
        urbanisation = (
            str(row.get("Urbanisation", "N/A"))
            if pd.notna(row.get("Urbanisation")) else "N/A"
        )
        densite   = safe_float(row.get("Densite_Pop", np.nan))
        taux_vacc = safe_float(row.get("Taux_Vaccination", np.nan))
        temp_moy  = safe_float(row.get("Temperature_Moy", np.nan))
        hum_moy   = safe_float(row.get("Humidite_Moy", np.nan))

        fill_color  = colormap(min(cas_obs, max_cases))
        line_color  = "#b71c1c" if cas_obs >= seuil_alerte_epidemique else "#555555"
        line_weight = 2.5       if cas_obs >= seuil_alerte_epidemique else 0.5

        badge = (
            '<span style="background:#d32f2f;color:white;padding:2px 8px;'
            'border-radius:10px;font-size:11px;">⚠️ ALERTE</span>'
            if cas_obs >= seuil_alerte_epidemique else ""
        )

        popup_html = f"""
        <div style="font-family:Arial;font-size:13px;width:360px;line-height:1.5;">
          <div style="background:#1976d2;color:white;padding:10px 14px;
                      border-radius:6px 6px 0 0;margin:-10px -10px 10px -10px;">
            <b style="font-size:15px;">🏥 {aire_name}</b> {badge}
          </div>
          <b style="color:#d32f2f;">🦠 Épidémiologique</b>
          <table style="width:100%;border-collapse:collapse;margin:6px 0;">
            <tr style="background:#ffeaea;">
              <td style="padding:5px 8px;">🔴 <b>Cas observés</b></td>
              <td style="padding:5px 8px;text-align:right;">
                <b style="font-size:18px;color:#d32f2f;">{cas_obs}</b></td></tr>
            <tr>
              <td style="padding:5px 8px;">📊 Taux d'attaque</td>
              <td style="padding:5px 8px;text-align:right;">
                {fmt_val(taux_attaque, "{:.1f}", " /10 000 enf.")}</td></tr>
          </table>
          <b style="color:#1565c0;">👥 Population</b>
          <table style="width:100%;border-collapse:collapse;margin:6px 0;">
            <tr style="background:#e3f2fd;">
              <td style="padding:5px 8px;">👥 Pop. totale</td>
              <td style="padding:5px 8px;text-align:right;">
                {fmt_val(pop_totale, "{:,.0f}")}</td></tr>
            <tr>
              <td style="padding:5px 8px;">👶 Enfants 0-14 ans</td>
              <td style="padding:5px 8px;text-align:right;">
                {fmt_val(pop_enfants, "{:,.0f}")}</td></tr>
            <tr style="background:#e3f2fd;">
              <td style="padding:5px 8px;">📐 Densité</td>
              <td style="padding:5px 8px;text-align:right;">
                {fmt_val(densite, "{:.1f}", " hab/km²")}</td></tr>
            <tr>
              <td style="padding:5px 8px;">🏙️ Habitat</td>
              <td style="padding:5px 8px;text-align:right;">
                <b>{urbanisation}</b></td></tr>
          </table>
          <b style="color:#2e7d32;">💉 Vaccination &amp; Climat</b>
          <table style="width:100%;border-collapse:collapse;margin:6px 0;">
            <tr style="background:#e8f5e9;">
              <td style="padding:5px 8px;">💉 Taux vaccination</td>
              <td style="padding:5px 8px;text-align:right;">
                {fmt_val(taux_vacc, "{:.1f}", "%")}</td></tr>
            <tr>
              <td style="padding:5px 8px;">🌡️ Température moy.</td>
              <td style="padding:5px 8px;text-align:right;">
                {fmt_val(temp_moy, "{:.1f}", " °C")}</td></tr>
            <tr style="background:#e8f5e9;">
              <td style="padding:5px 8px;">💧 Humidité moy.</td>
              <td style="padding:5px 8px;text-align:right;">
                {fmt_val(hum_moy, "{:.1f}", " %")}</td></tr>
          </table>
        </div>"""

        # ── BUGS 2 & 3 CORRIGÉS ──────────────────────────────
        try:
            geom = row["geometry"]
            if geom is None or geom.is_empty:
                continue

            # BUG 3 : envelopper dans un Feature GeoJSON complet
            geojson_feature = {
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": {}
            }

            folium.GeoJson(
                geojson_feature,
                style_function=lambda x, c=fill_color, w=line_weight, bc=line_color: {
                    "fillColor": c,
                    "color": bc,
                    "weight": w,
                    "fillOpacity": 0.7,
                    "opacity": 0.9
                },
                tooltip=folium.Tooltip(
                    f"<b>{aire_name}</b><br>{cas_obs} cas", sticky=True
                ),
                popup=folium.Popup(popup_html, max_width=420)
            ).add_to(m)
            n_poly_ok += 1

        except Exception as e:
            # BUG 2 : erreur visible au lieu de silencieuse
            n_poly_err += 1
            continue  # on continue sans bloquer l'app

    # ── Diagnostic polygones ─────────────────────────────────
    if n_poly_err > 0:
        st.warning(
            f"⚠️ {n_poly_err} polygone(s) ignoré(s) sur "
            f"{n_poly_ok + n_poly_err} total — vérifiez les géométries du shapefile."
        )
    if n_poly_ok == 0:
        st.error(
            "❌ Aucun polygone n'a pu être ajouté à la carte. "
            "Vérifiez que sa_gdf_with_cases contient des géométries valides en EPSG:4326."
        )

   

    # ── Légende personnalisée ────────────────────────────────
    mc3 = max(max_cases / 3, 1)
    legend_html = f"""
    <div style="position:fixed;bottom:50px;left:50px;width:240px;background:white;
    border:2px solid grey;z-index:9999;font-size:13px;padding:12px;
    border-radius:6px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);">
      <p style="margin:0 0 8px;font-weight:bold;">📊 Légende</p>
      <p style="margin:4px 0;">
        <span style="background:#e8f5e9;padding:2px 10px;border:1px solid #ccc;">Faible</span>
        0–{mc3:.0f} cas</p>
      <p style="margin:4px 0;">
        <span style="background:#ffeb3b;padding:2px 10px;border:1px solid #ccc;">Moyen</span>
        {mc3:.0f}–{2*mc3:.0f} cas</p>
      <p style="margin:4px 0;">
        <span style="background:#f44336;color:white;padding:2px 10px;">Élevé</span>
        &gt; {2*mc3:.0f} cas</p>
      <hr style="margin:8px 0;">
      <p style="margin:4px 0;color:#d32f2f;">
        <b>⚠️ Seuil alerte :</b> {seuil_alerte_epidemique} cas/sem.</p>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

   
    # use_container_width=True → width=1200 (compatible toutes versions streamlit-folium)
    st_folium(m, width=1200, height=650, returned_objects=[], key="carte_principale_tab2")
    # ── Métriques synthèse ───────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        nal = len(
            sa_gdf_with_cases[
                sa_gdf_with_cases["Cas_Observes"] >= seuil_alerte_epidemique
            ]
        )
        st.metric("⚠️ Aires en alerte", nal, f"{nal / len(sa_gdf) * 100:.1f}%")
    with col2:
        nsc = len(sa_gdf_with_cases[sa_gdf_with_cases["Cas_Observes"] == 0])
        st.metric("✅ Aires sans cas", nsc, f"{nsc / len(sa_gdf) * 100:.1f}%")
    with col3:
        d_moy = safe_float(sa_gdf_with_cases["Densite_Pop"].mean())
        st.metric("👥 Densité moy.", fmt_val(d_moy, "{:.1f}", " hab/km²"))
# ============================================================
# TAB 3 — Modelisation
# ============================================================
with tab3:
    st.header("🔬 Modélisation Prédictive par Semaines Épidémiologiques")

    def generer_semaines_futures(derniere_sem, derniere_an, n_weeks):
        futures = []
        sem, an = derniere_sem, derniere_an
        for _ in range(n_weeks):
            sem += 1
            if sem > 52:
                sem = 1
                an += 1
            futures.append({
                "SemaineLabel": f"{an}-S{sem:02d}",
                "SemaineEpi": sem,
                "Annee": an,
                "sort_key": an * 100 + sem
            })
        return futures

    st.markdown(
        f"""<div class="info-box"><b>⚙️ Configuration de la prédiction</b><br>
        - Dernière semaine de données : <b>S{derniere_semaine_epi:02d} {derniere_annee}</b><br>
        - Période de prédiction : <b>{pred_mois} mois ({n_weeks_pred} semaines)</b><br>
        - Modèle sélectionné : <b>{modele_choisi}</b><br>
        - Mode importance : <b>{mode_importance}</b><br>
        - Seuils configurés : Baisse {seuil_baisse}%, Hausse {seuil_hausse}%
        </div>""",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Lancer la Modélisation Prédictive", type="primary",
                     use_container_width=True, key="btn_model_rougeole"):
            st.session_state.prediction_rougeole_lancee = True
    with col2:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="btn_reset_rougeole"):
            st.session_state.prediction_rougeole_lancee = False

    if not st.session_state.prediction_rougeole_lancee:
        st.info("👆 Cliquez sur le bouton ci-dessus pour lancer la modélisation")
        st.stop()

    # ── Agrégation hebdomadaire par aire (contrat historique conservé) ───
    weekly_features = df.groupby(["Aire_Sante", "Annee", "Semaine_Epi"]).agg(
        CasObserves=("ID_Cas", "count"),
        AgeMoyen=("Age_Mois", "mean")
    ).reset_index()

    weekly_features["sort_key"] = weekly_features["Annee"] * 100 + weekly_features["Semaine_Epi"]
    weekly_features["SemaineLabel"] = (
        weekly_features["Annee"].astype(str) + "-S" +
        weekly_features["Semaine_Epi"].astype(str).str.zfill(2)
    )
    weekly_features = weekly_features.sort_values(["Aire_Sante", "sort_key"]).reset_index(drop=True)

    if len(weekly_features) < 10:
        st.error("❌ Données insuffisantes pour entraîner le modèle "
                 f"(seulement {len(weekly_features)} lignes hebdomadaires valides). "
                 "Réduisez les filtres temporels ou utilisez le mode démo.")
        st.stop()

    # ── Modélisation via le noyau `epimodel` ─────────────────────────────
    # Une seule fonction construit les variables, à l'entraînement comme à la
    # prévision : plus aucun décalage train/inférence. Le panneau est complété
    # (aires x semaines, zéros explicites) et l'index temporel est continu,
    # donc les années ne sont plus écrasées entre elles.
    with st.spinner("🤖 Préparation du panneau et entraînement…"):
        try:
            panel_r, design_builder_r, static_r, prep_info_r = \
                epi_app_bridge.prepare_rougeole(
                    df_cases_weekly=weekly_features,
                    sa_gdf_enrichi=sa_gdf_enrichi,
                    df_vaccination=vaccination_df,
                    df_linelist=df,
                    area_col="Aire_Sante",
                )
            res_r = epi_app_bridge.run_modelling_rougeole(
                panel=panel_r,
                design_builder=design_builder_r,
                algo=modele_choisi,
                n_weeks_pred=n_weeks_pred,
                objective=("poisson" if objectif_rougeole.startswith("Poisson")
                           else "squared_error"),
            )
        except Exception as _e:
            st.error(f"❌ Modélisation impossible : {_e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

    st.session_state["rougeole_model"] = res_r
    st.session_state["rougeole_panel_info"] = prep_info_r

    future_df = res_r["future_df"]
    metrics_r = res_r["metrics"]
    feature_cols = res_r["feature_cols"]
    mae = metrics_r["mae"]
    rmse = metrics_r["rmse"]
    r2 = metrics_r["r2"]
    cv_mean = metrics_r["cv_r2_mean"]
    cv_std = metrics_r["cv_r2_std"]
    cv_mae = metrics_r["cv_mae_mean"]

    if future_df is None or future_df.empty:
        st.error("❌ Aucune prédiction générée. Vérifiez que les données contiennent "
                 "suffisamment de semaines (minimum 4) par aire de santé.")
        st.stop()

    # ── Métriques du modèle ────────────────────────────────────────────
    st.subheader("📊 Performance du Modèle")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² CV temporel", f"{cv_mean:.3f} ±{cv_std:.3f}",
                  help="Découpage bloqué par semaine avec embargo : c'est la "
                       "métrique de généralisation à utiliser.")
    with col2:
        st.metric("MAE CV", f"{cv_mae:.1f} cas",
                  help="Erreur absolue moyenne hors échantillon (1 semaine)")
    with col3:
        _moy_obs = float(weekly_features["CasObserves"].mean())
        st.metric("MAE / moyenne", f"{(cv_mae / _moy_obs * 100) if _moy_obs > 0 else float('nan'):.1f} %",
                  help="Erreur relative à la moyenne observée")
    with col4:
        st.metric("R² in-sample", f"{r2:.3f}",
                  help="⚠️ Mesuré sur les données d'entraînement : toujours optimiste")

    st.caption(
        f"🤖 {modele_choisi} · objectif **{metrics_r['objectif']}** · "
        f"{metrics_r['n_features']} variables · {metrics_r['n_train']:,} observations "
        f"d'entraînement. Validation : {metrics_r.get('protocole', 'temporelle bloquée par semaine')}."
    )

    # ── Diagnostic de variabilité temporelle (action F5) ──────────────
    # Une variable constante dans le temps mais différente d'une aire à l'autre
    # passe le filtre de sélection, car elle varie globalement. Elle n'explique
    # pourtant aucune dynamique — ni pic, ni saison — et agit seulement comme
    # décalage de niveau entre aires.
    _dyn_r = res_r.get("feature_dynamics")
    _clim_inv_r = res_r.get("climat_invariant") or []
    if _dyn_r is not None and len(_dyn_r):
        _n_temp_r = int((_dyn_r["role"] == "temporelle").sum())
        _n_niv_r = int((_dyn_r["role"] == "niveau par aire").sum())
        st.caption(
            f"🧭 Variabilité : **{_n_temp_r}** variable(s) varient dans le temps au "
            f"sein des aires, **{_n_niv_r}** sont des niveaux par aire (couverture "
            f"vaccinale, urbanisation, population…) — utiles pour situer le risque, "
            f"incapables d'expliquer une dynamique.")
    if _clim_inv_r:
        st.warning(
            f"⚠️ **Climat invariant dans le temps** : {', '.join(_clim_inv_r)}. Ces "
            f"variables sont retenues par le modèle parce qu'elles diffèrent entre "
            f"aires, mais elles ne varient pas au fil des semaines : elles ne peuvent "
            f"expliquer ni pic ni saison. Pour que le climat apporte de l'information "
            f"prédictive, il faut des précipitations, températures et humidités "
            f"**hebdomadaires** (NASA POWER, ERA5), pas des moyennes par aire.")
    st.info(
        "ℹ️ L'ancien « R² test » provenait d'un `train_test_split` **aléatoire** : des "
        "semaines futures se retrouvaient dans l'entraînement, ce qui surestimait "
        "fortement la performance. La pondération manuelle des variables (mode Expert) "
        "était par ailleurs **sans effet** sur les arbres, invariants par mise à "
        "l'échelle d'une variable ; elle est remplacée par une vraie sélection de "
        "variables et un objectif de perte adapté aux comptages."
    )

    _folds_r = res_r.get("cv_folds")
    if _folds_r is not None and len(_folds_r):
        with st.expander("🧪 Détail de la validation temporelle par fold", expanded=False):
            st.dataframe(
                _folds_r[["fold", "train_weeks", "test_weeks", "n", "mae", "rmse",
                          "r2", "bias", "agg_ratio"]].rename(columns={
                              "fold": "Fold", "train_weeks": "Semaines entraînement",
                              "test_weeks": "Semaines test", "n": "Obs.", "mae": "MAE",
                              "rmse": "RMSE", "r2": "R²", "bias": "Biais",
                              "agg_ratio": "Total prédit/observé"}),
                hide_index=True, use_container_width=True)

    # ── Importance des variables ───────────────────────────────────────
    imp_df = res_r.get("importance")
    if imp_df is not None and len(imp_df):
        st.subheader("🔍 Importance des Variables")
        _imp_show = imp_df.head(20).rename(
            columns={"variable": "Variable", "importance": "Importance",
                     "importance_pct": "Importance (%)"})
        fig_imp = px.bar(
            _imp_show.sort_values("Importance"), x="Importance", y="Variable",
            orientation="h", title="Importance des variables — modèle ML",
            color="Importance", color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "Les variables de couverture vaccinale, de densité d'enfants et "
            "d'urbanisation sont des **covariables statiques** : elles expliquent les "
            "différences structurelles de risque entre aires de santé."
        )

    # ── Génération des prédictions futures ─────────────────────────────

    # ── Courbe épidémique avec prédictions ─────────────────────
    st.subheader("📈 Courbe Épidémique avec Prédictions")
    weekly_obs = weekly_cases[["Semaine_Label", "Cas", "sort_key"]].copy()
    weekly_obs.columns = ["SemaineLabel", "Valeur", "sort_key"]
    weekly_obs["Type"] = "Observé"

    weekly_pred_global = future_df.groupby(["SemaineLabel", "sort_key"])["CasPredits"].sum().reset_index()
    weekly_pred_global.columns = ["SemaineLabel", "sort_key", "Valeur"]
    weekly_pred_global["Type"] = "Prédit"

    combined = pd.concat([weekly_obs, weekly_pred_global], ignore_index=True)
    combined = combined.sort_values("sort_key")

    fig_pred = px.line(combined, x="SemaineLabel", y="Valeur", color="Type",
                       color_discrete_map={"Observé": "#d32f2f", "Prédit": "#1976d2"},
                       title=f"Courbe épidémique observée + prédictions ({n_weeks_pred} semaines)",
                       markers=True)
    derniere_label = weekly_cases.iloc[-1]["Semaine_Label"]
    fig_pred.add_shape(
        type="line",
        x0=derniere_label, x1=derniere_label,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )
    fig_pred.add_annotation(
        x=derniere_label,
        y=1.0,
        xref="x", yref="paper",
        text="Fin données réelles",
        showarrow=False,
        font=dict(color="gray", size=11),
        xanchor="left",
        yanchor="top"
    )
    fig_pred.update_layout(xaxis=dict(tickangle=-45, nticks=25),
                           height=450, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Synthèse des risques par aire ──────────────────────────
    st.subheader("🎯 Synthèse des Risques par Aire de Santé")
    risk_rows = []
    for aire in future_df["Aire_Sante"].unique():
        aire_pred = future_df[future_df["Aire_Sante"] == aire]
        cas_pred_total = aire_pred["CasPredits"].sum()
        semaine_pic    = aire_pred.loc[aire_pred["CasPredits"].idxmax(), "SemaineLabel"]
        aire_obs       = weekly_features[weekly_features["Aire_Sante"] == aire]
        cas_obs_m      = aire_obs["CasObserves"].mean() if len(aire_obs) > 0 else 0
        variation_pct  = (cas_pred_total / n_weeks_pred - cas_obs_m) / (cas_obs_m + 1) * 100

        if variation_pct >= seuil_hausse:
            cat = "Forte hausse"
        elif variation_pct <= -seuil_baisse:
            cat = "Forte baisse"
        elif variation_pct > 0:
            cat = "Légère hausse"
        else:
            cat = "Stable/baisse"

        risk_rows.append({
            "Aire_Sante": aire, "CasPreditsTotal": round(cas_pred_total, 1),
            "VariationPct": round(variation_pct, 1),
            "CategorieVariation": cat, "SemainePic": semaine_pic
        })

    risk_df = pd.DataFrame(risk_rows).sort_values("CasPreditsTotal", ascending=False)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_hausse = len(risk_df[risk_df["CategorieVariation"] == "Forte hausse"])
        st.metric("🔺 Fortes hausses", n_hausse)
    with col2:
        n_baisse = len(risk_df[risk_df["CategorieVariation"] == "Forte baisse"])
        st.metric("🔻 Fortes baisses", n_baisse)
    with col3:
        total_pred = risk_df["CasPreditsTotal"].sum()
        st.metric("📊 Total cas prédits", f"{int(total_pred):,}")
    with col4:
        moy_hebdo_pred = total_pred / n_weeks_pred if n_weeks_pred > 0 else 0
        st.metric("📅 Moy. hebdo prédite", f"{moy_hebdo_pred:.1f}")

    # ── Graphique barres par aire ──────────────────────────────
    color_map_cat = {
        "Forte hausse":  "#f44336",
        "Légère hausse": "#ff9800",
        "Stable/baisse": "#4caf50",
        "Forte baisse":  "#2196f3"
    }
    fig_risk = px.bar(
        risk_df.head(20), x="Aire_Sante", y="CasPreditsTotal",
        color="CategorieVariation", color_discrete_map=color_map_cat,
        title="Top 20 aires — Cas prédits et catégorie de risque",
        text="CasPreditsTotal"
    )
    fig_risk.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_risk.update_layout(xaxis_tickangle=-45, height=500,
                           template="plotly_white", showlegend=True)
    st.plotly_chart(fig_risk, use_container_width=True)

    st.dataframe(
        risk_df.style.format({"CasPreditsTotal": "{:.1f}", "VariationPct": "{:.1f}"})
               .background_gradient(subset=["CasPreditsTotal"], cmap="Reds"),
        use_container_width=True, height=400
    )

    # ── Heatmap prédictions ────────────────────────────────────
    st.subheader("🌡️ Heatmap des Prédictions — Top 15 Aires les Plus à Risque")

    # Sélection des 15 aires avec le plus de cas prédits totaux
    top15_aires = (
        future_df.groupby("Aire_Sante")["CasPredits"]
        .sum()
        .nlargest(15)
        .index
        .tolist()
    )

    heatmap_data = future_df[future_df["Aire_Sante"].isin(top15_aires)].pivot_table(
        index="Aire_Sante", columns="SemaineLabel",
        values="CasPredits", aggfunc="sum"
    ).fillna(0)

    # Tri des colonnes en ordre chronologique
    if len(heatmap_data.columns) > 0:
        sort_map = future_df[["SemaineLabel", "sort_key"]].drop_duplicates().set_index("SemaineLabel")["sort_key"]
        cols_sorted = sorted(heatmap_data.columns, key=lambda c: sort_map.get(c, 0))
        heatmap_data = heatmap_data[cols_sorted]

    if len(heatmap_data) > 0:
        h_heat = max(400, min(600, len(heatmap_data) * 35))
        fig_hm = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=list(heatmap_data.columns),
            y=list(heatmap_data.index),
            colorscale=[
                [0.0,  "rgb(255,255,255)"],
                [0.05, "rgb(255,245,220)"],
                [0.25, "rgb(255,200,100)"],
                [0.50, "rgb(255,140,40)"],
                [0.75, "rgb(220,50,20)"],
                [1.0,  "rgb(120,0,0)"],
            ],
            colorbar=dict(title="Cas prédits", thickness=15),
            hovertemplate="<b>%{y}</b><br>Semaine %{x}<br>Cas prédits : %{z}<extra></extra>",
        ))
        fig_hm.update_layout(
            title=f"Prédictions — Top 15 aires les plus à risque ({n_weeks_pred} semaines)",
            xaxis_title="Semaine",
            yaxis_title="Aire de Santé",
            height=h_heat,
            template="plotly_white",
            xaxis=dict(tickangle=-60, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
            margin=dict(l=160, r=80, t=60, b=120)
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption(f"ℹ️ Heatmap limitée aux {len(heatmap_data)} aires avec le plus grand nombre de cas prédits totaux.")
    else:
        st.info("ℹ️ Pas de données suffisantes pour la heatmap.")

    # ── Cartes géographiques prédictions ──────────────────────
    st.subheader("🗺️ Cartographie des Prédictions")
    gdf_predictions = sa_gdf_enrichi.merge(
        risk_df, left_on="health_area", right_on="Aire_Sante", how="left"
    )
    gdf_predictions["CasPreditsTotal"]    = gdf_predictions["CasPreditsTotal"].fillna(0)
    gdf_predictions["CategorieVariation"] = gdf_predictions["CategorieVariation"].fillna("Stable/baisse")

    try:
        center_lat_p = float(gdf_predictions.geometry.centroid.y.mean())
        center_lon_p = float(gdf_predictions.geometry.centroid.x.mean())
        if np.isnan(center_lat_p) or np.isnan(center_lon_p):
            center_lat_p, center_lon_p = 15.0, 2.0
    except Exception:
        center_lat_p, center_lon_p = 15.0, 2.0

    aires_critiques = gdf_predictions[gdf_predictions["CategorieVariation"] == "Forte hausse"]

    # Carte zones à risque élevé
    st.subheader("🚨 Carte des Zones à Risque Élevé")
    if len(aires_critiques) > 0:
        m_risque = folium.Map(location=[center_lat_p, center_lon_p],
                              zoom_start=6, tiles="CartoDB positron")
        folium.GeoJson(
            gdf_predictions,
            style_function=lambda x: {
                "fillColor": "#e0e0e0", "color": "#999999",
                "weight": 1, "fillOpacity": 0.3
            },
            name="Toutes les aires"
        ).add_to(m_risque)

        for idx, row in aires_critiques.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda x: {
                    "fillColor": "#ff0000", "color": "#8B0000",
                    "weight": 3, "fillOpacity": 0.6
                }
            ).add_to(m_risque)
            folium.Marker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                popup=folium.Popup(
                    f"""<div style="width:250px;font-family:Arial;">
                    <h4 style="color:red;margin:0;">⚠️ ALERTE</h4>
                    <p><b>{row["health_area"]}</b></p>
                    <p>Cas prédits : <b>{row["CasPreditsTotal"]}</b></p>
                    <p>Hausse : <b style="color:red;">{row["VariationPct"]:.1f}%</b></p>
                    <p>Pic : {row["SemainePic"]}</p></div>""",
                    max_width=300
                ),
                icon=folium.Icon(color="red", icon="exclamation-sign")
            ).add_to(m_risque)

        st_folium(m_risque, width=1200, height=600,
                  key="carte_risque_rougeole", returned_objects=[])
        st.error(f"🚨 **{len(aires_critiques)} aires identifiées à risque élevé** — Intervention prioritaire recommandée")
    else:
        st.success("✅ Aucune zone à risque élevé identifiée dans les prédictions")

    # Carte chaleur prédictions
    heat_data_pred = []
    if gdf_predictions["CasPreditsTotal"].sum() > 0:
        st.subheader("🌡️ Carte de Chaleur des Cas Prédits")
        heat_data_pred = [
            [row.geometry.centroid.y, row.geometry.centroid.x, row["CasPreditsTotal"]]
            for idx, row in gdf_predictions.iterrows()
            if row["CasPreditsTotal"] > 0 and row.geometry is not None
        ]
        if len(heat_data_pred) > 0:
            m_heat = folium.Map(location=[center_lat_p, center_lon_p],
                                zoom_start=6, tiles="CartoDB positron")
            HeatMap(heat_data_pred, min_opacity=0.3, max_opacity=0.8,
                    radius=25, blur=20,
                    gradient={0.0: "blue", 0.3: "lime", 0.5: "yellow",
                               0.7: "orange", 1.0: "red"}
            ).add_to(m_heat)
            st_folium(m_heat, width=1200, height=600,
                      key="heatmap_chaleur_pred_rougeole", returned_objects=[])
            st.info("ℹ️ Les zones rouges/oranges indiquent les concentrations de cas prédits les plus élevées")

    # ── Alertes et recommandations ─────────────────────────────
    st.subheader("🔔 Alertes et Recommandations")
    forte_hausse = risk_df[risk_df["CategorieVariation"] == "Forte hausse"]
    if len(forte_hausse) > 0:
        st.error(f"🚨 {len(forte_hausse)} aires en **FORTE HAUSSE** (>{seuil_hausse}%)")
        with st.expander("📋 Détails des aires critiques", expanded=True):
            st.dataframe(
                forte_hausse[["Aire_Sante", "CasPreditsTotal", "VariationPct", "SemainePic"]]
                .style.format({"CasPreditsTotal": "{:.0f}", "VariationPct": "{:.1f}"}),
                use_container_width=True
            )
        st.markdown("**Actions recommandées :**")
        st.markdown("- Intensifier la surveillance épidémiologique")
        st.markdown("- Préparer campagne de vaccination réactive (CVR)")
        st.markdown("- Renforcer stocks de vaccins et intrants")
        st.markdown("- Communication précoce aux équipes terrain")
    else:
        st.success("✅ Aucune aire en forte hausse détectée")

    forte_baisse = risk_df[risk_df["CategorieVariation"] == "Forte baisse"]
    if len(forte_baisse) > 0:
        st.success(f"📉 {len(forte_baisse)} aires en **FORTE BAISSE** (>{seuil_baisse}%)")
        with st.expander("📋 Aires en amélioration"):
            st.dataframe(
                forte_baisse[["Aire_Sante", "CasPreditsTotal", "VariationPct"]]
                .style.format({"CasPreditsTotal": "{:.0f}", "VariationPct": "{:.1f}"}),
                use_container_width=True
            )

    # ── Téléchargements ────────────────────────────────────────
    st.subheader("💾 Téléchargements")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_predictions = future_df.to_csv(index=False)
        st.download_button(
            label="📥 Prédictions détaillées (CSV)",
            data=csv_predictions,
            file_name=f"predictions_rougeole_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True, key="dl_pred_csv"
        )
    with col2:
        csv_synthese = risk_df.to_csv(index=False)
        st.download_button(
            label="📥 Synthèse par aire (CSV)",
            data=csv_synthese,
            file_name=f"synthese_risque_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True, key="dl_synth_csv"
        )
    with col3:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            risk_df.to_excel(writer, sheet_name="Synthèse", index=False)
            future_df.to_excel(writer, sheet_name="Prédictions détaillées", index=False)
            heatmap_data.to_excel(writer, sheet_name="Heatmap")
        st.download_button(
            label="📥 Rapport complet (Excel)",
            data=output.getvalue(),
            file_name=f"rapport_predictions_rougeole_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, key="dl_rapport_excel"
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        geojson_predictions = gdf_predictions.to_json()
        st.download_button(
            label="📥 Carte prédictions (GeoJSON)",
            data=geojson_predictions,
            file_name=f"carte_predictions_rougeole_{datetime.now().strftime('%Y%m%d')}.geojson",
            mime="application/json",
            use_container_width=True, key="dl_geojson_pred"
        )
    with col5:
        if len(aires_critiques) > 0:
            geojson_risque = aires_critiques.to_json()
            st.download_button(
                label="📥 Zones à risque (GeoJSON)",
                data=geojson_risque,
                file_name=f"zones_risque_rougeole_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/json",
                use_container_width=True, key="dl_geojson_risque"
            )

    st.markdown("---")
    st.success("✅ Modélisation terminée avec succès !")
    st.info("💡 Ajustez les paramètres dans la sidebar pour relancer une nouvelle prédiction")
# ============================================================
# TAB 4 — VALIDATION RÉTROSPECTIVE (module externe)
# ============================================================
with tab4:
    # Reconstruire df_cases agrégé par aire + semaine (format attendu par le module)
    df_cases_val = (
        df.groupby(["Aire_Sante", "Annee", "Semaine_Epi"])
        .size()
        .reset_index(name="CasObserves")
        .rename(columns={"Aire_Sante": "health_area"})
    )
    df_cases_val["sort_key"] = df_cases_val["Annee"] * 100 + df_cases_val["Semaine_Epi"]

    create_validation_tab_rougeole(
        df_cases=df_cases_val,
        gdf_health=sa_gdf_enrichi,
        model_results=None
    )

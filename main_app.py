"""
============================================================
APPLICATION PRINCIPALE - PLATEFORME SURVEILLANCE ÉPIDÉMIOLOGIQUE
Développée pour Médecins Sans Frontières (MSF)
============================================================
"""

import streamlit as st
import os
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# Configuration de la page (DOIT être la première commande Streamlit)
st.set_page_config(
    page_title="MSF - Surveillance Épidémiologique",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# AUTHENTIFICATION
# ============================================================
# ── Emplacement des identifiants ─────────────────────────────────────────
# Correction (action I1 de l'audit) : le chemin n'est plus codé en dur. En
# production, les identifiants doivent vivre HORS du dépôt — variable
# d'environnement pointant vers un fichier non versionné, ou secret monté par
# l'orchestrateur. Le repli sur ./credentials.yaml est conservé pour le
# développement local.
CREDENTIALS_PATH = os.environ.get("EPI_CREDENTIALS_PATH", "credentials.yaml")

try:
    with open(CREDENTIALS_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

    # Nouvelle API : location doit être 'main', 'sidebar' ou 'unrendered'
    authenticator.login(location="main")

    if st.session_state.get("authentication_status"):
        authenticator.logout("Se déconnecter", location="sidebar")
        st.sidebar.success(f"Connecté : {st.session_state.get('name', '')}")
    elif st.session_state.get("authentication_status") is False:
        st.error("Nom d'utilisateur ou mot de passe incorrect")
        st.stop()
    else:
        st.warning("Merci de saisir vos identifiants.")
        st.stop()

except FileNotFoundError:
    st.error(
        f"Fichier d'identifiants introuvable : `{CREDENTIALS_PATH}`.\n\n"
        "Copiez `credentials.yaml.example` vers ce chemin, renseignez un mot de "
        "passe **haché** et une clé de cookie aléatoire, ou faites pointer la "
        "variable d'environnement `EPI_CREDENTIALS_PATH` vers un fichier placé "
        "hors du dépôt."
    )
    st.stop()
except Exception as e:
    st.error(f"Erreur d'authentification : {e}")
    st.stop()

# CSS personnalisé avec branding MSF
st.markdown("""
<style>
    .header-banner {
        background: linear-gradient(135deg, #E4032E 0%, #C4032A 100%);
        border-radius: 12px;
        padding: 2rem 1rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(228, 3, 46, 0.2);
    }
    .msf-logo {
        font-size: 1.3rem;
        font-weight: bold;
        letter-spacing: 3px;
        margin-bottom: 0.5rem;
    }
    .header-banner h1 { font-size: 2rem; margin: 0.5rem 0; font-weight: 600; }
    .header-banner p  { font-size: 1rem; margin: 0.3rem 0; opacity: 0.95; }
    .app-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #E4032E;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .app-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.12); }
    .app-card h3 { color: #E4032E; margin-top: 0; font-size: 1.8rem; font-weight: bold; }
    .app-card h4 { color: #58595B; margin-top: 0.3rem; font-size: 1.1rem; font-weight: normal; }
    .app-card p  { color: #333; line-height: 1.6; }
    .app-card ul { list-style: none; padding-left: 0; line-height: 1.7; color: #555; }
    .app-card li { margin: 0.4rem 0; }
    .app-card strong { color: #E4032E; font-weight: 600; }
    .stButton > button {
        width: 100%;
        background: #E4032E;
        color: white;
        border: none;
        padding: 0.9rem 2rem;
        font-size: 1.05rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(228, 3, 46, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: #C4032A;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(228, 3, 46, 0.4);
    }
    .info-section {
        background: #F8F9FA;
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        border-left: 4px solid #E4032E;
    }
    .info-section h2 { color: #E4032E; text-align: center; margin-bottom: 1rem; }
    .doc-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-top: 3px solid #E4032E;
        height: 100%;
    }
    .doc-card h3 { color: #E4032E; font-size: 1.3rem; margin-bottom: 0.5rem; }
    .doc-card p  { color: #58595B; line-height: 1.5; }
    .footer {
        text-align: center;
        color: #58595B;
        padding: 2rem;
        background: #F8F9FA;
        border-radius: 10px;
        border-top: 3px solid #E4032E;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALISATION DE L'ÉTAT
# ============================================================
if 'page_choice' not in st.session_state:
    st.session_state.page_choice = "Accueil"

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.selectbox(
        "Choisir une application",
        ["Accueil", "Paludisme", "Rougeole", "Manuel"],
        index=["Accueil", "Paludisme", "Rougeole", "Manuel"].index(st.session_state.page_choice)
    )
    if page != st.session_state.page_choice:
        st.session_state.page_choice = page
        st.rerun()

# ============================================================
# FONCTION POUR CHARGER LES APPLICATIONS
# ============================================================
def load_app(filename):
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()
            lines = code.split('\n')
            cleaned_lines = []
            skip_next = 0
            for line in lines:
                if skip_next > 0:
                    skip_next -= 1
                    if ')' in line:
                        skip_next = 0
                    continue
                if 'st.set_page_config' in line:
                    if ')' not in line:
                        skip_next = 10
                    continue
                cleaned_lines.append(line)
            exec('\n'.join(cleaned_lines), globals())
        else:
            st.error(f"❌ Fichier '{filename}' introuvable")
            if st.button("🏠 Retour à l'accueil"):
                st.session_state.page_choice = "Accueil"
                st.rerun()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de {filename}")
        st.code(str(e))
        with st.expander("📋 Détails de l'erreur"):
            import traceback
            st.code(traceback.format_exc())
        if st.button("🏠 Retour à l'accueil"):
            st.session_state.page_choice = "Accueil"
            st.rerun()

# ============================================================
# ROUTAGE
# ============================================================
if st.session_state.page_choice == "Paludisme":
    load_app("app_paludisme.py")
elif st.session_state.page_choice == "Rougeole":
    load_app("app_rougeole.py")
elif st.session_state.page_choice == "Manuel":
    load_app("app_manuel.py")
else:
    st.markdown("""
    <div class="header-banner">
        <div class="msf-logo">⚕️ MÉDECINS SANS FRONTIÈRES</div>
        <h1>Plateforme de Surveillance Épidémiologique</h1>
        <p>Outils d'analyse, cartographie et prédiction pour le paludisme et la rougeole</p>
        <p style="font-size:0.9rem; opacity:0.85;">Afrique de l'Ouest | Operational Research & Innovation</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin:1.5rem 0;">
        <h2 style="color:#E4032E; font-size:1.8rem;">Choisissez votre module d'analyse</h2>
        <p style="font-size:1.1rem; color:#58595B;">Cliquez sur les boutons ci-dessous ou utilisez le menu dans la barre latérale</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="app-card">
            <h3>🦟 Paludisme</h3>
            <h4>Outil d'analyse et de prédiction avancée</h4>
            <p>Cartographie interactive, données environnementales et climatiques pour identifier les zones à risque.</p>
            <ul>
                <li>• <strong>Cartographie dynamique</strong> avec popups enrichis</li>
                <li>• <strong>Données démographiques</strong> WorldPop</li>
                <li>• <strong>Analyse climatique</strong> NASA POWER API</li>
                <li>• <strong>Prédiction ML</strong> avec validation croisée temporelle</li>
                <li>• <strong>Clustering géographique</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🦟 Accéder à l'application Paludisme", key="btn_palu_home"):
            st.session_state.page_choice = "Paludisme"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="app-card">
            <h3>🦠 Rougeole</h3>
            <h4>Surveillance et prédiction par semaines épidémiologiques</h4>
            <p>Analyse des épidémies de rougeole avec suivi temporel précis et évaluation des couvertures vaccinales.</p>
            <ul>
                <li>• <strong>Suivi hebdomadaire</strong> par semaines épidémiologiques</li>
                <li>• <strong>Couverture vaccinale</strong> et susceptibilité</li>
                <li>• <strong>Prédiction avancée</strong> Gradient Boosting / Random Forest</li>
                <li>• <strong>Multi-pays</strong> : Niger, BFA, Mali, Mauritanie</li>
                <li>• <strong>Pyramide des âges</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🦠 Accéder à l'application Rougeole", key="btn_rougeole_home"):
            st.session_state.page_choice = "Rougeole"
            st.rerun()

    st.markdown("""
    <div class="info-section">
        <h2>📚 Documentation et Ressources</h2>
        <p style="text-align:center; color:#58595B;">Guides complets, méthodologies et bonnes pratiques</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="doc-card"><h3>📖 Manuel d\'utilisation</h3><p>Guide détaillé pour utiliser chaque module et interpréter les résultats.</p></div>', unsafe_allow_html=True)
        if st.button("📖 Consulter le manuel", key="btn_manuel_home"):
            st.session_state.page_choice = "Manuel"
            st.rerun()
    with col2:
        st.markdown('<div class="doc-card"><h3>🔬 Méthodologie</h3><p>Algorithmes ML, validation croisée temporelle et feature engineering.</p></div>', unsafe_allow_html=True)
        if st.button("🔬 Voir la méthodologie", key="btn_methodo_home"):
            st.session_state.page_choice = "Manuel"
            st.rerun()
    with col3:
        st.markdown('<div class="doc-card"><h3>💡 Glossaire</h3><p>Définitions des variables, lags, moyennes mobiles, ACP et concepts clés.</p></div>', unsafe_allow_html=True)
        if st.button("💡 Accéder au glossaire", key="btn_glossaire_home"):
            st.session_state.page_choice = "Manuel"
            st.rerun()

    st.markdown("""
    <div style="text-align:center; margin:2.5rem 0 1.5rem 0;">
        <h2 style="color:#E4032E; font-size:1.8rem;">⚙️ Caractéristiques Techniques</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="doc-card"><h3>🗺️ Cartographie</h3><ul style="list-style:none;padding:0;color:#58595B;"><li>✓ Folium interactif</li><li>✓ Popups enrichis</li><li>✓ Couches multiples</li><li>✓ Export GeoJSON</li></ul></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="doc-card"><h3>🤖 Machine Learning</h3><ul style="list-style:none;padding:0;color:#58595B;"><li>✓ Gradient Boosting</li><li>✓ Random Forest</li><li>✓ Validation temporelle</li><li>✓ R² > 0.80 typique</li></ul></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="doc-card"><h3>📊 Sources Données</h3><ul style="list-style:none;padding:0;color:#58595B;"><li>✓ NASA POWER API</li><li>✓ WorldPop (GEE)</li><li>✓ Rasters environnement</li><li>✓ Linelists épidémio</li></ul></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <p style="font-size:1.2rem;font-weight:bold;color:#E4032E;margin-bottom:0.5rem;">⚕️ MÉDECINS SANS FRONTIÈRES</p>
        <p style="font-size:1rem;margin:0.5rem 0;"><strong>Développé par Youssoupha MBODJI</strong></p>
        <p style="margin-top:1rem;font-size:0.9rem;">Version 3.0 | © 2026 MSF</p>
        <p style="font-size:0.85rem;margin-top:1rem;font-style:italic;color:#7f8c8d;">
            "Bringing medical assistance to people affected by conflict, epidemics, disasters, or exclusion from healthcare"
        </p>
    </div>
    """, unsafe_allow_html=True)

"""
============================================================
MANUEL D'UTILISATION - PLATEFORME SURVEILLANCE ÉPIDÉMIOLOGIQUE
Documentation complète Paludisme + Rougeole
============================================================
"""

import streamlit as st

# CSS personnalisé pour le manuel
st.markdown("""
<style>
    /* Styles des cartes info */
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .benefit-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .alert-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Titres dans les cartes */
    .info-card h3, .info-card h4, .info-card h5,
    .benefit-box h3, .benefit-box h4, .benefit-box h5,
    .warning-box h3, .warning-box h4, .warning-box h5,
    .alert-box h3, .alert-box h4, .alert-box h5 {
        margin-top: 0;
        color: #2c3e50;
    }
    
    /* Listes */
    .info-card ul, .benefit-box ul, .warning-box ul, .alert-box ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .info-card li, .benefit-box li, .warning-box li, .alert-box li {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.title("📚 Manuel d'Utilisation - Plateforme de Surveillance Épidémiologique")
st.markdown("*Guide complet pour l'utilisation des modules Paludisme et Rougeole*")
st.markdown("---")

# Onglets principaux
tab_palu, tab_rougeole, tab_glossaire = st.tabs([
    "🦟 Paludisme", 
    "🦠 Rougeole",
    "📖 Glossaire & Méthodologie"
])

# ============================================================
# TAB 1 : PALUDISME
# ============================================================

with tab_palu:
    st.header("🦟 Application de Surveillance du Paludisme")
    
    # Section 1 : Introduction
    st.markdown("## 📋 C'est quoi EpiPalu Predict ?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>En bref</h3>
            <p style="font-size:1.1rem;">EpiPalu Predict est un <b>outil intelligent</b> qui vous aide à :</p>
            <ul style="font-size:1.05rem; line-height:1.8;">
                <li><b>Visualiser</b> où se trouvent les cas de paludisme</li>
                <li><b>Comprendre</b> l'influence du climat (pluie, chaleur)</li>
                <li><b>Prévoir</b> où les cas vont augmenter (1 à 12 semaines)</li>
                <li><b>Alerter</b> les zones à risque élevé</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="benefit-box">
            <h4>Pourquoi c'est utile ?</h4>
            <ul style="line-height:1.8;">
                <li><b>Gagner du temps</b> : Analyse automatique en quelques clics</li>
                <li><b>Anticiper</b> : Préparer les interventions avant les pics</li>
                <li><b>Optimiser</b> : Mieux répartir les ressources (médicaments, moustiquaires)</li>
                <li><b>Décider</b> : S'appuyer sur des données objectives</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 2 : Sources de données
    st.markdown("## 📊 Sources de Données Intégrées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🌍 Données Géographiques</h4>
            <ul>
                <li><b>Aires de santé</b> : Zones administratives sanitaires (GeoJSON/Shapefile)</li>
                <li><b>Cours d'eau</b> : Réseau hydrographique (zones de reproduction moustiques)</li>
                <li><b>Altitude</b> : Modèle numérique de terrain (rasters)</li>
                <li><b>Zones inondables</b> : Risque d'inondation (rasters)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>🌡️ Données Climatiques</h4>
            <ul>
                <li><b>NASA POWER API</b> (gratuit) :
                    <ul>
                        <li>Température quotidienne (°C)</li>
                        <li>Précipitations (mm/jour)</li>
                        <li>Humidité relative (%)</li>
                    </ul>
                </li>
                <li><b>Agrégation</b> : Moyennes/totaux hebdomadaires par aire de santé</li>
                <li><b>Période</b> : Données disponibles depuis 1981</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>👥 Données Démographiques (NOUVEAU)</h4>
            <ul>
                <li><b>WorldPop</b> (Google Earth Engine) :
                    <ul>
                        <li><b>Population totale</b> : Nombre d'habitants par aire</li>
                        <li><b>Enfants 0-14 ans</b> : Population pédiatrique (plus vulnérable)</li>
                        <li><b>Densité de population</b> : Habitants/km²</li>
                        <li><b>Tranches d'âge détaillées</b> : 0-4, 5-9, 10-14... jusqu'à 30-34 ans</li>
                    </ul>
                </li>
                <li><b>Résolution</b> : 100m (précision quartier)</li>
                <li><b>Mise à jour</b> : Données annuelles (dernière : 2020)</li>
            </ul>
            <h5>🎯 Utilité pour l'analyse du paludisme</h5>
            <ul style="background:#fff9c4; padding:1rem; border-radius:5px;">
                <li><b>Calcul taux d'incidence</b> : Cas pour 10 000 habitants (indicateur épidémiologique standard)</li>
                <li><b>Priorisation zones à risque</b> : Densité forte + cas élevés = intervention urgente</li>
                <li><b>Estimation besoins</b> : Population enfants → doses médicaments/moustiquaires</li>
                <li><b>Coefficient d'ajustement prédictions</b> : Le modèle ajuste ses prévisions selon la pression démographique (risque relatif par zone)</li>
                <li><b>Identification poches vulnérables</b> : Enfants 0-14 ans = 60-80% des cas graves de paludisme</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>📋 Données Épidémiologiques</h4>
            <ul>
                <li><b>Cas hebdomadaires</b> : Nombre de cas confirmés par semaine et aire</li>
                <li><b>Décès</b> : Mortalité palustre</li>
                <li><b>Format attendu</b> : CSV avec colonnes health_area, week_, cases, deaths</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 3 : Guide d'utilisation
    st.markdown("## 🚀 Guide d'Utilisation Pas-à-Pas")
    
    with st.expander("**Étape 1️⃣ : Charger les aires de santé**", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>Comment faire ?</h4>
            <ol>
                <li>Dans la <b>sidebar</b> (barre latérale gauche), cliquez sur <b>"🗺️ Aires de Santé"</b></li>
                <li>Uploadez votre fichier (formats acceptés : .shp, .geojson, .zip)</li>
                <li>Le fichier doit contenir les colonnes :
                    <ul>
                        <li><code>health_area</code> : Nom de l'aire de santé</li>
                        <li><code>geometry</code> : Géométrie (polygones)</li>
                    </ul>
                </li>
            </ol>
            <h5>✅ Validation</h5>
            <p>Vous devez voir : <code>✓ X aires de santé chargées</code></p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 2️⃣ : Charger les cas de paludisme**", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>Format CSV attendu</h4>
            <p><b>Colonnes obligatoires :</b></p>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:0.5rem; border:1px solid #ddd;">Colonne</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Description</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Exemple</th>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><code>health_area</code></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Nom de l'aire (doit correspondre au fichier géographique)</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Dakar Centre</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><code>week_</code></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Numéro ou nom de semaine</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">S01, 2024-W01</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><code>cases</code></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Nombre de cas confirmés</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">45</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><code>deaths</code></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Nombre de décès (optionnel)</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">2</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 3️⃣ : Activer les données climatiques (optionnel mais recommandé)**", expanded=False):
        st.markdown("""
        <div class="benefit-box">
            <h4>⚡ API Climat - GRATUIT et RAPIDE</h4>
            <p><b>NASA POWER API</b> : Données météo depuis 1981, sans inscription</p>
            <h5>Activation :</h5>
            <ol>
                <li>Cochez <code>☑️ Activer API Climat</code> dans la sidebar</li>
                <li>Sélectionnez l'API : <b>NASA POWER (recommandé)</b></li>
                <li>Cliquez sur <b>"📥 Télécharger Données Climat"</b></li>
                <li>Patientez 10-30 secondes (selon nombre d'aires)</li>
            </ol>
            <h5>Variables obtenues (par aire et par semaine) :</h5>
            <ul>
                <li>🌡️ <b>Température moyenne</b> : Impact direct sur cycle de reproduction moustique</li>
                <li>🌧️ <b>Précipitations totales</b> : Gîtes larvaires (eau stagnante)</li>
                <li>💧 <b>Humidité relative</b> : Survie des moustiques adultes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h5>⚠️ Pourquoi c'est important ?</h5>
            <p>Le paludisme est une <b>maladie climatosensible</b> :</p>
            <ul>
                <li><b>Température optimale</b> : 25-30°C (accélère développement parasite dans moustique)</li>
                <li><b>Pluies</b> : Créent gîtes larvaires → explosion population moustiques 7-10 jours après</li>
                <li><b>Humidité > 60%</b> : Augmente longévité moustiques femelles (transmission prolongée)</li>
            </ul>
            <p><b>Impact sur prédictions :</b> +20-30% de précision avec climat vs sans climat</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 4️⃣ : Activer les données démographiques (NOUVEAU)**", expanded=False):
        st.markdown("""
        <div class="benefit-box">
            <h4>👥 WorldPop - Population haute résolution</h4>
            <p><b>Google Earth Engine</b> : Données populationnelles mondiales, résolution 100m</p>
            <h5>Activation :</h5>
            <ol>
                <li>Cochez <code>☑️ Activer WorldPop (GEE)</code> dans la sidebar</li>
                <li>Assurez-vous que GEE est connecté (voir <code>✓ GEE connecté</code> en haut)</li>
                <li>Cliquez sur <b>"📥 Extraire Population"</b></li>
                <li>Patientez 20-60 secondes (calcul par aire de santé)</li>
            </ol>
            <h5>Données extraites :</h5>
            <table style="width:100%; border-collapse:collapse; margin:1rem 0;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:0.5rem; border:1px solid #ddd;">Variable</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Description</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Utilisation</th>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Pop_Totale</b></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Nombre total d'habitants</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Dénominateur taux d'incidence</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Pop_Enfants_0_14</b></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Population pédiatrique</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Estimation besoins en MII/médicaments</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Densite_Pop</b></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Habitants par km²</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Risque de transmission (densité élevée = plus de contacts)</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Pop_M/F_0_4, 5_9...</b></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Tranches d'âge par sexe</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Pyramide des âges (visualisation)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box">
            <h5>🎯 Impact sur l'analyse et la prédiction</h5>
            <h6>1. Calcul automatique du taux d'incidence</h6>
            <p><code>Taux d'incidence = (Cas / Pop_Totale) × 10 000</code></p>
            <p>Permet de comparer le risque entre zones de tailles différentes</p>
            <h6>2. Coefficient d'ajustement démographique</h6>
            <p>Le modèle prédictif calcule un <b>coefficient de risque relatif</b> pour chaque aire :</p>
            <ul>
                <li><b>Coefficient > 1</b> : Zone à risque plus élevé que la moyenne (ex: forte densité + faible altitude)</li>
                <li><b>Coefficient < 1</b> : Zone à risque plus faible (ex: faible densité, zone urbaine bien drainée)</li>
                <li><b>Coefficient = 1</b> : Risque moyen</li>
            </ul>
            <p><b>Utilisation :</b> Les prédictions sont multipliées par ce coefficient pour mieux refléter le risque local</p>
            <h6>3. Priorisation des interventions</h6>
            <p>Le tableau de bord affiche automatiquement :</p>
            <ul>
                <li>🔴 <b>Zones prioritaires</b> : Cas élevés + population enfants élevée + densité forte</li>
                <li>🟡 <b>Zones à surveiller</b> : Taux d'incidence croissant + coefficient risque > 1.2</li>
            </ul>
            <h6>4. Estimation des besoins en ressources</h6>
            <p>Calculs automatiques basés sur Pop_Enfants_0_14 :</p>
            <ul>
                <li>• Moustiquaires (1 MII pour 2 enfants) : Pop_Enfants / 2</li>
                <li>• Doses TDR (20% population exposée) : Pop_Totale × 0.20</li>
                <li>• ACT (15% cas confirmés) : Cas_prédits × 0.15</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 5️⃣ : Analyser les données (Onglets)**", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>📊 Onglet 1 : Tableau de Bord</h4>
            <ul>
                <li><b>Métriques clés</b> : Total cas, décès, taux de létalité, population exposée</li>
                <li><b>Graphiques temporels</b> : Évolution hebdomadaire cas/décès + climat</li>
                <li><b>Top 10 aires à risque</b> : Classement par nombre de cas</li>
                <li><b>Pyramide des âges</b> (si WorldPop activé) : Structure démographique des zones affectées</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>🗺️ Onglet 2 : Cartographie</h4>
            <ul>
                <li><b>Carte climatique</b> (si données dispo) : Visualisation température/pluie/humidité par semaine</li>
                <li><b>Carte épidémiologique</b> :
                    <ul>
                        <li>Choroplèthe : Intensité couleur = nombre de cas</li>
                        <li>Cercles proportionnels : Taille = nombre de cas</li>
                        <li>Heatmap : Zones de concentration</li>
                    </ul>
                </li>
                <li><b>Popups enrichis</b> : Clic sur une zone affiche :
                    <ul>
                        <li>📊 Cas et décès</li>
                        <li>👥 Population et densité (si WorldPop)</li>
                        <li>🌡️ Climat (si API activée)</li>
                        <li>🌊 Environnement (altitude, inondation, distance rivière)</li>
                    </ul>
                </li>
                <li><b>Couches activables</b> : Rivières, zones inondables, altitude</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>🔮 Onglet 3 : Prédiction</h4>
            <ul>
                <li><b>Configuration</b> :
                    <ul>
                        <li>Sélection algorithme (Gradient Boosting recommandé)</li>
                        <li>Période prédiction (1-12 mois)</li>
                        <li>Options avancées : PCA, clustering spatial, lag spatial</li>
                    </ul>
                </li>
                <li><b>Résultats</b> :
                    <ul>
                        <li>Graphique prédictions vs données réelles</li>
                        <li>Métriques performance (R², MAE, RMSE)</li>
                        <li>Intervalle de confiance</li>
                        <li>Alertes zones à risque (cas prédits > seuil)</li>
                    </ul>
                </li>
                <li><b>Carte prédictive</b> : Visualisation spatiale des prévisions</li>
                <li><b>Export résultats</b> : CSV avec prédictions par aire et semaine</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>📈 Onglet 4 : Analyse de Corrélation</h4>
            <ul>
                <li><b>Matrice de corrélation complète</b> : Toutes variables vs toutes variables</li>
                <li><b>Corrélations avec cas</b> :
                    <ul>
                        <li>Positives : Variables augmentent avec les cas (ex: pluie, humidité)</li>
                        <li>Négatives : Variables diminuent avec les cas (ex: altitude)</li>
                    </ul>
                </li>
                <li><b>Scatter plots</b> : Visualisation corrélations fortes (|r| > 0.3)</li>
                <li><b>Coefficient d'ajustement population</b> : Risque relatif par zone (si WorldPop)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>💾 Onglet 5 : Export</h4>
            <p>Téléchargez toutes vos données :</p>
            <ul>
                <li><b>GeoJSON</b> : Aires de santé, rivières</li>
                <li><b>CSV</b> : Cas, climat, population, prédictions</li>
                <li><b>ZIP complet</b> : Archive avec tous les fichiers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 4 : Conseils d'utilisation
    st.markdown("## 💡 Conseils et Bonnes Pratiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="benefit-box">
            <h4>✅ Pour des prédictions optimales</h4>
            <ul>
                <li><b>Données historiques</b> : Au moins 20-30 semaines de données</li>
                <li><b>Activer API Climat</b> : +20-30% précision</li>
                <li><b>Activer WorldPop</b> : Coefficient ajustement démographique</li>
                <li><b>Ajouter rasters environnement</b> : Inondation, altitude, rivières</li>
                <li><b>Utiliser Gradient Boosting</b> : Meilleur algorithme pour séries temporelles</li>
                <li><b>Activer options avancées</b> :
                    <ul>
                        <li>PCA : Si > 15 variables (évite sur-apprentissage)</li>
                        <li>Clustering spatial : Si zones géographiques hétérogènes</li>
                        <li>Lag spatial : Si transmission inter-zones importante</li>
                    </ul>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Limites et précautions</h4>
            <ul>
                <li><b>Qualité données</b> : Garbage in, garbage out
                    <ul>
                        <li>Vérifiez cohérence (pas de valeurs aberrantes)</li>
                        <li>Complétude : Évitez semaines manquantes</li>
                    </ul>
                </li>
                <li><b>Prédictions long terme</b> : Plus c'est loin, moins c'est précis
                    <ul>
                        <li>1 à 4 semaines : horizon recommandé</li>
                        <li>1 à 2 mois : indicatif, à interpréter avec prudence</li>
                        <li>au-delà : le modèle extrapole une tendance, il ne prédit
                            plus une dynamique épidémique</li>
                        <li>Ces horizons sont des ordres de grandeur. La précision
                            effective est celle que rapportent le CV‑R² et le MAE CV
                            après entraînement sur vos données.</li>
                    </ul>
                </li>
                <li><b>Événements exceptionnels</b> : Le modèle ne prédit pas :
                    <ul>
                        <li>Épidémies inhabituelles (nouveau sérotype)</li>
                        <li>Catastrophes naturelles soudaines</li>
                        <li>Campagnes massives (distributions MII)</li>
                    </ul>
                </li>
                <li><b>WorldPop</b> : Données 2020 (peut être obsolète zones urbanisation rapide)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2 : ROUGEOLE (COMPLET)
# ============================================================

with tab_rougeole:
    st.header("🦠 Application de Surveillance de la Rougeole")
    
    # Section 1 : Introduction
    st.markdown("## 📋 Qu'est-ce que l'application Rougeole ?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>En bref</h3>
            <p style="font-size:1.1rem;">Outil spécialisé de <b>surveillance et prédiction</b> des épidémies de rougeole :</p>
            <ul style="font-size:1.05rem; line-height:1.8;">
                <li><b>Suivi temporel précis</b> : Analyse par semaines épidémiologiques</li>
                <li><b>Détection précoce</b> : Alertes automatiques basées sur seuils historiques</li>
                <li><b>Évaluation couverture vaccinale</b> : Identification poches de susceptibilité</li>
                <li><b>Prédiction ML</b> : Anticipation flambées 4-12 semaines à l'avance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="benefit-box">
            <h4>Pourquoi spécifique Rougeole ?</h4>
            <ul style="line-height:1.8;">
                <li><b>Haute contagiosité</b> : R₀ = 12-18 (vs paludisme non transmissible inter-humain)</li>
                <li><b>Vaccination clé</b> : Seuil immunité collective 95% (analyse couverture cruciale)</li>
                <li><b>Épidémies explosives</b> : Détection rapide = intervention précoce</li>
                <li><b>Cibles vaccinales</b> : Enfants 0-14 ans = 90% des cas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Bloc explicatif détaillé sur R0 et 95%
    st.markdown("""
    <div class="info-card">
        <h4>Comprendre R₀ et le seuil de 95&nbsp;% de vaccination</h4>
        <p><b>R₀</b> (taux de reproduction de base) correspond au nombre moyen de personnes infectées par un cas dans une population totalement sensible.</p>
        <ul>
            <li>Pour la rougeole, la plupart des études situent R₀ entre <b>12 et 18</b> dans des populations non immunisées.</li>
            <li>Le <b>seuil d’immunité collective</b> se calcule approximativement par <code>1 - 1/R₀</code>, soit &gt; 90&nbsp;% pour la rougeole.</li>
            <li>En pratique, les programmes de vaccination visent une <b>couverture &ge; 95&nbsp;%</b> chez les enfants pour casser les chaînes de transmission.</li>
        </ul>
        <p><i>Lecture simple :</i> avec un virus aussi contagieux, laisser plus de 5–10&nbsp;% d’enfants non vaccinés suffit pour alimenter des flambées.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 2 : Sources de données
    st.markdown("## 📊 Sources de Données Intégrées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🌍 Données Géographiques</h4>
            <ul>
                <li><b>Multi-pays</b> : Support Niger, Burkina Faso, Mali, Mauritanie</li>
                <li><b>Base locale</b> : Fichier <code>ao_hlthArea.zip</code> intégré
                    <ul>
                        <li>Filtrage automatique par code ISO3 pays</li>
                        <li>Colonnes : <code>iso3</code>, <code>health_area</code>, <code>geometry</code></li>
                    </ul>
                </li>
                <li><b>Upload personnalisé</b> : Shapefile/GeoJSON custom accepté</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>👥 Données Démographiques (WorldPop)</h4>
            <ul>
                <li><b>Population par tranches d'âge détaillées</b> :
                    <ul>
                        <li>0-4 ans, 5-9 ans, 10-14 ans (cibles vaccination)</li>
                        <li>15-19 ans, 20-24 ans, 25-29 ans, 30-34 ans</li>
                        <li>Désagrégation par sexe (M/F)</li>
                    </ul>
                </li>
                <li><b>Pyramide des âges interactive</b> : Visualisation structure démographique</li>
                <li><b>Densité de population</b> : Impact sur vitesse de propagation</li>
            </ul>
            <h5>🎯 Utilité spécifique Rougeole</h5>
            <ul style="background:#fff9c4; padding:1rem; border-radius:5px;">
                <li><b>Taux d'attaque par âge</b> : (Cas 0-14 ans / Pop 0-14 ans) × 10 000</li>
                <li><b>Estimation doses vaccins</b> : Pop 0-14 ans non vaccinée × 2 doses</li>
                <li><b>Priorisation géographique</b> : Zones avec forte proportion enfants + faible vaccination</li>
                <li><b>Modélisation transmission</b> : Densité forte = R effectif élevé</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>📋 Données Épidémiologiques</h4>
            <p><b>Deux formats acceptés :</b></p>
            <h5>Format 1 : Agrégé (recommandé)</h5>
            <table style="width:100%; border-collapse:collapse; margin:0.5rem 0;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:0.3rem; border:1px solid #ddd;">Colonne</th>
                    <th style="padding:0.3rem; border:1px solid #ddd;">Description</th>
                </tr>
                <tr>
                    <td style="padding:0.3rem; border:1px solid #ddd;"><code>health_area</code></td>
                    <td style="padding:0.3rem; border:1px solid #ddd;">Nom aire santé</td>
                </tr>
                <tr>
                    <td style="padding:0.3rem; border:1px solid #ddd;"><code>Semaine_Epi</code></td>
                    <td style="padding:0.3rem; border:1px solid #ddd;">Semaine épidémiologique (ex: 2024-W05)</td>
                </tr>
                <tr>
                    <td style="padding:0.3rem; border:1px solid #ddd;"><code>Cas_Total</code></td>
                    <td style="padding:0.3rem; border:1px solid #ddd;">Nombre de cas</td>
                </tr>
            </table>
            <h5>Format 2 : Linelist individuelle</h5>
            <table style="width:100%; border-collapse:collapse; margin:0.5rem 0;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:0.3rem; border:1px solid #ddd;">Colonne</th>
                    <th style="padding:0.3rem; border:1px solid #ddd;">Description</th>
                </tr>
                <tr>
                    <td style="padding:0.3rem; border:1px solid #ddd;"><code>Aire_Sante</code></td>
                    <td style="padding:0.3rem; border:1px solid #ddd;">Lieu du cas</td>
                </tr>
                <tr>
                    <td style="padding:0.3rem; border:1px solid #ddd;"><code>Date_Debut_Eruption</code></td>
                    <td style="padding:0.3rem; border:1px solid #ddd;">Date début éruption cutanée</td>
                </tr>
            </table>
            <p><i>→ Agrégation automatique par semaine épidémiologique</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>💉 Couverture Vaccinale (optionnel)</h4>
            <ul>
                <li><b>Format</b> : CSV avec colonnes :
                    <ul>
                        <li><code>health_area</code> : Aire de santé</li>
                        <li><code>Taux_Vaccination</code> : % population vaccinée (0-100)</li>
                    </ul>
                </li>
                <li><b>Utilisation</b> :
                    <ul>
                        <li>Identification zones sous-vaccinées (< 80%)</li>
                        <li>Corrélation couverture vs incidence</li>
                        <li>Priorisation campagnes de rattrapage</li>
                    </ul>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 3 : Guide d'utilisation (continué dans PARTIE 3)
    # Section 3 : Guide d'utilisation ROUGEOLE (suite)
    st.markdown("## 🚀 Guide d'Utilisation Pas-à-Pas")
    
    with st.expander("**Étape 1️⃣ : Choisir le mode (Réel vs Démo)**", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Mode d'utilisation</h4>
            <ul>
                <li><b>📊 Données réelles</b> : Uploadez vos propres fichiers</li>
                <li><b>🧪 Mode démo</b> : Génération automatique données fictives pour tester l'app
                    <ul>
                        <li>129 aires de santé simulées</li>
                        <li>52 semaines de données</li>
                        <li>Épidémie fictive avec pic semaine 15-20</li>
                    </ul>
                </li>
            </ul>
            <p><i>💡 Conseil : Commencez par le mode démo pour comprendre le fonctionnement</i></p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 2️⃣ : Charger les données géographiques**", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>Option 1 : Fichier local (multi-pays)</h4>
            <ol>
                <li>Sélectionnez <b>"Fichier local (ao_hlthArea.zip)"</b></li>
                <li>Choisissez le pays : Niger, Burkina Faso, Mali ou Mauritanie</li>
                <li>→ Filtrage automatique des aires du pays sélectionné</li>
            </ol>
            <h4>Option 2 : Upload personnalisé</h4>
            <ol>
                <li>Sélectionnez <b>"Upload personnalisé"</b></li>
                <li>Uploadez votre Shapefile/GeoJSON (.zip, .shp, .geojson)</li>
                <li>Colonnes requises : <code>health_area</code>, <code>geometry</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 3️⃣ : Charger les linelists rougeole**", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>En mode Données réelles :</h4>
            <ol>
                <li>Uploadez votre CSV de linelist</li>
                <li>L'app détecte automatiquement le format (agrégé ou individuel)</li>
                <li>Validation :
                    <ul>
                        <li>✅ Correspondance noms aires avec fichier géographique</li>
                        <li>✅ Format dates/semaines valide</li>
                        <li>⚠️ Affichage warnings si données incohérentes</li>
                    </ul>
                </li>
            </ol>
            <h4>En mode Démo :</h4>
            <p>Génération automatique de 52 semaines avec :</p>
            <ul>
                <li>Tendance saisonnière (pic fin hiver/début printemps)</li>
                <li>Variabilité géographique réaliste</li>
                <li>Corrélation densité population / cas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 4️⃣ : (Optionnel) Ajouter couverture vaccinale**", expanded=False):
        st.markdown("""
        <div class="benefit-box">
            <h4>💉 Pourquoi ajouter la vaccination ?</h4>
            <ul>
                <li><b>Identification gaps immunitaires</b> : Zones < 80% = risque épidémie</li>
                <li><b>Explication épidémies</b> : Forte incidence souvent corrélée faible vaccination</li>
                <li><b>Ciblage interventions</b> : Prioriser campagnes dans zones sous-vaccinées à forte incidence</li>
            </ul>
            <h5>Seuils OMS :</h5>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="background:#ffebee;">
                    <td style="padding:0.5rem; border:1px solid #ddd;">< 80%</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">🔴 <b>Très insuffisant</b> - Risque épidémie majeure</td>
                </tr>
                <tr style="background:#fff9c4;">
                    <td style="padding:0.5rem; border:1px solid #ddd;">80-94%</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">🟡 <b>Insuffisant</b> - Risque flambées localisées</td>
                </tr>
                <tr style="background:#e8f5e9;">
                    <td style="padding:0.5rem; border:1px solid #ddd;">≥ 95%</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">🟢 <b>Objectif atteint</b> - Immunité collective</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 5️⃣ : Analyser les données (Onglets)**", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>📊 Onglet 1 : Tableau de Bord</h4>
            <ul>
                <li><b>Métriques clés</b> :
                    <ul>
                        <li>Total cas observés</li>
                        <li>Nombre d'aires affectées</li>
                        <li>Population exposée (si WorldPop)</li>
                        <li>Couverture vaccinale moyenne (si données dispo)</li>
                    </ul>
                </li>
                <li><b>Graphiques temporels</b> :
                    <ul>
                        <li>Courbe épidémique (cas par semaine)</li>
                        <li>Tendance vaccination vs cas (si données vaccin)</li>
                    </ul>
                </li>
                <li><b>Top 10 aires</b> : Classement par incidence cumulée</li>
                <li><b>Pyramide des âges</b> : Structure démographique zones affectées (si WorldPop)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>🗺️ Onglet 2 : Cartographie</h4>
            <ul>
                <li><b>Visualisations disponibles</b> :
                    <ul>
                        <li>Choroplèthe : Intensité couleur = nombre de cas</li>
                        <li>Cercles proportionnels : Taille = incidence</li>
                        <li>Heatmap : Concentration géographique</li>
                    </ul>
                </li>
                <li><b>Popups détaillés</b> (clic sur aire) :
                    <ul>
                        <li>📊 Nombre de cas observés</li>
                        <li>👥 Population totale et enfants 0-14 ans</li>
                        <li>📏 Densité population</li>
                        <li>💉 Taux vaccination (si disponible)</li>
                        <li>🎯 Taux d'attaque (cas pour 10 000 enfants)</li>
                    </ul>
                </li>
                <li><b>Filtres</b> :
                    <ul>
                        <li>Période temporelle (plage semaines)</li>
                        <li>Sélection aires spécifiques</li>
                    </ul>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>🔮 Onglet 3 : Prédiction</h4>
            <ul>
                <li><b>Paramètres</b> :
                    <ul>
                        <li>Algorithme : Gradient Boosting (défaut), Random Forest, Ridge, Lasso, Decision Tree</li>
                        <li>Période prédiction : 1-12 mois (4-48 semaines)</li>
                        <li>Validation : Time Series Split 5-fold</li>
                    </ul>
                </li>
                <li><b>Features utilisées automatiquement</b> :
                    <ul>
                        <li>Lags temporels (1-4 semaines)</li>
                        <li>Moyennes mobiles (2-8 semaines)</li>
                        <li>Saisonnalité (sin/cos semaine année)</li>
                        <li>Population enfants (si WorldPop)</li>
                        <li>Densité population (si WorldPop)</li>
                        <li>Taux vaccination (si disponible)</li>
                    </ul>
                </li>
                <li><b>Résultats</b> :
                    <ul>
                        <li>Graphique prédictions vs réel</li>
                        <li>Métriques : R², MAE, RMSE</li>
                        <li>Alertes zones à risque (cas prédits > seuil épidémique)</li>
                        <li>Carte prédictive interactive</li>
                    </ul>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>📈 Onglet 4 : Alertes Épidémiques</h4>
            <ul>
                <li><b>Seuil épidémique automatique</b> :
                    <ul>
                        <li>Calcul : Moyenne historique + 2 × écart-type</li>
                        <li>Par aire de santé (seuils locaux)</li>
                    </ul>
                </li>
                <li><b>Classification zones</b> :
                    <ul>
                        <li>🔴 <b>Alerte rouge</b> : Cas actuels > 2 × seuil</li>
                        <li>🟡 <b>Alerte jaune</b> : Cas actuels > seuil</li>
                        <li>🟢 <b>Normal</b> : Cas < seuil</li>
                    </ul>
                </li>
                <li><b>Export CSV</b> : Liste zones en alerte + recommandations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>💾 Onglet 5 : Export</h4>
            <ul>
                <li>GeoJSON aires de santé</li>
                <li>CSV cas hebdomadaires</li>
                <li>CSV prédictions</li>
                <li>CSV alertes</li>
                <li>ZIP complet (tous les fichiers)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 4 : Conseils spécifiques Rougeole
    st.markdown("## 💡 Conseils Spécifiques Rougeole")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="benefit-box">
            <h4>✅ Pour une surveillance optimale</h4>
            <ul>
                <li><b>Données historiques</b> : Au moins 6 mois (26 semaines) pour capturer saisonnalité</li>
                <li><b>WorldPop essentiel</b> : Population enfants 0-14 ans = calcul taux d'attaque précis</li>
                <li><b>Couverture vaccinale</b> : Permet d'expliquer 60-80% des flambées (zones sous-vaccinées)</li>
                <li><b>Algorithme</b> : XGBoost est proposé par défaut. Les essais
                    menés lors de l'audit montrent que les six algorithmes disponibles
                    se tiennent en quelques pour cent les uns des autres — le choix de
                    l'algorithme importe beaucoup moins que la qualité des données et
                    du protocole de validation.</li>
                <li><b>Prédictions court terme</b> : 1 à 4 semaines. La précision réelle
                    dépend de la régularité de la notification ; aucune valeur de R²
                    n'est garantie à l'avance. Fiez-vous au CV‑R² et au MAE CV affichés
                    après entraînement, pas à une valeur annoncée.</li>
                <li><b>Saisonnalité</b> : Pics hivernaux (janvier-mars) en Afrique de l'Ouest</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="benefit-box">
            <h4>🎯 Interprétation seuils épidémiques</h4>
            <p><b>Seuil OMS rougeole :</b></p>
            <ul>
                <li><b>5 cas pour 10 000 enfants < 15 ans</b> par semaine</li>
                <li>OU <b>3 cas liés</b> (même chaîne transmission) en 4 semaines</li>
            </ul>
            <p><i>→ L'app calcule automatiquement le seuil adapté à chaque zone</i></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Limites et précautions</h4>
            <ul>
                <li><b>Sous-déclaration</b> : Rougeole souvent sous-diagnostiquée
                    <ul>
                        <li>Multiplicateur estimé : 1 cas rapporté = 3-10 cas réels</li>
                        <li>Solutions : Triangulation avec campagnes de rattrapage</li>
                    </ul>
                </li>
                <li><b>Épidémies explosives</b> : Délai 2-3 semaines → épidémie déjà avancée
                    <ul>
                        <li>Importance surveillance syndromique (cas suspects)</li>
                        <li>Réaction rapide < 72h dès confirmation</li>
                    </ul>
                </li>
                <li><b>Campagnes de vaccination</b> : Changent drastiquement la dynamique
                    <ul>
                        <li>Modèle ne prédit pas impact campagnes futures</li>
                        <li>Recalibrer après campagne massive</li>
                    </ul>
                </li>
                <li><b>Population mobile</b> : Mouvements transfrontaliers → sous-estimation risque zones frontalières</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box">
            <h4>🚨 Quand déclencher riposte vaccinale ?</h4>
            <p><b>Critères OMS :</b></p>
            <ol>
                <li><b>Confirmation épidémie</b> : Cas > seuil épidémique 2 semaines consécutives</li>
                <li><b>Couverture vaccinale < 80%</b> dans la zone</li>
                <li><b>Taux d'attaque > 50/10 000 enfants</b> (cumulé sur 4 semaines)</li>
            </ol>
            <p><b>Action :</b> Campagne vaccination réactive (CVR) dans rayon 30 km autour du cluster</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 3 : GLOSSAIRE & MÉTHODOLOGIE (COMPLET)
# ============================================================

with tab_glossaire:
    st.header("📖 Glossaire des Variables & Méthodologie")
    
    glossary_tabs = st.tabs([
        "📅 Variables Temporelles",
        "📊 Historique Cas",
        "🌡️ Climat",
        "🌍 Environnement",
        "👥 Démographiques",
        "🧮 Méthodes Avancées",
        "🧠 Concepts avancés"
    ])
    
    # TAB : Variables Temporelles
    with glossary_tabs[0]:
        st.markdown("""
        <div class="info-card">
            <h4>Numéro de semaine (week_num)</h4>
            <p><b>Signification :</b> Numéro séquentiel de la semaine (1, 2, 3...)</p>
            <p><b>Utilité :</b> Capture la tendance générale dans le temps</p>
            <p><b>Exemple :</b> Semaine 20 → printemps (hausse attendue paludisme)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Saisonnalité (sin_week, cos_week)</h4>
            <p><b>Problème si on garde seulement le numéro de semaine :</b></p>
            <ul>
                <li>La semaine 52 et la semaine 1 semblent très éloignées (52 vs 1),</li>
                <li>alors qu’elles sont en réalité <b>côte à côte dans le calendrier</b> (fin décembre – début janvier).</li>
            </ul>
            <p><b>Encodage circulaire :</b> on projette la semaine sur un cercle :</p>
            <ul>
                <li><code>sin_week = sin(2π × semaine / 52)</code></li>
                <li><code>cos_week = cos(2π × semaine / 52)</code></li>
            </ul>
            <p>Sur ce cercle, les semaines 1 et 52 sont presque au même endroit, ce qui permet au modèle de <b>respecter la nature cyclique</b> de la saisonnalité.</p>
            <p><b>Intérêt :</b> cet encodage aide le modèle à apprendre les <b>pics saisonniers récurrents</b> (par exemple pour la rougeole au Sahel).</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Harmoniques supplémentaires (sin_week2, cos_week2)</h4>
            <p><b>Signification :</b> Capture cycles plus rapides (bi-annuels)</p>
            <p><b>Calcul :</b> sin(4π × semaine / 52) et cos(4π × semaine / 52)</p>
            <p><b>Utilité :</b> Modélise saisons pluies multiples (ex: 2 pics par an)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB : Historique Cas
    with glossary_tabs[1]:
        st.markdown("""
        <div class="info-card">
            <h4>Lags temporels (cases_lag1, cases_lag2, cases_lag4)</h4>
            <p><b>Signification :</b> Nombre de cas 1, 2 ou 4 semaines avant</p>
            <p><b>Utilité :</b> <b>Variable la plus importante</b> - Tendance récente</p>
            <p><b>Exemple :</b> 50 cas en S24 → Prédiction S25 ≈ 48-52 cas</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Moyennes mobiles (cases_ma2, cases_ma4, cases_ma8)</h4>
            <p><b>Signification :</b> Moyenne des 2, 4 ou 8 dernières semaines</p>
            <p><b>Utilité :</b> Lisse les fluctuations, montre tendance globale</p>
            <p><b>Calcul :</b> MA2 = (S-1 + S-2) / 2</p>
            <p><b>Avantage :</b> Moins sensitive aux pics isolés</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Moyenne et variabilité sur 4 semaines (RollingMean4, RollingStd4)</h4>
            <h5>Définitions</h5>
            <ul>
                <li><b>RollingMean4</b> = moyenne des cas sur les 4 semaines précédentes.</li>
                <li><b>RollingStd4</b> = variabilité (écart-type) des cas sur les 4 dernières semaines.</li>
            </ul>
            <h5>Pourquoi 4 semaines&nbsp;?</h5>
            <ul>
                <li>La rougeole a une <b>incubation d’environ 7–14 jours</b> : une génération de cas toutes les 1–2 semaines.</li>
                <li>Sur 4 semaines, on couvre typiquement <b>2 à 3 générations</b> de transmission.</li>
                <li>Cette fenêtre permet de capturer les <b>chaînes de transmission encore actives</b> au moment de la prédiction.</li>
            </ul>
            <h5>Interprétation</h5>
            <ul>
                <li><b>RollingMean4 élevé</b> = tendance épidémique installée (niveau de fond élevé).</li>
                <li><b>RollingStd4 élevé</b> = situation instable, avec des pics et creux fréquents.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Taux de croissance (growth_rate)</h4>
            <p><b>Signification :</b> Variation % entre 2 semaines consécutives</p>
            <p><b>Formule :</b> (Cas<sub>S</sub> - Cas<sub>S-1</sub>) / Cas<sub>S-1</sub></p>
            <p><b>Exemple :</b> 40→50 cas → +25% (croissance rapide)</p>
            <p><b>Utilité :</b> Détecte accélérations/décélérations épidémiques</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Accélération (acceleration)</h4>
            <p><b>Signification :</b> Variation du taux de croissance (dérivée seconde)</p>
            <p><b>Formule :</b> growth_rate(S) - growth_rate(S-1)</p>
            <p><b>Utilité :</b> Détecte changements de dynamique</p>
            <ul>
                <li>Accélération positive : Épidémie s'amplifie</li>
                <li>Accélération négative : Épidémie ralentit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Min/Max glissants (cases_min4, cases_max4, cases_min8, cases_max8)</h4>
            <p><b>Signification :</b> Valeurs extrêmes sur fenêtres 4 et 8 semaines</p>
            <p><b>Utilité :</b> Capture amplitude variations récentes</p>
            <p><b>Exemple :</b> Max_4 très élevé → Pic récent = zone à risque</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Dates épidémiologiques : début d’éruption vs notification</h4>
            <p><b>Date_Debut_Eruption</b> = moment où les symptômes caractéristiques apparaissent (fièvre, éruption). C’est à partir de là que le patient est réellement contagieux.</p>
            <p><b>Date_Notification</b> = moment où le cas est enregistré dans le système (fiche remplie, saisie, validation), souvent <b>7 à 14 jours plus tard</b>.</p>
            <h5>Pourquoi corriger ce délai&nbsp;?</h5>
            <ul>
                <li>Si on se base sur la <b>notification</b>, on observe l’épidémie avec un décalage pouvant aller jusqu’à 2 semaines.</li>
                <li>En recodant les cas selon la <b>Date_Debut_Eruption</b>, on se rapproche du moment réel de transmission.</li>
                <li>La plateforme utilise cette correction pour une <b>détection plus précoce</b> des signaux (J0 au lieu de J+14).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB : Climat
    with glossary_tabs[2]:
        st.markdown("""
        <div class="info-card">
            <h4>Température moyenne (temp_api)</h4>
            <p><b>Signification :</b> Température moyenne hebdomadaire en degrés Celsius</p>
            <p><b>Source :</b> NASA POWER API</p>
            <p><b>Impact paludisme :</b></p>
            <ul>
                <li>< 18°C : Transmission nulle (parasite ne se développe pas)</li>
                <li>25-30°C : Optimum (cycle sporogonique 10-12 jours)</li>
                <li>> 34°C : Ralentissement (mortalité moustiques)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Précipitations totales (precip_api)</h4>
            <p><b>Signification :</b> Cumul pluies hebdomadaires en millimètres</p>
            <p><b>Impact paludisme :</b></p>
            <ul>
                <li>Création gîtes larvaires (eau stagnante)</li>
                <li>Délai d'action : 7-10 jours (émergence moustiques adultes)</li>
                <li>Seuil critique : > 50mm/semaine → explosion vectorielle</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Humidité relative (humidity_api)</h4>
            <p><b>Signification :</b> Humidité moyenne hebdomadaire en %</p>
            <p><b>Impact paludisme :</b></p>
            <ul>
                <li>< 60% : Mortalité élevée moustiques (déshydratation)</li>
                <li>> 60% : Longévité accrue → plus de piqûres infectantes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB : Environnement
    with glossary_tabs[3]:
        st.markdown("""
        <div class="info-card">
            <h4>Niveau d'inondation (flood_mean)</h4>
            <p><b>Signification :</b> Hauteur d'eau moyenne zone inondable (raster)</p>
            <p><b>Utilité :</b> Zones inondables = gîtes larvaires permanents</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Altitude (elevation_mean)</h4>
            <p><b>Signification :</b> Altitude moyenne de l'aire en mètres</p>
            <p><b>Impact paludisme :</b></p>
            <ul>
                <li>< 1000m : Transmission intense</li>
                <li>1000-1500m : Transmission modérée</li>
                <li>> 1500m : Transmission faible/nulle</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Distance rivière (dist_river)</h4>
            <p><b>Signification :</b> Distance centroïde aire → cours d'eau le plus proche (km)</p>
            <p><b>Utilité :</b> Proximité rivière = risque accru (reproduction <i>Anopheles</i>)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB : Démographiques
    with glossary_tabs[4]:
        st.markdown("""
        <div class="info-card">
            <h4>Population totale (Pop_Totale)</h4>
            <p><b>Source :</b> WorldPop (Google Earth Engine)</p>
            <p><b>Utilité :</b> Dénominateur pour taux d'incidence</p>
            <p><b>Formule :</b> Taux incidence = (Cas / Pop_Totale) × 10 000</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Population enfants 0-14 ans (Pop_Enfants_0_14)</h4>
            <p><b>Signification :</b> Somme des tranches 0-4, 5-9, 10-14 ans</p>
            <p><b>Utilité Paludisme :</b></p>
            <ul>
                <li>Groupe le plus vulnérable (immunité faible)</li>
                <li>60-80% des cas graves et décès</li>
                <li>Calcul besoins moustiquaires imprégnées (1 MII / 2 enfants)</li>
            </ul>
            <p><b>Utilité Rougeole :</b></p>
            <ul>
                <li>90% des cas totaux (forte susceptibilité)</li>
                <li>Calcul doses vaccins (Pop_Enfants × 2 doses)</li>
                <li>Taux d'attaque = (Cas 0-14 ans / Pop_Enfants) × 10 000</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Densité de population (Densite_Pop)</h4>
            <p><b>Signification :</b> Habitants par km²</p>
            <p><b>Utilité :</b></p>
            <ul>
                <li><b>Paludisme :</b> Densité forte + proximité gîtes = transmission intense</li>
                <li><b>Rougeole :</b> Densité forte = R effectif élevé (contagion rapide)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Tranches d'âge détaillées (Pop_M/F_0_4, 5_9...)</h4>
            <p><b>Signification :</b> Population par sexe (M/F) et tranche de 5 ans</p>
            <p><b>Utilité :</b></p>
            <ul>
                <li>Pyramide des âges (visualisation structure démographique)</li>
                <li>Ciblage interventions par âge (ex: rougeole < 5 ans prioritaire)</li>
                <li>Estimation besoins vaccins/médicaments par tranche</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB : Méthodes Avancées
    with glossary_tabs[5]:
        st.markdown("## 🧮 Méthodologie de Modélisation Avancée")
        
        st.markdown("""
        <div class="info-card">
            <h3>📐 Analyse en Composantes Principales (ACP)</h3>
            <h4>Principe</h4>
            <p>L'ACP transforme un ensemble de variables <b>corrélées</b> en un ensemble réduit de variables <b>non-corrélées</b> (composantes principales) qui capturent la majorité de la variance.</p>
            <h5>Exemple Concret</h5>
            <p><b>Situation initiale :</b></p>
            <ul>
                <li>Température, Humidité, Précipitations <i>(Fortement corrélées)</i></li>
                <li>Cas S-1, Cas S-2, Moyenne mobile 4W <i>(Redondance temporelle)</i></li>
                <li>50 variables au total <i>(Risque de sur-apprentissage)</i></li>
            </ul>
            <p><b>Après ACP :</b></p>
            <ul>
                <li><b>PC1</b> (40% variance) : Composante climatique globale (température + humidité)</li>
                <li><b>PC2</b> (25% variance) : Tendance temporelle (lags + moyennes mobiles)</li>
                <li><b>PC3</b> (15% variance) : Variabilité saisonnière</li>
                <li>...</li>
                <li><b>Total 8 composantes</b> capturent 95% de l'information</li>
            </ul>
            <h5>Interprétation des Composantes</h5>
            <p>Chaque composante est une <b>combinaison linéaire</b> des variables originales. Les coefficients indiquent la contribution de chaque variable.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="benefit-box">
                <h5>Avantages ACP</h5>
                <ul>
                    <li><b>Réduit complexité</b> : 50 → 8 variables</li>
                    <li><b>Élimine redondance</b> : Décolle variables corrélées</li>
                    <li><b>Améliore généralisation</b> : Moins de sur-apprentissage</li>
                    <li><b>Accélère calculs</b> : Moins de dimensions</li>
                    <li><b>Visualisation</b> : 2-3 composantes suffisent</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h5>Limites</h5>
                <ul>
                    <li>Perd interprétabilité directe</li>
                    <li>Nécessite scaling préalable</li>
                    <li>Linéaire (pas optimal si non-linéarités fortes)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="info-card">
            <h3>📍 Clustering Spatial (K-Means)</h3>
            <h4>Principe</h4>
            <p>Identifier des <b>groupes de zones géographiques homogènes</b> ayant des profils épidémiologiques similaires.</p>
            <h5>Algorithme K-Means</h5>
            <ol>
                <li><b>Entrée :</b> Coordonnées géographiques (latitude, longitude) de chaque aire de santé</li>
                <li><b>Initialisation :</b> Sélection aléatoire de k centres (ex: k=5)</li>
                <li><b>Attribution :</b> Chaque zone assignée au centre le plus proche</li>
                <li><b>Mise à jour :</b> Recalcul des centres comme moyenne des zones du groupe</li>
                <li><b>Itération :</b> Répéter jusqu'à stabilité</li>
            </ol>
            <h5>Utilité en Épidémiologie</h5>
            <ul>
                <li><b>Capture hétérogénéité spatiale</b> : Zones urbaines vs rurales, côtières vs intérieures</li>
                <li><b>Améliore prédictions</b> : Le modèle apprend des patterns spécifiques à chaque cluster</li>
                <li><b>Stratégies ciblées</b> : Interventions adaptées par groupe géographique</li>
            </ul>
            <h5>Exemple de Clustering</h5>
            <table style="width:100%; border-collapse:collapse; margin:1rem 0;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:0.5rem; border:1px solid #ddd;">Cluster</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Caractéristiques</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Cas Moy.</th>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Cluster 0</b> (Côtier)</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Basse altitude, près rivières, forte humidité</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>120/sem</b></td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Cluster 1</b> (Urbain)</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Dense, assainissement variable</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>85/sem</b></td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Cluster 2</b> (Rural plaine)</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Rizières, marais, forte transmission</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>150/sem</b></td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Cluster 3</b> (Montagne)</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Altitude > 800m, faible transmission</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>20/sem</b></td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Cluster 4</b> (Semi-aride)</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Faibles précipitations, transmission saisonnière</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>45/sem</b></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h5>Variables Cluster - Le modèle crée des variables binaires</h5>
                <p><code>cluster_0 = [1, 0, 0, 0, 0]  # Zone côtière</code></p>
                <p><code>cluster_1 = [0, 1, 0, 0, 0]  # Zone urbaine</code></p>
                <p><code>cluster_2 = [0, 0, 1, 0, 0]  # Rural plaine</code></p>
                <p><i>Le modèle apprend poids spécifiques pour chaque cluster</i></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="benefit-box">
                <h5>Paramètre k</h5>
                <ul>
                    <li><b>k=3</b> : Influence locale</li>
                    <li><b>k=5</b> : Équilibre (recommandé)</li>
                    <li><b>k=10</b> : Influence régionale</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="info-card">
            <h3>🌐 Lag Spatial</h3>
            <h4>Principe</h4>
            <p>Le <b>lag spatial</b> mesure l'influence des zones <b>voisines</b> sur le nombre de cas d'une zone.</p>
            <p><i>Hypothèse :</i> Si mes voisins ont beaucoup de cas, j'ai probablement plus de risques (migration moustiques, mouvements population).</p>
            <h5>Calcul - Pour une zone <b>i</b>, on calcule la <b>moyenne pondérée</b> des cas des k voisins les plus proches</h5>
            <p><code>Lag_spatial(zone_i) = Σ w_ij × Cas_j  pour j = k voisins les plus proches</code></p>
            <p><code>où w_ij = 1 / distance(i, j)  (poids inversement proportionnel à la distance)</code></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="benefit-box">
                <h5>Utilité</h5>
                <ul>
                    <li><b>Capture autocorrélation spatiale</b> : Les cas se regroupent géographiquement</li>
                    <li><b>Détecte clusters épidémiques</b> : Zones "hot spots"</li>
                    <li><b>Améliore prédictions</b> : +5-10% de précision en zones denses</li>
                    <li><b>Modélise diffusion</b> : Propagation géographique</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h5>Paramètre Clé : k</h5>
                <ul>
                    <li><b>k=3</b> : Influence très locale (voisins immédiats)</li>
                    <li><b>k=5</b> : Équilibre (recommandé)</li>
                    <li><b>k=10</b> : Influence régionale (peut lisser trop)</li>
                </ul>
                <p><i>💡 En pratique, k=5 fonctionne bien pour la plupart des contextes épidémiologiques.</i></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="info-card">
            <h3>⏱️ Validation Croisée Temporelle</h3>
            <h4>Pourquoi spéciale pour séries temporelles ?</h4>
            <p>En épidémiologie, <b>l'ordre temporel est crucial</b>. On ne peut pas tester le modèle sur des données <i>antérieures</i> à celles d'entraînement (ça n'a pas de sens de "prédire le passé" !)</p>
            <h5>Time Series Split (5 Folds) - Exemple avec données de 52 semaines</h5>
            <p><code>Fold 1:  Entraînement [S1-S30]  →  Test [S31-S40]  →  r² = 0.82</code></p>
            <p><code>Fold 2:  Entraînement [S1-S35]  →  Test [S36-S44]  →  r² = 0.78</code></p>
            <p><code>Fold 3:  Entraînement [S1-S40]  →  Test [S41-S48]  →  r² = 0.85</code></p>
            <p><code>Fold 4:  Entraînement [S1-S44]  →  Test [S45-S50]  →  r² = 0.80</code></p>
            <p><code>Fold 5:  Entraînement [S1-S48]  →  Test [S49-S52]  →  r² = 0.83</code></p>
            <p><b>Performance finale : r² = 0.82 ± 0.03 → Robuste !</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="benefit-box">
                <h5>Avantages</h5>
                <ul>
                    <li><b>Réaliste</b> : Simule vraie utilisation (prédire futur avec passé)</li>
                    <li><b>Détecte sur-apprentissage</b> : Si r² entraînement >> r² test</li>
                    <li><b>Mesure robustesse</b> : Écart-type faible = modèle stable</li>
                    <li><b>Compare algorithmes</b> : Choix objectif du meilleur</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h5>Interprétation Résultats</h5>
                <table style="width:100%; border-collapse:collapse;">
                    <tr style="background:#f5f5f5;">
                        <th style="padding:0.5rem; border:1px solid #ddd;">Écart-type r²</th>
                        <th style="padding:0.5rem; border:1px solid #ddd;">Signification</th>
                    </tr>
                    <tr style="background:#e8f5e9;">
                        <td style="padding:0.5rem; border:1px solid #ddd;">< 0.05</td>
                        <td style="padding:0.5rem; border:1px solid #ddd;">🟢 Très stable</td>
                    </tr>
                    <tr style="background:#fff9c4;">
                        <td style="padding:0.5rem; border:1px solid #ddd;">0.05-0.10</td>
                        <td style="padding:0.5rem; border:1px solid #ddd;">🟡 Acceptable</td>
                    </tr>
                    <tr style="background:#ffebee;">
                        <td style="padding:0.5rem; border:1px solid #ddd;">> 0.10</td>
                        <td style="padding:0.5rem; border:1px solid #ddd;">🔴 Instable</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="benefit-box">
            <h3>📊 Schéma Pipeline Complet</h3>
            <h4>1️⃣ FEATURES ENGINEERING</h4>
            <p><b>Variables créées :</b></p>
            <ul style="font-size:0.9rem;">
                <li>Lags temporels (1-4 sem.)</li>
                <li>Moyennes mobiles (2-8 sem.)</li>
                <li>Volatilité (écart-types)</li>
                <li>Taux croissance</li>
                <li>Accélération (dérivée 2)</li>
                <li>Min/Max glissants</li>
                <li>Harmoniques saisonnières</li>
            </ul>
            <h4>2️⃣ ACP (Optionnel)</h4>
            <p><b>Réduction dimensionnelle :</b></p>
            <ul style="font-size:0.9rem;">
                <li>Standardisation données</li>
                <li>Extraction composantes</li>
                <li>Conservation 95% variance</li>
                <li>Ex: 50 vars → 8 composantes</li>
                <li>Évite sur-apprentissage</li>
                <li>Calculs plus rapides</li>
            </ul>
            <h4>3️⃣ SPATIAL</h4>
            <p><b>Analyse géographique :</b></p>
            <ul style="font-size:0.9rem;">
                <li>K-means clustering (5 groupes)</li>
                <li>Lag spatial pondéré</li>
                <li>Influence voisins (k=5)</li>
                <li>Autocorrélation spatiale</li>
                <li>Matrice distances</li>
            </ul>
            <h4>4️⃣ VALIDATION</h4>
            <p><b>Test rigoureux :</b></p>
            <ul style="font-size:0.9rem;">
                <li>Time Series Split 5 folds</li>
                <li>Validation croisée</li>
                <li>Métriques : r², MAE, RMSE</li>
                <li>Moyenne ± Écart-type</li>
                <li>Robustesse temporelle</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB : Concepts avancés (fusion de l'ancien manuel)
    with glossary_tabs[6]:
        st.markdown("## 🧠 Concepts avancés Rougeole & Modèles")
        
        st.markdown("""
        <div class="info-card">
            <h3>R₀ & immunité collective (rappel)</h3>
            <p><b>R₀</b> = taux de reproduction de base : nombre moyen de personnes infectées par un cas dans une population totalement sensible.</p>
            <ul>
                <li>Pour la rougeole, R₀ est souvent estimé entre <b>12 et 18</b>.</li>
                <li>Le <b>seuil d’immunité collective</b> ≈ <code>1 - 1/R₀</code> → plus de 90&nbsp;%.</li>
                <li>Les programmes visent une <b>couverture vaccinale &ge; 95&nbsp;%</b> pour casser les chaînes de transmission.</li>
            </ul>
            <p><i>Idée clé :</i> maladie ultra contagieuse → besoin d’une vaccination quasi universelle.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>Gradient Boosting : principe et paramètres</h3>
            <h4>Idée générale</h4>
            <p>Le Gradient Boosting construit une <b>série de petits arbres de décision</b> plutôt qu’un seul gros arbre :</p>
            <ul>
                <li>Un premier arbre fait une prédiction approximative.</li>
                <li>On calcule les <b>erreurs</b> de cet arbre.</li>
                <li>Un deuxième arbre est entraîné pour corriger ces erreurs.</li>
                <li>On répète la procédure ; la somme de tous les arbres donne le modèle final.</li>
            </ul>
            <h4>Paramètres utilisés dans la plateforme</h4>
            <ul>
                <li><b>n_estimators = 200</b> : ~200 petits arbres successifs.</li>
                <li><b>learning_rate = 0.05</b> : pas d’apprentissage <b>petit</b> pour avancer de façon prudente.</li>
                <li><b>max_depth = 4</b> : arbres peu profonds, qui apprennent des règles simples.</li>
            </ul>
            <p><b>Compromis :</b> capacité à capturer des relations non linéaires tout en limitant le sur-apprentissage.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>Métriques de performance : R², MAE, RMSE, CV‑R² ± σ</h3>
            <h4>R² (coefficient de détermination)</h4>
            <p>Mesure la proportion de la variabilité des cas observés que le modèle explique (entre 0 et 1).</p>
            <ul>
                <li><b>R² = 1</b> : prédiction parfaite.</li>
                <li><b>R² ≈ 0</b> : pas mieux qu’une moyenne constante.</li>
            </ul>
            <h4>MAE (Mean Absolute Error)</h4>
            <p>Erreur absolue moyenne entre les cas prédits et observés.</p>
            <p><i>Lecture :</i> "en moyenne, on se trompe de X cas par semaine".</p>
            <h4>RMSE (Root Mean Squared Error)</h4>
            <p>Racine carrée de la moyenne des erreurs au carré, pénalise davantage les grosses erreurs.</p>
            <h4>Validation croisée (CV‑R² ± σ)</h4>
            <ul>
                <li>Les données sont découpées en <b>blocs temporels</b> : chaque bloc
                    de test est strictement postérieur à son bloc d'entraînement.</li>
                <li>Une <b>période d'embargo</b> sépare l'entraînement du test, pour
                    tenir compte du délai de notification et de la durée d'incubation.</li>
                <li>On calcule le <b>R² moyen</b> (CV‑R²) et son <b>écart-type</b> (σ),
                    ainsi que le <b>MAE CV</b> — la métrique la plus robuste sur des
                    comptages très dispersés.</li>
                <li>⚠️ Le <b>R² in-sample</b> affiché à côté est mesuré sur les données
                    d'entraînement : il est toujours optimiste et ne doit pas servir à
                    juger le modèle. C'est le <b>CV‑R²</b> qui compte.</li>
            </ul>
            <p><b>Règle pratique :</b></p>
            <ul>
                <li>Juger sur le <b>CV‑R²</b> et le <b>MAE CV</b>, jamais sur le R²
                    in-sample.</li>
                <li>Comparer toujours au <b>modèle de référence</b> (naïf saisonnier :
                    même semaine l'an dernier). Un CV‑R² élevé sans gain sur cette
                    référence ne signifie pas que le modèle apporte quelque chose.</li>
                <li>Vérifier le <b>rapport total prédit ÷ total observé</b> : il doit
                    être proche de 1. Un modèle peut avoir un bon R² et se tromper
                    systématiquement de niveau.</li>
                <li>Sur un panneau regroupant des aires de tailles très différentes, le
                    R² est dominé par les écarts entre aires : le <b>MAE rapporté à la
                    moyenne observée</b> est plus parlant.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7f8c8d; padding:2rem;">
    <h4>📧 Contact Support Technique</h4>
    <p>📧 Email : youssoupha.mbodji@example.com</p>
    <p>💬 Questions fréquentes : FAQ (à venir)</p>
    <p>📖 Documentation complète : Manuel utilisateur</p>
    <p style="margin-top:1rem;">Version 3.0 | Développé par <b>Youssoupha MBODJI</b></p>
    <p>© 2026 - Licence Open Source MIT</p>
</div>
""", unsafe_allow_html=True)

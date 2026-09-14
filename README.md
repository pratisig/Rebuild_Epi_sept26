# 🦟 SurveillanceEpi – Paludisme Sahel

> 🇬🇧 [English version below / Version anglaise ci-dessous](#-surveillanceepi--malaria-sahel-english)

Plateforme Streamlit pour la **surveillance épidémiologique et la modélisation prédictive du paludisme** au Sahel, intégrant données sanitaires hebdomadaires, données climatiques (NASA POWER / Open‑Meteo), population (WorldPop) et données environnementales (inondation, élévation, rivières).

Cette application est pensée pour des **équipes santé publique** (PNLP, districts, ONG, agences ONU) : pas besoin d'être développeur pour l'utiliser, mais des compétences de base en fichiers CSV et en interprétation de graphiques sont utiles.

---

## 1. Fonctionnalités principales

- 📁 **Chargement guidé des données** depuis la barre latérale (shapefile d'aires de santé, cas hebdomadaires CSV, rasters, rivières).
- 🌍 **Cartographie interactive** des cas, décès, indicateurs démographiques et environnementaux (choroplèthes, cercles proportionnels, heatmap).
- 🤖 **Modélisation prédictive** du nombre de cas par aire de santé, avec modèles de Machine Learning (RandomForest, GradientBoosting, ExtraTrees).
- 🔮 **Prédictions à court terme (1–12 semaines)** pour anticiper les pics et prioriser les aires à risque.
- 🌦️ **API climat** (NASA POWER, Open‑Meteo) pour récupérer température, précipitations et humidité par aire de santé et par semaine.
- 👥 **Intégration WorldPop** (Google Earth Engine) pour population totale, enfants 0–14 ans, densité, pyramide des âges simplifiée ou détaillée.
- 🌊 **Données environnementales** : inondations (raster), élévation (raster), distance aux rivières.
- 📈 **Tableaux de bord** : indicateurs clés (cas, décès, létalité, incidence, taux d'attaque) et courbes temporelles.
- 📊 **Analyse avancée** : corrélations entre cas, climat, population, environnement.
- 📥 **Export** des données enrichies et des prédictions (onglet Export).

---

## 2. Architecture générale

L'application est une **app Streamlit** avec plusieurs onglets :

1. **📊 Dashboard**  
   Synthèse des indicateurs (cas, décès, incidence, population, climat) + courbes temporelles et top 10 aires.

2. **🗺️ Cartographie**  
   Carte interactive (Folium) avec différentes visualisations : choroplèthes, cercles proportionnels, heatmap, couches rasters et rivières.

3. **🤖 Modélisation**  
   Configuration des modèles ML, lancement de la modélisation, performances (MAE, RMSE, R², CV), prédictions, heatmap des risques.

4. **🗺️ Carte Prédictions**  
   Visualisation cartographique des prédictions par aire de santé et par semaine future.

5. **📈 Analyse Avancée**  
   Corrélations entre cas, climat et variables environnementales, avec visualisations.

6. **📥 Export**  
   Téléchargement des jeux de données intermédiaires et des prédictions.

La barre latérale (« sidebar ») gère :
- le **chargement des données obligatoires** (aires de santé, cas hebdomadaires),
- le **chargement des données optionnelles** (WorldPop, rasters, rivières),
- l'**activation des API climat** et le téléchargement des données.

---

## 3. Données attendues

### 3.1. Aires de santé (géométrie)

Deux options :

1. **Fichier local fourni** : `data/ao_hlthArea.zip`  
   - Contient un shapefile avec les aires de santé pour plusieurs pays.  
   - Le pays est choisi dans la sidebar via le code ISO3 (ex. `bfa`, `mli`, `ner`, `mrt`).

2. **Fichier uploadé** (GeoJSON / SHP / ZIP)

#### Colonnes attendues / détectées automatiquement

L'application essaie de détecter la colonne du nom d'aire parmi :
`health_area`, `health_are`, `name_fr`, `name`, `NAME`, `nom`, `NOM`

Si rien n'est trouvé, des noms génériques sont créés (`aire_1`, `aire_2`, ...).

> **Important** : la couche doit être en WGS84 (`EPSG:4326`) avec des polygones ou multipolygones valides.

---

### 3.2. Cas hebdomadaires (CSV)

Le séparateur est détecté automatiquement (`;`, `,`, `\t`).

#### Colonnes minimales

| Colonne interne | Variantes reconnues | Obligatoire |
|---|---|---|
| `health_area` | `health_area`, `aire_de_sante`, `aire_sante`, `aire`, `aire_sanitaire`, `zone_sanitaire`, `district` | ✅ |
| `week_` | `week_`, `week`, `semaine`, `semaine_epi`, `epiweek`, `epi_week`, `semaine_epidemiologique` | ✅ |
| `cases` | `cases`, `nb_cas`, `nombre_de_cas`, `n_cas`, `cas`, `cas_palu`, `cas_paludisme` | ✅ |
| `year` | `year`, `annee`, `année`, `annee_epi`, `year_epi`, `epi_year` | ⭐ Fortement recommandée |
| `deaths` | `deaths`, `death`, `deces`, `décès`, `nb_deces`, `nb_deces_palu` | ❌ Optionnelle (défaut = 0) |

#### Format des semaines

- Numéro entier (1, 2, 3…), ou format type `2019-W01`, `2019-S01`, `S01` → normalisé automatiquement en entier.
- Si `year` est présente, une colonne `period` est créée au format `YYYY-Sxx` (ex. `2019-S01`).

#### Exemple de structure CSV minimale

```text
health_area;year;week_;cases;deaths
bobo_dioulasso;2019;1;12;0
bobo_dioulasso;2019;2;18;1
ouagadougou;2019;1;5;0
```

---

### 3.3. Données population (WorldPop via GEE)

Récupérées automatiquement si Google Earth Engine est configuré. Variables ajoutées par aire :

| Variable | Description |
|---|---|
| `Pop_Totale` | Population totale |
| `Pop_Enfants_0_14` | Population 0–14 ans |
| `Densite_Pop` | Habitants / km² |
| `Pop_MALE_*` / `Pop_FEMALE_*` | Pyramide des âges par tranches (0–4, 5–9, 10–14, 15–19, 20–24, 25–29, 30–34) |

> Sans GEE, l'application fonctionne sans indicateurs démographiques.

---

### 3.4. Rasters environnementaux

| Données | Format | Variable interne | Description |
|---|---|---|---|
| Inondation | TIF raster | `flood_mean`, `flood_risk` | Intensité / probabilité d'inondation |
| Élévation | TIF raster | `elevation_mean` | Altitude moyenne (m) |
| Rivières | GeoJSON / SHP / ZIP | `dist_river` | Distance au cours d'eau le plus proche (km) |

Indices calculés automatiquement :
- `flood_risk` = inondation × (1 / dist_river)
- `climate_index` = température + humidité
- `temp_precip_interaction` = température × précipitations

---

### 3.5. Données climat via API

| Source | Type | Variables | Clé API |
|---|---|---|---|
| NASA POWER | Journalier par point | `T2M` (temp), `PRECTOTCORR` (précip), `RH2M` (humidité) | ❌ Gratuite |
| Open‑Meteo Archive | Journalier par point | `temperature_2m_mean`, `precipitation_sum`, `relative_humidity_2m_mean` | ❌ Gratuite |

Agrégation hebdomadaire automatique : `temp_api`, `temp_api_min`, `temp_api_max`, `precip_api`, `precip_api_max`, `humidity_api`, `humidity_api_min`, `humidity_api_max`.

---

## 4. Installation et exécution locale

### 4.1. Prérequis

- **Python** ≥ 3.9  
- **Git**  
- Accès internet (pour les API climat et GEE)

### 4.2. Cloner le dépôt

```bash
git clone https://github.com/pratisig/surveillanceEpi.git
cd surveillanceEpi
```

### 4.3. Environnement virtuel (recommandé)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate            # Windows
```

### 4.4. Installer les dépendances

```bash
pip install -r requirements.txt
```

Installation manuelle si pas de `requirements.txt` :

```bash
pip install streamlit pandas geopandas numpy folium streamlit-folium rasterio matplotlib seaborn scikit-learn pillow requests plotly earthengine-api branca
```

#### Bibliothèques utilisées

| Bibliothèque | Rôle |
|---|---|
| `streamlit` | Interface web interactive |
| `pandas`, `numpy` | Traitement des données tabulaires |
| `geopandas`, `shapely` | Données géographiques (SHP, GeoJSON) |
| `folium`, `streamlit-folium`, `branca` | Cartes interactives Leaflet |
| `rasterio` | Lecture des rasters TIF |
| `scikit-learn` | Modèles ML (RandomForest, GradientBoosting, ExtraTrees, PCA, pipelines) |
| `matplotlib`, `seaborn`, `plotly` | Graphiques et visualisations |
| `requests` | Appels API HTTP (NASA POWER, Open‑Meteo) |
| `Pillow` | Traitement d'images |
| `earthengine-api` | Accès WorldPop via Google Earth Engine |
| `difflib`, `unicodedata` | Correction floue des noms d'aires de santé |

### 4.5. Configuration Google Earth Engine (optionnelle)

1. Créer un compte GEE : https://earthengine.google.com/
2. Créer un service account et télécharger la clé JSON.
3. Ajouter un secret `GEE_SERVICE_ACCOUNT` contenant le contenu JSON de la clé :
   - **Streamlit Cloud** : via les Secrets de l'application dans le tableau de bord Streamlit.
   - **Local** : créer un fichier `.streamlit/secrets.toml` :
     ```toml
     GEE_SERVICE_ACCOUNT = '{ "type": "service_account", ... }'
     ```

### 4.6. Lancer l'application

```bash
streamlit run main_app.py
```

Ouvre http://localhost:8501 dans votre navigateur.

---

## 5. Guide d'utilisation pas à pas

### Étape 1 – Charger les aires de santé
1. Sidebar → **« 📍 Données Obligatoires »**.
2. Choisir le fichier local (sélectionner le pays) ou importer votre propre GeoJSON/SHP/ZIP.
3. Message `✅ XX aires chargées` confirme le succès.

### Étape 2 – Charger le CSV de cas
1. Sidebar → **« 📁 Chargement des Données »** → **« Cas hebdomadaires (CSV) »**.
2. Colonnes reconnues → message `✅ Colonnes requises détectées`.
3. Un aperçu du tableau et les statistiques de la série apparaissent.

### Étape 3 – Filtres
Sidebar → **« 🔍 Filtres »** : filtrer par années, semaines, aires de santé. Les filtres s'appliquent à l'ensemble des onglets.

### Étape 4 – Activer WorldPop (optionnel)
Si GEE est configuré, la population par aire est récupérée et intégrée automatiquement.

### Étape 5 – Charger les rasters environnementaux (optionnel)
Sidebar → **« 🌍 Données Environnementales »** : importer les rasters TIF (inondation, élévation) et la couche rivières.

### Étape 6 – API Climat (optionnel)
1. Sidebar → **« 🌦️ API Climat »** → cocher **« Activer API Climat »**.
2. Choisir la source (NASA POWER ou Open‑Meteo).
3. Sélectionner les années dans les filtres.
4. Cliquer **« 🚀 Télécharger Données Climatiques »**.

### Étape 7 – Explorer le Dashboard
Onglet **📊 Dashboard** : indicateurs clés (cas, décès, incidence, taux d'attaque), courbes temporelles, top 10 aires.

### Étape 8 – Cartographie
Onglet **🗺️ Cartographie** : choroplèthe / cercles / heatmap, avec couches rasters et popups détaillés par aire.

### Étape 9 – Lancer la modélisation
Onglet **🤖 Modélisation** :
1. Choisir l'algorithme (RandomForest recommandé), l'horizon (1–12 semaines), le mode (Simple / Expert).
2. Cliquer **« 🚀 LANCER MODÉLISATION »**.
3. Résultats : métriques de performance, tableau des prédictions par aire, heatmap des risques, liste des zones à risque.

### Étape 10 – Carte des prédictions
Onglet **🗺️ Carte Prédictions** : visualisation spatiale des prédictions semaine par semaine.

### Étape 11 – Analyse avancée
Onglet **📈 Analyse Avancée** : matrice de corrélations entre cas, climat, population, environnement.

### Étape 12 – Export
Onglet **📥 Export** : télécharger les données enrichies et les prédictions en CSV.

---

## 6. Interprétation des métriques de validation

### 6.1. Principe fondamental : MAE et RMSE sont des métriques absolues

> ⚠️ **Règle d'or** : un MAE ou RMSE isolé, sans connaître la moyenne des valeurs observées, est **impossible à interpréter**.

Le MAE et le RMSE sont exprimés dans **la même unité que la variable cible** (ici : nombre de cas hebdomadaires). Ils « grandissent » proportionnellement aux valeurs absolues du district :

- **MAE = 500 sur une moyenne de 4 000 cas/semaine** → erreur relative de **12,5 %** → ✅ Excellent
- **MAE = 500 sur une moyenne de 600 cas/semaine** → erreur relative de **83 %** → 🔴 Inacceptable

Un district à forte charge (capitale, grand centre urbain) aura mécaniquement un MAE plus élevé qu'un petit district rural, sans que cela signifie que le modèle soit moins bon.

---

### 6.2. Les métriques de base

| Métrique | Formule | Interprétation | Sensibilité |
|---|---|---|---|
| **MAE** | Moyenne des `|observé − prédit|` | Erreur moyenne absolue en nb de cas | Égale sur tous les points |
| **RMSE** | Racine de la moyenne des `(observé − prédit)²` | Pénalise davantage les grosses erreurs | Sensible aux pics extrêmes |
| **R²** | 1 − (variance résiduelle / variance totale) | % de variance expliquée (0 = nul, 1 = parfait) | Relative au modèle nul |
| **R² CV** | R² moyen sur les folds de validation croisée temporelle | Fiabilité réelle en situation de prédiction | Robuste au surapprentissage |

**Rapport RMSE/MAE** : si RMSE > 2 × MAE, cela signale la présence de **pics extrêmes mal captés** par le modèle. Il faut alors vérifier si des semaines épidémiques exceptionnelles (flambées) biaisent l'évaluation.

---

### 6.3. Métriques normalisées — comparer entre districts

Pour comparer les performances entre districts de tailles différentes, on utilise des métriques **relatives** :

> ⚠️ **Cette version du code ne calcule pas de MAPE.** Une version antérieure de
> ce document présentait le MAPE comme « la métrique de référence » avec un barème
> d'interprétation en quatre niveaux. Aucune métrique de ce nom n'est calculée ni
> affichée par l'application. Le barème a donc été retiré : il invitait à une
> confiance qu'aucune mesure ne fondait.

#### sMAPE — Mean Absolute Percentage Error symétrique

C'est la métrique relative **effectivement calculée** par le noyau de modélisation
(`epimodel/validation.py`) :

```
sMAPE = (1/n) × Σ |observé − prédit| / ((|observé| + |prédit|) / 2) × 100
```

Le MAPE classique divise par la valeur observée : il n'est pas défini lorsque
celle-ci vaut zéro, ce qui est fréquent en basse transmission et dans les aires à
faible charge — précisément celles où la surveillance a le plus d'importance. Le
sMAPE évite cette division par zéro et reste borné entre 0 et 200 %.

| sMAPE | Lecture indicative |
|---|---|
| < 25 % | erreur faible au regard du niveau de charge |
| 25 – 50 % | tendance exploitable, chiffres exacts à nuancer |
| > 50 % | ordre de grandeur seulement |

Ces repères sont des ordres de grandeur, pas des seuils de décision : la valeur à
comparer est celle de votre **modèle de référence** (voir MASE ci-dessous), pas un
nombre absolu.

#### Erreur relative à la moyenne observée

```
MAE / moyenne des cas observés × 100
```

C'est la lecture la plus directe pour un décideur : « le modèle se trompe en
moyenne de X % du niveau habituel de cas ». Elle est calculée par le noyau et
affichée dans l'onglet de modélisation.

#### CV(RMSE) — Coefficient de Variation du RMSE

```
CV(RMSE) = RMSE / moyenne des valeurs observées × 100
```

| Seuil CV(RMSE) | Interprétation |
|---|---|
| < 15 % | 🟢 Très bon |
| 15 – 25 % | 🟡 Acceptable |
| > 25 % | 🔴 Modèle à améliorer |

#### MASE — Mean Absolute Scaled Error

Compare le modèle à une **baseline naïve** (prédire la valeur de la semaine précédente). **MASE < 1** signifie que votre modèle bat la baseline la plus simple. MASE > 1 indique que le modèle est moins bon qu'une simple règle « semaine suivante = semaine actuelle ».

---

### 6.4. Grille de lecture contextuelle selon le niveau de charge

| Situation | MAE/RMSE brut | Erreur relative | Interprétation |
|---|---|---|---|
| District forte charge (moy. > 1 000 cas/sem.) | MAE > 200 | MAE/moy. < 20 % | ✅ Normal — erreur relative acceptable |
| District forte charge (moy. > 1 000 cas/sem.) | MAE > 400 | MAE/moy. > 40 % | ⚠️ Élevé — vérifier outliers ou données manquantes |
| District faible charge (moy. < 200 cas/sem.) | MAE > 100 | MAE/moy. > 50 % | 🔴 Critique — modèle peu fiable sur ce district |
| Tout niveau | RMSE >> MAE (ratio > 2) | — | ⚠️ Présence de pics extrêmes mal captés |
| Tout niveau | R² élevé mais gain nul sur le modèle de référence | — | ⚠️ Le R² reflète les écarts entre districts, pas une capacité de prévision |

---

### 6.5. Seuils de qualité pour le R² et le R² en validation croisée

| R² CV | Interprétation | Décision |
|---|---|---|
| > 0,85 | 🟢 Excellent | Modèle utilisable pour alerte précoce |
| 0,70 – 0,85 | 🟡 Bon | Fiable pour planification logistique |
| 0,50 – 0,70 | 🟠 Modéré | Tendances exploitables — chiffres à nuancer |
| < 0,50 | 🔴 Insuffisant | Enrichir les données (météo, vecteurs, démographie) |

> ⚠️ Un R² élevé sur les données d'entraînement avec un R² CV faible révèle du **surapprentissage** : le modèle a mémorisé l'historique sans apprendre les vrais patterns. Toujours se fier au **R² CV**, pas au R² simple.

---

### 6.6. Validation croisée temporelle — nombre de folds et stratégie

#### Pourquoi pas un k-fold classique ?

Un k-fold classique **mélange aléatoirement** les données d'entraînement et de test. Pour des séries temporelles, cela crée une **fuite temporelle** (*data leakage*) : le modèle « voit le futur » pendant l'entraînement, ce qui gonfle artificiellement les métriques. **Il ne faut jamais utiliser un k-fold classique sur des données épidémiologiques hebdomadaires.**

#### Stratégies recommandées

| Stratégie | Description | Quand l'utiliser |
|---|---|---|
| **Walk-forward / Time Series Split** | Chaque fold ajoute des semaines à l'entraînement, teste sur les suivantes | Standard — respecte l'ordre temporel |
| **Blocked CV** | Blocs continus sans chevauchement | Données courtes (< 2 ans) |
| **Expanding window** | L'entraînement commence toujours depuis le début | Séries rares ou districts avec peu de semaines |

#### Nombre de folds recommandé

| Volume de données | Folds recommandés | Remarque |
|---|---|---|
| < 2 ans (~104 semaines) | **5 folds** | Minimum viable |
| 2 – 3 ans (~150 semaines) | **5 à 7 folds** | Bon équilibre biais/variance |
| ≥ 3 ans (~156+ semaines) | **7 à 10 folds** | Meilleure estimation, plus de calcul |
| > 10 folds | ❌ Déconseillé avec RandomForest | Temps de calcul quadratique, gain marginal |

> **Règle pratique** : avec 5 folds sur 2 ans de données hebdomadaires, chaque fold de test représente environ 8 semaines (~2 mois), ce qui est cohérent avec un horizon de prédiction opérationnel pour le paludisme saisonnier.

---

### 6.7. Résumé visuel de lecture des résultats

```
Après modélisation, lire dans cet ordre :

1. R² CV        → Le modèle est-il globalement fiable ?
                   > 0,70 : oui → continuer
                   < 0,50 : non → ajouter des données ou changer d'algorithme

2. MAE / moyenne observée
                → L'erreur est-elle acceptable proportionnellement ?
                   < 20 % : oui → les prédictions sont exploitables
                   > 50 % : non → les chiffres exacts ne sont pas fiables

3. MAE brut     → En combien de cas me trompe-t-on en moyenne ?
                   Toujours lire avec la moyenne observée du district.
                   Exemple : MAE = 350 / moyenne = 2 800 → 12,5 % → acceptable

4. RMSE/MAE     → Y a-t-il des pics extrêmes mal captés ?
                   Ratio > 2 → vérifier les semaines épidémiques atypiques

5. Graphique    → Visuellement, les courbes observée et prédite se suivent-elles ?
   Observé vs     Un bon R² avec un mauvais suivi visuel est un signal d'alerte.
   Prédit
```

---

## 7. Contexte et valeur ajoutée

Au Sahel, les pics de paludisme suivent la saison des pluies (juillet–octobre) de façon régulière. Pourtant, **connaître le « quand » ne suffit pas pour décider** :

- **Combien de cas exactement** dans quelle aire de santé spécifique cette semaine ?
- Quelle aire sera **plus touchée que d'habitude** cette année (anomalie pluviométrique, déplacement de population) ?
- **Où concentrer les stocks** de moustiquaires, d'ACT, de RDT en priorité ?

L'outil apporte une **granularité spatiale et quantitative** que la saisonnalité seule ne fournit pas :

1. **Priorisation géographique** : parmi 50 aires, lesquelles méritent une intervention urgente cette semaine ?
2. **Détection d'anomalie locale** : une aire sort-elle de son profil habituel (flambée atypique) ?
3. **Allocation proportionnelle** des ressources selon les cas prédits par zone.
4. **Traçabilité** : les décisions de déploiement sont documentées et comparables d'une année sur l'autre.

L'horizon de **4–8 semaines** est précisément le délai logistique actionnable (commandes de médicaments, rotations de personnel, campagnes CPS).

---

## 8. Limites

- **Horizon court** (1–12 semaines) : pas conçu pour des projections stratégiques annuelles.
- **Qualité des données en entrée** : sous-déclaration, erreurs de saisie ou incohérences temporelles biaisent les prédictions.
- **Pas de modèle mécanistique** : l'outil est statistique (ML), pas une simulation biologique du cycle du paludisme.
- **Anomalies exceptionnelles** non anticipées : ruptures de stock, changements de politique, résistances émergentes.
- **Proxys environnementaux** : les données climatiques et d'environnement sont des approximations sur des centroïdes, pas des mesures de terrain.

---

## 9. FAQ terrain — Questions d'un chef de district

### 🗓️ « Le pic de palu arrive chaque année en septembre, à quoi sert cet outil ? »
La saisonnalité dit *quand* surveiller. L'outil dit *où et combien* agir. Parmi vos 30 aires de santé, lesquelles auront 400 cas cette semaine et lesquelles en auront 20 ? C'est cette différence qui guide l'allocation des moustiquaires et des ACT.

### 📂 « Mon CSV est refusé, que faire ? »
Vérifier que :
1. Le fichier contient bien les colonnes `health_area`, `week_` et `cases` (ou leurs variantes reconnues).
2. Le séparateur est `;`, `,` ou tabulation.
3. Les noms d'aires de santé dans le CSV correspondent à ceux de la couche géographique (orthographe identique). L'application propose des corrections automatiques si les noms sont proches.

### 🗺️ « Mes aires de santé n'apparaissent pas sur la carte »
Les noms dans le CSV doivent correspondre exactement (ou approximativement) à ceux du shapefile. Vérifier dans la sidebar → expander « ⚠️ Aires du CSV sans géométrie » pour voir les noms manquants, et l'expander « 🧠 Propositions de corrections » pour les correspondances suggérées.

### 🌡️ « L'API climat ne se charge pas »
- Vérifier la connexion internet.
- Sélectionner au moins une année dans les filtres avant de lancer le téléchargement climatique.
- Si NASA POWER échoue, essayer Open‑Meteo, qui ne nécessite pas de clé API.

### 🤖 « Le modèle dit qu'une aire aura 500 cas mais elle n'en a jamais eu plus de 50. Est-ce fiable ? »
Vérifier le R² et le MAE. Un résultat aberrant indique souvent :
1. Des données d'historique insuffisantes (moins de 2 ans recommandés).
2. Des noms d'aires non correspondants entre CSV et shapefile (des cas se cumulent sur une seule aire).
3. Des valeurs extrêmes dans les données (vérifier si une semaine a un nombre de cas anormalement élevé).

### 📊 « Quelle différence entre incidence et taux d'attaque ? »
- **Incidence** : nombre de cas pour 10 000 habitants (toute la population).
- **Taux d'attaque enfants** : nombre de cas pour 1 000 enfants de 0–14 ans (population cible prioritaire).

### 🔢 « Mon MAE est de 480 cas. C'est trop élevé ? »
Pas nécessairement. Comparez le MAE à la **moyenne hebdomadaire observée** de votre district :
- Si la moyenne est 3 500 cas/semaine → MAE de 480 = **13,7 %** d'erreur → ✅ Excellent
- Si la moyenne est 800 cas/semaine → MAE de 480 = **60 %** d'erreur → 🔴 Insuffisant

Divisez le **MAE affiché** par la moyenne hebdomadaire observée de votre district : c'est la lecture en pourcentage la plus fiable. (Le MAPE n'est pas calculé par cette version du code — voir la section 6.3.)

### 🔮 « Sur combien de semaines puis-je faire confiance aux prédictions ? »
En pratique, la fiabilité décroît au-delà de 4–6 semaines. Pour les 2–4 semaines suivantes, les prédictions sont suffisamment fiables pour guider les décisions logistiques. L'interface affiche le R² en validation croisée pour vous aider à estimer la confiance.

### 💾 « Comment exporter les résultats pour les partager avec le PNLP ? »
Onglet **📥 Export** : télécharger les données enrichies (CSV) et les prédictions. Ces fichiers peuvent être importés dans Excel, QGIS ou partagés par email.

### 🏥 « L'application fonctionne-t-elle sans connexion internet ? »
Oui, pour les fonctionnalités de base (chargement CSV, cartographie, modélisation). Non, pour l'API climat (NASA POWER / Open‑Meteo) et WorldPop (GEE) qui nécessitent une connexion.

### 🌍 « Peut-on l'utiliser pour d'autres pays que le Burkina, Mali, Niger, Mauritanie ? »
Oui, si vous fournissez votre propre couche d'aires de santé (GeoJSON ou SHP) via l'option « Charger un fichier » dans la sidebar.

---

## 10. Public cible et contribution

- Programmes Nationaux de Lutte contre le Paludisme (PNLP)
- Directions de la lutte contre la maladie
- Districts sanitaires / régions
- ONG, agences des Nations Unies, chercheurs

Pour toute contribution (code, documentation, idées de fonctionnalités), ouvrir une **issue** ou une **pull request** dans ce dépôt.

---

---

# 🦟 SurveillanceEpi — Malaria Sahel (English)

> 🇫🇷 [Version française ci-dessus / French version above](#-surveillanceepi--paludisme-sahel)

Streamlit platform for **epidemiological surveillance and predictive modeling of malaria** in the Sahel, integrating weekly health data, climate data (NASA POWER / Open‑Meteo), population data (WorldPop), and environmental data (floods, elevation, rivers).

Designed for **public health teams** (NMCPs, districts, NGOs, UN agencies): no developer skills required, but basic knowledge of CSV files and chart reading is helpful.

---

## E1. Key Features

- 📁 **Guided data loading** from the sidebar (health area shapefile, weekly cases CSV, rasters, rivers).
- 🌍 **Interactive mapping** of cases, deaths, demographic and environmental indicators (choropleth, proportional circles, heatmap).
- 🤖 **Predictive modeling** of cases per health area using Machine Learning (RandomForest, GradientBoosting, ExtraTrees).
- 🔮 **Short-term predictions (1–12 weeks)** to anticipate peaks and prioritize high-risk areas.
- 🌦️ **Climate APIs** (NASA POWER, Open‑Meteo) for temperature, precipitation and humidity per health area per week.
- 👥 **WorldPop integration** (Google Earth Engine) for total population, children 0–14, density, age pyramid.
- 🌊 **Environmental data**: flood raster, elevation raster, distance to rivers.
- 📈 **Dashboards**: key indicators (cases, deaths, fatality rate, incidence, attack rate) and time series charts.
- 📊 **Advanced analysis**: correlations between cases, climate, population, environment.
- 📥 **Export** enriched datasets and predictions.

---

## E2. Application Structure

The app has 6 tabs:

| Tab | Content |
|---|---|
| 📊 Dashboard | KPIs + time series + top 10 areas |
| 🗺️ Mapping | Interactive Folium map with choropleth/circles/heatmap |
| 🤖 Modeling | ML configuration, metrics, predictions table, risk heatmap |
| 🗺️ Prediction Map | Spatial view of predictions by area and future week |
| 📈 Advanced Analysis | Correlation matrix across all variables |
| 📥 Export | Download enriched data and predictions |

The **sidebar** manages: mandatory data loading (health areas + weekly cases), optional data (WorldPop, rasters, rivers), and climate API activation.

---

## E3. Expected Data

### E3.1. Health Areas (geometry)

- **Built-in file** `data/ao_hlthArea.zip`: select country by ISO3 code (`bfa`, `mli`, `ner`, `mrt`).
- **Upload your own** GeoJSON / SHP / ZIP.
- Auto-detected name column candidates: `health_area`, `health_are`, `name_fr`, `name`, `NAME`, `nom`, `NOM`.
- Must be in WGS84 (`EPSG:4326`), polygons or multipolygons.

---

### E3.2. Weekly Cases (CSV)

Auto-detected separator (`;`, `,`, `\t`).

| Internal column | Accepted variants | Required |
|---|---|---|
| `health_area` | `health_area`, `area`, `district`, `zone_sanitaire` | ✅ |
| `week_` | `week_`, `week`, `semaine`, `epiweek`, `epi_week` | ✅ |
| `cases` | `cases`, `nb_cas`, `cas`, `cas_palu`, `cas_paludisme` | ✅ |
| `year` | `year`, `annee`, `epi_year`, `year_epi` | ⭐ Strongly recommended |
| `deaths` | `deaths`, `death`, `deces`, `nb_deces` | ❌ Optional (defaults to 0) |

**Week format**: integer (1, 2, 3…) or `2019-W01`, `2019-S01`, `S01` → auto-normalized.

If `year` is provided, a `period` column is created in the format `YYYY-Sxx` (e.g., `2019-S01`).

**Minimal CSV example:**
```text
health_area;year;week_;cases;deaths
bobo_dioulasso;2019;1;12;0
bobo_dioulasso;2019;2;18;1
```

---

### E3.3. Population (WorldPop via GEE)

Auto-retrieved if GEE is configured. Added variables per area:

| Variable | Description |
|---|---|
| `Pop_Totale` | Total population |
| `Pop_Enfants_0_14` | Population aged 0–14 |
| `Densite_Pop` | Inhabitants / km² |
| `Pop_MALE_*` / `Pop_FEMALE_*` | Age pyramid by bracket (0–4, 5–9, 10–14, 15–19, 20–24, 25–29, 30–34) |

---

### E3.4. Environmental Rasters

| Data | Format | Internal variable | Description |
|---|---|---|---|
| Flood | TIF raster | `flood_mean`, `flood_risk` | Flood intensity / probability |
| Elevation | TIF raster | `elevation_mean` | Mean altitude (m) |
| Rivers | GeoJSON / SHP / ZIP | `dist_river` | Distance to nearest river (km) |

---

### E3.5. Climate API

| Source | Type | Key variables | API key |
|---|---|---|---|
| NASA POWER | Daily point data | Temperature, precipitation, humidity | ❌ Free |
| Open‑Meteo Archive | Daily point data | Temperature, precipitation, humidity | ❌ Free |

Weekly aggregation: `temp_api`, `temp_api_min/max`, `precip_api`, `precip_api_max`, `humidity_api`, `humidity_api_min/max`.

---

## E4. Local Installation

```bash
# Clone
git clone https://github.com/pratisig/surveillanceEpi.git
cd surveillanceEpi

# Virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run main_app.py
```

App opens at http://localhost:8501.

**Key libraries:**

| Library | Role |
|---|---|
| `streamlit` | Web UI |
| `pandas`, `numpy` | Data processing |
| `geopandas`, `shapely` | Geospatial data |
| `folium`, `streamlit-folium` | Interactive Leaflet maps |
| `rasterio` | TIF raster reading |
| `scikit-learn` | ML models (RandomForest, GradientBoosting, ExtraTrees, PCA) |
| `matplotlib`, `seaborn`, `plotly` | Charts and visualization |
| `requests` | HTTP API calls |
| `earthengine-api` | WorldPop via GEE |

**GEE Setup (optional):**
1. Create a GEE account at https://earthengine.google.com/
2. Create a service account and download the JSON key.
3. Add a `GEE_SERVICE_ACCOUNT` secret containing the full JSON key content.

---

## E5. Usage Guide

1. **Load health areas**: Sidebar → pick built-in country or upload your own layer.
2. **Load weekly cases CSV**: Sidebar → upload CSV → confirm column recognition.
3. **Apply filters**: Sidebar → filter by year, week, health area.
4. **Optional – WorldPop**: Automatically retrieved if GEE is configured.
5. **Optional – Environmental rasters**: Sidebar → upload flood/elevation TIF and rivers layer.
6. **Optional – Climate API**: Enable, select source, select years, click Download.
7. **Dashboard tab**: Review KPIs and time trends.
8. **Mapping tab**: Explore choropleth/circles/heatmap with detailed popups.
9. **Modeling tab**: Choose algorithm (RandomForest recommended), horizon (1–12 weeks), mode (Simple/Expert) → Launch → Review metrics and predictions.
10. **Prediction Map tab**: Spatial visualization of predictions week by week.
11. **Advanced Analysis tab**: Correlations across all variables.
12. **Export tab**: Download enriched CSV and predictions.

---

## E6. Metrics Interpretation & Validation Guide

### E6.1. The Golden Rule: MAE and RMSE Are Absolute Metrics

> ⚠️ **Key principle**: MAE or RMSE alone, without knowing the mean of observed values, is **impossible to interpret**.

MAE and RMSE are expressed in **the same unit as the target variable** (here: weekly case counts). They scale proportionally with the district's absolute burden:

- **MAE = 500 on a mean of 4,000 cases/week** → relative error = **12.5%** → ✅ Excellent
- **MAE = 500 on a mean of 600 cases/week** → relative error = **83%** → 🔴 Unacceptable

A high-burden district (capital city, large urban centre) will mechanically show a higher MAE than a small rural district, without meaning the model performs worse.

---

### E6.2. Core Metrics

| Metric | Formula | Interpretation | Sensitivity |
|---|---|---|---|
| **MAE** | Mean of `|observed − predicted|` | Average absolute error in number of cases | Equal weight on all points |
| **RMSE** | Square root of mean `(observed − predicted)²` | Penalizes large errors more heavily | Sensitive to extreme spikes |
| **R²** | 1 − (residual variance / total variance) | % of variance explained (0 = none, 1 = perfect) | Relative to null model |
| **CV R²** | Mean R² across temporal validation folds | True predictive reliability, out-of-sample | Robust to overfitting |

**RMSE/MAE ratio**: if RMSE > 2 × MAE, the model is poorly capturing **extreme epidemic peaks**. Check for exceptional outbreak weeks biasing the evaluation.

---

### E6.3. Normalized Metrics — Comparing Across Districts

> ⚠️ **This version of the code does not compute MAPE.** An earlier revision of
> this document presented MAPE as "the reference metric" with a four-level
> interpretation scale. No metric by that name is computed or displayed by the
> application. The scale has therefore been removed: it invited a confidence that
> no measurement supported.

#### sMAPE — Symmetric Mean Absolute Percentage Error

This is the relative metric **actually computed** by the modelling core
(`epimodel/validation.py`):

```
sMAPE = (1/n) × Σ |observed − predicted| / ((|observed| + |predicted|) / 2) × 100
```

Classic MAPE divides by the observed value, so it is undefined when that value is
zero — common in low-transmission settings and low-burden areas, precisely where
surveillance matters most. sMAPE avoids that division by zero and stays bounded
between 0 and 200%.

| sMAPE | Indicative reading |
|---|---|
| < 25% | small error relative to the burden level |
| 25 – 50% | trend usable, exact figures to be nuanced |
| > 50% | order of magnitude only |

These are orders of magnitude, not decision thresholds: the value to compare
against is that of your **reference model** (see MASE below), not an absolute
number.

#### Error relative to the observed mean

```
MAE / mean observed cases × 100
```

The most direct reading for a decision-maker: "the model is wrong by X% of the
usual case level on average". Computed by the core and displayed in the
modelling tab.

#### CV(RMSE) — Coefficient of Variation of RMSE

```
CV(RMSE) = RMSE / mean observed value × 100
```

| CV(RMSE) threshold | Interpretation |
|---|---|
| < 15% | 🟢 Very good |
| 15–25% | 🟡 Acceptable |
| > 25% | 🔴 Model needs improvement |

#### MASE — Mean Absolute Scaled Error

Compares the model against a **naïve baseline** (predicting last week's value). **MASE < 1** means your model outperforms the simplest possible rule. MASE > 1 means the model is worse than "next week = this week."

---

### E6.4. Contextual Reading Grid by Burden Level

| Situation | Raw MAE/RMSE | Relative error | Interpretation |
|---|---|---|---|
| High-burden district (mean > 1,000 cases/week) | MAE > 200 | MAE/mean < 20% | ✅ Normal — relative error acceptable |
| High-burden district (mean > 1,000 cases/week) | MAE > 400 | MAE/mean > 40% | ⚠️ High — check for outliers or missing data |
| Low-burden district (mean < 200 cases/week) | MAE > 100 | MAE/mean > 50% | 🔴 Critical — model unreliable for this district |
| Any level | RMSE >> MAE (ratio > 2) | — | ⚠️ Extreme peaks poorly captured |
| Any level | High R² but no gain over the reference model | — | ⚠️ R² reflects between-district variance, not forecasting skill |

---

### E6.5. Quality Thresholds for R² and Cross-Validated R²

| CV R² | Interpretation | Decision |
|---|---|---|
| > 0.85 | 🟢 Excellent | Model suitable for early warning |
| 0.70–0.85 | 🟡 Good | Reliable for logistical planning |
| 0.50–0.70 | 🟠 Moderate | Trends useful — exact figures to be nuanced |
| < 0.50 | 🔴 Insufficient | Add more data (climate, vectors, demographics) |

> ⚠️ A high training R² with a low CV R² signals **overfitting**: the model memorized the history without learning true patterns. Always rely on **CV R²**, not raw R².

---

### E6.6. Temporal Cross-Validation — Number of Folds and Strategy

#### Why Not Standard k-Fold?

Standard k-fold **randomly shuffles** training and test data. For time series, this creates **temporal leakage** (*data leakage*): the model "sees the future" during training, artificially inflating metrics. **Never use standard k-fold on weekly epidemiological data.**

#### Recommended Strategies

| Strategy | Description | When to use |
|---|---|---|
| **Walk-forward / Time Series Split** | Each fold expands training, tests on subsequent weeks | Standard — respects temporal order |
| **Blocked CV** | Non-overlapping contiguous blocks | Short data (< 2 years) |
| **Expanding window** | Training always starts from the beginning | Sparse data or districts with few weeks |

#### Recommended Number of Folds

| Data volume | Recommended folds | Note |
|---|---|---|
| < 2 years (~104 weeks) | **5 folds** | Minimum viable |
| 2–3 years (~150 weeks) | **5–7 folds** | Good bias/variance balance |
| ≥ 3 years (~156+ weeks) | **7–10 folds** | Better estimate, more compute |
| > 10 folds | ❌ Not recommended with RandomForest | Quadratic compute cost, marginal gain |

> **Practical rule**: with 5 folds on 2 years of weekly data, each test fold covers ~8 weeks (~2 months) — consistent with the operational prediction horizon for seasonal malaria.

---

### E6.7. Step-by-Step Results Reading Guide

```
After modeling, read in this order:

1. CV R²       → Is the model globally reliable?
                  > 0.70: yes → continue
                  < 0.50: no → add data or change algorithm

2. MAE / observed mean
               → Is the error acceptable relative to the data magnitude?
                  < 20%: yes → predictions are usable
                  > 30%: no → exact figures are unreliable

3. Raw MAE     → How many cases off is the model on average?
                  Always read against the observed mean for this district.
                  Example: MAE = 350 / mean = 2,800 → 12.5% → acceptable

4. RMSE/MAE    → Are there extreme peaks poorly captured?
                  Ratio > 2 → check for atypical epidemic weeks

5. Chart       → Do the observed and predicted curves visually track?
   Observed       A good R² with poor visual tracking is a warning signal.
   vs Predicted
```

---

## E7. Context and Value

Seasonal patterns in the Sahel tell you *when* to be vigilant. This tool tells you *where* and *how much* to act:

- Which of your 30 health areas will have 400 cases this week vs. 20?
- Which area is showing an unusual spike compared to historical patterns?
- How should you allocate bednets, ACTs, and RDTs proportionally?

The **4–8 week horizon** is precisely the actionable logistical window: drug orders, staff rotations, and seasonal chemoprevention (SMC) campaigns all require this lead time.

---

## E8. Limitations

- **Short horizon** (1–12 weeks): not designed for annual strategic projections.
- **Data quality dependence**: under-reporting, entry errors, and temporal inconsistencies bias predictions.
- **Statistical, not mechanistic**: ML-based, not a biological simulation of the malaria cycle.
- **Exceptional anomalies** not anticipated: drug stock-outs, policy changes, emerging resistance.
- **Environmental proxies**: climate and environmental data are approximations at centroids, not field measurements.

---

## E9. Field FAQ

### 🗓️ "The malaria peak comes every September anyway — why do I need this tool?"
Seasonality tells you *when* to watch. The tool tells you *where and how much* to act. Among your 30 health areas, which ones will have 400 cases this week and which will have 20? That difference drives your allocation of bednets and ACTs.

### 📂 "My CSV is rejected. What should I do?"
1. Ensure columns `health_area`, `week_`, and `cases` (or their variants) are present.
2. Use `;`, `,`, or tab as separator.
3. Verify that area names in the CSV match those in the shapefile. The app suggests fuzzy corrections automatically.

### 🗺️ "My health areas don't appear on the map"
Check Sidebar → expander "⚠️ Areas from CSV without geometry" for missing names, and "🧠 Name correction suggestions" for fuzzy matches.

### 🤖 "The model predicts 500 cases but this area has never had more than 50. Is it reliable?"
Check R² and MAE. Outlier predictions usually indicate: (1) insufficient history (at least 2 years recommended), (2) mismatched area names causing case accumulation, or (3) extreme outlier values in the historical data.

### 🔢 "My MAE is 480 cases. Is that too high?"
Not necessarily. Compare MAE to the **observed weekly mean** for your district:
- Mean = 3,500 cases/week → MAE of 480 = **13.7%** → ✅ Excellent
- Mean = 800 cases/week → MAE of 480 = **60%** → 🔴 Insufficient

Divide the **displayed MAE** by your district's mean weekly observed cases: that is the most reliable percentage reading. (MAPE is not computed by this version of the code — see section E6.3.)

### 🔮 "How many weeks ahead can I trust the predictions?"
Reliability drops beyond 4–6 weeks. For the next 2–4 weeks, predictions are reliable enough to guide logistical decisions. Check the CV R² displayed after modeling for confidence estimation.

### 🌍 "Can I use this for countries other than Burkina, Mali, Niger, Mauritania?"
Yes — upload your own health area layer (GeoJSON or SHP) via the "Upload file" option in the sidebar.

---

## E10. Target Audience & Contribution

- National Malaria Control Programs (NMCPs)
- Disease control departments
- Health districts / regions
- NGOs, UN agencies, researchers

To contribute (code, documentation, feature ideas), open an **issue** or **pull request** in this repository.

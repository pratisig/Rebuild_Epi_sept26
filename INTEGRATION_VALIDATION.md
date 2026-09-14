# Guide d'intégration — Onglet Validation Rétrospective

Le module `validation_tab.py` est prêt. Il suffit de faire **3 modifications** dans `app_paludisme.py`.

---

## Modification 1 — Importer le module (en haut du fichier)

Ajouter cette ligne **après** la ligne `from prediction_map_tab import create_prediction_map_tab` :

```python
from validation_tab import create_validation_tab
```

---

## Modification 2 — Ajouter l'onglet dans `st.tabs()`

Cherchoz cette ligne :

```python
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", 
    "🗺️ Cartographie", 
    "🤖 Modélisation", 
   "🗺️ Carte Prédictions",
   "📈 Analyse Avancée",
    "📥 Export"
])
```

Remplacez par :

```python
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard",
    "🗺️ Cartographie",
    "🤖 Modélisation",
    "🗺️ Carte Prédictions",
    "📈 Analyse Avancée",
    "📥 Export",
    "🔬 Validation",
])
```

---

## Modification 3 — Ajouter le contenu de l'onglet (juste avant ou après `with tab6`)

Ajoutez ce bloc **après** le bloc `with tab6:` (Export) :

```python
# ============================================================
# TAB 7 – VALIDATION RÉTROSPECTIVE
# ============================================================
with tab7:
    create_validation_tab(
        df_cases=st.session_state.df_cases,
        gdf_health=st.session_state.gdf_health,
        model_results=st.session_state.get("model_results"),
    )
```

---

## Ce que l'onglet Validation produit

### Entrées attendues
- `df_cases` : le même CSV de cas que pour la modélisation (colonnes `health_area`, `week_`, `cases`)
- Variables climatiques (`temp_api`, `precip_api`, `humidity_api`) utilisées automatiquement si présentes
- Variables environnementales (`flood_mean`, `dist_river`, `elevation_mean`) utilisées si présentes

### Sorties
| Section | Contenu |
|---|---|
| **Métriques globales** | MAE, RMSE, R², R² CV, Biais |
| **Tableau par fold** | Métriques pour chaque fenêtre temporelle |
| **Graphique Observé vs Prédit** | Courbes avec marqueurs de pics et intervalle ±20% |
| **Analyse des résidus** | Scatter résidus + histogramme par fold |
| **Performance par aire** | Tableau trié par MAE, graphique barres coloré |
| **Importance des variables** | Graphique barres horizontales |
| **Synthèse automatique** | Constats calculés + limites documentées |
| **Export CSV** | 3 fichiers téléchargeables (détails, métriques folds, aires) |

### Algorithmes disponibles
- **RandomForest** : Robuste, recommandé par défaut
- **GradientBoosting** : Plus précis sur données régulières
- **ExtraTrees** : Le plus rapide

### Paramètres configurables
- Nombre de folds temporels (2–6)
- Seuil de détection des pics (%)
- Aire de santé pour analyse détaillée

---

## Interprétation des métriques

| Métrique | Excellent | Bon | Moyen | Faible |
|---|---|---|---|---|
| MAE | < 5 cas/sem | 5–15 | 15–30 | > 30 |
| RMSE | < 8 cas/sem | 8–20 | 20–40 | > 40 |
| R² | > 0.80 | 0.65–0.80 | 0.45–0.65 | < 0.45 |
| R² CV | > 0.70 | 0.55–0.70 | 0.40–0.55 | < 0.40 |
| Biais | ±2 cas | ±5 cas | ±10 cas | > ±10 cas |

> ⚠️ Le **R² CV** est la métrique la plus honnête : elle mesure la performance sur des données jamais vues par le modèle.

---

## Différence avec la validation croisée de l'onglet Modélisation

| | Onglet Modélisation (Tab 3) | Onglet Validation (Tab 7) |
|---|---|---|
| **Objectif** | Entraîner et prédire l'avenir | Valider la fiabilité sur le passé |
| **Données test** | Semaines futures inconnues | Semaines passées cachées |
| **Métriques** | R² CV résumé | MAE, RMSE, R², Biais, pics |
| **Analyse spatiale** | Prédictions par aire | Performance par aire |
| **Export** | Prédictions futures CSV | Résidus + métriques CSV |

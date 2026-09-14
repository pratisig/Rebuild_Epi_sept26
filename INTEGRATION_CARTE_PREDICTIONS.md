# 🗺️ Intégration de la Cartographie des Prédictions

## Vue d'ensemble

Ce guide explique comment intégrer le nouveau module de cartographie des prédictions dans `app_paludisme.py`.

## Fonctionnalités ajoutées

✅ **Carte interactive** avec choroplèthe et cercles proportionnels  
✅ **Popups détaillés** avec toutes les informations par aire de santé  
✅ **Système d'alertes** configurable par percentile  
✅ **Statistiques en temps réel** (total, maximum, moyenne, alertes)  
✅ **Export CSV** des prédictions  
✅ **Compatibilité complète** avec le modèle de prédiction existant  

---

## 🛠️ Étape 1 : Placer le fichier

Le fichier `prediction_map_tab.py` doit être dans le **même répertoire** que `app_paludisme.py`.

```
votre_projet/
├── app_paludisme.py
├── prediction_map_tab.py  ← NOUVEAU FICHIER
├── data/
└── ...
```

---

## 📝 Étape 2 : Modifier app_paludisme.py

### 2.1 Ajouter l'import au début du fichier

**Chercher cette section** (vers les lignes 20-30) :

```python
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
# ... autres imports ...
```

**Ajouter après les imports existants :**

```python
# Import module cartographie prédictions
from prediction_map_tab import create_prediction_map_tab
```

---

### 2.2 Modifier la déclaration des onglets

**Chercher cette ligne** (vers la ligne 2000-2500, selon la version) :

```python
tabs = st.tabs(["📊 Dashboard", "🗺️ Cartographie Interactive", "🤖 Modélisation", "📤 Export"])
```

**Remplacer par :**

```python
tabs = st.tabs([
    "📊 Dashboard", 
    "🗺️ Cartographie Interactive", 
    "🤖 Modélisation", 
    "🗺️ Carte Prédictions",  # ← NOUVEAU
    "📤 Export"
])
```

---

### 2.3 Ajouter le code du nouvel onglet

**Chercher le bloc de modélisation** (vers ligne 2800-3000) :

```python
with tabs[2]:  # Modélisation
    st.markdown("### 🤖 Modélisation Prédictive")
    # ... code existant ...
```

**Ajouter JUSTE APRÈS ce bloc :**

```python
# ============================================================
# TAB 3 : CARTOGRAPHIE DES PRÉDICTIONS
# ============================================================

with tabs[3]:  # Carte Prédictions
    create_prediction_map_tab(
        gdf_health=gdf_health, 
        model_results=st.session_state.get('model_results')
    )
```

---

### 2.4 Ajuster l'index du tab Export

**Chercher :**

```python
with tabs[3]:  # Export
    st.markdown("### 📤 Export des Données")
```

**Remplacer par :**

```python
with tabs[4]:  # Export  ← MODIFIÉ : 3 → 4
    st.markdown("### 📤 Export des Données")
```

---

## 🔍 Résumé des modifications

| Fichier | Lignes concernées | Modification |
|---------|-------------------|-------------|
| `app_paludisme.py` | ~25 (imports) | Ajouter `from prediction_map_tab import create_prediction_map_tab` |
| `app_paludisme.py` | ~2200 (tabs) | Ajouter `"🗺️ Carte Prédictions"` dans la liste |
| `app_paludisme.py` | ~2900 (après modélisation) | Ajouter bloc `with tabs[3]:` |
| `app_paludisme.py` | ~3200 (export) | Changer `tabs[3]` → `tabs[4]` |

---

## ✅ Vérification

Après intégration, vous devriez voir **5 onglets** :

1. 📊 Dashboard
2. 🗺️ Cartographie Interactive
3. 🤖 Modélisation
4. 🗺️ Carte Prédictions **← NOUVEAU**
5. 📤 Export

---

## 🎯 Utilisation

### Workflow utilisateur

1. **Charger les données** (CSV cas + shapefile aires de santé)
2. **Aller dans l'onglet "Modélisation"** et lancer une prédiction
3. **Aller dans l'onglet "Carte Prédictions"** → La carte se génère automatiquement

### Contrôles disponibles

| Contrôle | Description |
|---------|-------------|
| **Sélection semaine** | Choisir la semaine à visualiser |
| **Mode visualisation** | Choroplèthe / Cercles / Combiné |
| **Seuil d'alerte** | Percentile 50-95% pour définir les alertes |

### Contenu des popups

Chaque aire de santé affiche :

✅ **Prédiction principale** (nombre de cas + indicateur alerte)  
✅ **Données démographiques** (population totale, enfants 0-14 ans, densité)  
✅ **Taux d'incidence prédit** (pour 10 000 habitants)  
✅ **Conditions climatiques** (température, précipitations, humidité)  
✅ **Facteurs environnementaux** (risque inondation, distance rivière, altitude)  

---

## 🚨 Dépannage

### Erreur : `ModuleNotFoundError: No module named 'prediction_map_tab'`

**Cause** : Le fichier `prediction_map_tab.py` n'est pas dans le bon répertoire.  
**Solution** : Vérifier que les deux fichiers sont au même niveau.

### Erreur : `KeyError: 'df_predictions'`

**Cause** : Aucune prédiction n'a été lancée.  
**Solution** : Aller dans l'onglet "Modélisation" et lancer une prédiction d'abord.

### Onglet vide ou message d'avertissement

**Cause** : Les données de prédiction ne sont pas dans `st.session_state`.  
**Solution** : Relancer la modélisation. Vérifier que le code de modélisation sauvegarde bien :

```python
st.session_state['model_results'] = {
    'df_predictions': df_predictions,
    'model_info': {...}
}
```

### Popups ne s'affichent pas

**Cause** : Problème de version Folium.  
**Solution** : Vérifier la version :

```bash
pip show folium
# Doit être >= 0.14.0
```

Si version antérieure :

```bash
pip install --upgrade folium
```

---

## 🔧 Personnalisation

### Changer les couleurs de la carte

Dans `prediction_map_tab.py`, ligne ~150 :

```python
colormap = linear.YlOrRd_09.scale(...)  # Jaune-Orange-Rouge
```

Options disponibles :
- `linear.YlGnBu_09` (Jaune-Vert-Bleu)
- `linear.RdPu_09` (Rouge-Violet)
- `linear.PuRd_09` (Violet-Rouge)
- `linear.Viridis_09` (Bleu-Vert-Jaune)

### Modifier le seuil d'alerte par défaut

Ligne ~60 :

```python
value=75,  # Changer cette valeur (50-95)
```

### Ajouter des données dans les popups

Chercher la section `# Construction du HTML du popup` (ligne ~180) et ajouter :

```python
if 'votre_colonne' in row and not pd.isna(row['votre_colonne']):
    popup_html += f"""
    <div style="...">
        <span>{row['votre_colonne']}</span>
    </div>
    """
```

---

## 📚 Références

- [Documentation Folium](https://python-visualization.github.io/folium/)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [GeoPandas User Guide](https://geopandas.org/en/stable/)

---

## ❓ Support

En cas de problème :

1. Vérifier que toutes les étapes ont été suivies
2. Consulter les logs de Streamlit (`streamlit run app_paludisme.py`)
3. Vérifier que les données de prédiction existent : 
   - Ouvrir l'onglet "Modélisation"
   - Lancer une prédiction
   - Attendre la fin du calcul
   - Retourner dans "Carte Prédictions"

---

**Version** : 1.0  
**Date** : Mars 2026  
**Auteur** : System Integration Team  
**Compatibilité** : EpiMonitoring v3.0+

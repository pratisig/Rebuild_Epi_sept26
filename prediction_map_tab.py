# ============================================================
# MODULE CARTOGRAPHIE DES PRÉDICTIONS
# À intégrer dans app_paludisme.py après l'onglet Modélisation
# ============================================================

import streamlit as st
import folium
from folium import Popup, Tooltip, CircleMarker
import pandas as pd
import numpy as np
from branca.colormap import linear

def create_prediction_map_tab(gdf_health, model_results):
    """
    Crée un onglet cartographie des prédictions avec popups détaillés
    
    Args:
        gdf_health: GeoDataFrame des aires de santé
        model_results: Dictionnaire contenant 'df_future' et 'model_info'
    """
    
    st.markdown("### 🗺️ Cartographie des Prédictions")
    st.info("📍 Carte interactive des cas prédits avec popups détaillés par aire de santé")
    
    if model_results is None or 'df_future' not in model_results:
        st.warning("⚠️ Aucune prédiction disponible. Veuillez d'abord exécuter la modélisation.")
        return
    
    df_pred = model_results['df_future'].copy()

    # ── ★ Nettoyage universel : supprimer espaces dans tous les noms de colonnes
    df_pred.columns = df_pred.columns.str.strip()

    # ── Normalisation des noms de colonnes ──────────────────────
    col_rename = {}
    for col in df_pred.columns:
        col_stripped = col.strip()
        # weeknum* -> week_num
        if col_stripped in ('weeknum', 'week_num_', 'weeknum_') and 'week_num' not in df_pred.columns:
            col_rename[col] = 'week_num'
        # predictedcases -> predicted_cases
        if col_stripped == 'predictedcases' and 'predicted_cases' not in df_pred.columns:
            col_rename[col] = 'predicted_cases'
    if col_rename:
        df_pred = df_pred.rename(columns=col_rename)

    # health_area
    if 'health_area' not in df_pred.columns:
        for candidate in ['healtharea', 'Aire_Sante', 'aire_sante', 'area', 'name']:
            if candidate in df_pred.columns:
                df_pred = df_pred.rename(columns={candidate: 'health_area'})
                break
    
    # ── Vérification de sécurité ────────────────────────────────
    required_cols = ['health_area', 'week_num', 'predicted_cases']
    missing = [c for c in required_cols if c not in df_pred.columns]
    if missing:
        st.error(f"❌ Colonnes manquantes dans les prédictions : {missing}")
        st.write("Colonnes disponibles :", list(df_pred.columns))
        return

    # ============================================================
    # CONFIGURATION UTILISATEUR
    # ============================================================
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        weeks_available = sorted(df_pred['week_num'].unique())
        selected_week = st.selectbox(
            "📅 Semaine à visualiser",
            weeks_available,
            index=len(weeks_available)-1 if weeks_available else 0,
            key="pred_map_week"
        )
    
    with col2:
        viz_mode = st.radio(
            "Mode de visualisation",
            ["Choroplèthe", "Cercles proportionnels", "Combiné"],
            key="pred_viz_mode"
        )
    
    with col3:
        alert_threshold = st.slider(
            "Seuil d'alerte (percentile)",
            min_value=50,
            max_value=95,
            value=75,
            step=5,
            key="pred_alert_threshold"
        )
    
    # ============================================================
    # PRÉPARATION DES DONNÉES
    # ============================================================
    
    df_week = df_pred[df_pred['week_num'] == selected_week].copy()
    
    if df_week.empty:
        st.error(f"❌ Aucune prédiction pour la semaine {selected_week}")
        return
    
    # Fusionner avec géométries
    gdf_map = gdf_health.merge(
        df_week,
        left_on='health_area',
        right_on='health_area',
        how='left'
    )
    
    # Remplir valeurs manquantes
    gdf_map['predicted_cases'] = gdf_map['predicted_cases'].fillna(0)
    
    # Calculer seuil d'alerte
    threshold_value = gdf_map['predicted_cases'].quantile(alert_threshold / 100)
    gdf_map['is_alert'] = gdf_map['predicted_cases'] > threshold_value
    
    # ============================================================
    # STATISTIQUES
    # ============================================================
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pred = int(gdf_map['predicted_cases'].sum())
        st.metric("📊 Cas prédits totaux", f"{total_pred:,}")
    
    with col2:
        max_pred = int(gdf_map['predicted_cases'].max())
        max_area = gdf_map.loc[gdf_map['predicted_cases'].idxmax(), 'health_area']
        st.metric("🔴 Maximum", f"{max_pred:,}", delta=f"{max_area}")
    
    with col3:
        avg_pred = int(gdf_map['predicted_cases'].mean())
        st.metric("📈 Moyenne", f"{avg_pred:,}")
    
    with col4:
        n_alerts = gdf_map['is_alert'].sum()
        st.metric("⚠️ Aires en alerte", f"{n_alerts}", delta=f"{n_alerts/len(gdf_map)*100:.1f}%")
    
    # ============================================================
    # CRÉATION DE LA CARTE
    # ============================================================
    
    # Centre de la carte
    center_lat = gdf_map.geometry.centroid.y.mean()
    center_lon = gdf_map.geometry.centroid.x.mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='CartoDB positron'
    )
    
    # ============================================================
    # COUCHE CHOROPLÈTHE
    # ============================================================
    
    if viz_mode in ["Choroplèthe", "Combiné"]:
        
        # Colormap
        colormap = linear.YlOrRd_09.scale(
            gdf_map['predicted_cases'].min(),
            gdf_map['predicted_cases'].max()
        )
        colormap.caption = 'Cas prédits'
        
        # GeoJson avec style et popups
        def style_function(feature):
            pred_value = feature['properties'].get('predicted_cases', 0)
            return {
                'fillColor': colormap(pred_value) if pred_value > 0 else '#ffffff',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6 if not feature['properties'].get('is_alert') else 0.8
            }
        
        def highlight_function(feature):
            return {
                'fillColor': '#ff6b6b',
                'color': '#c92a2a',
                'weight': 3,
                'fillOpacity': 0.9
            }
        
        # Créer popups détaillés
        for idx, row in gdf_map.iterrows():
            
            # Construction du HTML du popup
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 350px; padding: 10px;">
                
                <!-- En-tête -->
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 15px; border-radius: 8px 8px 0 0; 
                            margin: -10px -10px 15px -10px;">
                    <h3 style="margin: 0; font-size: 18px; font-weight: bold;">
                        📍 {row['health_area'].upper()}
                    </h3>
                    <p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.9;">
                        Semaine {selected_week} | Prédiction ML
                    </p>
                </div>
                
                <!-- Prédiction principale -->
                <div style="background: {'#ffe3e3' if row['is_alert'] else '#e7f5ff'}; 
                            padding: 15px; border-radius: 8px; margin-bottom: 15px; 
                            border-left: 4px solid {'#fa5252' if row['is_alert'] else '#339af0'};">
                    <div style="font-size: 14px; color: #495057; margin-bottom: 5px;">
                        {'🚨 ALERTE' if row['is_alert'] else '📊 Prévision'}
                    </div>
                    <div style="font-size: 32px; font-weight: bold; 
                                color: {'#c92a2a' if row['is_alert'] else '#1971c2'};">
                        {int(row['predicted_cases']):,}
                    </div>
                    <div style="font-size: 12px; color: #868e96; margin-top: 5px;">
                        cas prédits
                    </div>
                </div>
                
                <!-- Données démographiques -->
                <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="font-size: 13px; font-weight: bold; color: #495057; margin-bottom: 8px;">
                        👥 Données population
                    </div>
            """
            
            # Population totale
            if 'Pop_Totale' in row and not pd.isna(row['Pop_Totale']):
                pop_tot = int(row['Pop_Totale'])
                popup_html += f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #868e96; font-size: 12px;">Population totale:</span>
                        <span style="font-weight: bold; font-size: 12px; color: #495057;">{pop_tot:,}</span>
                    </div>
                """
            
            # Enfants 0-14 ans
            if 'Pop_Enfants_0_14' in row and not pd.isna(row['Pop_Enfants_0_14']):
                pop_enf = int(row['Pop_Enfants_0_14'])
                popup_html += f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #868e96; font-size: 12px;">Enfants 0-14 ans:</span>
                        <span style="font-weight: bold; font-size: 12px; color: #495057;">{pop_enf:,}</span>
                    </div>
                """
            
            # Densité
            if 'Densite_Pop' in row and not pd.isna(row['Densite_Pop']):
                densite = float(row['Densite_Pop'])
                popup_html += f"""
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #868e96; font-size: 12px;">Densité pop.:</span>
                        <span style="font-weight: bold; font-size: 12px; color: #495057;">{densite:.1f} hab/km²</span>
                    </div>
                """
            
            popup_html += "</div>"
            
            # Taux d'incidence
            if 'Pop_Totale' in row and not pd.isna(row['Pop_Totale']) and row['Pop_Totale'] > 0:
                incidence = (row['predicted_cases'] / row['Pop_Totale']) * 10000
                popup_html += f"""
                <div style="background: #fff3bf; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="font-size: 13px; font-weight: bold; color: #495057; margin-bottom: 5px;">
                        📈 Taux d'incidence prédit
                    </div>
                    <div style="font-size: 20px; font-weight: bold; color: #f59f00;">
                        {incidence:.1f}
                    </div>
                    <div style="font-size: 11px; color: #868e96;">pour 10 000 habitants</div>
                </div>
                """
            
            # Données climatiques si disponibles
            climate_data = []
            if 'temp_api' in row and not pd.isna(row['temp_api']):
                climate_data.append(('🌡️ Température', f"{row['temp_api']:.1f}°C"))
            if 'precip_api' in row and not pd.isna(row['precip_api']):
                climate_data.append(('🌧️ Précipitations', f"{row['precip_api']:.1f} mm"))
            if 'humidity_api' in row and not pd.isna(row['humidity_api']):
                climate_data.append(('💧 Humidité', f"{row['humidity_api']:.1f}%"))
            
            if climate_data:
                popup_html += """
                <div style="background: #e7f5ff; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="font-size: 13px; font-weight: bold; color: #495057; margin-bottom: 8px;">
                        🌍 Conditions climatiques
                    </div>
                """
                for label, value in climate_data:
                    popup_html += f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #868e96; font-size: 12px;">{label}:</span>
                        <span style="font-weight: bold; font-size: 12px; color: #1971c2;">{value}</span>
                    </div>
                    """
                popup_html += "</div>"
            
            # Facteurs de risque environnementaux
            env_factors = []
            if 'flood_mean' in row and not pd.isna(row['flood_mean']):
                env_factors.append(('💦 Risque inondation', f"{row['flood_mean']:.2f}"))
            if 'dist_river' in row and not pd.isna(row['dist_river']):
                env_factors.append(('🏞️ Distance rivière', f"{row['dist_river']:.1f} km"))
            if 'elevation_mean' in row and not pd.isna(row['elevation_mean']):
                env_factors.append(('⛰️ Altitude moyenne', f"{row['elevation_mean']:.0f} m"))
            
            if env_factors:
                popup_html += """
                <div style="background: #f3f0ff; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 13px; font-weight: bold; color: #495057; margin-bottom: 8px;">
                        🌿 Facteurs environnementaux
                    </div>
                """
                for label, value in env_factors:
                    popup_html += f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #868e96; font-size: 12px;">{label}:</span>
                        <span style="font-weight: bold; font-size: 12px; color: #7950f2;">{value}</span>
                    </div>
                    """
                popup_html += "</div>"
            
            popup_html += "</div>"
            
            # Créer le popup
            popup = folium.Popup(popup_html, max_width=400)
            
            # Ajouter tooltip simple
            tooltip = folium.Tooltip(
                f"<b>{row['health_area'].upper()}</b><br>" +
                f"Prédiction: {int(row['predicted_cases'])} cas",
                sticky=True
            )
            
            # Ajouter la géométrie
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x, pred=row['predicted_cases'], alert=row['is_alert']: {
                    'fillColor': colormap(pred) if pred > 0 else '#ffffff',
                    'color': '#c92a2a' if alert else 'black',
                    'weight': 2 if alert else 1,
                    'fillOpacity': 0.8 if alert else 0.6
                },
                highlight_function=highlight_function,
                tooltip=tooltip,
                popup=popup
            ).add_to(m)
        
        colormap.add_to(m)
    
    # ============================================================
    # COUCHE CERCLES PROPORTIONNELS
    # ============================================================
    
    if viz_mode in ["Cercles proportionnels", "Combiné"]:
        
        # Normaliser tailles
        max_pred = gdf_map['predicted_cases'].max()
        min_radius = 5
        max_radius = 30
        
        for idx, row in gdf_map.iterrows():
            if row['predicted_cases'] > 0:
                
                # Calculer rayon proportionnel
                normalized = (row['predicted_cases'] / max_pred) if max_pred > 0 else 0
                radius = min_radius + (max_radius - min_radius) * normalized
                
                # Couleur selon alerte
                color = '#c92a2a' if row['is_alert'] else '#1971c2'
                
                # Tooltip simple
                tooltip_text = (
                    f"<b>{row['health_area'].upper()}</b><br>"
                    f"Prédiction: {int(row['predicted_cases'])} cas"
                )
                
                # Cercle
                folium.CircleMarker(
                    location=[row.geometry.centroid.y, row.geometry.centroid.x],
                    radius=radius,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    weight=2,
                    tooltip=tooltip_text
                ).add_to(m)
    
    # ============================================================
    # CONTRÔLES ET AFFICHAGE
    # ============================================================
    
    folium.LayerControl().add_to(m)
    
    # Afficher la carte
    st.components.v1.html(m._repr_html_(), height=600, scrolling=False)
    
    # ============================================================
    # TABLEAU RÉCAPITULATIF
    # ============================================================
    
    st.markdown("### 📋 Tableau récapitulatif des prédictions")
    
    # Préparer données d'affichage
    display_cols = ['health_area', 'predicted_cases']
    for c in ['Pop_Totale', 'Pop_Enfants_0_14', 'is_alert']:
        if c in gdf_map.columns:
            display_cols.append(c)
        else:
            gdf_map[c] = np.nan if c != 'is_alert' else False
            display_cols.append(c)

    df_display = gdf_map[display_cols].copy()
    
    # Calculer taux
    if 'Pop_Totale' in df_display.columns:
        df_display['Taux_10000'] = (
            df_display['predicted_cases'] / df_display['Pop_Totale'] * 10000
        ).round(2)
    
    # Renommer colonnes
    rename_map = {
        'health_area': 'Aire de santé',
        'predicted_cases': 'Cas prédits',
        'Pop_Totale': 'Population totale',
        'Pop_Enfants_0_14': 'Enfants 0-14 ans',
        'is_alert': 'Alerte',
        'Taux_10000': 'Taux p.10000'
    }
    df_display = df_display.rename(columns=rename_map)
    
    # Trier par prédiction
    df_display = df_display.sort_values('Cas prédits', ascending=False)
    
    # Formater alerte
    if 'Alerte' in df_display.columns:
        df_display['Alerte'] = df_display['Alerte'].map({True: '🚨 OUI', False: '✅ Non'})
    
    # Afficher
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )
    
    # Export
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger le tableau (CSV)",
        data=csv,
        file_name=f"predictions_semaine_{selected_week}.csv",
        mime="text/csv"
    )

# ============================================================
# FONCTION D'INTÉGRATION DANS L'APP PRINCIPALE
# ============================================================

def integrate_into_main_app():
    """
    Instructions pour intégrer ce module dans app_paludisme.py
    
    ÉTAPE 1: Importer ce module au début du fichier
    ```python
    from prediction_map_tab import create_prediction_map_tab
    ```
    
    ÉTAPE 2: Ajouter l'onglet après "Modélisation"
    Chercher la ligne avec:
    ```python
    tabs = st.tabs(["📊 Dashboard", "🗺️ Cartographie Interactive", "🤖 Modélisation", "📤 Export"])
    ```
    
    Remplacer par:
    ```python
    tabs = st.tabs(["📊 Dashboard", "🗺️ Cartographie Interactive", 
                    "🤖 Modélisation", "🗺️ Carte Prédictions", "📤 Export"])
    ```
    
    ÉTAPE 3: Ajouter le nouveau tab après le tab[2] (Modélisation)
    ```python
    with tabs[3]:  # Carte Prédictions
        create_prediction_map_tab(gdf_health, st.session_state.get('model_results'))
    ```
    
    ÉTAPE 4: Ajuster l'index du tab Export (devient tabs[4] au lieu de tabs[3])
    """
    pass

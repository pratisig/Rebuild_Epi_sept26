# ============================================================
# MODULE VALIDATION RÉTROSPECTIVE DU MODÈLE PRÉDICTIF
# Intégration dans app_paludisme.py
# Usage : from validation_tab import create_validation_tab
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def safe_r2(y_true, y_pred):
    """R² sécurisé — retourne NaN si variance nulle."""
    try:
        if np.std(y_true) < 1e-6:
            return np.nan
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def detect_peaks(series, window=3, threshold_pct=50):
    """
    Détecte les pics épidémiques dans une série temporelle.
    Un pic = valeur > moyenne mobile × (1 + threshold_pct/100)
    Retourne un masque booléen.
    """
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    return series > rolling_mean * (1 + threshold_pct / 100)


def compute_peak_detection_metrics(y_true, y_pred, weeks, window=3, threshold_pct=50):
    """
    Évalue la capacité du modèle à détecter les pics épidémiques.
    Retourne : sensitivity, specificity, false_alarm_rate
    """
    true_s = pd.Series(y_true.values if hasattr(y_true, "values") else y_true)
    pred_s = pd.Series(y_pred.values if hasattr(y_pred, "values") else y_pred)

    true_peaks = detect_peaks(true_s, window, threshold_pct)
    pred_peaks = detect_peaks(pred_s, window, threshold_pct)

    tp = (true_peaks & pred_peaks).sum()
    fp = (~true_peaks & pred_peaks).sum()
    fn = (true_peaks & ~pred_peaks).sum()
    tn = (~true_peaks & ~pred_peaks).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    false_alarm  = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "sensitivity":     round(sensitivity * 100, 1) if not np.isnan(sensitivity) else None,
        "specificity":     round(specificity * 100, 1) if not np.isnan(specificity) else None,
        "false_alarm_pct": round(false_alarm  * 100, 1) if not np.isnan(false_alarm)  else None,
        "n_true_peaks":    int(true_peaks.sum()),
        "n_detected":      int((true_peaks & pred_peaks).sum()),
        "n_missed":        int(fn),
        "n_false_alarms":  int(fp),
    }


def build_features(df, feature_cols):
    """Construit la matrice X en gérant les NaN."""
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def interpret_r2(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if val >= 0.80: return "🟢 Excellent"
    if val >= 0.65: return "🟡 Bon"
    if val >= 0.45: return "🟠 Moyen"
    return "🔴 Faible"


def interpret_bias(val):
    if abs(val) < 2:  return "✅ Bien centré"
    if val > 0:       return "⬆️ Sur-estimation"
    return              "⬇️ Sous-estimation"


def interpret_mae_contextual(mae, mean_cases):
    """
    Interprète le MAE de façon relative à la moyenne des cas observés.
    Le MAE absolu n'a de sens que rapporté à l'ordre de grandeur des données.
    """
    if mean_cases is None or mean_cases <= 0:
        return "❓ Moyenne des cas non disponible pour contextualiser"

    mae_pct = (mae / mean_cases) * 100
    rel_str = f"{mae_pct:.1f}% de la moyenne observée ({mean_cases:.0f} cas)"

    if mae_pct <= 15:
        return f"🟢 Excellent — erreur de {mae:.0f} cas soit {rel_str}. Le modèle est très précis."
    if mae_pct <= 30:
        return f"🟡 Bon — erreur de {mae:.0f} cas soit {rel_str}. Précision acceptable pour la planification."
    if mae_pct <= 50:
        return f"🟠 Moyen — erreur de {mae:.0f} cas soit {rel_str}. Interprétez avec prudence ; envisagez l'ajout de variables explicatives."
    return f"🔴 Élevé — erreur de {mae:.0f} cas soit {rel_str}. MAE élevé MAIS NORMAL si vos données ont des pics à 500–2000 cas. Vérifiez le R² et la sensibilité aux pics."


def interpret_rmse_contextual(rmse, mean_cases):
    """
    Interprète le RMSE de façon relative à la moyenne des cas.
    Le RMSE pénalise davantage les erreurs sur les pics.
    """
    if mean_cases is None or mean_cases <= 0:
        return "❓ Moyenne des cas non disponible"

    rmse_pct = (rmse / mean_cases) * 100

    if rmse_pct <= 20:
        return f"🟢 Excellent — RMSE = {rmse:.0f} cas ({rmse_pct:.1f}% de la moyenne). Bonne gestion des erreurs importantes."
    if rmse_pct <= 40:
        return f"🟡 Bon — RMSE = {rmse:.0f} cas ({rmse_pct:.1f}% de la moyenne). Quelques pics génèrent des erreurs élevées ponctuellement."
    if rmse_pct <= 70:
        return f"🟠 Moyen — RMSE = {rmse:.0f} cas ({rmse_pct:.1f}% de la moyenne). Les pics sont difficiles à prévoir précisément — comportement attendu dans le Sahel."
    return f"🔴 RMSE élevé = {rmse:.0f} cas ({rmse_pct:.1f}% de la moyenne). Normal si données très hétérogènes ou pics extrêmes. Comparez au R² CV pour confirmer l'utilité globale."


def build_interpretations_list(global_mae, global_rmse, cv_r2_mean, global_bias,
                                ci_std, sens_mean, peak_str, best_area, best_mae,
                                worst_area, worst_mae, mean_cases):
    """Génère la liste de phrases d'interprétation enrichie pour le rapport."""
    mae_interp  = interpret_mae_contextual(global_mae, mean_cases)
    rmse_interp = interpret_rmse_contextual(global_rmse, mean_cases)
    verdict_r2  = interpret_r2(cv_r2_mean)
    verdict_bias = interpret_bias(global_bias)

    lines = [
        f"R² CV moyen = {cv_r2_mean:.3f} → {verdict_r2} (métrique principale de généralisation)",
        f"MAE (erreur absolue moyenne) : {mae_interp}",
        f"RMSE (erreur quadratique) : {rmse_interp}",
        f"Biais pondéré : {verdict_bias} ({global_bias:+.1f} cas/semaine en moyenne)",
        f"Intervalle de confiance : ±{ci_std:.1f} cas (±1σ résidus réels — non fictif)",
        f"Détection de pics : {peak_str}",
        f"Aire la mieux prédite : {best_area} (MAE = {best_mae})",
        f"Aire la moins bien prédite : {worst_area} (MAE = {worst_mae})",
        "Note : un MAE de 300-600 cas est NORMAL si vos zones ont des pics saisonniers à 1000-3000 cas.",
        "Limites : lags supposent continuité ; chocs non-climatiques non captés ; validation interne.",
    ]
    return lines


# ============================================================
# GUIDE PÉDAGOGIQUE DES MÉTRIQUES (bloc réutilisable)
# ============================================================

def _show_metrics_guide():
    """
    Affiche un encadré pédagogique expliquant chaque métrique
    de façon compréhensible pour un non-technicien de terrain.
    """
    with st.expander("📚 Comprendre les métriques — Guide pour non-techniciens", expanded=False):
        st.markdown("""
### 🎯 À quoi servent ces métriques ?

Après avoir entraîné le modèle, on lui présente des données qu'il n'a **jamais vues**.
On compare ce qu'il prédit à ce qui s'est réellement passé.
Les métriques ci-dessous mesurent **à quel point il se trompe** — et dans quel sens.

---

### 📏 MAE — Erreur Absolue Moyenne
> *"En moyenne, de combien de cas le modèle se trompe-t-il ?"*

**Comment la lire :**
- Si MAE = **50 cas** et votre moyenne est **200 cas/semaine** → le modèle se trompe de 25% → 🟡 Bon
- Si MAE = **50 cas** et votre moyenne est **2000 cas/semaine** → le modèle se trompe de 2.5% → 🟢 Excellent
- ⚠️ **Le MAE seul ne veut rien dire** : il faut toujours le diviser par la moyenne de vos données

**Analogie :** Si vous estimez chaque jour le nombre de patients à la clinique et que vous vous trompez en moyenne de 10 patients, c'est excellent si la clinique reçoit 500 patients, mais problématique si elle en reçoit 15.

**Seuils relatifs (MAE / moyenne des cas) :**
| < 15% | 15–30% | 30–50% | > 50% |
|--------|--------|--------|-------|
| 🟢 Excellent | 🟡 Bon | 🟠 Moyen | 🔴 Élevé |

---

### 📐 RMSE — Racine de l'Erreur Quadratique Moyenne
> *"Le modèle gère-t-il bien les semaines de forte épidémie ?"*

**Comment la lire :**
- Le RMSE **pénalise plus sévèrement** les grosses erreurs que le MAE
- Un RMSE beaucoup plus grand que le MAE signifie que quelques semaines de pic génèrent de grosses erreurs
- Dans le Sahel, RMSE > MAE est **normal** : les semaines de pic paludisme sont très difficiles à prévoir précisément

**Exemple :** MAE = 80 cas, RMSE = 250 cas → les semaines "normales" sont bien prédites, mais les semaines de pic (500+ cas) génèrent des erreurs importantes. C'est acceptable.

**Seuils relatifs (RMSE / moyenne des cas) :**
| < 20% | 20–40% | 40–70% | > 70% |
|--------|--------|--------|-------|
| 🟢 Excellent | 🟡 Bon | 🟠 Moyen | 🔴 Élevé |

---

### 📈 R² — Coefficient de Détermination
> *"Le modèle explique-t-il les variations de l'épidémie ?"*

**Comment le lire :**
- R² = **1.0** → le modèle est parfait (impossible en pratique)
- R² = **0.80** → le modèle explique 80% des variations → 🟢 Excellent
- R² = **0.60** → il explique 60% des variations → 🟡 Acceptable
- R² = **0.0** → le modèle ne vaut pas mieux qu'utiliser la moyenne → 🔴 Inutile
- R² < 0 → le modèle est pire que la moyenne → très mauvais

**⚠️ Piège :** Le R² "global" est calculé sur les données d'entraînement ET de test combinées. Il peut paraître bon même si le modèle ne généralise pas bien. **Préférez toujours le R² CV.**

| > 0.80 | 0.65–0.80 | 0.45–0.65 | < 0.45 |
|--------|-----------|-----------|--------|
| 🟢 Excellent | 🟡 Bon | 🟠 Moyen | 🔴 Faible |

---

### 🔁 R² CV — R² de Validation Croisée *(métrique principale)*
> *"Le modèle sera-t-il encore fiable sur de nouvelles données ?"*

**C'est la métrique la plus importante de cet onglet.**

- Il est calculé en testant le modèle sur **des semaines qu'il n'a jamais vues** (folds)
- Un R² CV de 0.70 signifie : si vous utilisez ce modèle l'année prochaine, il expliquera ~70% des variations futures
- Il est toujours **inférieur au R² global** — c'est normal

**Règle d'or :**
| R² CV > 0.70 | 0.55–0.70 | 0.40–0.55 | < 0.40 |
|--------------|-----------|-----------|--------|
| 🟢 Fiable pour la planification | 🟡 Utile avec prudence | 🟠 Indicatif seulement | 🔴 Non fiable |

---

### ⚖️ Biais — Sur-estimation ou Sous-estimation systématique
> *"Le modèle a-t-il tendance à toujours prédire trop haut ou trop bas ?"*

- **Biais positif (+10 cas)** → le modèle surestime systématiquement → risque d'alerte excessive
- **Biais négatif (−10 cas)** → le modèle sous-estime → risque de sous-réaction terrain
- **Biais ≈ 0** → prédictions centrées → idéal

**Impact terrain :**
- Biais positif → surstockage de médicaments, surcharge des équipes
- Biais négatif → rupture de stock, réponse tardive aux pics

---

### 📊 Intervalle de confiance ±1σ (zone bleue sur le graphique)
> *"Dans quelle fourchette se situe probablement la vraie valeur ?"*

- C'est la **dispersion réelle des erreurs de prédiction** observées pendant la validation
- ±1σ contient ~68% des vraies valeurs si les erreurs sont distribuées normalement
- ±2σ (non affiché) contiendrait ~95% des vraies valeurs
- Plus cette zone est étroite, plus le modèle est précis

---

### 🔴 Sensibilité aux pics — Détection des épidémies
> *"Le modèle prévient-il bien avant les semaines de forte épidémie ?"*

- **Sensibilité = 80%** → sur 10 semaines de pic réel, le modèle en prédit 8 → 🟢
- **Fausses alarmes = 15%** → sur 10 semaines "normales", le modèle déclenche une fausse alerte dans 1.5 → acceptable
- Pour la planification médicale, **une sensibilité élevée** (détecter un maximum de vrais pics) est prioritaire sur les fausses alarmes

---

> 💡 **En résumé :** Regardez d'abord le **R² CV** pour savoir si le modèle est fiable, puis le **MAE%** pour quantifier l'erreur en termes opérationnels, puis le **biais** pour détecter une dérive systématique.
        """)


# ============================================================
# CALCUL DE LA VALIDATION (appelé une seule fois au clic)
# ============================================================

def _run_validation_compute(df_model, available_features, algo_choice, n_splits, peak_threshold):
    """
    Exécute les folds TimeSeriesSplit et retourne un dict complet
    stocké dans st.session_state pour persister entre les re-renders.
    """
    X         = build_features(df_model, available_features)
    y         = df_model["cases"].values
    weeks_all = df_model["week_num"].values
    areas_all = df_model["health_area"].values

    if algo_choice == "RandomForest":
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif algo_choice == "GradientBoosting":
        estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        estimator = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics            = []
    all_true                = []
    all_pred                = []
    all_weeks               = []
    all_areas               = []
    fold_labels             = []
    feature_importance_list = []

    progress = st.progress(0)
    status   = st.empty()

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        status.text(
            f"Fold {fold_idx + 1}/{n_splits} — "
            f"Entraînement sur {len(train_idx)} obs, test sur {len(test_idx)} obs…"
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx],      y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = np.maximum(pipeline.predict(X_test), 0)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = safe_r2(y_test, y_pred)
        bias = float(np.mean(y_pred - y_test))

        peak_metrics = compute_peak_detection_metrics(
            pd.Series(y_test), pd.Series(y_pred),
            weeks_all[test_idx], threshold_pct=peak_threshold
        )

        fold_metrics.append({
            "Fold":                 f"Fold {fold_idx + 1}",
            "N train":              len(train_idx),
            "N test":               len(test_idx),
            "Semaines test":        f"{weeks_all[test_idx].min()}–{weeks_all[test_idx].max()}",
            "MAE":                  round(mae,  2),
            "RMSE":                 round(rmse, 2),
            "R²":                   round(r2,   3) if not np.isnan(r2) else None,
            "Biais":                round(bias, 2),
            "Sensibilité pics (%)": peak_metrics["sensitivity"],
            "Fausses alarmes (%)":  peak_metrics["false_alarm_pct"],
            "Pics réels":           peak_metrics["n_true_peaks"],
            "Pics détectés":        peak_metrics["n_detected"],
            "Pics manqués":         peak_metrics["n_missed"],
        })

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        all_weeks.extend(weeks_all[test_idx].tolist())
        all_areas.extend(areas_all[test_idx].tolist())
        fold_labels.extend([f"Fold {fold_idx + 1}"] * len(test_idx))

        if hasattr(pipeline["model"], "feature_importances_"):
            imp = pipeline["model"].feature_importances_
            feature_importance_list.append(dict(zip(available_features, imp)))

        progress.progress((fold_idx + 1) / n_splits)

    progress.empty()
    status.empty()

    df_results = pd.DataFrame({
        "health_area": all_areas,
        "week_num":    all_weeks,
        "observed":    all_true,
        "predicted":   all_pred,
        "fold":        fold_labels,
        "residual":    [p - t for p, t in zip(all_pred, all_true)],
        "abs_error":   [abs(p - t) for p, t in zip(all_pred, all_true)],
    })

    df_metrics = pd.DataFrame(fold_metrics)

    # Métriques globales
    global_mae  = mean_absolute_error(all_true, all_pred)
    global_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    global_r2   = safe_r2(all_true, all_pred)
    mean_cases  = float(np.mean(all_true)) if all_true else None

    # Biais pondéré par taille de fold
    fold_sizes  = [m["N test"] for m in fold_metrics]
    fold_biases = [m["Biais"]  for m in fold_metrics]
    total_obs   = sum(fold_sizes)
    global_bias = float(
        sum(b * n for b, n in zip(fold_biases, fold_sizes)) / total_obs
    ) if total_obs > 0 else float(np.mean(np.array(all_pred) - np.array(all_true)))

    cv_r2_mean = df_metrics["R²"].dropna().mean()

    residuals_all = np.array(all_pred) - np.array(all_true)
    ci_std = float(np.std(residuals_all))

    if "Sensibilité pics (%)" in df_metrics.columns:
        sens_values = [
            v for v in df_metrics["Sensibilité pics (%)"].tolist()
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        sens_mean = float(np.mean(sens_values)) if sens_values else None
    else:
        sens_mean = None

    n_peaks_total    = int(df_metrics["Pics réels"].sum())    if "Pics réels"    in df_metrics.columns else 0
    n_detected_total = int(df_metrics["Pics détectés"].sum()) if "Pics détectés" in df_metrics.columns else 0

    area_metrics = df_results.groupby("health_area").apply(
        lambda g: pd.Series({
            "MAE":              round(mean_absolute_error(g["observed"], g["predicted"]), 1),
            "RMSE":             round(np.sqrt(mean_squared_error(g["observed"], g["predicted"])), 1),
            "R²":               round(safe_r2(g["observed"], g["predicted"]), 3),
            "Biais moyen":      round((g["predicted"] - g["observed"]).mean(), 1),
            "N semaines":       len(g),
            "Cas observés moy": round(g["observed"].mean(), 1),
        })
    ).reset_index()
    area_metrics["MAE/Moy(%)"] = (
        area_metrics["MAE"] / area_metrics["Cas observés moy"].replace(0, np.nan) * 100
    ).round(1)
    area_metrics["Qualité"] = area_metrics["R²"].apply(interpret_r2)
    area_metrics = area_metrics.sort_values("MAE", ascending=True).reset_index(drop=True)

    imp_df = None
    if feature_importance_list:
        imp_df = pd.DataFrame(feature_importance_list).mean().reset_index()
        imp_df.columns = ["Variable", "Importance"]
        imp_df = imp_df.sort_values("Importance", ascending=True)

    return {
        "df_results":        df_results,
        "df_metrics":        df_metrics,
        "area_metrics":      area_metrics,
        "imp_df":            imp_df,
        "global_mae":        global_mae,
        "global_rmse":       global_rmse,
        "global_r2":         global_r2,
        "global_bias":       global_bias,
        "cv_r2_mean":        cv_r2_mean,
        "ci_std":            ci_std,
        "sens_mean":         sens_mean,
        "n_peaks_total":     n_peaks_total,
        "n_detected_total":  n_detected_total,
        "algo_choice":       algo_choice,
        "n_splits":          n_splits,
        "peak_threshold":    peak_threshold,
        "mean_cases":        mean_cases,
    }


# ============================================================
# FONCTION PRINCIPALE DE L'ONGLET
# ============================================================

def create_validation_tab(df_cases, gdf_health=None, model_results=None):
    """
    Onglet complet de validation rétrospective.

    Args:
        df_cases      : DataFrame des cas (doit contenir week_num, health_area, cases)
        gdf_health    : GeoDataFrame optionnel pour analyse spatiale
        model_results : Résultats du modèle principal (optionnel, pour comparaison)
    """

    st.markdown("## 🔬 Validation Rétrospective du Modèle Prédictif")
    st.info(
        "📋 **Principe** : Le modèle est entraîné sur des données passées, puis testé sur des "
        "périodes qu'il n'a jamais vues. On compare ensuite ses prédictions aux valeurs réellement "
        "observées pour mesurer sa fiabilité."
    )

    if df_cases is None or df_cases.empty:
        st.warning("⚠️ Aucune donnée épidémiologique chargée. Veuillez d'abord importer votre CSV de cas.")
        return

    # ── Normalisation des colonnes ──────────────────────────────
    df = df_cases.copy()
    if "week_num" not in df.columns and "week_" in df.columns:
        df["week_num"] = pd.to_numeric(df["week_"], errors="coerce")
    if "week_num" not in df.columns:
        st.error("❌ Colonne 'week_num' introuvable. Vérifiez votre fichier CSV.")
        return
    if "cases" not in df.columns:
        st.error("❌ Colonne 'cases' introuvable.")
        return
    if "health_area" not in df.columns:
        st.error("❌ Colonne 'health_area' introuvable.")
        return

    df["cases"]    = pd.to_numeric(df["cases"],    errors="coerce").fillna(0)
    df["week_num"] = pd.to_numeric(df["week_num"], errors="coerce")
    df = df.dropna(subset=["week_num"]).copy()
    df["week_num"] = df["week_num"].astype(int)

    # ── Lags et features temporelles ───────────────────────────
    df = df.sort_values(["health_area", "week_num"])
    for lag in [1, 2, 4]:
        df[f"cases_lag_{lag}"] = df.groupby("health_area")["cases"].shift(lag)
    for win in [2, 4]:
        df[f"cases_ma_{win}"] = df.groupby("health_area")["cases"].transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
    df["sin_week"]    = np.sin(2 * np.pi * df["week_num"] / 52)
    df["cos_week"]    = np.cos(2 * np.pi * df["week_num"] / 52)
    df["growth_rate"] = df.groupby("health_area")["cases"].pct_change().fillna(0).clip(-5, 5)

    climate_features = [c for c in df.columns if c.endswith("_api")]
    env_features = [c for c in df.columns if c in
                    ["flood_mean", "dist_river", "elevation_mean", "precipitation_mean",
                     "humidity_mean", "temp_mean", "flood_risk", "climate_index"]]
    demo_features = [c for c in df.columns if c in
                     ["Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop",
                      "incidence_rate", "child_risk", "demo_pressure"]]

    base_features = ["cases_lag_1", "cases_lag_2", "cases_lag_4",
                     "cases_ma_2", "cases_ma_4", "sin_week", "cos_week", "growth_rate"]
    all_features       = base_features + climate_features + env_features + demo_features
    available_features = [f for f in all_features if f in df.columns]

    df_model = df.dropna(subset=["cases_lag_1", "cases_lag_2"]).copy()

    if len(df_model) < 20:
        st.error("❌ Données insuffisantes pour la validation (minimum 20 observations après calcul des lags).")
        return

    # ── Informations contextuelles sur les données ─────────────
    data_mean_cases = float(df_model["cases"].mean())
    data_max_cases  = float(df_model["cases"].max())
    n_weeks_total   = df_model["week_num"].nunique()
    n_areas         = df_model["health_area"].nunique()

    # ============================================================
    # PANNEAU DE CONFIGURATION
    # ============================================================
    st.markdown("### ⚙️ Configuration de la validation")

    # ── Aide au choix du nombre de folds ───────────────────────
    with st.expander("❓ Quel nombre de folds choisir ?", expanded=False):
        # Calcul dynamique selon les semaines disponibles
        min_train_weeks = max(26, int(n_weeks_total * 0.4))
        fold_guide = []
        for k in range(2, 7):
            obs_per_fold = n_weeks_total // (k + 1)
            min_train    = int(n_weeks_total * k / (k + 1))
            fold_guide.append({
                "Folds (k)": k,
                "Obs. test / fold (≈)": obs_per_fold * n_areas,
                "Train minimum (semaines)": min_train,
                "Recommandé si …": (
                    "⚠️ Données < 1 an" if k == 2 else
                    "✅ 1–2 ans de données" if k == 3 else
                    "✅ 2–3 ans de données" if k == 4 else
                    "✅ 3+ ans de données" if k == 5 else
                    "⚠️ 4+ ans requis"
                )
            })
        st.dataframe(pd.DataFrame(fold_guide), hide_index=True, use_container_width=True)
        st.markdown(f"""
**Vos données** : `{n_weeks_total}` semaines distinctes × `{n_areas}` aires → `{len(df_model)}` observations après calcul des lags.

**Règle d'or** :
- `k = 3` → valeur par défaut, robuste pour la plupart des jeux de données (1–3 ans)
- `k = 4–5` → meilleure stabilité si vous avez 3 ans et plus
- `k = 2` → uniquement si moins d'un an de données
- Augmenter k réduit la variance de l'estimation mais diminue la taille de chaque fold test

**Seuil de détection des pics** : 50% est le défaut. Baissez-le (30%) pour capter les pics modérés, montez-le (70–80%) pour ne signaler que les pics extrêmes.
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        algo_choice = st.selectbox(
            "🤖 Algorithme",
            ["RandomForest", "GradientBoosting", "ExtraTrees"],
            index=0, key="val_algo"
        )
    with col2:
        n_splits = st.slider(
            "📊 Nombre de folds temporels",
            min_value=2, max_value=6, value=3,
            help="k=3 → recommandé pour 1–3 ans de données. k=4–5 → 3+ ans.",
            key="val_splits"
        )
    with col3:
        peak_threshold = st.slider(
            "📈 Seuil détection pics (%)",
            min_value=20, max_value=100, value=50,
            help="50% = pic si valeur > 1.5× la moyenne mobile locale. Baissez pour plus de sensibilité.",
            key="val_peak_threshold"
        )

    selected_area = st.selectbox(
        "🗺️ Aire de santé pour analyse détaillée",
        ["Toutes (agrégé)"] + sorted(df_model["health_area"].unique().tolist()),
        key="val_area"
    )

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        run_validation = st.button("▶️ Lancer la validation", type="primary", key="run_val")
    with col_btn2:
        reset_btn = st.button("🔄 Réinitialiser", key="reset_val",
                              help="Efface les résultats et permet de relancer avec d'autres paramètres")

    if reset_btn:
        st.session_state.pop("val_results", None)
        st.rerun()

    current_params = (algo_choice, n_splits, peak_threshold)
    stored = st.session_state.get("val_results")
    if stored is not None:
        stored_params = (stored["algo_choice"], stored["n_splits"], stored["peak_threshold"])
        if stored_params != current_params:
            st.session_state.pop("val_results", None)
            stored = None

    if run_validation and stored is None:
        with st.spinner("🔄 Validation en cours… veuillez patienter"):
            st.session_state["val_results"] = _run_validation_compute(
                df_model, available_features, algo_choice, n_splits, peak_threshold
            )
        stored = st.session_state["val_results"]

    if stored is None:
        st.markdown("""
        <div style="background:#f0f7ff;padding:1rem;border-radius:8px;border-left:4px solid #339af0;margin-top:1rem;">
        <b>ℹ️ Comment ça fonctionne ?</b><br><br>
        1. Le modèle est entraîné sur la <b>première partie</b> de vos données<br>
        2. Il prédit les cas des <b>semaines suivantes</b> qu'il n'a jamais vues<br>
        3. On compare ses prédictions aux <b>valeurs réellement observées</b><br>
        4. On évalue sa capacité à <b>détecter les pics épidémiques</b> à l'avance
        </div>
        """, unsafe_allow_html=True)
        return

    # ============================================================
    # RÉCUPÉRATION DES RÉSULTATS
    # ============================================================
    df_results       = stored["df_results"]
    df_metrics       = stored["df_metrics"]
    area_metrics     = stored["area_metrics"]
    imp_df           = stored["imp_df"]
    global_mae       = stored["global_mae"]
    global_rmse      = stored["global_rmse"]
    global_r2        = stored["global_r2"]
    global_bias      = stored["global_bias"]
    cv_r2_mean       = stored["cv_r2_mean"]
    ci_std           = stored["ci_std"]
    sens_mean        = stored["sens_mean"]
    n_peaks_total    = stored["n_peaks_total"]
    n_detected_total = stored["n_detected_total"]
    mean_cases       = stored.get("mean_cases", data_mean_cases)

    peak_str = (
        f"{n_detected_total}/{n_peaks_total} pics détectés ({sens_mean:.1f}% de sensibilité)"
        if sens_mean is not None else "Non calculé"
    )
    verdict_r2   = interpret_r2(cv_r2_mean)
    verdict_bias = interpret_bias(global_bias)

    best_area  = area_metrics.iloc[0]["health_area"]  if not area_metrics.empty else "—"
    best_mae   = area_metrics.iloc[0]["MAE"]           if not area_metrics.empty else "—"
    worst_area = area_metrics.iloc[-1]["health_area"]  if not area_metrics.empty else "—"
    worst_mae  = area_metrics.iloc[-1]["MAE"]          if not area_metrics.empty else "—"

    mae_pct  = (global_mae  / mean_cases * 100) if mean_cases else None
    rmse_pct = (global_rmse / mean_cases * 100) if mean_cases else None

    st.success(
        f"✅ Validation terminée — {stored['algo_choice']} | "
        f"{stored['n_splits']} folds | seuil pics {stored['peak_threshold']}%"
    )

    # ============================================================
    # MÉTRIQUES GLOBALES
    # ============================================================
    st.markdown("---")
    st.markdown("### 📊 Résultats globaux")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE globale",  f"{global_mae:.1f} cas",
                delta=f"{mae_pct:.0f}% de la moyenne" if mae_pct else None)
    col2.metric("RMSE globale", f"{global_rmse:.1f} cas",
                delta=f"{rmse_pct:.0f}% de la moyenne" if rmse_pct else None)
    col3.metric("R² global",    f"{global_r2:.3f}" if global_r2 and not np.isnan(global_r2) else "N/A",
                delta=interpret_r2(global_r2))
    col4.metric("R² CV moyen",  f"{cv_r2_mean:.3f}" if not np.isnan(cv_r2_mean) else "N/A",
                delta=interpret_r2(cv_r2_mean))
    col5.metric("Biais moyen",  f"{global_bias:+.1f} cas",
                delta=interpret_bias(global_bias))

    # ── Guide pédagogique des métriques ────────────────────────
    _show_metrics_guide()

    # ── Interprétation contextuelle MAE/RMSE ───────────────────
    st.markdown("#### 📖 Interprétation contextuelle des métriques")
    st.markdown(f"""
<div style="background:#f8f9fa;padding:1.2rem;border-radius:10px;border-left:5px solid #3182CE;font-size:0.93em;">

<b>📌 Contexte de vos données</b><br>
• Moyenne observée : <b>{mean_cases:.0f} cas/semaine</b> &nbsp;|&nbsp; Maximum observé : <b>{data_max_cases:.0f} cas</b> &nbsp;|&nbsp; {n_weeks_total} semaines × {n_areas} aires

<hr style="margin:10px 0;">

<b>MAE = {global_mae:.1f} cas</b> → {interpret_mae_contextual(global_mae, mean_cases)}<br><br>
<b>RMSE = {global_rmse:.1f} cas</b> → {interpret_rmse_contextual(global_rmse, mean_cases)}<br><br>

<b>🔑 Règle clé :</b> MAE et RMSE ne sont jamais "trop élevés" en valeur absolue ;
 ils doivent être lus en % de la moyenne de vos données.
 Un MAE de 500 est <i>excellent</i> si vos zones ont en moyenne 3000 cas/semaine,
 et <i>médiocre</i> si la moyenne est 200 cas/semaine.
 Le vrai juge est le <b>R² CV</b>.
</div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Tableau des seuils — absolu ET relatif"):
        st.markdown(f"""
| Métrique | Excellent | Bon | Moyen | Faible |
|---|---|---|---|---|
| **MAE** (% de la moyenne) | < 15% | 15–30% | 30–50% | > 50% |
| **RMSE** (% de la moyenne) | < 20% | 20–40% | 40–70% | > 70% |
| **MAE** absolu (contexte Sahel, moy. ~{mean_cases:.0f} cas) | < {mean_cases*0.15:.0f} | {mean_cases*0.15:.0f}–{mean_cases*0.30:.0f} | {mean_cases*0.30:.0f}–{mean_cases*0.50:.0f} | > {mean_cases*0.50:.0f} |
| **R²** | > 0.80 | 0.65–0.80 | 0.45–0.65 | < 0.45 |
| **R² CV** | > 0.70 | 0.55–0.70 | 0.40–0.55 | < 0.40 |
| **Biais** | ± 2 cas | ± 5% moy. | ± 10% moy. | > ± 10% moy. |

> ⚠️ **Le R² CV est la métrique principale** — elle mesure la vraie capacité de généralisation.
> Le MAE/RMSE élevés sont normaux dans le Sahel où les pics saisonniers peuvent atteindre 10× la moyenne.
> Un modèle avec R² CV = 0.70 et MAE = 400 cas (mais moyenne = 1500 cas) est **bon**.
        """)

    # ============================================================
    # TABLEAU PAR FOLD
    # ============================================================
    st.markdown("### 📋 Détail par fold temporel")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # ============================================================
    # GRAPHIQUE OBSERVÉ vs PRÉDIT
    # ============================================================
    st.markdown("### 📈 Observé vs Prédit")

    if selected_area == "Toutes (agrégé)":
        df_plot = df_results.groupby("week_num").agg(
            observed=("observed",  "sum"),
            predicted=("predicted", "sum"),
        ).reset_index()
        title_plot = "Tous les cas (somme toutes aires)"
    else:
        df_plot = df_results[df_results["health_area"] == selected_area].copy()
        df_plot = df_plot.groupby("week_num").agg(
            observed=("observed",  "sum"),
            predicted=("predicted", "sum"),
        ).reset_index()
        title_plot = f"Aire de santé : {selected_area}"

    df_plot = df_plot.sort_values("week_num")
    df_plot["is_peak"]    = detect_peaks(df_plot["observed"], threshold_pct=peak_threshold)
    df_plot["pred_upper"] = df_plot["predicted"] + ci_std
    df_plot["pred_lower"] = (df_plot["predicted"] - ci_std).clip(lower=0)

    fig_obs_pred = go.Figure()
    fig_obs_pred.add_trace(go.Scatter(
        x=df_plot["week_num"].tolist() + df_plot["week_num"].tolist()[::-1],
        y=df_plot["pred_upper"].tolist() + df_plot["pred_lower"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(99,179,237,0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"Intervalle ±1σ résidus ({ci_std:.1f} cas)",
        showlegend=True,
    ))
    fig_obs_pred.add_trace(go.Scatter(
        x=df_plot["week_num"], y=df_plot["predicted"],
        mode="lines", name="Prédit",
        line=dict(color="#3182CE", width=2, dash="dash"),
    ))
    fig_obs_pred.add_trace(go.Scatter(
        x=df_plot["week_num"], y=df_plot["observed"],
        mode="lines+markers", name="Observé",
        line=dict(color="#2F855A", width=2),
        marker=dict(size=5),
    ))
    df_peaks = df_plot[df_plot["is_peak"]]
    if not df_peaks.empty:
        fig_obs_pred.add_trace(go.Scatter(
            x=df_peaks["week_num"], y=df_peaks["observed"],
            mode="markers", name="Pic épidémique",
            marker=dict(color="#E53E3E", size=10, symbol="star"),
        ))
    fig_obs_pred.update_layout(
        title=title_plot,
        xaxis_title="Semaine épidémiologique",
        yaxis_title="Nombre de cas",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_obs_pred, use_container_width=True)
    st.caption(
        f"📐 Intervalle de confiance (zone bleue) = ±1σ des résidus réels de validation "
        f"({ci_std:.1f} cas) — calculé sur l'ensemble des folds, non fictif."
    )

    # ============================================================
    # GRAPHIQUE RÉSIDUS
    # ============================================================
    st.markdown("### 📉 Analyse des résidus")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        fig_resid = px.scatter(
            df_results, x="predicted", y="residual",
            color="fold",
            labels={"predicted": "Valeur prédite", "residual": "Résidu (prédit − observé)"},
            title="Résidus vs valeurs prédites",
            opacity=0.6,
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Biais nul")
        st.plotly_chart(fig_resid, use_container_width=True)

    with col_r2:
        fig_hist = px.histogram(
            df_results, x="residual",
            nbins=30, color="fold",
            labels={"residual": "Résidu (prédit − observé)"},
            title="Distribution des résidus",
            opacity=0.75,
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ============================================================
    # ANALYSE SPATIALE PAR AIRE
    # ============================================================
    st.markdown("### 🗺️ Performance par aire de santé")
    st.info(
        "💡 La colonne **MAE/Moy(%)** indique l'erreur **relative** à la moyenne des cas de chaque aire. "
        "C'est la colonne la plus pertinente pour comparer des aires de tailles différentes."
    )
    st.dataframe(area_metrics, use_container_width=True, hide_index=True)

    fig_area = px.bar(
        area_metrics.sort_values("MAE", ascending=False).head(20),
        x="health_area", y="MAE",
        color="R²",
        color_continuous_scale="RdYlGn",
        title="MAE par aire de santé (top 20 — les plus élevées en premier)",
        labels={"health_area": "Aire de santé", "MAE": "MAE (cas/semaine)"},
    )
    fig_area.update_xaxes(tickangle=45)
    st.plotly_chart(fig_area, use_container_width=True)

    # ============================================================
    # IMPORTANCE DES VARIABLES
    # ============================================================
    fig_importance = None
    if imp_df is not None:
        st.markdown("### 🔑 Importance des variables (moyenne sur tous les folds)")
        fig_importance = px.bar(
            imp_df.tail(15),
            x="Importance", y="Variable",
            orientation="h",
            title="Variables les plus influentes dans les prédictions",
            color="Importance",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        with st.expander("ℹ️ Interprétation des variables importantes"):
            st.markdown("""
- **cases_lag_1 / lag_2 / lag_4** : L'historique récent est la variable la plus prédictive.
- **sin_week / cos_week** : Capturent la saisonnalité cyclique.
- **cases_ma_2 / ma_4** : Lissage de la tendance récente.
- **temp_api / precip_api / humidity_api** : Variables climatiques.
- **flood_mean / dist_river** : Risque environnemental lié aux gîtes larvaires.
- **Pop_Totale / Pop_Enfants_0_14** : Taille de la population à risque.
            """)

    # ============================================================
    # SYNTHÈSE ET RECOMMANDATIONS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🧾 Synthèse et interprétation")

    st.markdown(f"""
<div style="background:#f8f9fa;padding:1.5rem;border-radius:10px;border-left:5px solid #3182CE;">
<h4>📌 Constats automatiques</h4>

- **Performance globale** : R² CV moyen = `{cv_r2_mean:.3f}` → {verdict_r2}
- **Précision absolue** : MAE = `{global_mae:.1f} cas` ({mae_pct:.1f}% de la moyenne de {mean_cases:.0f} cas) → {interpret_mae_contextual(global_mae, mean_cases).split('—')[0].strip()}
- **Précision sur pics** : RMSE = `{global_rmse:.1f} cas` ({rmse_pct:.1f}% de la moyenne) → {interpret_rmse_contextual(global_rmse, mean_cases).split('—')[0].strip()}
- **Biais pondéré** : {verdict_bias} (biais moyen pondéré = `{global_bias:+.1f} cas`)
- **Intervalle de confiance** : ±{ci_std:.1f} cas (±1σ résidus réels de validation)
- **Détection de pics** : {peak_str}
- **Aire la mieux prédite** : `{best_area}` (MAE = {best_mae})
- **Aire la moins bien prédite** : `{worst_area}` (MAE = {worst_mae})

<h4>⚠️ Limites à documenter</h4>

- Les lags temporels supposent une continuité des données
- Le modèle ne capte pas les chocs non-climatiques (déplacements, ruptures de stock)
- Validation interne — une validation sur une nouvelle année serait plus rigoureuse

</div>
    """, unsafe_allow_html=True)

    # ============================================================
    # EXPORT CSV
    # ============================================================
    st.markdown("### 📥 Télécharger les données brutes (CSV)")

    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            "📄 Données détaillées (CSV)",
            data=df_results.to_csv(index=False).encode("utf-8"),
            file_name="validation_details.csv",
            mime="text/csv",
        )
    with col_dl2:
        st.download_button(
            "📊 Métriques par fold (CSV)",
            data=df_metrics.to_csv(index=False).encode("utf-8"),
            file_name="validation_metriques_fold.csv",
            mime="text/csv",
        )
    with col_dl3:
        st.download_button(
            "🗺️ Performance par aire (CSV)",
            data=area_metrics.to_csv(index=False).encode("utf-8"),
            file_name="validation_par_aire.csv",
            mime="text/csv",
        )

    # ============================================================
    # EXPORT RAPPORT PDF / WORD / HTML — branding MSF
    # ============================================================
    metrics_rapport = {
        "MAE globale":        f"{global_mae:.1f} cas ({mae_pct:.1f}% de la moyenne)" if mae_pct else f"{global_mae:.1f} cas",
        "RMSE globale":       f"{global_rmse:.1f} cas ({rmse_pct:.1f}% de la moyenne)" if rmse_pct else f"{global_rmse:.1f} cas",
        "Moyenne observée":   f"{mean_cases:.0f} cas/semaine",
        "Maximum observé":    f"{data_max_cases:.0f} cas",
        "R² global":          f"{global_r2:.3f}" if global_r2 and not np.isnan(global_r2) else "N/A",
        "R² CV moyen":        f"{cv_r2_mean:.3f} → {verdict_r2}" if not np.isnan(cv_r2_mean) else "N/A",
        "Biais pondéré":      f"{global_bias:+.1f} cas → {verdict_bias}",
        "IC ±1σ résidus":     f"±{ci_std:.1f} cas",
        "Détection pics":     peak_str,
        "Algorithme":         algo_choice,
        "Folds":              str(n_splits),
    }

    tables_data_rapport = [
        (df_metrics.columns.tolist(),  df_metrics.fillna("-").astype(str).values.tolist()),
        (area_metrics.columns.tolist(), area_metrics.fillna("-").astype(str).values.tolist()),
    ]

    interpretations_rapport = build_interpretations_list(
        global_mae, global_rmse, cv_r2_mean, global_bias,
        ci_std, sens_mean, peak_str, best_area, best_mae,
        worst_area, worst_mae, mean_cases
    )

    figures_rapport = [fig_obs_pred, fig_resid, fig_hist]
    if fig_importance is not None:
        figures_rapport.append(fig_importance)

    try:
        from report_generator import rapport_streamlit_widget
        rapport_streamlit_widget(
            title="Rapport de Validation Rétrospective — Paludisme",
            subtitle=f"Algorithme : {algo_choice} | {n_splits} folds temporels | Aire : {selected_area}",
            metrics=metrics_rapport,
            figures=figures_rapport,
            tables_data=tables_data_rapport,
            interpretations=interpretations_rapport,
            maladie="Paludisme",
            key_prefix="val_palu",
        )
    except ImportError:
        st.error("❌ Module report_generator.py introuvable dans le répertoire de l'application.")
    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport : {e}")

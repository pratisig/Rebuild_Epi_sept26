# ============================================================
# MODULE VALIDATION RÉTROSPECTIVE DU MODÈLE PRÉDICTIF — ROUGEOLE
# Intégration dans app_rougeole.py
# Usage : from validation_tab_rougeole import create_validation_tab_rougeole
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Branding MSF Rougeole
MSF_RED   = "#E4032E"
MSF_DARK  = "#C4032A"
MEAS_BLUE = "#1565C0"
MEAS_WARN = "#E65100"

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
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    return series > rolling_mean * (1 + threshold_pct / 100)


def compute_peak_detection_metrics(y_true, y_pred, weeks, window=3, threshold_pct=50):
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
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else np.nan

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
    return               "⬇️ Sous-estimation"


def interpret_mae_contextual(mae, mean_cases):
    if mean_cases is None or mean_cases <= 0:
        return "❓ Moyenne des cas non disponible pour contextualiser"
    mae_pct = (mae / mean_cases) * 100
    rel_str = f"{mae_pct:.1f}% de la moyenne observée ({mean_cases:.0f} cas)"
    if mae_pct <= 15:
        return f"🟢 Excellent — erreur de {mae:.0f} cas soit {rel_str}. Le modèle est très précis."
    if mae_pct <= 30:
        return f"🟡 Bon — erreur de {mae:.0f} cas soit {rel_str}. Précision acceptable pour la planification."
    if mae_pct <= 50:
        return f"🟠 Moyen — erreur de {mae:.0f} cas soit {rel_str}. Envisagez l'ajout de données de couverture vaccinale."
    return f"🔴 Élevé — erreur de {mae:.0f} cas soit {rel_str}. Normal si flambées à 500–2000 cas. Vérifiez le R² CV et la sensibilité."


def interpret_rmse_contextual(rmse, mean_cases):
    if mean_cases is None or mean_cases <= 0:
        return "❓ Moyenne des cas non disponible"
    rmse_pct = (rmse / mean_cases) * 100
    if rmse_pct <= 20:
        return f"🟢 Excellent — RMSE = {rmse:.0f} cas ({rmse_pct:.1f}% de la moyenne). Bonne gestion des flambées."
    if rmse_pct <= 40:
        return f"🟡 Bon — RMSE = {rmse:.0f} cas ({rmse_pct:.1f}% de la moyenne). Quelques flambées génèrent des erreurs ponctuelles."
    if rmse_pct <= 70:
        return f"🟠 Moyen — RMSE = {rmse:.0f} cas ({rmse_pct:.1f}% de la moyenne). Les flambées rougeole sont difficiles à prévoir précisément — comportement attendu."
    return f"🔴 RMSE élevé = {rmse:.0f} cas ({rmse_pct:.1f}% de la moyenne). Normal si données très hétérogènes ou flambées extrêmes. Comparez au R² CV."


def build_interpretations_list(global_mae, global_rmse, cv_r2_mean, global_bias,
                               ci_std, sens_mean, peak_str, best_area, best_mae,
                               worst_area, worst_mae, mean_cases):
    mae_interp   = interpret_mae_contextual(global_mae, mean_cases)
    rmse_interp  = interpret_rmse_contextual(global_rmse, mean_cases)
    verdict_r2   = interpret_r2(cv_r2_mean)
    verdict_bias = interpret_bias(global_bias)
    lines = [
        f"R² CV moyen = {cv_r2_mean:.3f} → {verdict_r2} (métrique principale de généralisation)",
        f"MAE (erreur absolue moyenne) : {mae_interp}",
        f"RMSE (erreur quadratique) : {rmse_interp}",
        f"Biais pondéré : {verdict_bias} ({global_bias:+.1f} cas/semaine en moyenne)",
        f"Intervalle de confiance : ±{ci_std:.1f} cas (±1σ résidus réels — non fictif)",
        f"Détection de flambées : {peak_str}",
        f"Aire la mieux prédite : {best_area} (MAE = {best_mae})",
        f"Aire la moins bien prédite : {worst_area} (MAE = {worst_mae})",
        "Note : un MAE élevé en valeur absolue est NORMAL lors de flambées rougeole intenses.",
        "Limites : lags supposent continuité ; ruptures de chaîne du froid non captées ; validation interne.",
    ]
    return lines


# ============================================================
# GUIDE PÉDAGOGIQUE DES MÉTRIQUES — Rougeole
# ============================================================

def _show_metrics_guide_rougeole():
    with st.expander("📚 Comprendre les métriques — Guide pour non-techniciens", expanded=False):
        st.markdown("""
### 🎯 À quoi servent ces métriques ?

Après avoir entraîné le modèle, on lui présente des données qu'il n'a **jamais vues**.
On compare ce qu'il prédit à ce qui s'est réellement passé.
Les métriques ci-dessous mesurent **à quel point il se trompe** — et dans quel sens.

---

### 📏 MAE — Erreur Absolue Moyenne
> *"En moyenne, de combien de cas le modèle se trompe-t-il ?"*

**Seuils relatifs (MAE / moyenne des cas) :**
| < 15% | 15–30% | 30–50% | > 50% |
|--------|--------|--------|-------|
| 🟢 Excellent | 🟡 Bon | 🟠 Moyen | 🔴 Élevé |

---

### 📐 RMSE — Racine de l'Erreur Quadratique Moyenne
> *"Le modèle gère-t-il bien les semaines de flambée rougeole ?"*

| < 20% | 20–40% | 40–70% | > 70% |
|--------|--------|--------|-------|
| 🟢 Excellent | 🟡 Bon | 🟠 Moyen | 🔴 Élevé |

---

### 📈 R² CV — R² de Validation Croisée *(métrique principale)*
> *"Le modèle sera-t-il encore fiable sur de nouvelles données ?"*

| R² CV > 0.70 | 0.55–0.70 | 0.40–0.55 | < 0.40 |
|--------------|-----------|-----------|--------|
| 🟢 Fiable | 🟡 Utile avec prudence | 🟠 Indicatif | 🔴 Non fiable |

---

### 💉 Couverture Vaccinale

Si `coverage_vac`, `coverage_mcv1` ou `sia_coverage` est présent, elle est intégrée comme variable explicative.
Une forte importance = zones à faible couverture = zones à risque de flambée.
        """)


# ============================================================
# CALCUL DE LA VALIDATION
# ============================================================

def _run_validation_compute_rougeole(df_model, available_features, algo_choice, n_splits, peak_threshold):
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

    global_mae  = mean_absolute_error(all_true, all_pred)
    global_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    global_r2   = safe_r2(all_true, all_pred)
    mean_cases  = float(np.mean(all_true)) if all_true else None

    fold_sizes  = [m["N test"] for m in fold_metrics]
    fold_biases = [m["Biais"]  for m in fold_metrics]
    total_obs   = sum(fold_sizes)
    global_bias = float(
        sum(b * n for b, n in zip(fold_biases, fold_sizes)) / total_obs
    ) if total_obs > 0 else float(np.mean(np.array(all_pred) - np.array(all_true)))

    cv_r2_mean    = df_metrics["R²"].dropna().mean()
    residuals_all = np.array(all_pred) - np.array(all_true)
    ci_std        = float(np.std(residuals_all))

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
        }),
        include_groups=False
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

def create_validation_tab_rougeole(df_cases, gdf_health=None, model_results=None):
    """
    Onglet complet de validation rétrospective — Rougeole.

    Args:
        df_cases      : DataFrame des cas rougeole
                        Colonnes attendues : week_num (ou Semaine_Epi…), health_area, cases (ou CasObserves…)
                        Colonnes optionnelles : coverage_vac, doses_administered,
                                                year, deaths, pop_total, pop_enfants
        gdf_health    : GeoDataFrame optionnel
        model_results : Résultats du modèle principal (optionnel)
    """

    # ── En-tête branding MSF Rougeole ──────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{MSF_RED},{MSF_DARK});
                color:white;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
      <h2 style="margin:0;">🔬 Validation Rétrospective — Rougeole</h2>
      <p style="margin:0.3rem 0 0 0;opacity:0.9;font-size:0.95rem;">
        Évaluation de la performance prédictive du modèle sur des données historiques rougeole
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "📋 **Principe** : Le modèle est entraîné sur des données passées (rougeole), puis testé sur des "
        "périodes qu'il n'a jamais vues. Pour la rougeole, on évalue aussi la couverture vaccinale "
        "comme facteur explicatif clé."
    )

    if df_cases is None or df_cases.empty:
        st.warning("⚠️ Aucune donnée épidémiologique rougeole chargée.")
        return

    df = df_cases.copy()

    # ── Normalisation colonne semaine ───────────────────────────
    WEEK_ALIASES = [
        "week_num", "week_", "Semaine_Epi", "semaine_epi", "semaine",
        "Semaine", "SEMAINE", "week", "Week", "WEEK", "epi_week",
        "SE", "se", "s_epi", "S_epi", "sem_epi", "no_semaine",
        "numero_semaine", "week_number", "wk", "WK"
    ]
    if "week_num" not in df.columns:
        for alias in WEEK_ALIASES:
            if alias in df.columns:
                df["week_num"] = pd.to_numeric(df[alias], errors="coerce")
                break
    if "week_num" not in df.columns:
        for col in df.columns:
            if any(kw in col.lower() for kw in ["week", "semai", "se_", "_se", "epi"]):
                df["week_num"] = pd.to_numeric(df[col], errors="coerce")
                break
    if "week_num" not in df.columns:
        st.error(
            f"❌ Colonne de semaine introuvable. "
            f"Colonnes détectées : `{', '.join(df.columns.tolist())}`\n\n"
            "Renommez votre colonne de semaine en `week_` ou `Semaine_Epi`."
        )
        return

    # ── Normalisation colonne 'cases' ───────────────────────────
    # Liste exhaustive incluant les noms réels du projet surveillanceEpi
    CASES_ALIASES = [
        "cases", "Cases", "CASES",
        # ← NOM RÉEL dans les données du projet
        "CasObserves", "casobserves", "CASOBSERVES",
        "Cas_Observes", "cas_observes", "CAS_OBSERVES",
        "CasObservés", "Cas_Observés",
        # Autres variantes courantes
        "Cas_Total", "cas_total", "CAS_TOTAL",
        "Cas", "cas", "CAS",
        "nb_cas", "Nb_Cas", "NB_CAS",
        "nombre_cas", "Nombre_Cas", "NOMBRE_CAS",
        "nbcas", "nbre_cas",
        "count", "Count", "total_cas", "Total_Cas",
        "confirmed", "Confirmed",
        "cas_confirmes", "Cas_Confirmes",
    ]
    if "cases" not in df.columns:
        for alias in CASES_ALIASES:
            if alias in df.columns:
                df["cases"] = pd.to_numeric(df[alias], errors="coerce").fillna(0)
                break
    # Fallback générique : toute colonne contenant "cas" ou "case"
    if "cases" not in df.columns:
        for col in df.columns:
            if any(kw in col.lower() for kw in ["cas", "case", "count", "confirm"]):
                df["cases"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                break
    if "cases" not in df.columns:
        st.error(
            f"❌ Colonne 'cases' introuvable.\n\n"
            f"Colonnes détectées : `{', '.join(df.columns.tolist())}`\n\n"
            "Renommez votre colonne de cas en `CasObserves`, `Cas_Total`, `cas` ou `cases`."
        )
        return

    # ── Normalisation colonne 'health_area' ────────────────────
    AREA_ALIASES = [
        "health_area", "HEALTH_AREA",
        "Aire_Sante", "aire_sante", "Aire_de_Sante",
        "name_fr", "NAME", "nom", "NOM",
        "aire", "zone_sante", "district", "localite",
        "nom_aire", "fosa", "nom_fosa",
    ]
    if "health_area" not in df.columns:
        for alias in AREA_ALIASES:
            if alias in df.columns:
                df["health_area"] = df[alias].astype(str).str.strip()
                break
    if "health_area" not in df.columns:
        st.error(
            f"❌ Colonne 'health_area' introuvable.\n\n"
            f"Colonnes détectées : `{', '.join(df.columns.tolist())}`\n\n"
            "Renommez votre colonne d'aire en `health_area` ou `Aire_Sante`."
        )
        return

    df["cases"]    = pd.to_numeric(df["cases"],    errors="coerce").fillna(0)
    df["week_num"] = pd.to_numeric(df["week_num"], errors="coerce")
    df = df.dropna(subset=["week_num"]).copy()
    df["week_num"] = df["week_num"].astype(int)

    # ── Features temporelles et lags ───────────────────────────
    df = df.sort_values(["health_area", "week_num"])
    for lag in [1, 2, 4, 8]:
        df[f"cases_lag_{lag}"] = df.groupby("health_area")["cases"].shift(lag)
    for win in [2, 4]:
        df[f"cases_ma_{win}"] = df.groupby("health_area")["cases"].transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
    df["sin_week"]    = np.sin(2 * np.pi * df["week_num"] / 52)
    df["cos_week"]    = np.cos(2 * np.pi * df["week_num"] / 52)
    df["growth_rate"] = df.groupby("health_area")["cases"].pct_change().fillna(0).clip(-5, 5)

    vac_features     = [c for c in df.columns if c in
                        ["coverage_vac", "doses_administered", "vaccination_rate",
                         "coverage_mcv1", "coverage_mcv2", "sia_coverage"]]
    climate_features = [c for c in df.columns if c.endswith("_api")]
    demo_features    = [c for c in df.columns if c in
                        ["Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop",
                         "incidence_rate", "child_risk", "demo_pressure"]]

    base_features = ["cases_lag_1", "cases_lag_2", "cases_lag_4", "cases_lag_8",
                     "cases_ma_2", "cases_ma_4", "sin_week", "cos_week", "growth_rate"]
    all_features       = base_features + vac_features + climate_features + demo_features
    available_features = [f for f in all_features if f in df.columns]

    df_model = df.dropna(subset=["cases_lag_1", "cases_lag_2"]).copy()

    if len(df_model) < 20:
        st.error("❌ Données insuffisantes pour la validation (minimum 20 observations après calcul des lags).")
        return

    data_mean_cases = float(df_model["cases"].mean())
    data_max_cases  = float(df_model["cases"].max())
    n_weeks_total   = df_model["week_num"].nunique()
    n_areas         = df_model["health_area"].nunique()

    # ============================================================
    # PANNEAU DE CONFIGURATION
    # ============================================================
    st.markdown("### ⚙️ Configuration de la validation")

    with st.expander("❓ Quel nombre de folds choisir ?", expanded=False):
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
**Vos données** : `{n_weeks_total}` semaines × `{n_areas}` aires → `{len(df_model)}` observations.

**Règle d'or rougeole** :
- `k = 3` → valeur par défaut (1–3 ans de données)
- `k = 4–5` → si vous avez 3 ans et plus
- `k = 2` → uniquement si moins d'un an de données
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        algo_choice = st.selectbox(
            "🤖 Algorithme",
            ["RandomForest", "GradientBoosting", "ExtraTrees"],
            index=0, key="val_rougeole_algo"
        )
    with col2:
        n_splits = st.slider(
            "📊 Nombre de folds temporels",
            min_value=2, max_value=6, value=3,
            help="k=3 → recommandé pour 1–3 ans de données.",
            key="val_rougeole_splits"
        )
    with col3:
        peak_threshold = st.slider(
            "📈 Seuil détection flambées (%)",
            min_value=20, max_value=100, value=50,
            help="50% = flambée si valeur > 1.5× la moyenne mobile locale.",
            key="val_rougeole_peak_threshold"
        )

    selected_area = st.selectbox(
        "🗺️ Aire de santé pour analyse détaillée",
        ["Toutes (agrégé)"] + sorted(df_model["health_area"].unique().tolist()),
        key="val_rougeole_area"
    )

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        run_validation = st.button("▶️ Lancer la validation rougeole", type="primary", key="run_val_rougeole")
    with col_btn2:
        reset_btn = st.button("🔄 Réinitialiser", key="reset_val_rougeole",
                              help="Efface les résultats et permet de relancer avec d'autres paramètres")

    if reset_btn:
        st.session_state.pop("val_results_rougeole", None)
        st.rerun()

    current_params = (algo_choice, n_splits, peak_threshold)
    stored = st.session_state.get("val_results_rougeole")
    if stored is not None:
        stored_params = (stored["algo_choice"], stored["n_splits"], stored["peak_threshold"])
        if stored_params != current_params:
            st.session_state.pop("val_results_rougeole", None)
            stored = None

    if run_validation and stored is None:
        with st.spinner("🔄 Validation en cours… veuillez patienter"):
            st.session_state["val_results_rougeole"] = _run_validation_compute_rougeole(
                df_model, available_features, algo_choice, n_splits, peak_threshold
            )
        stored = st.session_state["val_results_rougeole"]

    if stored is None:
        st.markdown(f"""
        <div style="background:#fff5f5;padding:1rem;border-radius:8px;
                    border-left:4px solid {MSF_RED};margin-top:1rem;">
        <b>ℹ️ Comment ça fonctionne ?</b><br><br>
        1. Le modèle est entraîné sur la <b>première partie</b> de vos données rougeole<br>
        2. Il prédit les cas des <b>semaines suivantes</b> qu'il n'a jamais vues<br>
        3. On compare ses prédictions aux <b>valeurs réellement observées</b><br>
        4. On évalue sa capacité à <b>détecter les flambées rougeole</b> à l'avance<br>
        5. La <b>couverture vaccinale</b> est incluse comme variable explicative si disponible
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
        f"{n_detected_total}/{n_peaks_total} flambées détectées ({sens_mean:.1f}% de sensibilité)"
        if sens_mean is not None else "Non calculé"
    )
    verdict_r2   = interpret_r2(cv_r2_mean)
    verdict_bias = interpret_bias(global_bias)
    best_area    = area_metrics.iloc[0]["health_area"]  if not area_metrics.empty else "—"
    best_mae     = area_metrics.iloc[0]["MAE"]           if not area_metrics.empty else "—"
    worst_area   = area_metrics.iloc[-1]["health_area"]  if not area_metrics.empty else "—"
    worst_mae    = area_metrics.iloc[-1]["MAE"]          if not area_metrics.empty else "—"
    mae_pct      = (global_mae  / mean_cases * 100) if mean_cases else None
    rmse_pct     = (global_rmse / mean_cases * 100) if mean_cases else None

    st.success(
        f"✅ Validation terminée — {stored['algo_choice']} | "
        f"{stored['n_splits']} folds | seuil flambées {stored['peak_threshold']}%"
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

    _show_metrics_guide_rougeole()

    st.markdown("#### 📖 Interprétation contextuelle des métriques")
    st.markdown(f"""
<div style="background:#fff5f5;padding:1.2rem;border-radius:10px;border-left:5px solid {MSF_RED};font-size:0.93em;">

<b>📌 Contexte de vos données rougeole</b><br>
• Moyenne observée : <b>{mean_cases:.0f} cas/semaine</b> &nbsp;|&nbsp; Maximum observé : <b>{data_max_cases:.0f} cas</b> &nbsp;|&nbsp; {n_weeks_total} semaines × {n_areas} aires
{'&nbsp;|&nbsp; 💉 Couverture vaccinale : <b>Disponible</b>' if any(c in df_model.columns for c in ["coverage_vac", "coverage_mcv1", "sia_coverage"]) else ''}

<hr style="margin:10px 0;">

<b>MAE = {global_mae:.1f} cas</b> → {interpret_mae_contextual(global_mae, mean_cases)}<br><br>
<b>RMSE = {global_rmse:.1f} cas</b> → {interpret_rmse_contextual(global_rmse, mean_cases)}<br><br>

<b>🔑 Règle clé :</b> MAE et RMSE ne sont jamais "trop élevés" en valeur absolue ;
ils doivent être lus en % de la moyenne de vos données.
Le vrai juge est le <b>R² CV</b>.
</div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Tableau des seuils — absolu ET relatif"):
        st.markdown(f"""
| Métrique | Excellent | Bon | Moyen | Faible |
|---|---|---|---|---|
| **MAE** (% de la moyenne) | < 15% | 15–30% | 30–50% | > 50% |
| **RMSE** (% de la moyenne) | < 20% | 20–40% | 40–70% | > 70% |
| **MAE** absolu (moy. ~{mean_cases:.0f} cas) | < {mean_cases*0.15:.0f} | {mean_cases*0.15:.0f}–{mean_cases*0.30:.0f} | {mean_cases*0.30:.0f}–{mean_cases*0.50:.0f} | > {mean_cases*0.50:.0f} |
| **R² CV** | > 0.70 | 0.55–0.70 | 0.40–0.55 | < 0.40 |
| **Biais** | ± 2 cas | ± 5% moy. | ± 10% moy. | > ± 10% moy. |
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
        title_plot = "Tous les cas rougeole (somme toutes aires)"
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
        fillcolor="rgba(228,3,46,0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"Intervalle ±1σ résidus ({ci_std:.1f} cas)",
        showlegend=True,
    ))
    fig_obs_pred.add_trace(go.Scatter(
        x=df_plot["week_num"], y=df_plot["predicted"],
        mode="lines", name="Prédit",
        line=dict(color=MSF_RED, width=2, dash="dash"),
    ))
    fig_obs_pred.add_trace(go.Scatter(
        x=df_plot["week_num"], y=df_plot["observed"],
        mode="lines+markers", name="Observé",
        line=dict(color=MEAS_BLUE, width=2),
        marker=dict(size=5),
    ))
    df_peaks = df_plot[df_plot["is_peak"]]
    if not df_peaks.empty:
        fig_obs_pred.add_trace(go.Scatter(
            x=df_peaks["week_num"], y=df_peaks["observed"],
            mode="markers", name="Flambée rougeole",
            marker=dict(color=MEAS_WARN, size=12, symbol="star"),
        ))
    fig_obs_pred.update_layout(
        title=title_plot,
        xaxis_title="Semaine épidémiologique",
        yaxis_title="Nombre de cas rougeole",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_obs_pred, use_container_width=True)
    st.caption(
        f"📐 Intervalle de confiance (zone rouge) = ±1σ des résidus réels de validation "
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
        fig_resid.add_hline(y=0, line_dash="dash", line_color=MSF_RED, annotation_text="Biais nul")
        st.plotly_chart(fig_resid, use_container_width=True)

    with col_r2:
        fig_hist = px.histogram(
            df_results, x="residual",
            nbins=30, color="fold",
            labels={"residual": "Résidu (prédit − observé)"},
            title="Distribution des résidus",
            opacity=0.75,
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color=MSF_RED)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ============================================================
    # ANALYSE SPATIALE PAR AIRE
    # ============================================================
    st.markdown("### 🗺️ Performance par aire de santé")
    st.info(
        "💡 La colonne **MAE/Moy(%)** indique l'erreur **relative** à la moyenne des cas de chaque aire."
    )
    st.dataframe(area_metrics, use_container_width=True, hide_index=True)

    fig_area = px.bar(
        area_metrics.sort_values("MAE", ascending=False).head(20),
        x="health_area", y="MAE",
        color="R²",
        color_continuous_scale="RdYlGn",
        title="MAE par aire de santé — rougeole (top 20)",
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
            title="Variables les plus influentes — rougeole",
            color="Importance",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        with st.expander("ℹ️ Interprétation des variables importantes — Rougeole"):
            st.markdown("""
- **cases_lag_1 / lag_2 / lag_4 / lag_8** : Historique récent — le lag 8 est spécifique rougeole.
- **sin_week / cos_week** : Saisonnalité cyclique (saison sèche).
- **coverage_vac / coverage_mcv1 / sia_coverage** : Couverture vaccinale — variable clé.
- **cases_ma_2 / ma_4** : Lissage de la tendance récente.
- **Pop_Enfants_0_14** : Population la plus à risque.
            """)

    # ============================================================
    # SYNTHÈSE ET RECOMMANDATIONS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🧾 Synthèse et interprétation")

    st.markdown(f"""
<div style="background:#fff5f5;padding:1.5rem;border-radius:10px;border-left:5px solid {MSF_RED};">
<h4>📌 Constats automatiques — Rougeole</h4>

- **Performance globale** : R² CV moyen = `{cv_r2_mean:.3f}` → {verdict_r2}
- **Précision absolue** : MAE = `{global_mae:.1f} cas` ({mae_pct:.1f}% de la moyenne de {mean_cases:.0f} cas)
- **Précision sur flambées** : RMSE = `{global_rmse:.1f} cas` ({rmse_pct:.1f}% de la moyenne)
- **Biais pondéré** : {verdict_bias} (biais moyen pondéré = `{global_bias:+.1f} cas`)
- **Intervalle de confiance** : ±{ci_std:.1f} cas (±1σ résidus réels de validation)
- **Détection de flambées** : {peak_str}
- **Aire la mieux prédite** : `{best_area}` (MAE = {best_mae})
- **Aire la moins bien prédite** : `{worst_area}` (MAE = {worst_mae})

<h4>⚠️ Limites à documenter</h4>

- Les lags temporels supposent une continuité des données
- Le modèle ne capte pas les chocs non-vaccinaux (mouvements de population, rupture de chaîne du froid)
- La couverture vaccinale doit être disponible pour de meilleures performances
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
            file_name="validation_rougeole_details.csv",
            mime="text/csv",
        )
    with col_dl2:
        st.download_button(
            "📊 Métriques par fold (CSV)",
            data=df_metrics.to_csv(index=False).encode("utf-8"),
            file_name="validation_rougeole_metriques_fold.csv",
            mime="text/csv",
        )
    with col_dl3:
        st.download_button(
            "🗺️ Performance par aire (CSV)",
            data=area_metrics.to_csv(index=False).encode("utf-8"),
            file_name="validation_rougeole_par_aire.csv",
            mime="text/csv",
        )

    # ============================================================
    # EXPORT RAPPORT
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
        "Détection flambées": peak_str,
        "Algorithme":         algo_choice,
        "Folds":              str(n_splits),
        "Couverture vac.":    "Disponible" if any(c in df_model.columns for c in ["coverage_vac", "coverage_mcv1", "sia_coverage"]) else "Non chargée",
    }

    tables_data_rapport = [
        (df_metrics.columns.tolist(),   df_metrics.fillna("-").astype(str).values.tolist()),
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
            title="Rapport de Validation Rétrospective — Rougeole",
            subtitle=f"Algorithme : {algo_choice} | {n_splits} folds temporels | Aire : {selected_area}",
            metrics=metrics_rapport,
            figures=figures_rapport,
            tables_data=tables_data_rapport,
            interpretations=interpretations_rapport,
            maladie="Rougeole",
            key_prefix="val_rougeole",
        )
    except ImportError:
        st.error("❌ Module report_generator.py introuvable dans le répertoire de l'application.")
    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport rougeole : {e}")

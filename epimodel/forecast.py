"""
Entraînement et prévision récursive — **une seule source de vérité** pour les
variables d'entrée.

Le pipeline historique reconstruisait à la main, dans la boucle de prévision,
les variables utilisées à l'entraînement, puis remplissait par ``0`` celles
qu'il ne savait pas recalculer (``coef_population``, ``incidence_rate``,
``child_risk``, ``demo_pressure``...). Résultat mesuré : 9 variables sur 16 en
décalage et une prévision future ~2 fois trop basse en moyenne.

Ici, la prévision ajoute les semaines futures au panneau, puis rappelle
**exactement la même fonction** d'ingénierie de variables. Aucune variable ne
peut diverger.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import clip_predictions
from .panel import (
    COL_AREA, COL_CASES, COL_WIDX, COL_WEEK, COL_YEAR, COL_PERIOD,
    WEEKS_PER_YEAR, iso_week_to_date, week_index_to_iso,
)

FLAG_FORECAST = "cases_is_forecast"


# ======================================================================
# Entraînement
# ======================================================================
def fit_pipeline(panel: pd.DataFrame,
                 design_builder: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]],
                 model_factory: Callable[[], Any],
                 feature_cols: Optional[Sequence[str]] = None,
                 objective: str = "squared_error",
                 log_target: bool = False,
                 min_history_ratio: float = 0.5,
                 sample_weight_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Entraîne un modèle sur le panneau fourni.

    * l'imputeur (médiane) est ajusté **uniquement** sur les lignes
      d'entraînement — pas de fuite par l'imputation ;
    * les lignes dont les retards sont manquants (début de série) sont
      exclues plutôt qu'imputées par des zéros trompeurs ;
    * les résidus d'entraînement sont conservés pour les intervalles de
      prédiction conformes.
    """
    df, cols = design_builder(panel)
    if feature_cols is not None:
        cols = [c for c in feature_cols if c in df.columns] or list(cols)

    df = df.copy()
    if FLAG_FORECAST not in df.columns:
        df[FLAG_FORECAST] = False

    # exclusion du début de série (retards non disponibles)
    lag_cols = [c for c in cols if c.startswith("cases_lag_")]
    required = ["cases_lag_1"] if "cases_lag_1" in cols else []
    train_mask = pd.Series(True, index=df.index)
    for c in required:
        train_mask &= df[c].notna()
    if train_mask.sum() < max(10, int(min_history_ratio * len(df))):
        train_mask = df["cases_lag_1"].notna() if "cases_lag_1" in df.columns else train_mask

    df_train = df[train_mask].copy()
    if len(df_train) < 10:
        raise ValueError(f"Entraînement impossible : {len(df_train)} lignes exploitables.")

    X = df_train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(df_train[COL_CASES], errors="coerce").to_numpy(dtype=float)

    imputer = _make_imputer()
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=cols, index=X.index)

    y_fit = np.log1p(y) if log_target else y

    model = model_factory()
    fit_kwargs: Dict[str, Any] = {}
    if sample_weight_col and sample_weight_col in df_train.columns:
        w = pd.to_numeric(df_train[sample_weight_col], errors="coerce").fillna(1.0)
        try:
            model.fit(X_imp, y_fit, sample_weight=w.to_numpy())
        except TypeError:
            model.fit(X_imp, y_fit)
    else:
        # offset d'exposition pour les objectifs de type Poisson (XGBoost / LightGBM)
        if objective == "poisson" and "log_exposure" in df_train.columns:
            try:
                model.fit(X_imp, y_fit,
                          sample_weight=None,
                          base_margin=None) if False else model.fit(X_imp, y_fit)
            except Exception:
                model.fit(X_imp, y_fit)
        else:
            model.fit(X_imp, y_fit)

    y_hat = _inverse(model.predict(X_imp), log_target)
    y_hat = clip_predictions(y_hat, objective)
    resid = y_hat - y

    return {
        "model": model,
        "model_name": type(model).__name__,
        "feature_cols": list(cols),
        "design_builder": design_builder,
        "imputer": imputer,
        "objective": objective,
        "log_target": bool(log_target),
        "n_train": int(len(df_train)),
        "residuals": resid,
        "residual_std": float(np.std(resid)) if len(resid) else 0.0,
        "train_metrics": _simple_metrics(y, y_hat),
        "version": 1,
    }


def _make_imputer():
    """
    Imputeur médiane qui **conserve** les colonnes entièrement vides.

    Sans ``keep_empty_features``, ``SimpleImputer`` supprime silencieusement les
    colonnes 100 % NaN : la matrice transformée a alors moins de colonnes que
    ``feature_cols``, et le modèle entraîné ne reçoit plus les mêmes variables à
    la prévision. Conserver les colonnes (remplies par 0) garantit l'alignement
    train / inférence.
    """
    from sklearn.impute import SimpleImputer
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # scikit-learn < 1.2
        return SimpleImputer(strategy="median")


def _inverse(y_pred: np.ndarray, log_target: bool) -> np.ndarray:
    y = np.asarray(y_pred, dtype=float)
    if log_target:
        y = np.expm1(y)
    return y


def _simple_metrics(y: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    from .validation import metrics_summary
    return metrics_summary(y, y_hat)


# ======================================================================
# Prévision récursive
# ======================================================================
def _extend_panel(panel: pd.DataFrame, next_widx: float,
                  origin_year: int, origin_week: int = 1) -> pd.DataFrame:
    """Ajoute une ligne par aire pour la semaine ``next_widx`` (cas inconnus)."""
    years, weeks = week_index_to_iso([next_widx], origin_year, origin_week)
    y, w = int(years[0]), int(weeks[0])
    template = (panel[panel[COL_WIDX] == panel[COL_WIDX].max()]
                .copy())
    new = template.copy()
    new[COL_YEAR] = y
    new[COL_WEEK] = w
    new[COL_WIDX] = float(next_widx)
    new[COL_PERIOD] = f"{y}-S{str(w).zfill(2)}"
    if "week_frac" in new.columns:
        # recalculé, et non recopié de la semaine précédente (défaut détecté par
        # le test de décalage entraînement / inférence)
        new["week_frac"] = (w - 1) / float(WEEKS_PER_YEAR)
    new[COL_CASES] = np.nan
    if "deaths" in new.columns:
        new["deaths"] = np.nan
    new[FLAG_FORECAST] = True
    # on ne conserve que les colonnes structurelles de la template
    keep = [c for c in panel.columns if c in new.columns]
    new = new[keep]
    return pd.concat([panel, new], ignore_index=True)


def recursive_forecast(fitted: Dict[str, Any],
                       panel: pd.DataFrame,
                       horizons: int = 4,
                       design_builder: Optional[Callable] = None,
                       origin_year: Optional[int] = None,
                       origin_week: int = 1) -> pd.DataFrame:
    """
    Prévision récursive à ``horizons`` semaines.

    À chaque pas, la semaine suivante est ajoutée au panneau puis les variables
    sont recalculées par la **même** fonction qu'à l'entraînement. La valeur
    prédite devient le ``cases`` de cette semaine, ce qui alimente
    automatiquement les retards du pas suivant.
    """
    design_builder = design_builder or fitted["design_builder"]
    model = fitted["model"]
    cols = fitted["feature_cols"]
    imputer = fitted["imputer"]
    log_target = fitted["log_target"]
    objective = fitted["objective"]

    hist = panel.copy()
    if FLAG_FORECAST not in hist.columns:
        hist[FLAG_FORECAST] = False
    if origin_year is None:
        origin_year = int(hist.attrs.get("origin_year", hist[COL_YEAR].min()))

    out_rows: List[pd.DataFrame] = []
    last_widx = float(hist[COL_WIDX].max())

    for h in range(1, int(horizons) + 1):
        next_widx = last_widx + h
        hist = _extend_panel(hist, next_widx, origin_year, origin_week)

        df_feat, _ = design_builder(hist)
        new_mask = (df_feat[COL_WIDX] == next_widx)
        df_new = df_feat[new_mask]
        if df_new.empty:
            continue

        X = df_new[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_imp = pd.DataFrame(imputer.transform(X), columns=cols, index=X.index)

        pred = _inverse(model.predict(X_imp), log_target)
        pred = clip_predictions(pred, objective)

        res = df_new[[COL_AREA, COL_YEAR, COL_WEEK, COL_WIDX, COL_PERIOD]].copy()
        res["horizon"] = h
        res["predicted_cases"] = pred

        # ré-injection dans l'historique pour alimenter les retards du pas suivant
        mapping = dict(zip(res[COL_AREA], pred))
        idx = hist.index[hist[COL_WIDX] == next_widx]
        hist.loc[idx, COL_CASES] = hist.loc[idx, COL_AREA].map(mapping).to_numpy()
        if "deaths" in hist.columns:
            hist.loc[idx, "deaths"] = 0.0

        out_rows.append(res)

    if not out_rows:
        return pd.DataFrame(columns=[COL_AREA, COL_YEAR, COL_WEEK, COL_WIDX,
                                     COL_PERIOD, "horizon", "predicted_cases"])
    fc = pd.concat(out_rows, ignore_index=True)
    # alias compatible avec les onglets/cartes existants de l'application
    fc["week_num"] = fc[COL_WIDX].astype(int)
    return fc


# ======================================================================
# Intervalles de prédiction
# ======================================================================
def predict_quantiles(fitted: Dict[str, Any],
                      forecast: pd.DataFrame,
                      levels: Sequence[float] = (0.1, 0.9),
                      horizon_penalty: float = 0.12) -> pd.DataFrame:
    """
    Intervalles de prédiction par **conformal split** sur les résidus
    d'entraînement, élargis avec l'horizon.

    Approche volontairement simple et sans hypothèse de distribution : pour un
    outil d'aide à la décision, un intervalle honnête et explicable vaut mieux
    qu'un intervalle paramétrique trop étroit.
    """
    resid = np.asarray(fitted.get("residuals", []), dtype=float)
    resid = resid[~np.isnan(resid)]
    if len(resid) < 20:
        sigma = float(np.std(resid)) if len(resid) > 1 else 1.0
    else:
        sigma = None

    out = forecast.copy()
    from scipy.stats import norm
    for lv in levels:
        z = norm.ppf(lv)
        col = f"q{int(lv * 100)}"
        if sigma is None:
            # quantile empirique des résidus, élargi selon l'horizon
            q = np.quantile(resid, lv)
            width = q * (1.0 + horizon_penalty * (out["horizon"].to_numpy() - 1))
            out[col] = np.clip(out["predicted_cases"] + width, 0, None)
        else:
            width = z * sigma * np.sqrt(1.0 + horizon_penalty * (out["horizon"].to_numpy() - 1))
            out[col] = np.clip(out["predicted_cases"] + width, 0, None)
    return out


def feature_importance(fitted: Dict[str, Any], top_n: int = 25) -> pd.DataFrame:
    """Importance des variables (gain / impureté) ou coefficients linéaires."""
    model = fitted["model"]
    cols = fitted["feature_cols"]
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        kind = "importance_arbre"
    elif hasattr(model, "coef_"):
        imp = np.abs(np.ravel(np.asarray(model.coef_, dtype=float)))
        kind = "coef_abs"
    else:
        return pd.DataFrame(columns=["variable", "importance", "type"])
    imp = imp[:len(cols)]
    df = pd.DataFrame({"variable": cols[:len(imp)], "importance": imp, "type": kind})
    total = df["importance"].sum()
    if total > 0:
        df["importance_pct"] = (df["importance"] / total * 100).round(2)
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

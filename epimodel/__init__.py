"""
epimodel — noyau de modélisation prédictive pour la surveillance épidémiologique.

Objectif : fournir à EpiPrediction un pipeline de prévision **sans fuite de
données**, **sans décalage entraînement/inférence**, avec :

* un panneau équilibré (toutes les aires x toutes les semaines, zéros explicites) ;
* un index temporel continu ISO-8601 (plus d'écrasement multi-années) ;
* des covariables **statiques** (altitude, pente, NDVI, eau, urbain, population) ;
* une zoo de modèles : RandomForest, GradientBoosting, ExtraTrees, **XGBoost**,
  **LightGBM**, HistGradientBoosting, Ridge ;
* une validation **temporelle bloquée par semaine** avec embargo, multi-horizons,
  comparée à des **baselines naïves** (persistance, saisonnier, moyenne) ;
* des intervalles de prédiction (quantiles / conforme).

Aucune dépendance à Streamlit : le module est testable et réutilisable ailleurs.
Les noms de colonnes historiques de l'application sont conservés.
"""
from .panel import (  # noqa: F401
    build_panel,
    iso_week_index,
    iso_week_to_date,
    week_index_to_iso,
)
from .features import (  # noqa: F401
    add_climate_features,
    add_population_features,
    add_seasonality,
    add_spatial_features,
    add_static_features,
    add_temporal_features,
    build_design_matrix,
    describe_feature_dynamics,
    invariant_climate_variables,
    select_features,
)
from .models import (  # noqa: F401
    MODEL_REGISTRY,
    available_models,
    describe_model,
    has_lightgbm,
    has_xgboost,
    make_model,
)
from .validation import (  # noqa: F401
    backtest,
    metrics_summary,
    naive_baselines,
    temporal_cv_splits,
)
from .forecast import (  # noqa: F401
    fit_pipeline,
    predict_quantiles,
    recursive_forecast,
)

__version__ = "1.0.0"

__all__ = [
    "build_panel", "iso_week_index", "iso_week_to_date", "week_index_to_iso",
    "add_temporal_features", "add_seasonality", "add_static_features",
    "add_population_features", "add_spatial_features", "add_climate_features",
    "build_design_matrix", "select_features",
    "describe_feature_dynamics", "invariant_climate_variables",
    "MODEL_REGISTRY", "available_models", "make_model", "describe_model",
    "has_xgboost", "has_lightgbm",
    "temporal_cv_splits", "backtest", "metrics_summary", "naive_baselines",
    "fit_pipeline", "recursive_forecast", "predict_quantiles",
]

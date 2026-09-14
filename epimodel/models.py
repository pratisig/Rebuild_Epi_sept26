"""
Zoo de modèles pour la prévision épidémiologique.

Ajouts principaux par rapport au pipeline historique :

* **XGBoost** (``reg:squarederror`` ou ``count:poisson``) — boosting régularisé,
  généralement supérieur à ``GradientBoostingRegressor`` de scikit-learn en
  précision comme en vitesse sur des panneaux de plusieurs milliers de lignes ;
* **LightGBM** — très rapide, utile quand le nombre d'aires est important ;
* **HistGradientBoosting** — toujours disponible (scikit-learn), sert de
  solution de repli si xgboost/lightgbm ne sont pas installés ;
* objectifs **Poisson** avec offset d'exposition : les cas sont des comptages,
  la variance croît avec la moyenne ; un objectif gaussien sur des comptages
  hétérogènes est mal spécifié ;
* objectifs **quantiles** (q10/q50/q90) pour produire des intervalles de
  prédiction, indispensables à un outil d'aide à la décision ;
* les arbres étant invariants par transformation monotone de chaque variable,
  la « pondération manuelle des variables » de l'ancien mode Expert n'avait
  **aucun effet** sur RF/GB/DT. Elle est remplacée par un mécanisme qui agit
  réellement : ``sample_weight`` et sélection de variables.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

# ----------------------------------------------------------------------
# Détection des bibliothèques optionnelles
# ----------------------------------------------------------------------
def has_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# Paramètres par défaut (volontairement sobres : petits panneaux, risque
# de surapprentissage élevé)
# ----------------------------------------------------------------------
# Mono-thread par défaut. Mesure effectuée sur un panneau réel de
# 11 160 observations x 42 variables (XGBoost 600 arbres, `hist`) :
#   n_jobs=1  ->  1.9 s      n_jobs=-1 -> 47.4 s      n_jobs=4 -> 75.1 s
# Sur des panneaux de cette taille, l'overhead OpenMP dépasse largement le gain
# parallèle ; le multi-threading n'a d'intérêt qu'au-delà de ~10^6 observations.
# `THREADS` reste configurable pour les déploiements sur gros serveur.
import os as _os

THREADS = 1


def _n_jobs() -> int:
    return int(THREADS) if int(THREADS) > 0 else (_os.cpu_count() or 1)


_TREE_KW = dict(random_state=42)

_POISSON_ALIASES = {
    "xgboost": "count:poisson",
    "lightgbm": "poisson",
    "hist": "poisson",
    "sklearn_gb": "poisson",
}
_QUANTILE_ALIASES = {
    "xgboost": "reg:quantileerror",
    "lightgbm": "quantile",
    "hist": "quantile",
}


def _xgboost(objective: str = "squared_error", quantile: Optional[float] = None,
             **overrides) -> Any:
    from xgboost import XGBRegressor
    params: Dict[str, Any] = dict(
        n_estimators=600,
        learning_rate=0.045,
        max_depth=5,
        min_child_weight=5.0,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        reg_alpha=0.0,
        gamma=0.0,
        tree_method="hist",
        random_state=42,
        n_jobs=_n_jobs(),
    )
    if objective == "poisson":
        params["objective"] = "count:poisson"
    elif objective == "quantile":
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = quantile if quantile is not None else 0.5
    else:
        params["objective"] = "reg:squarederror"
    params.update(overrides)
    return XGBRegressor(**params)


def _lightgbm(objective: str = "squared_error", quantile: Optional[float] = None,
              **overrides) -> Any:
    from lightgbm import LGBMRegressor
    params: Dict[str, Any] = dict(
        n_estimators=600,
        learning_rate=0.045,
        num_leaves=31,
        max_depth=6,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=_n_jobs(),
        verbose=-1,
    )
    if objective == "poisson":
        params["objective"] = "poisson"
    elif objective == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = quantile if quantile is not None else 0.5
    else:
        params["objective"] = "regression"
    params.update(overrides)
    return LGBMRegressor(**params)


def _hist(objective: str = "squared_error", quantile: Optional[float] = None,
          **overrides) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    params: Dict[str, Any] = dict(
        max_iter=500, learning_rate=0.05, max_depth=6,
        min_samples_leaf=10, l2_regularization=1.0,
        random_state=42, early_stopping=False,
    )
    if objective == "poisson":
        params["loss"] = "poisson"
    elif objective == "quantile":
        params["loss"] = "quantile"
        params["quantile"] = quantile if quantile is not None else 0.5
    else:
        params["loss"] = "squared_error"
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def _random_forest(**overrides) -> Any:
    from sklearn.ensemble import RandomForestRegressor
    params = dict(n_estimators=400, max_depth=None, min_samples_leaf=3,
                  max_features=0.5, n_jobs=_n_jobs(), **_TREE_KW)
    params.update(overrides)
    return RandomForestRegressor(**params)


def _gradient_boosting(**overrides) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=4,
                  min_samples_leaf=5, subsample=0.9, **_TREE_KW)
    params.update(overrides)
    return GradientBoostingRegressor(**params)


def _extra_trees(**overrides) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor
    params = dict(n_estimators=400, min_samples_leaf=3, max_features=0.7,
                  n_jobs=_n_jobs(), **_TREE_KW)
    params.update(overrides)
    return ExtraTreesRegressor(**params)


def _ridge(**overrides) -> Any:
    from sklearn.linear_model import Ridge
    params = dict(alpha=1.0)
    params.update(overrides)
    return Ridge(**params)


# ----------------------------------------------------------------------
# Registre
# ----------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "XGBoost": {
        "factory": _xgboost, "requires": "xgboost",
        "supports_objective": True, "supports_quantile": True,
        "label": "XGBoost (recommandé)",
        "help": ("Boosting régularisé (hist). Meilleur compromis précision / "
                 "robustesse sur panneaux épidémiologiques ; supporte Poisson "
                 "et les quantiles."),
    },
    "LightGBM": {
        "factory": _lightgbm, "requires": "lightgbm",
        "supports_objective": True, "supports_quantile": True,
        "label": "LightGBM",
        "help": "Boosting très rapide, adapté aux grands nombres d'aires de santé.",
    },
    "HistGradientBoosting": {
        "factory": _hist, "requires": None,
        "supports_objective": True, "supports_quantile": True,
        "label": "HistGradientBoosting (scikit-learn)",
        "help": "Boosting natif scikit-learn : toujours disponible, gère les NaN.",
    },
    "GradientBoosting": {
        "factory": _gradient_boosting, "requires": None,
        "supports_objective": False, "supports_quantile": False,
        "label": "GradientBoosting (historique)",
        "help": "Algorithme conservé pour compatibilité avec les analyses antérieures.",
    },
    "RandomForest": {
        "factory": _random_forest, "requires": None,
        "supports_objective": False, "supports_quantile": False,
        "label": "RandomForest (historique)",
        "help": "Algorithme conservé pour compatibilité ; robuste mais peu extrapole.",
    },
    "ExtraTrees": {
        "factory": _extra_trees, "requires": None,
        "supports_objective": False, "supports_quantile": False,
        "label": "ExtraTrees",
        "help": "Forêts extrêmement aléatoires ; rapide, utile en comparaison.",
    },
    "Ridge": {
        "factory": _ridge, "requires": None,
        "supports_objective": False, "supports_quantile": False,
        "label": "Ridge (référence linéaire)",
        "help": "Référence linéaire : si un arbre ne bat pas Ridge, les "
                "variables posent problème.",
    },
}

DEFAULT_MODEL = "XGBoost" if has_xgboost() else "HistGradientBoosting"


def available_models() -> list:
    """Noms des modèles utilisables dans l'environnement courant."""
    out = []
    for name, spec in MODEL_REGISTRY.items():
        req = spec.get("requires")
        if req == "xgboost" and not has_xgboost():
            continue
        if req == "lightgbm" and not has_lightgbm():
            continue
        out.append(name)
    return out


def describe_model(name: str) -> str:
    spec = MODEL_REGISTRY.get(name)
    return spec["help"] if spec else ""


def make_model(name: str = DEFAULT_MODEL,
               objective: str = "squared_error",
               quantile: Optional[float] = None,
               **overrides) -> Any:
    """
    Instancie un modèle du registre.

    ``objective`` : ``"squared_error"`` | ``"poisson"`` | ``"quantile"``.
    Si le modèle choisi ne supporte pas l'objectif demandé, on retombe sur
    ``squared_error`` (et le appelant peut alors utiliser une transformation
    log1p — voir :func:`epimodel.forecast.fit_pipeline`).
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Modèle inconnu : {name!r}. Disponibles : {available_models()}")
    spec = MODEL_REGISTRY[name]
    req = spec.get("requires")
    if req == "xgboost" and not has_xgboost():
        raise RuntimeError("xgboost n'est pas installé (pip install xgboost)")
    if req == "lightgbm" and not has_lightgbm():
        raise RuntimeError("lightgbm n'est pas installé (pip install lightgbm)")

    obj = objective
    if objective != "squared_error" and not spec["supports_objective"]:
        obj = "squared_error"
    if obj == "quantile" and not spec["supports_quantile"]:
        obj = "squared_error"

    factory: Callable = spec["factory"]
    if obj == "squared_error" and "objective" not in overrides:
        try:
            return factory(**overrides)
        except TypeError:
            return factory()
    try:
        return factory(objective=obj, quantile=quantile, **overrides)
    except TypeError:
        return factory(**overrides)


def resolve_objective(model_name: str, objective: str) -> str:
    """Objectif réellement applicable par le modèle demandé."""
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        return "squared_error"
    if objective == "squared_error":
        return "squared_error"
    if not spec["supports_objective"]:
        return "squared_error"
    return objective


def clip_predictions(y_pred: np.ndarray, objective: str) -> np.ndarray:
    """Les cas sont des comptages : jamais négatifs."""
    y = np.asarray(y_pred, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(y, 0.0, None)

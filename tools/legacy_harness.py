"""
Harnais d'exécution du VRAI code applicatif (aucune réécriture de logique).

Principe
--------
1. On parse `app_paludisme.py` / `app_rougeole.py` avec `ast`.
2. On compile toutes les `FunctionDef` de niveau module  -> vraies fonctions livrées.
3. On compile le bloc `with tab3:` (l'onglet Modélisation) -> vrai code livré.
4. On exécute le tout avec un stub Streamlit (`tools/st_stub.py`).

Ainsi les métriques / sorties obtenues proviennent du code réellement embarqué
dans l'application, pas d'une copie.
"""
from __future__ import annotations

import ast
import os
import sys
import types
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_stub import StopException, StreamlitStub  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------
# Extraction AST du code réel
# ----------------------------------------------------------------------
# Noms que le harnais doit garder sous son contrôle : ils sont fournis par le
# stub et ne doivent jamais être écrasés par les imports du module audité.
_RESERVE = ("st",)


def _apply_module_imports(source: str, filename: str, ns: Dict[str, Any]) -> None:
    """
    Exécute les imports de niveau module du fichier audité.

    Sans cette étape, les fonctions extraites n'ont pas accès aux modules que
    l'application importe elle-même : ``population_cache_key`` utilisait
    ``hashlib`` et levait un ``NameError`` silencieux, avalé par son propre
    ``except Exception``. Résultat : la clé de cache retombait sur son empreinte
    de repli et restait aveugle au contenu — exactement le défaut qu'elle était
    censée corriger.

    Chaque import est tenté isolément, et les noms réservés au stub sont
    restaurés ensuite. Cette précaution est indispensable : dès que Streamlit est
    réellement installé dans l'environnement, ``import streamlit as st`` réussit
    et remplace le stub par le vrai module — les appels d'interface du bloc
    testé partent alors dans le runtime réel au lieu d'être capturés, et le
    harnais ne mesure plus rien.

    Les imports qui échouent sont ignorés sans interrompre le reste.
    """
    reserves = {k: ns.get(k) for k in _RESERVE}
    tree = ast.parse(source, filename=filename)
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        try:
            mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(mod)
            exec(compile(mod, filename, "exec"), ns)
        except Exception:  # noqa: BLE001
            continue
    for k, v in reserves.items():
        if v is not None:
            ns[k] = v


def _compile_functions(source: str, filename: str, ns: Dict[str, Any]) -> None:
    tree = ast.parse(source, filename=filename)
    fn_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if fn_nodes:
        mod = ast.Module(body=fn_nodes, type_ignores=[])
        ast.fix_missing_locations(mod)
        exec(compile(mod, filename, "exec"), ns)


def _compile_block(source: str, filename: str, tab_name: str, ns: Dict[str, Any]) -> str:
    """Compile le bloc `with <tab_name>:` de niveau module. Retourne le code source."""
    tree = ast.parse(source, filename=filename)
    target = None
    for node in tree.body:
        if isinstance(node, ast.With):
            for item in node.items:
                ce = item.context_expr
                if isinstance(ce, ast.Name) and ce.id == tab_name:
                    target = node
                    break
        if target is not None:
            break
    if target is None:
        raise RuntimeError(f"Bloc `with {tab_name}:` introuvable dans {filename}")
    seg_start = target.lineno - 1
    seg_end = getattr(target, "end_lineno", None)
    src_lines = source.splitlines()[seg_start:seg_end]
    block_src = "\n".join(src_lines)
    mod = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, filename, "exec")
    ns[f"__block_{tab_name}__"] = code
    return block_src


def load_module_context(path: str, tab_name: str = "tab3",
                        stub: Optional[StreamlitStub] = None,
                        extra_globals: Optional[Dict[str, Any]] = None):
    """Charge le contexte d'exécution (vraies fonctions + bloc onglet compilé)."""
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    st = stub or StreamlitStub()
    ns: Dict[str, Any] = {
        "__name__": os.path.basename(path),
        "st": st,
        "np": np,
        "pd": pd,
    }
    # Imports réellement utilisés par les fonctions extraites
    import warnings
    import difflib
    import unicodedata
    import re
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                                  ExtraTreesRegressor)
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.spatial.distance import cdist
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    import sys as _sys, os as _os
    _repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _repo not in _sys.path:
        _sys.path.insert(0, _repo)
    import epimodel as em
    import epi_app_bridge
    import io, json, base64, tempfile, zipfile
    from io import BytesIO, StringIO
    try:
        import openpyxl  # noqa: F401
    except Exception:
        openpyxl = None
    from epimodel.static_covariates import (STATIC_COLUMNS as GEE_STATIC_COLUMNS,
                                            describe_coverage as static_coverage)

    ns.update({
        "px": px, "go": go, "datetime": datetime, "timedelta": timedelta,
        "em": em, "epi_app_bridge": epi_app_bridge,
        "io": io, "os": os, "json": json, "base64": base64,
        "tempfile": tempfile, "zipfile": zipfile,
        "BytesIO": BytesIO, "StringIO": StringIO, "openpyxl": openpyxl,
        "GEE_STATIC_COLUMNS": GEE_STATIC_COLUMNS, "static_coverage": static_coverage,
        "warnings": warnings, "difflib": difflib, "unicodedata": unicodedata, "re": re,
        "LinearRegression": LinearRegression, "Ridge": Ridge, "Lasso": Lasso,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "train_test_split": train_test_split, "cross_val_score": cross_val_score,
        "TimeSeriesSplit": TimeSeriesSplit,
        "StandardScaler": StandardScaler, "RobustScaler": RobustScaler,
        "LabelEncoder": LabelEncoder,
        "KMeans": KMeans, "PCA": PCA, "Pipeline": Pipeline,
        "SimpleImputer": SimpleImputer,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error, "r2_score": r2_score,
        "cdist": cdist,
    })
    if extra_globals:
        ns.update(extra_globals)

    _apply_module_imports(source, path, ns)
    _compile_functions(source, path, ns)
    block_src = _compile_block(source, path, tab_name, ns)
    return ns, st, block_src


def run_tab3(path: str, stub: StreamlitStub, variables: Dict[str, Any],
             tab_name: str = "tab3") -> Dict[str, Any]:
    """
    Exécute le vrai bloc `with tab3:` avec les variables d'entrée fournies.
    Retourne {'stub':..., 'ns':..., 'error':...}
    """
    ns, st, _ = load_module_context(path, tab_name=tab_name, stub=stub,
                                    extra_globals=variables)

    # objets de contexte pour les `with tabN:`
    class _TabCtx:
        def __enter__(self_inner):
            return None

        def __exit__(self_inner, *exc):
            return False

    for name in ("tab1", "tab2", "tab3", "tab4", "tab5", "tab6", "tab7"):
        ns[name] = _TabCtx()

    err = None
    try:
        exec(ns[f"__block_{tab_name}__"], ns)
    except StopException as e:
        err = f"st.stop(): {e}"
    except Exception as e:  # noqa: BLE001
        import traceback
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    return {"stub": st, "ns": ns, "error": err}

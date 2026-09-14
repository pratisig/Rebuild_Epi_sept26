"""
Validation temporelle honnête.

Trois règles non négociables pour un outil d'aide à la décision :

1. **Découpage par semaine, pas par ligne.** Les données sont un panneau
   (aires x semaines). ``TimeSeriesSplit`` appliqué à un tableau trié par aire
   — comme dans le pipeline historique — découpe en réalité **par aire de
   santé** : chaque fold contient alors l'intégralité de la période (y compris
   le futur) pour d'autres aires. Ici on découpe sur les valeurs distinctes de
   ``week_index``.

2. **Embargo.** Les variables de retard (jusqu'à 52 semaines) font que la
   dernière fenêtre d'entraînement contient de l'information postérieure à la
   coupure. On retire ``embargo`` semaines entre train et test.

3. **Comparaison à des baselines naïves.** Un R² de 0.9 ne veut rien dire si
   la persistance fait 0.95. On calcule systématiquement le *skill score*
   contre la meilleure baseline.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .panel import COL_AREA, COL_CASES, COL_WEEK, COL_WIDX


# ======================================================================
# Découpages temporels
# ======================================================================
def temporal_cv_splits(week_index: np.ndarray,
                       n_splits: int = 5,
                       embargo: int = 4,
                       min_train_frac: float = 0.4) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Validation croisée temporelle **bloquée par semaine**, fenêtre expanding.

    Retourne une liste de couples ``(train_idx, test_idx)`` d'indices
    positionnels. Les semaines de test sont toujours strictement postérieures
    aux semaines d'entraînement, séparées par ``embargo`` semaines.
    """
    wi = np.asarray(week_index, dtype=float)
    uniq = np.unique(wi[~np.isnan(wi)])
    if len(uniq) < n_splits + 2:
        raise ValueError(
            f"Nombre de semaines distinctes insuffisant ({len(uniq)}) "
            f"pour {n_splits} folds temporels.")

    n = len(uniq)
    first_test = max(int(np.ceil(n * min_train_frac)), 2)
    boundaries = np.linspace(first_test, n, n_splits + 1).astype(int)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        start, end = boundaries[k], boundaries[k + 1]
        if end <= start:
            continue
        test_weeks = set(uniq[start:end].tolist())
        train_weeks = set(uniq[:max(0, start - embargo)].tolist())
        if not train_weeks or not test_weeks:
            continue
        tr = np.where(np.isin(wi, list(train_weeks)))[0]
        te = np.where(np.isin(wi, list(test_weeks)))[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        splits.append((tr, te))
    return splits


def check_no_leakage(df: pd.DataFrame, train_idx: np.ndarray,
                     test_idx: np.ndarray) -> Dict[str, object]:
    """Vérifie qu'aucune semaine de test n'est présente dans l'entraînement."""
    tr_weeks = set(df.iloc[train_idx][COL_WIDX].unique().tolist())
    te_weeks = set(df.iloc[test_idx][COL_WIDX].unique().tolist())
    overlap = sorted(tr_weeks & te_weeks)
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_week_min": float(min(tr_weeks)) if tr_weeks else None,
        "train_week_max": float(max(tr_weeks)) if tr_weeks else None,
        "test_week_min": float(min(te_weeks)) if te_weeks else None,
        "test_week_max": float(max(te_weeks)) if te_weeks else None,
        "week_overlap": overlap,
        "leakage": bool(len(overlap) > 0),
        "train_areas": int(df.iloc[train_idx][COL_AREA].nunique()),
        "test_areas": int(df.iloc[test_idx][COL_AREA].nunique()),
    }


# ======================================================================
# Métriques
# ======================================================================
def metrics_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, R², sMAPE, biais — sur des comptages (>= 0)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(yt)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {"n": 0}

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    var = float(np.var(yt))
    r2 = float(1.0 - np.mean(err ** 2) / var) if var > 1e-12 else float("nan")

    denom = np.abs(yt) + np.abs(yp)
    safe = np.where(denom > 1e-12, denom, 1.0)
    smape = float(np.mean(np.where(denom > 1e-12, 2.0 * np.abs(err) / safe, 0.0)) * 100)

    mean_y = float(np.mean(yt))
    return {
        "n": int(len(yt)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "smape": smape,
        "bias": float(np.mean(err)),
        "mae_pct_mean": float(mae / mean_y * 100) if mean_y > 1e-12 else float("nan"),
        "mean_observed": mean_y,
        "total_observed": float(np.sum(yt)),
        "total_predicted": float(np.sum(yp)),
        "agg_ratio": float(np.sum(yp) / np.sum(yt)) if np.sum(yt) > 1e-12 else float("nan"),
    }


# ======================================================================
# Baselines naïves (références obligatoires)
# ======================================================================
def naive_baselines(df: pd.DataFrame, horizons: Sequence[int] = (1,)) -> pd.DataFrame:
    """
    Quatre références classiques en prévision épidémiologique :

    * ``persistance``      : la semaine t vaut la semaine t-1 ;
    * ``saisonnier_naif``  : la semaine t vaut la même semaine ISO de l'année
      précédente (t-52) ;
    * ``moyenne_aire``     : moyenne historique de l'aire ;
    * ``moyenne_saison``   : moyenne de l'aire pour la même semaine ISO.

    Toutes sont calculées **uniquement** sur le passé de chaque observation.
    """
    d = df.sort_values([COL_AREA, COL_WIDX]).reset_index(drop=True).copy()
    g = d.groupby(COL_AREA, sort=False)[COL_CASES]

    d["_persist_1"] = g.shift(1)
    d["_sais_52"] = g.shift(52)
    d["_mean_area"] = g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    d["_mean_season"] = (d.groupby([COL_AREA, COL_WEEK], sort=False)[COL_CASES]
                         .transform(lambda s: s.shift(1).expanding(min_periods=1).mean()))

    rows = []
    for name, col in [("persistance", "_persist_1"),
                      ("saisonnier_naif", "_sais_52"),
                      ("moyenne_aire", "_mean_area"),
                      ("moyenne_saison", "_mean_season")]:
        sub = d[[COL_CASES, col]].dropna()
        if len(sub) < 10:
            continue
        m = metrics_summary(sub[COL_CASES].to_numpy(), sub[col].to_numpy())
        m["modele"] = name
        m["horizon"] = 1
        rows.append(m)
    return pd.DataFrame(rows)


# ======================================================================
# Backtest multi-horizons (prévision récursive réelle)
# ======================================================================
def backtest(panel: pd.DataFrame,
             design_builder: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]],
             model_factory: Callable[[], object],
             horizons: Sequence[int] = (1, 2, 4, 8),
             n_origins: int = 12,
             min_train_weeks: int = 52,
             feature_cols: Optional[Sequence[str]] = None,
             objective: str = "squared_error",
             log_target: bool = False,
             with_baselines: bool = True,
             verbose: bool = False) -> pd.DataFrame:
    """
    Backtest en origine glissante : pour plusieurs dates de coupure, on
    entraîne sur le passé puis on prédit **récursivement** h = 1..H semaines,
    exactement comme en production. C'est la seule évaluation qui mesure ce que
    l'outil promet (prévoir les semaines à venir).

    Les baselines naïves sont évaluées **sur les mêmes jeux d'évaluation**, ce
    qui permet un calcul de skill score sans biais.

    ``design_builder`` doit être la fonction qui ajoute les variables à un
    panneau (typiquement :func:`epimodel.features.build_design_matrix`
    partiellement appliquée) : la même fonction est utilisée pour
    l'entraînement et la prévision.
    """
    from .forecast import recursive_forecast

    wi_all = np.sort(panel[COL_WIDX].dropna().unique())
    if len(wi_all) <= min_train_weeks + max(horizons) + 1:
        raise ValueError("Historique insuffisant pour le backtest multi-horizons.")

    usable = wi_all[min_train_weeks: len(wi_all) - max(horizons)]
    if len(usable) == 0:
        raise ValueError("Aucune origine de backtest possible avec cet historique.")
    n_origins = max(1, min(n_origins, len(usable)))
    origins = np.unique(np.linspace(usable[0], usable[-1], n_origins).round().astype(int))

    # référence rapide cas[(aire, semaine)]
    truth = (panel[[COL_AREA, COL_WIDX, COL_CASES]]
             .set_index([COL_AREA, COL_WIDX])[COL_CASES])
    hist_mean = panel.groupby(COL_AREA)[COL_CASES].mean().to_dict()

    def _cases(area, widx):
        v = truth.get((area, widx), np.nan)
        return float(v) if v is not None and not pd.isna(v) else np.nan

    records = []
    pooled: Dict[Tuple[str, int], Dict[str, List[float]]] = {}

    def _push_pool(key: Tuple[str, int], y: np.ndarray, p: np.ndarray):
        d = pooled.setdefault(key, {"y": [], "p": []})
        d["y"].extend(np.asarray(y, dtype=float).tolist())
        d["p"].extend(np.asarray(p, dtype=float).tolist())

    for origin in origins:
        train_panel = panel[panel[COL_WIDX] <= origin].copy()
        if train_panel[COL_WIDX].nunique() < min_train_weeks:
            continue

        fitted = _fit_on_panel(train_panel, design_builder, model_factory,
                               feature_cols=feature_cols, objective=objective,
                               log_target=log_target)
        fc = recursive_forecast(
            fitted, panel=panel[panel[COL_WIDX] <= origin].copy(),
            horizons=int(max(horizons)), design_builder=design_builder)

        for h in horizons:
            sel = fc[fc["horizon"] == h]
            if sel.empty:
                continue
            rows = []
            for _, r in sel.iterrows():
                area = r[COL_AREA]
                target_w = r[COL_WIDX]
                y = _cases(area, target_w)
                if np.isnan(y):
                    continue
                rows.append({
                    "y": y,
                    "modele": _cases(area, origin),                 # persistance
                    "saisonnier_naif": _cases(area, target_w - 52),  # même semaine ISO n-1
                    "moyenne_aire": hist_mean.get(area, np.nan),
                    "pred": float(r["predicted_cases"]),
                })
            if not rows:
                continue
            d = pd.DataFrame(rows)

            m = metrics_summary(d["y"].to_numpy(), d["pred"].to_numpy())
            m["modele_nom"] = fitted["model_name"]
            m["horizon"] = int(h)
            m["origin"] = int(origin)
            m["type"] = "modele"
            records.append(m)
            _push_pool((fitted["model_name"], int(h)), d["y"].to_numpy(),
                       d["pred"].to_numpy())

            if with_baselines:
                for bname in ["modele", "saisonnier_naif", "moyenne_aire"]:
                    if bname == "modele":
                        continue
                    sub = d[["y", bname]].dropna()
                    if len(sub) < 10:
                        continue
                    mb = metrics_summary(sub["y"].to_numpy(), sub[bname].to_numpy())
                    mb["modele_nom"] = bname
                    mb["horizon"] = int(h)
                    mb["origin"] = int(origin)
                    mb["type"] = "baseline"
                    records.append(mb)
                    _push_pool((bname, int(h)), sub["y"].to_numpy(),
                               sub[bname].to_numpy())
        if verbose:
            print(f"  origine t={origin} : {len(fc)} prédictions")

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.rename(columns={"modele": "_dummy"}) if "modele" in df.columns else df
    agg = (df.groupby(["modele_nom", "horizon", "type"], as_index=False)
             .agg({"mae": "mean", "rmse": "mean", "r2": "mean",
                   "smape": "mean", "bias": "mean", "agg_ratio": "mean",
                   "mae_pct_mean": "mean", "n": "sum"}))
    agg = agg.rename(columns={"modele_nom": "modele"})

    # métriques POOLÉES : toutes les origines mises en commun pour un horizon.
    # Le R² moyen par origine est instable (variances hétérogènes selon la
    # période) ; la version poolée est celle à citer dans un rapport.
    pooled_rows = []
    for (name, h), d in pooled.items():
        pm = metrics_summary(np.asarray(d["y"]), np.asarray(d["p"]))
        pm["modele"] = name
        pm["horizon"] = h
        pooled_rows.append(pm)
    pdf = pd.DataFrame(pooled_rows)
    if not pdf.empty:
        pdf["type"] = np.where(pdf["modele"].isin(["saisonnier_naif", "moyenne_aire"]),
                               "baseline", "modele")
        agg = agg.merge(pdf[["modele", "horizon", "mae", "rmse", "r2", "smape",
                             "bias", "agg_ratio", "mae_pct_mean"]]
                        .rename(columns={c: f"{c}_pool" for c in
                                         ["mae", "rmse", "r2", "smape", "bias",
                                          "agg_ratio", "mae_pct_mean"]}),
                        on=["modele", "horizon"], how="left")

    # skill score MAE vs meilleure baseline (poolé)
    if not pdf.empty:
        base = pdf[pdf["modele"].isin(["saisonnier_naif", "moyenne_aire"])]
        best = (base.groupby("horizon")["mae"].min().rename("best_baseline_mae")
                .reset_index())
        agg = agg.merge(best, on="horizon", how="left")
        agg["skill_mae_vs_baseline"] = (1.0 - agg["mae_pool"] / agg["best_baseline_mae"]).round(4)
    return agg.sort_values(["type", "modele", "horizon"]).reset_index(drop=True)


def _fit_on_panel(train_panel, design_builder, model_factory,
                  feature_cols=None, objective="squared_error",
                  log_target=False):
    from .forecast import fit_pipeline
    return fit_pipeline(train_panel, design_builder=design_builder,
                        model_factory=model_factory, feature_cols=feature_cols,
                        objective=objective, log_target=log_target)


def skill_score(m_model: float, m_baseline: float) -> float:
    """
    Réduction relative d'erreur par rapport à la baseline (1 = parfait,
    <= 0 = pas mieux que la baseline).
    """
    if m_baseline is None or m_baseline <= 1e-12 or np.isnan(m_baseline):
        return float("nan")
    return float(1.0 - m_model / m_baseline)

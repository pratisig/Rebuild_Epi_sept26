"""
Benchmark EpiPrediction — pipeline historique vs noyau `epimodel`.

Produit `reports/benchmark_results.json` + `reports/benchmark_*.csv`,
consommés par `tools/generate_audit_report.py`.

Tout le volet « historique » est obtenu en exécutant le VRAI code de
`app_paludisme.py` (via `tools/legacy_harness.py`), pas une copie.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import epimodel as em  # noqa: E402
from epimodel.features import build_neighbour_weights  # noqa: E402
from epimodel.validation import (backtest, metrics_summary, naive_baselines,  # noqa: E402,F401
                                 skill_score, temporal_cv_splits)
from legacy_harness import load_module_context, run_tab3  # noqa: E402
from make_synthetic_data import build_dataset  # noqa: E402
from st_stub import StreamlitStub  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Version ORIGINALE (commit c5dde9b) extraite du dépôt : c'est elle que le
# banc d'essai « legacy » doit exécuter, pas le fichier corrigé du dépôt.
APP_PALU = os.path.join(REPO, "tools", "_baseline", "app_paludisme_orig.py")
OUT_DIR = os.path.join(REPO, "reports")
HORIZONS = (1, 2, 4, 8)


# ======================================================================
# Helpers
# ======================================================================
def make_static(ds, with_population=True):
    cols = ["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]
    if with_population:
        return ds.df_static.merge(ds.df_population[cols], on="health_area", how="outer")
    return ds.df_static.copy()


def design_factory(ds, use_static=True, use_population=True, use_spatial=True,
                   lags=em.features.DEFAULT_LAGS):
    static = make_static(ds, with_population=use_population) if use_static else (
        ds.df_population[["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]]
        if use_population else None)
    W = build_neighbour_weights(ds.gdf, k=5) if use_spatial else None
    groups = list(em.features.DEFAULT_GROUP_ORDER)
    if not use_spatial:
        groups = [g for g in groups if g != "spatial"]
    if not use_static:
        groups = [g for g in groups if g != "static"]
    if not use_population:
        groups = [g for g in groups if g != "population"]
    return partial(em.build_design_matrix, static_df=static, gdf=ds.gdf,
                   neighbour_weights=W, n_clusters=5 if use_spatial else 0,
                   groups=groups, lags=lags)


# ======================================================================
# PARTIE 1 — métriques telles qu'affichées par l'application
# ======================================================================
def partie1_legacy_metrics(ds, algos=("RandomForest", "GradientBoosting", "ExtraTrees")):
    rows = []
    for algo in algos:
        stub = StreamlitStub(selectbox_values={"Algorithme": algo},
                             slider_values={"Semaines à prévoir": 4, "Seuil alerte": 75})
        df_cases = ds.df_cases.copy()
        df_cases["health_area"] = df_cases["health_area"].astype(str).str.strip().str.lower()
        gdf_health = ds.gdf.merge(
            ds.df_population[["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]],
            on="health_area", how="left")
        for k in ["gdf_health", "df_cases", "temp_raster", "flood_raster", "rivers_gdf",
                  "precipitation_raster", "humidity_raster", "elevation_raster",
                  "model_results", "df_climate_aggregated"]:
            stub.session_state[k] = None
        stub.session_state["gdf_health"] = gdf_health
        stub.session_state["df_cases"] = df_cases
        stub.session_state["dfpopulation"] = ds.df_population
        res = run_tab3(APP_PALU, stub, {
            "df_cases": df_cases, "gdf_health": gdf_health,
            "years_selected": sorted(ds.df_cases["year"].unique().tolist()),
            "iso3pays": "ner"}, tab_name="tab3")
        if res["error"]:
            rows.append({"algorithme": algo, "erreur": res["error"][:200]})
            continue
        mr = stub.session_state["model_results"]
        m = mr["metrics"]
        df_model, df_future = mr["df_model"], mr["df_future"]
        obs = float(df_model["cases"].mean())
        pred = float(df_future["predicted_cases"].mean())
        rows.append({
            "algorithme": algo,
            "R2_affiche_in_sample": round(float(m["r2"]), 4),
            "R2_CV_affiche": round(float(m["cv_r2_mean"]), 4),
            "R2_CV_ecart_type": round(float(m["cv_r2_std"]), 4),
            "MAE_in_sample": round(float(m["mae"]), 2),
            "RMSE_in_sample": round(float(m["rmse"]), 2),
            "n_features_modele": len(mr["feature_cols"]),
            "n_obs_modele": int(len(df_model)),
            "n_obs_brutes": int(len(ds.df_cases)),
            "ratio_pred_obs": round(pred / obs, 3) if obs else None,
            "erreur": None,
        })
        print(f"  [P1] {algo}: R²={m['r2']:.4f} CV-R²={m['cv_r2_mean']:.4f} "
              f"ratio_préd/obs={pred / obs:.3f}")
    return pd.DataFrame(rows)


# ======================================================================
# PARTIE 2 — d'où vient le R² ≈ 0.99 ? (fuite + découpage)
# ======================================================================
def partie2_leakage(ds):
    """
    Utilise les VRAIES fonctions de feature-engineering de app_paludisme.py
    (`create_advanced_features`, `create_population_features`) mais sur un
    panneau correct, puis compare trois protocoles d'évaluation.
    """
    ns, st, _ = load_module_context(APP_PALU, tab_name="tab3", stub=StreamlitStub())
    create_advanced_features = ns["create_advanced_features"]
    create_population_features = ns["create_population_features"]

    panel = em.build_panel(ds.df_cases)
    df = panel.copy()
    # `create_advanced_features` (code réel de l'app) attend une colonne week_num
    df["week_num"] = df["week_index"]
    df = create_advanced_features(df)
    df = df.merge(make_static(ds), on="health_area", how="left")
    df = create_population_features(df)

    all_feats = [c for c in ["cases_lag_1", "cases_lag_2", "cases_lag_4", "cases_ma_2",
                             "cases_ma_4", "growth_rate", "sin_week", "cos_week",
                             "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop",
                             "incidence_rate", "child_risk", "demo_pressure",
                             "coef_population"] if c in df.columns]
    leaky = ["incidence_rate", "child_risk", "demo_pressure"]
    clean_feats = [c for c in all_feats if c not in leaky]

    d = df.dropna(subset=["cases_lag_1"]).reset_index(drop=True)
    y = d["cases"].to_numpy(dtype=float)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline

    def _model():
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                      random_state=42, n_jobs=-1))])

    def _eval(feats, splitter):
        # même nettoyage que le code livré : `X.replace([inf,-inf], nan)`
        X = (d[feats].apply(pd.to_numeric, errors="coerce")
             .replace([np.inf, -np.inf], np.nan))
        tr_all, te_all = [], []
        for tr, te in splitter:
            mdl = _model().fit(X.iloc[tr], y[tr])
            tr_all.append(te)
            te_all.append(np.clip(mdl.predict(X.iloc[te]), 0, None))
        te_idx = np.concatenate(tr_all)
        y_pred = np.concatenate(te_all)
        m = metrics_summary(y[te_idx], y_pred)
        # R² in-sample (entraînement sur tout)
        full = _model().fit(X, y)
        m["r2_in_sample"] = metrics_summary(y, np.clip(full.predict(X), 0, None))["r2"]
        return m

    n = len(d)
    kf = list(KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(n)))
    # découpage temporel honnête sur week_index
    ts = temporal_cv_splits(d["week_index"].to_numpy(), n_splits=5, embargo=4)

    res = {
        "protocole_A_split_aleatoire_avec_fuite": _eval(all_feats, kf),
        "protocole_B_temporel_avec_fuite": _eval(all_feats, ts),
        "protocole_C_temporel_sans_fuite": _eval(clean_feats, ts),
        "features_avec_fuite": leaky,
        "features_totales": len(all_feats),
        "n_obs": int(n),
    }
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  [P2] {k}: R²={v['r2']:.4f}  MAE={v['mae']:.1f}  "
                  f"R² in-sample={v['r2_in_sample']:.4f}")
    return res


# ======================================================================
# PARTIE 3 — comparaison des modèles sur protocole corrigé
# ======================================================================
def partie3_models(ds, models=None, horizons=HORIZONS, n_origins=12, tag="complet"):
    if models is None:
        models = [m for m in ["RandomForest", "GradientBoosting", "ExtraTrees",
                              "HistGradientBoosting", "XGBoost", "LightGBM", "Ridge"]
                  if m in em.available_models()]
    panel = em.build_panel(ds.df_cases)
    db = design_factory(ds)
    df_feat, cols = db(panel)
    print(f"  [P3:{tag}] {len(cols)} variables, {len(panel)} lignes")

    frames = []
    for i, name in enumerate(models):
        t0 = time.time()
        try:
            agg = backtest(panel, db, lambda n=name: em.make_model(n),
                           horizons=horizons, n_origins=n_origins,
                           min_train_weeks=60, with_baselines=(i == 0),
                           verbose=False)
        except Exception as e:  # noqa: BLE001
            print(f"    {name}: ÉCHEC {type(e).__name__}: {e}")
            continue
        agg["config"] = tag
        agg["secondes"] = round(time.time() - t0, 1)
        frames.append(agg)
        # même filtrage : ne lire que les lignes du modèle, pas les références
        _m1 = agg[(agg.horizon == 1) & (agg.type != "baseline")]
        _m4 = agg[(agg.horizon == 4) & (agg.type != "baseline")]
        print(f"    {name:22s} h=1 MAE={_m1['mae_pool'].iloc[0]:8.1f} "
              f"R²pool={_m1['r2_pool'].iloc[0]:.3f} | "
              f"h=4 MAE={_m4['mae_pool'].iloc[0]:8.1f} "
              f"R²pool={_m4['r2_pool'].iloc[0]:.3f}  ({agg['secondes'].iloc[0]}s)")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    bl = out[out["type"] == "baseline"].copy() if "type" in out.columns else pd.DataFrame()
    if not bl.empty:
        print("  [P3] baselines naïves (mêmes jeux d'évaluation) :")
        for _, r in bl[bl.horizon == 1].iterrows():
            print(f"    {r['modele']:18s} MAE={r['mae']:8.1f}  R²pool={r['r2_pool']:.3f}")
    return out, bl


# ======================================================================
# PARTIE 4 — ablations
# ======================================================================
def partie4_ablations(ds, models=("XGBoost", "RandomForest"), horizons=(1, 4), n_origins=12):
    panel = em.build_panel(ds.df_cases)
    configs = {
        "complet (statiques + spatial + lags 52)": design_factory(ds),
        "sans covariables statiques": design_factory(ds, use_static=False),
        "sans lag spatial": design_factory(ds, use_spatial=False),
        "lags legacy (1,2,4 uniquement)": design_factory(ds, lags=(1, 2, 4)),
    }
    rows = []
    for label, db in configs.items():
        for name in models:
            if name not in em.available_models():
                continue
            try:
                agg = backtest(panel, db, lambda n=name: em.make_model(n),
                               horizons=horizons, n_origins=n_origins,
                               min_train_weeks=60)
            except Exception as e:  # noqa: BLE001
                print(f"    {label} / {name}: ÉCHEC {e}")
                continue
            agg["config"] = label
            rows.append(agg)
            # ATTENTION : `backtest` ajoute des lignes de référence (type
            # 'baseline'). Il faut les exclure, sinon on lit les chiffres
            # d'une baseline et toutes les configurations semblent identiques.
            m1 = agg[(agg.horizon == 1) & (agg.type != "baseline")]
            m4 = agg[(agg.horizon == 4) & (agg.type != "baseline")]
            print(f"    {label:42s} {name:12s} "
                  f"h=1 MAE={m1['mae_pool'].iloc[0]:8.1f} "
                  f"R²pool={m1['r2_pool'].iloc[0]:.3f} | "
                  f"h=4 MAE={m4['mae_pool'].iloc[0]:8.1f} "
                  f"R²pool={m4['r2_pool'].iloc[0]:.3f}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ======================================================================
# PARTIE 5 — objectif Poisson + intervalles
# ======================================================================
def partie5_objectives(ds, horizons=(1, 4), n_origins=12):
    panel = em.build_panel(ds.df_cases)
    db = design_factory(ds)
    rows = []
    for label, factory in [
        ("XGBoost squared_error", lambda: em.make_model("XGBoost", objective="squared_error")),
        ("XGBoost poisson", lambda: em.make_model("XGBoost", objective="poisson")),
        ("LightGBM poisson", lambda: em.make_model("LightGBM", objective="poisson")),
        ("XGBoost log1p", lambda: em.make_model("XGBoost", objective="squared_error")),
    ]:
        log_target = label.endswith("log1p")
        try:
            agg = backtest(panel, db, factory, horizons=horizons, n_origins=n_origins,
                           min_train_weeks=60, log_target=log_target,
                           with_baselines=False)
        except Exception as e:  # noqa: BLE001
            print(f"    {label}: ÉCHEC {e}")
            continue
        agg["config"] = label
        agg["modele"] = label
        rows.append(agg)
        print(f"    {label:24s} h=1 MAE={agg.loc[agg.horizon == 1, 'mae'].iloc[0]:8.1f} "
              f"R²={agg.loc[agg.horizon == 1, 'r2'].iloc[0]:.3f} "
              f"agg_ratio={agg.loc[agg.horizon == 1, 'agg_ratio'].iloc[0]:.3f}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ======================================================================
# PARTIE 6 — expérience de contrôle (altitude sans effet)
# ======================================================================
def partie6_control(ds_with, ds_without, horizons=(1, 4), n_origins=12):
    out = {}
    for label, ds in [("altitude_avec_effet", ds_with), ("altitude_sans_effet", ds_without)]:
        panel = em.build_panel(ds.df_cases)
        rows = []
        for cfg, db in [("avec_statiques", design_factory(ds)),
                        ("sans_statiques", design_factory(ds, use_static=False))]:
            agg = backtest(panel, db, lambda: em.make_model("XGBoost"),
                           horizons=horizons, n_origins=n_origins, min_train_weeks=60,
                           with_baselines=False)
            agg["config"] = cfg
            rows.append(agg)
        df = pd.concat(rows, ignore_index=True)
        out[label] = df
        a = df.loc[(df.config == "avec_statiques") & (df.horizon == 1), "mae"].iloc[0]
        b = df.loc[(df.config == "sans_statiques") & (df.horizon == 1), "mae"].iloc[0]
        print(f"  [P6] {label}: MAE h=1 avec={a:.1f} sans={b:.1f} "
              f"gain={100 * (1 - a / b):.1f}%")
    return out


# ======================================================================
# PARTIE 7 — importance des variables du modèle retenu
# ======================================================================
def partie7_importance(ds):
    panel = em.build_panel(ds.df_cases)
    db = design_factory(ds)
    fitted = em.fit_pipeline(panel, design_builder=db,
                             model_factory=lambda: em.make_model("XGBoost"))
    imp = em.forecast.feature_importance(fitted, top_n=25)
    print("  [P7] top 12 variables (XGBoost) :")
    for _, r in imp.head(12).iterrows():
        print(f"    {r['variable']:20s} {r['importance_pct']:.2f}%")
    return imp, fitted


# ======================================================================
def _save(results):
    with open(os.path.join(OUT_DIR, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    only = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n=== PARTIE 1 : métriques affichées par l'application (code réel) ===")
    ds = build_dataset()
    results["meta_dataset"] = ds.meta
    p1 = partie1_legacy_metrics(ds)
    p1.to_csv(os.path.join(OUT_DIR, "benchmark_p1_legacy.csv"), index=False)
    results["p1_legacy"] = json.loads(p1.to_json(orient="records"))

    print("\n=== PARTIE 2 : origine du R² élevé (fuite + découpage) ===")
    try:
        p2 = partie2_leakage(ds)
    except Exception as e:
        import traceback; traceback.print_exc(); p2 = {"erreur": str(e)}
    results["p2_leakage"] = {k: (v if not isinstance(v, dict) else
                                 {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                                  for kk, vv in v.items()})
                             for k, v in p2.items()}

    _save(results)
    print("\n=== PARTIE 3 : comparaison des modèles (protocole corrigé) ===")
    p3, bl = partie3_models(ds)
    p3.to_csv(os.path.join(OUT_DIR, "benchmark_p3_models.csv"), index=False)
    if not bl.empty:
        bl.to_csv(os.path.join(OUT_DIR, "benchmark_p3_baselines.csv"), index=False)
    results["p3_models"] = json.loads(p3.to_json(orient="records"))
    results["p3_baselines"] = json.loads(bl.to_json(orient="records")) if not bl.empty else []

    _save(results)
    print("\n=== PARTIE 4 : ablations ===")
    p4 = partie4_ablations(ds)
    p4.to_csv(os.path.join(OUT_DIR, "benchmark_p4_ablations.csv"), index=False)
    results["p4_ablations"] = json.loads(p4.to_json(orient="records"))

    _save(results)
    print("\n=== PARTIE 5 : objectifs (Poisson / log1p) ===")
    p5 = partie5_objectives(ds)
    p5.to_csv(os.path.join(OUT_DIR, "benchmark_p5_objectives.csv"), index=False)
    results["p5_objectives"] = json.loads(p5.to_json(orient="records"))

    _save(results)
    print("\n=== PARTIE 6 : expérience de contrôle sur l'altitude ===")
    ds_no = build_dataset(altitude_effect=False, seed=999)
    p6 = partie6_control(ds, ds_no)
    for k, v in p6.items():
        v.to_csv(os.path.join(OUT_DIR, f"benchmark_p6_{k}.csv"), index=False)
    results["p6_control"] = {k: json.loads(v.to_json(orient="records")) for k, v in p6.items()}

    _save(results)
    print("\n=== PARTIE 7 : importance des variables ===")
    imp, fitted = partie7_importance(ds)
    imp.to_csv(os.path.join(OUT_DIR, "benchmark_p7_importance.csv"), index=False)
    results["p7_importance"] = json.loads(imp.to_json(orient="records"))
    results["p7_train_metrics"] = fitted["train_metrics"]
    results["n_features_final"] = len(fitted["feature_cols"])
    results["feature_cols_final"] = fitted["feature_cols"]

    _save(results)
    print("\n→ reports/benchmark_results.json écrit")


if __name__ == "__main__":
    main()

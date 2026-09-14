"""
Tests du noyau `epimodel` et du pipeline historique.

Ces tests exécutent le code réellement livré :
* `epimodel.*` pour le nouveau noyau ;
* `app_paludisme.py` (extrait par AST via `tools/legacy_harness.py`) pour les
  tests de non-régression du diagnostic.

Lancer :  python -m pytest tests -q
"""
from __future__ import annotations

import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import epimodel as em  # noqa: E402
from epimodel.features import build_neighbour_weights  # noqa: E402
from epimodel.panel import iso_week_to_date  # noqa: E402
from epimodel.validation import check_no_leakage, temporal_cv_splits  # noqa: E402

from make_synthetic_data import build_dataset  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def ds():
    return build_dataset(years=(2022, 2023, 2024))


@pytest.fixture(scope="module")
def panel(ds):
    return em.build_panel(ds.df_cases)


@pytest.fixture(scope="module")
def static(ds):
    from epi_app_bridge import build_static_table
    return build_static_table(ds.df_population, ds.df_static)


@pytest.fixture(scope="module")
def design_builder(ds, static):
    W = build_neighbour_weights(ds.gdf, k=5)
    return partial(em.build_design_matrix, static_df=static, gdf=ds.gdf,
                   neighbour_weights=W, n_clusters=5)


# ----------------------------------------------------------------------
# 1. Panneau : plus d'écrasement multi-années
# ----------------------------------------------------------------------
def test_panel_no_multiyear_collapse(ds, panel):
    n_areas = ds.df_cases["health_area"].nunique()
    n_weeks = panel["week_index"].nunique()
    assert len(panel) == n_areas * n_weeks
    # 3 années ISO x 52 semaines
    assert n_weeks == 156
    # chaque (aire, année, semaine) est unique -> aucune somme entre années
    dup = panel.duplicated(subset=["health_area", "year", "week_"]).sum()
    assert dup == 0
    # la semaine 1 de chaque année reste distincte
    abala = panel[(panel["health_area"] == "abala") & (panel["week_"] == 1)]
    assert len(abala) == 3
    assert abala["cases"].nunique() >= 1
    assert abala["week_index"].nunique() == 3


def test_panel_completes_grid_with_zeros(ds, panel):
    """Les semaines absentes du linelist deviennent des zéros explicites."""
    per_area = panel.groupby("health_area").size()
    assert per_area.nunique() == 1
    assert (panel["cases"] == 0).sum() > 0


def test_iso_week_handles_week_53():
    # 2020 est une année ISO à 53 semaines
    d = iso_week_to_date(2020, 53)
    assert (d.year, d.month) == (2020, 12)
    # la conversion historique (1er janvier + n semaines) décale
    from datetime import datetime, timedelta
    legacy = datetime(2020, 1, 1) + timedelta(weeks=52)
    assert legacy.date() != d


# ----------------------------------------------------------------------
# 2. Absence de fuite de la cible
# ----------------------------------------------------------------------
def test_features_do_not_use_current_week(panel, design_builder):
    """
    Perturber les cas d'une semaine T ne doit modifier AUCUNE variable de la
    semaine T (uniquement celles des semaines suivantes).
    """
    df0, cols = design_builder(panel)
    area = panel["health_area"].iloc[0]
    T = float(panel[panel["health_area"] == area]["week_index"].max()) - 5

    perturbed = panel.copy()
    m = (perturbed["health_area"] == area) & (perturbed["week_index"] == T)
    assert m.sum() == 1
    perturbed.loc[m, "cases"] = perturbed.loc[m, "cases"] + 10000.0

    df1, _ = design_builder(perturbed)
    key = ["health_area", "week_index"]
    a = df0.set_index(key)[cols].loc[(area, T)]
    b = df1.set_index(key)[cols].loc[(area, T)]
    diff = (a.fillna(-9e9) - b.fillna(-9e9)).abs()
    assert diff.max() < 1e-9, f"fuite détectée sur : {list(diff[diff > 1e-9].index)}"

    # et les semaines suivantes DOIVENT bouger (les lags fonctionnent)
    nxt = (df1[(df1["health_area"] == area) & (df1["week_index"] == T + 1)]
           ["cases_lag_1"].iloc[0])
    assert abs(nxt - df0[(df0["health_area"] == area)
                         & (df0["week_index"] == T + 1)]["cases_lag_1"].iloc[0]) > 0


def test_population_rates_are_not_target_leaks(panel, design_builder):
    df, cols = design_builder(panel)
    assert "incidence_rate" in cols
    sub = df.dropna(subset=["incidence_rate", "Pop_Totale"])
    recon = sub["incidence_rate"] * sub["Pop_Totale"] / 1e4
    # l'ancien code permettait de retrouver exactement la cible
    assert np.corrcoef(recon, sub["cases_lag_1"])[0, 1] > 0.999
    assert abs(np.corrcoef(recon, sub["cases"])[0, 1]) < 0.999


# ----------------------------------------------------------------------
# 3. Découpage temporel sans fuite
# ----------------------------------------------------------------------
def test_temporal_cv_has_no_week_overlap(panel, design_builder):
    df, _ = design_builder(panel)
    d = df[df["cases_lag_1"].notna()].reset_index(drop=True)
    splits = temporal_cv_splits(d["week_index"].to_numpy(), n_splits=5, embargo=4)
    assert len(splits) == 5
    for tr, te in splits:
        info = check_no_leakage(d, tr, te)
        assert info["leakage"] is False
        assert info["train_week_max"] < info["test_week_min"]


def test_legacy_sorting_would_leak(panel, design_builder):
    """
    Le tri historique (par aire puis semaine) fait que TimeSeriesSplit découpe
    par aire : le test contient les mêmes semaines que l'entraînement.
    """
    from sklearn.model_selection import TimeSeriesSplit
    df, _ = design_builder(panel)
    legacy_order = df.sort_values(["health_area", "week_index"]).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=5)
    leaks = 0
    for tr, te in tscv.split(legacy_order):
        if check_no_leakage(legacy_order, tr, te)["leakage"]:
            leaks += 1
    assert leaks == 5


# ----------------------------------------------------------------------
# 4. Ordre des variables déterministe
# ----------------------------------------------------------------------
def test_feature_order_is_deterministic(panel, design_builder):
    _, c1 = design_builder(panel)
    _, c2 = design_builder(panel.sample(frac=1.0, random_state=7).reset_index(drop=True))
    assert c1 == c2
    assert len(set(c1)) == len(c1)


# ----------------------------------------------------------------------
# 5. Zoo de modèles
# ----------------------------------------------------------------------
def test_xgboost_is_available_and_registered():
    assert em.has_xgboost()
    assert "XGBoost" in em.available_models()
    m = em.make_model("XGBoost")
    assert m.__class__.__name__ == "XGBRegressor"


def test_poisson_and_quantile_objectives():
    mp = em.make_model("XGBoost", objective="poisson")
    assert mp.get_params()["objective"] == "count:poisson"
    mq = em.make_model("XGBoost", objective="quantile", quantile=0.9)
    assert mq.get_params()["objective"] == "reg:quantileerror"
    # un modèle qui ne supporte pas l'objectif retombe proprement
    assert em.models.resolve_objective("RandomForest", "poisson") == "squared_error"


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        em.make_model("ModeleInexistant")


# ----------------------------------------------------------------------
# 6. Prévision récursive : pas de décalage train / inférence
# ----------------------------------------------------------------------
def test_forecast_has_no_train_serve_skew(panel, design_builder):
    fitted = em.fit_pipeline(panel, design_builder=design_builder,
                             model_factory=lambda: em.make_model("XGBoost"))
    fc = em.recursive_forecast(fitted, panel=panel.copy(), horizons=3,
                               design_builder=design_builder)
    assert {"health_area", "week_num", "predicted_cases"}.issubset(fc.columns)
    assert (fc["predicted_cases"] >= 0).all()
    assert fc["week_index"].max() == panel["week_index"].max() + 3

    # toute variable d'entraînement est bien renseignée à la prévision
    df_feat, cols = design_builder(panel)
    hist = panel.copy()
    # on rejoue la prévision pas à pas pour inspecter les variables réellement
    # présentées au modèle au premier pas
    from epimodel.forecast import _extend_panel
    # on prolonge sur TOUT l'horizon : une harmonique de Fourier s'annule
    # légitimement à certaines semaines (sin = 0 en semaine 1), donc le critère
    # d'effondrement doit porter sur l'ensemble des semaines prévues et non sur
    # la première seule.
    ext = hist.copy()
    oy = int(hist.attrs.get("origin_year", hist["year"].min()))
    ow = int(hist.attrs.get("origin_week", 1))
    for h in range(1, 4):
        ext = _extend_panel(ext, float(hist["week_index"].max()) + h, oy, ow)
    df_ext, _ = design_builder(ext)
    new = df_ext[df_ext["week_index"] > float(hist["week_index"].max())]
    assert new["week_index"].nunique() == 3
    missing = [c for c in cols if c not in new.columns]
    assert missing == []
    # aucune variable informative à l'entraînement ne s'effondre à zéro sur
    # l'ensemble de l'horizon de prévision
    train_ok = [c for c in cols
                if pd.to_numeric(df_feat[c], errors="coerce").abs().mean() > 1e-6]
    collapsed = [c for c in train_ok
                 if pd.to_numeric(new[c], errors="coerce").abs().mean() == 0.0]
    assert collapsed == [], f"variables effondrées à 0 à la prévision : {collapsed}"
    # week_frac doit être recalculé sur les lignes de prévision, pas recopié
    if "week_frac" in ext.columns:
        wf = ext.loc[ext["week_index"] > float(hist["week_index"].max()),
                     ["week_", "week_frac"]].drop_duplicates()
        assert (abs(wf["week_frac"] - (wf["week_"] - 1) / 52.1775) < 1e-9).all(), \
            "week_frac non recalculé sur les lignes de prévision"


def test_forecast_scale_is_consistent_with_history(panel, design_builder):
    """Le pipeline historique sous-estimait d'un facteur ~2 : plus le cas ici."""
    fitted = em.fit_pipeline(panel, design_builder=design_builder,
                             model_factory=lambda: em.make_model("XGBoost"))
    fc = em.recursive_forecast(fitted, panel=panel.copy(), horizons=4,
                               design_builder=design_builder)
    last8 = (panel.sort_values(["health_area", "week_index"])
             .groupby("health_area").tail(8).groupby("health_area")["cases"].mean())
    pred = fc.groupby("health_area")["predicted_cases"].mean()
    ratio = (pred / last8.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    med = float(ratio.median())
    assert 0.5 < med < 2.0, f"dérive du forecast : ratio médian {med:.3f}"


# ----------------------------------------------------------------------
# 7. Validation honnête
# ----------------------------------------------------------------------
def test_metrics_summary_basic():
    from epimodel.validation import metrics_summary
    m = metrics_summary(np.array([10.0, 20.0, 30.0]), np.array([10.0, 20.0, 30.0]))
    assert m["mae"] == 0.0 and abs(m["r2"] - 1.0) < 1e-12


def test_naive_baselines_computed(panel, design_builder):
    from epimodel.validation import naive_baselines
    df, _ = design_builder(panel)
    bl = naive_baselines(df)
    assert not bl.empty
    assert {"persistance", "saisonnier_naif", "moyenne_aire"}.issubset(set(bl["modele"]))


# ----------------------------------------------------------------------
# 8. Non-régression du diagnostic du code historique
# ----------------------------------------------------------------------
def test_legacy_pipeline_still_shows_documented_defects(ds):
    """
    Verrouille le diagnostic documenté dans le rapport d'audit : tant que le
    bloc `with tab3:` d'origine est présent, il écrase les années et produit un
    R² in-sample > 0.95. Si un jour ce bloc est remplacé, ce test doit être mis
    à jour (il sert de trace exécutable du constat).
    """
    from legacy_harness import run_tab3
    from st_stub import StreamlitStub

    df_cases = ds.df_cases.copy()
    df_cases["health_area"] = df_cases["health_area"].astype(str).str.strip().str.lower()
    gdf_health = ds.gdf.merge(
        ds.df_population[["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]],
        on="health_area", how="left")

    stub = StreamlitStub(selectbox_values={"Algorithme": "RandomForest"},
                         slider_values={"Semaines à prévoir": 4, "Seuil alerte": 75})
    for k in ["gdf_health", "df_cases", "temp_raster", "flood_raster", "rivers_gdf",
              "precipitation_raster", "humidity_raster", "elevation_raster",
              "model_results", "df_climate_aggregated"]:
        stub.session_state[k] = None
    stub.session_state["gdf_health"] = gdf_health
    stub.session_state["df_cases"] = df_cases
    stub.session_state["dfpopulation"] = ds.df_population

    # Version ORIGINALE (commit c5dde9b) extraite du dépôt : c'est elle qui doit
    # encore présenter les défauts documentés, pas le fichier corrigé du dépôt.
    baseline = os.path.join(REPO, "tools", "_baseline", "app_paludisme_orig.py")
    if not os.path.exists(baseline):
        pytest.skip("version originale absente (tools/_baseline)")
    res = run_tab3(baseline, stub, {
        "df_cases": df_cases, "gdf_health": gdf_health,
        "years_selected": sorted(ds.df_cases["year"].unique().tolist()),
        "iso3pays": "ner"}, tab_name="tab3")

    if res["error"] and "with tab3" in res["error"]:
        pytest.skip("bloc historique remplacé — diagnostic à re-vérifier")
    assert res["error"] is None, res["error"]

    mr = stub.session_state["model_results"]
    # 1) écrasement multi-années : le modèle voit moins de lignes que les données
    assert len(mr["df_model"]) < len(ds.df_cases)
    # 2) R² in-sample affiché comme métrique principale
    assert mr["metrics"]["r2"] > 0.95
    # 3) découpage présenté comme temporel mais spatial
    assert mr["metrics"]["cv_r2_mean"] > 0.5

    # 4) preuve analytique de la fuite, établie avec la FONCTION RÉELLE de
    #    construction des variables de l'application d'origine (extraite du
    #    module) : `df_model` ne conserve que les composantes principales, les
    #    variables brutes n'y figurent donc pas.
    ns = res["ns"]
    if "create_advanced_features" in ns:
        # la fonction historique attend une colonne `week_num` (et non `week_`)
        _dfc = df_cases.copy()
        _dfc["week_num"] = _dfc["week_"]
        if "Pop_Totale" not in _dfc.columns:
            _dfc = _dfc.merge(
                ds.df_population[["health_area", "Pop_Totale", "Pop_Enfants_0_14",
                                  "Densite_Pop"]], on="health_area", how="left")
        feats = ns["create_advanced_features"](_dfc)
        need = {"incidence_rate", "Pop_Totale", "cases", "cases_ma_2", "cases_lag_1"}
        if need.issubset(feats.columns):
            d = feats.dropna(subset=["incidence_rate", "Pop_Totale"])
            assert len(d) > 10
            # cases_t = incidence_rate x Pop_Totale / 1e4  -> corrélation 1
            recon = d["incidence_rate"] * d["Pop_Totale"] / 1e4
            assert np.corrcoef(recon, d["cases"])[0, 1] > 0.999
            # cases_t = 2 x cases_ma_2 - cases_lag_1        -> corrélation 1
            dm = feats.dropna(subset=["cases_ma_2", "cases_lag_1"])
            recon_ma = 2 * dm["cases_ma_2"] - dm["cases_lag_1"]
            assert np.corrcoef(recon_ma, dm["cases"])[0, 1] > 0.999


# ----------------------------------------------------------------------
# Covariables climatiques : disponibilité réelle (action C5)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def climate_weekly(panel):
    """Climat hebdomadaire synthétique, déterministe et repérable par sa valeur."""
    cd = panel[["health_area", "year", "week_"]].copy()
    cd["precip_api"] = np.arange(len(cd), dtype=float)
    cd["temp_api"] = 25.0
    cd["humidity_api"] = 50.0
    return cd


def test_climate_features_accept_a_real_climate_frame(panel, climate_weekly):
    """
    Non-régression : `COL_YEAR` était utilisé dans `add_climate_features` sans
    être importé, ce qui levait `NameError` dès qu'un vrai climat était fourni.
    Le chemin avec données climatiques n'avait donc jamais pu s'exécuter.
    """
    out = em.add_climate_features(panel, climate_df=climate_weekly)
    for c in ["temp_api", "precip_api", "humidity_api", "precip_lag_4",
              "precip_lag_6", "precip_lag_8", "precip_ma_4_8", "temp_ma_4"]:
        assert c in out.columns, f"colonne climatique absente : {c}"
    assert out["precip_api"].notna().any()


def test_climate_availability_lag_uses_only_published_values(panel, climate_weekly):
    """
    Avec un délai de publication de 2 semaines, la valeur attribuée à la
    semaine t doit être celle de t-2 : c'est la seule qui sera réellement
    connue au moment de prévoir.
    """
    base = em.add_climate_features(panel, climate_df=climate_weekly, availability_lag=0)
    lag2 = em.add_climate_features(panel, climate_df=climate_weekly, availability_lag=2)
    aire = "health_area"
    a0 = base[base[aire] == base[aire].iloc[0]].reset_index(drop=True)
    a2 = lag2[lag2[aire] == lag2[aire].iloc[0]].reset_index(drop=True)

    assert a2["precip_api"].iloc[9] == a0["precip_api"].iloc[7]
    # les semaines sans antécédent publié sont NaN, pas approximées
    assert a2["precip_api"].iloc[:2].isna().all()
    assert a0["precip_api"].iloc[:2].notna().all()


def test_climate_availability_lag_composes_with_epi_lags(panel, climate_weekly):
    """`precip_lag_4` avec un délai de 2 désigne la pluie de t-6, disponible à t."""
    base = em.add_climate_features(panel, climate_df=climate_weekly, availability_lag=0)
    lag2 = em.add_climate_features(panel, climate_df=climate_weekly, availability_lag=2)
    aire = "health_area"
    a0 = base[base[aire] == base[aire].iloc[0]].reset_index(drop=True)
    a2 = lag2[lag2[aire] == lag2[aire].iloc[0]].reset_index(drop=True)
    assert a2["precip_lag_4"].iloc[9] == a0["precip_api"].iloc[3]


def test_climate_availability_lag_defaults_to_prior_behaviour(panel, climate_weekly):
    """Le défaut (0) reproduit exactement le comportement antérieur."""
    aire = "health_area"
    defaut = em.add_climate_features(panel, climate_df=climate_weekly)
    explicite = em.add_climate_features(panel, climate_df=climate_weekly, availability_lag=0)
    pd.testing.assert_frame_equal(defaut, explicite)


def test_climate_availability_lag_reaches_design_matrix(panel, climate_weekly):
    """Le paramètre est bien propagé par `build_design_matrix`."""
    aire = "health_area"
    first = panel[aire].iloc[0]
    base = em.build_design_matrix(panel, climate_df=climate_weekly,
                                  climate_availability_lag=0)[0]
    lag2 = em.build_design_matrix(panel, climate_df=climate_weekly,
                                  climate_availability_lag=2)[0]
    b0 = base[base[aire] == first].reset_index(drop=True)
    b2 = lag2[lag2[aire] == first].reset_index(drop=True)
    assert b2["precip_api"].iloc[9] == b0["precip_api"].iloc[7]

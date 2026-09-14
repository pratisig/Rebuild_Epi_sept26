"""
Diagnostic empirique du pipeline de prédiction LIVRÉ dans app_paludisme.py
(onglet « Modélisation », bloc `with tab3:`).

Tout ce qui est mesuré ici est produit par le code réel de l'application,
exécuté via tools/legacy_harness.py (aucune réécriture de la logique).
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legacy_harness import run_tab3  # noqa: E402
from make_synthetic_data import build_dataset  # noqa: E402
from st_stub import StreamlitStub  # noqa: E402

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Version ORIGINALE (commit c5dde9b) extraite du dépôt : c'est elle que le
# banc d'essai « legacy » doit exécuter, pas le fichier corrigé du dépôt.
APP_PALU = os.path.join(REPO, "tools", "_baseline", "app_paludisme_orig.py")


def prepare_inputs(ds, years):
    """Prépare les variables d'entrée exactement comme le fait la sidebar."""
    df_cases = ds.df_cases.copy()
    df_cases["health_area"] = df_cases["health_area"].astype(str).str.strip().str.lower()

    gdf_health = ds.gdf.copy()
    gdf_health = gdf_health.merge(
        ds.df_population[["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]],
        on="health_area", how="left")

    stub = StreamlitStub(
        selectbox_values={"Algorithme": "RandomForest"},
        slider_values={"Semaines à prévoir": 4, "Seuil alerte": 75},
    )
    ss = stub.session_state
    for key in ["gdf_health", "df_cases", "temp_raster", "flood_raster", "rivers_gdf",
                "precipitation_raster", "humidity_raster", "elevation_raster",
                "model_results", "df_climate_aggregated"]:
        ss[key] = None
    ss["gdf_health"] = gdf_health
    ss["df_cases"] = df_cases
    ss["dfpopulation"] = ds.df_population
    ss["flood_raster"] = None
    ss["elevation_raster"] = None
    ss["rivers_gdf"] = None

    variables = {
        "df_cases": df_cases,
        "gdf_health": gdf_health,
        "years_selected": list(years),
        "iso3pays": "ner",
    }
    return stub, variables


def run_legacy(ds, years, algo="RandomForest"):
    stub, variables = prepare_inputs(ds, years)
    stub.selectbox_values["Algorithme"] = algo
    res = run_tab3(APP_PALU, stub, variables, tab_name="tab3")
    return res


# ======================================================================
def main():
    out = {}

    # ---------- 1) Exécution du pipeline livré (3 ans, config par défaut) ----------
    ds3 = build_dataset(years=(2022, 2023, 2024))
    res = run_legacy(ds3, years=[2022, 2023, 2024])
    print("=" * 78)
    print("1) EXÉCUTION DU CODE LIVRÉ — 3 années sélectionnées (config recommandée)")
    print("=" * 78)
    if res["error"]:
        print("ERREUR:\n", res["error"][:4000])
        out["execution_error"] = res["error"]
    else:
        mr = res["stub"].session_state["model_results"]
        print("model_results keys :", sorted(mr.keys()))
        print("metrics            :", {k: round(float(v), 4) for k, v in mr["metrics"].items()})
        print("n features         :", len(mr["feature_cols"]))
        print("features (ordre)   :", mr["feature_cols"])
        df_model = mr["df_model"]
        df_future = mr["df_future"]
        print("df_model shape     :", df_model.shape)
        print("df_future shape    :", df_future.shape)
        out["legacy_3y"] = {
            "metrics": {k: float(v) for k, v in mr["metrics"].items()},
            "n_features": len(mr["feature_cols"]),
            "feature_cols": list(mr["feature_cols"]),
            "df_model_shape": list(df_model.shape),
            "df_future_shape": list(df_future.shape),
        }

        # ---- 2) COLLAPSUS MULTI-ANNÉES -------------------------------------
        print()
        print("=" * 78)
        print("2) COLLAPSUS MULTI-ANNÉES — groupby(['health_area','week_'])")
        print("=" * 78)
        src = ds3.df_cases.copy()
        src["health_area"] = src["health_area"].astype(str).str.strip().str.lower()
        collapsed = src.groupby(["health_area", "week_"], as_index=False).agg({"cases": "sum"})
        abala = src[src["health_area"] == "abala"]
        abala_c = collapsed[collapsed["health_area"] == "abala"]
        print(f"lignes brutes 'abala' (3 ans)        : {len(abala)}")
        print(f"lignes après groupby 'abala'         : {len(abala_c)}  (max théorique 52)")
        print(f"semaine 1 brute par année            : "
              f"{abala[abala['week_'] == 1][['year', 'cases']].to_dict('records')}")
        s1 = abala_c[abala_c["week_"] == 1]["cases"].iloc[0]
        print(f"semaine 1 après groupby              : {s1}  <-- somme des 3 années")
        n_weeks_src = src["week_"].nunique()
        print(f"semaines distinctes source           : {n_weeks_src} (1..52)")
        print(f"observations modélisation            : {len(df_model)}  "
              f"(vs {len(src)} lignes brutes, soit {len(src) - len(df_model)} écrasées)")
        out["multiyear_collapse"] = {
            "raw_rows_abala_3y": int(len(abala)),
            "rows_after_groupby_abala": int(len(abala_c)),
            "week1_by_year": abala[abala["week_"] == 1][["year", "cases"]].to_dict("records"),
            "week1_after_groupby": int(s1),
            "raw_rows_total": int(len(src)),
            "model_rows_total": int(len(df_model)),
            "rows_lost": int(len(src) - len(df_model)),
        }

        # ---- 3) FUITE TEMPORELLE : ordre des données avant TimeSeriesSplit ---
        print()
        print("=" * 78)
        print("3) « VALIDATION CROISÉE TEMPORELLE » — ordre réel des données")
        print("=" * 78)
        order = df_model[["health_area", "week_num"]].reset_index(drop=True)
        is_sorted_by_time = order["week_num"].is_monotonic_increasing
        print("df_model trié chronologiquement (week_num croissant) ?", bool(is_sorted_by_time))
        first_block = order.head(len(order) // 5)["health_area"].nunique()
        print(f"1er quintile des lignes : {first_block} aires distinctes "
              f"(sur {order['health_area'].nunique()})")
        # ce que TimeSeriesSplit coupe réellement
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        folds = []
        for tr, te in tscv.split(df_model):
            folds.append({
                "train_weeks": sorted(df_model.iloc[tr]["week_num"].unique())[:3],
                "train_week_range": [int(df_model.iloc[tr]["week_num"].min()),
                                     int(df_model.iloc[tr]["week_num"].max())],
                "test_week_range": [int(df_model.iloc[te]["week_num"].min()),
                                    int(df_model.iloc[te]["week_num"].max())],
                "train_areas": int(df_model.iloc[tr]["health_area"].nunique()),
                "test_areas": int(df_model.iloc[te]["health_area"].nunique()),
                "areas_in_both": int(len(set(df_model.iloc[tr]["health_area"])
                                         & set(df_model.iloc[te]["health_area"]))),
                "future_weeks_in_train": int(
                    (df_model.iloc[tr]["week_num"].max()
                     > df_model.iloc[te]["week_num"].min())),
            })
        for i, f in enumerate(folds, 1):
            print(f"  Fold {i}: train weeks {f['train_week_range']} | test weeks "
                  f"{f['test_week_range']} | aires train {f['train_areas']} | "
                  f"aires test {f['test_areas']} | aires communes {f['areas_in_both']} | "
                  f"semaines FUTURES présentes dans train : {bool(f['future_weeks_in_train'])}")
        out["temporal_cv_leakage"] = {
            "df_model_sorted_by_time": bool(is_sorted_by_time),
            "folds": folds,
        }

        # ---- 4) R² IN-SAMPLE vs R² réellement hors-échantillon ---------------
        print()
        print("=" * 78)
        print("4) MÉTRIQUES AFFICHÉES : R² in-sample vs vraie performance temporelle")
        print("=" * 78)
        m = mr["metrics"]
        print(f"R² affiché (fit sur TOUT X, predict sur TOUT X) : {m['r2']:.4f}")
        print(f"R² CV affiché (TimeSeriesSplit sur données triées par AIRE) : "
              f"{m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}")
        out["headline_metrics"] = {k: float(v) for k, v in m.items()}

        # ---- 5) DÉCALAGE TRAIN / SERVE ---------------------------------------
        print()
        print("=" * 78)
        print("5) DÉCALAGE ENTRAÎNEMENT / INFÉRENCE (train-serve skew)")
        print("=" * 78)
        # Les features réellement consommées par le modèle sont celles de l'ACP ;
        # df_future contient les colonnes BRUTES construites pas à pas par la
        # boucle de prévision -> comparaison directe train vs inférence.
        feat = list(mr["pca_info"]["feature_names"]) if mr.get("pca_info") else mr["feature_cols"]
        skew = {}
        for c in feat:
            train_mean = (float(pd.to_numeric(df_model[c], errors="coerce").mean())
                          if c in df_model.columns else float("nan"))
            serv_mean = (float(pd.to_numeric(df_future[c], errors="coerce").mean())
                         if c in df_future.columns else None)
            skew[c] = {"train_mean": train_mean, "forecast_mean": serv_mean}
        printed = 0
        n_flagged = 0
        for c, v in skew.items():
            tm, sm = v["train_mean"], v["forecast_mean"]
            flag = ""
            if sm is None:
                flag = "  <-- ABSENTE des prédictions (remplacée par 0)"
            elif not np.isnan(tm) and abs(tm) > 1e-9 and abs(sm - tm) / max(abs(tm), 1e-9) > 0.5:
                flag = "  <-- écart > 50 %"
            if flag:
                n_flagged += 1
                if printed < 30:
                    print(f"  {c:26s} train={tm:12.4f}  "
                          f"prédiction={sm if sm is None else round(sm, 4)}{flag}")
                    printed += 1
        print(f"  → {n_flagged}/{len(feat)} features présentent un décalage train/inférence")
        out["train_serve_skew"] = {"n_features": len(feat), "n_flagged": n_flagged,
                                   "detail": skew}

        # ---- 6) PRÉDICTIONS FUTURES : dérive ---------------------------------
        print()
        print("=" * 78)
        print("6) PRÉDICTIONS FUTURES vs historique récent (dérive du forecast récursif)")
        print("=" * 78)
        hist_mean = float(df_model["cases"].mean())
        pred_mean = float(df_future["predicted_cases"].mean())
        print(f"moyenne observée (df_model)   : {hist_mean:.2f}")
        print(f"moyenne prédite (df_future)   : {pred_mean:.2f}")
        print(f"ratio prédit/observé          : {pred_mean / max(hist_mean, 1e-9):.3f}")
        # par aire : moyenne des 4 dernières semaines observées de chaque aire
        last4 = (df_model.sort_values(["health_area", "week_num"])
                 .groupby("health_area").tail(4)
                 .groupby("health_area")["cases"].mean())
        fut_avg = df_future.groupby("health_area")["predicted_cases"].mean()
        cmp = pd.concat([last4.rename("obs"), fut_avg.rename("pred")], axis=1).dropna()
        cmp["ratio"] = cmp["pred"] / cmp["obs"].replace(0, np.nan)
        print(f"ratio par aire : médiane {cmp['ratio'].median():.3f} | "
              f"p10 {cmp['ratio'].quantile(.1):.3f} | p90 {cmp['ratio'].quantile(.9):.3f}")
        print(f"aires avec prédiction nulle : "
              f"{int((df_future.groupby('health_area')['predicted_cases'].sum() == 0).sum())}"
              f"/{df_future['health_area'].nunique()}")
        out["forecast_drift"] = {
            "observed_mean": hist_mean, "predicted_mean": pred_mean,
            "ratio": pred_mean / max(hist_mean, 1e-9),
            "area_ratio_median": float(cmp["ratio"].median()),
            "area_ratio_p10": float(cmp["ratio"].quantile(.1)),
            "area_ratio_p90": float(cmp["ratio"].quantile(.9)),
        }

        # ---- 7) PCA : ajustée sur toutes les données -------------------------
        print()
        print("=" * 78)
        print("7) ACP / imputation / centrage ajustés sur TOUTES les données")
        print("=" * 78)
        pca_info = mr["pca_info"]
        print("n_components           :", pca_info["n_components"])
        print("variance cumulée       :", round(float(pca_info["total_variance_explained"]), 4))
        print("features utilisées ACP :", len(pca_info["feature_names"]))
        out["pca"] = {
            "n_components": int(pca_info["n_components"]),
            "total_variance_explained": float(pca_info["total_variance_explained"]),
            "n_features_in_pca": len(pca_info["feature_names"]),
        }

    # ---------- 8) Non-déterminisme de l'ordre des features -------------------
    print()
    print("=" * 78)
    print("8) ORDRE DES FEATURES : list(set(...)) -> non déterministe")
    print("=" * 78)
    import subprocess
    snippet = (
        "import sys,os;sys.path.insert(0,'tools');"
        "cols=['week_num','sin_week','cos_week','cases_lag_1','cases_lag_2','cases_lag_4',"
        "'cases_ma_2','cases_ma_4','growth_rate','Pop_Totale','Densite_Pop','incidence_rate',"
        "'child_risk','demo_pressure','coef_population','spatial_lag','flood_mean'];"
        "print('|'.join(list(set(cols))))"
    )
    orders = set()
    for seed_env in ["0", "1", "2", "3", "4"]:
        env = dict(os.environ, PYTHONHASHSEED=seed_env)
        r = subprocess.run([sys.executable, "-c", snippet], cwd=REPO, env=env,
                           capture_output=True, text=True)
        orders.add(r.stdout.strip())
    print(f"ordres distincts obtenus sur 5 processus : {len(orders)}")
    for o in list(orders)[:3]:
        print("   ", o)
    out["feature_order_variants"] = len(orders)

    with open(os.path.join(REPO, "reports", "legacy_diagnostic.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print("\n→ rapports/legacy_diagnostic.json écrit")


if __name__ == "__main__":
    os.makedirs(os.path.join(REPO, "reports"), exist_ok=True)
    main()

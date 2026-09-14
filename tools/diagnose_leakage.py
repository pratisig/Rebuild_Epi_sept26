"""
Mesure précise de l'impact des fuites de la cible dans le pipeline historique.

Utilise les VRAIES fonctions de `app_paludisme.py`
(``create_advanced_features``, ``create_population_features``) extraites par AST,
appliquées sur un panneau correct, puis évalue sur un découpage **temporel
honnête** en retirant progressivement les variables fautives.

Fuites testées
--------------
1. ``cases_ma_2`` / ``cases_ma_4`` : moyennes mobiles calculées SANS décalage
   (``x.rolling(w).mean()`` au lieu de ``x.shift(1).rolling(w).mean()``) ->
   la moyenne de la semaine t contient ``cases_t``.
2. ``incidence_rate`` / ``child_risk`` / ``demo_pressure`` : calculés avec les
   cas de la semaine courante alors que la population est aussi une variable
   d'entrée -> reconstruction exacte de la cible.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import epimodel as em  # noqa: E402
from epimodel.validation import metrics_summary, temporal_cv_splits  # noqa: E402
from legacy_harness import load_module_context  # noqa: E402
from make_synthetic_data import build_dataset  # noqa: E402
from st_stub import StreamlitStub  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PALU = os.path.join(REPO, "app_paludisme.py")


def main():
    ds = build_dataset()
    ns, _, _ = load_module_context(APP_PALU, tab_name="tab3", stub=StreamlitStub())
    create_advanced_features = ns["create_advanced_features"]
    create_population_features = ns["create_population_features"]

    panel = em.build_panel(ds.df_cases)
    df = panel.copy()
    df["week_num"] = df["week_index"]
    df = create_advanced_features(df)

    static = ds.df_static.merge(
        ds.df_population[["health_area", "Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]],
        on="health_area", how="outer")
    df = df.merge(static, on="health_area", how="left")
    df = create_population_features(df)

    # ── démonstration analytique des deux fuites ────────────────────────
    d0 = df.dropna(subset=["cases_ma_2", "cases_lag_1"])
    recon_ma = 2 * d0["cases_ma_2"] - d0["cases_lag_1"]
    corr_ma = float(np.corrcoef(recon_ma, d0["cases"])[0, 1])
    err_ma = float(np.abs(recon_ma - d0["cases"]).max())
    d1 = df.dropna(subset=["incidence_rate", "Pop_Totale"])
    recon_inc = d1["incidence_rate"] * d1["Pop_Totale"] / 1e4
    corr_inc = float(np.corrcoef(recon_inc, d1["cases"])[0, 1])
    err_inc = float(np.abs(recon_inc - d1["cases"]).max())

    print("DÉMONSTRATION ANALYTIQUE DES FUITES")
    print(f"  cases_t = 2*cases_ma_2 - cases_lag_1  → corrélation {corr_ma:.6f}, "
          f"erreur max {err_ma:.3g}")
    print(f"  cases_t = incidence_rate*Pop/1e4      → corrélation {corr_inc:.6f}, "
          f"erreur max {err_inc:.3g}")

    # ── évaluation sur protocole temporel honnête ───────────────────────
    base = ["cases_lag_1", "cases_lag_2", "cases_lag_4", "growth_rate",
            "sin_week", "cos_week"]
    pop_leak = ["incidence_rate", "child_risk", "demo_pressure"]
    ma_leak = ["cases_ma_2", "cases_ma_4"]
    statics = ["Pop_Totale", "Pop_Enfants_0_14", "Densite_Pop"]

    protocols = {
        "P1_historique_complet (fuites MA + population)":
            base + ma_leak + pop_leak + statics,
        "P2_sans_fuite_moyennes_mobiles":
            base + pop_leak + statics,
        "P3_sans_fuite_population":
            base + ma_leak + statics,
        "P4_sans_aucune_fuite":
            base + statics,
    }

    d = df.dropna(subset=["cases_lag_1"]).reset_index(drop=True)
    y = d["cases"].to_numpy(dtype=float)
    ts = temporal_cv_splits(d["week_index"].to_numpy(), n_splits=5, embargo=4)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline

    def _model():
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                      random_state=42, n_jobs=-1))])

    def _eval(feats, splitter):
        feats = [c for c in feats if c in d.columns]
        X = (d[feats].apply(pd.to_numeric, errors="coerce")
             .replace([np.inf, -np.inf], np.nan))
        yt, yp = [], []
        for tr, te in splitter:
            mdl = _model().fit(X.iloc[tr], y[tr])
            yt.append(y[te])
            yp.append(np.clip(mdl.predict(X.iloc[te]), 0, None))
        yt = np.concatenate(yt)
        yp = np.concatenate(yp)
        m = metrics_summary(yt, yp)
        full = _model().fit(X, y)
        m["r2_in_sample"] = metrics_summary(y, np.clip(full.predict(X), 0, None))["r2"]
        m["n_features"] = len(feats)
        return m

    kf = list(KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(d))))

    print("\nÉVALUATION (RandomForest 300 arbres, mêmes données)")
    print(f"{'protocole':46s} {'MAE':>9s} {'MAE%':>7s} {'R²':>7s} {'R²insamp':>9s}")
    results = {
        "reconstruction_ma_corr": corr_ma,
        "reconstruction_ma_erreur_max": err_ma,
        "reconstruction_incidence_corr": corr_inc,
        "reconstruction_incidence_erreur_max": err_inc,
        "protocols": {},
    }
    for label, feats in protocols.items():
        m = _eval(feats, ts)
        results["protocols"][label] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                       for k, v in m.items()}
        print(f"{label:46s} {m['mae']:9.1f} {m['mae_pct_mean']:6.1f}% "
              f"{m['r2']:7.3f} {m['r2_in_sample']:9.3f}")

    # référence : découpage aléatoire (comme l'ancien `train_test_split`)
    m_rand = _eval(protocols["P4_sans_aucune_fuite"], kf)
    results["reference_split_aleatoire_sans_fuite"] = {
        k: float(v) for k, v in m_rand.items()}
    print(f"{'[réf] split aléatoire, sans fuite':46s} {m_rand['mae']:9.1f} "
          f"{m_rand['mae_pct_mean']:6.1f}% {m_rand['r2']:7.3f} "
          f"{m_rand['r2_in_sample']:9.3f}")

    out = os.path.join(REPO, "reports", "leakage_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()

"""
Construction du panneau épidémiologique.

Deux défauts majeurs du pipeline historique sont corrigés ici :

1. **Écrasement multi-années** — l'ancien code faisait
   ``groupby(["health_area", "week_"])`` : la semaine 1 de 2022, 2023 et 2024
   étaient *additionnées* en une seule observation. On construit désormais un
   index temporel continu ``week_index`` (ISO-8601) qui distingue les années.

2. **Panneau troué** — les semaines sans cas notifié sont absentes des
   linelists, donc ``shift(1)`` ne correspondait pas à « la semaine précédente »
   mais à « la ligne précédente » (parfois 5 semaines plus tard). On complète
   explicitement la grille (aires x semaines) avec des zéros.

Les noms de colonnes historiques sont conservés :
``health_area``, ``week_``, ``year``, ``cases``, ``deaths``, ``period``.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Colonnes du contrat de données de l'application (à ne pas renommer)
COL_AREA = "health_area"
COL_WEEK = "week_"          # semaine épidémiologique ISO (1..53)
COL_YEAR = "year"           # année ISO
COL_CASES = "cases"
COL_DEATHS = "deaths"
COL_PERIOD = "period"       # "AAAA-Snn"
COL_WIDX = "week_index"     # index temporel continu (nombre de semaines depuis t0)

WEEKS_PER_YEAR = 52.1775    # 365.2425 / 7 : moyenne ISO (gère les semaines 53)


# ----------------------------------------------------------------------
# Index temporel ISO-8601
# ----------------------------------------------------------------------
def iso_week_to_date(year: int, week: int) -> date:
    """
    Lundi de la semaine ISO ``week`` de l'année ISO ``year``.

    Corrige l'ancien ``week_to_date_range`` qui utilisait
    ``1er_janvier + timedelta(weeks=n-1)`` — approximation qui décale d'une
    semaine sur ~3 années sur 4 et ignore totalement les semaines 53.
    """
    year = int(year)
    week = int(week)
    iso_max = date(year, 12, 28).isocalendar()[1]   # dernier n° de semaine ISO de l'année
    week = max(1, min(week, iso_max))
    return date.fromisocalendar(year, week, 1)


def iso_week_index(year, week, origin_year: Optional[int] = None) -> np.ndarray:
    """
    Index temporel continu en semaines depuis l'origine.

    Implémentation vectorisée : on convertit (année ISO, semaine) en date puis
    on compte le nombre de jours depuis l'origine. Gère correctement les
    années multiples et les semaines 53.
    """
    y = pd.to_numeric(pd.Series(np.asarray(year)), errors="coerce").astype("Int64")
    w = pd.to_numeric(pd.Series(np.asarray(week)), errors="coerce").astype("Int64")
    valid = y.notna() & w.notna()

    # nombre maximal de semaines ISO par année concernée
    iso_max = y.map(lambda v: date(int(v), 12, 28).isocalendar()[1] if pd.notna(v) else pd.NA)
    w = np.minimum(w, iso_max)

    origin = iso_week_to_date(int(origin_year), 1) if origin_year is not None else None

    dates = pd.Series(pd.NaT, index=y.index, dtype="datetime64[ns]")
    for i in y.index[valid]:
        dates.iloc[i] = pd.Timestamp(iso_week_to_date(int(y.iloc[i]), int(w.iloc[i])))

    if origin is None:
        origin = dates.min()
    else:
        origin = pd.Timestamp(origin)

    idx = (dates - origin).dt.days / 7.0
    out = np.full(len(y), np.nan, dtype=float)
    out[valid.to_numpy()] = idx[valid].to_numpy()
    return out


def week_index_to_iso(week_index, origin_year: int, origin_week: int = 1
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse de :func:`iso_week_index` -> (année ISO, semaine ISO)."""
    origin = iso_week_to_date(origin_year, origin_week)
    wi = np.asarray(week_index, dtype=float)
    years, weeks = [], []
    for v in wi:
        if np.isnan(v):
            years.append(np.nan)
            weeks.append(np.nan)
            continue
        d = origin + timedelta(days=int(round(v)) * 7)
        iso = d.isocalendar()
        years.append(int(iso[0]))
        weeks.append(int(iso[1]))
    return np.asarray(years, dtype=float), np.asarray(weeks, dtype=float)


# ----------------------------------------------------------------------
# Construction du panneau
# ----------------------------------------------------------------------
def build_panel(df_cases: pd.DataFrame,
                years: Optional[Sequence[int]] = None,
                areas: Optional[Iterable[str]] = None,
                complete_grid: bool = True,
                fill_zero: bool = True) -> pd.DataFrame:
    """
    Construit le panneau de modélisation.

    Paramètres
    ----------
    df_cases : DataFrame brut avec au minimum ``health_area``, ``week_``, ``cases``
        (``year`` et ``deaths`` optionnels mais fortement recommandés).
    years : restreindre à ces années ISO.
    areas : restreindre à ces aires de santé.
    complete_grid : compléter la grille aires x semaines (recommandé).
    fill_zero : les cellules absentes deviennent ``cases = 0``.

    Retour
    ------
    DataFrame trié par (``health_area``, ``week_index``) avec les colonnes
    ``week_index`` (temps continu), ``period`` et ``week_frac`` (position dans
    l'année, 0..1) en plus des colonnes d'origine.
    """
    df = df_cases.copy()

    # ── normalisation minimale (mêmes règles que l'application) ────────
    df[COL_AREA] = df[COL_AREA].astype(str).str.strip().str.lower()
    df[COL_WEEK] = pd.to_numeric(df[COL_WEEK], errors="coerce")
    df[COL_CASES] = pd.to_numeric(df[COL_CASES], errors="coerce").fillna(0)
    df = df[df[COL_WEEK].notna()].copy()
    df[COL_WEEK] = df[COL_WEEK].astype(int)

    if COL_DEATHS not in df.columns:
        df[COL_DEATHS] = 0
    df[COL_DEATHS] = pd.to_numeric(df[COL_DEATHS], errors="coerce").fillna(0)

    # ── année ISO ───────────────────────────────────────────────────────
    if COL_YEAR in df.columns and df[COL_YEAR].notna().any():
        df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce")
        # une année manquante est déduite du champ `period` si possible
        if df[COL_YEAR].isna().any() and COL_PERIOD in df.columns:
            inferred = (df[COL_PERIOD].astype(str).str.slice(0, 4)
                        .pipe(pd.to_numeric, errors="coerce"))
            df[COL_YEAR] = df[COL_YEAR].fillna(inferred)
        df[COL_YEAR] = df[COL_YEAR].fillna(df[COL_YEAR].mode().iloc[0]
                                           if df[COL_YEAR].notna().any() else 2024)
        df[COL_YEAR] = df[COL_YEAR].astype(int)
    else:
        df[COL_YEAR] = int(pd.Timestamp.today().year)

    if years is not None:
        df = df[df[COL_YEAR].isin(list(years))].copy()
    if areas is not None:
        wanted = {str(a).strip().lower() for a in areas}
        df = df[df[COL_AREA].isin(wanted)].copy()

    # ── agrégation sur le VRAI couple (aire, année ISO, semaine ISO) ─────
    agg = {COL_CASES: "sum", COL_DEATHS: "sum"}
    extra_cols = [c for c in df.columns
                  if c not in (COL_AREA, COL_WEEK, COL_YEAR, COL_CASES, COL_DEATHS,
                               COL_PERIOD, COL_WIDX)]
    for c in extra_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            agg[c] = "mean"
    panel = (df.groupby([COL_AREA, COL_YEAR, COL_WEEK], as_index=False)
               .agg(agg))

    # ── grille complète (aires x semaines) ──────────────────────────────
    if complete_grid:
        all_areas = pd.DataFrame({COL_AREA: sorted(panel[COL_AREA].unique())})
        # plage temporelle couverte, exprimée en (année, semaine) réelles
        y_min, y_max = int(panel[COL_YEAR].min()), int(panel[COL_YEAR].max())
        grid_rows = []
        for y in range(y_min, y_max + 1):
            w_max = date(y, 12, 28).isocalendar()[1]
            for w in range(1, w_max + 1):
                grid_rows.append((y, w))
        grid_time = pd.DataFrame(grid_rows, columns=[COL_YEAR, COL_WEEK])
        grid = all_areas.assign(_k=1).merge(grid_time.assign(_k=1), on="_k").drop(columns="_k")
        panel = grid.merge(panel, on=[COL_AREA, COL_YEAR, COL_WEEK], how="left")
        if fill_zero:
            panel[COL_CASES] = panel[COL_CASES].fillna(0)
            panel[COL_DEATHS] = panel[COL_DEATHS].fillna(0)
        for c in extra_cols:
            if c in panel.columns:
                panel[c] = pd.to_numeric(panel[c], errors="coerce")

    # ── index temporel continu ──────────────────────────────────────────
    origin_year = int(panel[COL_YEAR].min())
    panel[COL_WIDX] = iso_week_index(panel[COL_YEAR], panel[COL_WEEK],
                                     origin_year=origin_year).astype(float)
    panel = panel.sort_values([COL_AREA, COL_WIDX]).reset_index(drop=True)

    # ── champs dérivés stables ─────────────────────────────────────────
    panel[COL_PERIOD] = (panel[COL_YEAR].astype(int).astype(str) + "-S"
                         + panel[COL_WEEK].astype(int).astype(str).str.zfill(2))
    panel["week_frac"] = (panel[COL_WEEK] - 1) / WEEKS_PER_YEAR
    panel.attrs["origin_year"] = origin_year
    panel.attrs["origin_week"] = 1
    return panel


def panel_summary(panel: pd.DataFrame) -> dict:
    """Statistiques de contrôle qualité du panneau."""
    n_areas = int(panel[COL_AREA].nunique())
    n_weeks = int(panel[COL_WIDX].nunique())
    expected = n_areas * n_weeks
    return {
        "n_areas": n_areas,
        "n_weeks": n_weeks,
        "n_rows": int(len(panel)),
        "n_rows_expected": expected,
        "completeness_pct": round(100.0 * len(panel) / max(expected, 1), 2),
        "n_zero_rows": int((panel[COL_CASES] == 0).sum()),
        "zero_pct": round(100.0 * float((panel[COL_CASES] == 0).mean()), 2),
        "total_cases": float(panel[COL_CASES].sum()),
        "week_index_min": float(panel[COL_WIDX].min()),
        "week_index_max": float(panel[COL_WIDX].max()),
        "years": sorted(int(v) for v in panel[COL_YEAR].dropna().unique()),
    }

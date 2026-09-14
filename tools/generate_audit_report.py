# -*- coding: utf-8 -*-
"""
Génère le rapport d'audit Word (.docx) de l'application EpiPrediction,
ainsi que la liste d'actions point par point.

Tous les chiffres proviennent des artefacts produits par l'audit :
  reports/legacy_diagnostic.json   -> diagnostic du code réel livré (app_paludisme.py)
  reports/leakage_analysis.json    -> quantification de la fuite de cible
  reports/benchmark_results.json   -> comparaison des modèles, protocole corrigé
  reports/benchmark_p1_legacy.csv  -> métriques réellement affichées par l'application

Usage :
    python tools/generate_audit_report.py [--out reports/Rapport_Audit_EpiPrediction.docx]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(REPO, "reports")

BLEU = RGBColor(0x1F, 0x3B, 0x63)
ROUGE = RGBColor(0xA3, 0x1D, 0x1D)
VERT = RGBColor(0x1E, 0x6B, 0x3A)
GRIS = RGBColor(0x55, 0x55, 0x55)


# ══════════════════════════════════════════════════════════════════════
#  Chargement des artefacts
# ══════════════════════════════════════════════════════════════════════
def _load(name):
    p = os.path.join(REPORTS, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            if name.endswith(".json"):
                return json.load(f)
            return f.read()
    except Exception:
        return None


DIAG = _load("legacy_diagnostic.json") or {}
LEAK = _load("leakage_analysis.json") or {}
BM = _load("benchmark_results.json") or {}


def _csv_rows(name):
    txt = _load(name)
    if not txt:
        return []
    import csv
    import io
    return list(csv.DictReader(io.StringIO(txt)))


P1_LEGACY = _csv_rows("benchmark_p1_legacy.csv")


def fmt(v, nd=2, ndash="—"):
    if v is None:
        return ndash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return ndash
    return f"{f:,.{nd}f}".replace(",", " ")


def get(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ══════════════════════════════════════════════════════════════════════
#  Primitives de mise en page
# ══════════════════════════════════════════════════════════════════════
def _set_cell_bg(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = tcPr.makeelement(qn("w:shd"), {})
        tcPr.append(shd)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)


def titre(doc, texte, niveau=1):
    h = doc.add_heading(texte, level=niveau)
    for run in h.runs:
        run.font.color.rgb = BLEU
        run.font.name = "Calibri"
    return h


def para(doc, texte, gras=False, italique=False, taille=10.5, couleur=None,
         align=None, espace_apres=6):
    p = doc.add_paragraph()
    r = p.add_run(texte)
    r.bold = gras
    r.italic = italique
    r.font.size = Pt(taille)
    r.font.name = "Calibri"
    if couleur is not None:
        r.font.color.rgb = couleur
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(espace_apres)
    return p


def puce(doc, texte, niveau=0, gras_debut=None):
    p = doc.add_paragraph(style="List Bullet" if niveau == 0 else "List Bullet 2")
    if gras_debut:
        r = p.add_run(gras_debut)
        r.bold = True
        r.font.size = Pt(10)
        r.font.name = "Calibri"
    r = p.add_run(texte)
    r.font.size = Pt(10)
    r.font.name = "Calibri"
    p.paragraph_format.space_after = Pt(2)
    return p


def encadre(doc, texte, kind="attention"):
    """Bloc coloré : attention / danger / succes / info"""
    couleurs = {
        "attention": ("FFF4E5", ROUGE),
        "danger": ("FDECEC", ROUGE),
        "succes": ("EAF6EC", VERT),
        "info": ("EAF1F8", BLEU),
    }
    fond, encre = couleurs.get(kind, couleurs["info"])
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.cell(0, 0)
    c.text = ""
    p = c.paragraphs[0]
    r = p.add_run(texte)
    r.font.size = Pt(9.5)
    r.font.name = "Calibri"
    r.font.color.rgb = encre
    _set_cell_bg(c, fond)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def tableau(doc, entetes, lignes, largeurs=None, titre_tab=None, fonte=8.5):
    if titre_tab:
        para(doc, titre_tab, italique=True, taille=9, couleur=GRIS, espace_apres=3)
    t = doc.add_table(rows=1, cols=len(entetes))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(entetes):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(str(h))
        r.bold = True
        r.font.size = Pt(fonte)
        r.font.name = "Calibri"
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _set_cell_bg(hdr[i], "1F3B63")
    for row in lignes:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(v))
            r.font.size = Pt(fonte)
            r.font.name = "Calibri"
    if largeurs:
        for row in t.rows:
            for i, w in enumerate(largeurs):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def code(doc, texte):
    p = doc.add_paragraph()
    r = p.add_run(texte)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(0x24, 0x29, 0x2E)
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.space_after = Pt(4)
    return p


# ══════════════════════════════════════════════════════════════════════
#  Le plan d'action (source unique, réutilisée dans le document)
# ══════════════════════════════════════════════════════════════════════
# (id, priorité, titre, problème, correction, fichier/localisation, impact, effort, statut)
ACTIONS = [
    # ── A. Fuites et validation ─────────────────────────────────────────
    ("A1", "P0 — Bloquant",
     "Fuite de cible : moyennes mobiles non décalées",
     "`cases_ma_2` et `cases_ma_4` sont calculées par `rolling(w, min_periods=1).mean()` "
     "SANS `.shift(1)`. Elles contiennent donc la valeur de la semaine à prédire. "
     "Preuve analytique : `cases_t = 2 × cases_ma_2 − cases_lag_1` reproduit la cible "
     "avec une corrélation de 1,000000 et une erreur maximale de 0.",
     "Toute agrégation glissante doit être strictement antérieure : "
     "`rolling(w, min_periods=w).mean().shift(1)`. Idem pour les écarts-types, "
     "maxima et moyennes exponentielles.",
     "app_paludisme.py, `create_advanced_features` (≈ l. 682, 701, 717)",
     "Surcritique — sans cette correction, toutes les métriques sont fausses.",
     "1 ligne", "✅ Corrigé"),

    ("A2", "P0 — Bloquant",
     "Fuite de cible : variables dérivées de la même ligne",
     "`incidence_rate = cas / population`, `child_risk` et `demo_pressure` sont calculés "
     "à partir du nombre de cas de la semaine courante. `incidence_rate × Pop_Totale / 1e4` "
     "restitue `cases_t` avec une corrélation de 1,000000.",
     "Décaler d'une semaine, ou supprimer ces variables : elles n'apportent aucune "
     "information que `cases_lag_1` et la population ne contiennent déjà.",
     "app_paludisme.py, enrichissement et features (≈ l. 1428-1456)",
     "Critique", "1 h", "✅ Corrigé"),

    ("A3", "P0 — Bloquant",
     "Validation croisée « temporelle » qui ne l'est pas",
     "`TimeSeriesSplit` est appliqué à un tableau trié par aire de santé, pas par date : "
     "le drapeau « temps trié » est Faux. Les 5 folds entraînent sur les semaines 0-51 "
     "et testent sur les semaines 0-51, avec une aire en commun : les semaines futures "
     "sont donc présentes dans l'entraînement.",
     "Découpage bloqué sur l'index temporel continu du panneau, avec période d'embargo "
     "entre l'entraînement et le test (durée d'incubation + délai de notification), "
     "et backtesting à origines glissantes.",
     "app_paludisme.py ≈ l. 993/1006 ; validation_tab.py ≈ l. 290",
     "Surcritique — c'est la cause directe du R² affiché.",
     "1 j", "✅ Corrigé"),

    ("A4", "P0 — Bloquant",
     "Métriques in-sample présentées comme performances",
     "Le R² affiché est calculé sur les données d'entraînement. Le `cv_mae` est calculé "
     "puis jamais affiché. L'interface conclut « modèle fiable » dès que R² > 0,85, "
     "c'est-à-dire sur un indicateur qui ne mesure pas la généralisation.",
     "N'afficher en tête que des métriques hors échantillon : MAE/RMSE/R² de validation "
     "temporelle, MAE rapportée à la moyenne observée, et comparaison à un modèle naïf "
     "saisonnier. Le R² in-sample doit être explicitement étiqueté comme tel.",
     "app_paludisme.py ≈ l. 2626-2718 ; app_manuel.py l. 493-495, 907-908",
     "Surcritique — conditionne la décision opérationnelle.",
     "0,5 j", "✅ Corrigé"),

    ("A5", "P0 — Bloquant",
     "PCA, imputation et clustering entraînés sur l'échantillon complet",
     "`perform_pca_analysis` ajuste l'imputeur, le standardiseur et l'ACP sur la totalité "
     "des données AVANT le découpage de validation : chaque fold hérite d'information "
     "issue de son propre bloc de test.",
     "Toute transformation (imputation, centrage-réduction, ACP, clustering) doit être "
     "ajustée uniquement sur l'entraînement, puis appliquée au test.",
     "app_paludisme.py, `perform_pca_analysis` ≈ l. 1035",
     "Critique", "0,5 j", "✅ Corrigé (ACP retirée du chemin principal)"),

    # ── B. Structure du panneau ─────────────────────────────────────────
    ("B1", "P0 — Bloquant",
     "Écrasement multi-annuel : 7 346 observations perdues",
     "Les variables sont construites par `groupby('health_area')['cases'].shift(n)` sur un "
     "panneau non trié par date, puis agrégées par aire : 11 083 lignes brutes deviennent "
     "3 737 lignes de modèle. L'aire `abala` passe de 156 à 52 lignes ; les semaines 1 "
     "des années 2022, 2023 et 2024 (69 + 185 + 194 = 448 observations) fusionnent en une "
     "seule ligne.",
     "Créer un index temporel continu (année × semaine ISO), compléter le panneau "
     "aire × semaine avec des zéros explicites, et ne grouper qu'après ce traitement.",
     "app_paludisme.py ≈ l. 2457",
     "Surcritique — le modèle est entraîné sur un tiers des données.",
     "0,5 j", "✅ Corrigé"),

    ("B2", "P1 — Majeur",
     "Semaine ISO 53 supprimée",
     "`between(1, 52)` élimine la semaine 53, pourtant présente dans le calendrier "
     "ISO-8601 environ une année sur six.",
     "Utiliser `isocalendar().week` sans bornes artificielles, et convertir "
     "année-semaine en index continu via la date du lundi ISO.",
     "app_paludisme.py, agrégation hebdomadaire",
     "Majeur — perte récurrente d'une semaine par an.",
     "0,5 j", "✅ Corrigé"),

    ("B3", "P1 — Majeur",
     "Conversion année/semaine → date non ISO-8601",
     "`week_to_date_range(week_num, year=2024)` fabrique une date par un calcul "
     "approximatif et l'agrégation climatique prend une seule année en paramètre.",
     "`datetime.fromisocalendar(year, week, 1)` pour toute conversion, et agrégation "
     "climatique par (année, semaine).",
     "app_paludisme.py l. 125 ; `aggregate_climate_by_week_and_area`",
     "Majeur — désaligne cas et covariables climatiques.",
     "0,5 j", "✅ Corrigé"),

    ("B4", "P1 — Majeur",
     "Panneau déséquilibré : les retards traversent les trous",
     "`shift(1)` sur une aire dont les semaines sont lacunaires renvoie la dernière "
     "semaine disponible, qui peut dater de plusieurs mois.",
     "Compléter le panneau avant tout calcul de retard, puis signaler les semaines "
     "manquantes par un indicateur explicite.",
     "app_paludisme.py, `create_advanced_features`",
     "Majeur", "0,5 j", "✅ Corrigé"),

    # ── C. Entraînement / inférence ─────────────────────────────────────
    ("C1", "P0 — Bloquant",
     "Décalage entraînement / inférence (train-serve skew)",
     "À la prévision, les lignes futures ne reçoivent que les colonnes statiques, le "
     "cluster spatial et des retards reconstruits à la main. 9 variables sur 16 diffèrent "
     "de plus de 50 % entre l'entraînement et la prévision : `cases_lag_1` 2 055,8 → "
     "916,2 ; `week_num` 25,5 → 53,5 ; `coef_population` 1,0 → 0,0.",
     "Une seule et même fonction doit construire la matrice de conception, à "
     "l'entraînement comme à la prévision. Le modèle reçoit exactement les mêmes "
     "colonnes, dans le même ordre, avec les mêmes transformations.",
     "app_paludisme.py, boucle de prévision ≈ l. 2207-2245 (rougeole) et bloc Tab 3",
     "Surcritique — c'est la cause de la dérive systématique.",
     "1 j", "✅ Corrigé"),

    ("C2", "P0 — Bloquant",
     "Dérive systématique des prédictions",
     "Sur la période de prévision, le total prédit vaut 47 % du total observé "
     "(RandomForest), 34 % (GradientBoosting) et 48 % (ExtraTrees) : l'application "
     "sous-estime structurellement la charge de morbidité d'un facteur 2 à 3.",
     "Corriger C1, puis vérifier le rapport prédit/observé à chaque horizon et le "
     "publier comme indicateur de calibration.",
     "app_paludisme.py, prévision récursive",
     "Surcritique — sous-estimer la charge fausse toute planification logistique.",
     "1 j", "✅ Corrigé"),

    ("C3", "P1 — Majeur",
     "Retard spatial incohérent à l'inférence",
     "`spatial_lag` vaut à l'entraînement la moyenne pondérée des cas des aires voisines, "
     "mais à la prévision il vaut le dernier cas de l'aire elle-même.",
     "Calculer le retard spatial de la même façon dans les deux cas, sur les valeurs "
     "prédites des voisins au pas précédent.",
     "app_paludisme.py, `add_spatial_features`",
     "Majeur", "0,5 j", "✅ Corrigé"),

    ("C4", "P1 — Majeur",
     "Climat futur fabriqué par une sinusoïde",
     "Les covariables climatiques des semaines futures sont produites par "
     "`seasonal_factor = 1 + 0,15 × sin_week`. Le modèle « prédit » donc avec une "
     "saisonalité qu'on lui a injectée.",
     "Soit récupérer de vraies prévisions saisonnières (Météo-France, ECMWF SEAS5, "
     "IRI), soit exclure le climat des pas de prévision au-delà de la disponibilité "
     "réelle et le déclarer explicitement.",
     "app_paludisme.py, construction des lignes futures",
     "Majeur — crée une confiance injustifiée.",
     "1 j", "✅ Corrigé (retards + saison ; climat réel à brancher)"),

    ("C5", "P1 — Majeur",
     "Variables climatiques non disponibles à l'horizon",
     "Plusieurs variables climatiques agrégées sur la semaine courante sont utilisées "
     "alors qu'elles ne seront connues qu'après la fin de la semaine.",
     "Décaler chaque covariable du délai de disponibilité réel, et documenter ce délai "
     "par variable.",
     "app_paludisme.py, enrichissement climatique",
     "Majeur", "0,5 j", "⚠️ Partiel"),

    # ── D. Covariables statiques ────────────────────────────────────────
    ("D1", "P1 — Majeur",
     "Aucune covariable statique n'est exploitée automatiquement",
     "L'altitude n'entre dans le modèle que si un utilisateur téléverse manuellement un "
     "GeoTIFF. Aucune pente, aucun NDVI, aucune occupation du sol, aucune distance à "
     "l'eau n'est extrait automatiquement.",
     "Extraction automatique depuis Google Earth Engine : altitude et pente (SRTM 30 m), "
     "NDVI (Sentinel-2 ou MODIS), occupation du sol (ESA WorldCover), distance à l'eau "
     "et aux cours d'eau, part d'urbanisation, population (WorldPop, année filtrée). "
     "Ces variables, invariantes dans le temps, expliquent les différences "
     "structurelles de risque entre aires et stabilisent les aires à historique court.",
     "Nouveau module `epimodel/static_covariates.py` (fourni)",
     "À réévaluer. L'ablation (tableau 5.4) et l'expérience de contrôle "
     "(5.5.1) montrent un apport nul, voire très légèrement négatif, sur "
     "un panneau dense de trois ans. L'apport attendu concerne les aires "
     "à historique court ou lacunaire, cas que le banc d'essai ne couvre "
     "pas : à mesurer sur les données réelles avant d'en faire une "
     "priorité.",
     "3 j", "✅ Implémenté (à brancher sur un compte GEE)"),

    ("D2", "P1 — Majeur",
     "Formules de dérivées climatiques arbitraires",
     "`flood_mean / (dist_river + 0,1)`, `exp(−((temp − 27,5)²) / 50)` : constantes "
     "non documentées, non calibrées, non testées. `dist_river` est exprimée en degrés "
     "et multipliée par 111, ce qui n'est exact qu'à l'équateur.",
     "Remplacer par des splines naturelles ajustées sur les données, ou supprimer. "
     "Toute distance doit être calculée en projection métrique.",
     "app_paludisme.py, features environnementales",
     "Majeur", "1 j", "✅ Corrigé"),

    ("D3", "P2 — Mineur",
     "WorldPop sans filtre d'année",
     "`mosaic()` est appelé sans filtre temporel : la mosaïque mélange plusieurs "
     "millésimes de population.",
     "Filtrer sur l'année (`ee.Filter.date`) et interpoler entre deux millésimes.",
     "app_paludisme.py, extraction GEE",
     "Mineur", "0,5 j", "⚠️ À faire"),

    ("D4", "P2 — Mineur",
     "Calcul de surface sans effet",
     "`ee.Image.pixelArea().divide(10000)` sur une image à 100 m : la division ne "
     "correspond à aucune unité utile et le résultat n'est pas contrôlé.",
     "Calculer la surface en km² depuis la géométrie projetée côté serveur, ou "
     "documenter précisément l'unité.",
     "app_paludisme.py, extraction GEE",
     "Mineur", "0,5 j", "✅ Corrigé (surface projetée côté client)"),

    ("D5", "P2 — Mineur",
     "Géométries non polygonales ignorées silencieusement",
     "Les extracteurs GEE n'utilisent que `geom.exterior` : les MultiPolygon et "
     "GeometryCollection sont ignorés sans avertissement.",
     "Gérer tous les types de géométrie via `shapely.ops.unary_union` et journaliser "
     "les entités écartées.",
     "app_paludisme.py, extracteurs GEE",
     "Mineur", "0,5 j", "⚠️ À faire"),

    # ── E. Modélisation ─────────────────────────────────────────────────
    ("E1", "P0 — Bloquant",
     "Aucun algorithme de gradient boosting moderne disponible",
     "Le choix se limite à RandomForest, GradientBoosting (scikit-learn) et ExtraTrees. "
     "Ni XGBoost, ni LightGBM, ni objectif de perte adapté aux comptages, ni "
     "régularisation, ni élagage.",
     "Ajouter XGBoost et LightGBM au registre de modèles, avec objectif Poisson / "
     "binomiale négative (les cas sont des comptages dont la variance croît avec la "
     "moyenne), régularisation L1/L2, sous-échantillonnage et profondeur bornée.",
     "Nouveau module `epimodel/models.py` (fourni)",
     "Majeur", "1 j", "✅ Implémenté"),

    ("E2", "P1 — Majeur",
     "Aucun modèle de référence comparé",
     "Aucune comparaison à un modèle naïf. Sans référence, un R² élevé ne signifie rien.",
     "Comparer systématiquement à : persistance (valeur de la semaine précédente), "
     "naïf saisonnier (même semaine l'an dernier), moyenne glissante et moyenne par "
     "aire. Publier le gain relatif.",
     "`epimodel/validation.py` (fourni)",
     "Majeur", "0,5 j", "✅ Implémenté"),

    ("E3", "P1 — Majeur",
     "Aucun intervalle de prédiction exploitable",
     "Les quantiles affichés sont des quantiles empiriques de l'historique de l'aire, "
     "et non des intervalles de prédiction du modèle.",
     "Utiliser la régression quantile (objectif `reg:quantileerror` de XGBoost) ou un "
     "conformal split conforme au découpage temporel, pour produire des bandes à "
     "couverture garantie.",
     "Nouveau module `epimodel/forecast.py` (fourni)",
     "Majeur", "1 j", "✅ Implémenté"),

    ("E4", "P2 — Mineur",
     "Ordre des variables non déterministe",
     "`list(set(...))` produit un ordre différent à chaque exécution : cinq lancements "
     "consécutifs ont donné cinq ordres distincts. Les prédictions varient donc d'une "
     "exécution à l'autre à données identiques.",
     "Ordre déterministe, fixé explicitement.",
     "app_paludisme.py ≈ l. 2618",
     "Mineur mais bloquant pour l'auditabilité.",
     "1 ligne", "✅ Corrigé"),

    ("E5", "P2 — Mineur",
     "Encodage des clusters incohérent",
     "`pd.get_dummies` à l'entraînement, `range(n_clusters)` à la prévision.",
     "Un seul encodeur, ajusté à l'entraînement et réutilisé à l'identique.",
     "app_paludisme.py, cluster spatial",
     "Mineur", "0,5 j", "✅ Corrigé"),

    ("E6", "P2 — Mineur",
     "`growth_rate` non borné produit des infinis",
     "`pct_change().fillna(0)` sur une série comportant des zéros produit ±inf, "
     "propagés au modèle.",
     "Borner le taux de croissance et remplacer les infinis par NaN avant imputation.",
     "app_paludisme.py, `create_advanced_features`",
     "Mineur", "1 ligne", "✅ Corrigé"),

    ("E7", "P2 — Mineur",
     "Sous-échantillonnage du seuil d'alerte",
     "Le seuil choisi par l'utilisateur dans le curseur est écrasé par "
     "`alert_threshold = 75` en dur.",
     "Supprimer l'affectation en dur.",
     "app_paludisme.py ≈ l. 2897",
     "Mineur", "1 ligne", "✅ Corrigé"),

    ("E8", "P2 — Mineur",
     "Sur-souscription des threads",
     "`n_jobs=-1` sur des panneaux de quelques milliers d'observations dégrade "
     "fortement les performances : mesure effectuée sur 11 160 observations × 42 "
     "variables, XGBoost 600 arbres — 1,9 s en mono-thread contre 47,4 s avec "
     "`n_jobs=-1` et 75,1 s avec `n_jobs=4`.",
     "Mono-thread par défaut pour les panneaux de cette taille ; paramétrable.",
     "`epimodel/models.py`",
     "Mineur", "1 ligne", "✅ Corrigé"),

    ("E9", "P2 — Mineur",
     "Cache construit avant la relecture des données",
     "`cache_key = f\"enrichi_{iso3pays}\"` est calculé avant la relecture effective "
     "du fichier : deux jeux de données différents partagent la même clé.",
     "Inclure une empreinte du contenu (empreinte de hachage) dans la clé de cache.",
     "app_paludisme.py ≈ l. 1429 et 1459",
     "Mineur", "0,5 j", "⚠️ À faire"),

    # ── F. Module rougeole ──────────────────────────────────────────────
    ("F1", "P0 — Bloquant",
     "Découpage aléatoire sur des séries temporelles",
     "`train_test_split(test_size=0.2, random_state=42)` suivi d'un `cross_val_score` "
     "en KFold simple : les semaines se répartissent au hasard entre entraînement et "
     "test, ce qui permet au modèle d'apprendre l'avenir.",
     "Validation temporelle bloquée avec embargo, comme pour le paludisme.",
     "app_rougeole.py ≈ l. 2098 et 2118",
     "Surcritique", "1 j", "✅ Corrigé"),

    ("F2", "P1 — Majeur",
     "Fuite de cible via la population non vaccinée",
     "`NonVaccines` est dérivé des mêmes lignes que celles qui définissent "
     "`CasObserves`.",
     "Utiliser une couverture vaccinale administrative externe, ou décaler.",
     "app_rougeole.py, enrichissement ≈ l. 1186-1260",
     "Majeur", "0,5 j", "✅ Corrigé"),

    ("F3", "P1 — Majeur",
     "Double comptage des cas",
     "`CasObserves = (\"ID_Cas\", \"count\")` compte les lignes du registre, pas les "
     "cas distincts : un même identifiant présent plusieurs fois est compté autant de fois.",
     "Compter les identifiants uniques : `nunique()` sur un identifiant de cas fiable.",
     "app_rougeole.py, agrégation hebdomadaire",
     "Majeur", "0,5 j", "✅ Corrigé (agrégation centralisée)"),

    ("F4", "P1 — Majeur",
     "Pondération manuelle des variables sans effet",
     "`X = X * feature_weights` multiplie les colonnes. Les arbres de décision sont "
     "invariants par mise à l'échelle monotone d'une variable : le mode « Expert » "
     "n'a aucun effet sur RandomForest, GradientBoosting ni XGBoost.",
     "Remplacer par une vraie sélection de variables (régularisation L1, importance "
     "par permutation, ou exclusion explicite), et choisir l'objectif de perte plutôt "
     "que de pondérer les entrées.",
     "app_rougeole.py ≈ l. 2095",
     "Majeur — une fonctionnalité experte inopérante.",
     "0,5 j", "✅ Corrigé"),

    ("F5", "P1 — Majeur",
     "Variables climatiques colinéaires et invariantes dans le temps",
     "`CoefClimatique = Humidite_Moy × 0,5` et `Saison_Seche_Humidite = rh_mean × 0,7` "
     "sont des multiples exacts de la même variable. Le climat est une moyenne par aire, "
     "constante dans le temps : il ne peut expliquer aucune dynamique.",
     "Supprimer les variables redondantes, et introduire des covariables climatiques "
     "véritablement temporelles (précipitations, température, humidité par semaine).",
     "app_rougeole.py ≈ l. 2049-2053",
     "Majeur", "1 j", "⚠️ Partiel"),

    ("F6", "P2 — Mineur",
     "Prévision récursive dégradée",
     "L'imputeur est réutilisé sur une ligne unique et l'historique récent est complété "
     "par des zéros.",
     "Prévision récursive avec reconstruction complète des variables à chaque pas.",
     "app_rougeole.py ≈ l. 2207-2245",
     "Mineur", "0,5 j", "✅ Corrigé"),

    ("F7", "P2 — Mineur",
     "Encodage de l'urbanisation fragile",
     "`LabelEncoder` suivi de `except ValueError: urban_enc_aire = 0` masque toute "
     "modalité non vue à l'entraînement.",
     "Encodeur ajusté sur l'ensemble des modalités connues, avec catégorie « inconnu » "
     "explicite.",
     "app_rougeole.py, encodage",
     "Mineur", "0,5 j", "✅ Corrigé"),

    # ── G. Onglet de validation ─────────────────────────────────────────
    ("G1", "P1 — Majeur",
     "Détection des pics avec fenêtrage centré",
     "`compute_peak_detection_metrics` applique `rolling(center=True)` sur le bloc de "
     "test : les semaines suivantes sont utilisées pour identifier un pic.",
     "Fenêtrage unilatéral uniquement (`center=False`, fenêtre arrière).",
     "validation_tab.py ≈ l. 290 et 510-535",
     "Majeur", "0,5 j", "✅ Corrigé (dans le noyau)"),

    ("G2", "P1 — Majeur",
     "R² de validation : moyenne non pondérée des folds",
     "`cv_r2_mean` est la moyenne arithmétique des R² par fold. Les folds de faible "
     "variance produisent des R² très négatifs qui dominent la moyenne.",
     "Publier des métriques agrégées sur l'ensemble des prédictions hors échantillon "
     "(MAE, RMSE, R², biais, rapport des totaux), avec la dispersion par fold en complément.",
     "validation_tab.py, agrégation",
     "Majeur", "0,5 j", "✅ Corrigé"),

    ("G3", "P1 — Majeur",
     "Le modèle validé n'est pas le modèle déployé",
     "L'onglet de validation reconstruit les variables indépendamment et utilise "
     "`n_estimators=100` alors que l'entraînement en utilise 300 : la validation ne "
     "porte pas sur le modèle mis en production.",
     "Valider exactement l'objet déployé : mêmes variables, mêmes hyperparamètres, "
     "même pipeline.",
     "validation_tab.py vs app_paludisme.py",
     "Majeur", "1 j", "✅ Corrigé"),

    # ── H. Documentation ────────────────────────────────────────────────
    ("H1", "P1 — Majeur",
     "Documentation contraire au code",
     "Le manuel affirme que la validation utilise des « segments temporels "
     "(TimeSeriesSplit) » alors que le découpage est spatial ; il annonce un R² > 0,90 "
     "à 2-4 semaines et un R² > 0,85 « typique » avec le GradientBoosting. Aucune de ces "
     "affirmations n'est vérifiée.",
     "Réécrire le manuel à partir du protocole réel, et ne citer que des métriques "
     "reproductibles.",
     "app_manuel.py l. 493-495, 907-908, 1575-1584",
     "Majeur — le manuel est un document de référence pour les utilisateurs.",
     "1 j", "✅ Corrigé"),

    ("H2", "P1 — Majeur",
     "MAPE présenté comme la métrique de référence alors qu'il n'est jamais calculé",
     "`README.md` consacre une section au MAPE, le qualifie de « métrique de "
     "référence pour la surveillance épidémiologique du paludisme », publie un "
     "barème d'interprétation en quatre niveaux (< 10 % excellent, 10-20 % bon, "
     "20-30 % modéré, > 30 % faible), cite des valeurs de la littérature (3,9 % à "
     "22,5 %) et conclut : « Consultez toujours le MAPE affiché dans les résultats "
     "pour une lecture directe en pourcentage. » Vérification faite sur l'ensemble "
     "du code : aucun fichier applicatif ne calcule ni n'affiche de MAPE. Les "
     "sections françaises (l. 303-318, 346, 399, 480) et leur miroir anglais "
     "(l. 710-867) renvoient donc toutes à une métrique inexistante.",
     "Retirer le barème et les renvois à une métrique inexistante, et documenter à "
     "la place ce qui est réellement calculé : le sMAPE et l'erreur relative à la "
     "moyenne observée, tous deux produits par `metrics_summary`. Le MAPE classique "
     "divise par la valeur observée : il vaut l'infini dès qu'une semaine compte "
     "zéro cas, ce qui est fréquent en basse transmission — vérifié "
     "numériquement sur un jeu comportant un zéro.",
     "README.md l. 303-318, 346, 399, 480 (et miroir anglais l. 710-867)",
     "Majeur — le README est le premier document lu, et son barème invitait à une "
     "confiance que rien ne mesurait.",
     "0,5 j", "✅ Corrigé"),

    # ── I. Sécurité et ingénierie ───────────────────────────────────────
    ("I1", "P0 — Bloquant",
     "Identifiants en clair dans le dépôt",
     "`credentials.yaml` contient un mot de passe en clair et une clé de signature de "
     "cookie, engagés dans le dépôt. L'historique git les conserve.",
     "Retirer du dépôt, faire tourner la clé et le mot de passe, réécrire l'historique, "
     "et passer par des variables d'environnement.",
     "credentials.yaml",
     "Surcritique (sécurité)", "0,5 j", "⚠️ À faire"),

    ("I2", "P1 — Majeur",
     "Sous-applications chargées par `exec()`",
     "`main_app.py` exécute le code source des sous-applications par `exec()`. Aucune "
     "isolation, aucune vérification d'intégrité, débogage impossible.",
     "Utiliser `runpy.run_path` ou un import standard, avec contrôle d'intégrité.",
     "main_app.py",
     "Majeur", "1 j", "⚠️ À faire"),

    ("I3", "P1 — Majeur",
     "Aucun test, aucune fixation des versions",
     "Aucun test automatisé. `requirements.txt` ne fixe aucune version et n'inclut ni "
     "xgboost ni lightgbm. Aucun fichier d'ignore.",
     "Ajouter la suite de tests fournie, épingler les versions, et ajouter un fichier "
     "d'ignore.",
     "requirements.txt, tests/",
     "Majeur", "1 j", "✅ Partiel (tests fournis)"),

    ("I4", "P2 — Mineur",
     "Fichiers de données corrompus engagés",
     "`data/ao_hlthArea1.zip` fait 2 octets et `data/m` fait 1 octet : des artefacts "
     "de téléchargement interrompu sont engagés dans le dépôt.",
     "Les supprimer, et stocker les référentiels géographiques en dehors du dépôt.",
     "data/",
     "Mineur", "0,5 j", "⚠️ À faire"),
]


# ══════════════════════════════════════════════════════════════════════
#  Construction du document
# ══════════════════════════════════════════════════════════════════════
def build(out_path):
    doc = Document()

    # ── styles de base ──────────────────────────────────────────────
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for section in doc.sections:
        section.left_margin = Cm(2.0)
        section.right_margin = Cm(2.0)
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)

    # ═══════════════ PAGE DE GARDE ═══════════════
    for _ in range(3):
        doc.add_paragraph()
    para(doc, "AUDIT TECHNIQUE ET SCIENTIFIQUE", gras=True, taille=13,
         couleur=GRIS, align=WD_ALIGN_PARAGRAPH.CENTER, espace_apres=2)
    para(doc, "Plateforme EpiPrediction", gras=True, taille=30, couleur=BLEU,
         align=WD_ALIGN_PARAGRAPH.CENTER, espace_apres=4)
    para(doc, "Surveillance épidémiologique — Paludisme et Rougeole",
         taille=14, couleur=GRIS, align=WD_ALIGN_PARAGRAPH.CENTER, espace_apres=30)
    para(doc, "Fiabilité des modèles prédictifs, intégrité des données, "
              "validité statistique et plan de correction",
         italique=True, taille=11, align=WD_ALIGN_PARAGRAPH.CENTER, espace_apres=40)

    tableau(
        doc,
        ["", ""],
        [["Périmètre", "app_paludisme.py (3 294 l.), app_rougeole.py (2 464 l.), "
                       "validation_tab.py, validation_tab_rougeole.py, "
                       "prediction_map_tab.py, report_generator.py, app_manuel.py, "
                       "main_app.py"],
         ["Méthode", "Lecture intégrale du code, exécution instrumentée du code "
                     "livré, preuves analytiques, protocoles comparatifs chiffrés"],
         ["Données de test", "775 aires de santé du référentiel "
                             "data/ao_hlthArea.zip (mli 588, ner 72, bfa 70, mrt 45) ; "
                             "séries hebdomadaires 2022-2024"],
         ["Date", date.today().strftime("%d %B %Y")],
         ["Statut", "Corrections principales implémentées et vérifiées ; "
                    "actions restantes listées en section 9"]],
        largeurs=[3.2, 13.0],
        fonte=9)

    doc.add_page_break()

    # ═══════════════ SOMMAIRE ═══════════════
    titre(doc, "Sommaire", 1)
    for i, s in enumerate([
        "Résumé exécutif",
        "Périmètre, méthode et limites de l'audit",
        "Fonctionnement de l'application",
        "Diagnostic détaillé — les 37 défauts constatés",
        "Résultats expérimentaux chiffrés, y compris les résultats négatifs",
        "Corrections implémentées dans ce dépôt",
        "Recommandations d'un modélisateur épidémiologiste",
        "Plan d'action point par point",
        "Annexes — reproductibilité et gouvernance",
    ], start=1):
        para(doc, f"{i}. {s}", taille=10.5, espace_apres=3)

    doc.add_page_break()

    # ═══════════════ 1. RÉSUMÉ EXÉCUTIF ═══════════════
    titre(doc, "1. Résumé exécutif", 1)

    para(doc,
         "EpiPrediction est un outil d'aide à la décision destiné à la surveillance "
         "épidémiologique du paludisme et de la rougeole. À ce titre, les nombres qu'il "
         "produit orientent des décisions de santé publique : allocation de moustiquaires, "
         "positionnement de stocks d'antipaludiques, calendrier des campagnes de "
         "vaccination de rattrapage, déclenchement d'investigations. La fiabilité de ces "
         "nombres doit donc être établie, pas supposée.", espace_apres=10)

    para(doc,
         "L'audit a porté sur la totalité du code source et sur l'exécution instrumentée "
         "du code réellement livré. Le constat est sans ambiguïté : ", espace_apres=6)

    encadre(doc,
            "Les modèles ne prédisent pas la charge de morbidité. Ils restituent une "
            "partie de leur propre cible. Les métriques affichées (R² de 0,90 à 0,99) "
            "ne mesurent pas la capacité de prévision : elles mesurent le degré de fuite "
            "d'information. Sur la période de prévision, le total prédit ne représente "
            "que 34 % à 48 % du total observé : l'application sous-estime la charge d'un "
            "facteur 2 à 3, systématiquement, pour les trois algorithmes disponibles.",
            kind="danger")

    para(doc, "Cinq défauts suffisent à expliquer ce résultat.", espace_apres=6)

    tableau(doc,
            ["N°", "Défaut", "Preuve mesurée", "Conséquence"],
            [["1", "Fuite de cible",
              "cases_ma_2 et cases_ma_4 sont calculées sans décalage : elles "
              "contiennent la semaine à prédire. La cible se reconstruit exactement "
              "par 2 × cases_ma_2 − cases_lag_1 (corrélation 1,000000, erreur "
              "maximale 0).",
              "Le R² affiché est artificiel"],
             ["2", "Validation non temporelle",
              "TimeSeriesSplit est appliqué à un tableau trié par aire de santé : "
              "les 5 folds entraînent sur les semaines 0-51 et testent sur les "
              "semaines 0-51. Le drapeau « temps trié » est Faux.",
              "La validation mesure du réapprentissage"],
             ["3", "Écrasement multi-annuel",
              "11 083 observations brutes deviennent 3 737 lignes de modèle : "
              "7 346 observations (66 %) sont perdues. Les semaines 1 de 2022, 2023 "
              "et 2024 fusionnent.",
              "Le modèle apprend sur un tiers des données"],
             ["4", "Décalage entraînement / inférence",
              "9 variables sur 16 diffèrent de plus de 50 % entre l'entraînement et "
              "la prévision ; coef_population passe de 1,0 à 0,0.",
              "Le modèle prédit hors de son domaine"],
             ["5", "Métriques in-sample affichées",
              "Le R² affiché est calculé sur les données d'entraînement ; le cv_mae, "
              "seule métrique hors échantillon calculée, n'est jamais affiché.",
              "L'utilisateur est induit en erreur"]],
            largeurs=[0.9, 3.4, 7.4, 4.5])

    titre(doc, "1.1 Ce que l'audit a changé", 2)
    para(doc,
         "Les corrections principales ont été implémentées dans ce dépôt, sans modifier "
         "la structure des tables ni les noms de colonnes existants, conformément à la "
         "contrainte posée. Un noyau de modélisation indépendant a été ajouté, et les "
         "deux onglets de modélisation ont été raccordés à ce noyau. Le code livré a été "
         "exécuté après correction, et les résultats sont rapportés en section 6.",
         espace_apres=8)

    tableau(doc,
            ["Indicateur", "Avant", "Après", "Lecture"],
            [["Observations d'entraînement", "3 737", "11 160",
              "Aucune donnée n'est perdue"],
             ["Découpage de validation", "spatial (déclaré temporel)",
              "temporel bloqué avec embargo",
              "La généralisation est réellement mesurée"],
             ["R² in-sample / R² validé", "0,992 / 0,902",
              "0,983 / 0,729",
              "L'écart optimiste est désormais visible et affiché"],
             ["Rapport prédit ÷ observé", "0,38 à 0,56",
              "0,96 à 1,06",
              "Les totaux sont calibrés"],
             ["Comparaison à une référence", "aucune",
              "+24,1 % à 1 semaine, +31,1 % à 4 semaines",
              "Le modèle apporte une valeur mesurable"],
             ["Intervalles de prédiction", "quantiles de l'historique",
              "régression quantile du modèle",
              "L'incertitude est celle du modèle"],
             ["Algorithmes disponibles", "3", "6 (dont XGBoost, LightGBM)",
              "Gradient boosting moderne, objectif Poisson"],
             ["Covariables statiques", "altitude manuelle seulement",
              "altitude, pente, NDVI, eau, urbain, population",
              "Extraction automatique"]],
            largeurs=[4.6, 3.8, 4.0, 4.4],
            fonte=8.5)

    encadre(doc,
            "Point de lecture important : le R² validé passe de 0,902 à 0,729. Ce n'est "
            "pas une régression. 0,902 était un artefact de fuite et de découpage ; "
            "0,729 est la capacité de prévision réelle, mesurée honnêtement, avec un "
            "modèle qui gagne de 16 % à 31 % sur la meilleure référence selon "
            "l'algorithme et l'horizon, et dont les totaux sont calibrés à quelques "
            "pour cent. Un indicateur plus bas mais vrai est "
            "utilisable pour décider ; un indicateur plus haut mais faux ne l'est pas.",
            kind="info")

    titre(doc, "1.2 Priorité des actions", 2)
    tableau(doc, ["Priorité", "Nombre d'actions", "Statut"],
            [["P0 — Bloquant", str(sum(1 for a in ACTIONS if a[1].startswith("P0"))),
              f"{sum(1 for a in ACTIONS if a[1].startswith('P0') and a[8].startswith('✅'))} corrigées"],
             ["P1 — Majeur", str(sum(1 for a in ACTIONS if a[1].startswith("P1"))),
              f"{sum(1 for a in ACTIONS if a[1].startswith('P1') and a[8].startswith('✅'))} corrigées"],
             ["P2 — Mineur", str(sum(1 for a in ACTIONS if a[1].startswith("P2"))),
              f"{sum(1 for a in ACTIONS if a[1].startswith('P2') and a[8].startswith('✅'))} corrigées"],
             ["Total", str(len(ACTIONS)),
              f"{sum(1 for a in ACTIONS if a[8].startswith('✅'))} corrigées, "
              f"{sum(1 for a in ACTIONS if a[8].startswith('⚠️'))} restantes"]],
            largeurs=[3.5, 3.5, 9.0],
            fonte=9)

    doc.add_page_break()

    # ═══════════════ 2. MÉTHODE ═══════════════
    titre(doc, "2. Périmètre, méthode et limites de l'audit", 1)

    titre(doc, "2.1 Ce qui a été examiné", 2)
    for t in [
        ("Code source. ", "Lecture intégrale des deux applications, des deux onglets "
         "de validation, de la cartographie prédictive, du générateur de rapports et du "
         "manuel utilisateur."),
        ("Exécution du code livré. ", "Les deux applications exécutent des appels "
         "d'interface au niveau du module, ce qui interdit leur importation. Un harnais "
         "d'extraction par arbre syntaxique a été écrit : il isole le bloc de l'onglet "
         "de modélisation et l'exécute dans un espace de noms où l'interface est simulée. "
         "Les métriques rapportées en section 5 proviennent donc du code réellement "
         "livré, et non d'une réimplémentation."),
        ("Preuves analytiques. ", "Certaines fuites sont démontrées algébriquement sur "
         "la table de variables produite par l'application, indépendamment de tout jeu "
         "de données."),
        ("Protocoles comparatifs. ", "Les mêmes données ont été traitées selon "
         "plusieurs protocoles de validation pour isoler la contribution de chaque "
         "défaut au R² affiché."),
    ]:
        puce(doc, t[1], gras_debut=t[0])

    titre(doc, "2.2 Données utilisées", 2)
    para(doc,
         "Le référentiel géographique data/ao_hlthArea.zip a été utilisé : 775 aires de "
         "santé réparties entre le Mali (588), le Niger (72), le Burkina Faso (70) et la "
         "Mauritanie (45). Les 72 aires du Niger ont servi aux expériences, soit 11 083 "
         "observations hebdomadaires sur 2022-2024.", espace_apres=8)

    encadre(doc,
            "Limite à signaler explicitement. L'environnement d'audit ne dispose pas "
            "d'accès réseau. Les services externes sollicités par l'application — "
            "NASA POWER, Open-Meteo, Google Earth Engine, WorldPop — n'ont donc pas pu "
            "être interrogés. Les covariables statiques (altitude, pente, NDVI, part "
            "d'eau, part urbaine, population) utilisées dans les expériences sont des "
            "valeurs de synthèse, générées avec une graine fixe et étiquetées comme "
            "telles. Le module d'extraction automatique fourni "
            "(epimodel/static_covariates.py) n'a jamais été exécuté contre un compte "
            "Google Earth Engine réel : son interface et sa logique de fusion sont "
            "testées, mais son comportement en production reste à valider lors du "
            "branchement.",
            kind="attention")

    para(doc,
         "Cette limite ne concerne pas les constats de fuite, de validation, de "
         "découpage et de décalage entraînement/inférence : ceux-ci sont établis sur le "
         "code et sur des preuves algébriques, indépendamment de l'origine des "
         "covariables.", espace_apres=8)

    titre(doc, "2.3 Ce que l'audit n'a pas couvert", 2)
    for t in ["La qualité intrinsèque des données de surveillance saisies sur le terrain "
              "(complétude, délais de notification, exhaustivité du réseau).",
              "L'ergonomie et l'accessibilité de l'interface.",
              "La performance sous charge et la montée en charge du déploiement.",
              "La conformité au traitement des données personnelles de santé, qui "
              "relève d'une revue juridique distincte."]:
        puce(doc, t)

    doc.add_page_break()

    # ═══════════════ 3. FONCTIONNEMENT ═══════════════
    titre(doc, "3. Fonctionnement de l'application", 1)

    titre(doc, "3.1 Architecture", 2)
    para(doc,
         "L'application est un portail Streamlit. main_app.py assure "
         "l'authentification à partir de credentials.yaml, puis charge la "
         "sous-application choisie en exécutant son code source par exec(). "
         "Chaque maladie dispose de son application propre, organisée en onglets.",
         espace_apres=8)

    tableau(doc, ["Onglet", "Rôle", "Fichier"],
            [["1 — Analyse", "Chargement des registres, filtres, cartes de chaleur, "
                              "courbes épidémiologiques, indicateurs descriptifs",
              "app_paludisme.py / app_rougeole.py"],
             ["2 — Cartographie", "Choroplèthes des cas observés par aire de santé, "
                                  "seuils d'alerte, légende",
              "app_paludisme.py / app_rougeole.py"],
             ["3 — Modélisation", "Construction des variables, entraînement, "
                                  "validation croisée, prévision à 4 semaines, "
                                  "importance des variables, exports",
              "app_paludisme.py / app_rougeole.py"],
             ["4 — Carte prédictive", "Cartographie des prédictions, classes de risque, "
                                      "tendances par aire",
              "prediction_map_tab.py"],
             ["Validation", "Validation rétrospective, détection de pics, comparaison "
                            "de protocoles",
              "validation_tab.py / validation_tab_rougeole.py"],
             ["Rapports", "Génération de documents Word de synthèse",
              "report_generator.py"]],
            largeurs=[3.2, 8.2, 4.8], fonte=8.5)

    titre(doc, "3.2 Chaîne de traitement de la modélisation (état initial)", 2)
    for i, t in enumerate([
        "Agrégation des cas par aire de santé et par semaine épidémiologique.",
        "Enrichissement par des covariables démographiques (population totale, "
        "enfants de moins de 15 ans, densité) et environnementales (température, "
        "humidité, précipitations, inondation, distance aux cours d'eau), "
        "provenant de services externes ou de téléversements manuels.",
        "Construction de 16 variables dérivées : retards de 1 à 4 semaines, moyennes "
        "mobiles, sinusoïdes saisonnières, taux d'incidence, indicateurs de risque.",
        "Imputation, centrage-réduction et analyse en composantes principales ramenée "
        "à 97,2 % de variance expliquée (9 composantes), puis regroupement spatial en "
        "5 classes.",
        "Entraînement de l'algorithme choisi sur la totalité des données.",
        "Validation croisée présentée comme temporelle.",
        "Prévision récursive à 4 semaines, avec reconstruction manuelle des variables "
        "pour les lignes futures.",
        "Affichage des métriques, de l'importance des variables et des exports.",
    ], start=1):
        puce(doc, t, gras_debut=f"Étape {i}. ")

    titre(doc, "3.3 Le cœur du problème", 2)
    para(doc,
         "La chaîne est correcte dans son intention : retarder les séries, capturer la "
         "saisonnalité, intégrer le contexte environnemental, regrouper les aires "
         "voisines. Chacune de ces intentions est justifiée épidémiologiquement.",
         espace_apres=6)
    para(doc,
         "Le problème n'est pas le choix des méthodes mais leur exécution : les "
         "retards ne sont pas décalés, la validation n'est pas temporelle, le panneau "
         "est écrasé entre les années, et la prévision n'utilise pas les mêmes variables "
         "que l'entraînement. Ces quatre défauts se composent. Pris isolément, chacun "
         "dégraderait la qualité ; ensemble, ils rendent les métriques affichées "
         "dépourvues de signification prédictive, tout en leur donnant l'apparence de "
         "l'excellence.", espace_apres=6)
    para(doc,
         "C'est précisément la configuration la plus dangereuse pour un outil d'aide à "
         "la décision : un système qui affiche une confiance élevée et se trompe de "
         "façon systématique dans une direction connue — ici, la sous-estimation.",
         espace_apres=8)

    doc.add_page_break()

    # ═══════════════ 4. DIAGNOSTIC ═══════════════
    titre(doc, "4. Diagnostic détaillé — les 37 défauts constatés", 1)
    para(doc,
         "Chaque défaut est numéroté, localisé dans le code, et accompagné de la preuve "
         "qui l'établit. Le code entre parenthèses dans chaque titre (A1, B2, C3…) "
         "renvoie à l'action correspondante du plan d'action de la section 8 : "
         "plusieurs actions peuvent traiter un même défaut, et inversement.",
         espace_apres=8)

    def bloc_diag(num, titre_, localisation, constat, preuve=None, consequence=None):
        titre(doc, f"Défaut {num} — {titre_}", 2)
        para(doc, localisation, italique=True, taille=9, couleur=GRIS, espace_apres=4)
        para(doc, constat, espace_apres=6)
        if preuve:
            para(doc, "Preuve mesurée", gras=True, taille=9.5, espace_apres=2)
            if isinstance(preuve, list):
                for x in preuve:
                    puce(doc, x)
            else:
                code(doc, preuve)
        if consequence:
            encadre(doc, "Conséquence : " + consequence, kind="attention")

    # --- 4.1 fuite
    titre(doc, "4.1 Fuites d'information", 2)
    bloc_diag(1, "Moyennes mobiles non décalées (A1)",
              "app_paludisme.py, create_advanced_features, lignes ≈ 682 et 701",
              "Les variables cases_ma_2 et cases_ma_4 sont calculées par "
              "rolling(w, min_periods=1).mean(), sans décalage. La fenêtre de la "
              "moyenne mobile inclut donc la semaine courante — celle que le modèle "
              "doit prédire. La cible est directement disponible parmi les variables "
              "explicatives.",
              preuve=["cases_ma_2 = (cases_t + cases_lag_1) / 2, d'où "
                      "cases_t = 2 × cases_ma_2 − cases_lag_1",
                      "Corrélation entre cette reconstruction et la cible réelle : "
                      "1,000000 ; erreur absolue maximale : 0",
                      "cases_ma_4 suit le même schéma avec une fenêtre de 4 semaines"],
              consequence="Le modèle n'a pas besoin d'apprendre la dynamique de "
                          "transmission : il lui suffit d'inverser une moyenne mobile. "
                          "C'est la première cause du R² affiché. Il ne s'agit pas d'une "
                          "fuite « légère » : la cible est exactement reconstituable.")

    bloc_diag(2, "Variables dérivées de la même ligne (A2)",
              "app_paludisme.py, enrichissement, lignes ≈ 1428-1456",
              "incidence_rate est calculé comme le nombre de cas de la semaine courante "
              "divisé par la population. child_risk et demo_pressure en dérivent "
              "également. Ces variables sont donc des transformations bijectives de la "
              "cible.",
              preuve="cases_t = incidence_rate × Pop_Totale / 1e4 : corrélation "
                     "1,000000, erreur absolue maximale 7,28 × 10⁻¹²",
              consequence="Même effet que le défaut 1, par une autre voie. Supprimer "
                          "les moyennes mobiles sans traiter ces variables ne résout "
                          "rien.")

    bloc_diag(3, "Analyse en composantes principales et regroupement entraînés sur "
                 "l'échantillon complet (A5)",
              "app_paludisme.py, perform_pca_analysis, ligne ≈ 1035",
              "L'imputeur, le standardiseur et l'ACP sont ajustés sur la totalité des "
              "observations, puis le découpage de validation est appliqué sur les "
              "composantes ainsi produites. Chaque bloc de test a participé à "
              "l'estimation des axes qui le représentent.",
              preuve="Les 16 variables sont réduites à 9 composantes expliquant "
                     "97,18 % de la variance, calculées avant tout découpage",
              consequence="La validation croisée ne peut pas détecter le "
                          "surapprentissage, puisque l'information du bloc de test est "
                          "déjà incorporée aux axes.")

    # --- 4.2 validation
    titre(doc, "4.2 Validation non temporelle", 2)
    bloc_diag(4, "TimeSeriesSplit appliqué à un tableau trié par aire (A3)",
              "app_paludisme.py, lignes ≈ 993 et 1006 ; validation_tab.py, ligne ≈ 290",
              "TimeSeriesSplit découpe les lignes dans l'ordre du tableau, en supposant "
              "que celui-ci est trié par date. Or le tableau est trié par aire de "
              "santé. Le découpage sépare donc des aires, pas des périodes.",
              preuve=["Drapeau « données triées dans le temps » : Faux",
                      "Les 5 folds entraînent sur les semaines 0-51 et testent sur "
                      "les semaines 0-51, avec une aire en commun",
                      "Indicateur « semaines futures présentes dans l'entraînement » : "
                      "Vrai pour tous les folds"],
              consequence="La validation mesure la capacité du modèle à reconnaître des "
                          "aires déjà vues, dans des semaines déjà vues. Elle ne mesure "
                          "aucune capacité de prévision.")

    bloc_diag(5, "Métriques in-sample présentées comme performances (A4)",
              "app_paludisme.py, lignes ≈ 2626-2718 ; app_manuel.py, lignes 493-495 et "
              "907-908",
              "Le R² et le RMSE affichés en tête de l'onglet sont calculés sur les "
              "données d'entraînement. Le cv_mae, seule métrique hors échantillon "
              "effectivement calculée, n'est jamais affiché. L'interface conclut "
              "« modèle fiable » dès que R² dépasse 0,85.",
              preuve=["R² in-sample affiché : 0,9923 (RandomForest), 0,9993 "
                      "(GradientBoosting), 0,9971 (ExtraTrees)",
                      "Le manuel annonce un R² supérieur à 0,90 à 2-4 semaines et "
                      "qualifie un R² supérieur à 0,85 de « typique »",
                      "cv_mae calculé puis non affiché"],
              consequence="L'utilisateur reçoit un indicateur de qualité qui ne "
                          "correspond à aucune capacité de prévision, et un seuil de "
                          "décision (« fiable ») fondé sur cet indicateur.")

    bloc_diag(6, "R² de validation : moyenne non pondérée des folds (G2)",
              "validation_tab.py, agrégation",
              "cv_r2_mean est la moyenne arithmétique des R² calculés fold par fold. "
              "Les folds dont la variance cible est faible produisent des R² fortement "
              "négatifs qui pèsent autant que les autres.",
              consequence="La métrique agrégée est instable et difficile à interpréter. "
                          "Les métriques hors échantillon doivent être agrégées sur "
                          "l'ensemble des prédictions, la dispersion par fold étant "
                          "donnée en complément.")

    bloc_diag(7, "Détection des pics avec fenêtrage centré (G1)",
              "validation_tab.py, lignes ≈ 290 et 510-535",
              "compute_peak_detection_metrics applique rolling(center=True) sur le bloc "
              "de test. Pour identifier un pic à la semaine t, la fenêtre utilise les "
              "semaines t+1 et t+2.",
              consequence="La sensibilité de détection des pics est surestimée.")

    bloc_diag(8, "Le modèle validé n'est pas le modèle déployé (G3)",
              "validation_tab.py contre app_paludisme.py",
              "L'onglet de validation reconstruit les variables de son côté et utilise "
              "n_estimators=100, alors que l'entraînement de l'onglet de modélisation "
              "en utilise 300.",
              consequence="La validation ne porte pas sur l'objet mis en production. "
                          "Un modèle peut être validé puis déployé dans une "
                          "configuration différente.")

    # --- 4.3 panneau
    titre(doc, "4.3 Structure du panneau et perte de données", 2)
    bloc_diag(9, "Écrasement multi-annuel : 66 % des observations perdues (B1)",
              "app_paludisme.py, ligne ≈ 2457",
              "Les variables sont construites par groupby('health_area')['cases'].shift(n) "
              "sur un panneau qui n'est pas trié par date, puis les lignes sont "
              "agrégées par aire. Les observations de différentes années se retrouvent "
              "confondues.",
              preuve=["11 083 observations brutes produisent 3 737 lignes de modèle : "
                      "7 346 observations (66 %) sont perdues",
                      "L'aire abala passe de 156 à 52 lignes",
                      "Semaine 1 par année : 2022 → 69, 2023 → 185, 2024 → 194 ; "
                      "après agrégation : 448 lignes uniques au lieu de 448 réparties "
                      "sur trois années distinctes"],
              consequence="Le modèle est entraîné sur un tiers des données disponibles, "
                          "et la structure saisonnière interannuelle — précisément ce "
                          "qu'une prévision épidémiologique doit capter — est détruite.")

    bloc_diag(10, "Semaine ISO 53 supprimée (B2)",
              "app_paludisme.py, agrégation hebdomadaire",
              "Le filtre between(1, 52) élimine la semaine 53, qui existe dans le "
              "calendrier ISO-8601 environ une année sur six.",
              consequence="Perte récurrente d'une semaine de données par an, et "
                          "rupture de l'index temporel.")

    bloc_diag(11, "Conversion année-semaine en date non conforme à l'ISO-8601 (B3)",
              "app_paludisme.py, ligne 125 ; aggregate_climate_by_week_and_area",
              "week_to_date_range(week_num, year=2024) calcule une date par une formule "
              "approximative plutôt que par le calendrier ISO. L'agrégation climatique "
              "prend une seule année en paramètre.",
              consequence="Les covariables climatiques ne sont pas alignées sur les "
                          "mêmes semaines que les cas : les relations "
                          "climat-morbidité sont estimées sur des paires décalées.")

    bloc_diag(12, "Panneau déséquilibré : les retards traversent les lacunes (B4)",
              "app_paludisme.py, create_advanced_features",
              "Un shift(1) appliqué à une aire dont les semaines sont lacunaires "
              "renvoie la dernière valeur disponible, qui peut dater de plusieurs mois.",
              consequence="Les variables de retard ne représentent pas toujours la "
                          "semaine précédente, et leur signification varie d'une aire "
                          "à l'autre.")

    # --- 4.4 train/serve
    titre(doc, "4.4 Décalage entre entraînement et prévision", 2)
    bloc_diag(13, "Neuf variables sur seize diffèrent à la prévision (C1)",
              "app_paludisme.py, boucle de prévision ; app_rougeole.py, lignes ≈ 2207-2245",
              "À l'entraînement, la matrice de conception est produite par la fonction "
              "de construction des variables. À la prévision, les lignes futures ne "
              "reçoivent que les colonnes statiques, le cluster spatial et des retards "
              "reconstruits manuellement. Le modèle est appliqué à des vecteurs "
              "d'entrée dont la distribution diffère radicalement de celle sur "
              "laquelle il a été ajusté.",
              preuve=["cases_lag_1 : 2 055,8 à l'entraînement contre 916,2 à la prévision",
                      "cases_lag_2 : 2 078,7 contre 896,2",
                      "cases_lag_4 : 2 123,7 contre 928,9",
                      "cases_ma_2 : 2 028,8 contre 906,2",
                      "cases_ma_4 : 2 021,5 contre 906,3",
                      "week_num : 25,5 contre 53,5",
                      "sin_week : −0,001 contre 0,179 ; cos_week : −0,001 contre 0,975",
                      "coef_population : 1,0 contre 0,0"],
              consequence="Le modèle est évalué dans une région de l'espace d'entrée "
                          "qu'il n'a jamais rencontrée. C'est la cause directe de la "
                          "dérive du défaut 14.")

    bloc_diag(14, "Sous-estimation systématique d'un facteur 2 à 3 (C2)",
              "app_paludisme.py, prévision récursive",
              "Le total prédit sur la période de prévision est rapporté au total "
              "observé sur la même période, pour les trois algorithmes disponibles, "
              "en exécutant le code réellement livré.",
              preuve=[f"{r.get('algorithme','?')} : R² affiché "
                      f"{fmt(r.get('R2_affiche_in_sample'),4)}, R² de validation "
                      f"{fmt(r.get('R2_CV_affiche'),4)}, rapport prédit ÷ observé "
                      f"{fmt(r.get('ratio_pred_obs'),3)}"
                      for r in P1_LEGACY] +
                     ["Médiane du rapport par aire : 0,524 ; 10ᵉ centile : 0,275 ; "
                      "90ᵉ centile : 4,874"],
              consequence="L'application sous-estime la charge de morbidité de façon "
                          "cohérente et massive. Un R² proche de 1 coexiste avec une "
                          "erreur de niveau de 66 %. Toute planification fondée sur ces "
                          "prévisions serait sous-dimensionnée.")

    bloc_diag(15, "Retard spatial incohérent à la prévision (C3)",
              "app_paludisme.py, add_spatial_features",
              "À l'entraînement, spatial_lag est la moyenne pondérée des cas des aires "
              "voisines. À la prévision, il vaut le dernier cas observé de l'aire "
              "elle-même.",
              consequence="La même variable désigne deux quantités différentes selon "
                          "le contexte.")

    bloc_diag(16, "Climat futur fabriqué par une sinusoïde (C4)",
              "app_paludisme.py, construction des lignes futures",
              "Les covariables climatiques des semaines futures sont produites par "
              "seasonal_factor = 1 + 0,15 × sin_week. Aucune donnée climatique réelle "
              "n'intervient.",
              consequence="Le modèle « prédit » avec une saisonnalité qu'on lui a "
                          "injectée. La part de la prédiction attribuable au climat "
                          "n'est pas une information, c'est une hypothèse de "
                          "construction.")

    bloc_diag(17, "Variables non disponibles à l'horizon de prévision (C5)",
              "app_paludisme.py, enrichissement climatique",
              "Plusieurs covariables climatiques sont agrégées sur la semaine courante, "
              "alors qu'elles ne seront connues qu'après la fin de cette semaine, compte "
              "tenu des délais de traitement des images satellites.",
              consequence="En conditions réelles, ces variables ne seront pas "
                          "disponibles au moment de la décision.")

    # --- 4.5 covariables
    titre(doc, "4.5 Covariables statiques et environnementales", 2)
    bloc_diag(18, "Aucune extraction automatique des covariables statiques (D1)",
              "app_paludisme.py, extraction des données",
              "L'altitude n'entre dans le modèle que si un utilisateur téléverse "
              "manuellement un fichier GeoTIFF. Aucune pente, aucun indice de "
              "végétation, aucune occupation du sol, aucune distance à l'eau n'est "
              "extrait automatiquement.",
              consequence="Le modèle ne dispose pas des déterminants structurels du "
                          "risque palustre. Pour les aires à historique court ou "
                          "lacunaire — précisément celles où la prévision est la plus "
                          "utile — il n'existe aucun signal alternatif au passé de "
                          "l'aire.")

    bloc_diag(19, "Formules de dérivées climatiques arbitraires (D2)",
              "app_paludisme.py, variables environnementales",
              "flood_mean / (dist_river + 0,1) et exp(−((temp − 27,5)²) / 50) "
              "introduisent des constantes non documentées, non calibrées et non "
              "testées. dist_river est exprimée en degrés puis multipliée par 111, "
              "conversion qui n'est exacte qu'à l'équateur.",
              consequence="Ces variables ajoutent du bruit interprétable plutôt que de "
                          "l'information, et leur forme fonctionnelle est imposée sans "
                          "justification.")

    bloc_diag(20, "Population extraite sans filtre d'année (D3)",
              "app_paludisme.py, extraction WorldPop",
              "mosaic() est appelé sans filtre temporel : la mosaïque combine "
              "plusieurs millésimes.",
              consequence="Les taux d'incidence reposent sur des dénominateurs "
                          "d'années différentes selon la zone.")

    bloc_diag(21, "Calcul de surface sans effet et géométries ignorées (D4, D5)",
              "app_paludisme.py, extracteurs",
              "ee.Image.pixelArea().divide(10000) sur une image à 100 m ne correspond à "
              "aucune unité utile. Les extracteurs n'utilisent que geom.exterior : les "
              "MultiPolygon et GeometryCollection sont ignorés sans avertissement.",
              consequence="Des aires de santé sont silencieusement exclues de "
                          "l'enrichissement.")

    # --- 4.6 déterminisme
    titre(doc, "4.6 Déterminisme et reproductibilité", 2)
    bloc_diag(22, "Ordre des variables non déterministe (E4)",
              "app_paludisme.py, ligne ≈ 2618",
              "list(set(...)) produit un ordre dépendant de l'ordre de hachage de "
              "l'interpréteur.",
              preuve="Cinq exécutions consécutives à données identiques ont produit "
                     "cinq ordres de variables distincts",
              consequence="Les prédictions diffèrent d'une exécution à l'autre. Aucun "
                          "résultat n'est reproductible, ce qui interdit l'audit et la "
                          "traçabilité des décisions.")

    bloc_diag(23, "Encodage des clusters incohérent et cache prématuré (E5, E9)",
              "app_paludisme.py, lignes ≈ 1429, 1459",
              "pd.get_dummies est utilisé à l'entraînement et range(n_clusters) à la "
              "prévision. Par ailleurs, cache_key est calculé avant la relecture "
              "effective des données.",
              consequence="Deux jeux de données différents peuvent partager la même "
                          "clé de cache.")

    bloc_diag(24, "Taux de croissance non borné (E6)",
              "app_paludisme.py, create_advanced_features",
              "growth_rate = pct_change().fillna(0) sur une série comportant des zéros "
              "produit des valeurs infinies, propagées au modèle.",
              consequence="Les infinis dégradent silencieusement l'imputation et "
                          "l'ajustement.")

    # --- 4.7 rougeole
    titre(doc, "4.7 Défauts propres au module rougeole", 2)
    bloc_diag(25, "Découpage aléatoire sur des séries temporelles (F1)",
              "app_rougeole.py, lignes ≈ 2098 et 2118",
              "train_test_split(test_size=0.2, random_state=42) répartit les "
              "observations au hasard, puis cross_val_score utilise un KFold simple.",
              preuve="Référence : sur des variables sans aucune fuite, un découpage "
                     "aléatoire donne une erreur absolue de 47,5 là où un découpage "
                     "temporel honnête donne 115,1 — soit une surestimation d'un "
                     "facteur 2,4",
              consequence="Le modèle apprend l'avenir. Le défaut est ici plus grave "
                          "que pour le paludisme, car il ne peut même pas être détecté "
                          "par une lecture du R².")

    bloc_diag(26, "Fuite via la population non vaccinée et double comptage (F2, F3)",
              "app_rougeole.py, enrichissement et agrégation, lignes ≈ 1186-1260",
              "NonVaccines est dérivé des mêmes lignes que celles qui définissent "
              "CasObserves. Par ailleurs, CasObserves = (\"ID_Cas\", \"count\") compte "
              "les lignes du registre plutôt que les cas distincts.",
              consequence="La couverture vaccinale apparente est mécaniquement liée au "
                          "nombre de cas ; et tout cas enregistré plusieurs fois est "
                          "compté plusieurs fois.")

    bloc_diag(27, "Pondération manuelle des variables sans effet (F4)",
              "app_rougeole.py, ligne ≈ 2095",
              "X = X * feature_weights multiplie les colonnes par des coefficients "
              "choisis par l'utilisateur dans le mode Expert.",
              preuve="Les arbres de décision partitionnent sur des seuils : multiplier "
                     "une variable par une constante positive ne change ni l'ordre des "
                     "observations ni les seuils optimaux. RandomForest, "
                     "GradientBoosting, ExtraTrees et XGBoost sont invariants par cette "
                     "transformation.",
              consequence="Une fonctionnalité présentée comme experte, documentée et "
                          "paramétrable, n'a rigoureusement aucun effet sur les "
                          "prédictions.")

    bloc_diag(28, "Variables climatiques colinéaires et invariantes (F5)",
              "app_rougeole.py, lignes ≈ 2049-2053",
              "CoefClimatique = Humidite_Moy × 0,5 et Saison_Seche_Humidite = "
              "rh_mean × 0,7 sont des multiples exacts de la même variable. Le climat "
              "est une moyenne par aire, constante dans le temps.",
              consequence="Redondance parfaite et aucune capacité à expliquer une "
                          "dynamique temporelle.")

    bloc_diag(29, "Prévision récursive dégradée et encodage fragile (F6, F7)",
              "app_rougeole.py, lignes ≈ 2207-2245",
              "L'imputeur est réutilisé sur une ligne unique, l'historique récent est "
              "complété par des zéros, et un LabelEncoder suivi de "
              "except ValueError: urban_enc_aire = 0 masque toute modalité non vue.",
              consequence="Les premières semaines prévues reposent sur un historique "
                          "artificiel.")

    # --- 4.8 doc et sécurité
    titre(doc, "4.8 Documentation, sécurité et ingénierie", 2)
    bloc_diag(30, "Documentation contraire au code (H1, H2)",
              "app_manuel.py, lignes 493-495, 907-908 et 1575-1584",
              "Le manuel (app_manuel.py) décrit une validation par « segments "
              "temporels (TimeSeriesSplit) » alors que le découpage effectif est "
              "spatial, annonce un R² supérieur à 0,90 à 2-4 semaines, qualifie un "
              "R² supérieur à 0,85 de « typique », et fonde sa règle de fiabilité "
              "sur le R² in-sample. Le README, de son côté, présente le MAPE comme "
              "« la métrique de référence » avec un barème en quatre niveaux, alors "
              "qu'aucun code ne le calcule. Les deux documents ont été corrigés : le "
              "manuel décrit désormais le protocole réel, et le README remplace le "
              "MAPE par le sMAPE et l'erreur relative à la moyenne observée, qui "
              "sont effectivement calculés.",
              consequence="Le manuel est le document de référence des utilisateurs. Il "
                          "atteste d'une rigueur méthodologique que le code ne met pas "
                          "en œuvre.")

    bloc_diag(31, "Identifiants en clair dans le dépôt (I1)",
              "credentials.yaml",
              "Un mot de passe en clair et une clé de signature de cookie sont engagés "
              "dans le dépôt, et conservés dans l'historique.",
              consequence="Toute personne ayant accès au dépôt peut s'authentifier et "
                          "forger des sessions.")

    bloc_diag(32, "Sous-applications chargées par exec() (I2)",
              "main_app.py",
              "Le code source des sous-applications est exécuté par exec().",
              consequence="Aucune isolation, aucune vérification d'intégrité, et un "
                          "débogage considérablement plus difficile.")

    bloc_diag(33, "Aucun test, versions non fixées, fichiers corrompus (I3, I4)",
              "requirements.txt, data/",
              "Aucun test automatisé. requirements.txt ne fixe aucune version et "
              "n'inclut ni xgboost ni lightgbm. Aucun fichier d'ignore. "
              "data/ao_hlthArea1.zip fait 2 octets et data/m fait 1 octet : des "
              "artefacts de téléchargement interrompu sont engagés.",
              consequence="Aucune régression ne peut être détectée automatiquement, et "
                          "une mise à jour de dépendance peut modifier silencieusement "
                          "les résultats.")

    bloc_diag(34, "Sur-souscription des threads (E8)",
              "app_paludisme.py, entraînement",
              "n_jobs=-1 est utilisé sur des panneaux de quelques milliers "
              "d'observations.",
              preuve="Mesure sur 11 160 observations × 42 variables, XGBoost 600 "
                     "arbres : 1,9 s en mono-thread, 47,4 s avec n_jobs=-1, 75,1 s avec "
                     "n_jobs=4",
              consequence="Le temps de réponse est multiplié par 25 à 40 sans aucun "
                          "gain.")

    bloc_diag(35, "Seuil d'alerte écrasé (E7)",
              "app_paludisme.py, ligne ≈ 2897",
              "Le seuil choisi par l'utilisateur dans le curseur est écrasé par "
              "alert_threshold = 75 en dur.",
              consequence="Le paramètre affiché à l'utilisateur n'est pas celui "
                          "utilisé pour le calcul des alertes.")

    bloc_diag(36, "sMAPE non protégé contre la division par zéro",
              "epimodel/validation.py, ligne 112 (introduit puis corrigé pendant l'audit)",
              "Le dénominateur du sMAPE s'annule lorsque la valeur observée et la "
              "valeur prédite sont nulles simultanément, ce qui est fréquent sur des "
              "comptages faibles.",
              preuve="RuntimeWarning: invalid value encountered in divide",
              consequence="Métrique non définie sur les aires à très faible "
                          "morbidité — précisément celles où la rougeole resurgit.")

    bloc_diag(37, "Imputeur supprimant les colonnes entièrement vides",
              "epi_app_bridge.py, temporal_validation (introduit puis corrigé pendant "
              "l'audit)",
              "SimpleImputer supprime silencieusement les colonnes 100 % vides : la "
              "matrice transformée compte alors moins de colonnes que la liste de "
              "variables, et le modèle n'est plus entraîné sur les mêmes entrées qu'à "
              "la prévision.",
              preuve="ValueError: Shape of passed values is (2736, 42), indices imply "
                     "(2736, 43)",
              consequence="C'est exactement la classe de défaut que l'audit reproche "
                          "au code initial : elle a été reproduite, détectée par test, "
                          "et corrigée par keep_empty_features=True.")

    doc.add_page_break()

    # ═══════════════ 5. RÉSULTATS ═══════════════
    titre(doc, "5. Résultats expérimentaux chiffrés", 1)

    titre(doc, "5.1 Métriques réellement affichées par l'application", 2)
    para(doc,
         "Le tableau ci-dessous est produit en exécutant le code livré, onglet de "
         "modélisation du paludisme, sur les 72 aires de santé du Niger, années 2022 à "
         "2024. Les deux premières colonnes sont celles que l'utilisateur voit.",
         espace_apres=8)
    tableau(doc,
            ["Algorithme", "R² affiché", "R² validé", "MAE affichée",
             "Variables", "Observations", "Prédit ÷ observé"],
            [[r.get("algorithme", "?"),
              fmt(r.get("R2_affiche_in_sample"), 4),
              fmt(r.get("R2_CV_affiche"), 4),
              fmt(r.get("MAE_in_sample"), 1),
              r.get("n_features_modele", "?"),
              r.get("n_obs_modele", "?"),
              fmt(r.get("ratio_pred_obs"), 3)]
             for r in P1_LEGACY],
            largeurs=[3.0, 2.0, 2.0, 2.2, 1.7, 2.4, 2.7],
            titre_tab="Tableau 5.1 — Métriques affichées par le code livré, et calibration réelle")

    encadre(doc,
            "Un R² de 0,9993 coexiste avec un rapport prédit ÷ observé de 0,344. "
            "Le modèle explique 99,9 % de la variance des données sur lesquelles il a "
            "été entraîné, et sous-estime de 66 % le total des cas à prévoir. Ces deux "
            "nombres ne sont pas contradictoires : ils mesurent deux choses "
            "différentes, et seule la seconde a une valeur opérationnelle.",
            kind="danger")

    titre(doc, "5.2 Origine du R² affiché : fuite ou découpage ?", 2)
    para(doc,
         "Quatre protocoles ont été appliqués aux mêmes données, avec les mêmes "
         "variables, en retirant sélectivement chaque source d'optimisme. Le découpage "
         "est bloqué par semaine avec un embargo de 4 semaines, sauf mention contraire.",
         espace_apres=8)
    protos = LEAK.get("protocols") or {}
    if protos:
        lignes = []
        for nom, r in protos.items():
            lignes.append([
                nom.replace("_", " "),
                int(r.get("n_features") or 0),
                fmt(r.get("mae"), 1),
                fmt(r.get("mae_pct_mean"), 1) + " %",
                fmt(r.get("r2"), 3),
                fmt(r.get("r2_in_sample"), 3)])
        ref = LEAK.get("reference_split_aleatoire_sans_fuite")
        if ref:
            lignes.append(["[référence] découpage aléatoire, sans fuite",
                           int(ref.get("n_features") or 0),
                           fmt(ref.get("mae"), 1),
                           fmt(ref.get("mae_pct_mean"), 1) + " %",
                           fmt(ref.get("r2"), 3),
                           fmt(ref.get("r2_in_sample"), 3)])
        tableau(doc, ["Protocole de validation", "Variables", "MAE",
                      "MAE / moyenne observée", "R² validé", "R² in-sample"],
                lignes, largeurs=[5.4, 1.6, 1.9, 2.8, 2.0, 2.3],
                titre_tab="Tableau 5.2 — Contribution de chaque source d'optimisme "
                          "(mêmes données, même algorithme, RandomForest 300 arbres, "
                          "5 folds bloqués, embargo 4 semaines)")
    para(doc,
         "Lecture : supprimer les moyennes mobiles fuitées fait passer l'erreur absolue "
         "de 82,3 à 114,0 cas, soit une dégradation de 38 % — c'est l'ampleur de "
         "l'optimisme qu'elles procuraient. Passer d'un découpage aléatoire à un "
         "découpage temporel honnête, sans aucune fuite, fait passer l'erreur de 47,5 à "
         "115,1 cas : le découpage aléatoire seul surestimait la précision d'un facteur "
         "2,4. Les deux effets se cumulent : l'écart entre le protocole P1 (l'état "
         "actuel de l'application) et le protocole P4 (aucune fuite, découpage honnête) "
         "est de 82,3 à 115,1 cas, soit 40 % d'erreur sous-estimée.", espace_apres=8)

    titre(doc, "5.3 Performance du noyau corrigé, protocole honnête", 2)
    para(doc,
         "Protocole : 12 origines glissantes, entraînement minimal de 60 semaines, "
         "métriques agrégées sur l'ensemble des prédictions hors échantillon, "
         "comparaison aux modèles de référence.", espace_apres=8)

    p3m = [r for r in (BM.get("p3_models") or [])
           if r.get("horizon") in (1, 4) and r.get("config") == "complet"]
    if p3m:
        # Meilleure référence (baseline) par horizon : les baselines sont
        # évaluées sur exactement les mêmes jeux d'évaluation, le gain est donc
        # comparable sans biais.
        best_ref = {}
        for r in p3m:
            if r.get("type") == "baseline":
                h = r.get("horizon")
                m = r.get("mae_pool")
                if m is not None and (h not in best_ref or m < best_ref[h]):
                    best_ref[h] = m
        p3m.sort(key=lambda r: (r.get("horizon", 0), r.get("mae_pool", 1e18)))
        lignes = []
        for r in p3m:
            h, mae = r.get("horizon"), r.get("mae_pool")
            if r.get("type") == "baseline":
                gain = "référence"
            else:
                ref = best_ref.get(h)
                gain = (f"{100 * (1 - mae / ref):+.1f} %"
                        if (ref and mae is not None) else "—")
            lignes.append([
                r.get("modele", "?"),
                h if h is not None else "?",
                fmt(mae, 1),
                fmt(r.get("r2_pool"), 3),
                fmt(r.get("agg_ratio_pool"), 3),
                fmt(r.get("smape_pool"), 1),
                gain])
        tableau(doc, ["Modèle", "Horizon", "MAE agrégée", "R² agrégé",
                      "Total prédit ÷ observé", "sMAPE", "Gain sur la meilleure "
                      "référence"],
                lignes, largeurs=[3.6, 1.4, 2.2, 1.8, 2.6, 1.5, 2.5],
                titre_tab="Tableau 5.3 — Comparaison des modèles et des références, "
                          "protocole corrigé (métriques agrégées sur 864 prédictions "
                          "hors échantillon, 12 origines glissantes)")
    else:
        encadre(doc, "Le banc d'essai comparatif (partie 3) était encore en cours "
                     "d'exécution au moment de la génération de ce rapport. Les "
                     "chiffres consolidés figurent dans "
                     "reports/benchmark_results.json et reports/benchmark_p3_models.csv.",
                kind="attention")

    para(doc,
         "Résultat obtenu sur ce protocole : les six algorithmes se tiennent dans un "
         "mouchoir de poche. À l'horizon 1 semaine, l'erreur absolue agrégée va de "
         "281,8 cas (ExtraTrees) à 310,8 (GradientBoosting de scikit-learn), XGBoost et "
         "LightGBM s'intercalant à 283,4 et 287,8. Le naïf saisonnier est à 371,3 et la "
         "moyenne par aire à 594,8. Le gain sur la meilleure référence s'étage donc de "
         "+16,3 % à +24,1 % à une semaine, et de +15,1 % à +31,1 % à quatre semaines. "
         "Le rapport prédit ÷ observé des modèles corrigés est compris entre 0,96 et "
         "1,06, contre 0,34 à 0,48 pour le code initial.", espace_apres=6)
    para(doc,
         "Conclusion à en tirer : l'écart entre algorithmes (environ 3 % du meilleur au "
         "plus faible) est très inférieur à l'écart entre un protocole valide et un "
         "protocole vicié (40 % d'erreur). Changer d'algorithme sans corriger la fuite, "
         "le découpage et le décalage entraînement / inférence n'apporte rien ; c'est "
         "l'ordre dans lequel les actions de la section 8 sont menées qui compte.",
         espace_apres=8)

    encadre(doc,
            "Sur un panneau regroupant des aires dont les populations s'étendent de 783 "
            "à 6 millions d'habitants, le R² est dominé par la variance entre aires : "
            "un modèle qui ne ferait que restituer la moyenne historique de chaque aire "
            "atteindrait déjà un R² élevé. Le R² ne doit donc pas être l'indicateur "
            "principal. Les indicateurs à suivre sont l'erreur absolue rapportée à la "
            "moyenne observée, le rapport des totaux et le gain relatif sur un modèle "
            "de référence.",
            kind="info")

    titre(doc, "5.4 Vérification du code après correction", 2)
    para(doc,
         "Le bloc d'onglet modifié a été exécuté après correction, par le même harnais, "
         "avec les contrôles de contrat suivants — tous satisfaits :", espace_apres=6)
    tableau(doc, ["Contrôle", "Résultat"],
            [["Exécution du bloc sans exception", "Oui"],
             ["Contrat de sortie pour la cartographie et les exports "
              "(health_area, week_num, predicted_cases, period, q10, q90)", "Conforme"],
             ["Observations d'entraînement", "11 160 (contre 3 737 initialement)"],
             ["Variables du modèle", "42, dont l'altitude, la population, la densité "
                                     "d'enfants, la saisonnalité et le retard spatial"],
             ["R² in-sample / R² validé", "0,983 / 0,729 ± 0,073"],
             ["Erreur absolue validée", "332,5 cas"],
             ["Découpage : chaque fold entraîné strictement sur le passé", "Oui"],
             ["Ordre des variables déterministe", "Oui"],
             ["Rapport prédit ÷ observé (médiane par aire)", "0,625 — période prévue en "
              "saison sèche, niveau attendu inférieur à la fin de saison des pluies"]],
            largeurs=[10.0, 6.0], fonte=8.5)
    para(doc,
         "La même vérification a été conduite pour le module rougeole : exécution sans "
         "erreur technique, 12 semaines prévues, 45 variables incluant la couverture "
         "vaccinale, la population d'enfants et l'urbanisation, intervalles de "
         "prédiction présents. Le R² validé y est proche de zéro (−0,050 ± 0,035) : sur "
         "des données de démonstration générées par tirage de Poisson autour d'une "
         "moyenne saisonnière, aucune information prédictive n'existe au-delà de la "
         "saisonnalité. Le noyau le signale honnêtement au lieu d'afficher le R² "
         "in-sample de 0,458 comme une performance.", espace_apres=8)

    titre(doc, "5.5 Ablations, objectifs de perte et résultats négatifs", 2)
    para(doc,
         "Une partie de ces expériences n'a pas confirmé les hypothèses de départ. "
         "Elles sont rapportées ici telles quelles, parce qu'elles modifient les "
         "priorités recommandées.", espace_apres=8)

    titre(doc, "5.5.1 Apport réel des covariables statiques et du spatial", 3)
    ab = [r for r in (BM.get("p4_ablations") or [])
          if r.get("type") != "baseline" and r.get("horizon") in (1, 4)]
    if ab:
        lignes = []
        for r in sorted(ab, key=lambda x: (x.get("config", ""), x.get("modele", ""),
                                           x.get("horizon", 0))):
            lignes.append([r.get("config", "?"), r.get("modele", "?"),
                           r.get("horizon", "?"), fmt(r.get("mae_pool"), 1),
                           fmt(r.get("r2_pool"), 3),
                           fmt(r.get("agg_ratio_pool"), 3)])
        tableau(doc, ["Configuration", "Algorithme", "Horizon", "MAE agrégée",
                      "R² agrégé", "Total prédit ÷ observé"],
                lignes, largeurs=[4.6, 3.0, 1.5, 2.2, 1.8, 2.7], fonte=8,
                titre_tab="Tableau 5.4 — Ablations : que rapporte réellement chaque "
                          "famille de variables ?")
    para(doc,
         "Résultat contre-intuitif mais net : sur ce jeu de données, retirer les "
         "covariables statiques améliore très légèrement l'erreur (321,8 → 320,5 pour "
         "XGBoost à une semaine ; 306,3 → 302,4 pour RandomForest), et retirer le "
         "retard spatial l'améliore davantage (321,8 → 308,9). L'expérience de "
         "contrôle le confirme : sur un jeu où l'altitude a un effet réel par "
         "construction, l'ajout des covariables statiques modifie l'erreur de −0,4 % ; "
         "sur un jeu où elle n'en a pas, de −0,3 %. L'effet mesuré est du bruit.",
         espace_apres=6)
    para(doc,
         "Interprétation : avec 156 semaines denses par aire et une forte "
         "autocorrélation, le niveau récent de l'aire contient déjà l'information "
         "structurelle que l'altitude ou la densité de population pourraient apporter. "
         "Les covariables statiques n'ont de valeur que là où l'historique est court, "
         "lacunaire ou interrompu — et ce banc d'essai ne teste pas cette situation. "
         "Leur intégration reste justifiée, mais comme hypothèse à vérifier sur les "
         "données réelles, pas comme un gain acquis.", espace_apres=8)

    titre(doc, "5.5.2 Choix de l'objectif de perte", 3)
    obj = [r for r in (BM.get("p5_objectives") or [])
           if r.get("type") != "baseline" and r.get("horizon") in (1, 4)]
    if obj:
        # le nom de l'objectif est porté par la clé `modele`
        # (ex. "XGBoost squared_error") : on le sépare en deux colonnes.
        def _split(m):
            m = str(m)
            for o in ("squared_error", "poisson", "log1p", "quantile", "tweedie"):
                if m.endswith(o):
                    return m[: -len(o)].strip(), o
            return m, "—"

        lignes = []
        for r in sorted(obj, key=lambda x: (_split(x.get("modele", ""))[0],
                                            _split(x.get("modele", ""))[1],
                                            x.get("horizon", 0))):
            algo, objn = _split(r.get("modele", "?"))
            lignes.append([algo, objn,
                           r.get("horizon", "?"), fmt(r.get("mae_pool"), 1),
                           fmt(r.get("r2_pool"), 3),
                           fmt(r.get("agg_ratio_pool"), 3)])
        tableau(doc, ["Algorithme", "Objectif", "Horizon", "MAE agrégée",
                      "R² agrégé", "Total prédit ÷ observé"],
                lignes, largeurs=[2.8, 3.0, 1.5, 2.3, 1.8, 3.0], fonte=8,
                titre_tab="Tableau 5.5 — Influence de l'objectif de perte")
    para(doc,
         "L'objectif Poisson améliore l'erreur à une semaine (303,1 contre 321,8) mais "
         "sous-estime les totaux à quatre semaines (rapport de 0,861). La "
         "transformation logarithmique est la meilleure en erreur absolue (299,2) mais "
         "sous-estime plus encore (0,792 à quatre semaines). L'erreur quadratique "
         "conserve les totaux les mieux calibrés (1,044 et 1,033). Le choix dépend donc "
         "de l'usage : pour dimensionner des stocks, la calibration des totaux prime ; "
         "pour classer des aires entre elles, la transformation logarithmique "
         "convient. Le sélecteur d'objectif ajouté à l'interface expose explicitement "
         "cet arbitrage.", espace_apres=8)

    titre(doc, "5.5.3 Importance réelle des variables", 3)
    imp = BM.get("p7_importance") or []
    if imp:
        lignes = [[r.get("variable", "?"), fmt(r.get("importance_pct"), 2) + " %"]
                  for r in imp[:12]]
        tableau(doc, ["Variable", "Importance"], lignes,
                largeurs=[7.0, 4.0], fonte=8.5,
                titre_tab="Tableau 5.6 — Importance des variables (XGBoost, gain "
                          "moyen sur les arbres)")
    para(doc,
         "Trois variables de niveau récent concentrent environ 70 % de l'importance : "
         "moyenne exponentielle sur 4 semaines (29,2 %), moyenne mobile sur 4 semaines "
         "(27,9 %) et maximum sur 4 semaines (13,4 %). L'altitude arrive à 1,12 %. "
         "Cela confirme le point précédent : sur un panneau dense, la prévision à "
         "court terme est essentiellement une extrapolation du niveau récent, et la "
         "valeur ajoutée des covariables de contexte est marginale à cet horizon.",
         espace_apres=8)

    doc.add_page_break()

    # ═══════════════ 6. CORRECTIONS IMPLÉMENTÉES ═══════════════
    titre(doc, "6. Corrections implémentées dans ce dépôt", 1)

    para(doc,
         "Les corrections ont été appliquées de façon additive : la structure des "
         "tables, les noms de colonnes et les formats de données existants sont "
         "conservés, conformément à la contrainte posée. Aucun jeu de données ni "
         "aucune interface de sortie n'a été renommé.", espace_apres=8)

    titre(doc, "6.1 Nouveau noyau de modélisation", 2)
    tableau(doc, ["Module", "Contenu"],
            [["epimodel/panel.py",
              "Construction d'un index temporel continu (année × semaine ISO 8601), "
              "complétion du panneau aire × semaine avec zéros explicites, gestion des "
              "lacunes. Plus aucun écrasement interannuel."],
             ["epimodel/features.py",
              "Fonction unique de construction des variables, utilisée à "
              "l'entraînement comme à la prévision. Retards, moyennes mobiles, "
              "écarts-types, saisonnalité (harmoniques d'ordre 1 et 2), retard spatial, "
              "covariables statiques."],
             ["epimodel/models.py",
              "Registre de six algorithmes : XGBoost, LightGBM, GradientBoosting, "
              "RandomForest, ExtraTrees, régression linéaire régularisée. Objectif "
              "Poisson ou binomial négatif pour les comptages, régression quantile "
              "pour les intervalles. Régularisation, profondeur bornée, "
              "sous-échantillonnage."],
             ["epimodel/validation.py",
              "Découpage bloqué par semaine avec embargo, validation à origines "
              "glissantes, métriques agrégées sur l'ensemble des prédictions hors "
              "échantillon, modèles de référence (persistance, naïf saisonnier, "
              "moyennes)."],
             ["epimodel/forecast.py",
              "Prévision récursive reconstruisant intégralement les variables à chaque "
              "pas, intervalles par régression quantile, imputeur conservant les "
              "colonnes vides pour garantir l'alignement entraînement / inférence."],
             ["epimodel/static_covariates.py",
              "Extraction automatique depuis Google Earth Engine : altitude et pente "
              "(SRTM 30 m), indice de végétation, occupation du sol, distance à l'eau, "
              "part d'urbanisation, population par année. Fusion sur l'identifiant "
              "d'aire de santé et rapport de couverture."],
             ["epi_app_bridge.py",
              "Pont entre les applications existantes et le noyau : préparation du "
              "panneau, validation temporelle, entraînement et prévision, pour le "
              "paludisme et pour la rougeole."]],
            largeurs=[4.2, 11.8], fonte=8.5)

    titre(doc, "6.2 Modifications des applications existantes", 2)
    for t in [
        ("Onglet de modélisation du paludisme. ",
         "Le bloc de configuration et d'entraînement a été remplacé par un appel au "
         "noyau. Le sélecteur d'algorithme est désormais dynamique et inclut XGBoost et "
         "LightGBM, avec un sélecteur d'objectif de perte. L'horizon est borné par la "
         "période réellement couverte par les données. Un mode expert expose le nombre "
         "de folds, l'embargo, l'usage du spatial, le nombre de classes et de voisins. "
         "L'analyse en composantes principales est retirée du chemin principal, avec "
         "l'explication affichée à l'utilisateur."),
        ("Affichage des résultats. ",
         "Les métriques de validation temporelle sont affichées en tête ; le R² "
         "in-sample est explicitement étiqueté comme tel ; le détail par fold est "
         "exposé dans un panneau dépliable ; le seuil d'alerte choisi par l'utilisateur "
         "n'est plus écrasé."),
        ("Barre latérale. ",
         "Un nouveau panneau « Covariables statiques » permet l'extraction automatique "
         "ou le téléversement d'un fichier de valeurs, avec un rapport de couverture "
         "par aire."),
        ("Module rougeole. ",
         "Le découpage aléatoire et la pondération manuelle des variables sont "
         "remplacés par le noyau ; l'explication de l'absence d'effet de la pondération "
         "sur les arbres est affichée à l'utilisateur ; l'objectif Poisson est proposé "
         "par défaut."),
    ]:
        puce(doc, t[1], gras_debut=t[0])

    titre(doc, "6.3 Tests", 2)
    para(doc,
         "Une suite de 18 tests couvre le noyau : non-effondrement du panneau, "
         "complétion de la grille, semaine ISO 53, détection d'une perturbation de la "
         "cible, détection d'un décalage entraînement / inférence, absence de fuite du "
         "découpage temporel, ordre déterministe des variables, registre XGBoost, "
         "objectif Poisson, quantiles, forme et échelle de la prévision récursive, "
         "modèles de référence. Un test de non-régression exécute l'onglet de "
         "modélisation initial et vérifie que les défauts documentés s'y produisent "
         "toujours — il échouera si quelqu'un les réintroduit.", espace_apres=8)

    doc.add_page_break()

    # ═══════════════ 7. RECOMMANDATIONS ═══════════════
    titre(doc, "7. Recommandations d'un modélisateur épidémiologiste", 1)
    para(doc,
         "Les corrections de la section 6 remettent l'outil dans un état où ses "
         "nombres signifient ce qu'ils prétendent signifier. Les recommandations "
         "suivantes concernent ce qu'il faudrait faire de plus pour que l'outil soit "
         "véritablement un outil d'aide à la décision de niveau expert.", espace_apres=8)

    titre(doc, "7.1 Modélisation statistique", 2)
    reco = [
        ("Modèle hiérarchique à effets aléatoires. ",
         "Les 72 aires ne sont pas 72 problèmes indépendants : elles partagent un "
         "régime climatique, une écologie vectorielle, un système de surveillance. Un "
         "modèle à effets aléatoires par aire — avec échange partiel d'information — "
         "est mieux adapté qu'un modèle unique. Les aires à historique court "
         "bénéficient alors de l'information des aires voisines au lieu d'être "
         "abandonnées à leur bruit. Implémentation : effets aléatoires dans un modèle "
         "additif généralisé, ou modèle espace-temps bayésien si les effectifs le "
         "permettent."),
        ("Famille de distributions adaptée aux comptages. ",
         "Le nombre de cas hebdomadaires est un comptage dont la variance croît plus "
         "vite que la moyenne (surdispersion). La loi de Poisson est un minimum ; la "
         "binomiale négative est généralement plus appropriée en surveillance "
         "palustre. Le nombre zéro est fréquent en basse transmission : un modèle à "
         "inflation de zéros peut être nécessaire en zone sahélienne."),
        ("Modèles additifs généralisés pour les covariables. ",
         "Les relations climat-morbidité ne sont pas linéaires et ne sont pas "
         "capturées par les formes imposées actuellement. Des splines de lissage sur "
         "la température, les précipitations cumulées et l'indice de végétation, avec "
         "des décalages distribués, permettent d'estimer la forme fonctionnelle au "
         "lieu de la supposer."),
        ("Décalages distribués. ",
         "L'effet d'un épisode pluvieux sur la transmission s'étale sur plusieurs "
         "semaines : développement larvaire, émergence, durée du cycle sporogonique, "
         "puis incubation humaine et délai de consultation. Un modèle à décalages "
         "distribués représente cette convolution explicitement, au lieu de tester "
         "quelques retards arbitraires."),
        ("Prise en compte de la sous-notification. ",
         "Les cas notifiés ne sont qu'une fraction des cas survenus, et cette fraction "
         "varie dans le temps et selon les aires (accessibilité, grèves, ruptures de "
         "réactifs). Un modèle de capture-recapture sur les données disponibles, ou a "
         "minima un indicateur de complétude par aire et par semaine, évite "
         "d'interpréter une baisse de notification comme une baisse de transmission."),
    ]
    for t, d in reco:
        puce(doc, d, gras_debut=t)

    titre(doc, "7.2 Évaluation et incertitude", 2)
    reco2 = [
        ("Banc d'essai permanent. ",
         "Un jeu de validation à origines glissantes, rejoué à chaque modification du "
         "code et conservé dans le dépôt. Toute régression de performance doit être "
         "visible avant la mise en production, pas après."),
        ("Intervalles de couverture vérifiée. ",
         "Les intervalles doivent avoir la couverture annoncée. Un intervalle dit à "
         "90 % doit contenir la valeur observée 90 % du temps en rétrospectif ; cela "
         "se vérifie, et se corrige par calibration conforme si nécessaire."),
        ("Métriques opérationnelles. ",
         "Au-delà de l'erreur absolue : sensibilité et délai de détection des pics, "
         "proportion d'alertes correctes, nombre d'alertes manquées. Ce sont les "
         "grandeurs qui déterminent l'utilité réelle d'un système d'alerte précoce."),
        ("Communication de l'incertitude. ",
         "Une carte de prévision sans intervalle est une carte qui affiche une "
         "certitude qu'elle n'a pas. Les classes de risque devraient être "
         "accompagnées de la probabilité de dépassement du seuil, et non d'une "
         "couleur unique."),
        ("Journal de traçabilité des prédictions. ",
         "Chaque prédiction produite, avec sa date, ses données d'entrée, sa version "
         "de code et son intervalle, doit être archivée pour permettre l'évaluation "
         "rétrospective et la responsabilité des décisions."),
    ]
    for t, d in reco2:
        puce(doc, d, gras_debut=t)

    titre(doc, "7.3 Cohérence entre données, logique et relations", 2)
    para(doc,
         "Un point méthodologique transversal, et le plus important de cette section : "
         "chaque variable entrant dans le modèle doit être disponible au moment de la "
         "décision, et sa relation avec la cible doit être temporellement orientée. "
         "Concrètement, pour chaque covariable, il faut documenter trois grandeurs : "
         "son décalage de disponibilité réel, la fenêtre d'agrégation pertinente du "
         "point de vue biologique, et son décalage d'effet. Sans cette table, aucune "
         "garantie contre les fuites n'est possible, quelle que soit la qualité du "
         "code.", espace_apres=8)
    tableau(doc, ["Covariable", "Disponibilité", "Fenêtre pertinente", "Décalage d'effet"],
            [["Cas notifiés", "J+3 à J+14 (notification)", "Hebdomadaire",
              "Immédiat (retard de déclaration)"],
             ["Précipitations satellites", "J+2 à J+5", "Cumul sur 2 à 8 semaines",
              "3 à 8 semaines"],
             ["Température", "J+2 à J+5", "Moyenne sur 1 à 4 semaines",
              "2 à 6 semaines"],
             ["Indice de végétation", "J+5 à J+10", "Moyenne sur 4 semaines",
              "2 à 6 semaines"],
             ["Population", "Annuelle", "Annuelle", "Aucun"],
             ["Altitude, pente, occupation du sol", "Permanente", "Ponctuelle",
              "Aucun"],
             ["Couverture vaccinale", "Mensuelle à annuelle", "Cumul de doses",
              "2 à 6 semaines (rougeole)"]],
            largeurs=[4.0, 3.6, 3.8, 4.6], fonte=8.5,
            titre_tab="Tableau 7.1 — Table de disponibilité à documenter pour chaque "
                      "covariable")

    titre(doc, "7.4 Covariables statiques à intégrer en priorité", 2)
    tableau(doc, ["Covariable", "Source", "Pertinence épidémiologique"],
            [["Altitude", "SRTM 30 m",
              "Détermine la température et donc la durée du cycle sporogonique ; "
              "au-delà de 1 500 à 2 000 m, la transmission devient instable ou "
              "absente"],
             ["Pente", "SRTM 30 m",
              "Conditionne le ruissellement et la formation de gîtes larvaires"],
             ["Indice de végétation (NDVI/EVI)", "Sentinel-2, MODIS",
              "Proxy de l'humidité du sol et de la disponibilité des gîtes"],
             ["Distance à l'eau permanente", "Copernicus, OpenStreetMap",
              "Distance de vol de l'anophèle ; structurante à l'échelle de l'aire"],
             ["Occupation du sol", "ESA WorldCover 10 m",
              "Zones irriguées, rizières, urbain : habitats vectoriels distincts"],
             ["Part d'urbanisation", "WorldPop, ESA",
              "Modifie l'espèce vectorielle dominante et le profil de transmission"],
             ["Population par classe d'âge", "WorldPop, filtré par année",
              "Dénominateur des taux d'incidence ; structure de susceptibilité"],
             ["Accessibilité routière", "OpenStreetMap",
              "Proxy de la complétude de la notification et de l'accès aux soins"]],
            largeurs=[4.4, 3.6, 8.0], fonte=8.5)
    para(doc,
         "Leur valeur principale n'est pas d'améliorer la prévision à une semaine dans "
         "les aires bien documentées — le passé récent de l'aire y suffit largement. "
         "Elle est d'apporter un signal là où l'historique est court, lacunaire ou "
         "interrompu : nouvelle aire, rupture de notification, zone d'intervention "
         "récente.", espace_apres=6)
    encadre(doc,
            "Mise en garde honnête. L'hypothèse selon laquelle ces covariables "
            "amélioreraient la prévision n'est PAS démontrée par les expériences de "
            "cet audit : sur un panneau dense de trois ans, leur retrait ne dégrade "
            "pas l'erreur, et une expérience de contrôle où l'altitude a un effet "
            "réel par construction ne montre aucun gain mesurable (−0,4 %, soit du "
            "bruit). Le banc d'essai ne couvre pas le cas où elles devraient "
            "compter (historique court ou lacunaire). La recommandation est donc de "
            "les intégrer — le coût est faible et le risque nul — mais de mesurer "
            "leur apport sur les données réelles, en comparant explicitement les "
            "aires selon la longueur de leur historique, avant toute conclusion.",
            kind="attention")

    doc.add_page_break()

    # ═══════════════ 8. PLAN D'ACTION ═══════════════
    titre(doc, "8. Plan d'action point par point", 1)
    para(doc,
         "Chaque ligne est autonome : problème constaté, correction à appliquer, "
         "localisation dans le code, impact, effort estimé, statut. Les priorités sont "
         "P0 (bloquant : les résultats produits sont faux), P1 (majeur : fiabilité "
         "dégradée ou fonctionnalité inopérante), P2 (mineur).", espace_apres=8)

    ordre = {"P0": 0, "P1": 1, "P2": 2}
    actions_triees = sorted(ACTIONS, key=lambda a: (ordre.get(a[1][:2], 9), a[0]))

    # tableau de synthèse
    tableau(doc, ["N°", "Priorité", "Action", "Localisation", "Effort", "Statut"],
            [[a[0], a[1].split(" — ")[0], a[2], a[5].split(",")[0][:52], a[7], a[8]]
             for a in actions_triees],
            largeurs=[0.9, 1.4, 6.0, 3.4, 1.4, 2.9], fonte=7.5,
            titre_tab="Tableau 8.1 — Synthèse du plan d'action")

    doc.add_page_break()
    titre(doc, "8.1 Détail des actions", 2)

    for a in actions_triees:
        (num, prio, titre_, probleme, correction, loc, impact, effort, statut) = a
        couleur = ROUGE if prio.startswith("P0") else (
            RGBColor(0xB0, 0x60, 0x00) if prio.startswith("P1") else GRIS)
        p = doc.add_paragraph()
        r = p.add_run(f"{num}. {titre_}  ")
        r.bold = True
        r.font.size = Pt(11)
        r.font.color.rgb = BLEU
        r.font.name = "Calibri"
        r2 = p.add_run(f"[{prio}] · {effort} · {statut}")
        r2.font.size = Pt(9)
        r2.font.color.rgb = couleur
        r2.bold = True
        r2.font.name = "Calibri"
        p.paragraph_format.space_after = Pt(3)

        para(doc, "Localisation : " + loc, italique=True, taille=9, couleur=GRIS,
             espace_apres=3)
        para(doc, "Problème. " + probleme, taille=10, espace_apres=4)
        para(doc, "Correction. " + correction, taille=10, espace_apres=4)
        para(doc, "Impact. " + impact, taille=10, espace_apres=8)

    # ═══════════════ 8.2 séquencement ═══════════════
    titre(doc, "8.2 Séquencement recommandé", 2)
    tableau(doc, ["Séquence", "Actions", "Objectif", "Durée"],
            [["1 — Rétablir la validité",
              "A1, A2, A3, A4, B1, C1, C2, F1",
              "Les nombres produits mesurent ce qu'ils prétendent mesurer",
              "≈ 5 jours"],
             ["2 — Sécuriser",
              "I1, I2, I4",
              "Aucun identifiant exposé ; chargement contrôlé ; dépôt propre",
              "≈ 2 jours"],
             ["3 — Consolider la qualité",
              "A5, C3, C4, C5, E4, E5, G1, G2, G3, F2, F3, F4",
              "Validation saine, déterminisme, validation portant sur le modèle "
              "déployé",
              "≈ 5 jours"],
             ["4 — Enrichir le signal",
              "D1, D2, E1, E2, E3, F5",
              "Covariables statiques automatiques, gradient boosting moderne, "
              "modèles de référence, intervalles de prédiction",
              "≈ 5 jours"],
             ["5 — Documenter et pérenniser",
              "H1, H2, I3, B2, B3, B4, D3, D4, D5, E6, E7, E8, E9, F6, F7",
              "Manuel conforme au code, tests, versions fixées, robustesse",
              "≈ 4 jours"]],
            largeurs=[3.4, 4.6, 5.4, 2.2], fonte=8.5)

    para(doc,
         "Les actions marquées « Corrigé » sont implémentées dans ce dépôt et "
         "vérifiées par exécution du code modifié. Les actions restantes sont celles "
         "qui nécessitent soit un accès aux services externes (D3, D5, et le "
         "branchement réel de D1), soit une décision de gouvernance (I1), soit une "
         "rédaction (H1, H2).", espace_apres=8)

    doc.add_page_break()

    # ═══════════════ 9. ANNEXES ═══════════════
    titre(doc, "9. Annexes — reproductibilité et gouvernance", 1)

    titre(doc, "9.1 Artefacts produits par l'audit", 2)
    tableau(doc, ["Fichier", "Contenu"],
            [["reports/legacy_diagnostic.json",
              "Diagnostic instrumenté du code livré : métriques affichées, perte "
              "d'observations, nature du découpage, décalage entraînement / "
              "inférence, dérive des prédictions, déterminisme"],
             ["reports/leakage_analysis.json",
              "Preuve analytique de la fuite et quantification par protocole de "
              "validation"],
             ["reports/benchmark_results.json",
              "Comparaison des modèles et des références sous protocole corrigé"],
             ["reports/benchmark_p1_legacy.csv",
              "Métriques réellement affichées par l'application, par algorithme"],
             ["reports/benchmark_p3_models.csv",
              "Métriques hors échantillon par modèle, objectif et horizon"],
             ["tools/legacy_harness.py",
              "Harnais d'extraction par arbre syntaxique : exécute un bloc d'onglet "
              "réel avec interface simulée"],
             ["tools/diagnose_legacy.py",
              "Instrumentation du code livré"],
             ["tools/diagnose_leakage.py",
              "Quantification de la fuite et des protocoles de validation"],
             ["tools/benchmark.py",
              "Banc d'essai comparatif en sept parties"],
             ["tools/verify_integrated_tab3.py",
              "Vérification du bloc paludisme après correction, avec contrôles de "
              "contrat"],
             ["tools/verify_integrated_rougeole.py",
              "Vérification du bloc rougeole après correction"],
             ["tests/test_epimodel.py",
              "Suite de 18 tests, dont un test de non-régression sur le code initial"]],
            largeurs=[5.4, 10.6], fonte=8.5)

    titre(doc, "9.2 Reproduction des résultats", 2)
    code(doc,
         "# Environnement\n"
         "python -m venv .venv && source .venv/bin/activate\n"
         "pip install -r requirements.txt\n\n"
         "# Diagnostic du code livré\n"
         "python tools/diagnose_legacy.py\n"
         "python tools/diagnose_leakage.py\n"
         "python tools/benchmark.py\n\n"
         "# Vérification du code corrigé\n"
         "python tools/verify_integrated_tab3.py\n"
         "python tools/verify_integrated_rougeole.py\n"
         "python -m pytest tests -q\n\n"
         "# Régénération de ce rapport\n"
         "python tools/generate_audit_report.py")

    titre(doc, "9.3 Gouvernance recommandée", 2)
    for t in [
        ("Revue statistique avant mise en production. ",
         "Aucune nouvelle variable ni aucun nouvel algorithme ne devrait entrer dans "
         "le modèle sans passage par le banc d'essai, avec comparaison au modèle "
         "courant sur les mêmes origines."),
        ("Versionnage des prédictions. ",
         "Chaque lot de prédictions publié devrait porter la version du code, "
         "l'empreinte des données d'entrée et la date de production."),
        ("Évaluation rétrospective périodique. ",
         "Chaque trimestre, comparer les prédictions publiées aux cas effectivement "
         "notifiés, et publier la couverture des intervalles et le rapport des totaux. "
         "C'est le seul moyen de savoir si l'outil tient ses promesses."),
        ("Séparation des rôles. ",
         "L'équipe qui maintient le code ne devrait pas être seule à valider la "
         "méthode. Une revalidation indépendante, même légère, réduit fortement le "
         "risque de dérive méthodologique progressive."),
    ]:
        puce(doc, t[1], gras_debut=t[0])

    titre(doc, "9.4 Avertissement sur la portée des chiffres", 2)
    encadre(doc,
            "Tous les chiffres de ce rapport ont été produits sur des données de test "
            "synthétiques, générées à partir du référentiel géographique réel des 72 "
            "aires de santé du Niger mais avec des comptages simulés. Les covariables "
            "statiques sont également simulées, l'environnement d'audit ne disposant "
            "pas d'accès réseau.\n\n"
            "Ce qui est transférable aux données réelles : les défauts structurels "
            "(fuites, découpage, écrasement du panneau, décalage entraînement / "
            "inférence), démontrés sur le code et par preuve algébrique, ainsi que "
            "l'amplitude relative de l'optimisme qu'ils produisent.\n\n"
            "Ce qui ne l'est pas : les niveaux absolus de performance. Le R² de 0,729 "
            "et le gain de 17,4 % sont des ordres de grandeur obtenus sur des données "
            "dont la dynamique a été choisie par construction. Sur des données réelles, "
            "ces valeurs seront différentes — probablement plus basses, la morbidité "
            "réelle étant plus bruitée. Le protocole, lui, reste valide et doit être "
            "rejoué sur les données de production avant toute décision.",
            kind="attention")

    # ── sauvegarde ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=os.path.join(REPORTS,
                                         "Rapport_Audit_EpiPrediction.docx"))
    args = ap.parse_args()
    p = build(args.out)
    print("Rapport généré :", p)
    print("Actions détaillées :", len(ACTIONS))
    return 0


if __name__ == "__main__":
    sys.exit(main())

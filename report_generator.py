# ============================================================
# report_generator.py  v2.0
# Module de génération de rapports épidémiologiques complets
# Formats : PDF (fpdf2) · Word (python-docx) · HTML imprimable
# Branding MSF — Rouge #E4032E / Bleu Marine #1A2D4F
#
# SECTIONS SUPPORTÉES
#   1. Dashboard         — KPIs globaux, évolution hebdo, répartition
#   2. Cartographie      — Top aires à risque, tableau géographique
#   3. Modélisation      — Algo, variables, paramètres, importance
#   4. Validation        — Métriques CV, graphiques obs/prédit, résidus
#   5. Recommandations   — Synthèse terrain automatique
#
# USAGE
#   from report_generator import rapport_streamlit_widget
#
#   rapport_streamlit_widget(
#       title          = "Surveillance Paludisme — Niger 2024",
#       subtitle       = "Rapport de validation rétrospective",
#       maladie        = "Paludisme",
#       # ── sections facultatives (passer None pour ignorer) ──
#       dashboard_data = {
#           "kpis"          : {"Cas totaux": "12 450", ...},
#           "evolution_fig" : <Plotly Figure>,
#           "top_aires"     : pd.DataFrame,
#       },
#       carto_data     = {
#           "top_risque"  : pd.DataFrame,   # colonnes : Aire, Cas, Incidence, Risque
#           "carte_fig"   : <Plotly Figure>,
#       },
#       model_data     = {
#           "algorithme"   : "RandomForestRegressor",
#           "parametres"   : {"n_estimators": 200, ...},
#           "variables"    : ["pluie_lag1", "temp_moy", ...],
#           "importance_fig": <Plotly Figure>,
#       },
#       validation_data = {
#           "metrics"        : {"MAE": "3.2", ...},
#           "interpretations": ["Le modèle explique 82 % ..."],
#           "figures"        : [fig_obs_pred, fig_residus, ...],
#           "tables"         : [([cols], [rows]), ...],
#       },
#       key_prefix = "rapport_paludisme",
#   )
# ============================================================

import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import streamlit as st


# ─────────────────────────────────────────────────────────────
# UTILITAIRES COMMUNS
# ─────────────────────────────────────────────────────────────

def _fig_to_png_bytes(fig, width=900, height=380):
    """Convertit une figure Plotly en bytes PNG via kaleido."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


EMOJI_MAP = {
    "\u2014": "-", "\u2013": "-",
    "\u2019": "'", "\u2018": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2026": "...", "\u2022": "-",
    "\u2192": ">",
    "\u2265": ">=", "\u2264": "<=", "\u2260": "!=",
    "\u00B0": " deg", "\u200b": "", "\u00A0": " ",
    "\U0001F7E2": "OK ", "\U0001F7E1": "~ ",
    "\U0001F7E0": "! ", "\U0001F534": "X ",
    "\u2705": "OK ", "\u26A0": "! ", "\uFE0F": "",
    "\u2B06": "^", "\u2B07": "v",
    "\U0001F4CA": "", "\U0001F50D": "",
    "\U0001F4C8": "", "\U0001F4CB": "",
    "\U0001F517": "", "\U0001F5FA": "",
    "\U0001F9EA": "", "\U0001F3AF": "",
    "🟢": "OK ", "🟡": "~ ", "🟠": "! ", "🔴": "X ",
    "✅": "OK ", "⚠️": "! ", "⬆️": "^", "⬇️": "v",
    "📊": "", "🔍": "", "📈": "", "📋": "",
}


def _clean(text: str) -> str:
    """Nettoie le texte : emojis → ASCII, encodage latin-1 safe."""
    for char, repl in EMOJI_MAP.items():
        text = text.replace(char, repl)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _clean_html(text: str) -> str:
    """Nettoie uniquement les emojis (garde l'UTF-8 pour HTML)."""
    for char, repl in EMOJI_MAP.items():
        text = text.replace(char, repl)
    return text


def _df_to_rows(df, max_rows=25):
    """Convertit un DataFrame en (colonnes, lignes) pour les rapports."""
    if df is None or len(df) == 0:
        return [], []
    cols = list(df.columns)
    rows = [list(r) for r in df.head(max_rows).itertuples(index=False)]
    return cols, rows


# ─────────────────────────────────────────────────────────────
# FORMAT 1 — PDF (fpdf2)
# ─────────────────────────────────────────────────────────────

def generate_pdf(title, subtitle, maladie="Paludisme",
                 dashboard_data=None, carto_data=None,
                 model_data=None, validation_data=None):
    """
    Génère un rapport PDF multi-sections avec fpdf2.
    Retourne les bytes du PDF ou None si erreur.
    """
    try:
        from fpdf import FPDF, XPos, YPos

        MSF_RED   = (228,  3,  46)
        MSF_NAVY  = ( 26, 45,  79)
        DARK_GRAY = ( 50, 50,  50)
        LIGHT_BG  = (248, 249, 250)
        GREEN     = ( 39, 174,  96)
        ORANGE    = (230, 126,  34)

        class PDFReport(FPDF):
            def header(self):
                self.set_fill_color(*MSF_RED)
                self.rect(0, 0, 210, 12, "F")
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(255, 255, 255)
                self.set_xy(8, 2)
                self.cell(120, 8, _clean("MSF - Medecins Sans Frontieres | Surveillance Epidemiologique"))
                self.set_xy(138, 2)
                self.set_font("Helvetica", "", 8)
                self.cell(60, 8, _clean(datetime.now().strftime("%d/%m/%Y")), align="R")
                self.set_text_color(*DARK_GRAY)

            def footer(self):
                self.set_y(-12)
                self.set_fill_color(*MSF_NAVY)
                self.rect(0, self.get_y(), 210, 12, "F")
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(255, 255, 255)
                self.cell(0, 6,
                    _clean(f"Page {self.page_no()} | {title} | CONFIDENTIEL - Usage interne"),
                    align="C")

            def section_title(self, txt, color=None):
                self.ln(5)
                c = color or MSF_NAVY
                self.set_fill_color(*c)
                self.set_text_color(255, 255, 255)
                self.set_font("Helvetica", "B", 11)
                self.cell(0, 9, _clean(f"  {txt}"),
                          new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
                self.set_text_color(*DARK_GRAY)
                self.ln(2)

            def subsection(self, txt):
                self.ln(3)
                self.set_font("Helvetica", "B", 9)
                self.set_text_color(*MSF_NAVY)
                self.set_draw_color(*MSF_NAVY)
                self.set_line_width(0.4)
                self.cell(0, 6, _clean(txt),
                          new_x=XPos.LMARGIN, new_y=YPos.NEXT, border="B")
                self.set_text_color(*DARK_GRAY)
                self.ln(2)

            def kv_row(self, key, val, shade=False):
                self.set_fill_color(*(LIGHT_BG if shade else (255, 255, 255)))
                self.set_font("Helvetica", "B", 9)
                self.cell(70, 7, _clean(f"  {key}"), fill=True)
                self.set_font("Helvetica", "", 9)
                self.cell(0, 7, _clean(f"  {val}"),
                          fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            def table(self, cols, rows, col_w=None, max_rows=30):
                n = len(cols)
                cw = col_w or min(182 / max(n, 1), 40)
                self.set_fill_color(*MSF_NAVY)
                self.set_text_color(255, 255, 255)
                self.set_font("Helvetica", "B", 7)
                for c in cols:
                    self.cell(cw, 6, _clean(str(c)[:20]), border=1, fill=True)
                self.ln()
                self.set_text_color(*DARK_GRAY)
                self.set_font("Helvetica", "", 7)
                for ri, row in enumerate(rows[:max_rows]):
                    self.set_fill_color(*(LIGHT_BG if ri % 2 == 0 else (255, 255, 255)))
                    for cell in row:
                        self.cell(cw, 5, _clean(str(cell))[:20], border=1, fill=True)
                    self.ln()
                self.ln(3)

        pdf = PDFReport()
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.set_margins(12, 18, 12)

        # ── PAGE DE GARDE ──────────────────────────────────────
        pdf.add_page()
        pdf.ln(14)
        pdf.set_fill_color(*MSF_RED)
        pdf.rect(12, pdf.get_y(), 186, 2, "F")
        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 26)
        pdf.set_text_color(*MSF_NAVY)
        pdf.multi_cell(0, 13, _clean(title), align="C")
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 13)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 8, _clean(subtitle), align="C")
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(130, 130, 130)
        pdf.cell(0, 6, _clean(f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M UTC')}"),
                 align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)
        pdf.set_fill_color(*MSF_RED)
        pdf.rect(12, pdf.get_y(), 186, 2, "F")
        pdf.ln(12)

        # Encadré récapitulatif maladie
        pdf.set_fill_color(*LIGHT_BG)
        pdf.set_draw_color(*MSF_NAVY)
        pdf.set_line_width(0.6)
        pdf.rect(20, pdf.get_y(), 170, 28, "FD")
        pdf.set_xy(24, pdf.get_y() + 4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*MSF_NAVY)
        pdf.cell(0, 6, _clean(f"Maladie surveillee : {maladie}"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(24)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK_GRAY)
        sections_actives = []
        if dashboard_data:   sections_actives.append("Dashboard")
        if carto_data:       sections_actives.append("Cartographie")
        if model_data:       sections_actives.append("Modelisation")
        if validation_data:  sections_actives.append("Validation")
        pdf.cell(0, 6,
                 _clean("Sections : " + " | ".join(sections_actives) if sections_actives
                        else "Rapport de synthese epidemiologique"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(18)

        # Table des matières
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*MSF_NAVY)
        pdf.cell(0, 7, _clean("Contenu du rapport :"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK_GRAY)
        section_num = 1
        for sec, active in [
            ("Tableau de bord epidemiologique", dashboard_data),
            ("Analyse cartographique",           carto_data),
            ("Parametres de modelisation",        model_data),
            ("Validation retrospective",          validation_data),
            ("Recommandations terrain",           True),
            ("Limites et avertissements",         True),
        ]:
            if active:
                pdf.cell(0, 6, _clean(f"  {section_num}. {sec}"),
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                section_num += 1

        # ── SECTION 1 : DASHBOARD ──────────────────────────────
        if dashboard_data:
            pdf.add_page()
            pdf.section_title("1. Tableau de bord epidemiologique", MSF_NAVY)

            if dashboard_data.get("kpis"):
                pdf.subsection("Indicateurs cles de performance")
                for i, (k, v) in enumerate(dashboard_data["kpis"].items()):
                    pdf.kv_row(k, str(v), shade=(i % 2 == 0))
                pdf.ln(4)

            if dashboard_data.get("evolution_fig"):
                pdf.subsection("Evolution hebdomadaire")
                png = _fig_to_png_bytes(dashboard_data["evolution_fig"], 900, 320)
                if png:
                    pdf.image(io.BytesIO(png), x=12, w=185)
                pdf.ln(4)

            if dashboard_data.get("top_aires") is not None:
                pdf.subsection("Top aires de sante les plus touchees")
                cols, rows = _df_to_rows(dashboard_data["top_aires"], 20)
                if cols:
                    pdf.table(cols, rows)

        # ── SECTION 2 : CARTOGRAPHIE ──────────────────────────
        if carto_data:
            pdf.add_page()
            pdf.section_title("2. Analyse cartographique", MSF_NAVY)

            if carto_data.get("carte_fig"):
                pdf.subsection("Carte de distribution")
                png = _fig_to_png_bytes(carto_data["carte_fig"], 900, 500)
                if png:
                    pdf.image(io.BytesIO(png), x=12, w=185)
                pdf.ln(4)

            if carto_data.get("top_risque") is not None:
                pdf.subsection("Aires a risque eleve")
                cols, rows = _df_to_rows(carto_data["top_risque"], 20)
                if cols:
                    pdf.table(cols, rows)

        # ── SECTION 3 : MODÉLISATION ──────────────────────────
        if model_data:
            pdf.add_page()
            pdf.section_title("3. Parametres de modelisation", MSF_NAVY)

            if model_data.get("algorithme"):
                pdf.subsection("Algorithme utilise")
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(0, 6, _clean(f"  Algorithme : {model_data['algorithme']}"),
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(2)

            if model_data.get("parametres"):
                pdf.subsection("Hyperparametres")
                for i, (k, v) in enumerate(model_data["parametres"].items()):
                    pdf.kv_row(str(k), str(v), shade=(i % 2 == 0))
                pdf.ln(3)

            if model_data.get("variables"):
                pdf.subsection("Variables predictives utilisees")
                pdf.set_font("Helvetica", "", 9)
                for var in model_data["variables"]:
                    pdf.cell(0, 5, _clean(f"  - {var}"),
                             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(3)

            if model_data.get("importance_fig"):
                pdf.subsection("Importance des variables")
                png = _fig_to_png_bytes(model_data["importance_fig"], 900, 380)
                if png:
                    pdf.image(io.BytesIO(png), x=12, w=185)
                pdf.ln(4)

        # ── SECTION 4 : VALIDATION ────────────────────────────
        if validation_data:
            pdf.add_page()
            pdf.section_title("4. Validation retrospective", MSF_NAVY)

            metrics        = validation_data.get("metrics", {})
            interpretations = validation_data.get("interpretations", [])
            figures        = validation_data.get("figures", [])
            tables_data    = validation_data.get("tables", [])

            if metrics:
                pdf.subsection("Metriques de performance globales")
                for i, (k, v) in enumerate(metrics.items()):
                    pdf.kv_row(k, str(v), shade=(i % 2 == 0))
                pdf.ln(3)

            if interpretations:
                pdf.subsection("Interpretation automatique")
                pdf.set_font("Helvetica", "", 9)
                for line in interpretations:
                    pdf.multi_cell(0, 5, _clean(f"  - {line}"))
                pdf.ln(3)

            fig_titles = [
                "Observe vs Predit - serie temporelle",
                "Residus vs valeurs predites",
                "Distribution des residus",
                "Importance des variables",
            ]
            if figures:
                pdf.subsection("Visualisations")
                for i, fig in enumerate(figures):
                    if fig is None:
                        continue
                    png = _fig_to_png_bytes(fig)
                    if png:
                        ftitle = fig_titles[i] if i < len(fig_titles) else f"Graphique {i+1}"
                        pdf.set_font("Helvetica", "B", 9)
                        pdf.set_text_color(*MSF_NAVY)
                        pdf.cell(0, 6, _clean(f"  Figure {i+1} - {ftitle}"),
                                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                        pdf.set_text_color(*DARK_GRAY)
                        pdf.image(io.BytesIO(png), x=12, w=185)
                        pdf.ln(4)

            table_titles = [
                "Metriques par fold temporel",
                "Performance par aire de sante (top 20)",
            ]
            for tidx, (cols, rows) in enumerate(tables_data):
                ttitle = table_titles[tidx] if tidx < len(table_titles) else f"Tableau {tidx+1}"
                pdf.subsection(ttitle)
                pdf.table(cols, rows)

        # ── SECTION 5 : RECOMMANDATIONS ───────────────────────
        pdf.add_page()
        pdf.section_title("5. Recommandations terrain", GREEN)
        pdf.set_font("Helvetica", "", 9)
        reco_list = [
            "Prioriser les aires de sante affichant un indice de risque eleve pour les activites de riposte.",
            "Renforcer la surveillance active durant les semaines epidemiologiques de pic historique.",
            "Croiser les predictions avec les stocks de medicaments et les capacites d'hospitalisation.",
            "Mettre a jour les donnees d'entrainement du modele chaque trimestre pour maintenir la precision.",
            "Valider les predictions avec les equipes medicales locales avant toute decision logistique.",
            "Utiliser la carte de predictions pour planifier les campagnes de vaccination preventives.",
            "Documenter tout ecart important entre observations et predictions pour audit ulterieur.",
        ]
        for r in reco_list:
            pdf.multi_cell(0, 6, _clean(f"  > {r}"))
        pdf.ln(4)

        # ── SECTION 6 : LIMITES ───────────────────────────────
        pdf.section_title("6. Limites et avertissements", ORANGE)
        pdf.set_font("Helvetica", "I", 8)
        for d in [
            "Ce rapport est genere automatiquement par la plateforme MSF de surveillance epidemiologique.",
            "Les resultats refletent la qualite et l'exhaustivite des donnees importees.",
            "La validation est interne - une validation prospective sur une annee non entrainee est recommandee.",
            "L'horizon predictif fiable est de 4 a 6 semaines ; au-dela, l'incertitude augmente significativement.",
            "Ce document ne remplace pas l'expertise clinique et epidemiologique des equipes terrain.",
            f"Rapport genere le {datetime.now().strftime('%d/%m/%Y')} - A usage interne uniquement.",
        ]:
            pdf.multi_cell(0, 5, _clean(f"  {d}"))

        return bytes(pdf.output())

    except ImportError:
        return None
    except Exception as e:
        st.error(f"Erreur PDF : {e}")
        return None


# ─────────────────────────────────────────────────────────────
# FORMAT 2 — WORD (.docx)
# ─────────────────────────────────────────────────────────────

def generate_word(title, subtitle, maladie="Paludisme",
                  dashboard_data=None, carto_data=None,
                  model_data=None, validation_data=None):
    """
    Génère un rapport Word multi-sections avec python-docx.
    Retourne les bytes du .docx ou None si erreur.
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement

        MSF_RED  = RGBColor(228,  3,  46)
        MSF_NAVY = RGBColor( 26, 45,  79)
        WHITE    = RGBColor(255, 255, 255)
        GRAY     = RGBColor(130, 130, 130)

        def set_cell_bg(cell, hex_color):
            tc   = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd  = OxmlElement("w:shd")
            shd.set(qn("w:fill"), hex_color)
            shd.set(qn("w:val"),  "clear")
            tcPr.append(shd)

        def add_section(doc, num, title, color=MSF_NAVY):
            h = doc.add_heading(f"{num}. {title}", level=1)
            h.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in h.runs:
                run.font.color.rgb = color
                run.font.size      = Pt(13)
            return h

        def add_subsection(doc, title):
            h = doc.add_heading(title, level=2)
            for run in h.runs:
                run.font.color.rgb = MSF_NAVY
                run.font.size      = Pt(11)

        def add_kv_table(doc, kv_dict):
            tbl = doc.add_table(rows=1, cols=2)
            tbl.style = "Table Grid"
            hdr = tbl.rows[0].cells
            hdr[0].text = "Indicateur"
            hdr[1].text = "Valeur"
            for cell in hdr:
                set_cell_bg(cell, "1A2D4F")
                for run in cell.paragraphs[0].runs:
                    run.font.color.rgb = WHITE
                    run.font.bold      = True
                    run.font.size      = Pt(9)
            for i, (k, v) in enumerate(kv_dict.items()):
                r = tbl.add_row().cells
                r[0].text = _clean_html(str(k))
                r[1].text = _clean_html(str(v))
                if i % 2 == 0:
                    set_cell_bg(r[0], "F8F9FA")
                    set_cell_bg(r[1], "F8F9FA")
            doc.add_paragraph()

        def add_df_table(doc, df, max_rows=25):
            if df is None or len(df) == 0:
                return
            cols, rows = _df_to_rows(df, max_rows)
            tbl = doc.add_table(rows=1, cols=len(cols))
            tbl.style = "Table Grid"
            hdr = tbl.rows[0].cells
            for ci, col_name in enumerate(cols):
                hdr[ci].text = str(col_name)
                set_cell_bg(hdr[ci], "1A2D4F")
                for run in hdr[ci].paragraphs[0].runs:
                    run.font.color.rgb = WHITE
                    run.font.bold      = True
                    run.font.size      = Pt(8)
            for ri, row in enumerate(rows):
                row_cells = tbl.add_row().cells
                for ci, val in enumerate(row):
                    row_cells[ci].text = _clean_html(str(val))
                    if ri % 2 == 0:
                        set_cell_bg(row_cells[ci], "F8F9FA")
                    for run in row_cells[ci].paragraphs[0].runs:
                        run.font.size = Pt(8)
            doc.add_paragraph()

        def add_fig(doc, fig, caption, width=6.2):
            png = _fig_to_png_bytes(fig)
            if not png:
                return
            cap_p = doc.add_paragraph(caption)
            if cap_p.runs:
                cap_p.runs[0].font.bold      = True
                cap_p.runs[0].font.color.rgb = MSF_NAVY
                cap_p.runs[0].font.size      = Pt(9)
            doc.add_picture(io.BytesIO(png), width=Inches(width))
            doc.add_paragraph()

        # ── Document ──────────────────────────────────────────
        doc = Document()
        for section in doc.sections:
            section.top_margin    = Cm(2.5)
            section.bottom_margin = Cm(2.5)
            section.left_margin   = Cm(2.5)
            section.right_margin  = Cm(2.5)

        # Page de garde
        h = doc.add_heading(title, level=0)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in h.runs:
            run.font.color.rgb = MSF_NAVY
            run.font.size      = Pt(24)

        p = doc.add_paragraph(subtitle)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if p.runs:
            p.runs[0].font.italic      = True
            p.runs[0].font.color.rgb   = GRAY

        p2 = doc.add_paragraph(
            f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M UTC')}")
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if p2.runs:
            p2.runs[0].font.size      = Pt(9)
            p2.runs[0].font.color.rgb = GRAY

        p3 = doc.add_paragraph(f"Maladie : {maladie} | MSF — Médecins Sans Frontières")
        p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if p3.runs:
            p3.runs[0].font.bold      = True
            p3.runs[0].font.color.rgb = MSF_NAVY
        doc.add_page_break()

        sec_num = 1

        # ── Section 1 : Dashboard ────────────────────────────
        if dashboard_data:
            add_section(doc, sec_num, "Tableau de bord épidémiologique")
            sec_num += 1

            if dashboard_data.get("kpis"):
                add_subsection(doc, "Indicateurs clés de performance")
                add_kv_table(doc, dashboard_data["kpis"])

            if dashboard_data.get("evolution_fig"):
                add_subsection(doc, "Évolution hebdomadaire")
                add_fig(doc, dashboard_data["evolution_fig"],
                        "Évolution hebdomadaire des cas")

            if dashboard_data.get("top_aires") is not None:
                add_subsection(doc, "Top aires de santé les plus touchées")
                add_df_table(doc, dashboard_data["top_aires"])

        # ── Section 2 : Cartographie ─────────────────────────
        if carto_data:
            add_section(doc, sec_num, "Analyse cartographique")
            sec_num += 1

            if carto_data.get("carte_fig"):
                add_subsection(doc, "Carte de distribution")
                add_fig(doc, carto_data["carte_fig"], "Distribution géographique des cas",
                        width=6.0)

            if carto_data.get("top_risque") is not None:
                add_subsection(doc, "Aires à risque élevé")
                add_df_table(doc, carto_data["top_risque"])

        # ── Section 3 : Modélisation ─────────────────────────
        if model_data:
            add_section(doc, sec_num, "Paramètres de modélisation")
            sec_num += 1

            kv = {}
            if model_data.get("algorithme"):
                kv["Algorithme"] = model_data["algorithme"]
            if model_data.get("parametres"):
                kv.update({str(k): str(v)
                           for k, v in model_data["parametres"].items()})
            if kv:
                add_subsection(doc, "Configuration du modèle")
                add_kv_table(doc, kv)

            if model_data.get("variables"):
                add_subsection(doc, "Variables prédictives utilisées")
                for var in model_data["variables"]:
                    p = doc.add_paragraph(var, style="List Bullet")
                    if p.runs:
                        p.runs[0].font.size = Pt(10)
                doc.add_paragraph()

            if model_data.get("importance_fig"):
                add_subsection(doc, "Importance des variables")
                add_fig(doc, model_data["importance_fig"],
                        "Importance des variables prédictives")

        # ── Section 4 : Validation ───────────────────────────
        if validation_data:
            add_section(doc, sec_num, "Validation rétrospective")
            sec_num += 1

            metrics         = validation_data.get("metrics", {})
            interpretations = validation_data.get("interpretations", [])
            figures         = validation_data.get("figures", [])
            tables          = validation_data.get("tables", [])

            if metrics:
                add_subsection(doc, "Métriques de performance globales")
                add_kv_table(doc, metrics)

            if interpretations:
                add_subsection(doc, "Interprétation automatique")
                for line in interpretations:
                    p = doc.add_paragraph(_clean_html(line), style="List Bullet")
                    if p.runs:
                        p.runs[0].font.size = Pt(10)
                doc.add_paragraph()

            fig_titles = [
                "Observé vs Prédit — série temporelle",
                "Résidus vs valeurs prédites",
                "Distribution des résidus",
                "Importance des variables",
            ]
            for i, fig in enumerate(figures):
                if fig is None:
                    continue
                ftitle = fig_titles[i] if i < len(fig_titles) else f"Graphique {i+1}"
                add_subsection(doc, f"Figure {i+1} — {ftitle}")
                add_fig(doc, fig, ftitle)

            table_titles = [
                "Métriques par fold temporel",
                "Performance par aire de santé (top 20)",
            ]
            for tidx, (cols, rows) in enumerate(tables):
                ttitle = table_titles[tidx] if tidx < len(table_titles) else f"Tableau {tidx+1}"
                add_subsection(doc, ttitle)
                tbl = doc.add_table(rows=1, cols=len(cols))
                tbl.style = "Table Grid"
                hdr = tbl.rows[0].cells
                for ci, col_name in enumerate(cols):
                    hdr[ci].text = str(col_name)
                    set_cell_bg(hdr[ci], "1A2D4F")
                    for run in hdr[ci].paragraphs[0].runs:
                        run.font.color.rgb = WHITE
                        run.font.bold      = True
                        run.font.size      = Pt(8)
                for ri, row in enumerate(rows[:30]):
                    row_cells = tbl.add_row().cells
                    for ci, val in enumerate(row):
                        row_cells[ci].text = _clean_html(str(val))
                        if ri % 2 == 0:
                            set_cell_bg(row_cells[ci], "F8F9FA")
                        for run in row_cells[ci].paragraphs[0].runs:
                            run.font.size = Pt(8)
                doc.add_paragraph()

        # ── Section 5 : Recommandations ──────────────────────
        add_section(doc, sec_num, "Recommandations terrain",
                    color=RGBColor(39, 174, 96))
        sec_num += 1
        for r in [
            "Prioriser les aires de santé affichant un indice de risque élevé.",
            "Renforcer la surveillance active durant les semaines de pic historique.",
            "Croiser les prédictions avec les stocks de médicaments disponibles.",
            "Mettre à jour les données d'entraînement chaque trimestre.",
            "Valider les prédictions avec les équipes médicales locales.",
            "Utiliser la carte de prédictions pour planifier les campagnes préventives.",
            "Documenter tout écart important entre observations et prédictions.",
        ]:
            p = doc.add_paragraph(r, style="List Bullet")
            if p.runs:
                p.runs[0].font.size = Pt(10)
        doc.add_paragraph()

        # ── Section 6 : Limites ──────────────────────────────
        add_section(doc, sec_num, "Limites et avertissements",
                    color=RGBColor(230, 126, 34))
        for d in [
            "Ce rapport est généré automatiquement par la plateforme MSF.",
            "Les résultats reflètent la qualité et l'exhaustivité des données importées.",
            "Validation interne — une validation prospective est recommandée.",
            "Horizon prédictif fiable : 4 à 6 semaines maximum.",
            "Ce document ne remplace pas l'expertise clinique des équipes terrain.",
            f"Généré le {datetime.now().strftime('%d/%m/%Y')} — À usage interne uniquement.",
        ]:
            p = doc.add_paragraph(d, style="List Bullet")
            if p.runs:
                p.runs[0].font.size   = Pt(9)
                p.runs[0].font.italic = True

        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    except ImportError:
        return None
    except Exception as e:
        st.error(f"Erreur Word : {e}")
        return None


# ─────────────────────────────────────────────────────────────
# FORMAT 3 — HTML IMPRIMABLE
# ─────────────────────────────────────────────────────────────

def generate_html(title, subtitle, maladie="Paludisme",
                  dashboard_data=None, carto_data=None,
                  model_data=None, validation_data=None):
    """
    Génère un rapport HTML imprimable (CSS @media print).
    Retourne une string HTML UTF-8.
    """
    now_str = datetime.now().strftime("%d/%m/%Y à %H:%M UTC")

    def _rows_to_html(cols, rows, max_rows=30):
        ths = "".join(f"<th>{c}</th>" for c in cols)
        trs = ""
        for i, row in enumerate(rows[:max_rows]):
            cls = ' class="shade"' if i % 2 == 0 else ""
            tds = "".join(f"<td>{_clean_html(str(cell))}</td>" for cell in row)
            trs += f"<tr{cls}>{tds}</tr>"
        return f"<thead><tr>{ths}</tr></thead><tbody>{trs}</tbody>"

    def _fig_b64(fig, w=900, h=360):
        png = _fig_to_png_bytes(fig, w, h)
        if not png:
            return None
        return base64.b64encode(png).decode()

    def _df_to_html(df, max_rows=25):
        if df is None or len(df) == 0:
            return ""
        cols, rows = _df_to_rows(df, max_rows)
        return f"<table>{_rows_to_html(cols, rows)}</table>"

    # ── CSS ───────────────────────────────────────────────────
    css = """
    :root{--red:#E4032E;--navy:#1A2D4F;--bg:#F8F9FA;--div:#DEE2E6;
          --txt:#2D3748;--green:#27AE60;--orange:#E67E22;}
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
    body{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5pt;
         color:var(--txt);background:#fff;max-width:960px;margin:0 auto;padding:20px;}
    .report-header{background:var(--red);color:#fff;padding:18px 24px 14px;
                   border-radius:6px 6px 0 0;}
    .report-header .org{font-size:8.5pt;opacity:.85;letter-spacing:.06em;
                        text-transform:uppercase;}
    .report-header h1{font-size:20pt;font-weight:800;margin:8px 0 4px;line-height:1.2;}
    .report-header .sub{font-size:10pt;opacity:.9;font-style:italic;}
    .report-header .meta{font-size:8pt;opacity:.65;margin-top:8px;}
    .navy-bar{background:var(--navy);height:6px;margin-bottom:28px;}
    .toc{background:var(--bg);border:1px solid var(--div);border-radius:4px;
         padding:14px 20px;margin-bottom:28px;}
    .toc h3{color:var(--navy);font-size:10pt;margin-bottom:8px;}
    .toc ol{padding-left:20px;line-height:2;}
    section{margin-bottom:32px;}
    h2{font-size:11pt;font-weight:700;color:#fff;background:var(--navy);
       padding:7px 14px;border-radius:3px;margin-bottom:12px;}
    h2.green{background:var(--green);}
    h2.orange{background:var(--orange);}
    h3{font-size:10pt;font-weight:700;color:var(--navy);border-bottom:2px solid var(--div);
       padding-bottom:4px;margin:16px 0 8px;}
    table{width:100%;border-collapse:collapse;font-size:8.5pt;margin-top:6px;}
    thead tr{background:var(--navy);color:#fff;}
    thead th{padding:7px 9px;text-align:left;font-weight:600;}
    tbody tr.shade{background:var(--bg);}
    tbody td{padding:5px 9px;border-bottom:1px solid var(--div);}
    .kv-table td:first-child{font-weight:600;width:40%;}
    ul.items{padding-left:20px;line-height:1.9;}
    ul.items li{margin-bottom:3px;}
    .figure-block{margin:12px 0 18px;page-break-inside:avoid;}
    .figure-block img{width:100%;border:1px solid var(--div);border-radius:4px;
                      display:block;}
    .fig-caption{font-size:8.5pt;color:var(--navy);margin-bottom:5px;font-weight:700;}
    .kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
              gap:12px;margin-bottom:16px;}
    .kpi-card{background:var(--bg);border:1px solid var(--div);border-radius:6px;
              padding:10px 14px;}
    .kpi-card .kpi-label{font-size:8pt;color:#666;text-transform:uppercase;
                          letter-spacing:.04em;}
    .kpi-card .kpi-value{font-size:16pt;font-weight:800;color:var(--navy);
                          margin-top:4px;}
    .disclaimer{background:var(--bg);border-left:4px solid var(--orange);
                padding:12px 16px;font-size:9pt;font-style:italic;border-radius:0 4px 4px 0;}
    .disclaimer p{margin-bottom:5px;}
    .report-footer{background:var(--navy);color:#fff;text-align:center;
                   padding:10px;font-size:8pt;border-radius:0 0 6px 6px;margin-top:32px;}
    @media print{
      body{padding:0;max-width:100%;}
      .report-header,.report-footer{border-radius:0;}
      h2{page-break-after:avoid;}
      .figure-block{page-break-inside:avoid;}
      tr{page-break-inside:avoid;}
      .kpi-grid{grid-template-columns:repeat(4,1fr);}
    }
    """

    # ── Table des matières ────────────────────────────────────
    toc_items = []
    sec_num = 1
    for sec_label, active in [
        ("Tableau de bord épidémiologique", dashboard_data),
        ("Analyse cartographique",          carto_data),
        ("Paramètres de modélisation",       model_data),
        ("Validation rétrospective",         validation_data),
        ("Recommandations terrain",          True),
        ("Limites et avertissements",        True),
    ]:
        if active:
            anchor = f"sec{sec_num}"
            toc_items.append(f'<li><a href="#{anchor}">{sec_label}</a></li>')
            sec_num += 1

    toc_html = (f'<div class="toc"><h3>Contenu du rapport</h3>'
                f'<ol>{"".join(toc_items)}</ol></div>')

    # ── Sections ─────────────────────────────────────────────
    body = ""
    sec_num = 1

    # Dashboard
    if dashboard_data:
        inner = ""
        if dashboard_data.get("kpis"):
            kpi_cards = "".join(
                f'<div class="kpi-card"><div class="kpi-label">{_clean_html(str(k))}</div>'
                f'<div class="kpi-value">{_clean_html(str(v))}</div></div>'
                for k, v in dashboard_data["kpis"].items()
            )
            inner += f'<h3>Indicateurs clés</h3><div class="kpi-grid">{kpi_cards}</div>'
        if dashboard_data.get("evolution_fig"):
            b64 = _fig_b64(dashboard_data["evolution_fig"], 900, 320)
            if b64:
                inner += (f'<h3>Évolution hebdomadaire</h3>'
                          f'<div class="figure-block">'
                          f'<img src="data:image/png;base64,{b64}" '
                          f'alt="Evolution hebdomadaire" /></div>')
        if dashboard_data.get("top_aires") is not None:
            df_html = _df_to_html(dashboard_data["top_aires"])
            if df_html:
                inner += f'<h3>Top aires les plus touchées</h3>{df_html}'
        body += (f'<section id="sec{sec_num}">'
                 f'<h2>1. Tableau de bord épidémiologique</h2>{inner}</section>')
        sec_num += 1

    # Cartographie
    if carto_data:
        inner = ""
        if carto_data.get("carte_fig"):
            b64 = _fig_b64(carto_data["carte_fig"], 900, 480)
            if b64:
                inner += (f'<div class="figure-block">'
                          f'<p class="fig-caption">Distribution géographique des cas</p>'
                          f'<img src="data:image/png;base64,{b64}" '
                          f'alt="Carte distribution" /></div>')
        if carto_data.get("top_risque") is not None:
            df_html = _df_to_html(carto_data["top_risque"])
            if df_html:
                inner += f'<h3>Aires à risque élevé</h3>{df_html}'
        body += (f'<section id="sec{sec_num}">'
                 f'<h2>2. Analyse cartographique</h2>{inner}</section>')
        sec_num += 1

    # Modélisation
    if model_data:
        inner = ""
        kv = {}
        if model_data.get("algorithme"):
            kv["Algorithme"] = model_data["algorithme"]
        if model_data.get("parametres"):
            kv.update(model_data["parametres"])
        if kv:
            rows_html = "".join(
                f'<tr{"  class=\"shade\"" if i%2==0 else ""}>'
                f'<td>{_clean_html(str(k))}</td>'
                f'<td>{_clean_html(str(v))}</td></tr>'
                for i, (k, v) in enumerate(kv.items())
            )
            inner += (f'<h3>Configuration du modèle</h3>'
                      f'<table class="kv-table">'
                      f'<thead><tr><th>Paramètre</th><th>Valeur</th></tr></thead>'
                      f'<tbody>{rows_html}</tbody></table>')
        if model_data.get("variables"):
            vars_li = "".join(f"<li>{v}</li>" for v in model_data["variables"])
            inner += f'<h3>Variables prédictives</h3><ul class="items">{vars_li}</ul>'
        if model_data.get("importance_fig"):
            b64 = _fig_b64(model_data["importance_fig"])
            if b64:
                inner += (f'<h3>Importance des variables</h3>'
                          f'<div class="figure-block">'
                          f'<img src="data:image/png;base64,{b64}" '
                          f'alt="Importance variables" /></div>')
        body += (f'<section id="sec{sec_num}">'
                 f'<h2>3. Paramètres de modélisation</h2>{inner}</section>')
        sec_num += 1

    # Validation
    if validation_data:
        inner = ""
        metrics         = validation_data.get("metrics", {})
        interpretations = validation_data.get("interpretations", [])
        figures         = validation_data.get("figures", [])
        tables          = validation_data.get("tables", [])

        if metrics:
            rows_html = "".join(
                f'<tr{"  class=\"shade\"" if i%2==0 else ""}>'
                f'<td>{_clean_html(str(k))}</td>'
                f'<td>{_clean_html(str(v))}</td></tr>'
                for i, (k, v) in enumerate(metrics.items())
            )
            inner += (f'<h3>Métriques de performance</h3>'
                      f'<table class="kv-table">'
                      f'<thead><tr><th>Métrique</th><th>Valeur</th></tr></thead>'
                      f'<tbody>{rows_html}</tbody></table>')

        if interpretations:
            items = "".join(f"<li>{_clean_html(l)}</li>" for l in interpretations)
            inner += f'<h3>Interprétation automatique</h3><ul class="items">{items}</ul>'

        fig_titles = [
            "Observé vs Prédit — série temporelle",
            "Résidus vs valeurs prédites",
            "Distribution des résidus",
            "Importance des variables",
        ]
        for i, fig in enumerate(figures):
            if fig is None:
                continue
            b64 = _fig_b64(fig)
            if b64:
                ftitle = fig_titles[i] if i < len(fig_titles) else f"Graphique {i+1}"
                inner += (f'<div class="figure-block">'
                          f'<p class="fig-caption">Figure {i+1} — {ftitle}</p>'
                          f'<img src="data:image/png;base64,{b64}" alt="{ftitle}" /></div>')

        table_titles = [
            "Métriques par fold temporel",
            "Performance par aire de santé (top 20)",
        ]
        for tidx, (cols, rows) in enumerate(tables):
            ttitle = table_titles[tidx] if tidx < len(table_titles) else f"Tableau {tidx+1}"
            inner += (f'<h3>{ttitle}</h3>'
                      f'<table>{_rows_to_html(cols, rows)}</table>')

        body += (f'<section id="sec{sec_num}">'
                 f'<h2>4. Validation rétrospective</h2>{inner}</section>')
        sec_num += 1

    # Recommandations
    reco_items = "".join(f"<li>{r}</li>" for r in [
        "Prioriser les aires de santé affichant un indice de risque élevé.",
        "Renforcer la surveillance active durant les semaines de pic historique.",
        "Croiser les prédictions avec les stocks de médicaments disponibles.",
        "Mettre à jour les données d'entraînement chaque trimestre.",
        "Valider les prédictions avec les équipes médicales locales.",
        "Utiliser la carte de prédictions pour planifier les campagnes préventives.",
        "Documenter tout écart important entre observations et prédictions.",
    ])
    body += (f'<section id="sec{sec_num}">'
             f'<h2 class="green">5. Recommandations terrain</h2>'
             f'<ul class="items">{reco_items}</ul></section>')
    sec_num += 1

    # Limites
    disclaimers = "".join(f"<p>{d}</p>" for d in [
        "Ce rapport est généré automatiquement par la plateforme MSF de surveillance épidémiologique.",
        "Les résultats reflètent la qualité et l'exhaustivité des données importées.",
        "Validation interne — une validation prospective sur une nouvelle année est recommandée.",
        "Horizon prédictif fiable : 4 à 6 semaines maximum. L'incertitude augmente au-delà.",
        "Ce document ne remplace pas l'expertise clinique et épidémiologique des équipes terrain.",
        f"Généré le {datetime.now().strftime('%d/%m/%Y')} — À usage interne uniquement.",
    ])
    body += (f'<section id="sec{sec_num}">'
             f'<h2 class="orange">6. Limites et avertissements</h2>'
             f'<div class="disclaimer">{disclaimers}</div></section>')

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
<header class="report-header">
  <div class="org">MSF — Médecins Sans Frontières | Surveillance Épidémiologique</div>
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
  <div class="meta">Généré le {now_str} — Document confidentiel à usage interne | Maladie : {maladie}</div>
</header>
<div class="navy-bar"></div>
{toc_html}
{body}
<footer class="report-footer">
  {title} — {datetime.now().strftime('%d/%m/%Y')} — MSF Surveillance Épidémiologique — CONFIDENTIEL
</footer>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# WIDGET STREAMLIT UNIFIÉ — appelé depuis n'importe quel onglet
# ─────────────────────────────────────────────────────────────

def rapport_streamlit_widget(
    title,
    subtitle,
    maladie          = "Paludisme",
    # ── sections facultatives ──────────────────────────────
    dashboard_data   = None,   # dict : kpis, evolution_fig, top_aires
    carto_data       = None,   # dict : top_risque (DataFrame), carte_fig
    model_data       = None,   # dict : algorithme, parametres, variables, importance_fig
    validation_data  = None,   # dict : metrics, interpretations, figures, tables
    # ── compat. ancienne API (validation seule) ────────────
    metrics          = None,   # dict str→str  (ancien API)
    figures          = None,   # list Plotly   (ancien API)
    tables_data      = None,   # list tuples   (ancien API)
    interpretations  = None,   # list str      (ancien API)
    # ── options ────────────────────────────────────────────
    key_prefix       = "rapport",
):
    """
    Widget Streamlit : 3 boutons de téléchargement (PDF, Word, HTML).

    Supporte deux modes d'appel :
      1. Nouveau (multi-sections) — passer dashboard_data / carto_data / model_data / validation_data
      2. Ancien (validation seule) — passer metrics / figures / tables_data / interpretations
         (rétro-compatible avec le code existant dans validation_tab.py)
    """
    # Rétro-compatibilité : si appelé avec l'ancien API, emballer dans validation_data
    if validation_data is None and any(
        x is not None for x in [metrics, figures, tables_data, interpretations]
    ):
        validation_data = {
            "metrics":         metrics         or {},
            "interpretations": interpretations or [],
            "figures":         figures         or [],
            "tables":          tables_data     or [],
        }

    st.markdown("---")
    st.markdown("### 📄 Générer un rapport")
    st.info(
        "Choisissez le format :\n"
        "- **PDF** : archivage et partage officiel *(nécessite `fpdf2`)*\n"
        "- **Word** : éditable par les équipes avant envoi au PNLP *(nécessite `python-docx`)*\n"
        "- **HTML** : ouvrez dans le navigateur → `Ctrl+P` → Enregistrer en PDF *(aucune dépendance extra)*"
    )

    fname_base = (f"rapport_{maladie.lower().replace(' ','_')}_"
                  f"{datetime.now().strftime('%Y%m%d_%H%M')}")
    col1, col2, col3 = st.columns(3)

    kwargs = dict(
        title          = title,
        subtitle       = subtitle,
        maladie        = maladie,
        dashboard_data = dashboard_data,
        carto_data     = carto_data,
        model_data     = model_data,
        validation_data= validation_data,
    )

    with col1:
        if st.button("🔴 Générer PDF", key=f"{key_prefix}_pdf_btn",
                     use_container_width=True):
            with st.spinner("Génération du PDF…"):
                pdf_bytes = generate_pdf(**kwargs)
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Télécharger le PDF",
                    data=pdf_bytes,
                    file_name=f"{fname_base}.pdf",
                    mime="application/pdf",
                    key=f"{key_prefix}_pdf_dl",
                    use_container_width=True,
                )
                st.success("PDF prêt !")
            else:
                st.error("fpdf2 non installé — ajoutez `fpdf2` à requirements.txt")

    with col2:
        if st.button("🔵 Générer Word", key=f"{key_prefix}_word_btn",
                     use_container_width=True):
            with st.spinner("Génération du Word…"):
                docx_bytes = generate_word(**kwargs)
            if docx_bytes:
                st.download_button(
                    label="⬇️ Télécharger le Word",
                    data=docx_bytes,
                    file_name=f"{fname_base}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"{key_prefix}_word_dl",
                    use_container_width=True,
                )
                st.success("Word prêt !")
            else:
                st.error("python-docx non installé — ajoutez `python-docx` à requirements.txt")

    with col3:
        if st.button("🟢 Générer HTML", key=f"{key_prefix}_html_btn",
                     use_container_width=True):
            with st.spinner("Génération du HTML…"):
                html_str = generate_html(**kwargs)
            st.download_button(
                label="⬇️ Télécharger le HTML",
                data=html_str.encode("utf-8"),
                file_name=f"{fname_base}.html",
                mime="text/html",
                key=f"{key_prefix}_html_dl",
                use_container_width=True,
            )
            st.success("HTML prêt ! Ouvrez-le → Ctrl+P → Enregistrer en PDF")

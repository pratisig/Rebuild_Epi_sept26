# -*- coding: utf-8 -*-
"""
Test de fumée des applications Streamlit : exécute réellement chaque script
dans le runtime Streamlit (framework `AppTest`) et remonte toute exception.

Complément indispensable au harnais `legacy_harness.py` : celui-ci exécute un
bloc d'onglet isolé, il ne prouve pas que l'application complète démarre. Les
deux applications exécutent des appels d'interface au niveau du module, donc une
erreur d'import, de syntaxe ou de barre latérale n'apparaît qu'ici.

Usage :
    python tools/smoke_test_apps.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPS = ["main_app.py", "app_manuel.py", "app_paludisme.py", "app_rougeole.py"]

# Pages du portail : chacune doit être atteignable après authentification.
# Ce parcours est celui qui exerce réellement `load_app()` — sans lui, le
# routage de main_app.py ne s'exécute jamais et les sous-applications ne sont
# pas testées dans leur contexte d'appel.
PAGES = ["Accueil", "Manuel", "Paludisme", "Rougeole"]

# Nombre minimal d'éléments rendus par page, relevé sur le code fonctionnel.
# Sert de garde-fou : si `load_app()` cesse d'exécuter la sous-application, la
# page se rend vide au lieu de lever une exception — donc sans ce seuil le test
# passerait quand même.
RENDU_MINIMAL = {
    "Accueil":   {"markdown": 15},
    "Manuel":    {"markdown": 102, "expander": 10},
    "Paludisme": {"markdown": 21,  "selectbox": 2},
    "Rougeole":  {"markdown": 5,   "selectbox": 4},
}

# Messages d'erreur attendus en l'absence de données ou de compte Earth Engine :
# ce sont des invites à l'utilisateur, pas des dysfonctionnements.
ERREURS_ATTENDUES = (
    "Veuillez uploader",
    "GEE échec total",
    "authorize access to your Earth Engine",
    "WorldPop",
)


def main() -> int:
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("streamlit non installé — test de fumée ignoré")
        return 0

    echecs = []
    for app in APPS:
        chemin = os.path.join(REPO, app)
        if not os.path.exists(chemin):
            print(f"--- {app} : absent, ignoré")
            continue
        print(f"--- {app}")
        try:
            at = AppTest.from_file(chemin, default_timeout=300).run()
        except Exception as e:  # noqa: BLE001
            print(f"    ❌ échec au lancement : {type(e).__name__}: {e}")
            echecs.append(app)
            continue

        if at.exception:
            print(f"    ❌ {len(at.exception)} exception(s)")
            for e in at.exception[:3]:
                print(f"       {str(e.value)[:300]}")
            echecs.append(app)
            continue

        inattendues = [e.value for e in at.error
                       if not any(m in e.value for m in ERREURS_ATTENDUES)]
        print(f"    ✅ aucune exception | {len(at.error)} message(s) d'erreur "
              f"dont {len(inattendues)} inattendu(s)")
        for m in inattendues[:3]:
            print(f"       ⚠️ {m[:160]}")
        if inattendues:
            echecs.append(app)

    print()
    if echecs:
        print(f"❌ {len(echecs)} application(s) en échec : {echecs}")
        return 1
    print("✅ Toutes les applications démarrent sans exception")
    return 0


def main_routage() -> int:
    """
    Exécute le portail après authentification et parcourt ses quatre pages.

    L'authentification est franchie par l'état de session, comme pour un
    utilisateur déjà connecté. C'est le seul moyen d'atteindre le routage de
    `main_app.py` et donc le chargement réel des sous-applications.
    """
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("streamlit non installé — test de routage ignoré")
        return 0

    print("--- routage du portail (après authentification)")
    echecs = []
    for page in PAGES:
        at = AppTest.from_file(os.path.join(REPO, "main_app.py"),
                               default_timeout=300)
        at.session_state["authentication_status"] = True
        at.session_state["name"] = "Administrateur"
        at.session_state["page_choice"] = page
        try:
            at.run()
        except Exception as e:  # noqa: BLE001
            print(f"    ❌ {page} : échec au lancement : {type(e).__name__}: {e}")
            echecs.append(page)
            continue

        if at.exception:
            print(f"    ❌ {page} : {len(at.exception)} exception(s)")
            for e in at.exception[:2]:
                print(f"       {str(e.value)[:250]}")
            echecs.append(page)
            continue

        rendu = {
            "markdown": len(at.markdown),
            "expander": len(at.expander),
            "selectbox": len(at.selectbox),
        }
        attendu = RENDU_MINIMAL.get(page, {})
        insuffisant = {k: (rendu.get(k, 0), v) for k, v in attendu.items()
                       if rendu.get(k, 0) < v}
        if insuffisant:
            detail = ", ".join(f"{k}={a} < {b}" for k, (a, b) in insuffisant.items())
            print(f"    ❌ {page} : rendu insuffisant ({detail}) — la "
                  f"sous-application ne s'est probablement pas exécutée")
            echecs.append(page)
            continue

        print(f"    ✅ {page:10s} md={rendu['markdown']:3d} "
              f"exp={rendu['expander']:2d} sel={rendu['selectbox']}")

    if echecs:
        print(f"❌ {len(echecs)} page(s) en échec : {echecs}")
        return 1
    print("✅ Les quatre pages du portail se chargent et se rendent")
    return 0


if __name__ == "__main__":
    code = main()
    print()
    code = main_routage() or code
    sys.exit(code)

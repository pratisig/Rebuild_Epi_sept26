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


if __name__ == "__main__":
    sys.exit(main())

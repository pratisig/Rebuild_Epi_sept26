"""
Stub Streamlit minimal permettant d'exécuter des blocs d'application
(app_paludisme.py / app_rougeole.py) en mode headless, SANS réécrire leur logique.

Objectif : exécuter le VRAI code de l'application (extrait par AST) pour
diagnostiquer / valider le pipeline de prédiction réellement livré.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional


class StopException(Exception):
    """Équivalent de st.stop()."""


class _WidgetElement:
    """Objet permissif retourné par st.empty() / st.progress() / st.status()."""

    def __init__(self, sink: Optional[List[Any]] = None):
        self._sink = sink if sink is not None else []

    def __getattr__(self, name: str):
        def _noop(*args, **kwargs):
            self._sink.append((name, args, kwargs))
            return None
        return _noop

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Context:
    """Contexte de colonne / conteneur / spinner."""

    def __init__(self, stub: "StreamlitStub"):
        self._stub = stub

    def __getattr__(self, name: str):
        return getattr(self._stub, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _SessionState(dict):
    """st.session_state : dict + accès attribut."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


class StreamlitStub:
    """
    Remplace le module `streamlit`.

    selectbox_values : {substring_du_label: valeur_a_retourner}
    buttons          : liste de substrings pour lesquels st.button() renvoie True
                       (par défaut TOUS les boutons renvoient True)
    """

    def __init__(self,
                 selectbox_values: Optional[Dict[str, Any]] = None,
                 slider_values: Optional[Dict[str, Any]] = None,
                 radio_values: Optional[Dict[str, Any]] = None,
                 buttons_true: bool = True,
                 collect: bool = True):
        self.selectbox_values = selectbox_values or {}
        self.slider_values = slider_values or {}
        self.radio_values = radio_values or {}
        self.buttons_true = buttons_true
        self.collect = collect
        self.session_state = _SessionState()
        self.log: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.dataframes: List[Any] = []
        self.charts: List[Any] = []
        # alias utilisés par le code applicatif
        self.sidebar = _Context(self)
        self.cache_data = self._cache_decorator
        self.cache_resource = self._cache_decorator
        self.set_page_config = lambda *a, **k: None

    # ── décorateurs de cache : transparents ─────────────────────
    def _cache_decorator(self, *dargs, **dkwargs):
        def _wrap(fn: Callable):
            return fn
        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return _wrap

    # ── affichage ───────────────────────────────────────────────
    def _record(self, kind: str, *args, **kwargs):
        if self.collect:
            self.log.append(f"{kind}: {args[0] if args else ''}")

    def markdown(self, *a, **k):        self._record("markdown", *a, **k)
    def write(self, *a, **k):           self._record("write", *a, **k)
    def subheader(self, *a, **k):       self._record("subheader", *a, **k)
    def header(self, *a, **k):          self._record("header", *a, **k)
    def title(self, *a, **k):           self._record("title", *a, **k)
    def caption(self, *a, **k):         self._record("caption", *a, **k)
    def latex(self, *a, **k):           self._record("latex", *a, **k)
    def text(self, *a, **k):            self._record("text", *a, **k)
    def code(self, *a, **k):            self._record("code", *a, **k)
    def json(self, *a, **k):            self._record("json", *a, **k)
    def toast(self, *a, **k):           self._record("toast", *a, **k)
    def help(self, *a, **k):            self._record("help", *a, **k)
    def html(self, *a, **k):            self._record("html", *a, **k)

    def success(self, *a, **k):         self._record("success", *a, **k)
    def info(self, *a, **k):            self._record("info", *a, **k)

    def error(self, *a, **k):
        self.errors.append(str(a[0]) if a else "")
        self._record("error", *a, **k)

    def warning(self, *a, **k):
        self.warnings.append(str(a[0]) if a else "")
        self._record("warning", *a, **k)

    def exception(self, *a, **k):
        self.errors.append(str(a[0]) if a else "")

    def metric(self, label, value=None, delta=None, *a, **k):
        self.metrics[str(label)] = value
        self._record("metric", label, value)

    def dataframe(self, data=None, *a, **k):
        self.dataframes.append(data)
        self._record("dataframe")

    def data_editor(self, data=None, *a, **k):
        self.dataframes.append(data)
        return data

    def table(self, data=None, *a, **k):
        self.dataframes.append(data)

    def plotly_chart(self, fig=None, *a, **k):
        self.charts.append(fig)

    def pyplot(self, *a, **k):          self._record("pyplot")
    def map(self, *a, **k):             self._record("map")
    def image(self, *a, **k):           self._record("image")
    def download_button(self, *a, **k): return False
    def file_uploader(self, *a, **k):   return None
    def text_input(self, *a, **k):      return ""
    def number_input(self, label, value=0, *a, **k): return value
    def date_input(self, *a, **k):      return None
    def multiselect(self, label, options, default=None, *a, **k):
        return list(default) if default else list(options)
    def progress(self, value=None, *a, **k): return _WidgetElement()
    def empty(self, *a, **k):           return _WidgetElement()
    def status(self, *a, **k):          return _WidgetElement()
    def container(self, *a, **k):       return _Context(self)
    def expander(self, *a, **k):        return _Context(self)
    def form(self, *a, **k):            return _Context(self)
    def form_submit_button(self, *a, **k): return False
    def rerun(self, *a, **k):           return None
    def experimental_rerun(self, *a, **k): return None

    def stop(self, *a, **k):
        import traceback
        stack = "".join(traceback.format_stack()[-4:-1])
        raise StopException("st.stop()\n" + stack)

    @contextmanager
    def spinner(self, *a, **k):
        yield _WidgetElement()

    def columns(self, spec=1, *a, **k):
        if isinstance(spec, (list, tuple)):
            n = len(spec)
        else:
            n = int(spec)
        return [_Context(self) for _ in range(max(1, n))]

    def tabs(self, labels, *a, **k):
        return [_Context(self) for _ in labels]

    # ── widgets de saisie ───────────────────────────────────────
    def _pick(self, store: Dict[str, Any], label: str, options, default):
        for key, val in store.items():
            if key.lower() in str(label).lower():
                return val
        if default is not None:
            return default
        try:
            return list(options)[0]
        except Exception:
            return None

    def selectbox(self, label, options=(), index=0, *a, **k):
        forced = self._pick(self.selectbox_values, label, options, None)
        if forced is not None:
            return forced
        opts = list(options)
        if index and isinstance(index, int) and 0 <= index < len(opts):
            return opts[index]
        return opts[0] if opts else None

    def radio(self, label, options=(), index=0, *a, **k):
        forced = self._pick(self.radio_values, label, options, None)
        if forced is not None:
            return forced
        opts = list(options)
        if index and isinstance(index, int) and 0 <= index < len(opts):
            return opts[index]
        return opts[0] if opts else None

    def slider(self, label, min_value=None, max_value=None, value=None, *a, **k):
        forced = self._pick(self.slider_values, label, None, None)
        if forced is not None:
            return forced
        if value is not None:
            return value
        return min_value

    def checkbox(self, label, value=False, *a, **k):
        return bool(value)

    def toggle(self, label, value=False, *a, **k):
        return bool(value)

    def button(self, label="", *a, **k):
        bt = self.buttons_true
        if isinstance(bt, (list, tuple, set)):
            return any(str(pat).lower() in str(label).lower() for pat in bt)
        return bool(bt)

    def select_slider(self, label, options=(), value=None, *a, **k):
        return value if value is not None else (list(options)[0] if options else None)


def make_namespace(stub: StreamlitStub) -> Dict[str, Any]:
    """Espace de noms de base pour exécuter un bloc applicatif."""
    return {"st": stub}

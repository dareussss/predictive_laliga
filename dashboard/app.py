"""Dashboard La Liga.

    streamlit run dashboard/app.py

Fiecare grafic are si o vizualizare tabelara, pentru ca nicio valoare nu trebuie sa fie
accesibila doar prin culoare sau doar prin hover.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.data import active_palette
from dashboard.views import age, form, match, players, race
from models.dixon_coles import DEFAULT_XI
from models.form_forecast import DEFAULT_HORIZON

st.set_page_config(page_title="La Liga Analytics", page_icon="⚽", layout="wide")

PAGES = {
    "Predicție de meci": match.render,
    "Formă": form.render,
    "Jucători": players.render,
    "Golgheter": race.render,
    "Curba de vârstă": age.render,
}


def sidebar() -> str:
    """Navigatia si setarile globale; intoarce pagina aleasa."""
    st.sidebar.title("⚽ La Liga Analytics")
    st.sidebar.caption("Predicții de scor, formă și golgheter din 7.980 de meciuri.")

    page = st.sidebar.radio("Secțiune", list(PAGES), label_visibility="collapsed")

    st.sidebar.divider()
    st.session_state.xi = st.sidebar.select_slider(
        "Memoria modelului (xi)",
        options=[0.0005, 0.001, 0.0018, 0.003],
        value=DEFAULT_XI,
        help="Cât de repede uită meciurile vechi. 0.001 e valoarea aleasă prin backtesting.",
    )
    st.session_state.horizon = st.sidebar.slider("Etape pentru formă", 3, 10, DEFAULT_HORIZON)
    st.session_state.sims = st.sidebar.select_slider(
        "Simulări pentru golgheter", options=[2000, 5000, 10000, 20000], value=5000
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "**Surse.** Rezultate istorice: football-data.co.uk. Calendar și marcatori: "
        "football-data.org. Statistici de jucător: FBref, prin arhiva worldfootballR_data "
        "a lui Jason Zivkovic."
    )
    return page


def main() -> None:
    page = sidebar()
    PAGES[page](active_palette())


if __name__ == "__main__":
    main()

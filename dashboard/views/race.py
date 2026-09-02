"""Pagina cursei pentru titlul de golgheter."""

from __future__ import annotations

import streamlit as st

from dashboard.data import cached_race, table_view
from dashboard.theme import horizontal_bars


def render(palette) -> None:
    st.subheader("Cursa pentru golgheter 2026/27")
    st.caption(
        "Simulare Monte Carlo a sezonului. Incertitudinea vine din trei surse: câte goluri "
        "dă echipa, ce parte îi revine jucătorului, și norocul din fața porții."
    )

    race = cached_race(st.session_state.xi, st.session_state.sims)
    top = race.head(15)

    hover = [
        f"{row.player}<br>{row.team}"
        f"<br>bază: {row.base_goals} goluri în {row.base_matches} meciuri"
        f"<br>estimat: {row.expected_goals:.1f} goluri"
        f"<br>titlu {row.title_probability:.1%} · 20+ {row.p20_plus:.0%} · 25+ {row.p25_plus:.0%}"
        for row in top.itertuples(index=False)
    ]
    figure = horizontal_bars(
        top["player"].tolist(),
        top["title_probability"].tolist(),
        palette,
        value_labels=[f"{value:.1%}" for value in top["title_probability"]],
        hover=hover,
        height=470,
        highlight=top.iloc[0]["player"],
    )
    figure.update_xaxes(title=dict(text="probabilitate de a termina golgheter"), tickformat=".0%")
    st.plotly_chart(figure, width="stretch")

    st.markdown("##### Goluri așteptate")
    hover = [
        f"{row.player}<br>{row.expected_goals:.1f} goluri estimate"
        for row in top.itertuples(index=False)
    ]
    goals_figure = horizontal_bars(
        top["player"].tolist(),
        top["expected_goals"].tolist(),
        palette,
        value_labels=[f"{value:.1f}" for value in top["expected_goals"]],
        hover=hover,
        height=470,
    )
    goals_figure.update_xaxes(title=dict(text="goluri așteptate pe sezon"))
    st.plotly_chart(goals_figure, width="stretch")

    display = race[
        ["player", "team", "base_goals", "base_matches", "rate", "expected_goals",
         "title_probability", "p20_plus", "p25_plus"]
    ].copy()
    display.columns = [
        "jucător", "echipă", "goluri 25-26", "meciuri 25-26", "rată",
        "goluri estimate", "P(titlu) %", "P(20+) %", "P(25+) %",
    ]
    for column in ("rată", "goluri estimate"):
        display[column] = display[column].round(2)
    for column in ("P(titlu) %", "P(20+) %", "P(25+) %"):
        display[column] = (display[column] * 100).round(1)
    table_view(display)

    st.warning(
        "**Limitări asumate.** Loturile sunt cele de la finalul sezonului 2025-26 — "
        "transferurile verii 2026 nu sunt cunoscute de nicio sursă gratuită încă. Cele trei "
        "echipe promovate n-au jucători în datele de La Liga, deci golurile lor rămân "
        "nealocate: un golgheter surpriză de la Málaga e imposibil în acest model."
    )

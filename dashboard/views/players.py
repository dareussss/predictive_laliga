"""Pagina indexului de performanta a jucatorilor."""

from __future__ import annotations

import streamlit as st

from dashboard.data import cached_index, table_view
from dashboard.theme import horizontal_bars, scatter


def render(palette) -> None:
    st.subheader("Index de performanță ofensivă")
    st.caption(
        "npxG + xAG per 90 de minute, contractat spre media poziției. Cât de tare se "
        "contractă e valoarea care prezice cel mai bine sezonul următor, măsurată pe "
        "tranzițiile reale din istoric."
    )

    left, middle, right = st.columns([1, 1, 2])
    season = left.selectbox("Sezon", ["2024-25", "2023-24", "2022-23", "2021-22"], index=0)
    minimum = middle.slider("Minute minime", 180, 2000, 450, step=90)
    choice = right.radio(
        "Poziție", ["toate", "FW", "MF", "DF", "GK"], horizontal=True, label_visibility="collapsed"
    )

    index = cached_index(season, minimum)
    subset = index if choice == "toate" else index[index["group"] == choice]

    if subset.empty:
        st.warning("Niciun jucător pentru filtrele alese.")
        return

    top = subset.head(15)
    hover = [
        f"{row.name}<br>{row.squad}<br>{int(row.minutes)} minute"
        f"<br>index {row.index_per90:.2f} (brut {row.rate:.2f})"
        f"<br>contracție {row.shrinkage:.0%}"
        for row in top.itertuples(index=False)
    ]
    figure = horizontal_bars(
        top["name"].tolist(),
        top["index_per90"].tolist(),
        palette,
        value_labels=[f"{value:.2f}" for value in top["index_per90"]],
        hover=hover,
        height=470,
    )
    figure.update_xaxes(title=dict(text="index (npxG + xAG per 90)"))
    st.plotly_chart(figure, width="stretch")

    st.markdown("##### Efectul contracției")
    st.caption(
        "Fiecare punct e un jucător. Cu cât joacă mai puțin, cu atât indexul lui e tras mai "
        "tare spre media poziției — de aceea norul se îngustează spre stânga."
    )
    hover = [
        f"{row.name}<br>{int(row.minutes)} minute<br>brut {row.rate:.2f} → index {row.index_per90:.2f}"
        for row in subset.itertuples(index=False)
    ]
    dots = scatter(subset["minutes"].tolist(), subset["index_per90"].tolist(), palette, hover=hover)
    dots.update_xaxes(title=dict(text="minute jucate"))
    dots.update_yaxes(title=dict(text="index"))
    st.plotly_chart(dots, width="stretch")

    display = subset[
        ["name", "squad", "group", "age", "minutes", "goals", "assists", "rate", "index_per90", "z_score", "shrinkage"]
    ].copy()
    display.columns = [
        "jucător", "echipă", "poziție", "vârstă", "minute", "goluri", "assist",
        "brut", "index", "z", "contracție %",
    ]
    for column in ("brut", "index", "z"):
        display[column] = display[column].round(3)
    display["contracție %"] = (display["contracție %"] * 100).round(0)
    table_view(display)

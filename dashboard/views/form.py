"""Pagina de forma asteptata pe urmatoarele etape."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data import cached_form, cached_model, cached_team_fixtures, table_view
from dashboard.theme import horizontal_bars, stacked_outcomes


def render(palette) -> None:
    horizon = st.session_state.horizon
    st.subheader(f"Formă așteptată — următoarele {horizon} etape")
    st.caption(
        "Distribuția punctelor e calculată exact prin convoluție, nu simulată. Echipele "
        "marcate cu * n-au istoric recent și primesc ratingul mediu al promovatelor."
    )

    form = cached_form(st.session_state.xi, horizon)
    labels = [f"{row.echipa}{' *' if row.estimat else ''}" for row in form.itertuples(index=False)]
    hover = [
        f"{row.echipa}<br>{row.puncte:.2f} din {3 * horizon} puncte"
        f"<br>{row.acasa} meciuri acasă<br>P(≥7p) {row.min7:.0%}"
        for row in form.itertuples(index=False)
    ]

    figure = horizontal_bars(
        labels,
        form["puncte"].tolist(),
        palette,
        value_labels=[f"{value:.1f}" for value in form["puncte"]],
        hover=hover,
        height=560,
    )
    figure.update_xaxes(title=dict(text=f"puncte așteptate din {3 * horizon}"))
    st.plotly_chart(figure, width="stretch")

    display = form.drop(columns=["estimat"]).rename(
        columns={
            "echipa": "echipă",
            "puncte": "puncte așteptate",
            "min7": "P(≥7 puncte) %",
            "maxim": "P(maxim) %",
            "acasa": "meciuri acasă",
        }
    )
    display["puncte așteptate"] = display["puncte așteptate"].round(2)
    display["P(≥7 puncte) %"] = (display["P(≥7 puncte) %"] * 100).round(1)
    display["P(maxim) %"] = (display["P(maxim) %"] * 100).round(2)
    table_view(display)

    st.divider()
    _team_detail(palette, form, horizon)


def _team_detail(palette, form: pd.DataFrame, horizon: int) -> None:
    """Meciurile concrete din care rezultă punctele așteptate ale unei echipe."""
    _, promoted = cached_model(st.session_state.xi)
    teams = form["echipa"].tolist()

    st.markdown("##### Ce meciuri urmează")
    st.caption(
        "Clasamentul de sus spune cât; aici se vede de ce. Două echipe cu același rating "
        "pot avea puncte așteptate diferite doar din calendar."
    )

    left, _ = st.columns([1, 3])
    team = left.selectbox("Echipă", teams, index=0, label_visibility="collapsed")

    summary = form[form["echipa"] == team].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Puncte așteptate", f"{summary.puncte:.2f}", f"din {3 * horizon}", delta_color="off")
    columns[1].metric("Șanse de minim 7 puncte", f"{summary.min7:.0%}")
    columns[2].metric("Șanse de punctaj maxim", f"{summary.maxim:.2%}")
    columns[3].metric("Meciuri acasă", f"{int(summary.acasa)} din {horizon}")

    fixtures = cached_team_fixtures(st.session_state.xi, horizon, team)
    if fixtures.empty:
        st.warning("Nu există meciuri programate pentru echipa aleasă.")
        return

    labels = [
        f"et. {row.etapa} · {'vs' if row.acasa else 'la'} {row.adversar}"
        f"{' *' if row.adversar in promoted else ''}"
        for row in fixtures.itertuples(index=False)
    ]
    hover = [
        f"{'Acasă cu' if row.acasa else 'În deplasare la'} {row.adversar}"
        f"<br>{row.data.strftime('%d %b %Y')} · etapa {row.etapa}"
        f"<br>V {row.victorie:.0%} · X {row.egal:.0%} · Î {row.infrangere:.0%}"
        f"<br>goluri {row.goluri_marcate:.2f} – {row.goluri_primite:.2f}"
        f"<br>{row.puncte:.2f} puncte așteptate"
        for row in fixtures.itertuples(index=False)
    ]

    figure = stacked_outcomes(
        labels,
        fixtures["victorie"].tolist(),
        fixtures["egal"].tolist(),
        fixtures["infrangere"].tolist(),
        palette,
        hover=hover,
        annotations=[f"{value:.2f} p" for value in fixtures["puncte"]],
        height=90 + 46 * len(fixtures),
    )
    st.plotly_chart(figure, width="stretch")

    detail = pd.DataFrame(
        {
            "etapa": fixtures["etapa"],
            "data": fixtures["data"].dt.strftime("%d.%m.%Y"),
            "teren": ["acasă" if value else "deplasare" for value in fixtures["acasa"]],
            "adversar": fixtures["adversar"],
            "V %": (fixtures["victorie"] * 100).round(1),
            "X %": (fixtures["egal"] * 100).round(1),
            "Î %": (fixtures["infrangere"] * 100).round(1),
            "goluri marcate": fixtures["goluri_marcate"].round(2),
            "goluri primite": fixtures["goluri_primite"].round(2),
            "puncte așteptate": fixtures["puncte"].round(2),
        }
    )
    table_view(detail, "Vezi meciurile ca tabel")

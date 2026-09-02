"""Pagina de predictie a unui meci."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import cached_fixtures, cached_model, table_view
from dashboard.theme import apply_layout, sequential_scale

MAX_DISPLAY_GOALS = 6


def render(palette) -> None:
    st.subheader("Predicție de meci")
    st.caption(
        "Modelul produce o distribuție peste toate scorurile posibile. Rezultatul 1X2, "
        "peste/sub 2.5 goluri și „ambele înscriu” se citesc din aceeași matrice."
    )

    model, promoted = cached_model(st.session_state.xi)
    fixtures = cached_fixtures()
    options = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    if not options:
        options = sorted(team for team in model.teams if model.known(team))

    left, right, _ = st.columns([1, 1, 2])
    home = left.selectbox(
        "Gazdă", options, index=options.index("Barcelona") if "Barcelona" in options else 0
    )
    away = right.selectbox(
        "Oaspete", options, index=options.index("Real Madrid") if "Real Madrid" in options else 1
    )

    if home == away:
        st.warning("Alege două echipe diferite.")
        return

    prediction = model.predict(home, away)
    estimated = {home, away} & promoted
    if estimated:
        st.info(
            f"Rating estimat din prior-ul promovatelor pentru {', '.join(sorted(estimated))} — "
            "echipe fără istoric recent în La Liga."
        )

    columns = st.columns(3)
    columns[0].metric(f"1 — {home}", f"{prediction.home_win:.1%}")
    columns[1].metric("X — egal", f"{prediction.draw:.1%}")
    columns[2].metric(f"2 — {away}", f"{prediction.away_win:.1%}")

    columns = st.columns(4)
    columns[0].metric(
        "Goluri așteptate",
        f"{prediction.expected_home_goals:.2f} – {prediction.expected_away_goals:.2f}",
    )
    columns[1].metric(
        "Scor cel mai probabil",
        f"{prediction.most_likely_score[0]}–{prediction.most_likely_score[1]}",
        f"{prediction.most_likely_score_prob:.1%}",
        delta_color="off",
    )
    columns[2].metric("Peste 2.5 goluri", f"{prediction.over_2_5:.1%}")
    columns[3].metric("Ambele înscriu", f"{prediction.both_teams_score:.1%}")

    matrix = model.score_matrix(home, away)[: MAX_DISPLAY_GOALS + 1, : MAX_DISPLAY_GOALS + 1]
    axis = list(range(MAX_DISPLAY_GOALS + 1))
    annotations = [[f"{value:.1%}" if value >= 0.01 else "" for value in row] for row in matrix]
    hover = [[f"{home} {h} – {a} {away}<br>{matrix[h][a]:.2%}" for a in axis] for h in axis]

    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=axis,
            y=axis,
            colorscale=sequential_scale(palette),
            xgap=2,
            ygap=2,
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertext=hover,
            hoverinfo="text",
            colorbar=dict(
                title=dict(text="probabilitate", font=dict(size=11)),
                tickformat=".0%",
                outlinewidth=0,
                thickness=12,
            ),
        )
    )
    apply_layout(figure, palette, height=430)
    figure.update_xaxes(title=dict(text=f"goluri {away}"), showgrid=False, dtick=1)
    figure.update_yaxes(title=dict(text=f"goluri {home}"), dtick=1)

    st.markdown("##### Distribuția scorurilor")
    st.plotly_chart(figure, width="stretch")

    table = pd.DataFrame(
        matrix,
        index=[str(value) for value in axis],
        columns=[str(value) for value in axis],
    ).round(4)
    table.index.name = f"goluri {home}"
    table_view(table.reset_index(), "Vezi matricea ca tabel")

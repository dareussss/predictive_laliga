"""Pagina curbei de varsta."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data import cached_age_curve, table_view
from dashboard.theme import line


def render(palette) -> None:
    st.subheader("Curba de vârstă")
    st.caption(
        "Estimată prin metoda diferențelor: fiecare jucător e comparat cu el însuși de la un "
        "sezon la altul. Media producției pe vârstă ar fi înșelătoare, pentru că la 35 de ani "
        "mai joacă doar cei foarte buni."
    )

    curve, peak_age, n_transitions, results = cached_age_curve("goals_assists")

    ages = [int(value) for value in curve.index]
    values = [float(value) for value in curve.values]
    hover = [f"{age} ani<br>{value:+.3f} față de vârf" for age, value in zip(ages, values)]

    figure = line(ages, values, palette, hover=hover, marker_at=peak_age)
    figure.update_xaxes(title=dict(text="vârstă"), dtick=2, showgrid=False)
    figure.update_yaxes(
        title=dict(text="producție față de vârf"), showgrid=True, gridcolor=palette.gridline
    )
    st.plotly_chart(figure, width="stretch")

    st.markdown(
        f"Vârf la **{peak_age} ani**, estimat pe **{n_transitions:,}** tranziții sezon-la-sezon."
    )

    st.markdown("##### Ajută la predicție?")
    display = results.rename(columns={"rmse": "RMSE", "n": "tranziții testate"})
    display["RMSE"] = display["RMSE"].round(4)
    st.dataframe(display, width="stretch", hide_index=True)

    st.error(
        "**Nu.** Curba e corectă descriptiv — vârful la 26 de ani se potrivește cu literatura — "
        "dar aplicarea ei înrăutățește predicțiile, în fiecare grupă de vârstă testată. "
        "Contracția spre medie absoarbe deja aproape tot efectul: un jucător în declin produce "
        "mai puțin, iar contracția îl trage în jos oricum. De aceea ajustarea e dezactivată "
        "implicit în model."
    )

    table_view(pd.DataFrame({"vârstă": ages, "față de vârf": np.round(values, 4)}))

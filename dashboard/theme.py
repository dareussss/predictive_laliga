"""Paleta si sabloanele de grafic ale dashboard-ului.

Paleta e validata pentru daltonism; motivatia alegerilor de culoare e in README.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go

SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


@dataclass(frozen=True)
class Palette:
    """Rolurile de culoare pentru un mod de afisare."""

    surface: str
    page: str
    text_primary: str
    text_secondary: str
    muted: str
    gridline: str
    baseline: str
    series_1: str
    series_2: str
    series_3: str
    diverging_warm: str = "#e34948"
    neutral: str = "#c3c2b7"
    good: str = "#0ca30c"
    critical: str = "#d03b3b"


LIGHT = Palette(
    surface="#fcfcfb",
    page="#f9f9f7",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    series_1="#2a78d6",
    series_2="#eb6834",
    series_3="#1baf7a",
)

DARK = Palette(
    surface="#1a1a19",
    page="#0d0d0d",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    series_1="#3987e5",
    series_2="#d95926",
    series_3="#199e70",
    diverging_warm="#e66767",
    neutral="#383835",
)


def palette_for(base: str | None) -> Palette:
    """Alege paleta dupa tema activa a aplicatiei."""
    return DARK if (base or "light").lower() == "dark" else LIGHT


def sequential_scale(palette: Palette) -> list[list]:
    """Rampa continua albastra, pornind de la suprafata pentru valori nule."""
    steps = [palette.surface, *SEQUENTIAL_BLUE]
    return [[position / (len(steps) - 1), color] for position, color in enumerate(steps)]


def apply_layout(figure: go.Figure, palette: Palette, height: int = 420) -> go.Figure:
    """Cromatica comuna: fundal transparent, grila subtire, text pe tokeni de text."""
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            size=13,
            color=palette.text_secondary,
        ),
        margin=dict(l=8, r=8, t=8, b=8),
        hoverlabel=dict(
            bgcolor=palette.surface,
            bordercolor=palette.baseline,
            font=dict(color=palette.text_primary, size=13),
        ),
        showlegend=False,
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor=palette.gridline,
        gridwidth=1,
        zeroline=False,
        linecolor=palette.baseline,
        tickfont=dict(color=palette.muted),
    )
    figure.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor=palette.baseline,
        tickfont=dict(color=palette.muted),
    )
    return figure


def horizontal_bars(
    labels: list[str],
    values: list[float],
    palette: Palette,
    value_labels: list[str] | None = None,
    hover: list[str] | None = None,
    height: int = 420,
    highlight: str | None = None,
) -> go.Figure:
    """Bare orizontale cu o singura serie, cu evidentierea optionala a unei categorii."""
    if highlight is None:
        colors = [palette.series_1] * len(labels)
    else:
        colors = [
            palette.series_1 if label == highlight else palette.gridline for label in labels
        ]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=value_labels,
            textposition="outside",
            textfont=dict(color=palette.text_secondary, size=12),
            hovertext=hover,
            hoverinfo="text" if hover else "x+y",
            width=0.62,
        )
    )
    apply_layout(figure, palette, height=height)
    figure.update_layout(
        bargap=0.32,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, max(values) * 1.18 if values else 1]),
    )
    return figure


def stacked_outcomes(
    labels: list[str],
    win: list[float],
    draw: list[float],
    loss: list[float],
    palette: Palette,
    hover: list[str] | None = None,
    annotations: list[str] | None = None,
    height: int = 320,
) -> go.Figure:
    """Cate o bara pe meci, impartita in victorie / egal / infrangere."""
    figure = go.Figure()
    series = (
        ("Victorie", win, palette.series_1),
        ("Egal", draw, palette.neutral),
        ("Înfrângere", loss, palette.diverging_warm),
    )
    for name, values, color in series:
        figure.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                name=name,
                marker=dict(color=color, line=dict(color=palette.surface, width=1)),
                hovertext=hover,
                hoverinfo="text" if hover else "x+name",
                width=0.6,
            )
        )

    apply_layout(figure, palette, height=height)
    figure.update_layout(
        barmode="stack",
        bargap=0.34,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(color=palette.text_secondary),
        ),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 1.16], tickformat=".0%"),
    )

    if annotations:
        for position, text in enumerate(annotations):
            figure.add_annotation(
                x=1.02,
                y=labels[position],
                text=text,
                showarrow=False,
                xanchor="left",
                font=dict(color=palette.text_secondary, size=12),
            )
    return figure


def line(
    x: list,
    y: list,
    palette: Palette,
    hover: list[str] | None = None,
    marker_at: int | None = None,
    height: int = 380,
) -> go.Figure:
    """Linie cu o singura serie, 2px, cu un punct evidentiat optional."""
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=palette.series_1, width=2),
            hovertext=hover,
            hoverinfo="text" if hover else "x+y",
        )
    )
    if marker_at is not None and marker_at in x:
        position = list(x).index(marker_at)
        figure.add_trace(
            go.Scatter(
                x=[x[position]],
                y=[y[position]],
                mode="markers",
                marker=dict(
                    color=palette.series_1,
                    size=11,
                    line=dict(color=palette.surface, width=2),
                ),
                hoverinfo="skip",
            )
        )
    apply_layout(figure, palette, height=height)
    return figure


def scatter(
    x: list,
    y: list,
    palette: Palette,
    hover: list[str] | None = None,
    height: int = 440,
) -> go.Figure:
    """Nor de puncte cu o singura serie si inel de suprafata pe marcaje."""
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                color=palette.series_1,
                size=9,
                opacity=0.82,
                line=dict(color=palette.surface, width=2),
            ),
            hovertext=hover,
            hoverinfo="text" if hover else "x+y",
        )
    )
    apply_layout(figure, palette, height=height)
    figure.update_yaxes(showgrid=True, gridcolor=palette.gridline)
    return figure

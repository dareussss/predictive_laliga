"""Potrivirea numelor de echipe intre football-data.org si football-data.co.uk.

Cele doua surse nu au niciun nume identic. Numele canonic e cel din CSV-uri, fiind
sursa istorica. Rezolvarea esueaza zgomotos pentru un nume nemapat, intentionat.
"""

from __future__ import annotations

FOOTBALL_DATA_ORG_TO_CANONICAL: dict[str, str] = {
    "Athletic Club": "Ath Bilbao",
    "CA Osasuna": "Osasuna",
    "Club Atlético de Madrid": "Ath Madrid",
    "Deportivo Alavés": "Alaves",
    "Elche CF": "Elche",
    "FC Barcelona": "Barcelona",
    "Getafe CF": "Getafe",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "RC Celta de Vigo": "Celta",
    "RC Deportivo La Coruña": "La Coruna",
    "RCD Espanyol de Barcelona": "Espanol",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Real Betis Balompié": "Betis",
    "Real Madrid CF": "Real Madrid",
    "Real Racing Club de Santander": "Santander",
    "Real Sociedad de Fútbol": "Sociedad",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "Cádiz CF": "Cadiz",
    "CD Leganés": "Leganes",
    "Girona FC": "Girona",
    "Granada CF": "Granada",
    "RCD Mallorca": "Mallorca",
    "Real Oviedo": "Oviedo",
    "Real Sporting de Gijón": "Sp Gijon",
    "Real Valladolid CF": "Valladolid",
    "Real Zaragoza": "Zaragoza",
    "SD Eibar": "Eibar",
    "SD Huesca": "Huesca",
    "UD Almería": "Almeria",
    "UD Las Palmas": "Las Palmas",
}


class UnknownTeamError(KeyError):
    """Un nume de echipa care nu are corespondent canonic."""


def to_canonical(name: str) -> str:
    """Traduce un nume de la football-data.org in numele canonic din baza de date."""
    try:
        return FOOTBALL_DATA_ORG_TO_CANONICAL[name]
    except KeyError:
        raise UnknownTeamError(
            f"Echipa {name!r} nu are mapare canonica. "
            f"Adaug-o in ingestion/team_names.py."
        ) from None

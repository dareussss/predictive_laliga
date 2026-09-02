"""Schema bazei de date."""

from __future__ import annotations

from datetime import date as date_

from sqlalchemy import Date, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(8), unique=True)
    name: Mapped[str] = mapped_column(String(64))
    country: Mapped[str] = mapped_column(String(64))

    matches: Mapped[list["Match"]] = relationship(back_populates="league")

    def __repr__(self) -> str:
        return f"<League {self.code} {self.name}>"


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    def __repr__(self) -> str:
        return f"<Team {self.name}>"


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("season", "home_team_id", "away_team_id", name="uq_match_natural_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    season: Mapped[str] = mapped_column(String(7), index=True)
    date: Mapped[date_] = mapped_column(Date, index=True)

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)

    matchday: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(16))

    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)
    result: Mapped[str | None] = mapped_column(String(1))

    ht_home_goals: Mapped[int | None] = mapped_column(Integer)
    ht_away_goals: Mapped[int | None] = mapped_column(Integer)

    home_shots: Mapped[int | None] = mapped_column(Integer)
    away_shots: Mapped[int | None] = mapped_column(Integer)
    home_shots_on_target: Mapped[int | None] = mapped_column(Integer)
    away_shots_on_target: Mapped[int | None] = mapped_column(Integer)
    home_corners: Mapped[int | None] = mapped_column(Integer)
    away_corners: Mapped[int | None] = mapped_column(Integer)
    home_fouls: Mapped[int | None] = mapped_column(Integer)
    away_fouls: Mapped[int | None] = mapped_column(Integer)
    home_yellow: Mapped[int | None] = mapped_column(Integer)
    away_yellow: Mapped[int | None] = mapped_column(Integer)
    home_red: Mapped[int | None] = mapped_column(Integer)
    away_red: Mapped[int | None] = mapped_column(Integer)

    b365_home: Mapped[float | None] = mapped_column(Float)
    b365_draw: Mapped[float | None] = mapped_column(Float)
    b365_away: Mapped[float | None] = mapped_column(Float)
    avg_home: Mapped[float | None] = mapped_column(Float)
    avg_draw: Mapped[float | None] = mapped_column(Float)
    avg_away: Mapped[float | None] = mapped_column(Float)
    avg_close_home: Mapped[float | None] = mapped_column(Float)
    avg_close_draw: Mapped[float | None] = mapped_column(Float)
    avg_close_away: Mapped[float | None] = mapped_column(Float)

    league: Mapped["League"] = relationship(back_populates="matches")
    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])

    def __repr__(self) -> str:
        return (
            f"<Match {self.season} {self.date} "
            f"{self.home_team_id} {self.home_goals}-{self.away_goals} {self.away_team_id}>"
        )


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    fbref_url: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(96), index=True)
    born: Mapped[int | None] = mapped_column(Integer)
    nationality: Mapped[str | None] = mapped_column(String(8))

    seasons: Mapped[list["PlayerSeasonStats"]] = relationship(back_populates="player")

    def __repr__(self) -> str:
        return f"<Player {self.name} ({self.born})>"


class PlayerSeasonStats(Base):
    __tablename__ = "player_season_stats"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "squad", name="uq_player_season_squad"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    season: Mapped[str] = mapped_column(String(7), index=True)
    squad: Mapped[str] = mapped_column(String(64), index=True)

    position: Mapped[str | None] = mapped_column(String(16))
    age: Mapped[float | None] = mapped_column(Float)

    matches: Mapped[int | None] = mapped_column(Integer)
    starts: Mapped[int | None] = mapped_column(Integer)
    minutes: Mapped[int | None] = mapped_column(Integer)

    goals: Mapped[int | None] = mapped_column(Integer)
    assists: Mapped[int | None] = mapped_column(Integer)
    penalties: Mapped[int | None] = mapped_column(Integer)
    penalties_attempted: Mapped[int | None] = mapped_column(Integer)
    yellow: Mapped[int | None] = mapped_column(Integer)
    red: Mapped[int | None] = mapped_column(Integer)

    xg: Mapped[float | None] = mapped_column(Float)
    npxg: Mapped[float | None] = mapped_column(Float)
    xag: Mapped[float | None] = mapped_column(Float)

    progressive_carries: Mapped[int | None] = mapped_column(Integer)
    progressive_passes: Mapped[int | None] = mapped_column(Integer)

    player: Mapped["Player"] = relationship(back_populates="seasons")

    def __repr__(self) -> str:
        return f"<PlayerSeasonStats {self.player_id} {self.season} {self.squad}>"


class ScorerSeason(Base):
    """Marcatorii unui sezon, de la football-data.org.

    Tabel separat de `player_season_stats`: sursa nu are minute sau xG, dar acopera
    sezonul pe care arhiva FBref nu-l are.
    """

    __tablename__ = "scorer_seasons"
    __table_args__ = (
        UniqueConstraint("season", "player_name", "team", name="uq_scorer_season"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    season: Mapped[str] = mapped_column(String(7), index=True)
    player_name: Mapped[str] = mapped_column(String(96), index=True)
    team: Mapped[str] = mapped_column(String(64), index=True)

    date_of_birth: Mapped[str | None] = mapped_column(String(10))
    nationality: Mapped[str | None] = mapped_column(String(64))
    section: Mapped[str | None] = mapped_column(String(32))

    played_matches: Mapped[int | None] = mapped_column(Integer)
    goals: Mapped[int | None] = mapped_column(Integer)
    assists: Mapped[int | None] = mapped_column(Integer)
    penalties: Mapped[int | None] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"<ScorerSeason {self.season} {self.player_name} {self.goals}g>"

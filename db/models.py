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

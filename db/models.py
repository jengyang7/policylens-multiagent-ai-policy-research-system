from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class ResearchRun(Base):
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    # pending | awaiting_clarification | running | done | failed
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    # {questions: [...], answers: [...]} populated during human-in-the-loop clarification
    clarifications: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # {subtasks: [...]} set by the plan node
    plan: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    stats: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    subtasks: Mapped[list[Subtask]] = relationship(back_populates="run", cascade="all, delete-orphan")
    sources: Mapped[list[Source]] = relationship(back_populates="run", cascade="all, delete-orphan")
    report: Mapped[Optional[Report]] = relationship(back_populates="run", uselist=False, cascade="all, delete-orphan")


class Subtask(Base):
    __tablename__ = "subtasks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # pending | running | done | failed
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")

    run: Mapped[ResearchRun] = relationship(back_populates="subtasks")
    findings: Mapped[list[Finding]] = relationship(back_populates="subtask", cascade="all, delete-orphan")


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped[ResearchRun] = relationship(back_populates="sources")
    findings: Mapped[list[Finding]] = relationship(back_populates="source")


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    subtask_id: Mapped[str] = mapped_column(String, ForeignKey("subtasks.id", ondelete="CASCADE"), nullable=False)
    source_id: Mapped[str] = mapped_column(String, ForeignKey("sources.id", ondelete="SET NULL"), nullable=True)
    # Mirrors the Pydantic Finding schema in engine/extraction.py
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_span: Mapped[str] = mapped_column(Text, nullable=False)
    citation_url: Mapped[str] = mapped_column(Text, nullable=False)

    subtask: Mapped[Subtask] = relationship(back_populates="findings")
    source: Mapped[Optional[Source]] = relationship(back_populates="findings")


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (UniqueConstraint("run_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    structured: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    run: Mapped[ResearchRun] = relationship(back_populates="report")

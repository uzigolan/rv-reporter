from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Section(BaseModel):
    title: str
    body: str


class Alert(BaseModel):
    severity: str
    message: str


class Recommendation(BaseModel):
    priority: str
    action: str


class ReportOutput(BaseModel):
    report_type_id: str
    report_title: str
    summary: str
    sections: list[Section] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

"""Strict writer-only contract for narrative generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class WriterContract(ABC):
    """Strict interface for report narrative writing (generation only, no data prep)."""

    @abstractmethod
    def generate_narrative(
        self,
        *,
        prompt: str,
        sections_template: list[dict[str, Any]] | None = None,
        output_token_budget: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate report narrative (structured JSON) from a prompt.

        Args:
            prompt: Full prompt text for narrative generation
            sections_template: Optional list of section templates to structure output
            output_token_budget: Optional budget for output tokens

        Returns:
            Dict with:
              - report_type_id: str
              - report_title: str
              - summary: str
              - sections: list[dict] with title, body
              - alerts: list[dict] with severity, message
              - recommendations: list[dict] with priority, action
              - tables: list[dict] with rows
              - charts: list[dict] with data
              - metadata: dict with generation metadata
        """
        raise NotImplementedError

    def estimate_cost(
        self,
        *,
        prompt: str,
        estimated_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Estimate cost/token usage for narrative generation.

        Returns dict with keys: total_cost_usd_est, input_tokens_est, output_tokens_est, model
        """
        raise NotImplementedError

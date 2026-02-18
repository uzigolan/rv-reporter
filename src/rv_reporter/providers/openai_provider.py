from __future__ import annotations

import json
from typing import Any

from rv_reporter.providers.base import ReportProvider
from rv_reporter.report_types.registry import ReportTypeDefinition


class OpenAIResponsesProvider(ReportProvider):
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError("Install openai extra: pip install -e .[openai]") from exc
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self._client = OpenAI(**client_kwargs)
        self._model = model
        self.last_raw_response: dict[str, Any] | None = None

    def generate_report_json(
        self,
        definition: ReportTypeDefinition,
        csv_profile: dict[str, Any],
        metrics: dict[str, Any],
        user_prefs: dict[str, Any],
    ) -> dict[str, Any]:
        instructions = _model_instructions()
        input_payload = build_model_input_payload(definition, csv_profile, metrics, user_prefs)
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=[{"role": "user", "content": [{"type": "input_text", "text": json.dumps(input_payload)}]}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": f"{definition.report_type_id}_report",
                    "schema": definition.output_schema,
                    "strict": False,
                }
            },
        )
        if hasattr(response, "model_dump"):
            self.last_raw_response = response.model_dump()
        else:
            self.last_raw_response = {"raw_response_text": getattr(response, "output_text", "")}
        return json.loads(response.output_text)


def _model_instructions() -> str:
    return (
        "Produce a JSON object that strictly follows the provided JSON schema. "
        "Do not include keys that are not in the schema."
    )


def build_model_input_payload(
    definition: ReportTypeDefinition,
    csv_profile: dict[str, Any],
    metrics: dict[str, Any],
    user_prefs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_type_id": definition.report_type_id,
        "report_title": definition.title,
        "custom_instructions": definition.prompt_instructions,
        "csv_profile": csv_profile,
        "metrics": metrics,
        "user_prefs": user_prefs,
    }


def build_model_prompt_for_estimation(
    definition: ReportTypeDefinition,
    csv_profile: dict[str, Any],
    metrics: dict[str, Any],
    user_prefs: dict[str, Any],
) -> str:
    payload = build_model_input_payload(definition, csv_profile, metrics, user_prefs)
    return f"{_model_instructions()}\n{json.dumps(payload)}"

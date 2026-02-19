from __future__ import annotations

import json
import re
from typing import Any

from rv_reporter.providers.base import ReportProvider
from rv_reporter.providers.openai_provider import build_model_input_payload
from rv_reporter.report_types.registry import ReportTypeDefinition


class OpenAIChatCompletionsProvider(ReportProvider):
    """Use OpenAI-compatible Chat Completions APIs when Responses API is unavailable."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        json_repair_model: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError("Install openai extra: pip install -e .[openai]") from exc
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers
        self._client = OpenAI(**kwargs)
        self._model = model
        self._json_repair_model = json_repair_model
        self.last_raw_response: dict[str, Any] | None = None

    def generate_report_json(
        self,
        definition: ReportTypeDefinition,
        csv_profile: dict[str, Any],
        metrics: dict[str, Any],
        user_prefs: dict[str, Any],
    ) -> dict[str, Any]:
        payload = build_model_input_payload(definition, csv_profile, metrics, user_prefs)
        prompt = (
            "Return only valid JSON (no markdown fences) that matches this schema. "
            "Do not add extra keys. "
            f"Schema:\n{json.dumps(definition.output_schema)}\n\n"
            f"Input:\n{json.dumps(payload)}"
        )
        response = self._chat_complete(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a strict JSON report generator."},
                {"role": "user", "content": prompt},
            ],
        )
        if hasattr(response, "model_dump"):
            self.last_raw_response = response.model_dump()
        else:
            self.last_raw_response = {"raw_response_text": str(response)}

        content = ""
        if getattr(response, "choices", None):
            content = _extract_message_text(response.choices[0].message.content)
        try:
            return _parse_json_object(content)
        except Exception:  # noqa: BLE001
            repaired = self._repair_to_json(content=content, schema=definition.output_schema)
            return _parse_json_object(repaired)

    def _repair_to_json(self, content: str, schema: dict[str, Any]) -> str:
        repair_prompt = (
            "Convert the following content into one strict JSON object only. "
            "No markdown, no comments, no extra text.\n\n"
            f"Schema:\n{json.dumps(schema)}\n\n"
            f"Content:\n{content}"
        )
        repair_model = self._json_repair_model or self._model
        repair_response = self._chat_complete(
            model=repair_model,
            messages=[
                {"role": "system", "content": "You output strict valid JSON only."},
                {"role": "user", "content": repair_prompt},
            ],
        )
        if hasattr(repair_response, "model_dump"):
            self.last_raw_response = repair_response.model_dump()
        if getattr(repair_response, "choices", None):
            return _extract_message_text(repair_response.choices[0].message.content)
        return content

    def _chat_complete(self, model: str, messages: list[dict[str, Any]]) -> Any:
        # Some OpenRouter models ignore or reject response_format=json_object.
        try:
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception:  # noqa: BLE001
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )


def _extract_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates: list[str] = []
    candidates.append(cleaned)
    balanced = _extract_first_balanced_json_object(cleaned)
    if balanced:
        candidates.append(balanced)

    for raw_candidate in candidates:
        for candidate in (raw_candidate, _cleanup_common_json_issues(raw_candidate)):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:  # noqa: BLE001
                continue
    raise ValueError("Provider response did not contain a valid JSON object.")


def _extract_first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _cleanup_common_json_issues(text: str) -> str:
    fixed = text
    # Remove trailing commas before } or ].
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
    # Strip any leading noise before first JSON object.
    first_brace = fixed.find("{")
    if first_brace > 0:
        fixed = fixed[first_brace:]
    # Strip trailing noise after last }.
    last_brace = fixed.rfind("}")
    if last_brace >= 0:
        fixed = fixed[: last_brace + 1]
    return fixed.strip()

from __future__ import annotations

import json
import time
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from rv_reporter.providers.base import ReportProvider
from rv_reporter.providers.openai_chat_provider import _parse_json_object
from rv_reporter.providers.openai_provider import build_model_input_payload
from rv_reporter.report_types.registry import ReportTypeDefinition


class AnthropicMessagesProvider(ReportProvider):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = (base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.last_raw_response: dict[str, Any] | None = None

    def generate_report_json(
        self,
        definition: ReportTypeDefinition,
        csv_profile: dict[str, Any],
        metrics: dict[str, Any],
        user_prefs: dict[str, Any],
    ) -> dict[str, Any]:
        payload = build_model_input_payload(definition, csv_profile, metrics, user_prefs)
        user_prompt = (
            "Generate the final report payload according to the schema. "
            "Do not add extra keys. "
            "Write concise, structured content: for each section body prefer short lines and bullet points "
            "(markdown '-' bullets), not one long paragraph.\n\n"
            f"Schema:\n{json.dumps(definition.output_schema)}\n\n"
            f"Input:\n{json.dumps(payload)}"
        )
        body = {
            "model": self._model,
            "max_tokens": 3200,
            "system": "You are a strict JSON report generator.",
            "messages": [{"role": "user", "content": user_prompt}],
            "tools": [
                {
                    "name": "submit_report",
                    "description": "Submit the final report JSON object.",
                    "input_schema": definition.output_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "submit_report"},
        }
        req = request.Request(
            url=f"{self._base_url}/messages",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        raw = self._perform_request_with_retry(req)
        parsed = json.loads(raw)
        self.last_raw_response = parsed if isinstance(parsed, dict) else {"raw": raw}

        text = ""
        blocks = parsed.get("content", []) if isinstance(parsed, dict) else []
        if isinstance(blocks, list):
            for block in blocks:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "submit_report"
                ):
                    tool_input = block.get("input")
                    if isinstance(tool_input, dict):
                        return tool_input
                if isinstance(block, dict) and block.get("type") == "text":
                    text += str(block.get("text", ""))
        return _parse_json_object(text)

    def _perform_request_with_retry(self, req: request.Request) -> str:
        max_attempts = 3
        backoff_seconds = [0.8, 1.8]
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            try:
                with request.urlopen(req, timeout=120) as resp:  # noqa: S310
                    return resp.read().decode("utf-8")
            except HTTPError as exc:
                last_exc = exc
                # 529/503/502/504 are typically transient upstream overload/gateway issues.
                if exc.code in {529, 503, 502, 504} and attempt < max_attempts - 1:
                    time.sleep(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])
                    continue
                details = ""
                try:
                    details = exc.read().decode("utf-8", errors="ignore").strip()
                except Exception:  # noqa: BLE001
                    details = ""
                if exc.code == 529:
                    raise RuntimeError(
                        "Anthropic API is temporarily overloaded (HTTP 529). "
                        "Please retry in 10-30 seconds."
                    ) from exc
                if details:
                    raise RuntimeError(f"Anthropic API error HTTP {exc.code}: {details[:400]}") from exc
                raise RuntimeError(f"Anthropic API error HTTP {exc.code}.") from exc
            except URLError as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    time.sleep(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])
                    continue
                raise RuntimeError(f"Anthropic connection error: {exc.reason}") from exc

        if last_exc is not None:
            raise RuntimeError(f"Anthropic request failed: {last_exc}") from last_exc
        raise RuntimeError("Anthropic request failed for unknown reason.")

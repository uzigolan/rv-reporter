from __future__ import annotations

import json
import re
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
        self.last_usage: dict[str, int] | None = None
        self.last_usage_cost_usd: float | None = None

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
        self.last_usage = _extract_usage(parsed)
        if self.last_usage:
            self.last_usage_cost_usd = _estimate_usage_cost_usd(self._model, self.last_usage)
        else:
            self.last_usage_cost_usd = None

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


def _extract_usage(payload: dict[str, Any]) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    in_tokens = usage.get("input_tokens")
    out_tokens = usage.get("output_tokens")
    try:
        input_val = int(in_tokens)
        output_val = int(out_tokens)
    except (TypeError, ValueError):
        return None
    if input_val < 0 or output_val < 0:
        return None
    return {"input_tokens": input_val, "output_tokens": output_val}


def _estimate_usage_cost_usd(model: str, usage: dict[str, int]) -> float | None:
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    if input_tokens < 0 or output_tokens < 0:
        return None
    key = _canonical_anthropic_model_key(model)
    pricing_per_m = {
        "claude-sonnet-4": (3.0, 15.0),
        "claude-sonnet-4-5": (3.0, 15.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-opus-4": (15.0, 75.0),
        "claude-opus-4-1": (15.0, 75.0),
        "claude-opus-4-5": (5.0, 25.0),
        "claude-opus-4-6": (5.0, 25.0),
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-3-haiku": (0.25, 1.25),
        "claude-3-7-sonnet-latest": (3.0, 15.0),
    }
    prices = pricing_per_m.get(key)
    if prices is None:
        return None
    input_per_1m, output_per_1m = prices
    total = (input_tokens / 1_000_000) * input_per_1m + (output_tokens / 1_000_000) * output_per_1m
    return round(total, 6)


def _canonical_anthropic_model_key(model: str) -> str:
    key = str(model or "").strip().lower()
    tail = key.split("/", 1)[1] if "/" in key else key
    tail = tail.replace(".", "-")
    tail = re.sub(r"-\d{8}$", "", tail)
    if tail.startswith("claude-sonnet-4-6"):
        return "claude-sonnet-4-6"
    if tail.startswith("claude-sonnet-4-5"):
        return "claude-sonnet-4-5"
    if tail.startswith("claude-sonnet-4"):
        return "claude-sonnet-4"
    if tail.startswith("claude-opus-4-6"):
        return "claude-opus-4-6"
    if tail.startswith("claude-opus-4-5"):
        return "claude-opus-4-5"
    if tail.startswith("claude-opus-4-1"):
        return "claude-opus-4-1"
    if tail.startswith("claude-opus-4"):
        return "claude-opus-4"
    if tail.startswith("claude-haiku-4-5"):
        return "claude-haiku-4-5"
    if tail.startswith("claude-3-7-sonnet"):
        return "claude-3-7-sonnet-latest"
    if tail.startswith("claude-3-haiku") or tail.startswith("claude-3-5-haiku"):
        return "claude-3-haiku"
    return tail

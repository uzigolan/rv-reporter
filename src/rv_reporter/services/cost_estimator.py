from __future__ import annotations

import math
from dataclasses import dataclass


# Verified from OpenAI pricing docs on 2026-02-16.
PRICING_SOURCE_URL = "https://platform.openai.com/docs/pricing"
PRICING_VERIFIED_DATE = "2026-02-16"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float


MODEL_PRICING_USD: dict[str, ModelPricing] = {
    "gpt-5.2": ModelPricing(input_per_1m=1.75, output_per_1m=14.00),
    "gpt-5.1": ModelPricing(input_per_1m=1.25, output_per_1m=10.00),
    "gpt-5": ModelPricing(input_per_1m=1.25, output_per_1m=10.00),
    "gpt-5-mini": ModelPricing(input_per_1m=0.25, output_per_1m=2.00),
    "gpt-5-nano": ModelPricing(input_per_1m=0.05, output_per_1m=0.40),
    "gpt-4.1": ModelPricing(input_per_1m=2.00, output_per_1m=8.00),
    "gpt-4.1-mini": ModelPricing(input_per_1m=0.40, output_per_1m=1.60),
    "gpt-4.1-nano": ModelPricing(input_per_1m=0.10, output_per_1m=0.40),
}


def estimate_tokens(text: str) -> int:
    # Fast approximation for planning: ~4 characters per token for English-like JSON text.
    return max(1, math.ceil(len(text) / 4))


def estimate_openai_cost(
    *,
    model: str,
    prompt_text: str,
    estimated_output_tokens: int = 1200,
) -> dict[str, float | int | str]:
    if model not in MODEL_PRICING_USD:
        known = ", ".join(sorted(MODEL_PRICING_USD.keys()))
        raise ValueError(f"Unknown pricing for model '{model}'. Known models: {known}")

    pricing = MODEL_PRICING_USD[model]
    input_tokens = estimate_tokens(prompt_text)
    output_tokens = max(1, int(estimated_output_tokens))

    input_cost = (input_tokens / 1_000_000) * pricing.input_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_1m
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "input_cost_usd_est": round(input_cost, 6),
        "output_cost_usd_est": round(output_cost, 6),
        "total_cost_usd_est": round(total_cost, 6),
        "pricing_source_url": PRICING_SOURCE_URL,
        "pricing_verified_date": PRICING_VERIFIED_DATE,
    }

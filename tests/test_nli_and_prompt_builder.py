"""Tests for natural language intent extraction and prompt building."""

import pytest
from rv_reporter.agents.nli_extractor import NaturalLanguageIntentExtractor
from rv_reporter.providers.prompt_builder import PromptBuilder
from rv_reporter.report_types.registry import ReportTypeRegistry
from pathlib import Path


def test_nli_extractor_heuristic_keyword_matching():
    """Test that NLI extractor can match keywords to report types."""
    registry = ReportTypeRegistry(config_dir=Path("configs/report_types"))
    extractor = NaturalLanguageIntentExtractor(registry=registry)
    
    result = extractor.extract(
        user_description="analyze network queue latency issues",
        available_report_types=["network_queue_congestion", "twamp_session_health"],
    )
    
    assert result["report_type_id"] in ["network_queue_congestion", "twamp_session_health"]
    assert result["confidence"] > 0
    assert "reasoning" in result
    assert isinstance(result["user_prefs"], dict)


def test_nli_extractor_empty_description():
    """Test that NLI extractor handles empty descriptions gracefully."""
    extractor = NaturalLanguageIntentExtractor()
    
    result = extractor.extract(user_description="")
    
    assert result["report_type_id"] == ""
    assert result["confidence"] == 0.0
    assert "No user description" in result["reasoning"]


def test_nli_extractor_preference_extraction():
    """Test that NLI extractor can extract user preferences from description."""
    extractor = NaturalLanguageIntentExtractor()
    
    result = extractor.extract(
        user_description="I need a deep-dive technical analysis of performance issues for the executive team",
        available_report_types=["pm_export_health"],
    )
    
    # Should extract some preferences
    assert isinstance(result["user_prefs"], dict)


def test_prompt_builder_generates_valid_prompt(tmp_path):
    """Test that PromptBuilder generates valid generation prompts."""
    registry = ReportTypeRegistry(config_dir=Path("configs/report_types"))
    definition = registry.get("network_queue_congestion")
    
    csv_profile = {
        "row_count": 100,
        "column_count": 5,
        "dtypes": ["int64", "float64", "object"],
    }
    
    metrics = {
        "avg_queue_length": 42.5,
        "peak_queue_length": 150,
    }
    
    user_prefs = {
        "tone": "assertive",
        "audience": "technical",
    }
    
    prompt = PromptBuilder.build_generation_prompt(
        definition=definition,
        csv_profile=csv_profile,
        metrics=metrics,
        user_prefs=user_prefs,
    )
    
    # Verify prompt contains key elements
    assert "Report Type:" in prompt
    assert "network_queue_congestion" in prompt.lower() or "Data Summary" in prompt
    assert "Rows analyzed: 100" in prompt
    assert "Output Format" in prompt
    assert len(prompt) > 100


def test_prompt_builder_with_agent_plan(tmp_path):
    """Test that PromptBuilder includes agent plan in prompt."""
    registry = ReportTypeRegistry(config_dir=Path("configs/report_types"))
    definition = registry.get("network_queue_congestion")
    
    agent_plan = {
        "sections": [
            {
                "title": "Overview",
                "purpose": "High-level summary",
            },
            {
                "title": "Metrics",
                "purpose": "Detailed metrics",
            },
        ]
    }
    
    prompt = PromptBuilder.build_generation_prompt(
        definition=definition,
        csv_profile={"row_count": 100, "column_count": 5, "dtypes": []},
        metrics={},
        user_prefs={},
        agent_plan=agent_plan,
    )
    
    # Should include agent plan context
    assert "Overview" in prompt or "Metrics" in prompt or "Report Structure" in prompt


def test_prompt_builder_estimation_prompt_matches_generation():
    """Test that estimation and generation prompts are equivalent."""
    registry = ReportTypeRegistry(config_dir=Path("configs/report_types"))
    definition = registry.get("network_queue_congestion")
    
    csv_profile = {"row_count": 50, "column_count": 3, "dtypes": []}
    metrics = {"test": "value"}
    prefs = {}
    
    gen_prompt = PromptBuilder.build_generation_prompt(
        definition=definition,
        csv_profile=csv_profile,
        metrics=metrics,
        user_prefs=prefs,
    )
    
    est_prompt = PromptBuilder.build_estimation_prompt(
        definition=definition,
        csv_profile=csv_profile,
        metrics=metrics,
        user_prefs=prefs,
    )
    
    # Should be identical
    assert gen_prompt == est_prompt

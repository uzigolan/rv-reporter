from __future__ import annotations

from jsonschema import validate


def validate_report_schema(report_json: dict, output_schema: dict) -> None:
    validate(instance=report_json, schema=output_schema)

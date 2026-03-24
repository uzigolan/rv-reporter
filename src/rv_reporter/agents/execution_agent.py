from __future__ import annotations

from typing import Any

from rv_reporter.agents.models import IntentResolution, PreparedFacts
from rv_reporter.report_types.registry import ReportTypeDefinition
from rv_reporter.report_types.plugins import compute_report_metrics
from rv_reporter.services.ingest import load_csv_with_limit, validate_required_columns
from rv_reporter.services.profiler import profile_dataframe


class DeterministicExecutionAgent:
    """Runs the authoritative backend calculations that the writer layer must not invent."""

    def prepare_facts(
        self,
        *,
        intent: IntentResolution,
        definition: ReportTypeDefinition,
        ignore_columns: list[str] | None = None,
    ) -> PreparedFacts:
        prefs = dict(definition.default_prefs)
        prefs.update(intent.user_prefs)

        df = load_csv_with_limit(intent.csv_path, row_limit=intent.row_limit, sheet_name=intent.sheet_name)
        if ignore_columns:
            drop_columns = [column for column in ignore_columns if column in df.columns]
            if drop_columns:
                df = df.drop(columns=drop_columns)

        validate_required_columns(df, definition.required_columns)
        csv_profile = profile_dataframe(df)
        metrics = compute_report_metrics(
            metrics_profile=definition.metrics_profile,
            df=df,
            prefs=prefs,
            report_type_id=definition.report_type_id,
        )
        return PreparedFacts(definition=definition, prefs=prefs, csv_profile=csv_profile, metrics=metrics)

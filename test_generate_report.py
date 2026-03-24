import pandas as pd
from rv_reporter.report_types.plugins import invalidate_plugin_cache
from rv_reporter.orchestrator import run_pipeline

# Invalidate cache
invalidate_plugin_cache()

# Try to load and generate the report
try:
    json_path, html_path = run_pipeline(
        report_type_id="pm_telecom_sla_test",
        csv_path="samples/pm-csv-es.csv",
        output_dir="outputs/test_pm_telecom"
    )
    print(f"✓ Report generated successfully!")
    print(f"  - JSON: {json_path}")
    print(f"  - HTML: {html_path}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

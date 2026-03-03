from __future__ import annotations

import pandas as pd

from rv_reporter.report_types.plugins import ReportPluginManager


def test_ptp1588_smoke_build() -> None:
    manager = ReportPluginManager(plugin_dir="report_type_plugins")
    result = manager.compute_metrics(
        metrics_profile="ptp1588",
        df=pd.DataFrame(
            {
                "frame_time_epoch": [1000.0, 1000.0],
                "frame_len": [60, 68],
                "src_ip": ["", ""],
                "dst_ip": ["", ""],
                "transport": ["PTP", "PTP"],
                "src_port": [319, 320],
                "dst_port": [319, 320],
                "frame_protocols": ["eth:ptp", "eth:ptp"],
                "ptp_message_type": ["0x01", "0x09"],
                "ptp_sequence_id": [1, 1],
                "ptp_origin_ts_seconds": [1000, None],
                "ptp_origin_ts_nanoseconds": [100, None],
                "ptp_dr_receive_ts_seconds": [None, 1000],
                "ptp_dr_receive_ts_nanoseconds": [None, 27100],
                "ptp_correction_ns": [6500, 19000],
            }
        ),
        prefs={},
        report_type_id="ptp1588",
    )
    assert isinstance(result, dict)

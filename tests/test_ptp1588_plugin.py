from __future__ import annotations

import pandas as pd

from rv_reporter.report_types.plugins import compute_report_metrics


def test_ptp1588_computes_t4_t3_pairs_and_correction_anomalies() -> None:
    df = pd.DataFrame(
        [
            {
                "frame_time_epoch": 1000.0,
                "frame_len": 60,
                "frame_protocols": "eth:ptp",
                "transport": "PTP",
                "src_ip": "",
                "dst_ip": "",
                "src_port": 319,
                "dst_port": 319,
                "ptp_message_type": "0x01",
                "ptp_sequence_id": 10,
                "ptp_domain_number": 24,
                "ptp_source_port_identity": "1",
                "ptp_two_step": 0,
                "ptp_origin_ts_seconds": 1000,
                "ptp_origin_ts_nanoseconds": 100,
                "ptp_correction_ns": 6500,
            },
            {
                "frame_time_epoch": 1000.0,
                "frame_len": 68,
                "frame_protocols": "eth:ptp",
                "transport": "PTP",
                "src_ip": "",
                "dst_ip": "",
                "src_port": 320,
                "dst_port": 320,
                "ptp_message_type": "0x09",
                "ptp_sequence_id": 10,
                "ptp_domain_number": 24,
                "ptp_source_port_identity": "6",
                "ptp_two_step": 0,
                "ptp_dr_receive_ts_seconds": 1000,
                "ptp_dr_receive_ts_nanoseconds": 27100,
                "ptp_correction_ns": 19000,
            },
            {
                "frame_time_epoch": 1010.0,
                "frame_len": 60,
                "frame_protocols": "eth:ptp",
                "transport": "PTP",
                "src_ip": "",
                "dst_ip": "",
                "src_port": 319,
                "dst_port": 319,
                "ptp_message_type": "0x01",
                "ptp_sequence_id": 11,
                "ptp_domain_number": 24,
                "ptp_source_port_identity": "1",
                "ptp_two_step": 0,
                "ptp_origin_ts_seconds": 1010,
                "ptp_origin_ts_nanoseconds": 100,
                "ptp_correction_ns": 6500,
            },
            {
                "frame_time_epoch": 1010.0,
                "frame_len": 68,
                "frame_protocols": "eth:ptp",
                "transport": "PTP",
                "src_ip": "",
                "dst_ip": "",
                "src_port": 320,
                "dst_port": 320,
                "ptp_message_type": "0x09",
                "ptp_sequence_id": 11,
                "ptp_domain_number": 24,
                "ptp_source_port_identity": "6",
                "ptp_two_step": 0,
                "ptp_dr_receive_ts_seconds": 1010,
                "ptp_dr_receive_ts_nanoseconds": 85100,
                "ptp_correction_ns": 7_400_000_000,
            },
        ]
    )

    result = compute_report_metrics(
        metrics_profile="ptp1588",
        df=df,
        prefs={"t4_t3_outlier_ns": 60_000, "correction_anomaly_ns": 1_000_000},
        report_type_id="ptp1588",
    )

    assert result["ptp1588_t4_t3_analysis"]["t4_t3_stats"]["pairs"] == 2
    assert result["ptp1588_t4_t3_analysis"]["t4_t3_stats"]["max_ns"] == 85000.0
    assert result["ptp1588_correction_anomalies"]["count"] == 1
    assert any("correction-field anomalies" in a["message"] for a in result["alerts"])

import pandas as pd

from rv_reporter.services.metrics import _compute_wireshark_capture_health


def test_wireshark_metrics_extracts_ptp_port_health_signals() -> None:
    df = pd.DataFrame(
        [
            {
                "frame_time_epoch": 1000.000000,
                "frame_len": 128,
                "src_ip": "10.0.0.1",
                "dst_ip": "224.0.1.129",
                "transport": "PTP",
                "src_port": 319,
                "dst_port": 319,
                "frame_protocols": "eth:ip:udp:ptp",
                "ptp_message_type": 0,
                "ptp_domain_number": 24,
                "ptp_source_port_identity": "00:11:22:33:44:55:66:04",
                "ptp_correction_ns": 12,
                "ptp_origin_ts_seconds": 999,
                "ptp_origin_ts_nanoseconds": 999_999_900,
                "ptp_two_step": 1,
                "source_file": "port4.pcapng",
            },
            {
                "frame_time_epoch": 1000.000125,
                "frame_len": 128,
                "src_ip": "10.0.0.1",
                "dst_ip": "224.0.1.129",
                "transport": "PTP",
                "src_port": 320,
                "dst_port": 320,
                "frame_protocols": "eth:ip:udp:ptp",
                "ptp_message_type": 8,
                "ptp_domain_number": 24,
                "ptp_source_port_identity": "00:11:22:33:44:55:66:04",
                "ptp_correction_ns": 14,
                "ptp_origin_ts_seconds": 1000,
                "ptp_origin_ts_nanoseconds": 120,
                "ptp_two_step": 1,
                "source_file": "port4.pcapng",
            },
            {
                "frame_time_epoch": 1000.001000,
                "frame_len": 128,
                "src_ip": "10.0.0.2",
                "dst_ip": "224.0.1.129",
                "transport": "PTP",
                "src_port": 319,
                "dst_port": 319,
                "frame_protocols": "eth:ip:udp:ptp",
                "ptp_message_type": 0,
                "ptp_domain_number": 24,
                "ptp_source_port_identity": "00:11:22:33:44:55:66:03",
                "ptp_correction_ns": 200,
                "ptp_origin_ts_seconds": 1000,
                "ptp_origin_ts_nanoseconds": 900,
                "ptp_two_step": 1,
                "source_file": "port3.pcapng",
            },
            {
                "frame_time_epoch": 1000.002000,
                "frame_len": 128,
                "src_ip": "10.0.0.2",
                "dst_ip": "224.0.1.129",
                "transport": "PTP",
                "src_port": 319,
                "dst_port": 319,
                "frame_protocols": "eth:ip:udp:ptp",
                "ptp_message_type": 0,
                "ptp_domain_number": 24,
                "ptp_source_port_identity": "00:11:22:33:44:55:66:03",
                "ptp_correction_ns": 260,
                "ptp_origin_ts_seconds": 1000,
                "ptp_origin_ts_nanoseconds": 1_900,
                "ptp_two_step": 1,
                "source_file": "port3.pcapng",
            },
        ]
    )

    metrics = _compute_wireshark_capture_health(df, {})
    ptp_summary = metrics["ptp_summary"]
    assert ptp_summary["packets"] == 4
    assert ptp_summary["g8275_1_likely"] is True
    assert ptp_summary["sync_packets"] == 3
    assert ptp_summary["follow_up_packets"] == 1

    port_health = metrics["ptp_port_health"]
    assert any(r.get("port_number") == 3 for r in port_health)
    assert any(r.get("port_number") == 4 for r in port_health)
    source_cmp = metrics["ptp_source_comparison"]
    assert len(source_cmp) == 2
    assert source_cmp[0]["source_file"] == "port3.pcapng"
    assert source_cmp[1]["source_file"] == "port4.pcapng"
    assert metrics["ptp_message_time_trend"]
    assert metrics["ptp_message_time_trend"][0]["sync_packets"] >= 1
    assert metrics["ptp_time_sync_logic"]["lock_likelihood"] in {"low", "medium", "high"}
    assert metrics["ptp_time_sync_logic"]["focus_scope"] == "g8275_1_domain24"
    assert metrics["ptp_state_flow"]
    assert metrics["ptp_state_flow"][0]["state"] in {"stable", "warning", "unstable"}
    assert any("port 3 appears weaker than port 4" in a["message"] for a in metrics["alerts"])

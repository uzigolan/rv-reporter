def get_spec():
    return {
        "id": "telecom_session_health_threshold_sla",
        "title": "Telecom Session Health Threshold SLA Report",
        "description": "A report for evaluating telecom session health metrics against predefined SLA thresholds using TWAMP data.",
        "columns": [
            "twampControllerId","twampPeerAddrType","twampPeerAddr","twampContSessionId",
            "twampReportCurrentStartDateAndTime","twampReportCurrentElapsedTime","twampReportCurrentTxPackets",
            "twampReportCurrentRxValidPackets","twampReportCurrentLossPackets","twampReportCurrentAvailableSeconds",
            "twampReportCurrentDelayMin","twampReportCurrentDelayMax","twampReportCurrentDelaySum",
            "twampReportCurrentDelayAverage","twampReportCurrentDelayedPackets","twampReportCurrentPdvMax",
            "twampReportCurrentIpdvMax","twampReportCurrentIpdvSum","twampReportCurrentIpdvValidResults",
            "twampReportCurrentIpdvFwdMax","twampReportCurrentIpdvFwdSum","twampReportCurrentIpdvFwdValidResults",
            "twampReportCurrentIpdvBckMax","twampReportCurrentIpdvBckSum","twampReportCurrentIpdvBckValidResults",
            "twampReportCurrentReorderedFwd","twampReportCurrentReorderedBck","twampReportCurrentDuplicateFwd",
            "twampReportCurrentDuplicateBck","twampReportCurrentFragmentedFwd","twampReportCurrentFragmentedBck",
            "twampReportCurrentDelayFwdMin","twampReportCurrentDelayFwdMax","twampReportCurrentDelayFwdSum",
            "twampReportCurrentDelayBckMin","twampReportCurrentDelayBckMax","twampReportCurrentDelayBckSum",
            "twampReportCurrentDelayedPacketsFwd","twampReportCurrentDelayedPacketsBck","twampReportCurrentPdvMaxFwd",
            "twampReportCurrentPdvMaxBck","twampReportCurrentTxPacketsFwd","twampReportCurrentTxPacketsBck",
            "twampReportCurrentRxValidPacketsFwd","twampReportCurrentRxValidPacketsBck","twampReportCurrentLossPacketsFwd",
            "twampReportCurrentLossPacketsBck","twampReportCurrentAvailableSecondsFwd","twampReportCurrentAvailableSecondsBck",
            "twampReportCurrentRxSyncValidPacketsFwd","twampReportCurrentRxSyncValidPacketsBck","twampReportCurrentSyncSeconds",
            "twampReportCurrentReordered","twampReportCurrentDuplicate"
        ],
    }

def build(df, prefs, ctx):
    sla = prefs.get('sla_thresholds', {})
    delay_critical = sla.get('delay_critical', 50)
    delay_degraded = sla.get('delay_degraded', 10)
    jitter_critical = sla.get('jitter_critical', 30)
    jitter_degraded = sla.get('jitter_degraded', 10)
    packet_loss_critical = sla.get('packet_loss_critical', 5)
    packet_loss_degraded = sla.get('packet_loss_degraded', 1)

    results = {
        "total_sessions": len(df),
        "critical_delay_violations": len(df[df['twampReportCurrentDelayMax'].astype(float) > delay_critical]),
        "degraded_delay_violations": len(df[(df['twampReportCurrentDelayMax'].astype(float) <= delay_critical) &
                                             (df['twampReportCurrentDelayMax'].astype(float) > delay_degraded)]),
        "critical_packet_loss_violations": len(df[df['twampReportCurrentLossPackets'].astype(float) > packet_loss_critical]),
        "degraded_packet_loss_violations": len(df[(df['twampReportCurrentLossPackets'].astype(float) <= packet_loss_critical) &
                                                   (df['twampReportCurrentLossPackets'].astype(float) > packet_loss_degraded)]),
    }
    return results

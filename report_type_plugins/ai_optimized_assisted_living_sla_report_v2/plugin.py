def get_spec():
    return {
        "column_types": {
            "timestamp": "datetime",
            "device_name": "string",
            "property": "string"
        },
        "preferences": {
            "sla_response_threshold": 300,
            "sla_resolution_threshold": 900,
            "risk_score_weights": {
                "high_call_frequency": 30,
                "response_time": 25,
                "incomplete_incidents": 25,
                "resolution_time": 20
            },
            "caregiver_cost_range": [20, 30],
            "shift_plane_cost": 25
        }
    }

def build(df, prefs, ctx):
    import pandas as pd
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Sort by timestamp
    df = df.sort_values(by='timestamp')
    
    # Initialize metrics
    total_incidents = 0
    compliant_incidents = 0
    total_response_time = 0
    total_resolution_time = 0
    risk_scores = {}
    caregiver_performance = {}

    # Internal function to calculate incident metrics
    def calculate_incidents(grouped):
        nonlocal total_incidents, compliant_incidents, total_response_time, total_resolution_time
        incident_count = 0
        resp_time, res_time = None, None
        accepted, res_confirmed_res, res_confirmed_care = False, False, False
        for _, row in grouped.iterrows():
            if "Call Button Pressed" in row['property']:
                incident_count += 1
                call_time = row['timestamp']
            elif "Call Accepted" in row['property']:
                accepted = True
                if call_time:
                    resp_time = (row['timestamp'] - call_time).seconds
                    total_response_time += resp_time
            elif "Resolution Confirmed" in row['property']:
                if "Rm" in row['device_name']:
                    res_confirmed_res = True
                elif "Caregiver" in row['device_name']:
                    res_confirmed_care = True
                    if call_time:
                        res_time = (row['timestamp'] - call_time).seconds
                        total_resolution_time += res_time
                    
        is_compliant = (accepted and res_confirmed_res and res_confirmed_care)
        if is_compliant:
            compliant_incidents += 1
        total_incidents += incident_count

    # Group by resident
    for name, group in df.groupby('device_name'):
        calculate_incidents(group)

    # Calculate averages
    avg_resp_time = total_response_time / total_incidents if total_incidents > 0 else 0
    avg_res_time = total_resolution_time / total_incidents if total_incidents > 0 else 0
    sla_compliance_percentage = (compliant_incidents / total_incidents * 100) if total_incidents > 0 else 0

    # Placeholder risk and performance calculations
    risk_scores = {"Resident": {}, "Caregiver": {}}
    caregiver_performance = {}

    # Example risk score calculations, these can be detailed and driven by actual parsing above
    for name in df['device_name'].unique():
        if 'Rm' in name:
            risk_scores["Resident"][name] = 50  # placeholder score
        else:
            risk_scores["Caregiver"][name] = 50  # placeholder score

    return {
        "total_incidents": total_incidents,
        "percentage_sla_compliance": sla_compliance_percentage,
        "average_response_time": avg_resp_time,
        "average_resolution_time": avg_res_time,
        "resident_risk_scores": risk_scores["Resident"],
        "caregiver_risk_scores": risk_scores["Caregiver"],
        "caregiver_performance": caregiver_performance
    }

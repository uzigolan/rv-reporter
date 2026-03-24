def get_spec():
    return {
        "description": "Report analyzing SLA compliance and response times in assisted living environments.",
        "required_columns": ["timestamp", "device_name", "property"],
        "default_prefs": {"sla_response_threshold": 10, "sla_resolution_threshold": 30, "caregiver_hourly_rate": 25},
    }

def build(df, prefs, ctx):
    import pandas as pd
    
    # Sort by timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Extract residents and caregivers
    df['resident'] = df['device_name'].apply(lambda x: x if 'Rm ' in x else None)
    df['caregiver'] = df['device_name'].apply(lambda x: x if 'Caregiver ' in x else None)

    # Identify events
    call_pressed = df[df['property'] == 'Call Button Pressed']
    call_accepted = df[df['property'] == 'Call Accepted']
    resolution_confirmed = df[df['property'] == 'Resolution Confirmed']

    # Compute metrics
    incidents_count = len(call_pressed)
    response_times = (call_accepted['timestamp'].reset_index(drop=True) - 
                      call_pressed['timestamp'].reset_index(drop=True)).dt.total_seconds() / 60
    resolution_times = (resolution_confirmed['timestamp'].reset_index(drop=True)[1::2] - 
                        call_pressed['timestamp'].reset_index(drop=True)).dt.total_seconds() / 60
    compliance = len(resolution_confirmed) == 2 * len(call_pressed)
    avg_response_time = response_times.mean()
    avg_resolution_time = resolution_times.mean()
    
    # Determine overall risk level based on thresholds
    risk_level = 'Low'
    if avg_response_time > prefs['sla_response_threshold'] or avg_resolution_time > prefs['sla_resolution_threshold']:
        risk_level = 'Moderate'
        if any(resolution_times > prefs['sla_resolution_threshold']):
            risk_level = 'High'

    results = {
        "Total Incidents": incidents_count,
        "% SLA Compliance": (compliance / incidents_count) * 100,
        "Avg Response Time": avg_response_time,
        "Avg Resolution Time": avg_resolution_time,
        "Overall Risk Level": risk_level,
    }
    
    # More sections would be computed here...
    
    return results

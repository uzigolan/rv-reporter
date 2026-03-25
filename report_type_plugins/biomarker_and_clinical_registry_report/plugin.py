def get_spec():
    return {
        'id': 'biomarker_and_clinical_registry_report',
        'name': 'Biomarker and Clinical Registry Data Report',
        'description': 'Evaluates metrics and integrity of clinical device data.',
        'domain': 'healthcare',
        'mode': 'statistical_summary',
        'columns': ['timestamp', 'device_name', 'property'],
        'alerts': [
            {'condition': "missing_timestamps", 'message_template': "High number of missing timestamps detected.", 'severity': 'High'},
            {'condition': "invalid_timestamp_format", 'message_template': "Some timestamps are not in the correct YYYY-MM-DD format.", 'severity': 'Medium'},
            {'condition': "deviation_in_property", 'message_template': "Some 'property' values are slightly outside expected ranges.", 'severity': 'Low'}
        ],
        'charts': [
            {'chart_type': 'bar', 'title': 'Frequency of Device Usage', 'x_axis': 'device_name', 'y_axis': 'frequency'},
            {'chart_type': 'histogram', 'title': 'Property Distribution Histogram', 'x_axis': 'property value', 'y_axis': 'frequency'}
        ],
        'tables': [
            {'title': 'Descriptive Statistics Summary', 'columns': ['Statistic', 'Value']}
        ],
        'sections': [
            {'title': 'Introduction', 'purpose': "Introduce the report's goals and audience."},
            {'title': 'Data Schema Overview', 'purpose': "Describe expected data types and constraints."},
            {'title': 'Descriptive Statistics', 'purpose': "Present mean, median, and standard deviation for 'property'."},
            {'title': 'Distribution Analysis', 'purpose': "Analyze distributions for 'device_name' and 'property'."},
            {'title': 'Data Quality Metrics', 'purpose': "Assess data completeness, validity, and consistency."},
            {'title': 'Overall Assessment', 'purpose': "Provide a summary of key findings."}
        ],
        'recommendations': [
            {'action': "Ensure all devices log timestamps in the correct format to improve data consistency.", 'priority': 'High'},
            {'action': "Investigate outliers in the 'property' column to identify potential data entry errors.", 'priority': 'Medium'}
        ],
        'suggestions': [
            "Incorporate timestamp time zones to account for device location variations.",
            "Add a 'patient_id' column to enable patient-specific analysis."
        ]
    }

def build(df, prefs, ctx):
    import pandas as pd
    import numpy as np
    from collections import Counter

    results = {}
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    missing_timestamps = df['timestamp'].isnull().mean()

    # Descriptive statistics for 'property'
    property_numeric = pd.to_numeric(df['property'], errors='coerce')
    descriptive_stats = {
        'mean': property_numeric.mean(),
        'median': property_numeric.median(),
        'std': property_numeric.std()
    }

    # Frequency analysis for 'device_name'
    device_counts = Counter(df['device_name'])

    # Histogram data for 'property'
    histogram_data = np.histogram(property_numeric.dropna())

    results['descriptive_statistics'] = descriptive_stats
    results['device_usage_frequency'] = device_counts
    results['property_histogram'] = histogram_data

    # Data Quality Metrics
    results['missing_timestamps'] = missing_timestamps
    results['invalid_timestamp_format'] = df['timestamp'].isnull().sum()
    
    # Return results
    return results

import pandas as pd
import pytest

# The test will try to load the plugin module via a ReportPluginManager if available.
# If the manager is not installed in the test environment we fall back to directly importing the plugin module.

try:
    from report_plugin_manager import ReportPluginManager  # type: ignore
except Exception:
    # shim manager for smoke-test environments where the host manager isn't available
    class ReportPluginManager:
        def load_module(self, module):
            return module

import importlib
import sys

# import the plugin file as module; it is expected to be named plugin.py in same package when executed in host
import plugin as plugin_mod

def test_smoke_build_basic():
    # create a small deterministic dataframe with the minimal expected columns
    rows = [
        {
            "Entry OID": "1.3.6.1",
            "Date And Time (Local)": "2022-09-29 09:15:00",
            "Date And Time (UTC)": "2022-09-29 09:15:00+00:00",
            "System Uptime (Seconds)": "100",
            "Device ID": "devA",
            "Interval Length (Seconds)": "60"
        },
        {
            # normal next interval
            "Entry OID": "1.3.6.1",
            "Date And Time (Local)": "2022-09-29 09:16:00",
            "Date And Time (UTC)": "2022-09-29 09:16:00+00:00",
            "System Uptime (Seconds)": "160",
            "Device ID": "devA",
            "Interval Length (Seconds)": "60"
        },
        {
            # skip one interval to create a gap
            "Entry OID": "1.3.6.1",
            "Date And Time (Local)": "2022-09-29 09:18:00",
            "Date And Time (UTC)": "2022-09-29 09:18:05+00:00",
            "System Uptime (Seconds)": "20",  # uptime decreased -> reboot/reset
            "Device ID": "devA",
            "Interval Length (Seconds)": "60"
        }
    ]
    df = pd.DataFrame(rows)

    # load plugin via manager if available
    mgr = ReportPluginManager()
    # support manager.load_module or manager.load
    if hasattr(mgr, 'load_module'):
        plugin = mgr.load_module(plugin_mod)
    elif hasattr(mgr, 'load'):
        plugin = mgr.load(plugin_mod)
    else:
        plugin = plugin_mod

    spec = plugin.get_spec()
    assert spec.get('id') == 'snmp_time_sync_anomaly'

    result = plugin.build(df, prefs={"interval_seconds": 60, "drift_threshold_seconds": 1, "top_n_samples": 5})
    assert 'overall' in result
    assert result['overall']['row_count'] == 3
    # we expect at least one gap, one reset
    assert result['overall']['total_gaps'] >= 1
    assert result['overall']['total_uptime_resets'] >= 1
    assert isinstance(result['anomaly_samples'], list)

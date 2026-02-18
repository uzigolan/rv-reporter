from __future__ import annotations

import html
import json
import re

from jinja2 import Template


_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{{ report.report_title }}</title>
    <style>
      :root {
        --bg: #f1f5ff;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #475569;
        --accent: #0f766e;
        --accent2: #2563eb;
      }
      body { font-family: "Segoe UI", Tahoma, sans-serif; background: radial-gradient(circle at 10% -20%, #dbeafe 0%, #eef2ff 35%, var(--bg) 70%, #ecfeff 100%); color: var(--text); margin: 0; }
      main { max-width: 1120px; margin: 2rem auto; padding: 1rem; }
      body.ms-biomarker-wide main { max-width: 96vw; }
      section { background: rgba(255,255,255,0.9); border: 1px solid #dbe3ef; border-radius: 16px; box-shadow: 0 16px 36px rgba(15,23,42,0.09); padding: 1rem 1.25rem; margin-bottom: 1rem; backdrop-filter: blur(4px); }
      h1, h2 { margin: 0 0 0.75rem; }
      .summary { color: var(--muted); }
      .muted { color: var(--muted); }
      .chip { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 999px; background: #dbeafe; color: var(--accent2); margin-right: 0.4rem; font-size: 0.82rem; border: 1px solid #bfdbfe; }
      ul { margin: 0.5rem 0 0; padding-left: 1.1rem; }
      table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
      th, td { border-bottom: 1px solid #dbe3ef; padding: 0.45rem; text-align: left; vertical-align: top; }
      .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.7rem; }
      .stat-card { border: 1px solid #d9e3f0; border-radius: 12px; padding: 0.72rem 0.75rem; background: linear-gradient(135deg, #f8fbff, #eef6ff); }
      .stat-label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
      .stat-value { font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; }
      .stat-value.alert { color: #b91c1c; }
      .stat-value.info { color: #1d4ed8; }
      .stat-value.ok { color: #15803d; }
      .mono { font-family: Consolas, "Courier New", monospace; font-size: 0.85rem; }
      .chart-wrap { border: 1px solid #d9e3f0; border-radius: 12px; padding: 0.6rem; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); min-height: 280px; }
      .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
      .chart-title { font-size: 0.9rem; color: var(--muted); margin-bottom: 0.5rem; font-weight: 600; }
      .chart-head { display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; margin-bottom: 0.3rem; }
      .chart-export-btn { border: 1px solid #c7d2e3; background: #ffffff; color: #1d4ed8; border-radius: 8px; padding: 0.25rem 0.55rem; font-size: 0.76rem; font-weight: 600; cursor: pointer; }
      .chart-export-btn:hover { background: #eff6ff; border-color: #93c5fd; }
      .small { font-size: 0.84rem; }
      .table-scroll { overflow: auto; }
      .chart-wrap { height: 320px; overflow: hidden; }
      .chart-wrap canvas { width: 100% !important; height: calc(100% - 56px) !important; display: block; }
      td, th { overflow-wrap: anywhere; word-break: break-word; }
      @media print { .chart-export-btn { display: none !important; } }
      @media (max-width: 900px) { .chart-grid { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body class="{% if is_ms_biomarker %}ms-biomarker-wide{% endif %}">
    <main>
      <section>
        <h1>{{ report.report_title }}</h1>
        <p class="summary">{{ report.summary }}</p>
        {% for key, value in report.metadata.items() %}
          <span class="chip">{{ key }}: {{ value }}</span>
        {% endfor %}
      </section>

      {% for s in report.sections %}
      <section>
        <h2>{{ s.title }}</h2>
        {{ s.body_html | safe }}
      </section>
      {% endfor %}

      <section>
        <h2>Alerts</h2>
        {% if report.alerts %}
        <ul>
          {% for a in report.alerts %}
            <li><strong>{{ a.severity }}:</strong> {{ a.message }}</li>
          {% endfor %}
        </ul>
        {% else %}
        <p>No alerts.</p>
        {% endif %}
      </section>

      <section>
        <h2>Recommendations</h2>
        {% if report.recommendations %}
        <ul>
          {% for r in report.recommendations %}
            <li><strong>{{ r.priority }}:</strong> {{ r.action }}</li>
          {% endfor %}
        </ul>
        {% else %}
        <p>No recommendations.</p>
        {% endif %}
      </section>

      {% if is_network_queue %}
      <section>
        <h2>Queue Summary</h2>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-label">Interval Samples</div>
            <div class="stat-value">{{ network.summary.get('interval_samples', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Active Queues</div>
            <div class="stat-value">{{ network.summary.get('active_queues', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Overall Drop Ratio (Bytes)</div>
            <div class="stat-value {% if network.summary.get('overall_drop_ratio_bytes', 0) >= 0.2 %}alert{% endif %}">
              {{ "%.2f"|format(network.summary.get('overall_drop_ratio_bytes', 0) * 100) }}%
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Overall Drop Ratio (Frames)</div>
            <div class="stat-value {% if network.summary.get('overall_drop_ratio_frames', 0) >= 0.2 %}alert{% endif %}">
              {{ "%.2f"|format(network.summary.get('overall_drop_ratio_frames', 0) * 100) }}%
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Affected Intervals</div>
            <div class="stat-value {% if network.summary.get('affected_intervals_pct', 0) >= 40 %}alert{% endif %}">
              {{ "%.2f"|format(network.summary.get('affected_intervals_pct', 0)) }}%
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Risk Score</div>
            <div class="stat-value {% if network.summary.get('risk_score', 0) >= 50 %}alert{% endif %}">
              {{ network.summary.get('risk_score', 0) }} / 100
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Risk Band</div>
            <div class="stat-value {% if network.summary.get('risk_band', 'healthy') in ['degraded', 'critical'] %}alert{% endif %}">
              {{ network.summary.get('risk_band', 'healthy') }}
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2>L2/QoS Insights</h2>
        {% if network.l2_qos_insights %}
        <ul>
          {% for item in network.l2_qos_insights %}
            <li><strong>{{ item.confidence }}:</strong> {{ item.hypothesis }}</li>
          {% endfor %}
        </ul>
        {% else %}
        <p class="muted">No additional insights.</p>
        {% endif %}
      </section>

      <section>
        <h2>Top Congested Queues</h2>
        {% if network.top_queues %}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>NE</th>
                <th>Resource</th>
                <th>Queue</th>
                <th>Samples</th>
                <th>Mean Drop (Bytes)</th>
                <th>P95 Drop (Bytes)</th>
                <th>Dropped Bytes</th>
                <th>Dequeued Bytes</th>
              </tr>
            </thead>
            <tbody>
              {% for q in network.top_queues %}
              <tr>
                <td>{{ q.ne_name }}</td>
                <td>{{ q.resource_name }}</td>
                <td class="mono">{{ q.queue_block }}/{{ q.queue_number }}</td>
                <td>{{ q.samples }}</td>
                <td>{{ "%.2f"|format(q.mean_drop_ratio_bytes * 100) }}%</td>
                <td>{{ "%.2f"|format(q.p95_drop_ratio_bytes * 100) }}%</td>
                <td>{{ "{:,.0f}".format(q.dropped_bytes) }}</td>
                <td>{{ "{:,.0f}".format(q.dequeued_bytes) }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <p class="muted">No active queue intervals found.</p>
        {% endif %}
      </section>

      <section>
        <h2>Drop Ratio Trend</h2>
        <div class="chart-grid">
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Drop Ratio Over Time (%)</div>
              <button type="button" class="chart-export-btn" data-canvas-id="dropTrendChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="dropTrendChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Dropped Bytes Over Time</div>
              <button type="button" class="chart-export-btn" data-canvas-id="droppedBytesChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="droppedBytesChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Top Queues by Mean Drop Ratio (%)</div>
              <button type="button" class="chart-export-btn" data-canvas-id="topQueuesBarChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="topQueuesBarChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Share of Total Dropped Bytes</div>
              <button type="button" class="chart-export-btn" data-canvas-id="queueShareDonutChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="queueShareDonutChart"></canvas>
          </div>
        </div>
        <p class="small muted">Charts are interactive: hover to inspect values.</p>
      </section>

      <section>
        <h2>Hotspot Intervals</h2>
        {% if network.interval_hotspots %}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Drop Ratio %</th>
                <th>Dropped Bytes</th>
                <th>Dominant Queue</th>
                <th>Dominant Share %</th>
                <th>Likely Cause</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {% for row in network.interval_hotspots %}
              <tr>
                <td>{{ row.time }}</td>
                <td>{{ "%.2f"|format(row.drop_ratio_bytes * 100) }}</td>
                <td>{{ "{:,.0f}".format(row.dropped_bytes) }}</td>
                <td>{{ row.dominant_queue }}</td>
                <td>{{ "%.2f"|format(row.dominant_queue_share_pct) }}</td>
                <td>{{ row.likely_cause }}</td>
                <td>{{ row.confidence }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <p class="muted">No hotspot intervals found.</p>
        {% endif %}
      </section>
      {% endif %}

      {% if is_twamp %}
      <section>
        <h2>TWAMP Summary</h2>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-label">Samples</div>
            <div class="stat-value">{{ twamp.summary.get('samples', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Avg Discard Rate</div>
            <div class="stat-value {% if twamp.summary.get('avg_discard_rate_pct', 0) >= 0.15 %}alert{% endif %}">
              {{ "%.4f"|format(twamp.summary.get('avg_discard_rate_pct', 0)) }}%
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">P95 Discard Rate</div>
            <div class="stat-value {% if twamp.summary.get('p95_discard_rate_pct', 0) >= 0.15 %}alert{% endif %}">
              {{ "%.4f"|format(twamp.summary.get('p95_discard_rate_pct', 0)) }}%
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Avg Yellow Traffic</div>
            <div class="stat-value">{{ "%.4f"|format(twamp.summary.get('avg_yellow_traffic_pct', 0)) }}%</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Risk Score</div>
            <div class="stat-value {% if twamp.summary.get('risk_score', 0) >= 50 %}alert{% endif %}">
              {{ twamp.summary.get('risk_score', 0) }} / 100
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Risk Band</div>
            <div class="stat-value {% if twamp.summary.get('risk_band', 'healthy') in ['degraded', 'critical'] %}alert{% endif %}">
              {{ twamp.summary.get('risk_band', 'healthy') }}
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2>L2/QoS Insights</h2>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-label">Green Discard Share</div>
            <div class="stat-value">{{ "%.2f"|format(twamp.color_packet_share_pct.get('green_pct', 0)) }}%</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Yellow Discard Share</div>
            <div class="stat-value">{{ "%.2f"|format(twamp.color_packet_share_pct.get('yellow_pct', 0)) }}%</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Red Discard Share</div>
            <div class="stat-value">{{ "%.2f"|format(twamp.color_packet_share_pct.get('red_pct', 0)) }}%</div>
          </div>
        </div>
        {% if twamp.l2_qos_insights %}
        <ul>
          {% for item in twamp.l2_qos_insights %}
            <li><strong>{{ item.confidence }}:</strong> {{ item.hypothesis }}</li>
          {% endfor %}
        </ul>
        {% else %}
        <p class="muted">No additional L2/QoS insights.</p>
        {% endif %}
      </section>

      <section>
        <h2>Top Discard Hotspots</h2>
        {% if twamp.top_discard_intervals %}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Time (UTC)</th>
                <th>Discard Rate %</th>
                <th>Yellow Traffic %</th>
                <th>Discard Packets</th>
                <th>Forward Packets</th>
                <th>Delay Avg</th>
                <th>IPDV Max</th>
                <th>Likely Cause</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {% for row in twamp.top_discard_intervals %}
              <tr>
                <td>{{ row.time_utc }}</td>
                <td>{{ "%.4f"|format(row.discard_rate_pct) }}</td>
                <td>{{ "%.4f"|format(row.yellow_traffic_pct) }}</td>
                <td>{{ "{:,.0f}".format(row.discard_packets) }}</td>
                <td>{{ "{:,.0f}".format(row.forward_packets) }}</td>
                <td>{{ "%.4f"|format(row.delay_average) }}</td>
                <td>{{ "%.4f"|format(row.ipdv_max) }}</td>
                <td>{{ row.likely_cause }}</td>
                <td>{{ row.confidence }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <p class="muted">No interval hotspots found.</p>
        {% endif %}
      </section>

      <section>
        <h2>TWAMP Health Charts</h2>
        <div class="chart-grid">
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Discard Rate Over Time (%)</div>
              <button type="button" class="chart-export-btn" data-canvas-id="twampDiscardTrendChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="twampDiscardTrendChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Yellow Traffic Over Time (%)</div>
              <button type="button" class="chart-export-btn" data-canvas-id="twampYellowTrendChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="twampYellowTrendChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Discard Packets Over Time</div>
              <button type="button" class="chart-export-btn" data-canvas-id="twampDiscardPacketsChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="twampDiscardPacketsChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Discard Rate vs Yellow Traffic</div>
              <button type="button" class="chart-export-btn" data-canvas-id="twampScatterChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="twampScatterChart"></canvas>
          </div>
        </div>
      </section>
      {% endif %}

      {% if is_pm %}
      <section>
        <h2>PM Export Summary</h2>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-label">Parsed Records</div>
            <div class="stat-value">{{ pm.summary.get('records', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Table Types</div>
            <div class="stat-value">{{ pm.summary.get('table_types', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Discard Delta Total</div>
            <div class="stat-value {% if pm.summary.get('discard_delta_total', 0) > 0 %}alert{% endif %}">
              {{ "{:,.0f}".format(pm.summary.get('discard_delta_total', 0)) }}
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">CRC Delta Total</div>
            <div class="stat-value {% if pm.summary.get('crc_delta_total', 0) > 0 %}alert{% endif %}">
              {{ "{:,.0f}".format(pm.summary.get('crc_delta_total', 0)) }}
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">CPU Max</div>
            <div class="stat-value {% if pm.summary.get('cpu_max_pct', 0) >= 80 %}alert{% endif %}">
              {{ "%.2f"|format(pm.summary.get('cpu_max_pct', 0)) }}%
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Memory Max Used</div>
            <div class="stat-value {% if pm.summary.get('memory_max_used_pct', 0) >= 85 %}alert{% endif %}">
              {{ "%.2f"|format(pm.summary.get('memory_max_used_pct', 0)) }}%
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Disk Max Used</div>
            <div class="stat-value {% if pm.summary.get('disk_max_used_pct', 0) >= 85 %}alert{% endif %}">
              {{ "%.2f"|format(pm.summary.get('disk_max_used_pct', 0)) }}%
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Risk Score</div>
            <div class="stat-value {% if pm.summary.get('risk_score', 0) >= 50 %}alert{% endif %}">
              {{ pm.summary.get('risk_score', 0) }} / 100
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2>L2/System Insights</h2>
        {% if pm.l2_system_hypotheses %}
        <ul>
          {% for item in pm.l2_system_hypotheses %}
            <li><strong>{{ item.confidence }}:</strong> {{ item.hypothesis }}</li>
          {% endfor %}
        </ul>
        {% else %}
        <p class="muted">No additional L2/system insights.</p>
        {% endif %}
      </section>

      <section>
        <h2>Top Interfaces by Discard Delta</h2>
        {% if pm.top_interface_discards %}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>ifIndex</th>
                <th>Discard Delta</th>
                <th>Error Delta</th>
              </tr>
            </thead>
            <tbody>
              {% for row in pm.top_interface_discards %}
              <tr>
                <td class="mono">{{ row.ifIndex }}</td>
                <td>{{ "{:,.0f}".format(row.discard_delta) }}</td>
                <td>{{ "{:,.0f}".format(row.error_delta) }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <p class="muted">No interface discard deltas available.</p>
        {% endif %}
      </section>

      <section>
        <h2>PM Health Charts</h2>
        <div class="chart-grid">
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">
                Top Interface {{ "Discard Counter" if pm.summary.get('interface_signal_basis') == 'absolute_counter' else "Discard Delta" }}
              </div>
              <button type="button" class="chart-export-btn" data-canvas-id="pmInterfaceDiscardChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="pmInterfaceDiscardChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Top Interface {{ pm.summary.get('interface_error_signal_label', 'Error Delta') }}</div>
              <button type="button" class="chart-export-btn" data-canvas-id="pmInterfaceErrorChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="pmInterfaceErrorChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Parsed Rows by Table Type</div>
              <button type="button" class="chart-export-btn" data-canvas-id="pmTableCountsChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="pmTableCountsChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Resource Pressure Snapshot (%)</div>
              <button type="button" class="chart-export-btn" data-canvas-id="pmResourcePressureChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="pmResourcePressureChart"></canvas>
          </div>
        </div>
      </section>
      {% endif %}

      {% if is_jira %}
      <section>
        <h2>Jira Portfolio Summary</h2>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-label">Total Issues</div>
            <div class="stat-value">{{ jira.summary.get('total_issues', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Projects</div>
            <div class="stat-value">{{ jira.summary.get('projects', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Open</div>
            <div class="stat-value {% if jira.summary.get('open_issues', 0) > jira.summary.get('closed_issues', 0) %}alert{% endif %}">
              {{ jira.summary.get('open_issues', 0) }}
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">In Progress</div>
            <div class="stat-value info">{{ jira.summary.get('in_progress_issues', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Closed</div>
            <div class="stat-value ok">{{ jira.summary.get('closed_issues', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Closure Ratio</div>
            <div class="stat-value {% if jira.summary.get('closure_ratio', 0) < 0.1 %}alert{% endif %}">
              {{ "%.2f"|format(jira.summary.get('closure_ratio', 0) * 100) }}%
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2>Observed facts (space-level)</h2>
        <p class="muted">Opened / in-progress / closed by project with responsibility owners.</p>
        {% if jira.project_status_breakdown %}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Project</th>
                <th>Lead</th>
                <th>Top Active Assignee</th>
                <th>Open</th>
                <th>In Progress</th>
                <th>Closed</th>
                <th>Total</th>
                <th>Closure %</th>
              </tr>
            </thead>
            <tbody>
              {% for row in jira.project_status_breakdown[:12] %}
              <tr>
                <td class="mono">{{ row.project_key }}</td>
                <td>{{ row.project_lead }}</td>
                <td>{{ row.top_active_assignee }} ({{ row.top_active_assignee_issues }})</td>
                <td>{{ row.open_issues }}</td>
                <td>{{ row.in_progress_issues }}</td>
                <td>{{ row.closed_issues }}</td>
                <td>{{ row.total_issues }}</td>
                <td>{{ "%.2f"|format(row.closure_ratio * 100) }}%</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <p class="muted">No project breakdown available.</p>
        {% endif %}
      </section>

      <section>
        <h2>Responsibility and Stale Work</h2>
        <div class="chart-grid">
          <div class="chart-wrap">
            <div class="chart-title">Top Active Assignees (Who Owns Backlog)</div>
            {% if jira.responsible_workload %}
            <div class="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Assignee</th>
                    <th>Active Issues</th>
                  </tr>
                </thead>
                <tbody>
                  {% for row in jira.responsible_workload %}
                  <tr>
                    <td>{{ row.assignee }}</td>
                    <td>{{ row.active_issues }}</td>
                  </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
            {% else %}
            <p class="muted">No active assignee workload data.</p>
            {% endif %}
          </div>
          <div class="chart-wrap">
            <div class="chart-title">Oldest Active Issues (Stale Queue)</div>
            {% if jira.oldest_open_issues %}
            <div class="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Issue</th>
                    <th>Project</th>
                    <th>Status</th>
                    <th>Assignee</th>
                    <th>Age (days)</th>
                  </tr>
                </thead>
                <tbody>
                  {% for row in jira.oldest_open_issues[:10] %}
                  <tr>
                    <td class="mono">{{ row.issue_key }}</td>
                    <td class="mono">{{ row.project_key }}</td>
                    <td>{{ row.status }}</td>
                    <td>{{ row.assignee }}</td>
                    <td>{{ row.age_days }}</td>
                  </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
            {% else %}
            <p class="muted">No stale issue data.</p>
            {% endif %}
          </div>
        </div>
      </section>

      <section>
        <h2>Jira Status Charts</h2>
        <div class="chart-grid">
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">All Status Buckets</div>
              <button type="button" class="chart-export-btn" data-canvas-id="jiraStatusChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="jiraStatusChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Projects: Open / In Progress / Closed</div>
              <button type="button" class="chart-export-btn" data-canvas-id="jiraProjectStatusChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="jiraProjectStatusChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Assignees with Highest Active Backlog</div>
              <button type="button" class="chart-export-btn" data-canvas-id="jiraAssigneeBacklogChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="jiraAssigneeBacklogChart"></canvas>
          </div>
        </div>
      </section>
      {% endif %}

      {% if is_ms_biomarker %}
      <section>
        <h2>MS Registry Summary</h2>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-label">Rows</div>
            <div class="stat-value">{{ ms.summary.get('rows', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Unique Patients</div>
            <div class="stat-value">{{ ms.summary.get('unique_patients', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Biomarker Columns Used</div>
            <div class="stat-value">{{ ms.summary.get('biomarker_columns_used', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">High Sparse Biomarkers</div>
            <div class="stat-value {% if ms.summary.get('high_sparse_biomarkers', 0) > 0 %}alert{% endif %}">
              {{ ms.summary.get('high_sparse_biomarkers', 0) }}
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">EDSS Comparable Pairs</div>
            <div class="stat-value">{{ ms.summary.get('edss_pairs', 0) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">EDSS Worsened Ratio</div>
            <div class="stat-value {% if ms.summary.get('edss_worsened_ratio', 0) >= 0.4 %}alert{% endif %}">
              {{ "%.2f"|format(ms.summary.get('edss_worsened_ratio', 0) * 100) }}%
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Strong EDSS Contributors</div>
            <div class="stat-value {% if ms.summary.get('strong_contributors', 0) == 0 %}alert{% endif %}">
              {{ ms.summary.get('strong_contributors', 0) }}
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Weak/Non-contributing</div>
            <div class="stat-value info">
              {{ ms.summary.get('weak_contributors', 0) }}
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2>Clinical Progression Snapshot</h2>
        <div class="chart-grid">
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">EDSS Progression (Improved / Stable / Worsened)</div>
              <button type="button" class="chart-export-btn" data-canvas-id="msEdssProgressionChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="msEdssProgressionChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Course Distribution (Sample vs Last)</div>
              <button type="button" class="chart-export-btn" data-canvas-id="msCourseDistributionChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="msCourseDistributionChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Biomarker vs Sample EDSS Correlation</div>
              <button type="button" class="chart-export-btn" data-canvas-id="msBiomarkerEdssCorrChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="msBiomarkerEdssCorrChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Biomarker-to-Biomarker Correlation</div>
              <button type="button" class="chart-export-btn" data-canvas-id="msBiomarkerPairCorrChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="msBiomarkerPairCorrChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">Top Sparse Biomarkers (Missing %)</div>
              <button type="button" class="chart-export-btn" data-canvas-id="msSparseBiomarkersChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="msSparseBiomarkersChart"></canvas>
          </div>
          <div class="chart-wrap">
            <div class="chart-head">
              <div class="chart-title">SID Distribution</div>
              <button type="button" class="chart-export-btn" data-canvas-id="msSidDistributionChart" data-file-prefix="{{ report.metadata.get('report_id', report.report_type_id) }}">Export PNG</button>
            </div>
            <canvas id="msSidDistributionChart"></canvas>
          </div>
        </div>
      </section>

      <section>
        <h2>Data Completeness and Follow-up</h2>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Field</th>
                <th>Missing %</th>
                <th>Non-null Rows</th>
              </tr>
            </thead>
            <tbody>
              {% for row in ms.key_missingness %}
              <tr>
                <td>{{ row.column }}</td>
                <td>{{ "%.2f"|format(row.missing_pct * 100) }}</td>
                <td>{{ row.non_null }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        <p class="muted small">
          Follow-up pairs: {{ ms.followup_summary.get('pairs', 0) }},
          median: {{ ms.followup_summary.get('median_days', 0) }} days,
          p90: {{ ms.followup_summary.get('p90_days', 0) }} days,
          invalid date pairs: {{ ms.followup_summary.get('invalid_pairs', 0) }}.
        </p>
      </section>

      <section>
        <h2>Biomarkers Related to Sample EDSS</h2>
        <p class="muted small">Correlation method: {{ ms.summary.get('correlation_method', 'Spearman rank correlation (Pearson computed on ranked values)') }}.</p>
        {% if ms.biomarker_sample_edss_corr %}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Biomarker</th>
                <th>Spearman Corr (Sample EDSS)</th>
                <th>Abs Corr</th>
                <th>Direction</th>
                <th>Paired Rows</th>
              </tr>
            </thead>
            <tbody>
              {% for row in ms.biomarker_sample_edss_corr %}
              <tr>
                <td>{{ row.biomarker }}</td>
                <td>{{ "%.4f"|format(row.corr) }}</td>
                <td>{{ "%.4f"|format(row.abs_corr) }}</td>
                <td>{{ row.direction }}</td>
                <td>{{ row.paired_rows }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <p class="muted">Not enough paired data to estimate biomarker-to-Sample-EDSS correlations.</p>
        {% endif %}
      </section>

      <section>
        <h2>Biomarker Correlation by Sample EDSS Bands (0-3.5 vs 4+)</h2>
        <p class="muted small">Correlation method: {{ ms.summary.get('correlation_method', 'Spearman rank correlation (Pearson computed on ranked values)') }}.</p>
        {% if ms.biomarker_sample_edss_group_corr %}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Biomarker</th>
                <th>Corr (0-3.5)</th>
                <th>Rows (0-3.5)</th>
                <th>Corr (4+)</th>
                <th>Rows (4+)</th>
                <th>|Delta|</th>
              </tr>
            </thead>
            <tbody>
              {% for row in ms.biomarker_sample_edss_group_corr %}
              <tr>
                <td>{{ row.biomarker }}</td>
                <td>{% if row.low_corr is not none %}{{ "%.4f"|format(row.low_corr) }}{% else %}-{% endif %}</td>
                <td>{{ row.low_paired_rows }}</td>
                <td>{% if row.high_corr is not none %}{{ "%.4f"|format(row.high_corr) }}{% else %}-{% endif %}</td>
                <td>{{ row.high_paired_rows }}</td>
                <td>{{ "%.4f"|format(row.delta_abs) }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <p class="muted">Not enough grouped paired data to estimate banded Sample-EDSS biomarker correlations.</p>
        {% endif %}
      </section>

      <section>
        <h2>Biomarker Correlation by Last EDSS Bands (0-3.5 vs 4+)</h2>
        <p class="muted small">Correlation method: {{ ms.summary.get('correlation_method', 'Spearman rank correlation (Pearson computed on ranked values)') }}.</p>
        {% if ms.biomarker_last_edss_group_corr %}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Biomarker</th>
                <th>Corr (0-3.5)</th>
                <th>Rows (0-3.5)</th>
                <th>Corr (4+)</th>
                <th>Rows (4+)</th>
                <th>|Delta|</th>
              </tr>
            </thead>
            <tbody>
              {% for row in ms.biomarker_last_edss_group_corr %}
              <tr>
                <td>{{ row.biomarker }}</td>
                <td>{% if row.low_corr is not none %}{{ "%.4f"|format(row.low_corr) }}{% else %}-{% endif %}</td>
                <td>{{ row.low_paired_rows }}</td>
                <td>{% if row.high_corr is not none %}{{ "%.4f"|format(row.high_corr) }}{% else %}-{% endif %}</td>
                <td>{{ row.high_paired_rows }}</td>
                <td>{{ "%.4f"|format(row.delta_abs) }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <p class="muted">Not enough grouped paired data to estimate banded Last-EDSS biomarker correlations.</p>
        {% endif %}
      </section>

      <section>
        <h2>Biomarker Inter-correlation</h2>
        <p class="muted small">Correlation method: {{ ms.summary.get('correlation_method', 'Spearman rank correlation (Pearson computed on ranked values)') }}.</p>
        {% if ms.biomarker_pair_corr %}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Left Biomarker</th>
                <th>Right Biomarker</th>
                <th>Spearman Corr</th>
                <th>Abs Corr</th>
                <th>Paired Rows</th>
              </tr>
            </thead>
            <tbody>
              {% for row in ms.biomarker_pair_corr %}
              <tr>
                <td>{{ row.left_biomarker }}</td>
                <td>{{ row.right_biomarker }}</td>
                <td>{{ "%.4f"|format(row.corr) }}</td>
                <td>{{ "%.4f"|format(row.abs_corr) }}</td>
                <td>{{ row.paired_rows }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <p class="muted">Not enough paired coverage for robust biomarker-to-biomarker correlation estimation.</p>
        {% endif %}
        <h3 style="margin-top:0.9rem;">Correlation Matrix (Overall)</h3>
        <div class="table-scroll" id="msCorrMatrixTable"></div>
        <h3 style="margin-top:0.9rem;">Correlation Matrix (Sample EDSS 0-3.5)</h3>
        <div class="table-scroll" id="msCorrMatrixSampleLowTable"></div>
        <h3 style="margin-top:0.9rem;">Correlation Matrix (Sample EDSS 4+)</h3>
        <div class="table-scroll" id="msCorrMatrixSampleHighTable"></div>
        <h3 style="margin-top:0.9rem;">Correlation Matrix (Last EDSS 0-3.5)</h3>
        <div class="table-scroll" id="msCorrMatrixLastLowTable"></div>
        <h3 style="margin-top:0.9rem;">Correlation Matrix (Last EDSS 4+)</h3>
        <div class="table-scroll" id="msCorrMatrixLastHighTable"></div>
      </section>
      {% endif %}

      <section>
        <h2>Tables</h2>
        {% if is_network_queue or is_twamp or is_pm or is_jira or is_ms_biomarker %}
          <p class="muted">Detailed metric payload is available in the JSON artifact for this report.</p>
        {% else %}
        {% for t in report.tables %}
          <h3>{{ t.name }}</h3>
          {% if t.rows and t.rows[0] is mapping %}
            <table>
              <thead>
                <tr>
                  {% for key in t.rows[0].keys() %}
                  <th>{{ key }}</th>
                  {% endfor %}
                </tr>
              </thead>
              <tbody>
                {% for row in t.rows %}
                <tr>
                  {% for v in row.values() %}
                  <td>{{ v }}</td>
                  {% endfor %}
                </tr>
                {% endfor %}
              </tbody>
            </table>
          {% else %}
            <pre>{{ t.rows }}</pre>
          {% endif %}
        {% endfor %}
        {% endif %}
      </section>
    </main>
    {% if is_network_queue %}
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script>
      (function() {
        const points = {{ network.time_trend_json | safe }};
        const queues = {{ network.top_queues_json | safe }};
        if (!window.Chart) return;
        Chart.defaults.font.family = "'Segoe UI', Tahoma, sans-serif";
        Chart.defaults.font.size = 13;
        Chart.defaults.color = "#334155";

        const downsample = (arr, maxPoints) => {
          if (!Array.isArray(arr) || arr.length <= maxPoints) return arr || [];
          const step = Math.ceil(arr.length / maxPoints);
          return arr.filter((_, idx) => idx % step === 0);
        };
        const trend = downsample(points, 120);

        const labels = Array.isArray(trend) ? trend.map((p) => p.time) : [];
        const dropRatioValues = Array.isArray(trend) ? trend.map((p) => Number((p.drop_ratio_bytes * 100).toFixed(2))) : [];
        const droppedBytesValues = Array.isArray(trend) ? trend.map((p) => Number(p.dropped_bytes || 0)) : [];
        const truncate = (txt, maxLen = 28) => (txt && txt.length > maxLen ? txt.slice(0, maxLen - 1) + "..." : txt);
        const maxTicks = labels.length > 120 ? 6 : labels.length > 60 ? 8 : 12;
        const formatTimeLabel = (raw) => {
          const asText = String(raw || "");
          if (asText.length >= 16) return asText.slice(11, 16);
          return asText;
        };
        const staggerTick = (text, index) => (index % 2 === 0 ? [String(text || ""), ""] : ["", String(text || "")]);
        const topQueueLabels = Array.isArray(queues)
          ? queues.slice(0, 10).map((q) => truncate(`${q.resource_name} ${q.queue_block}/${q.queue_number}`))
          : [];
        const topQueueDropValues = Array.isArray(queues)
          ? queues.slice(0, 10).map((q) => Number(((q.mean_drop_ratio_bytes || 0) * 100).toFixed(2)))
          : [];
        const queueShareLabels = Array.isArray(queues)
          ? queues.slice(0, 8).map((q) => `${q.resource_name} ${q.queue_block}/${q.queue_number}`)
          : [];
        const queueShareValues = Array.isArray(queues)
          ? queues.slice(0, 8).map((q) => Number(q.dropped_bytes || 0))
          : [];

        const lineGradient = (ctx) => {
          const g = ctx.createLinearGradient(0, 0, 0, 240);
          g.addColorStop(0, "rgba(37, 99, 235, 0.35)");
          g.addColorStop(1, "rgba(37, 99, 235, 0.02)");
          return g;
        };

        const createChart = (id, cfg) => {
          const el = document.getElementById(id);
          if (!el) return;
          new Chart(el, cfg);
        };

        createChart("dropTrendChart", {
          type: "line",
          data: {
            labels: labels,
            datasets: [{
              label: "Drop Ratio %",
              data: dropRatioValues,
              borderColor: "#1d4ed8",
              backgroundColor: (ctx) => lineGradient(ctx.chart.ctx),
              borderWidth: 2.4,
              pointRadius: 0,
              pointHoverRadius: 3,
              tension: 0.32,
              fill: true
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                title: { display: true, text: "Drop Ratio (%)", color: "#334155", font: { weight: "600" } },
                beginAtZero: true,
                grid: { color: "rgba(148,163,184,0.18)" }
              },
              x: {
                title: { display: true, text: "Time (UTC)", color: "#334155", font: { weight: "600" } },
                ticks: { maxTicksLimit: maxTicks, callback: (value, index) => staggerTick(formatTimeLabel(labels[index]), index) },
                grid: { display: false }
              }
            },
            plugins: {
              legend: { display: true },
              tooltip: { mode: "index", intersect: false }
            }
          }
        });

        createChart("droppedBytesChart", {
          type: "line",
          data: {
            labels: labels,
            datasets: [{
              label: "Dropped Bytes",
              data: droppedBytesValues,
              borderColor: "#b45309",
              backgroundColor: "rgba(245, 158, 11, 0.14)",
              borderWidth: 2,
              pointRadius: 0,
              pointHoverRadius: 3,
              tension: 0.28,
              fill: true
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                title: { display: true, text: "Dropped Bytes", color: "#334155", font: { weight: "600" } },
                grid: { color: "rgba(148,163,184,0.18)" }
              },
              x: {
                title: { display: true, text: "Time (UTC)", color: "#334155", font: { weight: "600" } },
                ticks: { maxTicksLimit: maxTicks, callback: (value, index) => staggerTick(formatTimeLabel(labels[index]), index) },
                grid: { display: false }
              }
            },
            plugins: { legend: { display: true } }
          }
        });

        createChart("topQueuesBarChart", {
          type: "bar",
          data: {
            labels: topQueueLabels,
            datasets: [{
              label: "Mean Drop %",
              data: topQueueDropValues,
              borderRadius: 8,
              backgroundColor: [
                "#1d4ed8", "#0f766e", "#0369a1", "#7c3aed", "#b45309",
                "#be123c", "#0891b2", "#5b21b6", "#15803d", "#334155"
              ]
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
              x: {
                title: { display: true, text: "Mean Drop Ratio (%)", color: "#334155", font: { weight: "600" } },
                beginAtZero: true,
                grid: { color: "rgba(148,163,184,0.18)" }
              },
              y: {
                title: { display: true, text: "Queue", color: "#334155", font: { weight: "600" } },
                grid: { display: false },
                ticks: { autoSkip: false }
              }
            },
            plugins: { legend: { display: true } }
          }
        });

        createChart("queueShareDonutChart", {
          type: "doughnut",
          data: {
            labels: queueShareLabels,
            datasets: [{
              label: "Dropped Bytes Share",
              data: queueShareValues,
              backgroundColor: [
                "#1d4ed8", "#0f766e", "#0369a1", "#7c3aed",
                "#b45309", "#be123c", "#0891b2", "#334155"
              ],
              borderColor: "#ffffff",
              borderWidth: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: "bottom" }
            },
            cutout: "62%"
          }
        });

        document.querySelectorAll(".chart-export-btn").forEach((btn) => {
          btn.addEventListener("click", () => {
            const canvasId = btn.getAttribute("data-canvas-id");
            const prefix = btn.getAttribute("data-file-prefix") || "report";
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const a = document.createElement("a");
            a.href = canvas.toDataURL("image/png");
            a.download = `${prefix}.${canvasId}.png`;
            document.body.appendChild(a);
            a.click();
            a.remove();
          });
        });
      })();
    </script>
    {% endif %}
    {% if is_twamp %}
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script>
      (function() {
        if (!window.Chart) return;
        Chart.defaults.font.family = "'Segoe UI', Tahoma, sans-serif";
        Chart.defaults.font.size = 13;
        Chart.defaults.color = "#334155";
        const points = {{ twamp.time_trend_json | safe }};
        if (!Array.isArray(points) || points.length === 0) return;

        const downsample = (arr, maxPoints) => {
          if (!Array.isArray(arr) || arr.length <= maxPoints) return arr || [];
          const step = Math.ceil(arr.length / maxPoints);
          return arr.filter((_, idx) => idx % step === 0);
        };
        const trend = downsample(points, 120);

        const labels = trend.map((p) => p.time_utc);
        const discardRate = trend.map((p) => Number(p.discard_rate_pct || 0));
        const yellowPct = trend.map((p) => Number(p.yellow_traffic_pct || 0));
        const discardPackets = trend.map((p) => Number(p.discard_packets || 0));
        const scatterPoints = trend.map((p) => ({ x: Number(p.yellow_traffic_pct || 0), y: Number(p.discard_rate_pct || 0) }));
        const maxTicks = labels.length > 120 ? 6 : labels.length > 60 ? 8 : 12;
        const formatTimeLabel = (raw) => {
          const asText = String(raw || "");
          if (asText.length >= 16) return asText.slice(11, 16);
          return asText;
        };
        const staggerTick = (text, index) => (index % 2 === 0 ? [String(text || ""), ""] : ["", String(text || "")]);

        const createChart = (id, cfg) => {
          const el = document.getElementById(id);
          if (!el) return;
          new Chart(el, cfg);
        };

        createChart("twampDiscardTrendChart", {
          type: "line",
          data: {
            labels,
            datasets: [{
              label: "Discard Rate %",
              data: discardRate,
              borderColor: "#dc2626",
              backgroundColor: "rgba(220, 38, 38, 0.12)",
              borderWidth: 2.2,
              pointRadius: 0,
              pointHoverRadius: 3,
              tension: 0.28,
              fill: true,
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                title: { display: true, text: "Discard Rate (%)", color: "#334155", font: { weight: "600" } },
                beginAtZero: true,
                grid: { color: "rgba(148,163,184,0.18)" }
              },
              x: {
                title: { display: true, text: "Time (UTC)", color: "#334155", font: { weight: "600" } },
                ticks: { maxTicksLimit: maxTicks, callback: (value, index) => staggerTick(formatTimeLabel(labels[index]), index) },
                grid: { display: false }
              }
            },
            plugins: { legend: { display: true } }
          }
        });

        createChart("twampYellowTrendChart", {
          type: "line",
          data: {
            labels,
            datasets: [{
              label: "Yellow Traffic %",
              data: yellowPct,
              borderColor: "#ca8a04",
              backgroundColor: "rgba(202, 138, 4, 0.14)",
              borderWidth: 2.2,
              pointRadius: 0,
              pointHoverRadius: 3,
              tension: 0.3,
              fill: true,
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                title: { display: true, text: "Yellow Traffic (%)", color: "#334155", font: { weight: "600" } },
                beginAtZero: true,
                grid: { color: "rgba(148,163,184,0.18)" }
              },
              x: {
                title: { display: true, text: "Time (UTC)", color: "#334155", font: { weight: "600" } },
                ticks: { maxTicksLimit: maxTicks, callback: (value, index) => staggerTick(formatTimeLabel(labels[index]), index) },
                grid: { display: false }
              }
            },
            plugins: { legend: { display: true } }
          }
        });

        createChart("twampDiscardPacketsChart", {
          type: "bar",
          data: {
            labels,
            datasets: [{
              label: "Discard Packets",
              data: discardPackets,
              backgroundColor: "#0f766e",
              borderRadius: 6,
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                title: { display: true, text: "Discard Packets", color: "#334155", font: { weight: "600" } },
                beginAtZero: true,
                grid: { color: "rgba(148,163,184,0.18)" }
              },
              x: {
                title: { display: true, text: "Time (UTC)", color: "#334155", font: { weight: "600" } },
                ticks: { maxTicksLimit: maxTicks, callback: (value, index) => staggerTick(formatTimeLabel(labels[index]), index) },
                grid: { display: false }
              }
            },
            plugins: { legend: { display: true } }
          }
        });

        createChart("twampScatterChart", {
          type: "scatter",
          data: {
            datasets: [{
              label: "Intervals",
              data: scatterPoints,
              pointRadius: 4,
              pointBackgroundColor: "#1d4ed8",
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                title: { display: true, text: "Discard Rate (%)", color: "#334155", font: { weight: "600" } },
                beginAtZero: true,
                grid: { color: "rgba(148,163,184,0.18)" }
              },
              x: {
                title: { display: true, text: "Yellow Traffic (%)", color: "#334155", font: { weight: "600" } },
                beginAtZero: true,
                grid: { color: "rgba(148,163,184,0.18)" }
              }
            },
            plugins: { legend: { display: true } }
          }
        });

        document.querySelectorAll(".chart-export-btn").forEach((btn) => {
          btn.addEventListener("click", () => {
            const canvasId = btn.getAttribute("data-canvas-id");
            const prefix = btn.getAttribute("data-file-prefix") || "report";
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const a = document.createElement("a");
            a.href = canvas.toDataURL("image/png");
            a.download = `${prefix}.${canvasId}.png`;
            document.body.appendChild(a);
            a.click();
            a.remove();
          });
        });
      })();
    </script>
    {% endif %}
    {% if is_pm %}
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script>
      (function() {
        if (!window.Chart) return;
        Chart.defaults.font.family = "'Segoe UI', Tahoma, sans-serif";
        Chart.defaults.font.size = 13;
        Chart.defaults.color = "#334155";

        const topInterfaces = {{ pm.top_interface_discards_json | safe }};
        const tableCounts = {{ pm.table_counts_json | safe }};
        const summary = {{ pm.summary_json | safe }};

        const ifLabels = Array.isArray(topInterfaces) ? topInterfaces.map((row) => `ifIndex ${row.ifIndex}`) : [];
        const ifDiscard = Array.isArray(topInterfaces) ? topInterfaces.map((row) => Number(row.discard_delta || 0)) : [];
        const ifError = Array.isArray(topInterfaces) ? topInterfaces.map((row) => Number(row.error_chart_value || row.error_delta || 0)) : [];
        const discardLabel = summary.interface_signal_basis === "absolute_counter" ? "Discard Counter" : "Discard Delta";
        const errorSignalLabel = String(summary.interface_error_signal_label || "Error Delta");
        const tableLabels = Array.isArray(tableCounts) ? tableCounts.map((row) => row.name) : [];
        const tableValues = Array.isArray(tableCounts) ? tableCounts.map((row) => Number(row.count || 0)) : [];

        const createChart = (id, cfg) => {
          const el = document.getElementById(id);
          if (!el) return;
          new Chart(el, cfg);
        };

        createChart("pmInterfaceDiscardChart", {
          type: "bar",
          data: {
            labels: ifLabels,
            datasets: [{
              label: discardLabel,
              data: ifDiscard,
              borderRadius: 8,
              backgroundColor: "#dc2626",
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
              x: {
                title: { display: true, text: discardLabel, color: "#334155", font: { weight: "600" } },
                beginAtZero: true,
                grid: { color: "rgba(148,163,184,0.18)" }
              },
              y: {
                title: { display: true, text: "Interface", color: "#334155", font: { weight: "600" } },
                grid: { display: false }
              }
            },
            plugins: { legend: { display: true } }
          }
        });

        createChart("pmInterfaceErrorChart", {
          type: "bar",
          data: {
            labels: ifLabels,
            datasets: [{
              label: errorSignalLabel,
              data: ifError,
              borderRadius: 8,
              backgroundColor: "#0f766e",
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
              x: {
                title: { display: true, text: errorSignalLabel, color: "#334155", font: { weight: "600" } },
                beginAtZero: true,
                grid: { color: "rgba(148,163,184,0.18)" }
              },
              y: {
                title: { display: true, text: "Interface", color: "#334155", font: { weight: "600" } },
                grid: { display: false }
              }
            },
            plugins: { legend: { display: true } }
          }
        });

        createChart("pmTableCountsChart", {
          type: "doughnut",
          data: {
            labels: tableLabels,
            datasets: [{
              label: "Rows",
              data: tableValues,
              borderColor: "#ffffff",
              borderWidth: 2,
              backgroundColor: [
                "#1d4ed8", "#0f766e", "#0369a1", "#7c3aed", "#b45309",
                "#be123c", "#0891b2", "#334155", "#15803d", "#ea580c"
              ],
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: "bottom" } },
            cutout: "58%"
          }
        });

        createChart("pmResourcePressureChart", {
          type: "radar",
          data: {
            labels: ["CPU max %", "Memory max %", "Disk max %", "Risk score"],
            datasets: [{
              label: "Pressure",
              data: [
                Number(summary.cpu_max_pct || 0),
                Number(summary.memory_max_used_pct || 0),
                Number(summary.disk_max_used_pct || 0),
                Number(summary.risk_score || 0),
              ],
              borderColor: "#1d4ed8",
              backgroundColor: "rgba(29, 78, 216, 0.16)",
              pointBackgroundColor: "#1d4ed8",
              borderWidth: 2,
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              r: {
                suggestedMin: 0,
                suggestedMax: 100,
                angleLines: { color: "rgba(148,163,184,0.25)" },
                grid: { color: "rgba(148,163,184,0.2)" },
                pointLabels: { color: "#334155", font: { size: 12, weight: "600" } }
              }
            },
            plugins: { legend: { display: true } }
          }
        });

        document.querySelectorAll(".chart-export-btn").forEach((btn) => {
          btn.addEventListener("click", () => {
            const canvasId = btn.getAttribute("data-canvas-id");
            const prefix = btn.getAttribute("data-file-prefix") || "report";
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const a = document.createElement("a");
            a.href = canvas.toDataURL("image/png");
            a.download = `${prefix}.${canvasId}.png`;
            document.body.appendChild(a);
            a.click();
            a.remove();
          });
        });
      })();
    </script>
    {% endif %}
    {% if is_jira %}
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script>
      (function() {
        if (!window.Chart) return;
        Chart.defaults.font.family = "'Segoe UI', Tahoma, sans-serif";
        Chart.defaults.font.size = 13;
        Chart.defaults.color = "#334155";

        const summary = {{ jira.summary_json | safe }};
        const projects = {{ jira.project_status_breakdown_json | safe }};
        const assignees = {{ jira.responsible_workload_json | safe }};
        const staggerTick = (text, index) => (index % 2 === 0 ? [String(text || ""), ""] : ["", String(text || "")]);

        const createChart = (id, cfg) => {
          const el = document.getElementById(id);
          if (!el) return;
          new Chart(el, cfg);
        };

        createChart("jiraStatusChart", {
          type: "doughnut",
          data: {
            labels: ["Open", "In Progress", "Closed", "Other"],
            datasets: [{
              label: "Issues",
              data: [
                Number(summary.open_issues || 0),
                Number(summary.in_progress_issues || 0),
                Number(summary.closed_issues || 0),
                Number(summary.other_status_issues || 0),
              ],
              borderColor: "#ffffff",
              borderWidth: 2,
              backgroundColor: ["#dc2626", "#1d4ed8", "#16a34a", "#64748b"],
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: "bottom" } },
            cutout: "58%"
          }
        });

        const topProjects = Array.isArray(projects) ? projects.slice(0, 12) : [];
        const projectLabels = topProjects.map((p) => String(p.project_key || "UNKNOWN"));
        createChart("jiraProjectStatusChart", {
          type: "bar",
          data: {
            labels: projectLabels,
            datasets: [
              { label: "Open", data: topProjects.map((p) => Number(p.open_issues || 0)), backgroundColor: "#dc2626", borderRadius: 6 },
              { label: "In Progress", data: topProjects.map((p) => Number(p.in_progress_issues || 0)), backgroundColor: "#1d4ed8", borderRadius: 6 },
              { label: "Closed", data: topProjects.map((p) => Number(p.closed_issues || 0)), backgroundColor: "#16a34a", borderRadius: 6 },
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                stacked: true,
                title: { display: true, text: "Project" },
                ticks: { callback: (value, index) => staggerTick(projectLabels[index], index) }
              },
              y: { stacked: true, beginAtZero: true, title: { display: true, text: "Issue Count" } }
            },
            plugins: { legend: { display: true } }
          }
        });

        const topAssignees = Array.isArray(assignees) ? assignees.slice(0, 12) : [];
        createChart("jiraAssigneeBacklogChart", {
          type: "bar",
          data: {
            labels: topAssignees.map((a) => String(a.assignee || "Unassigned")),
            datasets: [{
              label: "Active Issues",
              data: topAssignees.map((a) => Number(a.active_issues || 0)),
              borderRadius: 8,
              backgroundColor: "#1d4ed8",
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
              x: { beginAtZero: true, title: { display: true, text: "Active Issues" } },
              y: { title: { display: true, text: "Assignee" } }
            },
            plugins: { legend: { display: true } }
          }
        });

        document.querySelectorAll(".chart-export-btn").forEach((btn) => {
          btn.addEventListener("click", () => {
            const canvasId = btn.getAttribute("data-canvas-id");
            const prefix = btn.getAttribute("data-file-prefix") || "report";
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const a = document.createElement("a");
            a.href = canvas.toDataURL("image/png");
            a.download = `${prefix}.${canvasId}.png`;
            document.body.appendChild(a);
            a.click();
            a.remove();
          });
        });
      })();
    </script>
    {% endif %}
    {% if is_ms_biomarker %}
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script>
      (function() {
        if (!window.Chart) return;
        Chart.defaults.font.family = "'Segoe UI', Tahoma, sans-serif";
        Chart.defaults.font.size = 13;
        Chart.defaults.color = "#334155";

        const summary = {{ ms.summary_json | safe }};
        const sid = {{ ms.sid_distribution_json | safe }};
        const sparse = {{ ms.sparse_biomarkers_json | safe }};
        const courses = {{ ms.course_distribution_json | safe }};
        const progression = {{ ms.edss_progression_json | safe }};
        const edssCorr = {{ ms.biomarker_sample_edss_corr_json | safe }};
        const pairCorr = {{ ms.biomarker_pair_corr_json | safe }};
        const corrMatrix = {{ ms.correlation_matrix_json | safe }};
        const corrMatrixSampleLow = {{ ms.correlation_matrix_sample_low_json | safe }};
        const corrMatrixSampleHigh = {{ ms.correlation_matrix_sample_high_json | safe }};
        const corrMatrixLastLow = {{ ms.correlation_matrix_last_low_json | safe }};
        const corrMatrixLastHigh = {{ ms.correlation_matrix_last_high_json | safe }};
        const staggerTick = (text, index) => (index % 2 === 0 ? [String(text || ""), ""] : ["", String(text || "")]);
        const courseLabels = Array.isArray(courses) ? courses.map((r) => String(r.course || "")) : [];

        const createChart = (id, cfg) => {
          const el = document.getElementById(id);
          if (!el) return;
          new Chart(el, cfg);
        };

        createChart("msEdssProgressionChart", {
          type: "bar",
          data: {
            labels: ["Improved", "Stable", "Worsened"],
            datasets: [{
              label: "Patients",
              data: [
                Number(progression.improved || 0),
                Number(progression.stable || 0),
                Number(progression.worsened || 0),
              ],
              borderRadius: 8,
              backgroundColor: ["#16a34a", "#64748b", "#dc2626"],
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { title: { display: true, text: "EDSS Change Category" } },
              y: { title: { display: true, text: "Count" }, beginAtZero: true }
            },
            plugins: { legend: { display: true } }
          }
        });

        createChart("msCourseDistributionChart", {
          type: "bar",
          data: {
            labels: courseLabels,
            datasets: [
              {
                label: "Sample Course",
                data: Array.isArray(courses) ? courses.map((r) => Number(r.sample_count || 0)) : [],
                backgroundColor: "#1d4ed8",
                borderRadius: 6,
              },
              {
                label: "Last Course",
                data: Array.isArray(courses) ? courses.map((r) => Number(r.last_count || 0)) : [],
                backgroundColor: "#0f766e",
                borderRadius: 6,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                stacked: false,
                title: { display: true, text: "Course" },
                ticks: { callback: (value, index) => staggerTick(courseLabels[index], index) }
              },
              y: { stacked: false, beginAtZero: true, title: { display: true, text: "Count" } }
            },
            plugins: { legend: { display: true } }
          }
        });

        const corrTop = Array.isArray(edssCorr) ? edssCorr.slice(0, 15) : [];
        createChart("msBiomarkerEdssCorrChart", {
          type: "bar",
          data: {
            labels: corrTop.map((r) => String(r.biomarker || "")),
            datasets: [{
              label: "Spearman Corr with Sample EDSS",
              data: corrTop.map((r) => Number(r.corr || 0)),
              borderRadius: 8,
              backgroundColor: corrTop.map((r) => Number(r.corr || 0) >= 0 ? "#dc2626" : "#1d4ed8"),
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                title: { display: true, text: "Biomarker" },
                ticks: { callback: (value, index) => staggerTick(corrTop[index] ? String(corrTop[index].biomarker || "") : "", index) }
              },
              y: { min: -1, max: 1, title: { display: true, text: "Correlation" } }
            },
            plugins: { legend: { display: true } }
          }
        });

        const pairTop = Array.isArray(pairCorr) ? pairCorr.slice(0, 12) : [];
        createChart("msBiomarkerPairCorrChart", {
          type: "bar",
          data: {
            labels: pairTop.map((r) => `${String(r.left_biomarker || "")} <> ${String(r.right_biomarker || "")}`),
            datasets: [{
              label: "Spearman Corr",
              data: pairTop.map((r) => Number(r.corr || 0)),
              borderRadius: 8,
              backgroundColor: pairTop.map((r) => Number(r.corr || 0) >= 0 ? "#0f766e" : "#7c3aed"),
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
              x: { min: -1, max: 1, title: { display: true, text: "Correlation" } },
              y: { title: { display: true, text: "Biomarker Pair" } }
            },
            plugins: { legend: { display: true } }
          }
        });

        const sparseTop = Array.isArray(sparse) ? sparse.slice(0, 12) : [];
        createChart("msSparseBiomarkersChart", {
          type: "bar",
          data: {
            labels: sparseTop.map((r) => String(r.biomarker || "")),
            datasets: [{
              label: "Missing %",
              data: sparseTop.map((r) => Number(r.missing_pct || 0)),
              borderRadius: 8,
              backgroundColor: "#f97316",
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
              x: { beginAtZero: true, max: 100, title: { display: true, text: "Missing %" } },
              y: { title: { display: true, text: "Biomarker" } }
            },
            plugins: { legend: { display: true } }
          }
        });

        createChart("msSidDistributionChart", {
          type: "doughnut",
          data: {
            labels: Array.isArray(sid) ? sid.map((r) => `SID ${r.sid}`) : [],
            datasets: [{
              label: "Rows",
              data: Array.isArray(sid) ? sid.map((r) => Number(r.count || 0)) : [],
              borderColor: "#ffffff",
              borderWidth: 2,
              backgroundColor: ["#1d4ed8", "#0f766e", "#7c3aed", "#ea580c", "#334155"],
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: "bottom" } },
            cutout: "58%"
          }
        });

        const renderCorrMatrix = (hostId, matrixPayload) => {
          const host = document.getElementById(hostId);
          if (!host) return;
          const columns = Array.isArray(matrixPayload.columns) ? matrixPayload.columns : [];
          const rows = Array.isArray(matrixPayload.rows) ? matrixPayload.rows : [];
          if (!columns.length || !rows.length) {
            host.innerHTML = '<p class="muted">Matrix unavailable: insufficient paired data.</p>';
            return;
          }
          const gradeColor = (absV, isSelf) => {
            if (isSelf) return "#d1d5db"; // self-correlation on diagonal
            if (absV >= 0.8) return "#166534"; // 100% ~ dark green
            if (absV >= 0.6) return "#4d9d2d";
            if (absV >= 0.4) return "#facc15"; // around 50% ~ yellow
            if (absV >= 0.2) return "#fef08a";
            return "#ffffff"; // none ~ white
          };
          let html = '<table class="ms-corr-matrix"><thead><tr><th style="font-size:11px;">Biomarker</th>';
          for (const col of columns) {
            html += `<th style="font-size:11px;white-space:nowrap;">${String(col)}</th>`;
          }
          html += "</tr></thead><tbody>";
          for (const row of rows) {
            const rowName = String(row.name || "");
            html += `<tr><td style="font-size:11px;white-space:nowrap;"><strong>${rowName}</strong></td>`;
            (row.values || []).forEach((value, idx) => {
              const n = (value === null || value === undefined) ? null : Number(value);
              const absV = (n === null || Number.isNaN(n)) ? 0 : Math.min(1, Math.abs(n));
              const colName = String(columns[idx] || "");
              const isSelf = rowName === colName;
              const bg = gradeColor(absV, isSelf);
              const title = (n === null || Number.isNaN(n))
                ? `${rowName} x ${colName}: no correlation value`
                : `${rowName} x ${colName}: corr=${n.toFixed(4)} (|corr| ${(absV * 100).toFixed(0)}%)`;
              html += `<td style="background:${bg};text-align:center;min-width:18px;height:16px;" title="${title}"></td>`;
            });
            html += "</tr>";
          }
          html += "</tbody></table>";
          html += `
            <div class="muted small" style="margin-top:0.55rem;display:flex;gap:0.65rem;align-items:center;flex-wrap:wrap;">
              <span><strong>Legend:</strong></span>
              <span style="display:inline-flex;align-items:center;gap:0.2rem;"><i style="display:inline-block;width:14px;height:14px;background:#166534;border:1px solid #d1d5db;"></i>100%</span>
              <span style="display:inline-flex;align-items:center;gap:0.2rem;"><i style="display:inline-block;width:14px;height:14px;background:#4d9d2d;border:1px solid #d1d5db;"></i>75%</span>
              <span style="display:inline-flex;align-items:center;gap:0.2rem;"><i style="display:inline-block;width:14px;height:14px;background:#facc15;border:1px solid #d1d5db;"></i>50%</span>
              <span style="display:inline-flex;align-items:center;gap:0.2rem;"><i style="display:inline-block;width:14px;height:14px;background:#fef08a;border:1px solid #d1d5db;"></i>25%</span>
              <span style="display:inline-flex;align-items:center;gap:0.2rem;"><i style="display:inline-block;width:14px;height:14px;background:#ffffff;border:1px solid #d1d5db;"></i>None</span>
              <span style="display:inline-flex;align-items:center;gap:0.2rem;"><i style="display:inline-block;width:14px;height:14px;background:#d1d5db;border:1px solid #9ca3af;"></i>Self</span>
            </div>
          `;
          host.innerHTML = html;
        };
        renderCorrMatrix("msCorrMatrixTable", corrMatrix);
        renderCorrMatrix("msCorrMatrixSampleLowTable", corrMatrixSampleLow);
        renderCorrMatrix("msCorrMatrixSampleHighTable", corrMatrixSampleHigh);
        renderCorrMatrix("msCorrMatrixLastLowTable", corrMatrixLastLow);
        renderCorrMatrix("msCorrMatrixLastHighTable", corrMatrixLastHigh);

        document.querySelectorAll(".chart-export-btn").forEach((btn) => {
          btn.addEventListener("click", () => {
            const canvasId = btn.getAttribute("data-canvas-id");
            const prefix = btn.getAttribute("data-file-prefix") || "report";
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const a = document.createElement("a");
            a.href = canvas.toDataURL("image/png");
            a.download = `${prefix}.${canvasId}.png`;
            document.body.appendChild(a);
            a.click();
            a.remove();
          });
        });
      })();
    </script>
    {% endif %}
  </body>
</html>
""".strip()


def render_html(report: dict) -> str:
    template = Template(_HTML_TEMPLATE)
    sections = _format_sections_for_display(report.get("sections", []))
    report_view = dict(report)
    report_view["sections"] = sections
    network = _extract_network_metrics(report)
    twamp = _extract_twamp_metrics(report)
    pm = _extract_pm_metrics(report)
    jira = _extract_jira_metrics(report)
    ms = _extract_ms_biomarker_metrics(report)
    return template.render(
        report=report_view,
        is_network_queue=report.get("report_type_id") == "network_queue_congestion" and network is not None,
        is_twamp=report.get("report_type_id") == "twamp_session_health" and twamp is not None,
        is_pm=report.get("report_type_id") == "pm_export_health" and pm is not None,
        is_jira=report.get("report_type_id") == "jira_issue_portfolio" and jira is not None,
        is_ms_biomarker=report.get("report_type_id") == "ms_biomarker_registry_health" and ms is not None,
        network=network or {},
        twamp=twamp or {},
        pm=pm or {},
        jira=jira or {},
        ms=ms or {},
    )


def _format_sections_for_display(sections: list[dict]) -> list[dict]:
    formatted: list[dict] = []
    for section in sections or []:
        title = str(section.get("title", ""))
        body = str(section.get("body", ""))
        formatted.append(
            {
                "title": title,
                "body": body,
                "body_html": _body_to_html(body),
            }
        )
    return formatted


def _body_to_html(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "<p></p>"

    mixed_html = _render_mixed_markdown_tables(lines)
    if mixed_html is not None:
        return mixed_html

    table_html = _markdown_table_to_html(lines)
    if table_html:
        return table_html

    bullet_lines = [line for line in lines if re.match(r"^[-*]\s+", line)]
    non_bullet_lines = [line for line in lines if line not in bullet_lines]
    if bullet_lines:
        parts: list[str] = []
        for line in non_bullet_lines:
            parts.append(f"<p>{_inline_format(line)}</p>")
        clean_bullets = [re.sub(r"^[-*]\s+", "", line) for line in bullet_lines]
        items = "".join(f"<li>{_inline_format(line)}</li>" for line in clean_bullets)
        parts.append(f"<ul>{items}</ul>")
        return "".join(parts)

    # If a section is one large paragraph, auto-split into bullets by sentence.
    if len(lines) == 1 and len(lines[0]) > 120:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", lines[0]) if s.strip()]
        if len(sentences) >= 3:
            items = "".join(f"<li>{_inline_format(s)}</li>" for s in sentences)
            return f"<ul>{items}</ul>"

    return "".join(f"<p>{_inline_format(line)}</p>" for line in lines)


def _markdown_table_to_html(lines: list[str]) -> str | None:
    pipe_lines = [line for line in lines if "|" in line]
    if len(pipe_lines) < 2:
        return None

    header_idx = None
    sep_idx = None
    for idx in range(len(pipe_lines) - 1):
        if _is_markdown_table_separator(pipe_lines[idx + 1]):
            header_idx = idx
            sep_idx = idx + 1
            break
    if header_idx is None or sep_idx is None:
        return None

    header_cells = _split_markdown_table_row(pipe_lines[header_idx])
    if not header_cells:
        return None

    body_rows: list[list[str]] = []
    for row_line in pipe_lines[sep_idx + 1 :]:
        cells = _split_markdown_table_row(row_line)
        if not cells:
            continue
        while len(cells) < len(header_cells):
            cells.append("")
        body_rows.append(cells[: len(header_cells)])

    if not body_rows:
        return None

    thead = "".join(f"<th>{_inline_format(cell)}</th>" for cell in header_cells)
    tbody_parts: list[str] = []
    for row in body_rows:
        tds = "".join(f"<td>{_inline_format(cell)}</td>" for cell in row)
        tbody_parts.append(f"<tr>{tds}</tr>")
    tbody = "".join(tbody_parts)
    return f"<div class='table-scroll'><table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>"


def _split_markdown_table_row(row: str) -> list[str]:
    trimmed = row.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [cell.strip() for cell in trimmed.split("|")]


def _is_markdown_table_separator(line: str) -> bool:
    cells = _split_markdown_table_row(line)
    if not cells:
        return False
    for cell in cells:
        compact = cell.replace(" ", "")
        if not compact:
            return False
        if not re.fullmatch(r":?-{3,}:?", compact):
            return False
    return True


def _render_mixed_markdown_tables(lines: list[str]) -> str | None:
    html_parts: list[str] = []
    idx = 0
    found_table = False

    while idx < len(lines):
        if idx + 1 < len(lines) and "|" in lines[idx] and _is_markdown_table_separator(lines[idx + 1]):
            table_lines = [lines[idx], lines[idx + 1]]
            idx += 2
            while idx < len(lines) and "|" in lines[idx]:
                table_lines.append(lines[idx])
                idx += 1
            table_html = _markdown_table_to_html(table_lines)
            if table_html:
                html_parts.append(table_html)
                found_table = True
                continue
            # Fallback: if parsing failed, keep original lines as paragraphs.
            for line in table_lines:
                html_parts.append(f"<p>{_inline_format(line)}</p>")
            continue

        html_parts.append(f"<p>{_inline_format(lines[idx])}</p>")
        idx += 1

    if not found_table:
        return None
    return "".join(html_parts)


def _inline_format(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\b([A-Za-z][A-Za-z0-9 /_-]{1,30}):\s", r"<strong>\1:</strong> ", escaped)
    return escaped


def _extract_network_metrics(report: dict) -> dict | None:
    if report.get("report_type_id") != "network_queue_congestion":
        return None
    tables = report.get("tables", [])
    for table in tables:
        if table.get("name") == "metrics_payload" and table.get("rows"):
            payload = table["rows"][0]
            summary = payload.get("summary", {})
            top_queues = payload.get("top_queues", [])
            time_trend = payload.get("time_trend", [])
            return _normalize_network_metrics(
                summary,
                top_queues,
                time_trend,
                payload.get("interval_hotspots", []),
                payload.get("l2_qos_insights", []),
            )
    return _normalize_network_metrics({}, [], [], [], [])


def _normalize_network_metrics(
    summary: dict,
    top_queues: list,
    time_trend: list,
    interval_hotspots: list,
    l2_qos_insights: list,
) -> dict:
    normalized_summary = {
        "interval_samples": int(summary.get("interval_samples", 0) or 0),
        "active_queues": int(summary.get("active_queues", 0) or 0),
        "overall_drop_ratio_bytes": float(summary.get("overall_drop_ratio_bytes", 0.0) or 0.0),
        "overall_drop_ratio_frames": float(summary.get("overall_drop_ratio_frames", 0.0) or 0.0),
    }
    return {
        "summary": normalized_summary,
        "top_queues": top_queues or [],
        "top_queues_json": json.dumps(top_queues or []),
        "interval_hotspots": interval_hotspots or [],
        "l2_qos_insights": l2_qos_insights or [],
        "time_trend_json": json.dumps(time_trend or []),
    }


def _extract_twamp_metrics(report: dict) -> dict | None:
    if report.get("report_type_id") != "twamp_session_health":
        return None
    tables = report.get("tables", [])
    for table in tables:
        if table.get("name") == "metrics_payload" and table.get("rows"):
            payload = table["rows"][0]
            return _normalize_twamp_metrics(
                payload.get("summary", {}),
                payload.get("top_discard_intervals", []),
                payload.get("time_trend", []),
                payload.get("color_packet_share_pct", {}),
                payload.get("l2_qos_insights", []),
            )
    return _normalize_twamp_metrics({}, [], [], {}, [])


def _normalize_twamp_metrics(
    summary: dict,
    top_discard_intervals: list,
    time_trend: list,
    color_packet_share_pct: dict,
    l2_qos_insights: list,
) -> dict:
    normalized_summary = {
        "samples": int(summary.get("samples", 0) or 0),
        "avg_discard_rate_pct": float(summary.get("avg_discard_rate_pct", 0.0) or 0.0),
        "p95_discard_rate_pct": float(summary.get("p95_discard_rate_pct", 0.0) or 0.0),
        "avg_yellow_traffic_pct": float(summary.get("avg_yellow_traffic_pct", 0.0) or 0.0),
        "total_forward_packets": int(summary.get("total_forward_packets", 0) or 0),
        "total_discard_packets": int(summary.get("total_discard_packets", 0) or 0),
        "avg_delay": float(summary.get("avg_delay", 0.0) or 0.0),
        "p95_ipdv_max": float(summary.get("p95_ipdv_max", 0.0) or 0.0),
    }
    return {
        "summary": normalized_summary,
        "top_discard_intervals": top_discard_intervals or [],
        "color_packet_share_pct": color_packet_share_pct or {},
        "l2_qos_insights": l2_qos_insights or [],
        "time_trend_json": json.dumps(time_trend or []),
    }


def _extract_pm_metrics(report: dict) -> dict | None:
    if report.get("report_type_id") != "pm_export_health":
        return None
    tables = report.get("tables", [])
    for table in tables:
        if table.get("name") == "metrics_payload" and table.get("rows"):
            payload = table["rows"][0]
            return _normalize_pm_metrics(
                payload.get("summary", {}),
                payload.get("top_interface_discards", []),
                payload.get("table_counts", {}),
                payload.get("l2_system_hypotheses", []),
            )
    return _normalize_pm_metrics({}, [], {}, [])


def _normalize_pm_metrics(
    summary: dict,
    top_interface_discards: list,
    table_counts: dict,
    l2_system_hypotheses: list,
) -> dict:
    normalized_summary = {
        "records": int(summary.get("records", 0) or 0),
        "table_types": int(summary.get("table_types", 0) or 0),
        "discard_delta_total": float(summary.get("discard_delta_total", 0.0) or 0.0),
        "error_delta_total": float(summary.get("error_delta_total", 0.0) or 0.0),
        "crc_delta_total": float(summary.get("crc_delta_total", 0.0) or 0.0),
        "cpu_avg_pct": float(summary.get("cpu_avg_pct", 0.0) or 0.0),
        "cpu_max_pct": float(summary.get("cpu_max_pct", 0.0) or 0.0),
        "memory_avg_used_pct": float(summary.get("memory_avg_used_pct", 0.0) or 0.0),
        "memory_max_used_pct": float(summary.get("memory_max_used_pct", 0.0) or 0.0),
        "disk_avg_used_pct": float(summary.get("disk_avg_used_pct", 0.0) or 0.0),
        "disk_max_used_pct": float(summary.get("disk_max_used_pct", 0.0) or 0.0),
        "risk_score": int(summary.get("risk_score", 0) or 0),
        "interface_signal_basis": str(summary.get("interface_signal_basis", "delta") or "delta"),
        "interface_error_signal": str(summary.get("interface_error_signal", "error_delta") or "error_delta"),
        "interface_error_signal_label": str(summary.get("interface_error_signal_label", "Error Delta") or "Error Delta"),
    }
    table_count_rows = [
        {"name": str(name), "count": int(count)} for name, count in sorted((table_counts or {}).items(), key=lambda x: x[1], reverse=True)
    ]
    return {
        "summary": normalized_summary,
        "summary_json": json.dumps(normalized_summary),
        "top_interface_discards": top_interface_discards or [],
        "top_interface_discards_json": json.dumps(top_interface_discards or []),
        "table_counts": table_count_rows,
        "table_counts_json": json.dumps(table_count_rows),
        "l2_system_hypotheses": l2_system_hypotheses or [],
    }


def _extract_jira_metrics(report: dict) -> dict | None:
    if report.get("report_type_id") != "jira_issue_portfolio":
        return None
    tables = report.get("tables", [])
    for table in tables:
        if table.get("name") == "metrics_payload" and table.get("rows"):
            payload = table["rows"][0]
            return _normalize_jira_metrics(
                payload.get("summary", {}),
                payload.get("project_status_breakdown", []),
                payload.get("responsible_workload", []),
                payload.get("oldest_open_issues", []),
            )
    return _normalize_jira_metrics({}, [], [], [])


def _normalize_jira_metrics(
    summary: dict,
    project_status_breakdown: list,
    responsible_workload: list,
    oldest_open_issues: list,
) -> dict:
    normalized_summary = {
        "total_issues": int(summary.get("total_issues", 0) or 0),
        "projects": int(summary.get("projects", 0) or 0),
        "open_issues": int(summary.get("open_issues", 0) or 0),
        "in_progress_issues": int(summary.get("in_progress_issues", 0) or 0),
        "closed_issues": int(summary.get("closed_issues", 0) or 0),
        "other_status_issues": int(summary.get("other_status_issues", 0) or 0),
        "closure_ratio": float(summary.get("closure_ratio", 0.0) or 0.0),
    }
    return {
        "summary": normalized_summary,
        "summary_json": json.dumps(normalized_summary),
        "project_status_breakdown": project_status_breakdown or [],
        "project_status_breakdown_json": json.dumps(project_status_breakdown or []),
        "responsible_workload": responsible_workload or [],
        "responsible_workload_json": json.dumps(responsible_workload or []),
        "oldest_open_issues": oldest_open_issues or [],
    }


def _extract_ms_biomarker_metrics(report: dict) -> dict | None:
    if report.get("report_type_id") != "ms_biomarker_registry_health":
        return None
    tables = report.get("tables", [])
    for table in tables:
        if table.get("name") == "metrics_payload" and table.get("rows"):
            payload = table["rows"][0]
            return _normalize_ms_biomarker_metrics(
                payload.get("summary", {}),
                payload.get("sid_distribution", []),
                payload.get("key_missingness", []),
                payload.get("sparse_biomarkers", []),
                payload.get("edss_progression", {}),
                payload.get("course_distribution", []),
                payload.get("followup_summary", {}),
                payload.get("biomarker_sample_edss_corr", payload.get("biomarker_last_edss_corr", [])),
                payload.get("biomarker_sample_edss_group_corr", []),
                payload.get("biomarker_last_edss_group_corr", []),
                payload.get("biomarker_pair_corr", []),
                payload.get("correlation_matrix", {}),
                payload.get("correlation_matrix_sample_low", {}),
                payload.get("correlation_matrix_sample_high", {}),
                payload.get("correlation_matrix_last_low", {}),
                payload.get("correlation_matrix_last_high", {}),
            )
    return _normalize_ms_biomarker_metrics({}, [], [], [], {}, [], {}, [], [], [], [], {}, {}, {}, {}, {})


def _normalize_ms_biomarker_metrics(
    summary: dict,
    sid_distribution: list,
    key_missingness: list,
    sparse_biomarkers: list,
    edss_progression: dict,
    course_distribution: list,
    followup_summary: dict,
    biomarker_sample_edss_corr: list,
    biomarker_sample_edss_group_corr: list,
    biomarker_last_edss_group_corr: list,
    biomarker_pair_corr: list,
    correlation_matrix: dict,
    correlation_matrix_sample_low: dict,
    correlation_matrix_sample_high: dict,
    correlation_matrix_last_low: dict,
    correlation_matrix_last_high: dict,
) -> dict:
    normalized_summary = {
        "rows": int(summary.get("rows", 0) or 0),
        "unique_patients": int(summary.get("unique_patients", 0) or 0),
        "biomarker_columns_used": int(summary.get("biomarker_columns_used", 0) or 0),
        "high_sparse_biomarkers": int(summary.get("high_sparse_biomarkers", 0) or 0),
        "strong_contributors": int(summary.get("strong_contributors", 0) or 0),
        "weak_contributors": int(summary.get("weak_contributors", 0) or 0),
        "edss_pairs": int(summary.get("edss_pairs", 0) or 0),
        "edss_worsened_ratio": float(summary.get("edss_worsened_ratio", 0.0) or 0.0),
        "correlation_method": str(
            summary.get("correlation_method", "Spearman rank correlation (Pearson computed on ranked values)")
        ),
    }
    normalized_edss = {
        "improved": int(edss_progression.get("improved", 0) or 0),
        "stable": int(edss_progression.get("stable", 0) or 0),
        "worsened": int(edss_progression.get("worsened", 0) or 0),
    }
    normalized_followup = {
        "pairs": int(followup_summary.get("pairs", 0) or 0),
        "median_days": int(followup_summary.get("median_days", 0) or 0),
        "p90_days": int(followup_summary.get("p90_days", 0) or 0),
        "invalid_pairs": int(followup_summary.get("invalid_pairs", 0) or 0),
    }
    return {
        "summary": normalized_summary,
        "summary_json": json.dumps(normalized_summary),
        "sid_distribution": sid_distribution or [],
        "sid_distribution_json": json.dumps(sid_distribution or []),
        "key_missingness": key_missingness or [],
        "sparse_biomarkers": sparse_biomarkers or [],
        "sparse_biomarkers_json": json.dumps(sparse_biomarkers or []),
        "edss_progression": normalized_edss,
        "edss_progression_json": json.dumps(normalized_edss),
        "course_distribution": course_distribution or [],
        "course_distribution_json": json.dumps(course_distribution or []),
        "followup_summary": normalized_followup,
        "biomarker_sample_edss_corr": biomarker_sample_edss_corr or [],
        "biomarker_sample_edss_corr_json": json.dumps(biomarker_sample_edss_corr or []),
        "biomarker_sample_edss_group_corr": biomarker_sample_edss_group_corr or [],
        "biomarker_sample_edss_group_corr_json": json.dumps(biomarker_sample_edss_group_corr or []),
        "biomarker_last_edss_group_corr": biomarker_last_edss_group_corr or [],
        "biomarker_last_edss_group_corr_json": json.dumps(biomarker_last_edss_group_corr or []),
        "biomarker_pair_corr": biomarker_pair_corr or [],
        "biomarker_pair_corr_json": json.dumps(biomarker_pair_corr or []),
        "correlation_matrix": correlation_matrix or {"columns": [], "rows": []},
        "correlation_matrix_json": json.dumps(correlation_matrix or {"columns": [], "rows": []}),
        "correlation_matrix_sample_low": correlation_matrix_sample_low or {"columns": [], "rows": []},
        "correlation_matrix_sample_low_json": json.dumps(correlation_matrix_sample_low or {"columns": [], "rows": []}),
        "correlation_matrix_sample_high": correlation_matrix_sample_high or {"columns": [], "rows": []},
        "correlation_matrix_sample_high_json": json.dumps(correlation_matrix_sample_high or {"columns": [], "rows": []}),
        "correlation_matrix_last_low": correlation_matrix_last_low or {"columns": [], "rows": []},
        "correlation_matrix_last_low_json": json.dumps(correlation_matrix_last_low or {"columns": [], "rows": []}),
        "correlation_matrix_last_high": correlation_matrix_last_high or {"columns": [], "rows": []},
        "correlation_matrix_last_high_json": json.dumps(correlation_matrix_last_high or {"columns": [], "rows": []}),
    }

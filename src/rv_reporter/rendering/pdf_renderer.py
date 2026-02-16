from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def render_pdf(html_path: str | Path, output_path: str | Path, fallback_report: dict[str, Any] | None = None) -> bool:
    """Render PDF from full HTML (preferred) and fallback to simplified template.

    Returns True on success, False when PDF backend is unavailable.
    """
    html_file = Path(html_path).resolve()
    pdf_file = Path(output_path).resolve()
    pdf_file.parent.mkdir(parents=True, exist_ok=True)

    if _render_pdf_with_playwright(html_file, pdf_file):
        return True
    if fallback_report is None:
        return False
    return _render_pdf_with_xhtml2pdf(fallback_report, pdf_file)


def _render_pdf_with_playwright(html_file: Path, pdf_file: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False

    if not html_file.exists():
        return False

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(html_file.as_uri(), wait_until="networkidle")
            page.add_style_tag(
                content="""
                .chart-export-btn { display: none !important; }
                body { background: #ffffff !important; }
                section {
                  background: #ffffff !important;
                  backdrop-filter: none !important;
                  -webkit-backdrop-filter: none !important;
                  box-shadow: none !important;
                }
                td, th, p, li, h1, h2, h3 { color: #111111 !important; }
                """
            )
            # Wait briefly for Chart.js/canvas initialization, then print.
            try:
                page.wait_for_function(
                    """
                    () => {
                      const canvases = Array.from(document.querySelectorAll("canvas"));
                      if (canvases.length === 0) return true;
                      if (!window.Chart) return false;
                      return canvases.every((c) => c.width > 0 && c.height > 0);
                    }
                    """,
                    timeout=5000,
                )
            except Exception:
                pass
            page.wait_for_timeout(900)
            page.pdf(
                path=str(pdf_file),
                format="A4",
                print_background=True,
                margin={"top": "10mm", "right": "8mm", "bottom": "10mm", "left": "8mm"},
            )
            browser.close()
        return True
    except Exception:
        return False


def _render_pdf_with_xhtml2pdf(report: dict[str, Any], output_path: Path) -> bool:
    try:
        from xhtml2pdf import pisa
    except Exception:
        return False

    html_content = _build_pdf_html(report)
    try:
        with output_path.open("wb") as target:
            result = pisa.CreatePDF(src=html_content, dest=target)
        return not bool(result.err)
    except Exception:
        return False


def _build_pdf_html(report: dict[str, Any]) -> str:
    title = _esc(str(report.get("report_title", "Report")))
    summary = _esc(str(report.get("summary", "")))
    metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
    sections = report.get("sections", []) if isinstance(report.get("sections"), list) else []
    alerts = report.get("alerts", []) if isinstance(report.get("alerts"), list) else []
    recommendations = report.get("recommendations", []) if isinstance(report.get("recommendations"), list) else []

    meta_html = "".join(f"<li><b>{_esc(str(k))}:</b> {_esc(str(v))}</li>" for k, v in metadata.items())
    section_html = "".join(
        f"<h3>{_esc(str(s.get('title', '')))}</h3><p>{_esc(str(s.get('body', '')))}</p>"
        for s in sections
        if isinstance(s, dict)
    )
    alerts_html = "".join(
        f"<li><b>{_esc(str(a.get('severity', 'info')))}:</b> {_esc(str(a.get('message', '')))}</li>"
        for a in alerts
        if isinstance(a, dict)
    )
    recs_html = "".join(
        f"<li><b>{_esc(str(r.get('priority', 'medium')))}:</b> {_esc(str(r.get('action', '')))}</li>"
        for r in recommendations
        if isinstance(r, dict)
    )

    return (
        "<html><head><meta charset='utf-8'/>"
        "<style>"
        "body{font-family:Helvetica,Arial,sans-serif;font-size:11pt;color:#111;}"
        "h1{font-size:20pt;margin:0 0 10px 0;}h2{font-size:14pt;margin:16px 0 6px 0;}"
        "h3{font-size:12pt;margin:10px 0 4px 0;}p{margin:0 0 8px 0;line-height:1.35;}"
        "ul{margin:0 0 10px 18px;}li{margin:0 0 4px 0;}"
        ".muted{color:#555;}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        f"<p class='muted'>{summary}</p>"
        "<h2>Metadata</h2>"
        f"<ul>{meta_html or '<li>None</li>'}</ul>"
        "<h2>Sections</h2>"
        f"{section_html or '<p>None</p>'}"
        "<h2>Alerts</h2>"
        f"<ul>{alerts_html or '<li>None</li>'}</ul>"
        "<h2>Recommendations</h2>"
        f"<ul>{recs_html or '<li>None</li>'}</ul>"
        "</body></html>"
    )


def _esc(value: str) -> str:
    return html.escape(value, quote=True)

/**
 * wizard/reasoning_chat.js
 * Reusable AI reasoning chat panel with endless feedback loop.
 *
 * Usage:
 *   const chat = ReasoningChat.create({
 *     containerId: "my-chat-panel",
 *     onResult: (result) => { ... },
 *   });
 *   chat.start(apiUrl, payload);
 *   // User clicks Send → chat.sendFeedback(text)
 */

window.ReasoningChat = (function () {
  "use strict";

  /**
   * Create a reasoning chat instance bound to a container element.
   * @param {object} opts
   * @param {string} opts.containerId  - ID of the chat panel container
   * @param {function} opts.onResult   - Called with parsed JSON after each AI response
   * @param {function} [opts.renderMessage] - Custom renderer: (role, content, container) => void
   */
  function create(opts) {
    const container = document.getElementById(opts.containerId);
    if (!container) {
      console.warn("ReasoningChat: container not found:", opts.containerId);
      return null;
    }

    const messagesEl = container.querySelector(".rc-messages");
    const inputEl = container.querySelector(".rc-input");
    const sendBtn = container.querySelector(".rc-send-btn");
    const statusEl = container.querySelector(".rc-status");

    let apiUrl = "";
    let basePayload = {};
    let lastRawResponse = "";
    let busy = false;

    /* ── Render helpers ─────────────────────────────────────────── */

    function addMessage(role, html) {
      const msg = document.createElement("div");
      msg.className = "rc-msg rc-msg-" + role;
      msg.innerHTML = html;
      messagesEl.appendChild(msg);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    }

    function setBusy(val) {
      busy = val;
      if (sendBtn) sendBtn.disabled = val;
      if (inputEl) inputEl.disabled = val;
      setStatus(val ? "AI is thinking…" : "");
    }

    /* ── Transform reasoning renderer ──────────────────────────── */

    function renderTransformResult(data) {
      let html = "";
      if (data.reasoning) {
        html += '<div class="rc-reasoning">' + escapeHtml(data.reasoning).replace(/\n/g, "<br>") + "</div>";
      }
      if (data.transformations && data.transformations.length) {
        html += '<div class="rc-section-title">🔧 Proposed Transformations (' + data.transformations.length + ")</div>";
        data.transformations.forEach(function (t, i) {
          html += '<div class="rc-bp-section">'
            + '<strong>' + (i + 1) + '. <span class="rc-chip">' + escapeHtml(t.type) + '</span></strong> '
            + '→ ' + (t.columns || []).map(escapeHtml).join(", ")
            + '<div class="rc-bp-purpose">' + escapeHtml(t.reason) + "</div></div>";
        });
      }
      if (data.new_columns && data.new_columns.length) {
        html += '<div class="rc-section-title">📋 Resulting Columns (' + data.new_columns.length + ")</div>";
        html += '<div class="rc-chip-row">';
        data.new_columns.forEach(function (c) {
          html += '<span class="rc-chip family">' + escapeHtml(c) + "</span>";
        });
        html += "</div>";
      }
      if (data.result_description) {
        html += '<div class="rc-section-title">📄 Result</div>';
        html += '<div class="rc-bp-summary">' + escapeHtml(data.result_description) + "</div>";
      }
      if (data.pandas_code) {
        html += '<div class="rc-section-title">🐍 Pandas Code</div>';
        html += '<pre style="background:#1e1e2e;color:#cdd6f4;padding:.6rem;border-radius:8px;font-size:.76rem;overflow-x:auto;max-height:250px;">'
          + escapeHtml(data.pandas_code) + "</pre>";
      }
      if (data.suggestions && data.suggestions.length) {
        html += '<div class="rc-section-title">💡 Suggestions</div><ul class="rc-suggestions">';
        data.suggestions.forEach(function (s) { html += "<li>" + escapeHtml(s) + "</li>"; });
        html += "</ul>";
      }
      return html;
    }

    /* ── Classification reasoning renderer ──────────────────────── */

    function renderClassifyResult(data) {
      let html = "";
      if (data.reasoning) {
        html += '<div class="rc-reasoning">' + escapeHtml(data.reasoning).replace(/\n/g, "<br>") + "</div>";
      }
      html += '<div class="rc-result-chips">';
      if (data.domain) html += '<span class="rc-chip rc-chip-domain">Domain: ' + escapeHtml(data.domain) + "</span>";
      if (data.family) html += '<span class="rc-chip rc-chip-family">Family: ' + escapeHtml(data.family) + "</span>";
      if (data.mode)   html += '<span class="rc-chip rc-chip-mode">Mode: ' + escapeHtml(data.mode) + "</span>";
      if (data.confidence) html += '<span class="rc-chip rc-chip-conf">Confidence: ' + escapeHtml(data.confidence) + "</span>";
      html += "</div>";
      if (data.alternatives && data.alternatives.length) {
        html += '<div class="rc-section-title">Alternative Classifications</div>';
        data.alternatives.forEach(function (alt) {
          html += '<div class="rc-alt">'
            + '<span class="rc-chip">' + escapeHtml(alt.domain) + " · " + escapeHtml(alt.family) + " · " + escapeHtml(alt.mode) + "</span> "
            + '<span class="rc-alt-why">' + escapeHtml(alt.why) + "</span></div>";
        });
      }
      if (data.suggestions && data.suggestions.length) {
        html += '<div class="rc-section-title">💡 Suggestions</div><ul class="rc-suggestions">';
        data.suggestions.forEach(function (s) { html += "<li>" + escapeHtml(s) + "</li>"; });
        html += "</ul>";
      }
      return html;
    }

    /* ── Blueprint reasoning renderer ───────────────────────────── */

    function renderBlueprintResult(data) {
      let html = "";
      if (data.title) html += '<div class="rc-bp-title">' + escapeHtml(data.title) + "</div>";
      if (data.summary_description) html += '<div class="rc-bp-summary">' + escapeHtml(data.summary_description) + "</div>";

      if (data.sections && data.sections.length) {
        html += '<div class="rc-section-title">📄 Sections (' + data.sections.length + ")</div>";
        data.sections.forEach(function (s, i) {
          html += '<div class="rc-bp-section"><strong>' + (i + 1) + ". " + escapeHtml(s.title) + "</strong>"
            + '<div class="rc-bp-purpose">' + escapeHtml(s.purpose) + "</div>"
            + '<div class="rc-bp-narrative">"' + escapeHtml(s.narrative_preview) + '"</div></div>';
        });
      }

      if (data.charts && data.charts.length) {
        html += '<div class="rc-section-title">📊 Charts (' + data.charts.length + ")</div>";
        data.charts.forEach(function (c) {
          html += '<div class="rc-bp-chart">📊 <strong>' + escapeHtml(c.title) + "</strong> (" + escapeHtml(c.chart_type) + ")"
            + " — X: " + escapeHtml(c.x_axis) + ", Y: " + escapeHtml(c.y_axis)
            + '<div class="rc-bp-desc">' + escapeHtml(c.description) + "</div></div>";
        });
      }

      if (data.tables && data.tables.length) {
        html += '<div class="rc-section-title">📋 Tables (' + data.tables.length + ")</div>";
        data.tables.forEach(function (t) {
          html += '<div class="rc-bp-table">📋 <strong>' + escapeHtml(t.title) + "</strong>"
            + " — Columns: [" + t.columns.map(escapeHtml).join(", ") + "]"
            + '<div class="rc-bp-desc">' + escapeHtml(t.row_description) + "</div></div>";
        });
      }

      if (data.alerts && data.alerts.length) {
        html += '<div class="rc-section-title">⚠ Alerts (' + data.alerts.length + ")</div>";
        data.alerts.forEach(function (a) {
          html += '<div class="rc-bp-alert"><span class="rc-chip rc-chip-' + escapeHtml(a.severity) + '">' + escapeHtml(a.severity) + "</span> "
            + escapeHtml(a.condition) + ': "' + escapeHtml(a.message_template) + '"</div>';
        });
      }

      if (data.recommendations && data.recommendations.length) {
        html += '<div class="rc-section-title">✅ Recommendations</div><ul>';
        data.recommendations.forEach(function (r) {
          html += "<li><strong>" + escapeHtml(r.priority) + ":</strong> " + escapeHtml(r.action) + "</li>";
        });
        html += "</ul>";
      }

      if (data.gaps && data.gaps.length) {
        html += '<div class="rc-section-title">🔍 Gaps & Missing Data</div><ul class="rc-gaps">';
        data.gaps.forEach(function (g) { html += "<li>" + escapeHtml(g) + "</li>"; });
        html += "</ul>";
      }

      if (data.suggestions && data.suggestions.length) {
        html += '<div class="rc-section-title">💡 Suggestions</div><ul class="rc-suggestions">';
        data.suggestions.forEach(function (s) { html += "<li>" + escapeHtml(s) + "</li>"; });
        html += "</ul>";
      }

      return html;
    }

    /* ── API call ───────────────────────────────────────────────── */

    async function callApi(payload) {
      setBusy(true);
      try {
        const resp = await fetch(apiUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await resp.json();
        if (!resp.ok) {
          addMessage("error", "❌ " + escapeHtml(result.error || "Unknown error"));
          setBusy(false);
          return null;
        }
        lastRawResponse = JSON.stringify(result, null, 2);

        // Render based on response shape
        let html;
        if (opts.renderMessage) {
          html = opts.renderMessage("assistant", result);
        } else if (result.transformations !== undefined) {
          html = renderTransformResult(result);
        } else if (result.reasoning !== undefined) {
          html = renderClassifyResult(result);
        } else if (result.sections !== undefined) {
          html = renderBlueprintResult(result);
        } else {
          html = "<pre>" + escapeHtml(lastRawResponse) + "</pre>";
        }
        addMessage("assistant", html);

        if (opts.onResult) opts.onResult(result);
        setBusy(false);
        return result;
      } catch (err) {
        addMessage("error", "❌ Network error: " + escapeHtml(String(err)));
        setBusy(false);
        return null;
      }
    }

    /* ── Public API ──────────────────────────────────────────────── */

    function start(url, payload) {
      apiUrl = url;
      basePayload = Object.assign({}, payload);
      messagesEl.innerHTML = "";
      addMessage("system", "🤖 Analyzing your data… I'll show my reasoning so you can give feedback.");
      callApi(payload);
    }

    function sendFeedback(text) {
      if (!text.trim() || busy) return;
      addMessage("user", escapeHtml(text));
      const payload = Object.assign({}, basePayload, {
        feedback: text,
        previous_reasoning: lastRawResponse,
        previous_blueprint: lastRawResponse,
      });
      callApi(payload);
    }

    function getLastResult() {
      return lastRawResponse;
    }

    /* ── Wire up send button + enter key ─────────────────────────── */

    if (sendBtn) {
      sendBtn.addEventListener("click", function () {
        if (inputEl && inputEl.value.trim()) {
          sendFeedback(inputEl.value.trim());
          inputEl.value = "";
        }
      });
    }
    if (inputEl) {
      inputEl.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          if (inputEl.value.trim()) {
            sendFeedback(inputEl.value.trim());
            inputEl.value = "";
          }
        }
      });
    }

    return { start: start, sendFeedback: sendFeedback, getLastResult: getLastResult, addMessage: addMessage };
  }

  /* ── Escape HTML to prevent XSS ──────────────────────────────── */

  function escapeHtml(str) {
    if (str == null) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /* ── Public module API ───────────────────────────────────────── */

  return { create: create };
})();

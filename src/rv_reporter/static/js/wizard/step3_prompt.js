/**
 * wizard/step3_prompt.js
 * Step 3 — Write & Validate Prompt
 *
 * Responsibilities:
 *   - Build a rich, domain/column-aware prompt when entering step 3
 *   - Show classification summary chips (domain · family · mode)
 *   - Run guardrail validation and render the score panel
 *   - Ask the AI to improve the prompt and explain what changed
 *
 * Depends on: data.js (RVWizard.data), state.js (RVWizard.state)
 * Config keys consumed: WIZARD_CONFIG.validatePromptUrl, WIZARD_CONFIG.improvePromptUrl
 */

window.RVWizard = window.RVWizard || {};

RVWizard.step3 = (function () {
  const data  = RVWizard.data;
  const state = RVWizard.state;

  /* ── Rich prompt builder ─────────────────────────────────────────── */
  function populateRichPrompt() {
    const pt = document.getElementById("prompt_text");
    const verb = data.DOMAIN_VERB[state.selectedDomain] || "Create a report";
    let prompt = verb;

    if (state.sourceColumns && state.sourceColumns.length) {
      prompt += ` for CSVs with these columns: ${state.sourceColumns.join(", ")}.`;
    } else {
      // Fall back to the example columns embedded in the domain template
      const tpl = data.PROMPT_TEMPLATES[state.selectedDomain] || "";
      const colMatch = tpl.match(/with(?:[\s\w]+columns:?)?\s([^.]+\.)/i);
      prompt += colMatch ? " for CSVs " + colMatch[0].replace(/^with/i, "with") : ".";
    }

    if (state.selectedMode && data.MODE_GOAL[state.selectedMode]) {
      prompt += " " + data.MODE_GOAL[state.selectedMode];
    }

    const questions = data.DOMAIN_CONTEXT_QUESTIONS[state.selectedDomain];
    if (questions && questions.length) {
      prompt += "\n\n" + questions.join("\n");
    }

    pt.value = prompt;
  }

  /* ── Classification summary chips ───────────────────────────────── */
  function refreshClassificationSummary() {
    const summary = document.getElementById("classification-summary");
    const dEl = document.getElementById("cs-domain");
    const fEl = document.getElementById("cs-family");
    const mEl = document.getElementById("cs-mode");
    let any = false;

    if (state.selectedDomain) {
      dEl.textContent = "domain: " + ((data.DOMAIN_LABELS || {})[state.selectedDomain] || state.selectedDomain.replace(/_/g, " "));
      dEl.style.display = "";
      any = true;
    } else {
      dEl.style.display = "none";
    }
    if (state.selectedFamily) {
      fEl.textContent = "family: " + ((data.FAMILY_LABELS || {})[state.selectedFamily] || state.selectedFamily.replace(/_/g, " "));
      fEl.style.display = "";
      any = true;
    } else {
      fEl.style.display = "none";
    }
    if (state.selectedMode) {
      mEl.textContent = "mode: " + ((data.MODE_LABELS || {})[state.selectedMode] || state.selectedMode.replace(/_/g, " "));
      mEl.style.display = "";
      any = true;
    } else {
      mEl.style.display = "none";
    }

    summary.style.display = any ? "flex" : "none";
  }

  /* ── Guardrail render ────────────────────────────────────────────── */
  function renderGuardrail(result) {
    const panel = document.getElementById("guardrail-panel");
    panel.style.display = "";

    const fill  = document.getElementById("guardrail-score-fill");
    const score = result.score ?? 0;
    fill.style.width      = score + "%";
    fill.style.background = score >= 80 ? "#15803d" : score >= 50 ? "#d97706" : "#b91c1c";

    const header = document.getElementById("guardrail-header");
    header.textContent = "Prompt Quality: " + score + "/100" + (result.valid ? " ✓" : " ✗");
    header.style.color = result.valid ? "#15803d" : "#b91c1c";

    function renderList(containerId, items, cls, prefix) {
      const el = document.getElementById(containerId);
      if (!items || !items.length) { el.style.display = "none"; el.innerHTML = ""; return; }
      el.style.display = "";
      el.innerHTML = items.map(i => `<div class="${cls}">${prefix} ${i}</div>`).join("");
    }
    renderList("guardrail-issues",      result.issues,      "guardrail-issue", "✗");
    renderList("guardrail-warnings",    result.warnings,    "guardrail-warn",  "⚠");
    renderList("guardrail-suggestions", result.suggestions, "guardrail-sugg",  "💡");
  }

  function renderPromptImprovements(changes) {
    const panel = document.getElementById("prompt-improve-panel");
    const container = document.getElementById("prompt-improve-changes");
    if (!panel || !container) return;
    if (!changes || !changes.length) {
      panel.style.display = "none";
      container.innerHTML = "";
      return;
    }
    panel.style.display = "";
    container.innerHTML = changes.map((item) => `<div>• ${item}</div>`).join("");
  }

  /* ── Check Prompt button ─────────────────────────────────────────── */
  document.getElementById("check-prompt-btn").addEventListener("click", async () => {
    const promptText = document.getElementById("prompt_text").value.trim();
    const btn = document.getElementById("check-prompt-btn");
    btn.disabled = true;
    btn.textContent = "Checking…";
    try {
      const resp = await fetch(window.WIZARD_CONFIG.validatePromptUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt_text: promptText, hint_domain: state.selectedDomain }),
      });
      renderGuardrail(await resp.json());
    } catch (_) {
      renderGuardrail({ valid: false, score: 0,
        issues: ["Could not reach validation endpoint."], warnings: [], suggestions: [] });
    } finally {
      btn.disabled = false;
      btn.textContent = "🛡 Check Prompt";
    }
  });

  document.getElementById("improve-prompt-btn").addEventListener("click", async () => {
    const promptText = document.getElementById("prompt_text").value.trim();
    const btn = document.getElementById("improve-prompt-btn");
    btn.disabled = true;
    btn.textContent = "Improving…";
    try {
      const resp = await fetch(window.WIZARD_CONFIG.improvePromptUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_text: promptText,
          hint_domain: state.selectedDomain,
          hint_family: state.selectedFamily,
          hint_mode: state.selectedMode,
          source_columns: state.sourceColumns || [],
        }),
      });
      const payload = await resp.json();
      if (!resp.ok) {
        throw new Error(payload.error || "Prompt improvement failed.");
      }
      if (payload.improved_prompt) {
        document.getElementById("prompt_text").value = payload.improved_prompt;
      }
      renderPromptImprovements(payload.changes || []);
      if (window.RVWizard && typeof window.RVWizard.showStep === "function") {
        window.RVWizard.showStep(3);
      }
    } catch (err) {
      renderPromptImprovements([err.message || "Could not improve prompt."]);
    } finally {
      btn.disabled = false;
      btn.textContent = "What Needs To Be Done Better?";
    }
  });

  return { populateRichPrompt, refreshClassificationSummary };
})();

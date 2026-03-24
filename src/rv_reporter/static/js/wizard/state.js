/**
 * wizard/state.js
 * Shared mutable state for the wizard and the pipeline highlight utility.
 * Must be loaded before any step files.
 */

window.RVWizard = window.RVWizard || {};

/* ── Shared state ────────────────────────────────────────────────────── */
RVWizard.state = {
  currentStep:    1,
  selectedDomain: "",
  selectedFamily: "",
  selectedMode:   "",
  sourceColumns:  [],   // populated by step1 after CSV metadata fetch
  sourceRowCount: null,
};

/* ── Pipeline highlight ──────────────────────────────────────────────── */
RVWizard.highlightPipelineStep = function (n) {
  for (let i = 1; i <= 7; i++) {
    const node  = document.getElementById("wf-node-" + i);
    const badge = document.getElementById("wf-badge-" + i);
    const arr   = document.getElementById("wf-arrow-" + i);
    if (!node || !badge) continue;

    const active = (i === n);
    const done   = (i < n);

    /* toggle CSS classes defined in macros.html */
    node.classList.toggle("is-active", active);
    node.classList.toggle("is-done", done);

    badge.textContent = done ? "✓" : String(i);

    if (arr) {
      const col = done ? "#15803d" : "#dbe5f0";
      arr.querySelector("path").setAttribute("stroke", col);
      arr.querySelector("polygon").setAttribute("fill", col);
      arr.querySelector("path").setAttribute("stroke-dasharray", done ? "0" : "4 2");
    }
  }
};

/* ── Wizard step navigation ──────────────────────────────────────────── */
RVWizard.showStep = function (n) {
  document.querySelectorAll(".wizard-step").forEach(el => el.classList.add("hidden"));
  const el = document.getElementById("step-" + n);
  if (el) el.classList.remove("hidden");
  RVWizard.state.currentStep = n;
  RVWizard.highlightPipelineStep(n);
  if (n === 3 && RVWizard.step3) {
    RVWizard.step3.refreshClassificationSummary();
  }
};

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
RVWizard.highlightPipelineStep = function (stepId) {
  /* stepId can be: 1, "1b", 2, "2b", 3, "3b", etc. */
  const stepList = ["1", "1b", "2", "3", "4", "5", "6", "7"];
  const activeIdx = stepList.indexOf(stepId.toString());
  
  stepList.forEach((step, idx) => {
    const node  = document.getElementById("wf-node-" + step);
    const badge = document.getElementById("wf-badge-" + step);
    const arr   = document.getElementById("wf-arrow-" + step);
    if (!node || !badge) return;

    const active = (idx === activeIdx);
    const done   = (idx < activeIdx);

    node.classList.toggle("is-active", active);
    node.classList.toggle("is-done", done);

    badge.textContent = done ? "✓" : step;

    if (arr) {
      const col = done ? "#15803d" : "#dbe5f0";
      arr.querySelector("path").setAttribute("stroke", col);
      arr.querySelector("polygon").setAttribute("fill", col);
      arr.querySelector("path").setAttribute("stroke-dasharray", done ? "0" : "4 2");
    }
  });
};

/* ── Wizard step navigation ──────────────────────────────────────────── */
RVWizard.showStep = function (n) {
  document.querySelectorAll(".wizard-step").forEach(el => el.classList.add("hidden"));
  const el = document.getElementById("step-" + n);
  if (el) el.classList.remove("hidden");
  RVWizard.state.currentStep = n;
  /* Map sub-steps to parent pipeline step for highlight */
  const pipelineMap = { "1b": 1, "2b": 2, "3b": 3 };
  const pipelineStep = pipelineMap[n] || (typeof n === "number" ? n : parseInt(n, 10) || 1);
  RVWizard.highlightPipelineStep(pipelineStep);
  if ((n === 3 || n === "3" || n === "3b") && RVWizard.step3) {
    RVWizard.step3.refreshClassificationSummary();
  }
};

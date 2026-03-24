/**
 * wizard/init.js
 * Wires up navigation buttons, form submit animation, and initial state.
 * Must be loaded last — after all step files are loaded.
 *
 * Depends on: state.js, step1_upload.js, step2_classify.js, step3_prompt.js
 */

(function () {
  const showStep = RVWizard.showStep;

  /* ── Navigation buttons ──────────────────────────────────────────── */
  document.getElementById("step1-next").addEventListener("click", () => showStep(2));
  document.getElementById("step1-skip").addEventListener("click", () => showStep(2));
  document.getElementById("step2-back").addEventListener("click", () => showStep(1));
  document.getElementById("step2-next").addEventListener("click", () => {
    RVWizard.step3.populateRichPrompt();
    showStep(3);
  });
  document.getElementById("step3-back").addEventListener("click", () => showStep(2));

  /* ── Form submit: trigger AI Draft animation on pipeline step 4 ── */
  document.getElementById("new_rt_form").addEventListener("submit", () => {
    RVWizard.highlightPipelineStep(4);
    sessionStorage.setItem("rv_wf_from_wizard", "1");
  });

  /* ── Initial state: highlight step 1 ────────────────────────────── */
  RVWizard.highlightPipelineStep(1);
})();

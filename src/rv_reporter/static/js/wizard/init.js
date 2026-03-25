/**
 * wizard/init.js
 * Wires up navigation buttons, form submit animation, and initial state.
 * Must be loaded last — after all step files and reasoning_chat.js.
 *
 * Depends on: state.js, step1_upload.js, step2_classify.js, step3_prompt.js, reasoning_chat.js
 */

(function () {
  const showStep = RVWizard.showStep;

  /* ── Navigation buttons ──────────────────────────────────────────── */
  document.getElementById("step1-next").addEventListener("click", () => {
    showStep("1b");
    /* Fetch AI preset recommendations when entering Step 1b */
    if (RVWizard.step1 && RVWizard.step1.fetchPresetRecommendations) {
      RVWizard.step1.fetchPresetRecommendations();
    }
  });
  document.getElementById("step1-skip").addEventListener("click", () => showStep(2));
  document.getElementById("step1b-back").addEventListener("click", () => showStep(1));
  document.getElementById("step1b-next").addEventListener("click", () => showStep(2));
  document.getElementById("step1b-skip").addEventListener("click", () => showStep(2));

  /* ── Step 1b: AI Transform Chat ──────────────────────────────────── */
  var transformChat = null;
  var lastTransformPandasCode = "";
  var applyTransformBtn = document.getElementById("btn-apply-ai-transform");
  var aiTransformStatus = document.getElementById("ai-transform-status");
  var aiTransformPreviewWrap = document.getElementById("ai-transform-preview-wrap");
  var existingCsvInput = document.getElementById("existing_csv_path");

  /* Auto-start AI transform analysis when step 1b becomes visible */
  (function autoStartTransformChat() {
    var observer = new MutationObserver(function () {
      var step1b = document.getElementById("step-1b");
      if (step1b && !step1b.classList.contains("hidden") && !transformChat) {
        var csvPath = existingCsvInput ? existingCsvInput.value : "";
        if (csvPath) {
          transformChat = ReasoningChat.create({
            containerId: "transform-reasoning-chat",
            onResult: function (data) {
              RVWizard.state._lastTransformPlan = data;
              if (data.pandas_code) {
                lastTransformPandasCode = data.pandas_code;
                if (applyTransformBtn) applyTransformBtn.style.display = "";
              }
            }
          });
          var sheetInput = document.getElementById("sheet_name");
          transformChat.start(WIZARD_CONFIG.transformReasoningUrl, {
            csv_path: csvPath,
            sheet_name: sheetInput ? sheetInput.value : ""
          });
        }
      }
    });
    var wizardForm = document.getElementById("new_rt_form");
    if (wizardForm) observer.observe(wizardForm, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
  })();

  if (applyTransformBtn) {
    applyTransformBtn.addEventListener("click", async function () {
      if (!lastTransformPandasCode || !existingCsvInput || !existingCsvInput.value) return;
      applyTransformBtn.disabled = true;
      if (aiTransformStatus) aiTransformStatus.textContent = "Applying transform…";
      try {
        var sheetInput = document.getElementById("sheet_name");
        var resp = await fetch(WIZARD_CONFIG.transformApplyUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            csv_path: existingCsvInput.value,
            sheet_name: sheetInput ? sheetInput.value : "",
            pandas_code: lastTransformPandasCode
          })
        });
        var result = await resp.json();
        if (!resp.ok || result.error) {
          if (aiTransformStatus) aiTransformStatus.textContent = "Error: " + (result.error || resp.statusText);
          return;
        }
        /* ── KEY: Swap the CSV path to the transformed file ───── */
        RVWizard.state._originalCsvPath = existingCsvInput.value;
        existingCsvInput.value = result.transformed_path;
        /* Reload source metadata with new columns */
        RVWizard.state.sourceColumns = result.columns || [];
        if (aiTransformStatus) {
          aiTransformStatus.textContent = "✅ Applied! " + result.row_count.toLocaleString() + " rows, " + result.columns.length + " columns. Using transformed CSV from now on.";
        }
        /* Render preview table */
        if (aiTransformPreviewWrap && result.sample_rows) {
          _renderAiTransformPreview(result, aiTransformPreviewWrap);
        }
        /* Re-fetch source metadata so step2/step3 see new columns */
        if (RVWizard.step1 && RVWizard.step1.loadSourceMetadata) {
          RVWizard.step1.loadSourceMetadata(result.transformed_path, sheetInput ? sheetInput.value : "");
        }
      } catch (err) {
        if (aiTransformStatus) aiTransformStatus.textContent = "Request failed: " + err;
      } finally {
        applyTransformBtn.disabled = false;
      }
    });
  }

  function _renderAiTransformPreview(result, wrap) {
    var cols = result.columns || [];
    var rows = result.sample_rows || [];
    var html = '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:8px;background:#fff;">';
    html += '<div style="padding:.45rem .75rem;background:#f0fdf4;border-bottom:1px solid var(--border);font-size:.78rem;font-weight:700;color:#065f46;">✅ Transformed Data Preview</div>';
    if (rows.length) {
      html += '<table style="border-collapse:collapse;font-size:.78rem;min-width:100%;">';
      html += '<thead><tr>' + cols.map(function (c) {
        return '<th style="padding:.35rem .6rem;border-bottom:1px solid var(--border);white-space:nowrap;text-align:left;color:var(--muted);font-size:.75rem;">' + c + '</th>';
      }).join('') + '</tr></thead><tbody>';
      rows.slice(0, 8).forEach(function (row) {
        html += '<tr>' + cols.map(function (c) {
          var v = row[c];
          return '<td style="padding:.3rem .6rem;border-bottom:1px solid #f0f0f0;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + (v != null ? String(v) : '') + '</td>';
        }).join('') + '</tr>';
      });
      html += '</tbody></table>';
    }
    html += '<div style="padding:.45rem .75rem;border-top:1px solid var(--border);font-size:.76rem;color:var(--muted);">' + Math.min(rows.length, 8) + ' of ' + result.row_count.toLocaleString() + ' rows</div>';
    html += '</div>';
    wrap.innerHTML = html;
    wrap.style.display = "";
  }

  document.getElementById("step2-back").addEventListener("click", () => showStep("1b"));
  document.getElementById("step2-next").addEventListener("click", () => {
    RVWizard.step3.populateRichPrompt();
    showStep(3);
  });
  document.getElementById("step3-back").addEventListener("click", () => showStep(2));

  /* ── Step 2b: Classification Reasoning ───────────────────────────── */
  var classifyChat = null;

  document.getElementById("step2-reason-btn").addEventListener("click", () => {
    showStep("2b");
    if (!classifyChat) {
      classifyChat = ReasoningChat.create({
        containerId: "classify-reasoning-chat",
        onResult: function (data) {
          /* Store the AI classification so user can apply it */
          RVWizard.state._lastClassifyReasoning = data;
        }
      });
    }
    /* Start (or restart) the reasoning with current columns/filename */
    var cols = RVWizard.state.sourceColumns || [];
    var filename = document.querySelector("input[name='source_file']")?.value ||
                   document.getElementById("source-file-input")?.files?.[0]?.name || "";
    classifyChat.start(WIZARD_CONFIG.classifyReasoningUrl, {
      columns: cols,
      filename: filename
    });
  });

  document.getElementById("step2b-back").addEventListener("click", () => showStep(2));

  document.getElementById("step2b-apply-btn").addEventListener("click", () => {
    /* Apply AI classification to the wizard selections */
    var data = RVWizard.state._lastClassifyReasoning;
    if (data && RVWizard.step2 && RVWizard.step2.applyRecommendation) {
      RVWizard.step2.applyRecommendation(data.domain, data.family, data.mode);
    }
    showStep(2);
  });

  document.getElementById("step2b-next").addEventListener("click", () => {
    RVWizard.step3.populateRichPrompt();
    showStep(3);
  });

  /* ── Step 3b: Report Blueprint ───────────────────────────────────── */
  var blueprintChat = null;

  document.getElementById("step3-next").addEventListener("click", () => {
    showStep("3b");
    if (!blueprintChat) {
      blueprintChat = ReasoningChat.create({
        containerId: "blueprint-reasoning-chat",
        onResult: function (data) {
          RVWizard.state._lastBlueprint = data;
          /* Store blueprint text in hidden field for form submission */
          var hidden = document.getElementById("blueprint-json");
          if (!hidden) {
            hidden = document.createElement("input");
            hidden.type = "hidden";
            hidden.name = "blueprint_json";
            hidden.id = "blueprint-json";
            document.getElementById("new_rt_form").appendChild(hidden);
          }
          hidden.value = JSON.stringify(data);
        }
      });
    }
    /* Start blueprint generation with current prompt + classification */
    var promptText = document.getElementById("prompt_text")?.value || "";
    var cols = RVWizard.state.sourceColumns || [];
    blueprintChat.start(WIZARD_CONFIG.blueprintUrl, {
      prompt_text: promptText,
      columns: cols,
      domain: RVWizard.state.selectedDomain,
      family: RVWizard.state.selectedFamily,
      mode:   RVWizard.state.selectedMode
    });
  });

  document.getElementById("step3b-back").addEventListener("click", () => showStep(3));

  document.getElementById("step3b-generate").addEventListener("click", () => {
    /* Submit the form with blueprint attached */
    RVWizard.highlightPipelineStep(4);
    sessionStorage.setItem("rv_wf_from_wizard", "1");
    document.getElementById("new_rt_form").submit();
  });

  /* ── Form submit: trigger AI Draft animation on pipeline step 4 ── */
  document.getElementById("new_rt_form").addEventListener("submit", () => {
    RVWizard.highlightPipelineStep(4);
    sessionStorage.setItem("rv_wf_from_wizard", "1");
  });

  /* ── Initial state: highlight step 1 ────────────────────────────── */
  RVWizard.highlightPipelineStep(1);
})();

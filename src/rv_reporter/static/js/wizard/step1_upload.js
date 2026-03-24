/**
 * wizard/step1_upload.js
 * Step 1 — Upload Source Data
 *
 * Responsibilities:
 *   - Toggle between "new file" and "recent file" upload modes
 *   - Fetch source metadata (columns, row count, sheets) from the server
 *   - Populate the sheet selector and source summary line
 *   - Store discovered columns in RVWizard.state.sourceColumns for use in step 3
 *
 * Depends on: state.js (RVWizard.state)
 * Config keys consumed: WIZARD_CONFIG.sourceMetadataUrl, WIZARD_CONFIG.sheetOptions,
 *                       WIZARD_CONFIG.selectedSheet
 */

window.RVWizard = window.RVWizard || {};

RVWizard.step1 = (function () {

  /* ── DOM refs ──────────────────────────────────────────────────────── */
  const uploadMode    = document.getElementById("upload_mode");
  const newUploadRow  = document.getElementById("new_upload_row");
  const recentRow     = document.getElementById("recent_upload_row");
  const uploadInput   = document.getElementById("csv_upload");
  const recentSelect  = document.getElementById("recent_upload_select");
  const existingInput = document.getElementById("existing_csv_path");
  const sheetWrap     = document.getElementById("sheet_wrap");
  const sheetInput    = document.getElementById("sheet_name");
  const sourceSummary = document.getElementById("source_summary");
  const sourceTotalLines = document.getElementById("source_total_lines");
  const samplePercentSummary = document.getElementById("sample_percent_summary");
  const samplePercentInputs = Array.from(document.querySelectorAll("input[name='source_sample_percent']"));

  function selectedSamplePercent() {
    const active = samplePercentInputs.find((input) => input.checked);
    return active ? Number(active.value || "100") : 100;
  }

  function refreshSamplePercentSummary() {
    if (!samplePercentSummary) return;
    const percent = selectedSamplePercent();
    const totalRows = RVWizard.state.sourceRowCount;
    if (Number.isFinite(totalRows) && totalRows > 0) {
      const sampledRows = Math.max(1, Math.ceil(totalRows * (percent / 100)));
      samplePercentSummary.textContent = `Using ${percent}% of the source for AI drafting (${sampledRows.toLocaleString()} of ${totalRows.toLocaleString()} rows).`;
      return;
    }
    samplePercentSummary.textContent = `Using ${percent}% of the source for AI drafting.`;
  }

  /* ── Sheet selector ──────────────────────────────────────────────── */
  function setSheetOptions(sheets, selectedSheet) {
    sheetInput.innerHTML = "";
    if (!sheets || !sheets.length) {
      sheetInput.innerHTML = '<option value="">Auto / not needed</option>';
      sheetInput.disabled = true;
      sheetWrap && sheetWrap.classList.add("field-disabled");
      return;
    }
    sheetWrap && sheetWrap.classList.remove("field-disabled");
    sheetInput.disabled = false;
    sheetInput.innerHTML = '<option value="">Choose sheet...</option>';
    sheets.forEach(s => {
      const o = document.createElement("option");
      o.value = s;
      o.textContent = s;
      if (selectedSheet && selectedSheet === s) o.selected = true;
      sheetInput.appendChild(o);
    });
  }

  /* ── Source summary line ─────────────────────────────────────────── */
  function renderSummary(profile) {
    if (!sourceSummary) return;
    if (!profile) {
      sourceSummary.textContent = "";
      RVWizard.state.sourceRowCount = null;
      if (sourceTotalLines) sourceTotalLines.textContent = "-";
      refreshSamplePercentSummary();
      return;
    }
    const parts = [];
    if (profile.file_type)                parts.push(profile.file_type.toUpperCase());
    if (profile.row_count != null)        parts.push(profile.row_count + " rows");
    if (profile.columns && profile.columns.length) parts.push(profile.columns.length + " columns");
    if (profile.selected_sheet)           parts.push("sheet: " + profile.selected_sheet);
    sourceSummary.textContent = parts.join(" • ");
    RVWizard.state.sourceRowCount = Number.isFinite(profile.row_count) ? profile.row_count : null;
    if (sourceTotalLines) {
      sourceTotalLines.textContent = profile.row_count != null ? Number(profile.row_count).toLocaleString() : "-";
    }
    refreshSamplePercentSummary();
  }

  /* ── Fetch metadata from server ──────────────────────────────────── */
  async function loadSourceMetadata(path, sheet) {
    if (!path) { renderSummary(null); setSheetOptions([], ""); return; }
    try {
      const url = window.WIZARD_CONFIG.sourceMetadataUrl
        + "?path=" + encodeURIComponent(path)
        + "&sheet_name=" + encodeURIComponent(sheet || "");
      const resp = await fetch(url);
      if (!resp.ok) return;
      const profile = await resp.json();
      renderSummary(profile);
      setSheetOptions(profile.sheets || [], profile.selected_sheet || sheet || "");
      RVWizard.state.sourceColumns = profile.columns || [];
      // Ask server to recommend classification based on columns
      const fname = path.split(/[\\/]/).pop() || "";
      fetchClassificationRecommendation(profile.columns || [], fname);
    } catch (_) { /* non-fatal */ }
  }

  /* ── Classification recommendation ──────────────────────────────── */
  async function fetchClassificationRecommendation(columns, filename) {
    const banner = document.getElementById("classification-recommendation");
    const chips  = document.getElementById("rec-chips");
    const applyBtn   = document.getElementById("rec-apply-btn");
    const dismissBtn = document.getElementById("rec-dismiss-btn");
    if (!banner || !chips || !columns.length) return;

    try {
      const resp = await fetch("/api/recommend-classification", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ columns, filename: filename || "" }),
      });
      if (!resp.ok) return;
      const rec = await resp.json();
      if (rec.confidence === "none") return;

      const labels = RVWizard.data || {};
      chips.innerHTML = "";
      if (rec.domain) {
        const c = document.createElement("span");
        c.className = "cs-chip";
        c.textContent = rec.domain.replace(/_/g, " ");
        c.style.cssText = "background:#e0edff;color:#184392;padding:.15rem .5rem;border-radius:6px;font-weight:600;font-size:.78rem;";
        chips.appendChild(c);
      }
      if (rec.family) {
        const c = document.createElement("span");
        c.className = "cs-chip";
        c.textContent = (labels.FAMILY_LABELS || {})[rec.family] || rec.family.replace(/_/g, " ");
        c.style.cssText = "background:#e0edff;color:#184392;padding:.15rem .5rem;border-radius:6px;font-weight:600;font-size:.78rem;";
        chips.appendChild(c);
      }
      if (rec.mode) {
        const c = document.createElement("span");
        c.className = "cs-chip";
        c.textContent = (labels.MODE_LABELS || {})[rec.mode] || rec.mode.replace(/_/g, " ");
        c.style.cssText = "background:#e0edff;color:#184392;padding:.15rem .5rem;border-radius:6px;font-weight:600;font-size:.78rem;";
        chips.appendChild(c);
      }

      // Store recommendation for the Apply button
      banner._rec = rec;
      banner.classList.remove("hidden");
      banner.style.display = "flex";

      applyBtn.onclick = function () {
        if (RVWizard.step2 && RVWizard.step2.applyRecommendation) {
          RVWizard.step2.applyRecommendation(rec.domain, rec.family, rec.mode);
        }
        banner.classList.add("hidden");
        banner.style.display = "none";
      };
      dismissBtn.onclick = function () {
        banner.classList.add("hidden");
        banner.style.display = "none";
      };
    } catch (_) { /* non-fatal */ }
  }

  /* ── Upload mode toggle ──────────────────────────────────────────── */
  function syncUploadModeUi() {
    const isRecent = uploadMode.value === "recent";
    newUploadRow && newUploadRow.classList.toggle("hidden", isRecent);
    recentRow    && recentRow.classList.toggle("hidden", !isRecent);
    if (uploadInput)  uploadInput.disabled  = isRecent;
    if (recentSelect) recentSelect.disabled = !isRecent;
    if (!isRecent && recentSelect) {
      recentSelect.value = "";
      existingInput.value = "";
    }
  }

  /* ── Event listeners ─────────────────────────────────────────────── */
  uploadMode.addEventListener("change", syncUploadModeUi);

  uploadInput && uploadInput.addEventListener("change", async () => {
    const file = uploadInput.files && uploadInput.files[0];
    if (!file) { renderSummary(null); setSheetOptions([], ""); return; }
    try {
      const fd = new FormData();
      fd.append("file", file);
      const resp = await fetch("/api/upload-source-metadata", { method: "POST", body: fd });
      if (!resp.ok) return;
      const profile = await resp.json();
      if (profile.error) return;
      existingInput.value = profile.path || "";
      renderSummary(profile);
      setSheetOptions(profile.sheets || [], profile.selected_sheet || "");
      RVWizard.state.sourceColumns = profile.columns || [];
      fetchClassificationRecommendation(profile.columns || [], file.name || "");
    } catch (_) { /* non-fatal */ }
  });

  recentSelect && recentSelect.addEventListener("change", () => {
    existingInput.value = recentSelect.value || "";
    if (!existingInput.value) { renderSummary(null); setSheetOptions([], ""); return; }
    loadSourceMetadata(existingInput.value, sheetInput.value || "");
  });

  sheetInput && sheetInput.addEventListener("change", () => {
    if (existingInput.value) loadSourceMetadata(existingInput.value, sheetInput.value || "");
  });
  samplePercentInputs.forEach((input) => {
    input.addEventListener("change", refreshSamplePercentSummary);
  });

  /* ── Init ────────────────────────────────────────────────────────── */
  syncUploadModeUi();
  if (existingInput.value) {
    loadSourceMetadata(existingInput.value, sheetInput.value || "");
  } else {
    setSheetOptions(
      window.WIZARD_CONFIG.sheetOptions || [],
      window.WIZARD_CONFIG.selectedSheet || ""
    );
  }
  refreshSamplePercentSummary();

  return { loadSourceMetadata };
})();

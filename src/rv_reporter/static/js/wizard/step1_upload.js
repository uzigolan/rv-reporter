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
        c.textContent = (labels.DOMAIN_LABELS || {})[rec.domain] || rec.domain.replace(/_/g, " ");
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

  /* ── Reshape: parse columns from textarea text ───────────────── */
  function parseSourceStructureText(text) {
    const raw = (text || "").trim();
    if (!raw) return null;
    let lines = raw.split(/\n/).map((l) => l.trim()).filter(Boolean);
    if (lines.length === 1 && lines[0].includes(",")) {
      lines = lines[0].split(",").map((l) => l.trim()).filter(Boolean);
    }
    const columns = [];
    const notes = {};
    for (const rawLine of lines) {
      const line = rawLine.replace(/^[-*\u2022\d.)\s]+/, "").trim();
      if (!line) continue;
      const colonIdx = line.indexOf(":");
      const col = colonIdx !== -1 ? line.slice(0, colonIdx).trim() : line;
      const note = colonIdx !== -1 ? line.slice(colonIdx + 1).trim() : "";
      if (!col || columns.includes(col)) continue;
      columns.push(col);
      if (note) notes[col] = note;
    }
    return columns.length ? { raw_text: raw, columns, notes } : null;
  }

  /* ── Task 2: Re-classify when reshape columns change ─────────── */
  const sourceStructureTextarea = document.getElementById("source_structure_text");
  let reshapeDebounceTimer = null;
  sourceStructureTextarea && sourceStructureTextarea.addEventListener("input", () => {
    clearTimeout(reshapeDebounceTimer);
    reshapeDebounceTimer = setTimeout(() => {
      const plan = parseSourceStructureText(sourceStructureTextarea.value);
      if (!plan || plan.columns.length < 2) return;
      const proposedCols = plan.columns.map((c) => c.toLowerCase());
      const fname = existingInput.value ? (existingInput.value.split(/[\\/]/).pop() || "") : "";
      fetchClassificationRecommendation(proposedCols, fname);
    }, 800);
  });

  /* ── Task 3: Preset schema buttons ───────────────────────────── */
  document.querySelectorAll(".schema-preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const presetId = btn.dataset.preset;
      const presets = ((RVWizard.data || {}).SCHEMA_PRESETS) || [];
      const preset = presets.find((p) => p.id === presetId);
      if (!preset || !sourceStructureTextarea) return;
      sourceStructureTextarea.value = preset.columns;
      // Trigger debounced classification
      clearTimeout(reshapeDebounceTimer);
      const proposedCols = preset.columns.split(/\n/).map((line) =>
        line.replace(/^[-*\u2022\d.)\s]+/, "").split(":")[0].trim().toLowerCase()
      ).filter(Boolean);
      if (proposedCols.length >= 2) {
        const fname = existingInput.value ? (existingInput.value.split(/[\\/]/).pop() || "") : "";
        fetchClassificationRecommendation(proposedCols, fname);
      }
    });
  });

  /* ── Task 1: Transform Preview ───────────────────────────────── */
  const previewTransformBtn = document.getElementById("btn-preview-transform");
  const transformPreviewWrap = document.getElementById("transform-preview-wrap");
  const transformPreviewStatus = document.getElementById("transform-preview-status");

  previewTransformBtn && previewTransformBtn.addEventListener("click", async () => {
    const sourcePath = existingInput.value;
    const structText = sourceStructureTextarea ? sourceStructureTextarea.value.trim() : "";
    if (!sourcePath) {
      if (transformPreviewStatus) transformPreviewStatus.textContent = "Upload a source file first.";
      return;
    }
    if (!structText) {
      if (transformPreviewStatus) transformPreviewStatus.textContent = "Enter columns in the Reshape panel first.";
      return;
    }
    const plan = parseSourceStructureText(structText);
    if (!plan) {
      if (transformPreviewStatus) transformPreviewStatus.textContent = "Could not parse any columns.";
      return;
    }
    if (transformPreviewStatus) transformPreviewStatus.textContent = "Transforming…";
    previewTransformBtn.disabled = true;
    try {
      const resp = await fetch("/api/transform-source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_path: sourcePath, source_structure_plan: plan }),
      });
      const result = await resp.json();
      if (!resp.ok || result.error) {
        if (transformPreviewStatus) transformPreviewStatus.textContent = "Error: " + (result.error || resp.statusText);
        return;
      }
      if (transformPreviewStatus) transformPreviewStatus.textContent =
        result.row_count.toLocaleString() + " rows → " + result.columns.length + " columns";
      _renderTransformPreview(result);
    } catch (_) {
      if (transformPreviewStatus) transformPreviewStatus.textContent = "Request failed.";
    } finally {
      previewTransformBtn.disabled = false;
    }
  });

  function _renderTransformPreview(result) {
    if (!transformPreviewWrap) return;
    const cols = result.columns || [];
    const rows = result.preview_rows || [];
    const conf = result.column_confidence || {};
    const badgeStyle = {
      direct:  "background:#d1fae5;color:#065f46;",
      derived: "background:#e0edff;color:#184392;",
      missing: "background:#fee2e2;color:#7f1d1d;",
    };
    let html = `<div style="overflow-x:auto;border:1px solid var(--border);border-radius:8px;background:#fff;">`;
    html += `<div style="padding:.45rem .75rem;background:#f8f9fa;border-bottom:1px solid var(--border);display:flex;gap:.4rem;flex-wrap:wrap;align-items:center;font-size:.76rem;">`;
    for (const col of cols) {
      const c = conf[col] || "missing";
      html += `<span style="${badgeStyle[c] || ""}padding:.15rem .45rem;border-radius:4px;font-weight:600;" title="${c}">${col}</span>`;
    }
    html += `</div>`;
    if (rows.length) {
      html += `<table style="border-collapse:collapse;font-size:.78rem;min-width:100%;">`;
      html += `<thead><tr>` + cols.map((c) =>
        `<th style="padding:.35rem .6rem;border-bottom:1px solid var(--border);white-space:nowrap;text-align:left;color:var(--muted);font-size:.75rem;">${c}</th>`
      ).join("") + `</tr></thead><tbody>`;
      for (const row of rows.slice(0, 8)) {
        html += `<tr>` + cols.map((c) => {
          const v = row[c];
          return `<td style="padding:.3rem .6rem;border-bottom:1px solid #f0f0f0;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${v != null ? String(v) : ""}</td>`;
        }).join("") + `</tr>`;
      }
      html += `</tbody></table>`;
    }
    html += `<div style="padding:.45rem .75rem;border-top:1px solid var(--border);font-size:.76rem;display:flex;gap:1rem;align-items:center;color:var(--muted);">`;
    html += `Showing ${Math.min(rows.length, 8)} of ${result.row_count.toLocaleString()} rows`;
    html += ` &nbsp;·&nbsp; <a href="/api/download-transformed?path=${encodeURIComponent(result.transformed_path)}" download style="color:#184392;font-weight:600;">⬇ Download transformed CSV</a>`;
    html += `</div></div>`;
    transformPreviewWrap.innerHTML = html;
    transformPreviewWrap.style.display = "";
  }

  /* ── Preset Recommendations ──────────────────────────────────────── */
  async function fetchPresetRecommendations() {
    const cols = RVWizard.state.sourceColumns || [];
    if (!cols.length) return;
    const container = document.getElementById("preset-recommendations-wrap");
    if (!container) return;

    try {
      const resp = await fetch("/api/recommend-preset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ columns: cols }),
      });
      if (!resp.ok) return;
      const data = await resp.json();
      const presets = data.presets || [];
      if (!presets.length) {
        container.innerHTML = `<div class="small muted">No preset matched your columns. Browse all presets below.</div>`;
        return;
      }

      let html = `<div style="margin-bottom:.75rem;"><div class="small" style="font-weight:700;margin-bottom:.35rem;color:var(--brand);">✨ AI-Recommended Presets</div>`;
      for (const preset of presets) {
        const confColor = preset.confidence === "high" ? "#059669" : preset.confidence === "medium" ? "#d97706" : "#6b7280";
        html += `<button type="button" class="schema-preset-btn" data-preset="${preset.id}" `;
        html += `style="font-size:.76rem; padding:.4rem .7rem; margin-right:.4rem; margin-bottom:.4rem; border:2px solid ${confColor}; border-radius:6px; background:#fff; cursor:pointer; color:var(--text); font-weight:600;">`;
        html += `${preset.label} `;
        html += `<span style="opacity:.6; padding-left:.3rem;">(${preset.confidence})</span>`;
        html += `</button>`;
      }
      html += `</div>`;
      container.innerHTML = html;
      container.style.display = "";

      // Re-wire preset buttons
      container.querySelectorAll(".schema-preset-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const presetId = btn.dataset.preset;
          const presets_data = ((RVWizard.data || {}).SCHEMA_PRESETS) || [];
          const preset = presets_data.find((p) => p.id === presetId);
          if (!preset || !sourceStructureTextarea) return;
          sourceStructureTextarea.value = preset.columns;
          sourceStructureTextarea.focus();
        });
      });
    } catch (_) { /* non-fatal */ }
  }

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

  return { loadSourceMetadata, fetchPresetRecommendations };
})();

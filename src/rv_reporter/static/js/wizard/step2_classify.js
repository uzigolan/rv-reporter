/**
 * wizard/step2_classify.js
 * Step 2 — Classify Your Report
 *
 * Responsibilities:
 *   - Render category, domain, family, and mode tile buttons
 *   - Track selections in RVWizard.state (selectedDomain, selectedFamily, selectedMode)
 *   - Sync hidden form inputs: hint_domain, hint_family, hint_mode
 *
 * Depends on: data.js (RVWizard.data), state.js (RVWizard.state)
 * Config keys consumed: WIZARD_CONFIG.modes (server-side full mode list for "generic" domain)
 */

window.RVWizard = window.RVWizard || {};

RVWizard.step2 = (function () {
  const data  = RVWizard.data;
  const state = RVWizard.state;

  function domainLabel(domain) {
    return (data.DOMAIN_LABELS || {})[domain] || domain.replace(/_/g, " ");
  }

  /* ── Tile factory ────────────────────────────────────────────────── */
  function makeTile(value, label, cssClass) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = cssClass;
    btn.dataset.value = value;
    btn.textContent = label;
    return btn;
  }

  function setActiveTile(container, value) {
    container.querySelectorAll("button").forEach(b =>
      b.classList.toggle("active", b.dataset.value === value)
    );
  }

  /* ── Reset downstream selections ────────────────────────────────── */
  function resetBelowDomain() {
    state.selectedFamily = "";
    state.selectedMode   = "";
    document.getElementById("hint_family").value = "";
    document.getElementById("hint_mode").value   = "";
    document.getElementById("family-row").classList.add("hidden");
    document.getElementById("mode-row").classList.add("hidden");
  }

  function resetBelowFamily() {
    state.selectedMode = "";
    document.getElementById("hint_mode").value = "";
    document.getElementById("mode-row").classList.add("hidden");
  }

  /* ── Category tiles ──────────────────────────────────────────────── */
  document.getElementById("category-tiles").addEventListener("click", e => {
    const tile = e.target.closest(".cat-tile");
    if (!tile) return;

    setActiveTile(document.getElementById("category-tiles"), tile.dataset.cat);
    state.selectedDomain = "";
    document.getElementById("hint_domain").value = "";
    resetBelowDomain();

    const domains = tile.dataset.domains.split(",").filter(Boolean);
    const domainTilesEl = document.getElementById("domain-tiles");
    domainTilesEl.innerHTML = "";
    domains.forEach(d => {
      domainTilesEl.appendChild(makeTile(d, domainLabel(d), "domain-tile"));
    });
    document.getElementById("domain-row").classList.remove("hidden");
  });

  /* ── Domain tiles ────────────────────────────────────────────────── */
  document.getElementById("domain-tiles").addEventListener("click", e => {
    const tile = e.target.closest(".domain-tile");
    if (!tile) return;

    state.selectedDomain = tile.dataset.value;
    document.getElementById("hint_domain").value = state.selectedDomain;
    setActiveTile(document.getElementById("domain-tiles"), state.selectedDomain);
    resetBelowDomain();

    const families = data.DOMAIN_FAMILY_MAP[state.selectedDomain] || [];
    const familyTilesEl = document.getElementById("family-tiles");
    familyTilesEl.innerHTML = "";
    families.forEach(f => {
      familyTilesEl.appendChild(makeTile(f, data.FAMILY_LABELS[f] || f, "family-tile"));
    });
    document.getElementById("family-row").classList.remove("hidden");
  });

  /* ── Family tiles ────────────────────────────────────────────────── */
  document.getElementById("family-tiles").addEventListener("click", e => {
    const tile = e.target.closest(".family-tile");
    if (!tile) return;

    state.selectedFamily = tile.dataset.value;
    document.getElementById("hint_family").value = state.selectedFamily;
    setActiveTile(document.getElementById("family-tiles"), state.selectedFamily);
    resetBelowFamily();

    // "generic" domain uses the full server-supplied mode list; others use the curated map
    const modes = (state.selectedDomain === "generic")
      ? (window.WIZARD_CONFIG.modes || [])
      : (data.DOMAIN_MODE_MAP[state.selectedDomain] || window.WIZARD_CONFIG.modes || []);

    const modeTilesEl = document.getElementById("mode-tiles");
    modeTilesEl.innerHTML = "";
    modes.forEach(m => {
      modeTilesEl.appendChild(makeTile(m, data.MODE_LABELS[m] || m, "mode-tile"));
    });
    document.getElementById("mode-row").classList.remove("hidden");
  });

  /* ── Mode tiles ──────────────────────────────────────────────────── */
  document.getElementById("mode-tiles").addEventListener("click", e => {
    const tile = e.target.closest(".mode-tile");
    if (!tile) return;
    state.selectedMode = tile.dataset.value;
    document.getElementById("hint_mode").value = state.selectedMode;
    setActiveTile(document.getElementById("mode-tiles"), state.selectedMode);
  });

  /* ── Programmatic selection (used by recommendation engine) ──────── */
  function applyRecommendation(domain, family, mode) {
    if (!domain) return;

    // 1. Find the category tile whose data-domains contains this domain
    const catTiles = document.querySelectorAll("#category-tiles .cat-tile");
    let matchedCat = null;
    catTiles.forEach(tile => {
      const domains = (tile.dataset.domains || "").split(",");
      if (domains.includes(domain)) matchedCat = tile;
    });
    if (matchedCat) {
      // Simulate category click
      setActiveTile(document.getElementById("category-tiles"), matchedCat.dataset.cat);
      const domains = matchedCat.dataset.domains.split(",").filter(Boolean);
      const domainTilesEl = document.getElementById("domain-tiles");
      domainTilesEl.innerHTML = "";
      domains.forEach(d => {
        domainTilesEl.appendChild(makeTile(d, domainLabel(d), "domain-tile"));
      });
      document.getElementById("domain-row").classList.remove("hidden");
    }

    // 2. Select domain
    state.selectedDomain = domain;
    document.getElementById("hint_domain").value = domain;
    setActiveTile(document.getElementById("domain-tiles"), domain);

    // 3. Populate and select family
    if (family) {
      const families = data.DOMAIN_FAMILY_MAP[domain] || [];
      const familyTilesEl = document.getElementById("family-tiles");
      familyTilesEl.innerHTML = "";
      families.forEach(f => {
        familyTilesEl.appendChild(makeTile(f, data.FAMILY_LABELS[f] || f, "family-tile"));
      });
      document.getElementById("family-row").classList.remove("hidden");
      state.selectedFamily = family;
      document.getElementById("hint_family").value = family;
      setActiveTile(document.getElementById("family-tiles"), family);
    }

    // 4. Populate and select mode
    if (mode) {
      const modes = (domain === "generic")
        ? (window.WIZARD_CONFIG.modes || [])
        : (data.DOMAIN_MODE_MAP[domain] || window.WIZARD_CONFIG.modes || []);
      const modeTilesEl = document.getElementById("mode-tiles");
      modeTilesEl.innerHTML = "";
      modes.forEach(m => {
        modeTilesEl.appendChild(makeTile(m, data.MODE_LABELS[m] || m, "mode-tile"));
      });
      document.getElementById("mode-row").classList.remove("hidden");
      state.selectedMode = mode;
      document.getElementById("hint_mode").value = mode;
      setActiveTile(document.getElementById("mode-tiles"), mode);
    }
  }

  return { applyRecommendation };
})();

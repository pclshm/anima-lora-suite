/* Anima LoRA Editor — frontend logic
   ─────────────────────────────────────────────────────────────────────────── */

const NUM_BLOCKS = 28;

// Global state
const state = {
  presets: {},                    // name -> [block indices] | null (Custom)
  // The fields below mirror the *active* layer so the existing single-LoRA code
  // paths keep working unchanged; switching layers swaps them in/out.
  impact: null,                   // null until first inspect
  blocksPresent: new Set(),       // blocks actually present in active LoRA
  inspected: false,
  previewReady: false,            // real Anima backend available? (gates BG generate)
  lastPreviewImage: null,         // data-URI of the most recent sample (for "use as background")
  // ─ LoRA layers ─ each is one LoRA with its own block config; preview & save
  // combine them all. The active layer drives the grid/config below.
  layers: [],                     // [{ id, name, path, inspected, inspectData, config, preset }]
  activeId: null,
  nextId: 1,
};

// ─── DOM helpers ────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const blockToggle = (i) => $(`block-${i}-toggle`);
const blockStrength = (i) => $(`block-${i}-strength`);
const blockStrengthVal = (i) => $(`block-${i}-strength-val`);
const blockCell = (i) => $(`block-${i}`);


// ─── Toast ──────────────────────────────────────────────────────────────────
let toastTimer = null;
function toast(msg, kind = '') {
  const el = $('toast');
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
}


// ─── Path inputs: quote-stripping + reusable cache ───────────────────────────
// Every filesystem-path field (LoRA in/out + the preview model paths) shares
// this machinery: pasted "Copy as path" quotes are stripped, and each distinct
// path is remembered per field so it can be re-picked from the input's
// <datalist> instead of pasted again. History lives apart from the per-field
// "current value" stores so clearing one never touches the other.
const PATHS_STORE = 'anima-path-history';
const PATHS_MAX = 8;
const PATH_FIELDS = [
  { key: 'lora_in',  input: 'lora-path',       list: 'dl-lora-path' },
  { key: 'lora_out', input: 'output-path',     list: 'dl-output-path' },
  { key: 'dit',      input: 'pv-path-dit',     list: 'dl-pv-path-dit' },
  { key: 'vae',      input: 'pv-path-vae',     list: 'dl-pv-path-vae' },
  { key: 'te',       input: 'pv-path-te',      list: 'dl-pv-path-te' },
];

// Strip whitespace and the surrounding quotes Windows "Copy as path" adds, so a
// pasted "G:\...\model.safetensors" resolves instead of looking set but failing.
function cleanPath(v) {
  let p = (v || '').trim();
  if (p.length >= 2 && p[0] === p[p.length - 1] && (p[0] === '"' || p[0] === "'")) {
    p = p.slice(1, -1).trim();
  }
  return p;
}

// Clean an input element in place; returns true if the displayed value changed.
function cleanPathField(el) {
  if (!el) return false;
  const cleaned = cleanPath(el.value);
  if (cleaned === el.value) return false;
  el.value = cleaned;
  return true;
}

// Read a path input, stripping quotes and reflecting the cleaned value back.
// Used at submit sites so even Enter-to-submit (no blur) never sends quotes.
function pathVal(id) {
  const el = $(id);
  if (!el) return '';
  cleanPathField(el);
  return el.value;
}

function loadPathHistory() {
  let h;
  try { h = JSON.parse(localStorage.getItem(PATHS_STORE) || 'null'); } catch { h = null; }
  return h && typeof h === 'object' ? h : {};
}

function rememberPath(key, value) {
  const v = cleanPath(value);
  if (!v) return;
  const hist = loadPathHistory();
  // Windows paths are case-insensitive, so dedupe on lowercase but keep the
  // newest casing, and move a reused path back to the front (most-recent-first).
  const prior = (hist[key] || []).filter((p) => p.toLowerCase() !== v.toLowerCase());
  hist[key] = [v, ...prior].slice(0, PATHS_MAX);
  try { localStorage.setItem(PATHS_STORE, JSON.stringify(hist)); } catch { /* private mode */ }
  renderPathOptions(key, hist[key]);
}

function renderPathOptions(key, items) {
  const meta = PATH_FIELDS.find((f) => f.key === key);
  if (!meta) return;
  const dl = $(meta.list);
  if (dl) {
    dl.replaceChildren(...(items || []).map((p) => {
      const opt = document.createElement('option');
      opt.value = p;
      return opt;
    }));
  }
  // The per-field "clear saved" control only matters when there's history.
  const clearBtn = document.querySelector(`[data-clear-path="${key}"]`);
  if (clearBtn) clearBtn.hidden = !(items && items.length);
}

function renderPathDatalists() {
  const hist = loadPathHistory();
  PATH_FIELDS.forEach(({ key }) => renderPathOptions(key, hist[key]));
}

// Selectively forget one field's saved paths — leaves every other field intact.
function clearPathHistory(key) {
  const hist = loadPathHistory();
  if (!hist[key] || !hist[key].length) return;
  delete hist[key];
  try { localStorage.setItem(PATHS_STORE, JSON.stringify(hist)); } catch { /* private mode */ }
  renderPathOptions(key, []);
  toast('cleared saved paths', 'success');
}

// Wire every "clear saved" button + render datalists from existing history.
function initPathCache() {
  document.querySelectorAll('[data-clear-path]').forEach((btn) => {
    btn.addEventListener('click', () => clearPathHistory(btn.dataset.clearPath));
  });
  renderPathDatalists();
}


// ─── Bootstrap ──────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  buildBlockGrid();
  initBlockLabels();
  wireGlobalControls();
  wireViewToggle();
  wireThemeSwitch();
  wireImmersiveToggle();
  initBackgrounds();
  initPathCache();
  await loadPresets();
  initLayers();
  await checkHealth();
  initPreview();
  initI2L();
  initQuickDock();
  wireCompressPanel();
});


// ─── LoRA layers ──────────────────────────────────────────────────────────────
// Each layer is one LoRA with its own block/edit config. The single grid + config
// panel below always edits the *active* layer; switching captures the current
// layer's UI into its object and rehydrates the target's. Preview and Save
// combine every layer (the backend merges the whole stack).

function defaultLayerConfig() {
  return {
    enabled_blocks: Array.from({ length: NUM_BLOCKS }, (_, i) => i),
    block_strengths: {},
    llm_adapter_enabled: true,
    llm_adapter_strength: 1.0,
    other_enabled: true,
    other_strength: 1.0,
    global_strength: 1.0,
  };
}

function baseName(p) {
  const n = (p || '').split(/[\\/]/).pop() || '';
  return n.replace(/\.safetensors$/i, '');
}

function makeLayer(path = '') {
  return {
    id: state.nextId++,
    name: path ? baseName(path) : '',
    path,
    inspected: false,
    inspectData: null,
    inspectedFor: null,   // the path we last inspected (dedupes auto-inspect)
    config: defaultLayerConfig(),
    preset: 'All Blocks',
    labels: [],           // [{ id, text, color, blocks:[int] }] — block annotations
  };
}

function activeLayer() {
  return state.layers.find((l) => l.id === state.activeId) || null;
}

function layerLabel(layer, idx) {
  return layer.name || `layer ${idx + 1}`;
}

function initLayers() {
  $('layer-add').addEventListener('click', () => addLayer());
  // Seed with a single layer that adopts the current (default) UI state.
  state.layers = [makeLayer('')];
  state.activeId = state.layers[0].id;
  renderLayerBar();
}

// Capture the live grid/config + path into the active layer object.
function syncActiveLayer() {
  const L = activeLayer();
  if (!L) return;
  L.path = pathVal('lora-path');
  L.config = collectEditConfig();
  L.preset = $('preset-select').value;
  L.name = L.path ? baseName(L.path) : '';
  L.labels = currentLabels;
}

// Rehydrate the shared grid/config UI from a layer's stored state.
function applyLayerToUI(L) {
  $('lora-path').value = L.path || '';

  const enabled = new Set(L.config.enabled_blocks || []);
  const bs = L.config.block_strengths || {};
  suppressCustom = true;
  for (let i = 0; i < NUM_BLOCKS; i++) {
    const on = enabled.has(i);
    blockToggle(i).checked = on;
    const sv = bs[i] !== undefined ? bs[i] : 1.0;
    blockStrength(i).value = sv;
    blockStrengthVal(i).textContent = (+sv).toFixed(2);
    blockCell(i).classList.toggle('disabled', !on);
  }
  $('llm-enabled').checked = !!L.config.llm_adapter_enabled;
  setRangeValue('llm-strength', 'llm-strength-val', L.config.llm_adapter_strength);
  $('other-enabled').checked = !!L.config.other_enabled;
  setRangeValue('other-strength', 'other-strength-val', L.config.other_strength);
  setRangeValue('global-strength', 'global-strength-val', L.config.global_strength);
  if (Array.from($('preset-select').options).some((o) => o.value === L.preset)) {
    $('preset-select').value = L.preset;
  }
  suppressCustom = false;

  // Restore this layer's inspection (impact meters, absent marks) or clear it.
  if (L.inspected && L.inspectData) applyInspectResult(L.inspectData);
  else resetInspectUI();

  // Restore this layer's block labels (from the layer, else by its file path).
  hydrateLabelsForActiveLayer();

  updateActiveLayerLabels();

  // A layer with a path we haven't analysed yet (e.g. switching to it) should
  // populate the AnimaBlock grid on its own.
  maybeAutoInspect();
}

function setRangeValue(rangeId, outId, value) {
  const r = $(rangeId);
  const v = value === undefined || value === null ? +r.value : value;
  r.value = v;
  const o = $(outId);
  if (o) o.textContent = (+v).toFixed(2);
}

// Reset the inspect panel + impact overlay for an un-inspected layer.
function resetInspectUI() {
  state.impact = null;
  state.blocksPresent = new Set();
  state.inspected = false;
  $('inspect-result').classList.add('hidden');
  $('arch-warn').classList.add('hidden');
  $('btn-validate').disabled = true;
  clearKeywordImpact();
  for (let i = 0; i < NUM_BLOCKS; i++) {
    setMeter(`block-${i}`, 0, 'negligible');
    $(`block-${i}-score`).textContent = '—';
    blockCell(i).dataset.impact = 'negligible';
    blockCell(i).classList.remove('absent');
  }
  setMeter('llm', 0, 'negligible');
  setMeter('other', 0, 'negligible');
}

function addLayer() {
  syncActiveLayer();
  const L = makeLayer('');
  state.layers.push(L);
  state.activeId = L.id;
  applyLayerToUI(L);
  renderLayerBar();
  $('lora-path').focus();
  toast(`added layer ${state.layers.length}`, 'success');
}

function removeLayer(id) {
  if (state.layers.length <= 1) { toast('keep at least one layer', ''); return; }
  const idx = state.layers.findIndex((l) => l.id === id);
  if (idx < 0) return;
  const wasActive = id === state.activeId;
  state.layers.splice(idx, 1);
  if (wasActive) {
    const next = state.layers[Math.min(idx, state.layers.length - 1)];
    state.activeId = next.id;
    applyLayerToUI(next);
  }
  renderLayerBar();
}

function switchLayer(id) {
  if (id === state.activeId) return;
  syncActiveLayer();
  state.activeId = id;
  applyLayerToUI(activeLayer());
  renderLayerBar();
}

function updateActiveLayerLabels() {
  const idx = state.layers.findIndex((l) => l.id === state.activeId);
  const L = activeLayer();
  const label = L ? layerLabel(L, idx < 0 ? 0 : idx) : 'layer 1';
  document.querySelectorAll('[data-active-layer-label]').forEach((el) => {
    el.textContent = label;
  });
  updateDockLayerLabel(idx < 0 ? 0 : idx);
}

// Reflect the active layer (and how many there are) in the dock's switch button.
function updateDockLayerLabel(idx) {
  const el = $('qd-layer-label');
  if (!el) return;
  const n = state.layers.length;
  el.textContent = n > 1 ? `Layer ${idx + 1}/${n}` : 'Layer 1';
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function renderLayerBar() {
  const bar = $('layer-bar');
  if (!bar) return;
  const addBtn = $('layer-add');
  bar.querySelectorAll('.layer-chip').forEach((c) => c.remove());

  state.layers.forEach((layer, idx) => {
    const chip = document.createElement('div');
    chip.className = 'layer-chip' + (layer.id === state.activeId ? ' is-active' : '');
    chip.dataset.layerId = String(layer.id);
    chip.setAttribute('role', 'tab');
    chip.setAttribute('aria-selected', layer.id === state.activeId ? 'true' : 'false');

    const main = document.createElement('button');
    main.type = 'button';
    main.className = 'layer-chip-main';
    main.title = layer.path || 'no LoRA selected yet';
    main.innerHTML =
      `<span class="layer-chip-idx">${idx + 1}</span>` +
      `<span class="layer-chip-name">${escapeHtml(layerLabel(layer, idx))}</span>` +
      (layer.inspected ? '<span class="layer-chip-dot" title="inspected">●</span>' : '');
    main.addEventListener('click', () => switchLayer(layer.id));

    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'layer-chip-remove';
    rm.title = 'Remove this layer';
    rm.textContent = '×';
    rm.disabled = state.layers.length <= 1;
    rm.addEventListener('click', (e) => { e.stopPropagation(); removeLayer(layer.id); });

    chip.appendChild(main);
    chip.appendChild(rm);
    bar.insertBefore(chip, addBtn);
  });

  updateActiveLayerLabels();
}

// Every layer that points at a LoRA, with its current edit config — what the
// preview and save endpoints consume to combine the stack.
function layersForRequest() {
  syncActiveLayer();
  return state.layers
    .filter((l) => l.path)
    .map((l) => ({ lora_path: l.path, config: l.config }));
}


// ─── Block labels ─────────────────────────────────────────────────────────────
// Annotate a set of AnimaBlock cells with a note, so you can keep track of which
// sliders you changed and why. A label is { id, text, color, blocks:[int] }. They
// live on the active layer (L.labels) and mirror to localStorage keyed by the
// LoRA's file path, so re-inspecting the same file brings its labels back.
const LABELS_STORE = 'anima-block-labels';
const LABEL_COLORS = ['#ff8fa3', '#f4c969', '#c9a4ff', '#5fd6c0', '#7ab8ff', '#ffa765'];

let currentLabels = [];   // mirrors the active layer's labels
let labeling = false;     // block-select mode active?
let labelDraft = null;    // { id|null, color, blocks:Set<int> } while composing

function loadLabelStore() {
  try { return JSON.parse(localStorage.getItem(LABELS_STORE) || '{}') || {}; }
  catch { return {}; }
}
function saveLabelStore(map) {
  try { localStorage.setItem(LABELS_STORE, JSON.stringify(map)); } catch { /* private mode */ }
}
function pathKey(path) { return (path || '').trim().toLowerCase(); }

function labelsForPath(path) {
  const key = pathKey(path);
  if (!key) return [];
  const arr = loadLabelStore()[key];
  return Array.isArray(arr) ? arr : [];
}

// Persist the active layer's labels under its file path (no-op without a path —
// labels still live in-memory on the layer until it gets one).
function persistCurrentLabels() {
  const L = activeLayer();
  if (L) L.labels = currentLabels;
  const key = pathKey(L && L.path);
  if (!key) return;
  const map = loadLabelStore();
  if (currentLabels.length) map[key] = currentLabels;
  else delete map[key];
  saveLabelStore(map);
}

// Point currentLabels at the active layer's labels (preferring ones already on
// the layer, else whatever's saved for its path), then repaint chips + markers.
function hydrateLabelsForActiveLayer() {
  const L = activeLayer();
  const fromLayer = L && Array.isArray(L.labels) && L.labels.length ? L.labels : null;
  currentLabels = fromLayer || labelsForPath(L && L.path);
  if (L) L.labels = currentLabels;
  exitLabelMode();
  renderLabels();
}

function newLabelId() {
  const seq = currentLabels.reduce((m, l) => Math.max(m, +(l.id || '').replace(/\D/g, '') || 0), 0);
  return `lbl-${seq + 1}-${currentLabels.length}`;
}

function initBlockLabels() {
  $('label-add-btn').addEventListener('click', () => {
    if (labeling) exitLabelMode();
    else enterLabelMode(null);
  });
  $('label-cancel').addEventListener('click', exitLabelMode);
  $('label-save').addEventListener('click', onSaveLabel);
  $('label-text').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); onSaveLabel(); }
    if (e.key === 'Escape') { e.preventDefault(); exitLabelMode(); }
  });
  renderLabelSwatches(LABEL_COLORS[0]);
  renderLabels();
}

// Begin composing a label. `existing` rehydrates the form to edit one in place.
function enterLabelMode(existing) {
  labeling = true;
  labelDraft = existing
    ? { id: existing.id, color: existing.color, blocks: new Set(existing.blocks) }
    : { id: null, color: LABEL_COLORS[currentLabels.length % LABEL_COLORS.length], blocks: new Set() };

  $('block-grid').classList.add('labeling');
  $('label-form').classList.remove('hidden');
  $('label-add-btn').classList.add('is-active');
  $('label-text').value = existing ? existing.text : '';
  renderLabelSwatches(labelDraft.color);
  syncSelectedCells();
  updateLabelHint();
  $('label-text').focus();
}

function exitLabelMode() {
  labeling = false;
  labelDraft = null;
  const grid = $('block-grid');
  if (grid) grid.classList.remove('labeling');
  for (let i = 0; i < NUM_BLOCKS; i++) blockCell(i).classList.remove('selected');
  const form = $('label-form');
  if (form) form.classList.add('hidden');
  const add = $('label-add-btn');
  if (add) add.classList.remove('is-active');
}

function toggleBlockSelection(i) {
  if (!labelDraft) return;
  if (labelDraft.blocks.has(i)) labelDraft.blocks.delete(i);
  else labelDraft.blocks.add(i);
  blockCell(i).classList.toggle('selected', labelDraft.blocks.has(i));
  updateLabelHint();
}

function syncSelectedCells() {
  for (let i = 0; i < NUM_BLOCKS; i++) {
    blockCell(i).classList.toggle('selected', !!labelDraft && labelDraft.blocks.has(i));
  }
}

function updateLabelHint() {
  const n = labelDraft ? labelDraft.blocks.size : 0;
  $('label-form-hint').textContent = n
    ? `${n} block${n === 1 ? '' : 's'} selected — name it, then Save.`
    : 'Click blocks in the grid to include them.';
}

function renderLabelSwatches(selected) {
  const wrap = $('label-swatches');
  if (!wrap) return;
  wrap.replaceChildren(...LABEL_COLORS.map((c) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'label-swatch' + (c === selected ? ' is-active' : '');
    b.style.setProperty('--swatch', c);
    b.title = 'Use this colour';
    b.setAttribute('aria-label', `colour ${c}`);
    b.addEventListener('click', () => {
      if (labelDraft) labelDraft.color = c;
      wrap.querySelectorAll('.label-swatch').forEach((s) => s.classList.remove('is-active'));
      b.classList.add('is-active');
    });
    return b;
  }));
}

function onSaveLabel() {
  if (!labelDraft) return;
  const text = $('label-text').value.trim();
  if (!text) { toast('name the label first', 'error'); $('label-text').focus(); return; }
  if (!labelDraft.blocks.size) { toast('pick at least one block', 'error'); return; }

  const blocks = [...labelDraft.blocks].sort((a, b) => a - b);
  if (labelDraft.id) {
    const L = currentLabels.find((l) => l.id === labelDraft.id);
    if (L) { L.text = text; L.color = labelDraft.color; L.blocks = blocks; }
  } else {
    currentLabels.push({ id: newLabelId(), text, color: labelDraft.color, blocks });
  }
  persistCurrentLabels();
  exitLabelMode();
  renderLabels();
  toast('label saved', 'success');
}

function deleteLabel(id) {
  const idx = currentLabels.findIndex((l) => l.id === id);
  if (idx < 0) return;
  currentLabels.splice(idx, 1);
  persistCurrentLabels();
  renderLabels();
  toast('label removed', '');
}

// Briefly pulse a label's member cells so you can see which blocks it covers.
let labelHiliteTimer = null;
function highlightLabelBlocks(id) {
  const label = currentLabels.find((l) => l.id === id);
  if (!label) return;
  clearTimeout(labelHiliteTimer);
  for (let i = 0; i < NUM_BLOCKS; i++) blockCell(i).classList.remove('label-hilite');
  label.blocks.forEach((i) => {
    const cell = blockCell(i);
    if (cell) cell.classList.add('label-hilite');
  });
  const first = blockCell(label.blocks[0]);
  if (first) first.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  labelHiliteTimer = setTimeout(() => {
    for (let i = 0; i < NUM_BLOCKS; i++) blockCell(i).classList.remove('label-hilite');
  }, 1600);
}

// A human-readable "which sliders" summary: each member block + its current
// strength, so the chip's tooltip records what you changed.
function labelStrengthSummary(label) {
  return label.blocks.map((i) => {
    const s = blockStrength(i);
    const v = s ? (+s.value).toFixed(2) : '1.00';
    const off = blockToggle(i) && !blockToggle(i).checked ? ' (off)' : '';
    return `${String(i).padStart(2, '0')} @ ${v}${off}`;
  }).join('  ·  ');
}

function renderLabels() {
  renderLabelChips();
  renderLabelMarkers();
}

function renderLabelChips() {
  const wrap = $('label-chips');
  if (!wrap) return;
  wrap.replaceChildren(...currentLabels.map((label) => {
    const chip = document.createElement('div');
    chip.className = 'label-chip';
    chip.style.setProperty('--chip', label.color);
    chip.title = `Blocks ${label.blocks.join(', ')}\n${labelStrengthSummary(label)}`;

    const name = document.createElement('button');
    name.type = 'button';
    name.className = 'label-chip-name';
    name.innerHTML =
      `<span class="label-chip-dot" aria-hidden="true"></span>` +
      `<span class="label-chip-text">${escapeHtml(label.text)}</span>` +
      `<span class="label-chip-count">${label.blocks.length}</span>`;
    name.title = 'Click to highlight · double-click to edit';
    name.addEventListener('click', () => highlightLabelBlocks(label.id));
    name.addEventListener('dblclick', () => enterLabelMode(label));

    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'label-chip-remove';
    rm.title = 'Remove label';
    rm.textContent = '×';
    rm.addEventListener('click', (e) => { e.stopPropagation(); deleteLabel(label.id); });

    chip.append(name, rm);
    return chip;
  }));
}

// Build CSS gradient colour-stops that split a bar into one hard-edged segment
// per label colour — so a block carrying several labels shows every colour.
function colorStops(colors) {
  if (colors.length === 1) return `${colors[0]} 0% 100%`;
  const seg = 100 / colors.length;
  return colors
    .map((c, i) => `${c} ${(i * seg).toFixed(2)}% ${((i + 1) * seg).toFixed(2)}%`)
    .join(', ');
}

// Colour-code each labelled cell: tint its border + wash, lay a segmented colour
// bar across the top (one segment per label), and list the label names as
// readable colour-coded tags. Multiple labels stack as bar segments + tags.
function renderLabelMarkers() {
  const byBlock = new Map();
  currentLabels.forEach((label) => {
    label.blocks.forEach((i) => {
      if (!byBlock.has(i)) byBlock.set(i, []);
      byBlock.get(i).push(label);
    });
  });
  for (let i = 0; i < NUM_BLOCKS; i++) {
    const cell = blockCell(i);
    let bar = cell.querySelector('.block-label-bar');
    let strip = cell.querySelector('.block-labels');
    const labels = byBlock.get(i);

    if (!labels) {
      cell.classList.remove('labeled');
      cell.style.removeProperty('--label-color');
      cell.style.removeProperty('--label-stops');
      if (bar) bar.remove();
      if (strip) strip.remove();
      continue;
    }

    const colors = labels.map((l) => l.color);
    cell.classList.add('labeled');
    cell.style.setProperty('--label-color', colors[0]);
    cell.style.setProperty('--label-stops', colorStops(colors));

    if (!bar) {
      bar = document.createElement('div');
      bar.className = 'block-label-bar';
      cell.insertBefore(bar, cell.firstChild);
    }
    bar.title = labels.map((l) => l.text).join(' · ');

    if (!strip) {
      strip = document.createElement('div');
      strip.className = 'block-labels';
      cell.appendChild(strip);
    }
    strip.replaceChildren(...labels.map((label) => {
      const tag = document.createElement('span');
      tag.className = 'block-label-tag';
      tag.style.setProperty('--chip', label.color);
      tag.title = label.text;
      tag.innerHTML =
        `<span class="block-label-tag-dot" aria-hidden="true"></span>` +
        `<span class="block-label-tag-text">${escapeHtml(label.text)}</span>`;
      return tag;
    }));
  }
}


// ─── Colour theme (light / dark / sakura / kurenai) ─────────────────────────
// The theme is also applied pre-paint by an inline <head> script (no flash);
// here we just wire the switcher buttons and keep the active state in sync.
const THEME_STORE = 'anima-theme';
const THEMES = ['light', 'dark', 'sakura', 'kurenai'];

function wireThemeSwitch() {
  const buttons = Array.from(document.querySelectorAll('.theme-btn'));

  const setTheme = (mode, persist = true) => {
    if (!THEMES.includes(mode)) mode = 'dark';
    document.documentElement.dataset.theme = mode;
    buttons.forEach((b) => {
      const active = b.dataset.theme === mode;
      b.classList.toggle('is-active', active);
      b.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    if (persist) {
      try { localStorage.setItem(THEME_STORE, mode); } catch { /* private mode */ }
    }
    // Each theme owns its background image — swap it in as the theme changes.
    applyThemeBackground(mode);
    refreshBgMenu();
  };

  buttons.forEach((b) => b.addEventListener('click', () => setTheme(b.dataset.theme)));

  // Reflect whatever the head script already applied (falls back to storage/dark)
  let current = document.documentElement.dataset.theme;
  if (!THEMES.includes(current)) {
    try { current = localStorage.getItem(THEME_STORE); } catch { current = null; }
  }
  setTheme(current || 'dark', false);
}


// ─── Immersive toggle — a 3-state cycle ─────────────────────────────────────────
// off    → normal panels
// reveal → drop the panels' frosted blur + heavy fill so the image shows through
// hidden → take the panels out of the way entirely, leaving just the background
// Persisted like the theme and mirrored pre-paint by the <head> script.
const IMMERSIVE_STORE = 'anima-immersive';
const IMMERSIVE_STATES = ['off', 'reveal', 'hidden'];
const IMMERSIVE_META = {
  off:    { glyph: '👁', title: 'Reveal background — show the image through the panels', toast: 'panels restored' },
  reveal: { glyph: '🌥', title: 'Hide panels — show only the background', toast: 'background revealed' },
  hidden: { glyph: '⛶',  title: 'Restore panels', toast: 'panels hidden — click to restore' },
};

function wireImmersiveToggle() {
  const btn = $('immersive-btn');
  if (!btn) return;
  const glyph = btn.querySelector('.immersive-glyph') || btn.querySelector('span');

  const apply = (state, persist = true) => {
    if (!IMMERSIVE_STATES.includes(state)) state = 'off';
    if (state === 'off') delete document.documentElement.dataset.immersive;
    else document.documentElement.dataset.immersive = state;
    const meta = IMMERSIVE_META[state];
    btn.setAttribute('aria-pressed', state === 'off' ? 'false' : 'true');
    btn.title = meta.title;
    if (glyph) glyph.textContent = meta.glyph;
    if (persist) {
      try { localStorage.setItem(IMMERSIVE_STORE, state); } catch { /* private mode */ }
    }
  };

  btn.addEventListener('click', () => {
    const cur = document.documentElement.dataset.immersive || 'off';
    const next = IMMERSIVE_STATES[(IMMERSIVE_STATES.indexOf(cur) + 1) % IMMERSIVE_STATES.length];
    apply(next);
    toast(IMMERSIVE_META[next].toast, 'success');
  });

  // Reflect whatever the head script already applied (or stored state).
  let state = document.documentElement.dataset.immersive;
  if (!state) {
    try { state = localStorage.getItem(IMMERSIVE_STORE) || 'off'; } catch { state = 'off'; }
  }
  if (state === 'on') state = 'reveal';  // back-compat with the old 2-state value
  apply(state, false);
}


// ─── Per-theme background image ───────────────────────────────────────────────
// Each theme can carry its own full-screen background. The image itself lives on
// disk (static/user-bg/<theme>.<ext>) — far too big for localStorage — and we
// keep only the small URL per theme here, mirroring it pre-paint in <head> so a
// reload never flashes the wrong backdrop. Sources: generate (Anima, sized to
// the viewport) or upload. CSS already paints it edge-to-edge via background-
// size:cover on .bg-waifu, so it always fills the complete background.
const THEME_BG_STORE = 'anima-theme-bg';
const BG_MAX_SIDE = 2048;   // cap the final (upscaled) long edge — cover scales the rest
const BG_UPSCALE = 2;       // hi-res-fix factor: render the base at 1/N, refine up to fill the screen

function currentTheme() {
  const t = document.documentElement.dataset.theme;
  return THEMES.includes(t) ? t : 'dark';
}

function loadThemeBgMap() {
  try { return JSON.parse(localStorage.getItem(THEME_BG_STORE) || '{}') || {}; }
  catch { return {}; }
}
function saveThemeBgMap(map) {
  try { localStorage.setItem(THEME_BG_STORE, JSON.stringify(map)); } catch { /* private mode */ }
}
function getThemeBg(theme) { return loadThemeBgMap()[theme] || null; }

// Point --waifu-image at the active theme's image (or fall back to the default
// waifu-bg/gradient mesh). No-op when the theme isn't the one on screen.
function applyThemeBackground(theme) {
  if (theme !== currentTheme()) return;
  const url = getThemeBg(theme);
  const root = document.documentElement.style;
  if (url) root.setProperty('--waifu-image', `url("${url}")`);
  else root.removeProperty('--waifu-image');
}

function storeThemeBg(theme, url) {
  const map = loadThemeBgMap(); map[theme] = url; saveThemeBgMap(map);
  applyThemeBackground(theme);
  refreshBgMenu();
}
function removeThemeBgLocal(theme) {
  const map = loadThemeBgMap(); delete map[theme]; saveThemeBgMap(map);
  applyThemeBackground(theme);
  refreshBgMenu();
}

// Base render size for a screen-filling background: keep the screen's exact
// aspect, target the device-pixel viewport (so it's sharp on HiDPI) capped at
// BG_MAX_SIDE, then divide by the hi-res-fix factor so the refine pass upscales
// back up to fill the screen. Snap to /8 (the VAE downscale). Returns both the
// base (what we render) and the upscale factor (how much the refine pass grows
// it), so caller and toast agree on the final size.
function bgViewportSize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const vw = Math.max(1, Math.round(window.innerWidth * dpr));
  const vh = Math.max(1, Math.round(window.innerHeight * dpr));
  const fit = Math.min(1, BG_MAX_SIDE / Math.max(vw, vh));
  // Snap to /16: the latent (px/8) is patchified 2×2 by the DiT, so the pixel
  // side must be a multiple of 16 or generation asserts.
  const snap = (n) => Math.max(64, Math.round(n) - (Math.round(n) % 16));
  const finalW = snap(vw * fit), finalH = snap(vh * fit);
  // Base = final / upscale; the refine pass restores it to ~final.
  return {
    width: snap(finalW / BG_UPSCALE),
    height: snap(finalH / BG_UPSCALE),
    upscale: BG_UPSCALE,
    finalW, finalH,
  };
}

let bgBusyFlag = false;
function bgBusy(on, label) {
  bgBusyFlag = on;
  document.querySelectorAll('[data-bg-action]').forEach((b) => { b.disabled = on; });
  if (on && label) toast(label);
  if (!on) refreshBgMenu();  // restore correct enabled/disabled state
}

// Upload the data-URI to the server, which writes it to disk and hands back a
// stable URL. Throws with a readable message so callers can toast it.
async function persistThemeBg(theme, dataUri) {
  const r = await fetch('/api/background', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ theme, image: dataUri }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || 'could not save background');
  storeThemeBg(theme, data.url);
}

async function generateThemeBackground() {
  if (bgBusyFlag) return;
  if (!state.previewReady) {
    toast('background generation needs the Anima GPU backend', 'error');
    return;
  }
  const theme = currentTheme();
  const { width, height, upscale, finalW, finalH } = bgViewportSize();
  const body = buildPreviewBody({
    width, height, upscale, seed: Math.floor(Math.random() * 2147483647),
  });
  bgBusy(true);
  // Same spinner + elapsed timer as the sample preview — background generation
  // runs the same engine, so show the same progress on the preview stage.
  const stopProgress = startPreviewProgress(`background ${finalW}×${finalH}…`);
  try {
    const r = await fetch('/api/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'generation failed');
    await persistThemeBg(theme, data.image);
    const m = data.meta || {};
    toast(`background set for ${theme} · ${m.width || finalW}×${m.height || finalH}`, 'success');
  } catch (e) {
    toast(e.message || 'network error', 'error');
  } finally {
    stopProgress();
    bgBusy(false);
  }
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = () => reject(new Error('could not read file'));
    fr.readAsDataURL(file);
  });
}

async function uploadThemeBackground(file) {
  if (bgBusyFlag || !file) return;
  if (!file.type.startsWith('image/')) { toast('pick an image file', 'error'); return; }
  const theme = currentTheme();
  bgBusy(true, 'uploading background…');
  try {
    await persistThemeBg(theme, await fileToDataURL(file));
    toast(`background set for ${theme}`, 'success');
  } catch (e) {
    toast(e.message || 'upload failed', 'error');
  } finally {
    bgBusy(false);
  }
}

// Adopt the most recent Live Preview sample as the current theme's background.
// The sample is rendered with the *current* edit (same model/LoRA/block config),
// so this gives a backdrop that exactly matches what the editor produces — no
// fresh generation, no random seed, no aspect-ratio surprise.
async function useSampleAsBackground() {
  if (bgBusyFlag) return;
  const dataUri = state.lastPreviewImage;
  if (!dataUri) { toast('generate a sample first', 'error'); return; }
  const theme = currentTheme();
  bgBusy(true, 'setting background…');
  try {
    await persistThemeBg(theme, dataUri);
    toast(`background set for ${theme}`, 'success');
  } catch (e) {
    toast(e.message || 'could not set background', 'error');
  } finally {
    bgBusy(false);
  }
}

async function clearThemeBackground() {
  if (bgBusyFlag) return;
  const theme = currentTheme();
  if (!getThemeBg(theme)) { toast('no background set for this theme', ''); return; }
  bgBusy(true, 'clearing background…');
  try {
    const r = await fetch('/api/background/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ theme }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'could not clear background');
    removeThemeBgLocal(theme);
    toast(`background cleared for ${theme}`, 'success');
  } catch (e) {
    toast(e.message || 'network error', 'error');
  } finally {
    bgBusy(false);
  }
}

// Keep both control surfaces (header popover + Live Preview row) in sync with
// the active theme and what's available.
function refreshBgMenu() {
  if (bgBusyFlag) return;  // leave everything disabled mid-action
  const theme = currentTheme();
  const hasBg = !!getThemeBg(theme);
  document.querySelectorAll('[data-bg-theme-label]').forEach((el) => { el.textContent = theme; });
  document.querySelectorAll('[data-bg-action="clear"]').forEach((b) => { b.disabled = !hasBg; });
  document.querySelectorAll('[data-bg-action="generate"]').forEach((b) => {
    b.disabled = !state.previewReady;
    b.title = state.previewReady
      ? 'Generate a background sized to your screen'
      : 'Needs the Anima GPU backend — see Live Preview';
  });
  const hasSample = !!state.lastPreviewImage;
  document.querySelectorAll('[data-bg-action="use-preview"]').forEach((b) => {
    b.disabled = !hasSample;
    b.title = hasSample
      ? 'Set the generated sample (your exact edits) as this theme’s background'
      : 'Generate a sample first';
  });
}

function wireBgPopover() {
  const btn = $('bg-menu-btn'), pop = $('bg-menu-pop');
  if (!btn || !pop) return;
  const close = () => { pop.hidden = true; btn.setAttribute('aria-expanded', 'false'); };
  const open = () => { pop.hidden = false; btn.setAttribute('aria-expanded', 'true'); refreshBgMenu(); };
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    pop.hidden ? open() : close();
  });
  pop.addEventListener('click', (e) => e.stopPropagation());
  document.addEventListener('click', () => { if (!pop.hidden) close(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
}

function initBackgrounds() {
  const fileInput = $('bg-upload-input');
  document.querySelectorAll('[data-bg-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.bgAction;
      if (action === 'generate') generateThemeBackground();
      else if (action === 'use-preview') useSampleAsBackground();
      else if (action === 'upload') { if (fileInput) fileInput.click(); }
      else if (action === 'clear') clearThemeBackground();
    });
  });
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      const f = fileInput.files && fileInput.files[0];
      fileInput.value = '';  // let the same file be re-picked later
      uploadThemeBackground(f);
    });
  }
  wireBgPopover();
  applyThemeBackground(currentTheme());
  refreshBgMenu();
}


// ─── Layer view mode (grid / list) ──────────────────────────────────────────
const VIEW_STORE = 'anima-layer-view';

function wireViewToggle() {
  const grid = $('block-grid');
  const buttons = [$('view-grid'), $('view-list')];

  const setView = (mode, persist = true) => {
    if (mode !== 'grid' && mode !== 'list') mode = 'grid';
    grid.dataset.view = mode;
    buttons.forEach((b) => {
      const active = b.dataset.view === mode;
      b.classList.toggle('is-active', active);
      b.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    if (persist) {
      try { localStorage.setItem(VIEW_STORE, mode); } catch { /* private mode */ }
    }
  };

  buttons.forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));

  // Restore last choice (defaults to the grid view in markup)
  let saved = null;
  try { saved = localStorage.getItem(VIEW_STORE); } catch { saved = null; }
  if (saved) setView(saved, false);
}


// ─── Build the 28-block grid ────────────────────────────────────────────────
function buildBlockGrid() {
  const grid = $('block-grid');
  const frag = document.createDocumentFragment();

  for (let i = 0; i < NUM_BLOCKS; i++) {
    const cell = document.createElement('div');
    cell.className = 'block-cell';
    cell.id = `block-${i}`;
    cell.dataset.impact = 'negligible';
    cell.innerHTML = `
      <div class="block-row">
        <span class="block-num">${String(i).padStart(2, '0')}</span>
        <span class="block-score" id="block-${i}-score">—</span>
        <input type="checkbox" class="block-checkbox" id="block-${i}-toggle" checked />
      </div>
      <div class="block-meter" title="estimated effect on the result">
        <span class="block-meter-track">
          <span class="block-meter-fill" id="block-${i}-meter-fill" data-impact="negligible"></span>
        </span>
      </div>
      <div class="block-meter block-meter-kw" title="keyword response (validation prompt)">
        <span class="block-meter-track">
          <span class="block-meter-fill block-meter-fill-kw" id="block-${i}-kw-meter-fill" data-impact="negligible"></span>
        </span>
      </div>
      <div class="block-strength">
        <input type="range" id="block-${i}-strength" min="-2" max="2" step="0.05" value="1" />
        <output id="block-${i}-strength-val">1.00</output>
      </div>
    `;
    frag.appendChild(cell);
  }
  grid.appendChild(frag);

  // Wire interactions
  for (let i = 0; i < NUM_BLOCKS; i++) {
    blockToggle(i).addEventListener('change', () => {
      blockCell(i).classList.toggle('disabled', !blockToggle(i).checked);
      markPresetCustom();
    });
    blockStrength(i).addEventListener('input', () => {
      blockStrengthVal(i).textContent = (+blockStrength(i).value).toFixed(2);
      markPresetCustom();
    });
    // While labelling, a click anywhere on the cell (but not on its checkbox /
    // slider) toggles the block's membership in the label being composed.
    blockCell(i).addEventListener('click', (e) => {
      if (!labeling) return;
      if (e.target.closest('input')) return;
      toggleBlockSelection(i);
    });
  }
}


// ─── Global controls ────────────────────────────────────────────────────────
function wireGlobalControls() {
  // Range slider value displays
  bindRange('global-strength', 'global-strength-val');
  bindRange('llm-strength', 'llm-strength-val');
  bindRange('other-strength', 'other-strength-val');

  // Inspect / Save buttons
  $('btn-inspect').addEventListener('click', onInspect);
  $('btn-save').addEventListener('click', onSave);

  // Validation keyword — attribute a concept to layers
  $('btn-validate').addEventListener('click', onValidate);
  $('validate-keyword').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); onValidate(); }
  });

  // LoRA in/out paths: strip pasted quotes, cache for reuse, auto-suggest output.
  const loraIn = $('lora-path'), loraOut = $('output-path');
  loraIn.addEventListener('paste', () => setTimeout(() => cleanPathField(loraIn), 0));
  loraIn.addEventListener('change', () => {
    cleanPathField(loraIn);
    rememberPath('lora_in', loraIn.value);
    // Reflect the new path on the active layer's chip immediately.
    const L = activeLayer();
    if (L) {
      L.path = loraIn.value;
      L.name = L.path ? baseName(L.path) : '';
      // A different file carries its own saved labels — swap them in.
      hydrateLabelsForActiveLayer();
      renderLayerBar();
    }
    autoSuggestOutputPath();
    // Update the AnimaBlock layers section to reflect the newly-set LoRA.
    maybeAutoInspect();
  });
  loraIn.addEventListener('blur', autoSuggestOutputPath);
  loraOut.addEventListener('paste', () => setTimeout(() => cleanPathField(loraOut), 0));
  loraOut.addEventListener('change', () => {
    cleanPathField(loraOut);
    rememberPath('lora_out', loraOut.value);
  });

  // Preset selector
  $('preset-select').addEventListener('change', onPresetChange);

  // Allow Enter to inspect
  loraIn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); onInspect(); }
  });
}

function bindRange(rangeId, outId, decimals = 2) {
  const r = $(rangeId), o = $(outId);
  const sync = () => { o.textContent = (+r.value).toFixed(decimals); };
  r.addEventListener('input', sync);
  sync();
}


// ─── Health check ──────────────────────────────────────────────────────────
async function checkHealth() {
  const pill = $('health-pill');
  try {
    const r = await fetch('/api/health');
    const data = await r.json();
    if (data.ok) {
      pill.dataset.state = 'ok';
      pill.textContent = `online · ${data.num_blocks} blocks`;
    } else {
      pill.dataset.state = 'error';
      pill.textContent = 'error';
    }
  } catch {
    pill.dataset.state = 'error';
    pill.textContent = 'offline';
  }
}


// ─── Presets ────────────────────────────────────────────────────────────────
async function loadPresets() {
  try {
    const r = await fetch('/api/presets');
    const data = await r.json();
    state.presets = data.presets;
    const sel = $('preset-select');
    sel.innerHTML = '';
    for (const name of Object.keys(data.presets)) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (name === 'All Blocks') opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (e) {
    toast('failed to load presets', 'error');
  }
}

function onPresetChange() {
  const name = $('preset-select').value;
  const blocks = state.presets[name];
  if (blocks === null) return;  // 'Custom' — leave UI as is

  const enabled = new Set(blocks);
  for (let i = 0; i < NUM_BLOCKS; i++) {
    const isOn = enabled.has(i);
    blockToggle(i).checked = isOn;
    blockStrength(i).value = 1.0;
    blockStrengthVal(i).textContent = '1.00';
    blockCell(i).classList.toggle('disabled', !isOn);
  }

  // 'All Off' should also turn off LLMAdapter and Other weights, matching the
  // ComfyUI preset behaviour.
  if (name === 'All Off') {
    $('llm-enabled').checked = false;
    $('other-enabled').checked = false;
  }
}

let suppressCustom = false;
function markPresetCustom() {
  if (suppressCustom) return;
  if ($('preset-select').value !== 'Custom') {
    $('preset-select').value = 'Custom';
  }
}


// ─── Inspect ────────────────────────────────────────────────────────────────
// Inspect the active layer's LoRA so the AnimaBlock grid reflects it. ``auto``
// is set when triggered by a path change / layer switch rather than the button:
// it stays quiet about a missing path and dedupes via the layer's inspectedFor.
async function onInspect(opts = {}) {
  const auto = !!opts.auto;
  const path = pathVal('lora-path');
  if (!path) {
    if (!auto) toast('paste a LoRA path first', 'error');
    return;
  }

  // Mark which path we're inspecting up front so a near-simultaneous change
  // event (e.g. Enter, which fires both keydown and change) doesn't double-run.
  const L0 = activeLayer();
  const prevInspectedFor = L0 ? L0.inspectedFor : undefined;
  if (L0) L0.inspectedFor = path;

  if (!auto) toast('inspecting…');
  const btn = $('btn-inspect');
  btn.disabled = true;

  try {
    const r = await fetch('/api/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lora_path: path }),
    });
    const data = await r.json();
    if (!r.ok) {
      if (L0) L0.inspectedFor = prevInspectedFor;  // allow a retry
      toast(data.error || 'inspect failed', 'error');
      return;
    }
    applyInspectResult(data);
    // Persist the analysis on the active layer so switching tabs restores it.
    const L = activeLayer();
    if (L) {
      L.path = path;
      L.name = baseName(path);
      L.inspected = true;
      L.inspectData = data;
      L.inspectedFor = path;
      L.config = collectEditConfig();   // post-inspect (absent blocks now unchecked)
      L.preset = $('preset-select').value;
      hydrateLabelsForActiveLayer();    // bring back any labels saved for this file
      renderLayerBar();
    }
    rememberPath('lora_in', path);  // a path that inspected cleanly is worth reusing
    toast('LoRA inspected', 'success');
  } catch (e) {
    if (L0) L0.inspectedFor = prevInspectedFor;  // allow a retry
    toast('network error', 'error');
  } finally {
    btn.disabled = false;
  }
}

// Inspect the active layer if its path is set and not already inspected, so the
// AnimaBlock layers section updates automatically on path change / layer switch.
function maybeAutoInspect() {
  const L = activeLayer();
  const path = pathVal('lora-path');
  if (path && (!L || L.inspectedFor !== path)) onInspect({ auto: true });
}

function applyInspectResult(data) {
  const { detected_architecture, is_anima, summary, impact } = data;
  state.impact = impact;
  state.blocksPresent = new Set(summary.blocks_present);
  state.inspected = true;
  // Size / precision drive the compression projection.
  state.sizeInfo = {
    file_size_bytes: data.file_size_bytes,
    payload_bytes: data.payload_bytes,
    dtype: data.dtype,
    profile: data.size_profile || null,
  };

  // Fill the inspect grid
  $('r-arch').textContent     = detected_architecture;
  $('r-total').textContent    = String(summary.total_keys);
  $('r-blocks').textContent   = `${summary.num_blocks} / ${NUM_BLOCKS}`;
  $('r-llm').textContent      = summary.has_llm_adapter ? 'yes' : 'absent';
  $('r-other').textContent    = String(summary.other_count);
  $('r-dtype').textContent    = data.dtype || '—';
  $('r-size').textContent     = data.file_size_bytes ? fmtBytes(data.file_size_bytes) : '—';
  updateCompressEstimate();

  // Strongest block from impact scores
  let strongest = null, max = -1;
  for (const [k, v] of Object.entries(impact.block_norm)) {
    if (v > max) { max = v; strongest = k; }
  }
  $('r-strongest').textContent = strongest === null ? '—' : `block ${strongest} (${max}%)`;

  // Warn if architecture clearly isn't Anima
  const warn = $('arch-warn');
  if (!is_anima) {
    warn.textContent = `This LoRA looks like ${detected_architecture}, not Anima. ` +
                       `Editing may produce a file that no Anima inference tool can load.`;
    warn.classList.remove('hidden');
  } else {
    warn.classList.add('hidden');
  }

  // Reveal panel + apply impact colors + contribution meters
  $('inspect-result').classList.remove('hidden');
  applyImpact(impact);
  markAbsentBlocks(state.blocksPresent);

  // Enable save and auto-suggest output path
  $('btn-save').disabled = false;
  autoSuggestOutputPath();

  // A fresh LoRA invalidates any previous keyword overlay
  $('btn-validate').disabled = false;
  clearKeywordImpact();
}

// Map a 0..100 score (relative to the strongest layer) to a legend band.
function impactBand(norm) {
  if (norm >= 70) return 'high';
  if (norm >= 40) return 'medium';
  if (norm >= 10) return 'low';
  return 'negligible';
}

// Paint each layer's estimated effect on the result. The score is the impact
// estimate (Frobenius norm of the low-rank update) normalised to 0..100 against
// the strongest layer, so the meter bar fills the full range and the colour
// band makes the strongest layers obvious at a glance.
//   • meter bar = estimated effect (0..100, relative to the strongest layer)
//   • pill %    = the same estimate, colour-coded by band
function applyImpact(impact) {
  for (let i = 0; i < NUM_BLOCKS; i++) {
    const norm = impact.block_norm[i] ?? 0;   // estimated effect, vs strongest layer
    const band = impactBand(norm);
    blockCell(i).dataset.impact = band;
    $(`block-${i}-score`).textContent = norm > 0 ? `${norm}%` : '—';
    setMeter(`block-${i}`, norm, band);
  }

  // LLMAdapter + Other affect the result too — estimate them on the same scale
  // (relative to the strongest block) so they share the legend bands.
  const maxBlock = impact.max_score || 0;
  const rel = (raw) => (maxBlock > 0 ? (100 * raw) / maxBlock : 0);
  const llm = rel(impact.llm_adapter_score || 0);
  const other = rel(impact.other_score || 0);
  setMeter('llm', llm, impactBand(llm));
  setMeter('other', other, impactBand(other));
}

// Fill a meter to its estimated-effect score and tint it (+ its % label, if any)
// by band. Width is clamped to 0..100 in case a component beats the top block.
function setMeter(prefix, normPct, band) {
  const fill = $(`${prefix}-meter-fill`);
  if (!fill) return;
  fill.style.width = `${Math.max(0, Math.min(100, normPct))}%`;
  fill.dataset.impact = band;
  const pct = $(`${prefix}-meter-pct`);
  if (pct) {
    pct.textContent = normPct >= 0.5 ? `${Math.round(normPct)}%` : '—';
    pct.dataset.impact = band;
  }
}

function markAbsentBlocks(presentSet) {
  for (let i = 0; i < NUM_BLOCKS; i++) {
    const absent = !presentSet.has(i);
    blockCell(i).classList.toggle('absent', absent);
    // If a block isn't in the LoRA, uncheck it (nothing to edit anyway)
    if (absent) {
      suppressCustom = true;
      blockToggle(i).checked = false;
      blockCell(i).classList.add('disabled');
      suppressCustom = false;
    }
  }
}


// ─── Validation prompt (keyword attribution) ────────────────────────────────
// "Which layers does this keyword light up?" — the server pushes the keyword
// through the model (activation-delta when the GPU backend is up, a cheaper
// cross-attention projection otherwise) and returns per-block scores shaped
// like the inspect impact. We paint them on the second (jade) meter row.
const KW_METHOD_LABEL = {
  activation:     'faithful · activation delta',
  cross_attn:     'approximate · cross-attention only',
  cross_attn_cpu: 'approximate · cross-attention only (CPU)',
  static:         'unavailable · prompt-free score',
};

async function onValidate() {
  if (!state.inspected) { toast('inspect a LoRA first', 'error'); return; }
  const keyword = $('validate-keyword').value.trim();
  if (!keyword) { toast('enter a keyword to validate', 'error'); return; }
  const path = pathVal('lora-path');
  if (!path) { toast('paste a LoRA path first', 'error'); return; }

  const btn = $('btn-validate');
  btn.disabled = true;
  const note = $('validate-note');
  note.classList.remove('hidden');
  note.dataset.method = 'pending';
  note.textContent = `attributing “${keyword}”… (first run loads the model — this can take a while)`;
  toast('validating…');

  try {
    const r = await fetch('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lora_path: path,
        keyword,
        config: collectEditConfig(),
        model_paths: previewModelPaths(),
      }),
    });
    const data = await r.json();
    if (!r.ok) {
      toast(data.error || 'validation failed', 'error');
      note.dataset.method = 'error';
      note.textContent = data.error || 'validation failed';
      return;
    }
    applyKeywordImpact(data);
    toast(`“${keyword}” attributed`, 'success');
  } catch (e) {
    toast('network error', 'error');
    note.dataset.method = 'error';
    note.textContent = 'network error';
  } finally {
    btn.disabled = false;
  }
}

function applyKeywordImpact(data) {
  const impact = data.impact || {};
  const blockNorm = impact.block_norm || {};
  document.body.classList.add('validated');

  for (let i = 0; i < NUM_BLOCKS; i++) {
    const norm = blockNorm[i] ?? 0;
    setMeter(`block-${i}-kw`, norm, impactBand(norm));
  }
  // LLMAdapter + Other on the same scale as the keyword block scores.
  const maxBlock = impact.max_score || 0;
  const rel = (raw) => (maxBlock > 0 ? (100 * raw) / maxBlock : 0);
  setMeter('llm-kw',   rel(impact.llm_adapter_score || 0), 'negligible');
  setMeter('other-kw', rel(impact.other_score || 0),       'negligible');

  const note = $('validate-note');
  note.classList.remove('hidden');
  note.dataset.method = data.method || 'static';
  const label = KW_METHOD_LABEL[data.method] || data.method || '';
  note.textContent = `“${data.keyword}” — ${label}. ${data.note || ''}`;
}

function clearKeywordImpact() {
  document.body.classList.remove('validated');
  const note = $('validate-note');
  note.classList.add('hidden');
  note.textContent = '';
  for (let i = 0; i < NUM_BLOCKS; i++) setMeter(`block-${i}-kw`, 0, 'negligible');
  setMeter('llm-kw', 0, 'negligible');
  setMeter('other-kw', 0, 'negligible');
}


// ─── Auto-suggest output path ───────────────────────────────────────────────
function autoSuggestOutputPath() {
  const inPath = cleanPath($('lora-path').value);
  if (!inPath) return;
  const out = $('output-path');
  if (out.value.trim()) return;  // user already typed something

  // With multiple layers the output is a merge of all of them, so name it
  // "_merged"; a single layer is just "_edited". Handle both / and \ separators.
  const multi = state.layers && state.layers.filter((l) => l.path).length > 1;
  const suffix = multi ? '_merged' : '_edited';
  const m = inPath.match(/^(.*?)(\.[^.\\/]+)$/);
  if (m) out.value = `${m[1]}${suffix}${m[2]}`;
  else   out.value = `${inPath}${suffix}.safetensors`;
}


// ─── Save ───────────────────────────────────────────────────────────────────
async function onSave() {
  const loras = layersForRequest();
  if (!loras.length) { toast('add at least one LoRA layer', 'error'); return; }
  const outPath = pathVal('output-path');
  if (!outPath) { toast('output path required', 'error'); return; }

  const btn = $('btn-save');
  btn.disabled = true;
  toast(loras.length > 1 ? `merging ${loras.length} layers…` : 'saving…');

  try {
    const compress = collectCompress();
    const reqBody = { output_path: outPath, loras };
    if (compress) reqBody.compress = compress;
    const r = await fetch('/api/edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody),
    });
    const data = await r.json();
    if (!r.ok) {
      toast(data.error || 'save failed', 'error');
      $('save-result').classList.remove('hidden');
      $('save-info').textContent = JSON.stringify(data, null, 2);
      return;
    }
    const info = data.info;
    $('save-result').classList.remove('hidden');
    $('save-info').textContent = formatSaveInfo(info);
    loras.forEach((l) => rememberPath('lora_in', l.lora_path));
    rememberPath('lora_out', outPath);
    toast(`saved → ${outPath.split(/[\\/]/).pop()}`, 'success');
  } catch (e) {
    toast('network error', 'error');
  } finally {
    btn.disabled = false;
  }
}

function formatSaveInfo(info) {
  // Multi-layer merge response (the only shape the UI sends now).
  const lines = [
    `output:                ${info.output_path}`,
    `layers merged:         ${info.merged_layers}`,
    ...info.input_paths.map((p, i) => `  layer ${i + 1}:            ${p.split(/[\\/]/).pop()}`),
    `tensors written:       ${info.output_tensor_count}`,
    `modules total:         ${info.modules_total}`,
    `  rank-concatenated:   ${info.modules_concatenated}`,
    `  passed through:      ${info.modules_passthrough}`,
    `architecture:          ${info.detected_architecture}`,
  ];
  if (info.collisions && info.collisions.length) {
    lines.push(`uncombinable modules:  ${info.collisions.length} (kept first layer's)`);
  }
  const c = info.compression;
  if (c) {
    const pct = c.ratio != null ? ` (${Math.round(c.ratio * 100)}% of original)` : '';
    lines.push(
      `size:                  ${fmtBytes(c.size_before)} → ${fmtBytes(c.size_after)}${pct}`,
      `precision:             ${c.dtype_before || '?'} → ${c.dtype_after || '?'}`,
    );
    if (c.svd && c.svd.pairs_reduced) {
      lines.push(`rank reduced:          ${c.svd.rank_before} → ${c.svd.rank_after}` +
                 ` across ${c.svd.pairs_reduced} module(s)`);
    }
    if (c.svd && c.svd.skipped && c.svd.skipped.length) {
      lines.push(`rank-reduction skipped: ${c.svd.skipped.length} non-2D module(s)`);
    }
  }
  return lines.join('\n');
}


// ─── Shared: build EditConfig from the current UI ────────────────────────────
function collectEditConfig() {
  const enabled = [];
  const block_strengths = {};
  for (let i = 0; i < NUM_BLOCKS; i++) {
    if (blockToggle(i).checked) {
      enabled.push(i);
      const s = parseFloat(blockStrength(i).value);
      if (s !== 1.0) block_strengths[i] = s;
    }
  }
  return {
    enabled_blocks: enabled,
    block_strengths,
    llm_adapter_enabled: $('llm-enabled').checked,
    llm_adapter_strength: parseFloat($('llm-strength').value),
    other_enabled: $('other-enabled').checked,
    other_strength: parseFloat($('other-strength').value),
    global_strength: parseFloat($('global-strength').value),
  };
}


// ─── Optimize size (compression) ─────────────────────────────────────────────
function fmtBytes(n) {
  if (!n && n !== 0) return '—';
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

// Read the compression panel into the { dtype, svd_rank|svd_energy } shape the
// API expects, or null when nothing is selected.
function collectCompress() {
  const dtype = $('compress-dtype').value;
  const svdOn = $('compress-svd-enabled').checked;
  const out = {};
  if (dtype && dtype !== 'keep') out.dtype = dtype;
  if (svdOn) {
    const mode = $('compress-svd-mode').value;
    const v = parseFloat($('compress-svd-value').value);
    if (mode === 'energy') {
      if (v > 0 && v < 100) out.svd_energy = v / 100;
    } else if (v >= 1) {
      out.svd_rank = Math.round(v);
    }
  }
  return Object.keys(out).length ? out : null;
}

function wireCompressPanel() {
  const svdToggle = $('compress-svd-enabled');
  const svdOpts = $('compress-svd-opts');
  const mode = $('compress-svd-mode');
  const valLabel = $('compress-svd-val-label');
  const valInput = $('compress-svd-value');

  const syncMode = () => {
    if (mode.value === 'energy') {
      valLabel.textContent = 'Energy kept (%)';
      valInput.min = 1; valInput.max = 99; valInput.step = 1;
      if (parseFloat(valInput.value) > 99) valInput.value = 95;
    } else {
      valLabel.textContent = 'Target rank';
      valInput.min = 1; valInput.removeAttribute('max'); valInput.step = 1;
    }
  };
  const refresh = () => {
    svdOpts.classList.toggle('hidden', !svdToggle.checked);
    updateCompressEstimate();
  };

  svdToggle.addEventListener('change', refresh);
  mode.addEventListener('change', () => { syncMode(); updateCompressEstimate(); });
  valInput.addEventListener('input', updateCompressEstimate);
  $('compress-dtype').addEventListener('change', updateCompressEstimate);
  syncMode();
  refresh();
}

// Bytes-per-element for a target dtype name (null = keep current).
const DTYPE_ELEM = {
  fp16: 2, bf16: 2, float16: 2, bfloat16: 2,
  fp8_e4m3fn: 1, fp8_e5m2: 1, fp8: 1,
  fp32: 4, float32: 4,
};

// Project the saved payload size from the inspect profile + selected options.
// dtype downcast and SVD-by-rank are both computed exactly from per-pair ranks;
// only SVD-by-energy can't be sized ahead of time (it depends on the spectrum).
function projectPayloadBytes(profile, opts) {
  const newElem = opts.dtype ? (DTYPE_ELEM[opts.dtype] ?? profile.elem_size)
                             : profile.elem_size;
  let pairNumel = 0;
  for (const [r0Str, numel] of Object.entries(profile.pairs_by_rank || {})) {
    const r0 = parseInt(r0Str, 10);
    pairNumel += (opts.svd_rank && opts.svd_rank < r0)
      ? (numel * opts.svd_rank) / r0
      : numel;
  }
  const floatNumel = pairNumel + (profile.fixed_float_numel || 0);
  return floatNumel * newElem + (profile.fixed_bytes || 0);
}

function updateCompressEstimate() {
  const box = $('compress-estimate');
  const chip = $('compress-summary');
  const c = collectCompress();
  if (!c) { chip.textContent = 'off'; }
  else {
    const parts = [];
    if (c.dtype) parts.push(c.dtype);
    if (c.svd_rank) parts.push(`rank ${c.svd_rank}`);
    if (c.svd_energy) parts.push(`${Math.round(c.svd_energy * 100)}% energy`);
    chip.textContent = parts.join(' · ');
  }

  const si = state.sizeInfo;
  if (!c || !si || !si.profile) { box.classList.add('hidden'); return; }

  const before = si.file_size_bytes || si.payload_bytes;
  // Header + non-tensor overhead between the actual file and the raw payload;
  // carry it onto the projection so before/after are measured the same way.
  const overhead = Math.max(0, (si.file_size_bytes || 0) - (si.payload_bytes || 0));

  // Energy-mode SVD can't be sized ahead of time (rank kept depends on the
  // spectrum); projectPayloadBytes ignores it, so the number reflects only the
  // dtype change and is an upper bound when energy mode is also on.
  const hasEnergy = !!c.svd_energy;
  const projected = projectPayloadBytes(si.profile, c) + overhead;
  const ratio = before ? projected / before : 1;

  const lines = [`Current: <b>${fmtBytes(before)}</b> (${si.dtype || '?'})`];
  if (hasEnergy && !c.dtype) {
    // Nothing computable client-side — only the exact size after saving.
    lines.push('Estimated: <b>smaller</b> — energy-based rank reduction sizes ' +
               'per module; exact size shown after saving.');
  } else {
    lines.push(`Estimated: <b>${fmtBytes(projected)}</b> ` +
               `(${Math.round(ratio * 100)}% of current)`);
    if (hasEnergy) {
      lines.push('<span class="compress-note">+ further savings from energy-based ' +
                 'rank reduction (sized after saving)</span>');
    }
  }
  box.innerHTML = lines.join('<br>');
  box.classList.remove('hidden');
}


// ─── Live Preview ─────────────────────────────────────────────────────────────
const PV_STORE = 'anima-preview-settings';

function initPreview() {
  // Range value displays
  bindRange('pv-steps', 'pv-steps-val', 0);
  bindRange('pv-cfg', 'pv-cfg-val', 1);
  bindRange('pv-eta', 'pv-eta-val', 2);

  $('btn-preview').addEventListener('click', onPreview);
  $('pv-rand-seed').addEventListener('click', () => {
    $('pv-seed').value = Math.floor(Math.random() * 2147483647);
    savePreviewSettings();
  });

  // Persist settings as the user tweaks them
  const persistIds = ['pv-prompt', 'pv-negative', 'pv-sampler', 'pv-scheduler',
                      'pv-steps', 'pv-cfg', 'pv-eta', 'pv-size', 'pv-seed',
                      'pv-path-dit', 'pv-path-vae', 'pv-path-te'];
  persistIds.forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener('change', savePreviewSettings);
  });
  // Model-path edits also re-check whether the real backend can light up.
  // Clean the field so pasted "quoted paths" lose their quotes on screen, and
  // cache each path for reuse via its <datalist>.
  PATH_FIELDS.filter((f) => f.input.startsWith('pv-path-')).forEach(({ key, input }) => {
    const el = $(input);
    el.addEventListener('paste', () => setTimeout(() => {
      if (cleanPathField(el)) savePreviewSettings();
    }, 0));
    el.addEventListener('change', () => {
      cleanPathField(el);
      savePreviewSettings();
      rememberPath(key, el.value);
      refreshPreviewCapabilities();
    });
  });

  loadPreviewCapabilities().then(() => {
    restorePreviewSettings();
  });
}

function previewModelPaths() {
  return {
    dit: cleanPath($('pv-path-dit').value),
    vae: cleanPath($('pv-path-vae').value),
    text_encoder: cleanPath($('pv-path-te').value),
  };
}

async function loadPreviewCapabilities() {
  try {
    const r = await fetch('/api/preview/capabilities');
    const caps = await r.json();
    // Populate sampler + scheduler dropdowns once
    fillSelect('pv-sampler', caps.samplers, 'res_2m');
    fillSelect('pv-scheduler', caps.schedulers, 'karras');
    applyCapabilities(caps);
  } catch (e) {
    setPreviewPill('error', 'preview offline');
  }
}

async function refreshPreviewCapabilities() {
  try {
    const r = await fetch('/api/preview/capabilities', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_paths: previewModelPaths() }),
    });
    applyCapabilities(await r.json());
  } catch { /* leave pill as-is */ }
}

function applyCapabilities(caps) {
  state.previewReady = caps.active_backend === 'anima';
  refreshBgMenu();  // gate the "Generate background" buttons on real-backend availability
  if (caps.active_backend === 'anima') {
    setPreviewPill('ok', `Anima · ${caps.device}`);
    $('pv-backend-hint').textContent = `Real generation ready on ${caps.device}.`;
  } else {
    const missing = caps.missing_models || [];
    const notFound = missing.some((m) => m.includes('not found'));
    setPreviewPill('error',
      !caps.cuda ? 'GPU required'
      : notFound ? 'model path not found'
      : 'set model paths');
    $('pv-backend-hint').textContent = caps.cuda
      ? `GPU detected. ${caps.reason}. Fill these to generate real Anima images.`
      : `${caps.reason}. Install a CUDA build of torch ` +
        `(https://pytorch.org/get-started/locally/), then run setup_preview. ` +
        `There is no CPU preview.`;
    // A wrong-but-set path is invisible while "Model paths" is collapsed — open
    // it so the user sees which one didn't resolve. Stay closed if no GPU.
    if (!caps.cuda) $('pv-advanced').open = false;
    else if (notFound) $('pv-advanced').open = true;
  }
}

function fillSelect(id, items, fallback) {
  const sel = $(id);
  if (sel.options.length) return;  // already populated
  (items || [fallback]).forEach((name) => {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    if (name === fallback) opt.selected = true;
    sel.appendChild(opt);
  });
}

function setPreviewPill(state, text) {
  const pill = $('preview-pill');
  pill.dataset.state = state;
  pill.textContent = text;
}

function savePreviewSettings() {
  const s = {
    prompt: $('pv-prompt').value, negative: $('pv-negative').value,
    sampler: $('pv-sampler').value, scheduler: $('pv-scheduler').value,
    steps: $('pv-steps').value, cfg: $('pv-cfg').value, eta: $('pv-eta').value,
    size: $('pv-size').value, seed: $('pv-seed').value,
    dit: $('pv-path-dit').value, vae: $('pv-path-vae').value, te: $('pv-path-te').value,
  };
  try { localStorage.setItem(PV_STORE, JSON.stringify(s)); } catch { /* private mode */ }
}

function restorePreviewSettings() {
  let s;
  try { s = JSON.parse(localStorage.getItem(PV_STORE) || 'null'); } catch { s = null; }
  if (!s) return;
  const set = (id, v) => { if (v !== undefined && v !== null && $(id)) $(id).value = v; };
  set('pv-prompt', s.prompt); set('pv-negative', s.negative);
  set('pv-sampler', s.sampler); set('pv-scheduler', s.scheduler);
  set('pv-steps', s.steps); set('pv-cfg', s.cfg); set('pv-eta', s.eta);
  set('pv-size', s.size); set('pv-seed', s.seed);
  set('pv-path-dit', s.dit); set('pv-path-vae', s.vae); set('pv-path-te', s.te);
  // Render the reuse-cache, seeding it with the last-used paths so they are
  // pickable even on first run after this feature shipped (before any `change`).
  rememberPath('dit', s.dit); rememberPath('vae', s.vae);
  rememberPath('te', s.te);
  renderPathDatalists();
  // refresh slider value labels
  ['pv-steps', 'pv-cfg', 'pv-eta'].forEach((id) => $(id).dispatchEvent(new Event('input')));
  if (s.dit || s.vae || s.te) refreshPreviewCapabilities();
}

// Assemble an /api/preview request from the current UI. `overrides` lets the
// background generator swap in viewport dimensions + a fresh seed.
function buildPreviewBody(overrides = {}) {
  const size = parseInt($('pv-size').value, 10);
  const upscale = parseFloat($('pv-upscale')?.value) || 1;
  let seed = parseInt($('pv-seed').value, 10);
  if (!Number.isFinite(seed) || seed < 0) seed = Math.floor(Math.random() * 2147483647);
  return {
    prompt: $('pv-prompt').value,
    negative: $('pv-negative').value,
    sampler: $('pv-sampler').value,
    scheduler: $('pv-scheduler').value,
    steps: parseInt($('pv-steps').value, 10),
    cfg: parseFloat($('pv-cfg').value),
    eta: parseFloat($('pv-eta').value),
    width: size, height: size,
    upscale,
    seed,
    loras: layersForRequest(),       // combine every layer in the preview
    model_paths: previewModelPaths(),
    ...overrides,
  };
}

// Show the preview stage's spinner + a live elapsed-seconds counter. Returns a
// stopper. Shared by the sample preview and the (same-engine) background
// generator so both surface identical progress instead of a silent wait.
function startPreviewProgress(label = 'generating…') {
  showPreviewSpinner(true);
  const t0 = performance.now();
  $('pv-progress').textContent = label;
  const timer = setInterval(() => {
    const s = ((performance.now() - t0) / 1000).toFixed(1);
    $('pv-progress').textContent = `${label} ${s}s`;
  }, 100);
  return () => { clearInterval(timer); showPreviewSpinner(false); };
}

async function onPreview() {
  const body = buildPreviewBody();

  const btn = $('btn-preview');
  btn.disabled = true;
  const stopProgress = startPreviewProgress('generating…');

  try {
    const r = await fetch('/api/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) {
      toast(data.error || 'preview failed', 'error');
      $('pv-meta').textContent = data.error || 'preview failed';
      return;
    }
    const img = $('pv-image');
    img.src = data.image;
    img.classList.add('loaded');
    $('pv-placeholder').classList.add('hidden');
    $('pv-meta').textContent = formatPreviewMeta(data.meta);
    state.lastPreviewImage = data.image;  // enable "Use this sample" as background
    refreshBgMenu();
    toast('sample generated', 'success');
  } catch (e) {
    toast('network error', 'error');
  } finally {
    stopProgress();
    btn.disabled = false;
  }
}

function showPreviewSpinner(on) {
  $('pv-spinner').classList.toggle('hidden', !on);
  $('pv-stage').classList.toggle('busy', on);
}

function formatPreviewMeta(m) {
  const tag = `Anima · ${m.device}`;
  const n = m.lora_count || 0;
  const loraTag = n > 1 ? `  ·  ${n} LoRAs combined`
                : n === 1 ? '  ·  edited LoRA applied' : '';
  const hiresTag = m.upscale > 1
    ? `  ·  hi-res ${m.base_width}×${m.base_height}→${m.width}×${m.height} (${m.upscale}×, ${m.hires_steps} steps @ ${m.hires_denoise})`
    : `  ·  ${m.width}×${m.height}`;
  return `${tag}  ·  ${m.sampler}/${m.scheduler}  ·  ${m.steps} steps  ·  cfg ${m.cfg}  ·  ` +
         `eta ${m.eta}  ·  seed ${m.seed}${hiresTag}  ·  ${m.elapsed_s}s` +
         loraTag;
}


// ─── Quick-action dock ─────────────────────────────────────────────────────────
// Fixed rail on the right. Run-actions delegate to the same handlers as the
// inline buttons; section jumps scroll-spy via IntersectionObserver. The dock's
// own "busy" state pulses an icon while the underlying button is disabled.
function initQuickDock() {
  const dock = $('quick-dock');
  if (!dock) return;

  const runBusy = async (qdBtn, underlyingId, fn) => {
    const under = $(underlyingId);
    if (under && under.disabled) return;   // already running / not ready
    qdBtn.classList.add('is-busy');
    try { await fn(); }
    finally { qdBtn.classList.remove('is-busy'); }
  };

  dock.querySelectorAll('[data-qd]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      switch (btn.dataset.qd) {
        case 'generate':
          await runBusy(btn, 'btn-preview', onPreview);
          break;
        case 'reroll':
          $('pv-seed').value = Math.floor(Math.random() * 2147483647);
          savePreviewSettings();
          await runBusy(btn, 'btn-preview', onPreview);
          break;
        case 'switch-layer':
          cycleLayer();
          break;
        case 'save':
          await runBusy(btn, 'btn-save', onSave);
          break;
        case 'releases':
          toggleReleasesPopover(btn);
          break;
      }
    });
  });

  // Section jumps
  dock.querySelectorAll('[data-qd-jump]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = $(btn.dataset.qdJump);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  initDockScrollSpy(dock);
  updateActiveLayerLabels();
  initReleasesPopover();
}

// ─── Releases popover ──────────────────────────────────────────────────────────
// Fetches the latest few GitHub Releases on first open and renders each asset
// as a direct download link. Results are cached for the session.
const RELEASES_REPO = 'pclshm/anima-lora-editor';
let _releasesCache = null;
let _releasesFetching = null;

function toggleReleasesPopover(btn) {
  const pop = $('qd-releases-pop');
  if (!pop) return;
  const open = !pop.hidden;
  if (open) closeReleasesPopover();
  else openReleasesPopover(btn);
}

function openReleasesPopover(btn) {
  const pop = $('qd-releases-pop');
  if (!pop) return;
  pop.hidden = false;
  btn.setAttribute('aria-expanded', 'true');
  loadReleases();
}

function closeReleasesPopover() {
  const pop = $('qd-releases-pop');
  const btn = $('qd-releases-btn');
  if (pop) pop.hidden = true;
  if (btn) btn.setAttribute('aria-expanded', 'false');
}

function initReleasesPopover() {
  const close = $('qd-releases-close');
  if (close) close.addEventListener('click', closeReleasesPopover);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const pop = $('qd-releases-pop');
      if (pop && !pop.hidden) closeReleasesPopover();
    }
  });
  document.addEventListener('click', (e) => {
    const pop = $('qd-releases-pop');
    const btn = $('qd-releases-btn');
    if (!pop || pop.hidden) return;
    if (pop.contains(e.target) || (btn && btn.contains(e.target))) return;
    closeReleasesPopover();
  });
}

async function loadReleases() {
  const body = $('qd-releases-body');
  if (!body) return;
  if (_releasesCache) { renderReleases(_releasesCache); return; }
  body.innerHTML = '<p class="qd-pop-status">Loading…</p>';
  try {
    if (!_releasesFetching) {
      _releasesFetching = fetch(
        `https://api.github.com/repos/${RELEASES_REPO}/releases?per_page=3`,
        { headers: { Accept: 'application/vnd.github+json' } },
      ).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      });
    }
    const data = await _releasesFetching;
    _releasesCache = Array.isArray(data) ? data : [];
    renderReleases(_releasesCache);
  } catch (err) {
    _releasesFetching = null;
    body.innerHTML =
      '<p class="qd-pop-status">Couldn’t load releases. ' +
      `<a href="https://github.com/${RELEASES_REPO}/releases" target="_blank" rel="noopener">Open on GitHub ↗</a></p>`;
  }
}

function renderReleases(releases) {
  const body = $('qd-releases-body');
  if (!body) return;
  if (!releases.length) {
    body.innerHTML =
      '<p class="qd-pop-status">No releases published yet. ' +
      'Builds will appear here once a <code>v*</code> tag is pushed.</p>';
    return;
  }
  body.innerHTML = '';
  for (const rel of releases) {
    const card = document.createElement('div');
    card.className = 'qd-release';
    const head = document.createElement('div');
    head.className = 'qd-release-head';
    const name = document.createElement('span');
    name.className = 'qd-release-name';
    name.textContent = rel.name || rel.tag_name || 'Release';
    const tag = document.createElement('span');
    tag.className = 'qd-release-tag';
    tag.textContent = rel.tag_name || '';
    head.append(name, tag);
    card.append(head);

    const meta = document.createElement('div');
    meta.className = 'qd-release-meta';
    const when = rel.published_at ? new Date(rel.published_at).toLocaleDateString() : '';
    meta.textContent = [when, rel.prerelease ? 'pre-release' : null]
      .filter(Boolean).join(' · ');
    if (meta.textContent) card.append(meta);

    const assets = Array.isArray(rel.assets) ? rel.assets : [];
    if (assets.length) {
      const list = document.createElement('ul');
      list.className = 'qd-asset-list';
      for (const a of assets) {
        const li = document.createElement('li');
        const link = document.createElement('a');
        link.className = 'qd-asset';
        link.href = a.browser_download_url;
        link.target = '_blank';
        link.rel = 'noopener';
        link.title = `Download ${a.name}`;
        const left = document.createElement('span');
        left.className = 'qd-asset-name';
        const ico = document.createElement('span');
        ico.setAttribute('aria-hidden', 'true');
        ico.textContent = assetIcon(a.name);
        const lbl = document.createElement('span');
        lbl.textContent = a.name;
        left.append(ico, lbl);
        const size = document.createElement('span');
        size.className = 'qd-asset-size';
        size.textContent = formatBytes(a.size);
        link.append(left, size);
        li.append(link);
        list.append(li);
      }
      card.append(list);
    } else {
      const link = document.createElement('a');
      link.className = 'qd-asset';
      link.href = rel.html_url;
      link.target = '_blank';
      link.rel = 'noopener';
      link.textContent = 'View on GitHub ↗';
      card.append(link);
    }
    body.append(card);
  }
}

function assetIcon(name) {
  const n = (name || '').toLowerCase();
  if (n.includes('windows') || n.endsWith('.exe')) return '🪟';
  if (n.includes('mac') || n.includes('darwin') || n.endsWith('.dmg')) return '';
  if (n.includes('linux') || n.endsWith('.appimage')) return '🐧';
  if (n.endsWith('.zip') || n.endsWith('.tar.gz') || n.endsWith('.tgz')) return '📦';
  return '⬇';
}

function formatBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

// Advance the active LoRA layer, wrapping around; toast which one is now live.
function cycleLayer() {
  if (state.layers.length <= 1) { toast('only one layer — add another to switch', ''); return; }
  const idx = state.layers.findIndex((l) => l.id === state.activeId);
  const next = state.layers[(idx + 1) % state.layers.length];
  switchLayer(next.id);
  const nIdx = state.layers.findIndex((l) => l.id === next.id);
  toast(`layer ${nIdx + 1} · ${layerLabel(next, nIdx)}`, 'success');
}

// Highlight the dot for whichever section is currently centred in the viewport.
function initDockScrollSpy(dock) {
  const dots = Array.from(dock.querySelectorAll('[data-qd-jump]'));
  const byId = new Map(dots.map((d) => [d.dataset.qdJump, d]));
  const sections = dots
    .map((d) => document.getElementById(d.dataset.qdJump))
    .filter(Boolean);
  if (!sections.length || !('IntersectionObserver' in window)) return;

  const visible = new Map();
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) visible.set(e.target.id, e.intersectionRatio);
      else visible.delete(e.target.id);
    });
    let bestId = null, bestRatio = -1;
    visible.forEach((ratio, id) => { if (ratio > bestRatio) { bestRatio = ratio; bestId = id; } });
    dots.forEach((d) => d.classList.toggle('is-current', d.dataset.qdJump === bestId));
  }, { rootMargin: '-40% 0px -40% 0px', threshold: [0, 0.25, 0.5, 1] });

  sections.forEach((s) => obs.observe(s));
}


// ─── Create from Images (i2L) ────────────────────────────────────────────────
// Image-to-LoRA (arXiv:2606.13809): predict a style LoRA from reference images
// in one forward pass. Talks to /api/i2lora/{capabilities,create}. Like Live
// Preview it needs an external artifact (a trained predictor checkpoint) but —
// unlike preview — the forward pass is small and runs fine on CPU.

const I2L_STORE = 'anima_i2l_settings_v1';

function initI2L() {
  bindRange('i2l-mult', 'i2l-mult-val', 2);

  $('btn-i2l-create').addEventListener('click', onI2LCreate);

  const persistIds = ['i2l-checkpoint', 'i2l-images', 'i2l-siglip',
                      'i2l-output', 'i2l-mult', 'i2l-gray'];
  persistIds.forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener('change', saveI2LSettings);
  });
  // Checkpoint / SigLIP changes re-probe what the predictor needs.
  ['i2l-checkpoint', 'i2l-siglip'].forEach((id) => {
    $(id).addEventListener('change', () => { cleanPathField($(id)); refreshI2LCapabilities(); });
  });

  restoreI2LSettings();
  refreshI2LCapabilities();
}

function i2lImagePaths() {
  return $('i2l-images').value.split('\n').map((s) => cleanPath(s)).filter(Boolean);
}

function setI2LPill(stateName, text) {
  const pill = $('i2l-pill');
  pill.dataset.state = stateName;
  pill.textContent = text;
}

async function refreshI2LCapabilities() {
  try {
    const r = await fetch('/api/i2lora/capabilities', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        checkpoint: cleanPath($('i2l-checkpoint').value),
        siglip_path: cleanPath($('i2l-siglip').value),
      }),
    });
    applyI2LCapabilities(await r.json());
  } catch {
    setI2LPill('error', 'i2L offline');
  }
}

function applyI2LCapabilities(caps) {
  const hint = $('i2l-encoder-hint');
  if (caps.ready) {
    setI2LPill('ok', `ready · ${caps.device}`);
  } else if (!caps.checkpoint_present) {
    setI2LPill('idle', 'set checkpoint');
  } else if (caps.needs_siglip) {
    setI2LPill('error', 'SigLIP2 needed');
    $('i2l-advanced').open = true;
  } else {
    setI2LPill('error', 'not ready');
  }
  // Surface what the loaded predictor expects.
  const s = caps.checkpoint_summary;
  if (s) {
    const enc = caps.needs_siglip
      ? `encoder: SigLIP2 (image_embed_dim ${s.image_embed_dim})`
      : 'encoder: fallback (no SigLIP2 needed — note: not a learned encoder)';
    hint.textContent = `${s.layers} layers · ${s.blocks} blocks · rank ${s.rank} · ${enc}. ${caps.reason}.`;
  } else {
    hint.textContent = caps.reason || '';
  }
}

async function onI2LCreate() {
  const checkpoint = cleanPath($('i2l-checkpoint').value);
  const images = i2lImagePaths();
  const output = cleanPath($('i2l-output').value);
  if (!checkpoint) { toast('predictor checkpoint required', 'error'); return; }
  if (!images.length) { toast('add at least one reference image', 'error'); return; }
  if (!output) { toast('output path required', 'error'); return; }

  const btn = $('btn-i2l-create');
  btn.disabled = true;
  toast(images.length > 1 ? `fusing ${images.length} images…` : 'predicting LoRA…');

  try {
    const r = await fetch('/api/i2lora/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        checkpoint,
        image_paths: images,
        output_path: output,
        siglip_path: cleanPath($('i2l-siglip').value),
        multiplier: parseFloat($('i2l-mult').value),
        gray: $('i2l-gray').checked,
      }),
    });
    const data = await r.json();
    $('i2l-result').hidden = false;
    if (!r.ok) {
      toast(data.error || 'create failed', 'error');
      $('i2l-result').textContent = data.error || JSON.stringify(data, null, 2);
      return;
    }
    $('i2l-result').textContent = formatI2LInfo(data.info);
    rememberPath('lora_in', output);  // so it shows up in the editor's path history
    toast(`created → ${output.split(/[\\/]/).pop()}`, 'success');
  } catch (e) {
    toast('network error', 'error');
  } finally {
    btn.disabled = false;
  }
}

function formatI2LInfo(info) {
  const lines = [
    `output:                ${info.output_path}`,
    `reference images:      ${info.num_reference_images}  (${info.num_image_tokens} tokens)`,
    `adapted layers:        ${info.num_layers}`,
    `rank / alpha:          ${info.rank} / ${info.alpha}`,
    `multiplier:            ${info.multiplier}`,
    `tensors written:       ${info.output_tensor_count}`,
    `encoder:               ${info.encoder}${info.encoder_is_real ? '' : '  (fallback — NOT trained/SigLIP2)'}`,
  ];
  if (info.gray_output_path) lines.push(`gray neutral LoRA:     ${info.gray_output_path}`);
  if (!info.encoder_is_real) {
    lines.push('', 'NOTE: fallback encoder — this LoRA is structurally valid but');
    lines.push('not meaningfully stylised. Use a SigLIP2-trained checkpoint for results.');
  }
  return lines.join('\n');
}

function saveI2LSettings() {
  const s = {
    checkpoint: $('i2l-checkpoint').value, images: $('i2l-images').value,
    siglip: $('i2l-siglip').value, output: $('i2l-output').value,
    mult: $('i2l-mult').value, gray: $('i2l-gray').checked,
  };
  try { localStorage.setItem(I2L_STORE, JSON.stringify(s)); } catch { /* private mode */ }
}

function restoreI2LSettings() {
  let s;
  try { s = JSON.parse(localStorage.getItem(I2L_STORE) || 'null'); } catch { s = null; }
  if (!s) return;
  if (s.checkpoint != null) $('i2l-checkpoint').value = s.checkpoint;
  if (s.images != null) $('i2l-images').value = s.images;
  if (s.siglip != null) $('i2l-siglip').value = s.siglip;
  if (s.output != null) $('i2l-output').value = s.output;
  if (s.mult != null) { $('i2l-mult').value = s.mult; $('i2l-mult-val').textContent = (+s.mult).toFixed(2); }
  if (s.gray != null) $('i2l-gray').checked = s.gray;
}

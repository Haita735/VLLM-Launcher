const NUMBER_FIELDS = {
  port: 'int',
  tensor_parallel_size: 'int',
  pipeline_parallel_size: 'int',
  max_model_len: 'int',
  max_num_seqs: 'int',
  max_num_batched_tokens: 'int',
  block_size: 'int',
  num_gpu_blocks_override: 'int',
  gpu_memory_utilization: 'float',
  swap_space: 'float',
  cpu_offload_gb: 'float',
  // SGLang-specific numerics
  sglang_max_total_tokens: 'int',
  sglang_dp_size: 'int',
  sglang_ep_size: 'int',
  sglang_max_mamba_cache_size: 'int',
  sglang_mamba_full_memory_ratio: 'float',
  sglang_speculative_num_steps: 'int',
  sglang_speculative_eagle_topk: 'int',
  sglang_speculative_num_draft_tokens: 'int',
};

// Fields collected as a string (or '' -> null). Tri-state selects count as text:
// they emit 'on'/'off'/'' and the server maps those to flags.
const TEXT_FIELDS = [
  'host', 'served_model_name', 'api_key', 'distributed_executor_backend', 'dtype',
  'quantization', 'linear_backend', 'attention_backend', 'mamba_backend', 'mamba_cache_dtype',
  'kv_cache_dtype', 'reasoning_parser', 'tool_call_parser', 'limit_mm_per_prompt',
  'enforce_eager', 'enable_chunked_prefill', 'enable_prefix_caching', 'trust_remote_code',
  'enable_auto_tool_choice',
  // SGLang selects (free value)
  'sglang_load_format', 'sglang_schedule_policy',
  'sglang_cuda_graph_backend_decode', 'sglang_cuda_graph_backend_prefill',
  'sglang_mamba_backend', 'sglang_mamba_ssm_dtype',
  // SGLang speculative / MTP (free values — algorithm list and model path)
  'sglang_speculative_algorithm', 'sglang_speculative_draft_model',
  // SGLang tri-state booleans
  'sglang_disable_radix_cache', 'sglang_disable_overlap_schedule',
  'sglang_enable_deterministic_inference', 'sglang_enable_multimodal',
  'sglang_enable_dp_attention',
];

// How many log lines the browser keeps and renders. The server's ring buffer
// is larger, but a new SSE connection only replays this many lines.
const LOG_LINE_CAP = 1000;

const state = {
  models: [],
  selected: null,
  system: null,
  running: false,
  logSeen: new Set(),
  logLines: [],
  streamAbort: null,
  dlSeen: new Set(),
  dlLines: [],
  chat: [],
  chatBusy: false,
  chatAbort: null,
};

const $ = (id) => document.getElementById(id);
// Safe for both text nodes and double-quoted attribute values.
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtBytes = (n) => {
  if (!n) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (_) { data = { detail: text.slice(0, 300) }; }
  if (!res.ok) {
    // FastAPI validation errors arrive as a list of {loc, msg}; everything else is a string.
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d) => `${(d.loc || []).slice(1).join('.') || 'request'}: ${d.msg}`).join('; ')
      : data.detail;
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return data;
}

/* ------------------------------------------------------------------ system */
const GIB = 1024 ** 3;

// Host RAM sits in the GPU strip because it is the resource that actually takes the
// desktop down during a load (kernel JIT, CPU offload); the guard thresholds come from
// /api/system so the card reflects what the server will enforce.
function ramCard(ram) {
  if (!ram || !ram.total_bytes) return '';
  const used = ram.total_bytes - (ram.available_bytes ?? ram.total_bytes);
  const pct = Math.round((used / ram.total_bytes) * 100);
  const guard = state.system?.ram_guard || {};
  const avail = (ram.available_bytes ?? 0) / GIB;
  const low = guard.min_free_gib && avail < guard.min_free_gib;
  const swap = ram.swap_total_bytes
    ? ` · swap ${fmtBytes(ram.swap_total_bytes - (ram.swap_free_bytes ?? 0))}/${fmtBytes(ram.swap_total_bytes)}`
    : '';
  const psi = ram.psi_full_avg10 != null && ram.psi_full_avg10 >= 1 ? ` · stall ${ram.psi_full_avg10.toFixed(0)}%` : '';
  // A systemd/Docker memory cap on the launcher applies to the engine too and is usually
  // the tighter bound, so show it next to the host figure.
  const cg = ram.cgroup || {};
  const cap = cg.high_bytes ?? cg.max_bytes;
  const cgroup = cap ? ` · cgroup ${fmtBytes(cg.anon_bytes ?? cg.current_bytes ?? 0)}/${fmtBytes(cap)}` : '';
  return `<div class="gpu-card ram-card${low ? ' low' : ''}" title="Launch refuses below ${guard.min_free_gib ?? '?'} GiB free; the watchdog kills the engine below ${guard.kill_gib ?? '?'} GiB or when the host is stalled on memory ${guard.psi_full_kill ?? '?'}% of the time${cap ? `. The launcher's cgroup is capped at ${fmtBytes(cg.max_bytes || cap)} (throttled above ${fmtBytes(cap)}); the engine inherits that cap.` : ''}">
    <div class="gpu-name"><span>Host RAM</span><span>${low ? 'low for launch' : `${avail.toFixed(1)} GiB free`}</span></div>
    <div class="bar"><span style="width:${pct}%"></span></div>
    <div class="gpu-meta">${fmtBytes(used)} / ${fmtBytes(ram.total_bytes)}${swap}${cgroup}${psi}</div>
  </div>`;
}

function renderGpus(gpus, ram) {
  $('gpu-strip').innerHTML = gpus.map((g) => {
    const pct = g.memory_total_mb ? Math.round((g.memory_used_mb / g.memory_total_mb) * 100) : 0;
    // Unified-memory parts (GB10) have no VRAM figure: the Host RAM card is the one that counts.
    const mem = g.unified_memory
      ? 'unified memory (see Host RAM)'
      : `${fmtBytes(g.memory_used_mb * 1048576)} / ${fmtBytes(g.memory_total_mb * 1048576)}`;
    return `<div class="gpu-card">
      <div class="gpu-name"><span>GPU ${g.index} · ${esc(g.name)}</span><span>sm_${g.compute_cap.replace('.', '')}</span></div>
      <div class="bar"><span style="width:${pct}%"></span></div>
      <div class="gpu-meta">${mem} · ${g.utilization}% · ${g.temperature}°C</div>
    </div>`;
  }).join('') + ramCard(ram);
}

function renderGpuPicker(gpus) {
  $('gpu-picker').innerHTML = gpus.map((g) => `
    <label><input type="checkbox" class="gpu-check" value="${g.index}" checked /> ${g.index}</label>
  `).join('');
  $('gpu-picker').querySelectorAll('.gpu-check').forEach((el) => el.addEventListener('change', schedulePreview));
}

// Which interpreter the selected engine will run under, or why launching it would fail.
function renderEngineHint() {
  const engine = $('f-engine').value || 'vllm';
  const info = state.system?.engines?.[engine];
  const hint = $('engine-hint');
  if (!info) { hint.textContent = ''; return; }
  if (!info.available) {
    hint.textContent = `${engine} is not installed on this machine - set ${engine.toUpperCase()}_PYTHON to the interpreter of the env that has it and restart the launcher.`;
    hint.classList.add('bad');
    return;
  }
  hint.classList.remove('bad');
  const versions = [info.version ? `v${info.version}` : null, info.torch ? `torch ${info.torch}` : null,
    info.cuda ? `CUDA ${info.cuda}` : null].filter(Boolean).join(', ');
  hint.textContent = `${engine}: ${info.python}${versions ? `  (${versions})` : ''}`;
}

async function loadSystem() {
  const sys = await api('/api/system');
  state.system = sys;
  const v = sys.versions || {};
  const sm = (sys.capability || '').replace('.', '') || '?';
  $('sys-subtitle').textContent =
    `vLLM ${v.vllm || '—'} · SGLang ${v.sglang || '—'} · torch ${v.torch || '?'} · CUDA ${v.cuda || '?'} · sm_${sm}`;
  renderEngineHint();
  renderGpus(sys.gpus || [], sys.ram);
  renderGpuPicker(sys.gpus || []);
  $('dl-hf-home').textContent = sys.hf_home || '(unset)';
  $('access-urls').innerHTML = (sys.access_urls || [])
    .map((url) => `<a href="${esc(url)}" class="url-chip">${esc(url.replace('http://', ''))}</a>`).join('');
}

async function pollGpus() {
  try {
    const { gpus, ram } = await api('/api/gpus');
    renderGpus(gpus, ram);
  } catch (_) { /* transient */ }
}

/* ------------------------------------------------------------------ models */
function modelCard(model) {
  const q = model.quantization || {};
  const tags = [];
  tags.push(`<span class="tag size">${fmtBytes(model.size_bytes)}</span>`);
  if (q.method) tags.push(`<span class="tag quant">${q.detail || q.method}</span>`);
  if (model.dtype) tags.push(`<span class="tag">${model.dtype}</span>`);
  if (model.max_position_embeddings) tags.push(`<span class="tag">${(model.max_position_embeddings / 1024).toFixed(0)}K ctx</span>`);
  if (model.multimodal) tags.push('<span class="tag mm">multimodal</span>');
  if (model.gguf_files.length) tags.push('<span class="tag">gguf</span>');
  // Per-engine saved-profile chips, so a user can see "saved for vLLM" vs "saved for SGLang"
  // at a glance. Legacy files only have vllm populated (auto-migrated server-side).
  const engineProfiles = state.profiles?.[model.id] || {};
  for (const [eng, data] of Object.entries(engineProfiles)) {
    tags.push(`<span class="tag saved" title="${esc((data?.spec ? 'saved' : '') + ' for ' + eng)}">⚙ ${eng}</span>`);
  }

  const dl = model.download || {};
  if (dl.state === 'missing') {
    tags.push('<span class="tag missing">not downloaded</span>');
  } else if (dl.state === 'partial') {
    const detail = dl.shards_expected ? `${dl.shards_present}/${dl.shards_expected} shards` : 'incomplete';
    tags.push(`<span class="tag partial">partial · ${detail}</span>`);
  } else if (dl.shards_expected) {
    tags.push(`<span class="tag ok">${dl.shards_expected} shards</span>`);
  }

  return `<div class="model-card state-${dl.state || 'ok'}${state.selected?.id === model.id ? ' active' : ''}" data-id="${esc(model.id)}">
    <div class="name">${esc(model.id)}</div>
    <div class="meta">${tags.join('')}</div>
    <div class="card-actions"><button class="ghost danger" data-del="${esc(model.id)}">delete</button></div>
  </div>`;
}

function renderModels() {
  const filter = $('model-filter').value.trim().toLowerCase();
  const visible = state.models.filter((m) => !filter || m.id.toLowerCase().includes(filter));
  $('model-list').innerHTML = visible.length
    ? visible.map(modelCard).join('')
    : `<div class="note">No models to show.${state.incompleteCount ? ` ${state.incompleteCount} incomplete hidden.` : ''}</div>`;
  $('model-list').querySelectorAll('.model-card').forEach((card) => {
    card.addEventListener('click', () => selectModel(card.dataset.id));
  });
  $('model-list').querySelectorAll('[data-del]').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = btn.dataset.del;
      const model = state.models.find((m) => m.id === id);
      const size = fmtBytes(model?.size_bytes || 0);
      // Large complete checkpoints are slow to re-fetch, so make the confirmation explicit.
      const heavy = (model?.size_bytes || 0) > 1024 ** 3;
      if (heavy) {
        const typed = prompt(`Permanently delete ${id} (${size}) from disk?\n\nRe-downloading takes a while. Type DELETE to confirm.`);
        if (typed !== 'DELETE') return;
      } else if (!confirm(`Delete ${id} (${size}) from disk?`)) {
        return;
      }
      try {
        const res = await api('/api/models/delete', { method: 'POST', body: JSON.stringify({ models: [id] }) });
        if (state.selected?.id === id) {
          state.selected = null;
          $('selected-model').textContent = 'No model selected';
          $('model-notes').innerHTML = '';
          $('cmd-preview').textContent = 'select a model…';
        }
        await loadModels(true);
        $('profile-status').textContent = `deleted ${id} — freed ${fmtBytes(res.freed_bytes)}`;
      } catch (err) { alert(err.message); }
    });
  });
}

async function loadModels(refresh = false) {
  const params = new URLSearchParams();
  if (refresh) params.set('refresh', 'true');
  if ($('show-incomplete').checked) params.set('include_missing', 'true');
  const data = await api(`/api/models?${params}`);
  state.models = data.models;
  state.incompleteCount = data.incomplete_count || 0;
  renderModels();
}

function selectModel(id) {
  const model = state.models.find((m) => m.id === id);
  if (!model) return;
  state.selected = model;
  $('selected-model').textContent = model.path;
  $('model-notes').innerHTML = (model.notes || [])
    .map((n) => `<div class="note ${n.level}">${esc(n.text)}</div>`).join('');

  const engineProfiles = state.profiles?.[model.id] || {};
  const engine = $('f-engine').value || 'vllm';
  const savedForEngine = engineProfiles[engine];
  const otherEngines = Object.keys(engineProfiles).filter((e) => e !== engine);

  if (savedForEngine) {
    applyFields(savedForEngine.spec);
    state.autoServedName = null;
    const when = new Date(savedForEngine.saved_at * 1000).toLocaleString();
    const note = otherEngines.length ? ` · other engine(s) saved: ${otherEngines.join(', ')}` : '';
    $('profile-status').textContent = `${engine} config restored (${when})${note}`;
  } else {
    // No saved config for this engine. If another engine has one, seed from it so a
    // user switching engines starts from a plausible base (shared fields carry over)
    // while SGLang-only fields stay empty. Otherwise fall back to suggested defaults.
    const seedFrom = otherEngines.length ? engineProfiles[otherEngines[0]] : null;
    if (seedFrom) {
      // Copy only shared / vLLM-shaped fields; drop SGLang-specific ones.
      const shared = { ...seedFrom.spec };
      Object.keys(shared).forEach((k) => { if (k.startsWith('sglang_') || k === 'engine') delete shared[k]; });
      applyFields(shared);
      // Clear the SGLang-only form controls so stale values don't leak in.
      [...Object.keys(NUMBER_FIELDS), ...TEXT_FIELDS].filter((n) => n.startsWith('sglang_')).forEach((n) => { const el=$(`f-${n}`); if (el) el.value = ''; });
      state.autoServedName = null;
      $('profile-status').textContent = `no ${engine} config — seeded from ${otherEngines.join(', ')}`;
    } else {
      applySuggestedDefaults(model);
      $('profile-status').textContent = 'no saved config - using suggested defaults';
    }
  }

  syncEngineUi();
  renderModels();
  $('launch-btn').disabled = state.running || model.download?.state === 'missing';
  schedulePreview();
}

/** Pre-fill flags that this checkpoint plus this GPU generation effectively require. */
function applySuggestedDefaults(model) {
  const cap = state.system?.capability_int || 0;
  const quant = JSON.stringify(model.quantization || {}).toLowerCase();

  if (!$('f-dtype').value && cap < 80 && (model.dtype || '').toLowerCase() === 'bfloat16') {
    $('f-dtype').value = 'float16';
  }
  // linear-backend is a vLLM flag (SGLang ignores it, and the preview would say so).
  if (!$('f-linear_backend').value && cap < 100 && /nvfp4|fp4/.test(quant) && $('f-engine').value !== 'sglang') {
    $('f-linear_backend').value = 'marlin';
  }
  if (!$('f-tensor_parallel_size').value && (state.system?.gpu_count || 1) > 1) {
    $('f-tensor_parallel_size').value = state.system.gpu_count;
  }
  if (!$('f-gpu_memory_utilization').value) $('f-gpu_memory_utilization').value = '0.90';

  // Served name tracks the selection unless it was hand-edited.
  const autoName = model.id.split('/').pop();
  const current = $('f-served_model_name').value.trim();
  if (!current || current === state.autoServedName) {
    $('f-served_model_name').value = autoName;
  }
  state.autoServedName = autoName;
}

/* ------------------------------------------------------------------ form */
function parseEnv(text) {
  const env = {};
  text.split('\n').forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) return;
    const idx = trimmed.indexOf('=');
    if (idx < 1) return;
    env[trimmed.slice(0, idx).trim()] = trimmed.slice(idx + 1).trim();
  });
  return env;
}

function collectSpec() {
  if (!state.selected) return null;
  const spec = { model: state.selected.id, engine: $('f-engine').value || 'vllm' };

  TEXT_FIELDS.forEach((name) => {
    const value = $(`f-${name}`).value.trim();
    spec[name] = value || null;
  });
  Object.entries(NUMBER_FIELDS).forEach(([name, kind]) => {
    const el = $(`f-${name}`);
    if (!el) return;
    const raw = el.value.trim();
    if (!raw) { spec[name] = null; return; }
    const value = kind === 'int' ? parseInt(raw, 10) : parseFloat(raw);
    spec[name] = Number.isNaN(value) ? null : value;
  });

  spec.host = spec.host || '0.0.0.0';
  spec.port = spec.port || 8000;
  spec.gpu_indices = [...document.querySelectorAll('.gpu-check')]
    .filter((el) => el.checked).map((el) => parseInt(el.value, 10));
  spec.extra_args = $('f-extra_args').value.trim();
  spec.env = parseEnv($('f-env').value);
  return spec;
}

function applyFields(spec) {
  if (typeof spec.engine === 'string' && spec.engine) $('f-engine').value = spec.engine;
  TEXT_FIELDS.forEach((name) => {
    const el = $(`f-${name}`);
    if (el) el.value = spec[name] ?? '';
  });
  Object.keys(NUMBER_FIELDS).forEach((name) => {
    const el = $(`f-${name}`);
    if (el) el.value = spec[name] ?? '';
  });
  $('f-extra_args').value = spec.extra_args || '';
  $('f-env').value = Object.entries(spec.env || {}).map(([k, v]) => `${k}=${v}`).join('\n');
  document.querySelectorAll('.gpu-check').forEach((el) => {
    el.checked = !spec.gpu_indices?.length || spec.gpu_indices.includes(parseInt(el.value, 10));
  });
  if (spec.engine) syncEngineUi();
}

function applySpec(spec) {
  applyFields(spec);
  if (spec.model) {
    selectModel(spec.model);
    applyFields(spec); // selectModel may have restored the per-model profile
    schedulePreview();
  }
}

let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPreview, 220);
}

// Show/hide the per-engine field blocks to match the selected engine. Shared
// flag sections (dtype, memory, behaviour …) stay visible for both.
function syncEngineUi() {
  const engine = $('f-engine').value || 'vllm';
  document.querySelectorAll('.engine-block[data-engine]').forEach((block) => {
    block.hidden = block.dataset.engine !== engine;
  });
}

async function refreshPreview() {
  const spec = collectSpec();
  if (!spec) return;
  try {
    const { command, env, notes } = await api('/api/preview', { method: 'POST', body: JSON.stringify(spec) });
    const envLine = Object.entries(env).map(([k, v]) => `${k}=${v}`).join(' ');
    $('cmd-preview').textContent = envLine ? `${envLine} \\\n  ${command}` : command;
    // Flag translations, dropped fields and the RAM-derived JIT sizing the server applied.
    $('preview-notes').innerHTML = (notes || [])
      .map((n) => `<div class="note ${/ignored|dropped|no equivalent|warning|exceeds/i.test(n) ? 'warn' : 'info'}">${esc(n)}</div>`).join('');
  } catch (err) {
    $('cmd-preview').textContent = `# ${err.message}`;
    $('preview-notes').innerHTML = '';
  }
}

/* ------------------------------------------------------------------ runtime */
function setStatus(status) {
  state.running = status.running;
  state.engine = status.engine || null;
  const pill = $('status-pill');
  const online = status.endpoint?.online;
  const external = !!status.external;
  pill.className = 'pill';
  const engineTag = status.engine ? ` · ${status.engine}` : '';
  // A kernel JIT build is the one loading phase where the engine prints nothing for
  // many minutes; name it in the pill so the page never looks hung.
  const jit = status.jit;
  const jitTag = jit ? ` · compiling kernels ${jit.done}/${jit.total}` : '';
  if (status.running && online) {
    pill.classList.add('running');
    pill.textContent = external ? `serving · external${engineTag}` : `serving${engineTag}`;
  }
  else if (status.running) { pill.classList.add('starting'); pill.textContent = `loading${engineTag}${jitTag}`; }
  else if (status.returncode) { pill.classList.add('error'); pill.textContent = `exit ${status.returncode}`; }
  else { pill.textContent = 'idle'; }
  pill.title = jit
    ? `${jit.op}: ${jit.done}/${jit.total} compile steps, ${jit.compilers} compiler process(es), ${Math.round(jit.elapsed_s / 60)} min so far. One-time; results are cached.`
    : '';

  const owner = external ? 'not owned by launcher' : `pid ${status.pid}`;
  $('endpoint-label').textContent = status.running
    ? `${owner} · :${status.port}${online ? ` · ${status.endpoint.models.join(', ')}` : ''}`
    : '';
  $('launch-btn').disabled = status.running
    || !state.selected
    || state.selected.download?.state === 'missing';
  $('stop-btn').disabled = !status.running || external;
  $('stop-btn').title = external
    ? 'Started outside this launcher - stop it from wherever it was launched.'
    : '';
  const models = status.endpoint?.models || [];
  $('chat-model').textContent = chatTarget(status, models);
}

// "model · engine · :port" so it is obvious which server the chat proxy will hit.
function chatTarget(status, models) {
  if (!status.running) return 'no model loaded';
  const where = [status.engine || (status.external ? 'external' : null), status.port ? `:${status.port}` : null]
    .filter(Boolean).join(' · ');
  return `${models[0] || 'model loading…'}${where ? ` · ${where}` : ''}`;
}

async function pollStatus() {
  try { setStatus(await api('/api/status')); } catch (_) { /* transient */ }
}

function appendLog(event) {
  appendLogs([event]);
}

// Coalesce redraws into one per animation frame: a backlog replay arrives
// across several network chunks, and rebuilding innerHTML per chunk would
// still stutter the page.
let logRenderQueued = false;
function scheduleLogRender() {
  if (logRenderQueued) return;
  logRenderQueued = true;
  requestAnimationFrame(() => {
    logRenderQueued = false;
    renderLogs();
  });
}

// Apply a batch of events; the actual redraw is coalesced via scheduleLogRender.
// The server replays a capped backlog on connect — rebuilding the terminal's
// innerHTML once per line (the original code) froze the page on load.
function appendLogs(events) {
  let grew = false;
  for (const event of events) {
    if (state.logSeen.has(event.sequence)) continue;
    state.logSeen.add(event.sequence);
    state.logLines.push(event);
    grew = true;
  }
  if (!grew) return;
  if (state.logLines.length > LOG_LINE_CAP) {
    state.logLines.splice(0, state.logLines.length - LOG_LINE_CAP);
    // Keep the dedupe set bounded to what is still rendered; sequences are monotonic so
    // anything older can never be replayed again.
    state.logSeen = new Set(state.logLines.map((e) => e.sequence));
  }
  scheduleLogRender();
}

function classify(line) {
  if (line.startsWith('$ ')) return 'cmd';
  if (/\b(ERROR|CRITICAL|Traceback|Error:)\b/.test(line)) return 'err';
  if (/\bWARNING\b/.test(line)) return 'warn';
  return '';
}

function renderLogs() {
  const term = $('terminal');
  if (!state.logLines.length) { term.textContent = 'No runtime output yet.'; return; }
  term.innerHTML = state.logLines.map((e) => {
    const stamp = new Date(e.timestamp * 1000).toLocaleTimeString();
    const cls = classify(e.line);
    const text = `${stamp}  ${e.line}`.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    return cls ? `<span class="${cls}">${text}</span>` : text;
  }).join('\n');
  if ($('autoscroll').checked) term.scrollTop = term.scrollHeight;
}

async function streamLogs() {
  state.streamAbort?.abort();
  const controller = new AbortController();
  state.streamAbort = controller;
  try {
    const res = await fetch('/api/logs/stream', { signal: controller.signal });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      const batch = [];
      lines.forEach((line) => {
        if (!line.startsWith('data: ')) return;
        try { batch.push(JSON.parse(line.slice(6))); } catch (_) { /* partial frame */ }
      });
      if (batch.length) appendLogs(batch);
    }
  } catch (err) {
    if (err.name !== 'AbortError') setTimeout(streamLogs, 3000);
    return;
  }
  setTimeout(streamLogs, 1500);
}

/* ------------------------------------------------------------------ tabs */
function initTabs() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
      ['launcher', 'download', 'chat'].forEach((name) => {
        $(`tab-${name}`).hidden = name !== tab.dataset.tab;
      });
      if (tab.dataset.tab === 'chat') refreshChatModel();
    });
  });
}

/* ------------------------------------------------------------------ download */
let resolveTimer = null;

async function resolveRepo() {
  const raw = $('dl-repo').value.trim();
  const box = $('dl-resolved');
  if (!raw) { box.textContent = ''; box.className = 'resolved'; return; }
  try {
    const { repo } = await api('/api/download/resolve', { method: 'POST', body: JSON.stringify({ repo: raw }) });
    box.textContent = `→ ${repo}`;
    box.className = 'resolved';
  } catch (err) {
    box.textContent = err.message;
    box.className = 'resolved bad';
  }
}

function fmtDur(s) {
  if (!Number.isFinite(s) || s <= 0) return '—';
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function renderDownloadProgress(status) {
  const wrap = $('dl-progress');
  const p = status.progress;
  // Show the bar while a download is running or has finished successfully; otherwise idle.
  const show = status.running || (status.repo !== null && status.returncode === 0 && p);
  wrap.hidden = !show;
  if (!show || !p) return;

  const total = p.total || 0, done = p.done || 0;
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  const fill = $('dl-progress-fill');
  const label = $('dl-progress-label'), detail = $('dl-progress-detail'), eta = $('dl-progress-eta');

  if (status.running) {
    const rate = p.rate || 0;
    label.textContent = total > 0 ? `${pct.toFixed(1)}% · ${fmtBytes(done)} / ${fmtBytes(total)}` : `downloading… ${fmtBytes(done)}`;
    eta.textContent = [
      p.eta ? `ETA ${fmtDur(p.eta)}` : null,
      rate > 0 ? `${(rate / 1048576).toFixed(1)} MiB/s` : null,
    ].filter(Boolean).join(' · ');
    fill.classList.toggle('indeterminate', total === 0);
    fill.style.width = `${pct}%`;
    const act = (p.files || []).filter((f) => f.state === 'active').length;
    detail.textContent = p.files?.length ? `${(p.files || []).filter((f) => f.state === 'done').length}/${p.files.length} files, ${act} in flight` : `elapsed ${Math.round(status.elapsed || 0)}s`;
  } else {
    label.textContent = `complete · ${fmtBytes(done)}${total ? ` (${fmtBytes(total)})` : ''}`;
    eta.textContent = status.repo ? `finish ${new Date().toLocaleTimeString()}` : '';
    fill.classList.remove('indeterminate');
    fill.style.width = '100%';
    detail.textContent = '';
  }
}

function setDownloadStatus(status) {
  $('dl-status').textContent = status.running
    ? `downloading ${status.repo}… ${Math.round(status.elapsed || 0)}s`
    : (status.repo ? `${status.repo} — exit ${status.returncode}` : 'idle');
  $('dl-start').disabled = status.running;
  $('dl-cancel').disabled = !status.running;
  renderDownloadProgress(status);
}

let dlRenderQueued = false;
function scheduleDlRender() {
  if (dlRenderQueued) return;
  dlRenderQueued = true;
  requestAnimationFrame(() => {
    dlRenderQueued = false;
    renderDownloadLogs();
  });
}

function renderDownloadLogs() {
  const term = $('dl-terminal');
  if (!state.dlLines.length) { term.textContent = 'No download output yet.'; return; }
  term.innerHTML = state.dlLines.map((e) => {
    const cls = classify(e.line);
    const text = e.line.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    return cls ? `<span class="${cls}">${text}</span>` : text;
  }).join('\n');
  term.scrollTop = term.scrollHeight;
}

async function streamDownloadLogs() {
  try {
    const res = await fetch('/api/download/logs/stream');
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      lines.forEach((line) => {
        if (!line.startsWith('data: ')) return;
        try {
          const event = JSON.parse(line.slice(6));
          if (state.dlSeen.has(event.sequence)) return;
          state.dlSeen.add(event.sequence);
          state.dlLines.push(event);
        } catch (_) { /* partial frame */ }
      });
      // Coalesce into one render per frame: huggingface-cli is chatty and the
      // buffer replay on reconnect is even chattier — per-line rebuilds froze
      // the page, and one render per animation frame is the safe floor.
      if (state.dlLines.length) {
        if (state.dlLines.length > LOG_LINE_CAP) {
          state.dlLines.splice(0, state.dlLines.length - LOG_LINE_CAP);
          state.dlSeen = new Set(state.dlLines.map((e) => e.sequence));
        }
        scheduleDlRender();
      }
    }
  } catch (_) { /* reconnect below */ }
  setTimeout(streamDownloadLogs, 2000);
}

/* ------------------------------------------------------------------ chat */
async function refreshChatModel() {
  try {
    const status = await api('/api/status');
    $('chat-model').textContent = chatTarget(status, status.endpoint?.models || []);
  } catch (_) { /* transient */ }
}

function chatParams() {
  const num = (id, kind = 'float') => {
    const raw = $(`chat-${id}`).value.trim();
    if (!raw) return null;
    const v = kind === 'int' ? parseInt(raw, 10) : parseFloat(raw);
    return Number.isNaN(v) ? null : v;
  };
  const stops = $('chat-stop-seq').value.split('\n').map((s) => s.trim()).filter(Boolean);
  return {
    temperature: num('temperature'),
    top_p: num('top_p'),
    top_k: num('top_k', 'int'),
    max_tokens: num('max_tokens', 'int'),
    presence_penalty: num('presence_penalty'),
    frequency_penalty: num('frequency_penalty'),
    repetition_penalty: num('repetition_penalty'),
    seed: num('seed', 'int'),
    stop: stops.length ? stops : null,
    enable_thinking: $('chat-thinking').checked,
    stream: $('chat-stream').checked,
  };
}

// Models served without a reasoning parser emit their thinking inline as <think>...</think>
// (Qwen3, DeepSeek-R1 templates put the opening tag in the prompt, so it may be absent).
// Re-derived from the whole accumulated text on every chunk so tags split across
// deltas are handled without a tokenizer-level state machine.
function splitThink(raw) {
  const open = raw.indexOf('<think>');
  const close = raw.indexOf('</think>');
  if (close >= 0 && (open < 0 || open < close)) {
    const start = open >= 0 ? open + '<think>'.length : 0;
    return { reasoning: raw.slice(start, close), content: raw.slice(close + '</think>'.length), thinking: false };
  }
  if (open >= 0 && !raw.slice(0, open).trim()) {
    return { reasoning: raw.slice(open + '<think>'.length), content: '', thinking: true };
  }
  return { reasoning: '', content: raw, thinking: false };
}

// Engines report streamed failures as {"error": {"message": ...}}; the launcher's own
// relay errors are plain strings.
function errorText(err) {
  if (typeof err === 'string') return err;
  if (err && typeof err === 'object') return err.message || err.detail || JSON.stringify(err);
  return String(err);
}

async function readError(res) {
  const text = await res.text();
  try { return JSON.parse(text).detail || text; } catch (_) { return text || res.statusText; }
}

function thinkBlock(m, i) {
  if (!$('chat-show-reasoning').checked || !m.reasoning) return '';
  const streaming = m.pending && !m.content;
  const secs = m.thinkStart && m.thinkEnd ? ((m.thinkEnd - m.thinkStart) / 1000).toFixed(1) : null;
  const label = streaming ? 'Thinking' : (secs ? `Thought for ${secs}s` : 'Thought process');
  return `<details class="think${streaming ? ' live' : ''}" data-idx="${i}"${m.thinkOpen ? ' open' : ''}>`
    + `<summary>${label}</summary>`
    + `<div class="think-body">${esc(m.reasoning)}</div>`
    + '</details>';
}

function renderChat() {
  const box = $('chat-messages');
  box.innerHTML = state.chat.map((m, i) => {
    const body = esc((m.content || '').replace(/^\s+/, ''));
    // While only reasoning has arrived, the disclosure itself is the progress indicator.
    const hideBubble = m.pending && !m.content && m.reasoning;
    const stopped = m.stopped ? '<span class="muted-text"> [stopped]</span>' : '';
    // finish_reason=length means the engine hit max_tokens mid-answer (thinking models spend
    // the budget reasoning first); without this the truncated reply reads as a bad model.
    const truncated = m.finishReason === 'length' ? '<span class="muted-text"> [cut off: max tokens reached - raise Max tokens]</span>' : '';
    const bubble = hideBubble
      ? ''
      : `<div class="bubble">${body}${m.pending ? '<span class="cursor-blink">▋</span>' : stopped + truncated}</div>`;
    return `<div class="msg ${m.role}${m.error ? ' error' : ''}">
      <span class="who">${m.role}</span>
      ${thinkBlock(m, i)}
      ${bubble}
      <div class="row-actions"><button class="ghost" data-copy="${i}">copy</button></div>
    </div>`;
  }).join('');

  box.querySelectorAll('[data-copy]').forEach((btn) => {
    btn.addEventListener('click', () => navigator.clipboard?.writeText(state.chat[btn.dataset.copy].content || ''));
  });
  box.querySelectorAll('details.think').forEach((el) => {
    el.addEventListener('toggle', () => {
      const m = state.chat[el.dataset.idx];
      if (!m || m.thinkOpen === el.open) return;
      m.thinkOpen = el.open;
      m.thinkPinned = true; // manual choice wins over auto-collapse
    });
  });
  box.querySelectorAll('.think.live .think-body').forEach((el) => { el.scrollTop = el.scrollHeight; });
  box.scrollTop = box.scrollHeight;
}

let renderQueued = false;
function scheduleChatRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; renderChat(); });
}

function buildMessages() {
  const messages = [];
  const system = $('chat-system').value.trim();
  if (system) messages.push({ role: 'system', content: system });
  const turns = state.chat;
  for (let i = 0; i < turns.length; i++) {
    const m = turns[i];
    if (m.role === 'user') {
      // A question whose answer never arrived (error, or stopped before the first token)
      // would leave two user turns back to back; drop the pair. The pending placeholder
      // after the newest question is not an answer yet, so that one stays.
      const reply = turns[i + 1];
      if (reply && reply.role === 'assistant' && !reply.pending && (reply.error || !reply.content.trim())) {
        i += 1;
        continue;
      }
      messages.push({ role: 'user', content: m.content });
    } else if (m.role === 'assistant' && !m.error && m.content.trim()) {
      messages.push({ role: 'assistant', content: m.content });
    }
  }
  return messages;
}

async function sendChat(reuseLast = false) {
  if (state.chatBusy) return;
  const input = $('chat-input');
  if (!reuseLast) {
    const text = input.value.trim();
    if (!text) return;
    state.chat.push({ role: 'user', content: text });
    input.value = '';
  }

  const params = chatParams();
  const assistant = { role: 'assistant', content: '', reasoning: '', raw: '', pending: true, thinkOpen: false };
  state.chat.push(assistant);
  state.chatBusy = true;
  $('chat-send').disabled = true;
  $('chat-stop').disabled = false;
  renderChat();

  const controller = new AbortController();
  state.chatAbort = controller;
  const started = performance.now();
  let tokens = 0;
  let parsedReasoning = false; // the engine's reasoning parser is active: never split tags ourselves

  const noteReasoning = () => {
    if (!assistant.thinkStart) {
      assistant.thinkStart = performance.now();
      if (!assistant.thinkPinned) assistant.thinkOpen = true;
    }
  };
  const noteContent = () => {
    if (assistant.reasoning && !assistant.thinkEnd) {
      assistant.thinkEnd = performance.now();
      if (!assistant.thinkPinned) assistant.thinkOpen = false;
    }
  };
  // Content deltas are accumulated raw and re-split so inline <think> tags work too.
  const applyContent = () => {
    if (parsedReasoning) { assistant.content = assistant.raw; return; }
    const parts = splitThink(assistant.raw);
    if (parts.reasoning) { assistant.reasoning = parts.reasoning; noteReasoning(); }
    assistant.content = parts.content;
    if (parts.content && !parts.thinking) noteContent();
  };

  try {
    // buildMessages() already leaves out the pending (empty) assistant placeholder, so the
    // history ends with the newest user turn. Slicing the last entry off here used to drop
    // that user message instead -> "Messages cannot be empty" on the first send.
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: buildMessages(), ...params }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(await readError(res));

    if (!params.stream) {
      const data = await res.json();
      if (data.error) throw new Error(errorText(data.error));
      const msg = data.choices?.[0]?.message;
      if (!msg) throw new Error('The engine returned no choices');
      const think = msg.reasoning ?? msg.reasoning_content;
      if (think) {
        parsedReasoning = true;
        assistant.reasoning = think;
      }
      assistant.raw = msg.content || '';
      applyContent();
      assistant.finishReason = data.choices[0].finish_reason || null;
      if (!assistant.thinkPinned) assistant.thinkOpen = false;
      tokens = data.usage?.completion_tokens || 0;
    } else {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      outer: for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') break outer;
          let json;
          try { json = JSON.parse(payload); } catch (_) { continue; }
          if (json.error) throw new Error(errorText(json.error));
          const choice = json.choices?.[0] || {};
          const delta = choice.delta || {};
          if (choice.finish_reason) assistant.finishReason = choice.finish_reason;
          // vLLM 0.26 streams `reasoning`; SGLang and older builds use `reasoning_content`.
          const think = delta.reasoning ?? delta.reasoning_content;
          if (think) {
            parsedReasoning = true;
            noteReasoning();
            assistant.reasoning += think;
          }
          if (delta.content) {
            assistant.raw += delta.content;
            if (parsedReasoning) noteContent();
            applyContent();
          }
          if (json.usage?.completion_tokens) tokens = json.usage.completion_tokens;
          scheduleChatRender();
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      assistant.stopped = true;
    } else {
      assistant.error = true;
      assistant.content = `Error: ${err.message}`;
    }
  } finally {
    assistant.pending = false;
    state.chatBusy = false;
    state.chatAbort = null;
    $('chat-send').disabled = false;
    $('chat-stop').disabled = true;
    const secs = (performance.now() - started) / 1000;
    $('chat-stats').textContent = tokens
      ? `${tokens} tokens · ${secs.toFixed(1)}s · ${(tokens / secs).toFixed(1)} tok/s`
      : `${secs.toFixed(1)}s`;
    renderChat();
  }
}

/* ------------------------------------------------------------------ presets */
async function loadProfiles() {
  const { profiles } = await api('/api/profiles');
  state.profiles = profiles;
  renderModels();
}

async function loadPresets(selectId = '') {
  const { presets } = await api('/api/presets');
  $('preset-select').innerHTML = '<option value="">Presets…</option>'
    + presets.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join('');
  $('preset-select').value = selectId;
  state.presets = presets;
}

/* ------------------------------------------------------------------ wiring */
function initTriStates() {
  document.querySelectorAll('select.tri').forEach((el) => {
    el.innerHTML = '<option value="">default</option><option value="on">on</option><option value="off">off</option>';
  });
}

function init() {
  initTriStates();
  initTabs();

  $('f-engine').addEventListener('change', () => {
    syncEngineUi();
    renderEngineHint();
    if (state.selected) selectModel(state.selected.id); // re-apply saved/profile for this engine
    schedulePreview();
  });
  syncEngineUi();

  $('dl-repo').addEventListener('input', () => {
    clearTimeout(resolveTimer);
    resolveTimer = setTimeout(resolveRepo, 300);
  });
  $('dl-start').addEventListener('click', async () => {
    try {
      setDownloadStatus(await api('/api/download', {
        method: 'POST',
        body: JSON.stringify({
          repo: $('dl-repo').value.trim(),
          revision: $('dl-revision').value.trim(),
          include: $('dl-include').value.trim(),
          endpoint: $('dl-endpoint').value.trim(),
        }),
      }));
    } catch (err) { alert(err.message); }
  });
  $('dl-cancel').addEventListener('click', async () => {
    try { setDownloadStatus(await api('/api/download/cancel', { method: 'POST' })); }
    catch (err) { alert(err.message); }
  });
  $('dl-clear').addEventListener('click', () => { state.dlLines = []; scheduleDlRender(); });

  $('chat-send').addEventListener('click', () => sendChat());
  $('chat-stop').addEventListener('click', () => state.chatAbort?.abort());
  $('chat-clear').addEventListener('click', () => { state.chat = []; $('chat-stats').textContent = ''; renderChat(); });
  $('chat-regen').addEventListener('click', () => {
    while (state.chat.length && state.chat[state.chat.length - 1].role === 'assistant') state.chat.pop();
    if (state.chat.length) sendChat(true);
  });
  $('chat-show-reasoning').addEventListener('change', renderChat);
  $('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });

  document.querySelectorAll('.config-body input, .config-body select, .config-body textarea')
    .forEach((el) => el.addEventListener('input', schedulePreview));

  $('model-filter').addEventListener('input', renderModels);
  $('show-incomplete').addEventListener('change', () => loadModels(false));
  $('rescan-btn').addEventListener('click', () => loadModels(true));
  $('clear-log').addEventListener('click', () => { state.logLines = []; scheduleLogRender(); });
  $('copy-cmd').addEventListener('click', () => navigator.clipboard?.writeText($('cmd-preview').textContent));

  $('launch-btn').addEventListener('click', async () => {
    const spec = collectSpec();
    if (!spec) return;
    $('launch-btn').disabled = true;
    state.logLines = [];
    state.logSeen.clear();
    try {
      setStatus(await api('/api/launch', { method: 'POST', body: JSON.stringify(spec) }));
      await loadProfiles();
      $('profile-status').textContent = 'config saved for this model';
    } catch (err) {
      appendLog({ sequence: -Date.now(), timestamp: Date.now() / 1000, line: `ERROR ${err.message}` });
      $('launch-btn').disabled = false;
    }
  });

  $('stop-btn').addEventListener('click', async () => {
    $('stop-btn').disabled = true;
    try { setStatus(await api('/api/stop', { method: 'POST' })); } catch (err) { alert(err.message); }
  });

  $('preset-save').addEventListener('click', async () => {
    const spec = collectSpec();
    if (!spec) { alert('Select a model first.'); return; }
    const name = prompt('Preset name', state.selected.id.split('/').pop());
    if (!name) return;
    const id = name.replace(/[^A-Za-z0-9._-]/g, '-').slice(0, 64);
    await api(`/api/presets/${id}`, { method: 'PUT', body: JSON.stringify({ id, name, spec }) });
    await loadPresets(id);
  });

  $('preset-delete').addEventListener('click', async () => {
    const id = $('preset-select').value;
    if (!id || !confirm(`Delete preset "${id}"?`)) return;
    await api(`/api/presets/${id}`, { method: 'DELETE' });
    await loadPresets();
  });

  $('preset-select').addEventListener('change', () => {
    const preset = state.presets?.find((p) => p.id === $('preset-select').value);
    if (preset) applySpec(preset.spec);
  });

  $('profile-save').addEventListener('click', async () => {
    const spec = collectSpec();
    if (!spec) { alert('Select a model first.'); return; }
    await api('/api/profiles', { method: 'PUT', body: JSON.stringify(spec) });
    await loadProfiles();
    $('profile-status').textContent = 'config saved for this model';
  });

  $('profile-reset').addEventListener('click', async () => {
    if (!state.selected) return;
    const engine = $('f-engine').value || 'vllm';
    // Drop only this engine's saved config so the other engine's is preserved.
    await api(`/api/profiles?model=${encodeURIComponent(state.selected.id)}&engine=${engine}`, { method: 'DELETE' });
    delete state.profiles[state.selected.id]?.[engine];
    if (!Object.keys(state.profiles[state.selected.id] || {}).length) delete state.profiles[state.selected.id];
    // Blank the launch fields only: the engine selector, host and port are not
    // per-model settings and blanking them made the form silently fall back to vLLM.
    [...TEXT_FIELDS, ...Object.keys(NUMBER_FIELDS)].forEach((name) => {
      const el = $(`f-${name}`);
      if (el && name !== 'host' && name !== 'port') el.value = '';
    });
    $('f-host').value = '0.0.0.0';
    $('f-port').value = '8000';
    $('f-extra_args').value = '';
    $('f-env').value = '';
    document.querySelectorAll('.gpu-check').forEach((el) => { el.checked = true; });
    state.autoServedName = null;
    selectModel(state.selected.id);
  });

  // Profiles must land before models render, otherwise an early card click sees an
  // empty state.profiles and falls back to suggested defaults over a real saved config.
  loadSystem().then(loadProfiles).then(() => loadModels()).then(loadPresets).catch((err) => {
    $('sys-subtitle').textContent = `error: ${err.message}`;
  });
  pollStatus();
  streamLogs();
  streamDownloadLogs();
  api('/api/download/status').then(setDownloadStatus).catch(() => {});
  setInterval(pollGpus, 5000);
  setInterval(pollStatus, 4000);
  setInterval(async () => {
    try { setDownloadStatus(await api('/api/download/status')); } catch (_) { /* transient */ }
  }, 3000);
}

init();

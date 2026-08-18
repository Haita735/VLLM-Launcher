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
};

const TEXT_FIELDS = [
  'host', 'served_model_name', 'api_key', 'distributed_executor_backend', 'dtype',
  'quantization', 'linear_backend', 'attention_backend', 'mamba_backend', 'mamba_cache_dtype',
  'kv_cache_dtype', 'reasoning_parser', 'tool_call_parser', 'limit_mm_per_prompt',
  'enforce_eager', 'enable_chunked_prefill', 'enable_prefix_caching', 'trust_remote_code',
  'enable_auto_tool_choice',
];

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
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

/* ------------------------------------------------------------------ system */
function renderGpus(gpus) {
  $('gpu-strip').innerHTML = gpus.map((g) => {
    const pct = g.memory_total_mb ? Math.round((g.memory_used_mb / g.memory_total_mb) * 100) : 0;
    return `<div class="gpu-card">
      <div class="gpu-name"><span>GPU ${g.index} · ${g.name}</span><span>sm_${g.compute_cap.replace('.', '')}</span></div>
      <div class="bar"><span style="width:${pct}%"></span></div>
      <div class="gpu-meta">${fmtBytes(g.memory_used_mb * 1048576)} / ${fmtBytes(g.memory_total_mb * 1048576)} · ${g.utilization}% · ${g.temperature}°C</div>
    </div>`;
  }).join('');
}

function renderGpuPicker(gpus) {
  $('gpu-picker').innerHTML = gpus.map((g) => `
    <label><input type="checkbox" class="gpu-check" value="${g.index}" checked /> ${g.index}</label>
  `).join('');
  $('gpu-picker').querySelectorAll('.gpu-check').forEach((el) => el.addEventListener('change', schedulePreview));
}

async function loadSystem() {
  const sys = await api('/api/system');
  state.system = sys;
  const v = sys.versions || {};
  $('sys-subtitle').textContent =
    `vLLM ${v.vllm || '?'} · torch ${v.torch || '?'} · CUDA ${v.cuda || '?'} · sm_${(sys.capability || '').replace('.', '') || '?'}`;
  renderGpus(sys.gpus || []);
  renderGpuPicker(sys.gpus || []);
  $('dl-hf-home').textContent = sys.hf_home || '(unset)';
  $('access-urls').innerHTML = (sys.access_urls || [])
    .map((url) => `<a href="${url}" class="url-chip">${url.replace('http://', '')}</a>`).join('');
}

async function pollGpus() {
  try {
    const { gpus } = await api('/api/gpus');
    renderGpus(gpus);
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
  if (state.profiles?.[model.id]) tags.push('<span class="tag saved">saved config</span>');

  const dl = model.download || {};
  if (dl.state === 'missing') {
    tags.push('<span class="tag missing">not downloaded</span>');
  } else if (dl.state === 'partial') {
    const detail = dl.shards_expected ? `${dl.shards_present}/${dl.shards_expected} shards` : 'incomplete';
    tags.push(`<span class="tag partial">partial · ${detail}</span>`);
  } else if (dl.shards_expected) {
    tags.push(`<span class="tag ok">${dl.shards_expected} shards</span>`);
  }

  return `<div class="model-card state-${dl.state || 'ok'}${state.selected?.id === model.id ? ' active' : ''}" data-id="${model.id}">
    <div class="name">${model.id}</div>
    <div class="meta">${tags.join('')}</div>
    <div class="card-actions"><button class="ghost danger" data-del="${model.id}">delete</button></div>
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
    .map((n) => `<div class="note ${n.level}">${n.text}</div>`).join('');

  const saved = state.profiles?.[model.id];
  if (saved) {
    applyFields(saved.spec);
    state.autoServedName = null;
    const when = new Date(saved.saved_at * 1000).toLocaleString();
    $('profile-status').textContent = `saved config restored (${when})`;
  } else {
    applySuggestedDefaults(model);
    $('profile-status').textContent = 'no saved config - using suggested defaults';
  }

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
  if (!$('f-linear_backend').value && cap < 100 && /nvfp4|fp4/.test(quant)) {
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
  const spec = { model: state.selected.id };

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
  TEXT_FIELDS.forEach((name) => { $(`f-${name}`).value = spec[name] ?? ''; });
  Object.keys(NUMBER_FIELDS).forEach((name) => {
    const el = $(`f-${name}`);
    if (el) el.value = spec[name] ?? '';
  });
  $('f-extra_args').value = spec.extra_args || '';
  $('f-env').value = Object.entries(spec.env || {}).map(([k, v]) => `${k}=${v}`).join('\n');
  document.querySelectorAll('.gpu-check').forEach((el) => {
    el.checked = !spec.gpu_indices?.length || spec.gpu_indices.includes(parseInt(el.value, 10));
  });
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

async function refreshPreview() {
  const spec = collectSpec();
  if (!spec) return;
  try {
    const { command, env } = await api('/api/preview', { method: 'POST', body: JSON.stringify(spec) });
    const envLine = Object.entries(env).map(([k, v]) => `${k}=${v}`).join(' ');
    $('cmd-preview').textContent = envLine ? `${envLine} \\\n  ${command}` : command;
  } catch (err) {
    $('cmd-preview').textContent = `# ${err.message}`;
  }
}

/* ------------------------------------------------------------------ runtime */
function setStatus(status) {
  state.running = status.running;
  const pill = $('status-pill');
  const online = status.endpoint?.online;
  const external = !!status.external;
  pill.className = 'pill';
  if (status.running && online) {
    pill.classList.add('running');
    pill.textContent = external ? 'serving · external' : 'serving';
  }
  else if (status.running) { pill.classList.add('starting'); pill.textContent = 'loading'; }
  else if (status.returncode) { pill.classList.add('error'); pill.textContent = `exit ${status.returncode}`; }
  else { pill.textContent = 'idle'; }

  const owner = external ? 'not owned by launcher' : `pid ${status.pid}`;
  $('endpoint-label').textContent = status.running
    ? `${owner} · :${status.port}${online ? ` · ${status.endpoint.models.join(', ')}` : ''}`
    : '';
  $('launch-btn').disabled = status.running
    || !state.selected
    || state.selected.download?.state === 'missing';
  $('stop-btn').disabled = !status.running || external;
  $('stop-btn').title = external
    ? 'Started outside this launcher — stop it from the InspireAI admin UI.'
    : '';
  const models = status.endpoint?.models || [];
  $('chat-model').textContent = status.running ? (models[0] || 'model loading…') : 'no model loaded';
}

async function pollStatus() {
  try { setStatus(await api('/api/status')); } catch (_) { /* transient */ }
}

function appendLog(event) {
  if (state.logSeen.has(event.sequence)) return;
  state.logSeen.add(event.sequence);
  state.logLines.push(event);
  if (state.logLines.length > 4000) state.logLines.splice(0, state.logLines.length - 4000);
  renderLogs();
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
      lines.forEach((line) => {
        if (!line.startsWith('data: ')) return;
        try { appendLog(JSON.parse(line.slice(6))); } catch (_) { /* partial frame */ }
      });
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

function setDownloadStatus(status) {
  $('dl-status').textContent = status.running
    ? `downloading ${status.repo}… ${Math.round(status.elapsed || 0)}s`
    : (status.repo ? `${status.repo} — exit ${status.returncode}` : 'idle');
  $('dl-start').disabled = status.running;
  $('dl-cancel').disabled = !status.running;
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
          renderDownloadLogs();
        } catch (_) { /* partial frame */ }
      });
    }
  } catch (_) { /* reconnect below */ }
  setTimeout(streamDownloadLogs, 2000);
}

/* ------------------------------------------------------------------ chat */
async function refreshChatModel() {
  try {
    const status = await api('/api/status');
    const models = status.endpoint?.models || [];
    $('chat-model').textContent = status.running
      ? (models[0] || 'model loading…')
      : 'no model loaded';
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

const esc = (s) => (s || '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

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
    const bubble = hideBubble
      ? ''
      : `<div class="bubble">${body}${m.pending ? '<span class="cursor-blink">▋</span>' : ''}</div>`;
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
  state.chat.filter((m) => !m.error).forEach((m) => {
    messages.push({ role: m.role, content: m.content });
  });
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
  const assistant = { role: 'assistant', content: '', reasoning: '', pending: true, thinkOpen: false };
  state.chat.push(assistant);
  state.chatBusy = true;
  $('chat-send').disabled = true;
  $('chat-stop').disabled = false;
  renderChat();

  const controller = new AbortController();
  state.chatAbort = controller;
  const started = performance.now();
  let tokens = 0;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: buildMessages().slice(0, -1), ...params }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    if (!params.stream) {
      const data = await res.json();
      const msg = data.choices?.[0]?.message || {};
      assistant.content = msg.content || '';
      assistant.reasoning = msg.reasoning ?? msg.reasoning_content ?? '';
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
          if (json.error) throw new Error(json.error);
          const delta = json.choices?.[0]?.delta || {};
          // vLLM 0.26 streams `reasoning`; other builds use `reasoning_content`.
          const think = delta.reasoning ?? delta.reasoning_content;
          if (think) {
            if (!assistant.thinkStart) {
              assistant.thinkStart = performance.now();
              if (!assistant.thinkPinned) assistant.thinkOpen = true;
            }
            assistant.reasoning += think;
          }
          if (delta.content) {
            if (assistant.reasoning && !assistant.thinkEnd) {
              assistant.thinkEnd = performance.now();
              if (!assistant.thinkPinned) assistant.thinkOpen = false;
            }
            assistant.content += delta.content;
          }
          if (json.usage?.completion_tokens) tokens = json.usage.completion_tokens;
          scheduleChatRender();
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      assistant.content += '\n\n[stopped]';
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
    + presets.map((p) => `<option value="${p.id}">${p.name}</option>`).join('');
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
        }),
      }));
    } catch (err) { alert(err.message); }
  });
  $('dl-cancel').addEventListener('click', async () => {
    try { setDownloadStatus(await api('/api/download/cancel', { method: 'POST' })); }
    catch (err) { alert(err.message); }
  });
  $('dl-clear').addEventListener('click', () => { state.dlLines = []; renderDownloadLogs(); });

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
  $('clear-log').addEventListener('click', () => { state.logLines = []; renderLogs(); });
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
    await api(`/api/profiles?model=${encodeURIComponent(state.selected.id)}`, { method: 'DELETE' });
    delete state.profiles[state.selected.id];
    document.querySelectorAll('.config-body input, .config-body select, .config-body textarea')
      .forEach((el) => { if (el.type !== 'checkbox') el.value = ''; });
    state.autoServedName = null;
    selectModel(state.selected.id);
  });

  loadSystem().then(() => loadModels()).then(loadProfiles).then(loadPresets).catch((err) => {
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

# vLLM Launcher

A LAN-only web UI for running [vLLM](https://github.com/vllm-project/vllm) or
[SGLang](https://github.com/sgl-project/sglang) on your own hardware.
Browse the models you've downloaded, tune the launch flags, start/stop the server, watch the
live log, pull new models from Hugging Face, and talk to whatever is loaded — all from one page.

Built for a homelab box running nvidia gpus.

Dark theme, three tabs — **Launcher**, **Download**, **Chat** — no build step, no dependencies
beyond what a vLLM or SGLang environment already has.

---

## Why

`vllm serve` has well over 200 flags. Remembering which combination a given checkpoint needs —
and which ones your GPU generation can actually support — gets old fast. This wraps it in
something you can drive from a phone on the couch.

Three things it does that a plain terminal doesn't:

- **Verifies downloads properly.** Hugging Face snapshots are symlinks into `blobs/`, which can
  dangle. It stats every weight file through the symlink, compares shard counts against
  `model.safetensors.index.json`, and looks for `.incomplete` blobs — so a half-finished 30 GB
  download is labelled `partial · 1/4 shards` instead of silently failing at load time.
- **Reads the checkpoint and warns you.** It parses `config.json` / `hf_quant_config.json` and
  compares against your GPU's compute capability, then tells you things like *"checkpoint is
  bfloat16 but this GPU is pre-Ampere; set dtype=float16"* before you waste ten minutes on a
  failed load.
- **Remembers per-model configs.** Whatever you launched with is saved against that model and
  restored next time you select it — separately for each engine.
- **Keeps the host alive.** Loading a model can take the whole desktop down through *CPU* RAM
  (kernel JIT, CPU offload), not VRAM. The launcher refuses to start when free RAM is already
  low, sizes JIT parallelism from the RAM that is actually free, and SIGKILLs the engine's
  process group if the box starts to thrash. See [Host RAM guards](#host-ram-guards).

---

## Requirements

| | |
|---|---|
| OS | Linux (uses `nvidia-smi`, `ip`, POSIX process groups, `/proc/meminfo`) |
| Python | 3.11+ |
| Packages | `fastapi`, `uvicorn`, `httpx`, `pydantic` — all already present in a vLLM/SGLang env |
| Engines | vLLM (developed against **0.25–0.26**) and/or SGLang (**0.5.x**); either may be absent |
| Downloads | the `hf` CLI (ships with `huggingface_hub`) |

No build step, no npm, no framework. The frontend is three static files.

---

## Quick start

```bash
git clone https://github.com/Haita735/VLLM-Launcher.git ~/vllm-launcher
cd ~/vllm-launcher

# run it with the python from the env that has vLLM installed - everything else
# (engine binaries, HF cache, model roots) is derived from that unless you override it
/path/to/envs/vllm/bin/python server.py
```

Open `http://<your-lan-ip>:7870`. The header lists the URLs it thinks it's reachable on.

### How engines are found

Each engine is addressed by the **interpreter of the environment that has it installed**; the
console script is optional. In order, the launcher checks the interpreter running it, sibling
conda envs named `vllm` / `sglang` (`~/miniconda3`, `~/anaconda3`, `~/miniforge3`,
`~/mambaforge`, `/opt/conda`, …), common venv locations (`~/vllm/.venv`, `~/.venvs/sglang`,
`~/sglang-env`, …) and finally `$PATH`. An engine that is not found is shown as unavailable in
the UI and refused with a clear message at launch instead of falling through to whatever
`python` is on `PATH`.

To pin one explicitly: `VLLM_PYTHON=/path/to/envs/vllm/bin/python` (or `SGLANG_PYTHON`), or the
older `VLLM_BIN` / `SGLANG_BIN` pointing at the executable.

Everything toolchain-related (`PATH`, `CUDA_HOME`, `LD_LIBRARY_PATH`) is derived per launch from
the *selected* engine's env — the pip-installed CUDA toolkit under `nvidia/cu13/` if present — so
an `EnvironmentFile` written for one engine never leaks its libraries into the other.

### Run it as a service

```bash
sudo cp vllm-launcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-launcher
```

Edit the `Environment=` lines in the unit first — they contain absolute paths for the reference
machine. The unit uses `Restart=always` and `StartLimitIntervalSec=0` so it comes back from a
crash loop instead of parking in `failed`.

> **Restarting the launcher kills a loaded model.** The engine is a child process and its stdout
> pipe belongs to the launcher. Unload from the UI first.

---

## Host RAM guards

The failure mode this protects against: the whole desktop hard-freezes while a model *loads*,
with nothing in the journal because the kernel was too thrashed to write it. On the reference
machine (30 GiB RAM) the cause was not the weights but SGLang's first-start FlashInfer JIT:
each `nvcc`/`cicc` job is ~6 GiB and an inherited `MAX_JOBS=4` ran four at once.

Three layers, all on by default:

1. **Pre-launch floor.** Launch is refused with `507` while `MemAvailable` is below
   `VLLM_LAUNCHER_MIN_FREE_RAM_GIB` (4 GiB).
2. **JIT sizing.** `MAX_JOBS` for the engine is computed from free RAM at launch time
   (`(free − headroom) / 6 GiB`, capped at 4, never *raised* above an inherited value) and
   `FLASHINFER_NVCC_THREADS=1`. Ninja keeps finished objects, so a killed compile resumes.
3. **Watchdog.** While the engine runs, a thread samples `/proc/meminfo` and
   `/proc/pressure/memory` twice a second and SIGKILLs the engine's process group when free RAM
   drops below `VLLM_LAUNCHER_RAM_KILL_GIB` (2 GiB) or the system is stalled on memory for three
   consecutive samples (PSI `full avg10` ≥ 25 %). The reason is written to the log.

The header shows host RAM next to the GPUs and turns red when a launch would be refused.

For a hard ceiling the kernel enforces even if the launcher itself is swapped out, add a cgroup
cap to the unit (`MemoryHigh=` / `MemoryMax=` in `vllm-launcher.service`, or
`systemctl set-property`). Size it to leave the desktop roughly half the machine: a cap of 23 GiB
on a 30 GiB box still froze it.

---

## Configuration

All optional; sensible defaults in brackets.

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_LAUNCHER_HOST` | `0.0.0.0` | bind address |
| `VLLM_LAUNCHER_PORT` | `7870` | bind port |
| `VLLM_PYTHON`, `SGLANG_PYTHON` | auto-discovered (see above) | interpreter of the env that has the engine |
| `VLLM_BIN`, `SGLANG_BIN` | auto-discovered | alternative: path to the engine's executable |
| `VLLM_LAUNCHER_MODEL_ROOTS` | `$HF_HOME/hub:~/models` | `:`-separated scan paths |
| `VLLM_LAUNCHER_TEMPLATES` | `./templates` | chat templates resolved by bare name |
| `VLLM_LAUNCHER_PROFILES` | `./profiles` | per-model saved configs |
| `VLLM_LAUNCHER_PRESETS` | `./presets` | named presets |
| `VLLM_LAUNCHER_LOG_LINES` | `4000` | log ring buffer size |
| `VLLM_LAUNCHER_EXTERNAL_PORT` | `8000` | port to adopt an externally started server from |
| `VLLM_LAUNCHER_EXTERNAL_API_KEY` | unset | key for that external endpoint, if it needs one |
| `VLLM_LAUNCHER_EXTERNAL_NAME` | generic wording | name of the other manager, for messages |
| `VLLM_LAUNCHER_MIN_FREE_RAM_GIB` | `4` | refuse to launch below this much free host RAM |
| `VLLM_LAUNCHER_RAM_KILL_GIB` | `2` | watchdog kills the engine below this |
| `VLLM_LAUNCHER_RAM_PSI_FULL` | `25` | … or when PSI `full avg10` exceeds this % |
| `VLLM_LAUNCHER_JIT_RAM_PER_JOB_GIB` | `6` | RAM budgeted per parallel nvcc job |
| `VLLM_LAUNCHER_JIT_MAX_JOBS` | `4` | hard cap on the derived `MAX_JOBS` |
| `VLLM_LAUNCHER_ALLOWED_CIDRS` | RFC1918 + loopback + link-local + `100.64.0.0/10` | who may connect |
| `VLLM_LAUNCHER_ALLOW_ANY` | unset | set to `1` to disable the network allowlist |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache root |

Nothing is hardcoded to a particular machine: with no environment set at all, the launcher
finds the engines beside the Python running it (or in sibling conda envs) and uses Hugging
Face's own default cache location.

### One config, two engines

The shared fields (dtype, quantization, TP/PP, max model len, GPU memory utilisation, KV cache
dtype, attention backend, parsers, …) are the same controls for both engines; the server maps
them onto SGLang's flag names (`--tp-size`, `--context-length`, `--mem-fraction-static`, …) and
translates vLLM-only *values* that SGLang's argparse would reject outright — `FLASH_ATTN` →
`fa3`, `kv-cache-dtype fp8` → `fp8_e4m3`, `qwen3_xml` → `qwen3_coder`, prefix caching off →
`--disable-radix-cache`, enforce eager → CUDA graphs disabled. Every translation, and every field
that has no SGLang equivalent and was dropped, is listed under the command preview and in the
log at launch. SGLang-only knobs (schedule policy, speculative/MTP, mamba, DP/EP) live in their
own section that appears when the engine is switched.

### Chat templates

Drop `.jinja` files in `templates/` and reference them with `--chat-template <name>.jinja` in
extra args. Bare names resolve against that directory, so a saved profile stays valid on a
different machine; absolute paths are passed through untouched.

`templates/qwen38-effort-tolerant.jinja` is Qwen3.8's stock template with one change: an
unrecognised `reasoning_effort` falls back to the default instead of `raise_exception`, which
otherwise turns any client sending `minimal`/`none` into a 400. It also accepts
`thinking_effort` as an alias.

### Sharing a GPU with another manager

If something else already serves on `VLLM_LAUNCHER_EXTERNAL_PORT`, the launcher adopts it
read-only: status, chat and the delete guard all see that model, Launch refuses with a 409
instead of fighting for the GPUs, and Stop is disabled because the process is not its child.

Model roots handle both layouts: Hugging Face caches (`models--org--name/snapshots/<rev>/`) and
plain directories containing a `config.json` or `*.gguf`.

---

## The three tabs

### Launcher

Model list on the left, launch config on the right, live terminal underneath.

Cards show size, quantization format, dtype, context length, shard count and download state.
Incomplete models are hidden by default — tick **incomplete** to reveal them. Every card has a
hover **delete** button; anything over 1 GiB requires typing `DELETE`, and the server refuses to
delete a model that's currently loaded.

Config is grouped into Server / Parallelism & devices / Precision & kernels / Memory & KV cache /
Behaviour, covering tensor & pipeline parallel, GPU selection, dtype, quantization, linear and
attention backend, mamba backend, max model length, KV cache dtype, block size, GPU memory
utilisation, max sequences, prefix caching, CPU offload, plus free-form extra args and env vars.

A **command preview** shows the exact `argv` before you commit to it, and there's a
**Save for this model** / **Reset** pair next to the auto-save indicator.

### Download

Paste a Hugging Face URL or a bare repo id — both work, and the parsed result is shown live:

```
https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/tree/main  →  unsloth/Qwen3.8-27B-NVFP4
https://evil.example.com/a/b                                →  rejected
```

Optional revision and include-glob fields (`*.safetensors *.json`). Runs `hf download` as a
subprocess with the output streamed to the page, and refreshes the model list when it finishes.
The launcher normally runs with `HF_HUB_OFFLINE=1`; this forces it off for the download only.

### Chat

A normal chat interface against the loaded model, proxied through the launcher so it works even
when the engine is bound to localhost and so the API key stays server-side.

Streaming with live token rendering, Stop, Regenerate, Clear, per-message copy, multi-turn
history, and tok/s stats. Parameters: system prompt, temperature, top_p, top_k, max_tokens,
presence / frequency / repetition penalty, seed, stop sequences, stream toggle, thinking toggle.

Reasoning models get a collapsible **Thinking** disclosure that stays open while the model
reasons, auto-collapses to `Thought for 2.7s` when the answer starts, and re-opens on click. It
works both with a server-side reasoning parser (`reasoning` / `reasoning_content` deltas) and
with raw inline `<think>…</think>` tags when no parser is configured. Stopping a stream tells
SGLang to abort the request (`/abort_request`) so it stops burning GPU time; vLLM aborts on its
own when the connection closes.

---

## HTTP API

Everything the UI does is a plain JSON endpoint.

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/system` | engine availability & versions, GPUs, RAM guard thresholds, access URLs |
| `GET` | `/api/gpus` | live GPU memory / utilisation / temperature plus host RAM |
| `GET` | `/api/models` | `?refresh=true` rescan, `?include_missing=true` include incomplete |
| `POST` | `/api/models/delete` | `{"models": ["org/name"]}` |
| `POST` | `/api/preview` | build the argv/env without running it; includes translation notes |
| `POST` | `/api/launch` | start the engine (`409` port busy, `507` low RAM); saves the profile |
| `POST` | `/api/stop` | SIGTERM the process group, SIGKILL after 30s, wait for workers |
| `GET` | `/api/status` | process state + `/v1/models` health probe |
| `GET` | `/api/logs`, `/api/logs/stream` | buffer / SSE |
| `GET`,`PUT`,`DELETE` | `/api/profiles` | per-model saved configs |
| `GET` | `/api/presets`, `PUT`/`DELETE` `/api/presets/{id}` | named presets |
| `POST` | `/api/download`, `/api/download/cancel`, `/api/download/resolve` | |
| `GET` | `/api/download/status`, `/api/download/logs`, `/api/download/logs/stream` | |
| `POST` | `/api/chat` | OpenAI-shaped, streams SSE |

The model itself is served by the engine directly on its own port (`8000` by default), so point
OpenAI-compatible clients at `http://<host>:8000/v1` — not at the launcher.

---

## Running old GPUs (Turing / Pascal)

Modern quantized checkpoints mostly assume Ada or Blackwell. A lot of them still run on older
cards, just through different kernels. Verified on **sm_75 (TITAN RTX)** with vLLM 0.26.0:

- **NVFP4 works.** There are no FP4 tensor cores before Blackwell, but vLLM falls back to
  weight-only **W4A16 via Marlin** — weights stay 4-bit in VRAM and are dequantized inside the
  GEMM. Look for `Using MarlinNvFp4LinearKernel for NVFP4 GEMM` in the log. Pin it with
  `linear-backend = marlin`.
- **FP8 works** the same way (`Fp8Config.get_min_capability()` returns `75`).
- **bfloat16 does not.** Set `dtype = float16`; vLLM logs `Casting torch.bfloat16 to torch.float16`.
- **FP8 KV cache does not** — needs sm_89+. Set `kv-cache-dtype = float16` explicitly: when the
  checkpoint ships fp8 KV scales (`kv_cache_scheme` in `config.json`), `auto` resolves to fp8 and
  `TRITON_ATTN` aborts with `native FP8 (fp8e4nv) requires SM89+`.
- **FlashAttention 2 needs sm_80+**, so attention auto-selects `TRITON_ATTN`. This is fine.
- **Gated DeltaNet / hybrid linear-attention models** (Qwen3.5/3.6/3.8) compile and run — the
  Triton kernels JIT to cubins for sm_75.

The launcher detects all of this and pre-fills the right flags when you select a model.

Reference numbers, Qwen3.6-35B-A3B-NVFP4 on 2× TITAN RTX, `tp=2`:

| | |
|---|---|
| Weights | 11.0 GiB/GPU |
| KV cache | 6.6 GiB/GPU → 562k tokens |
| First load | ~9.5 min (mostly Triton/torch.compile JIT) |
| Later loads | ~4.5 min (AOT cache warm) |
| Generation | 40–100 tok/s |

---

## Troubleshooting

**The whole machine froze while a model was loading**
Host RAM, not VRAM — see [Host RAM guards](#host-ram-guards). Check the log for a
`RAM WATCHDOG` line; if the box went down before the watchdog could act, lower
`VLLM_LAUNCHER_RAM_KILL_GIB`'s companion `VLLM_LAUNCHER_MIN_FREE_RAM_GIB`, or add a cgroup cap to
the unit. First SGLang starts JIT-compile FlashInfer kernels into `~/.cache/sglang`; later starts
reuse them.

**SGLang exits with code 2 immediately (`invalid choice`)**
A value SGLang's argparse does not know, usually from `Extra engine args` or a hand-edited
profile. The launcher translates the common vLLM spellings of the shared fields and lists what
it changed under the command preview; anything else must use SGLang's own vocabulary.

**`TypeError: GGUFConfig.override_quantization_method() got an unexpected keyword argument 'hf_config'`**
An out-of-date `vllm-gguf-plugin`. vLLM iterates every registered quantization method, so this
breaks *all* model loads, not just GGUF ones. `pip install -U vllm-gguf-plugin` (needs ≥ 0.0.5).

**`flashinfer-cubin version (x) does not match flashinfer version (y)`**
vLLM pins `flashinfer-python` to a version whose matching `flashinfer-cubin` may not be published
yet. The launcher sets `FLASHINFER_DISABLE_VERSION_CHECK=1` for spawned processes. Harmless on
pre-Ampere hardware, which can't use FlashInfer kernels anyway.

**Model loads then sits at "No available shared memory broadcast block found in 60 seconds"**
Normal. It's compiling Triton kernels and capturing CUDA graphs. Watch worker CPU — if it's
pegged, it's working. `--enforce-eager` skips this at some throughput cost.

**A model shows `partial` or `not downloaded`**
The weights are missing, dangling or half-fetched. Re-run the download; the log names the exact
shard count.

**Empty response with `finish_reason: "length"`**
Reasoning models spend the budget thinking before emitting an answer. Raise `max_tokens`.

**The launcher won't restart / hangs on shutdown**
Fixed via `timeout_graceful_shutdown=5` — without it, an open SSE log stream keeps uvicorn
waiting forever.

---

## Security

This is a LAN tool with **no authentication**. It starts processes and deletes directories.

What it does defend against:

- Requests from outside the configured CIDR allowlist get `403`.
- Launch arguments are assembled as an `argv` list and passed to `subprocess` without a shell,
  so there's no command-injection surface. Extra args go through `shlex.split`.
- `model` must match an entry the scanner discovered — arbitrary paths are rejected.
- Deletes resolve the target and require it to sit strictly inside a configured model root, so
  `../../etc` and root directories themselves are refused.
- Environment variable names are regex-validated.
- API keys live server-side; the chat proxy attaches them.

What it does **not** do: authenticate anyone. Treat access to port 7870 as equivalent to shell
access to the GPU box, and don't expose it to the internet. `presets/` and `profiles/` are
gitignored because they can contain API keys.

---

## Layout

```
server.py               FastAPI backend: discovery, launch, downloads, chat proxy, SSE
static/index.html       markup for all three tabs
static/app.js           client logic, no dependencies
static/styles.css       dark theme
templates/              chat templates, referenced by bare name from extra args
vllm-launcher.service   systemd unit
profiles/               per-model saved configs   (gitignored)
presets/                named presets             (gitignored)
```

## License

MIT

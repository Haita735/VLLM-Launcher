# vLLM Launcher

A LAN-only web UI for running [vLLM](https://github.com/vllm-project/vllm) on your own hardware.
Browse the models you've downloaded, tune the launch flags, start/stop the server, watch the
live log, pull new models from Hugging Face, and talk to whatever is loaded — all from one page.

Built for a homelab box with two NVIDIA TITAN RTX cards, but nothing is hardcoded to that setup.

Dark theme, three tabs — **Launcher**, **Download**, **Chat** — no build step, no dependencies
beyond what a vLLM environment already has.

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
  restored next time you select it.

---

## Requirements

| | |
|---|---|
| OS | Linux (uses `nvidia-smi`, `ip`, POSIX process groups) |
| Python | 3.11+ |
| Packages | `fastapi`, `uvicorn`, `httpx`, `pydantic` — all already present in a vLLM env |
| vLLM | any recent version; developed against **0.26.0** |
| Downloads | the `hf` CLI (ships with `huggingface_hub`) |

No build step, no npm, no framework. The frontend is three static files.

---

## Quick start

```bash
git clone https://github.com/animeclips0904-a11y/VLLM-Launcher.git ~/vllm-launcher
cd ~/vllm-launcher

# point it at the python env that has vLLM installed
VLLM_BIN=/path/to/envs/vllm/bin/vllm \
HF_HOME=/mnt/vllmdata \
/path/to/envs/vllm/bin/python server.py
```

Open `http://<your-lan-ip>:7870`. The header lists the URLs it thinks it's reachable on.

### Run it as a service

```bash
sudo cp vllm-launcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-launcher
```

Edit the `Environment=` lines in the unit first — they contain absolute paths for the reference
machine. The unit uses `Restart=always` and `StartLimitIntervalSec=0` so it comes back from a
crash loop instead of parking in `failed`.

> **Restarting the launcher kills a loaded model.** vLLM is a child process and its stdout pipe
> belongs to the launcher. Unload from the UI first.

---

## Configuration

All optional; sensible defaults in brackets.

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_LAUNCHER_HOST` | `0.0.0.0` | bind address |
| `VLLM_LAUNCHER_PORT` | `7870` | bind port |
| `VLLM_BIN` | `/home/andrew/miniconda3/envs/vllm/bin/vllm` | path to the `vllm` executable |
| `VLLM_LAUNCHER_MODEL_ROOTS` | `/mnt/vllmdata/hub:~/.cache/huggingface/hub:~/models` | `:`-separated scan paths |
| `VLLM_LAUNCHER_PROFILES` | `./profiles` | per-model saved configs |
| `VLLM_LAUNCHER_PRESETS` | `./presets` | named presets |
| `VLLM_LAUNCHER_LOG_LINES` | `4000` | log ring buffer size |
| `VLLM_LAUNCHER_ALLOWED_CIDRS` | RFC1918 + loopback + link-local + `100.64.0.0/10` | who may connect |
| `VLLM_LAUNCHER_ALLOW_ANY` | unset | set to `1` to disable the network allowlist |
| `HF_HOME` | `/mnt/vllmdata` | Hugging Face cache root |

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
when vLLM is bound to localhost and so the API key stays server-side.

Streaming with live token rendering, Stop, Regenerate, Clear, per-message copy, multi-turn
history, and tok/s stats. Parameters: system prompt, temperature, top_p, top_k, max_tokens,
presence / frequency / repetition penalty, seed, stop sequences, stream toggle, thinking toggle.

Reasoning models get a collapsible **Thinking** disclosure that stays open while the model
reasons, auto-collapses to `Thought for 2.7s` when the answer starts, and re-opens on click.

---

## HTTP API

Everything the UI does is a plain JSON endpoint.

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/system` | versions, GPUs, compute capability, access URLs |
| `GET` | `/api/gpus` | live memory / utilisation / temperature |
| `GET` | `/api/models` | `?refresh=true` rescan, `?include_missing=true` include incomplete |
| `POST` | `/api/models/delete` | `{"models": ["org/name"]}` |
| `POST` | `/api/preview` | build the argv without running it |
| `POST` | `/api/launch` | start vLLM; also saves the per-model profile |
| `POST` | `/api/stop` | SIGTERM the process group, SIGKILL after 30s |
| `GET` | `/api/status` | process state + `/v1/models` health probe |
| `GET` | `/api/logs`, `/api/logs/stream` | buffer / SSE |
| `GET`,`PUT`,`DELETE` | `/api/profiles` | per-model saved configs |
| `GET` | `/api/presets`, `PUT`/`DELETE` `/api/presets/{id}` | named presets |
| `POST` | `/api/download`, `/api/download/cancel`, `/api/download/resolve` | |
| `GET` | `/api/download/status`, `/api/download/logs`, `/api/download/logs/stream` | |
| `POST` | `/api/chat` | OpenAI-shaped, streams SSE |

The model itself is served by vLLM directly on its own port (`8000` by default), so point
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
vllm-launcher.service   systemd unit
profiles/               per-model saved configs   (gitignored)
presets/                named presets             (gitignored)
```

## License

MIT

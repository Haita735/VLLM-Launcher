#!/usr/bin/env python
"""LAN-only vLLM launcher: discover local models, configure launch flags, run and monitor `vllm serve`."""

from __future__ import annotations

import ipaddress
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
PRESET_DIR = Path(os.environ.get("VLLM_LAUNCHER_PRESETS", APP_DIR / "presets"))
PRESET_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR = Path(os.environ.get("VLLM_LAUNCHER_PROFILES", APP_DIR / "profiles"))
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

VLLM_BIN = Path(os.environ.get("VLLM_BIN", "/home/andrew/miniconda3/envs/vllm/bin/vllm"))

# Directories scanned for models. HF-cache layouts and plain checkpoint dirs are both handled.
MODEL_ROOTS = [
    Path(p).expanduser()
    for p in os.environ.get(
        "VLLM_LAUNCHER_MODEL_ROOTS",
        "/mnt/vllmdata/hub:~/.cache/huggingface/hub:~/models",
    ).split(":")
    if p.strip()
]

LOG_BUFFER = int(os.environ.get("VLLM_LAUNCHER_LOG_LINES", "4000"))
PRESET_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Networks allowed to reach the UI. Tailscale's 100.64.0.0/10 is not covered by
# ipaddress.is_private, so the ranges are listed explicitly.
DEFAULT_ALLOWED_CIDRS = (
    "127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,"
    "100.64.0.0/10,::1/128,fc00::/7,fe80::/10"
)
ALLOW_ANY_CLIENT = os.environ.get("VLLM_LAUNCHER_ALLOW_ANY", "").lower() in {"1", "true", "yes"}
ALLOWED_NETWORKS = [
    ipaddress.ip_network(cidr.strip(), strict=False)
    for cidr in os.environ.get("VLLM_LAUNCHER_ALLOWED_CIDRS", DEFAULT_ALLOWED_CIDRS).split(",")
    if cidr.strip()
]

app = FastAPI(title="vLLM Launcher", version="1.0.0")


# --------------------------------------------------------------------------------------
# LAN-only guard
# --------------------------------------------------------------------------------------
def client_allowed(host: str | None) -> bool:
    if ALLOW_ANY_CLIENT:
        return True
    try:
        addr = ipaddress.ip_address(host or "")
    except ValueError:
        return True  # unix socket or unknown transport
    if addr.version == 6 and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return any(addr in network for network in ALLOWED_NETWORKS)


@app.middleware("http")
async def restrict_to_private_networks(request: Request, call_next):
    host = request.client.host if request.client else None
    if not client_allowed(host):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"detail": f"Client {host} is outside the allowed networks."}, status_code=403
        )
    return await call_next(request)


# --------------------------------------------------------------------------------------
# Hardware probe
# --------------------------------------------------------------------------------------
def _nvidia_smi(fields: str) -> list[list[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def gpu_snapshot() -> list[dict]:
    rows = _nvidia_smi(
        "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,compute_cap"
    )
    gpus = []
    for row in rows:
        if len(row) < 7:
            continue
        gpus.append(
            {
                "index": int(row[0]),
                "name": row[1],
                "memory_total_mb": int(float(row[2])),
                "memory_used_mb": int(float(row[3])),
                "utilization": int(float(row[4])),
                "temperature": int(float(row[5])),
                "compute_cap": row[6],
            }
        )
    return gpus


def _local_addresses() -> list[str]:
    try:
        out = subprocess.run(
            ["ip", "-4", "-brief", "addr"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    addresses = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] == "lo":
            continue
        # Docker/libvirt bridges are not reachable from the LAN, so skip them.
        if parts[0].startswith(("docker", "br-", "veth", "virbr", "vmnet")):
            continue
        for cidr in parts[2:]:
            ip = cidr.split("/")[0]
            if client_allowed(ip):
                addresses.append(ip)
    return addresses


_system_cache: dict[str, Any] = {}


def system_info() -> dict:
    if _system_cache:
        info = dict(_system_cache)
        info["gpus"] = gpu_snapshot()
        return info

    versions = {"vllm": None, "torch": None, "cuda": None}
    try:
        out = subprocess.run(
            [
                str(VLLM_BIN.parent / "python"),
                "-c",
                "import vllm,torch,json;print(json.dumps({'vllm':vllm.__version__,"
                "'torch':torch.__version__,'cuda':torch.version.cuda}))",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        ).stdout
        versions.update(json.loads(out.strip().splitlines()[-1]))
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        pass

    gpus = gpu_snapshot()
    caps = [g["compute_cap"] for g in gpus]
    cap_major_minor = min((tuple(int(x) for x in c.split(".")) for c in caps), default=(0, 0))
    _system_cache.update(
        {
            "versions": versions,
            "vllm_bin": str(VLLM_BIN),
            "model_roots": [str(p) for p in MODEL_ROOTS],
            "capability": f"{cap_major_minor[0]}.{cap_major_minor[1]}" if caps else None,
            "capability_int": cap_major_minor[0] * 10 + cap_major_minor[1],
            "gpu_count": len(gpus),
            "access_urls": [
                f"http://{addr}:{os.environ.get('VLLM_LAUNCHER_PORT', '7870')}"
                for addr in _local_addresses()
            ],
        }
    )
    info = dict(_system_cache)
    info["gpus"] = gpus
    return info


# --------------------------------------------------------------------------------------
# Model discovery
# --------------------------------------------------------------------------------------
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt")


def _dir_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() or entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _describe_quantization(config: dict, path: Path) -> dict:
    qc = config.get("quantization_config") or {}
    # ModelOpt checkpoints keep the algorithm in a sidecar file instead of config.json.
    sidecar = _read_json(path / "hf_quant_config.json").get("quantization") or {}
    if not qc and not sidecar:
        return {"method": None, "format": None, "detail": None, "kv_cache": None}

    fmt = qc.get("format")
    formats = []
    for group in (qc.get("config_groups") or {}).values():
        gfmt = group.get("format")
        if gfmt and gfmt not in formats:
            formats.append(gfmt)
    for algo in (qc.get("quant_algo"), sidecar.get("quant_algo")):
        if algo and algo not in formats:
            formats.append(algo)

    kv = (
        qc.get("kv_cache_scheme")
        or qc.get("kv_cache_quant_algo")
        or sidecar.get("kv_cache_quant_algo")
    )
    kv_desc = None
    if isinstance(kv, dict):
        kv_desc = f"{kv.get('type', 'int')}{kv.get('num_bits', '')}"
    elif isinstance(kv, str):
        kv_desc = kv

    method = qc.get("quant_method") or ("modelopt" if sidecar.get("quant_algo") else None)
    return {
        "method": method,
        "format": fmt,
        "detail": ", ".join(formats) or fmt,
        "kv_cache": kv_desc,
    }


def _max_len(config: dict) -> int | None:
    for scope in (config, config.get("text_config") or {}):
        value = scope.get("max_position_embeddings")
        if isinstance(value, int):
            return value
    return None


def _weight_status(snapshot: Path, repo: Path | None) -> dict:
    """Verify weights are really on disk: HF snapshots are symlinks into blobs/ and can dangle."""
    present: list[str] = []
    dangling = 0
    total = 0
    for entry in snapshot.iterdir():
        if not entry.name.endswith(_WEIGHT_SUFFIXES):
            continue
        try:
            total += entry.stat().st_size  # follows the symlink into blobs/
            present.append(entry.name)
        except OSError:
            dangling += 1

    expected = None
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weight_map = _read_json(index).get("weight_map") or {}
        expected = len(set(weight_map.values())) or None

    incomplete = 0
    if repo is not None and (repo / "blobs").is_dir():
        incomplete = sum(1 for p in (repo / "blobs").iterdir() if p.name.endswith(".incomplete"))

    if not present:
        state = "missing"
    elif dangling or incomplete or (expected and len(present) < expected):
        state = "partial"
    else:
        state = "ok"

    return {
        "state": state,
        "shards_present": len(present),
        "shards_expected": expected,
        "dangling": dangling,
        "incomplete": incomplete,
        "bytes": total,
    }


def _model_entry(model_id: str, path: Path, source: str, repo: Path | None = None) -> dict | None:
    config = _read_json(path / "config.json")
    gguf_files = sorted(p.name for p in path.glob("*.gguf"))
    if not config and not gguf_files:
        return None

    architectures = config.get("architectures") or []
    download = _weight_status(path, repo)
    return {
        "id": model_id,
        "path": str(path),
        "source": source,
        "size_bytes": _dir_size(path),
        "architecture": architectures[0] if architectures else None,
        "model_type": config.get("model_type"),
        "dtype": config.get("dtype") or config.get("torch_dtype"),
        "max_position_embeddings": _max_len(config),
        "quantization": _describe_quantization(config, path),
        "gguf_files": gguf_files,
        "download": download,
        "has_weights": download["state"] == "ok",
        "multimodal": bool(config.get("vision_config") or config.get("audio_config")),
    }


def _scan_hf_cache(root: Path) -> list[dict]:
    entries = []
    for repo_dir in sorted(root.glob("models--*")):
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        revisions = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for revision in revisions:
            if not revision.is_dir():
                continue
            model_id = repo_dir.name.removeprefix("models--").replace("--", "/")
            entry = _model_entry(model_id, revision, str(root), repo=repo_dir)
            if entry:
                entries.append(entry)
            break
    return entries


def _scan_plain_dir(root: Path) -> list[dict]:
    entries = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        entry = _model_entry(child.name, child, str(root))
        if entry:
            entries.append(entry)
    return entries


_model_cache: dict[str, Any] = {"stamp": 0.0, "entries": []}


def discover_models(force: bool = False) -> list[dict]:
    if not force and time.time() - _model_cache["stamp"] < 60:
        return _model_cache["entries"]

    found: list[dict] = []
    for root in MODEL_ROOTS:
        if not root.is_dir():
            continue
        if any(root.glob("models--*")):
            found.extend(_scan_hf_cache(root))
        else:
            found.extend(_scan_plain_dir(root))

    # The same repo can live in several caches; keep the most complete copy.
    rank = {"ok": 2, "partial": 1, "missing": 0}
    best: dict[str, dict] = {}
    for entry in found:
        current = best.get(entry["id"])
        score = (rank[entry["download"]["state"]], entry["size_bytes"])
        if current is None or score > (rank[current["download"]["state"]], current["size_bytes"]):
            best[entry["id"]] = entry

    entries = sorted(best.values(), key=lambda e: e["id"].lower())
    for entry in entries:
        entry["notes"] = compatibility_notes(entry)
    _model_cache.update({"stamp": time.time(), "entries": entries})
    return entries


def compatibility_notes(entry: dict) -> list[dict]:
    """Hardware-specific advice derived from the checkpoint config and local compute capability."""
    cap = system_info().get("capability_int") or 0
    notes: list[dict] = []
    quant = entry.get("quantization") or {}
    detail = " ".join(str(v) for v in quant.values() if v).lower()
    download = entry.get("download") or {}

    if download.get("state") == "missing":
        notes.append(
            {"level": "warn", "text": "Only metadata is cached here - the weights are not downloaded."}
        )
    elif download.get("state") == "partial":
        bits = []
        if download.get("shards_expected"):
            bits.append(f"{download['shards_present']}/{download['shards_expected']} shards")
        if download.get("incomplete"):
            bits.append(f"{download['incomplete']} unfinished blob(s)")
        if download.get("dangling"):
            bits.append(f"{download['dangling']} broken symlink(s)")
        notes.append(
            {"level": "warn", "text": f"Incomplete download: {', '.join(bits)}. Re-run the download."}
        )

    if cap and cap < 80 and (entry.get("dtype") or "").lower() == "bfloat16":
        notes.append(
            {
                "level": "warn",
                "text": "Checkpoint is bfloat16 but this GPU is pre-Ampere; set dtype=float16.",
            }
        )
    if "nvfp4" in detail or "fp4" in detail:
        if cap and cap < 100:
            notes.append(
                {
                    "level": "info",
                    "text": "No native FP4 tensor cores: vLLM runs NVFP4 weight-only (W4A16) "
                    "through the Marlin kernel. Pin it with linear-backend=marlin.",
                }
            )
    if "fp8" in detail and cap and cap < 89:
        notes.append(
            {
                "level": "info",
                "text": "FP8 weights are dequantised by Marlin on this GPU; compute stays fp16.",
            }
        )
    if quant.get("kv_cache") and "8" in str(quant.get("kv_cache")) and cap and cap < 89:
        notes.append(
            {
                "level": "warn",
                "text": "Checkpoint ships fp8 KV scales, so kv-cache-dtype=auto resolves to fp8, "
                "which needs SM89+. Set kv-cache-dtype=float16 explicitly.",
            }
        )
    if entry.get("multimodal"):
        notes.append({"level": "info", "text": "Multimodal checkpoint; limit-mm-per-prompt applies."})
    return notes


# --------------------------------------------------------------------------------------
# Launch specification
# --------------------------------------------------------------------------------------
class LaunchSpec(BaseModel):
    model: str
    served_model_name: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None

    gpu_indices: list[int] = Field(default_factory=list)
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    distributed_executor_backend: str | None = None

    dtype: str | None = None
    quantization: str | None = None
    linear_backend: str | None = None
    attention_backend: str | None = None
    mamba_backend: str | None = None
    mamba_cache_dtype: str | None = None

    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    kv_cache_dtype: str | None = None
    block_size: int | None = None
    swap_space: float | None = None
    cpu_offload_gb: float | None = None
    num_gpu_blocks_override: int | None = None

    enforce_eager: str | None = None
    enable_chunked_prefill: str | None = None
    enable_prefix_caching: str | None = None
    trust_remote_code: str | None = None
    enable_auto_tool_choice: str | None = None

    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    limit_mm_per_prompt: str | None = None
    extra_args: str = ""
    env: dict[str, str] = Field(default_factory=dict)


_SIMPLE_FLAGS: list[tuple[str, str]] = [
    ("served_model_name", "--served-model-name"),
    ("api_key", "--api-key"),
    ("tensor_parallel_size", "--tensor-parallel-size"),
    ("pipeline_parallel_size", "--pipeline-parallel-size"),
    ("distributed_executor_backend", "--distributed-executor-backend"),
    ("dtype", "--dtype"),
    ("quantization", "--quantization"),
    ("linear_backend", "--linear-backend"),
    ("attention_backend", "--attention-backend"),
    ("mamba_backend", "--mamba-backend"),
    ("mamba_cache_dtype", "--mamba-cache-dtype"),
    ("max_model_len", "--max-model-len"),
    ("max_num_seqs", "--max-num-seqs"),
    ("max_num_batched_tokens", "--max-num-batched-tokens"),
    ("gpu_memory_utilization", "--gpu-memory-utilization"),
    ("kv_cache_dtype", "--kv-cache-dtype"),
    ("block_size", "--block-size"),
    ("swap_space", "--swap-space"),
    ("cpu_offload_gb", "--cpu-offload-gb"),
    ("num_gpu_blocks_override", "--num-gpu-blocks-override"),
    ("reasoning_parser", "--reasoning-parser"),
    ("tool_call_parser", "--tool-call-parser"),
    ("limit_mm_per_prompt", "--limit-mm-per-prompt"),
]

_TRISTATE_FLAGS: list[tuple[str, str]] = [
    ("enforce_eager", "enforce-eager"),
    ("enable_chunked_prefill", "enable-chunked-prefill"),
    ("enable_prefix_caching", "enable-prefix-caching"),
    ("trust_remote_code", "trust-remote-code"),
    ("enable_auto_tool_choice", "enable-auto-tool-choice"),
]

_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _validate_model(spec: LaunchSpec) -> str:
    """Only allow models that discovery found, to keep launch input off the free-form path."""
    known = {entry["id"]: entry for entry in discover_models()}
    if spec.model in known:
        return known[spec.model]["path"]
    known_paths = {entry["path"] for entry in known.values()}
    if spec.model in known_paths:
        return spec.model
    raise HTTPException(status_code=400, detail=f"Unknown model: {spec.model}")


def build_command(spec: LaunchSpec) -> tuple[list[str], dict[str, str]]:
    model_path = _validate_model(spec)

    if not 1 <= spec.port <= 65535:
        raise HTTPException(status_code=400, detail="Port must be between 1 and 65535")
    if spec.gpu_memory_utilization is not None and not 0.05 <= spec.gpu_memory_utilization <= 1.0:
        raise HTTPException(status_code=400, detail="gpu-memory-utilization must be in (0.05, 1.0]")

    argv: list[str] = [str(VLLM_BIN), "serve", model_path, "--host", spec.host, "--port", str(spec.port)]

    for field_name, flag in _SIMPLE_FLAGS:
        value = getattr(spec, field_name)
        if value is None or value == "":
            continue
        argv += [flag, str(value)]

    for field_name, flag in _TRISTATE_FLAGS:
        value = getattr(spec, field_name)
        if value == "on":
            argv.append(f"--{flag}")
        elif value == "off":
            argv.append(f"--no-{flag}")

    if spec.extra_args.strip():
        try:
            extra = shlex.split(spec.extra_args)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse extra args: {exc}") from exc
        if extra and not extra[0].startswith("-"):
            raise HTTPException(status_code=400, detail="Extra args must start with a flag")
        argv += extra

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/mnt/vllmdata")
    env.setdefault("HF_HUB_OFFLINE", "1")
    # vLLM 0.26 pins flashinfer-python==0.6.14 but flashinfer-cubin has no 0.6.14 release.
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if spec.gpu_indices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in sorted(set(spec.gpu_indices)))
    for key, value in spec.env.items():
        if not _ENV_KEY_RE.match(key):
            raise HTTPException(status_code=400, detail=f"Invalid environment variable name: {key}")
        env[key] = str(value)
    return argv, env


# --------------------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------------------
class EventLog:
    """Sequenced line buffer with SSE fan-out, shared by the runtime and the downloader."""

    def __init__(self, maxlen: int = LOG_BUFFER):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.events: deque = deque(maxlen=maxlen)
        self.sequence = 0

    def append(self, line: str):
        with self.condition:
            self.sequence += 1
            self.events.append(
                {"sequence": self.sequence, "timestamp": time.time(), "line": line.rstrip("\n")}
            )
            self.condition.notify_all()

    def clear(self):
        with self.condition:
            self.events.clear()
            self.sequence = 0

    def snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.events)

    def stream(self) -> Iterator[str]:
        last = 0
        while True:
            with self.condition:
                pending = [e for e in self.events if e["sequence"] > last]
                if not pending:
                    self.condition.wait(timeout=10)
                    pending = [e for e in self.events if e["sequence"] > last]
                if pending:
                    payloads = []
                    for event in pending:
                        last = event["sequence"]
                        payloads.append(f"data: {json.dumps(event)}\n\n")
                else:
                    payloads = [": heartbeat\n\n"]
            yield from payloads


@dataclass
class Runtime:
    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(init=False)
    events: deque = field(default_factory=lambda: deque(maxlen=LOG_BUFFER))
    sequence: int = 0
    process: subprocess.Popen | None = None
    spec: LaunchSpec | None = None
    command: list[str] = field(default_factory=list)
    started_at: float | None = None

    def __post_init__(self):
        self.condition = threading.Condition(self.lock)

    # -- logging ------------------------------------------------------------------
    def _append_locked(self, line: str):
        self.sequence += 1
        self.events.append(
            {"sequence": self.sequence, "timestamp": time.time(), "line": line.rstrip("\n")}
        )
        self.condition.notify_all()

    def log(self, line: str):
        with self.lock:
            self._append_locked(line)

    def stream(self) -> Iterator[str]:
        last = 0
        while True:
            with self.condition:
                pending = [e for e in self.events if e["sequence"] > last]
                if not pending:
                    self.condition.wait(timeout=10)
                    pending = [e for e in self.events if e["sequence"] > last]
                if pending:
                    payloads = []
                    for event in pending:
                        last = event["sequence"]
                        payloads.append(f"data: {json.dumps(event)}\n\n")
                else:
                    payloads = [": heartbeat\n\n"]
            yield from payloads

    # -- lifecycle ----------------------------------------------------------------
    def start(self, spec: LaunchSpec) -> dict:
        argv, env = build_command(spec)
        with self.lock:
            if self.process and self.process.poll() is None:
                raise HTTPException(status_code=409, detail="A vLLM process is already running")
            self.events.clear()
            self.sequence = 0
            self._append_locked(f"$ {shlex.join(argv)}")
            visible = env.get("CUDA_VISIBLE_DEVICES", "all")
            self._append_locked(f"CUDA_VISIBLE_DEVICES={visible}  HF_HOME={env.get('HF_HOME')}")
            try:
                process = subprocess.Popen(
                    argv,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                self._append_locked(f"Failed to start vLLM: {exc}")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            self.process = process
            self.spec = spec
            self.command = argv
            self.started_at = time.time()
            self._append_locked(f"Started vLLM pid={process.pid}")
        threading.Thread(target=self._pump, args=(process,), daemon=True).start()
        return self.status()

    def _pump(self, process: subprocess.Popen):
        try:
            if process.stdout:
                for line in process.stdout:
                    with self.lock:
                        if process is self.process:
                            self._append_locked(line)
        finally:
            code = process.wait()
            with self.lock:
                if process is self.process:
                    self._append_locked(f"vLLM exited with code {code}")

    def stop(self) -> dict:
        with self.lock:
            process = self.process
            if not process or process.poll() is not None:
                self._append_locked("No vLLM process is running.")
                return self._status_locked()
            self._append_locked(f"Stopping vLLM pid={process.pid} (SIGTERM)...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.log("Still alive after SIGTERM; sending SIGKILL...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.log(f"Failed to kill process group: {exc}")
        except (OSError, ProcessLookupError) as exc:
            self.log(f"Failed to signal process group: {exc}")
        return self.status()

    # -- status -------------------------------------------------------------------
    def _status_locked(self) -> dict:
        process = self.process
        running = bool(process and process.poll() is None)
        return {
            "running": running,
            "pid": process.pid if process else None,
            "returncode": process.poll() if process else None,
            "uptime": time.time() - self.started_at if running and self.started_at else None,
            "model": self.spec.model if self.spec else None,
            "port": self.spec.port if self.spec else None,
            "command": self.command,
        }

    def status(self) -> dict:
        with self.lock:
            snapshot = self._status_locked()
            port = self.spec.port if self.spec else None
            key = self.spec.api_key if self.spec else None
        snapshot["endpoint"] = self._probe(port, key) if port else {"online": False}
        return snapshot

    @staticmethod
    def _probe(port: int, api_key: str | None) -> dict:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            response = httpx.get(
                f"http://127.0.0.1:{port}/v1/models",
                headers=headers,
                timeout=httpx.Timeout(1.5, connect=0.4),
            )
            response.raise_for_status()
            models = [item.get("id") for item in response.json().get("data", []) if item.get("id")]
            return {"online": True, "models": models, "error": None}
        except (httpx.HTTPError, ValueError) as exc:
            return {"online": False, "models": [], "error": str(exc)}


runtime = Runtime()


# --------------------------------------------------------------------------------------
# Hugging Face downloads
# --------------------------------------------------------------------------------------
HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
HF_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HF_REVISION_RE = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")


def parse_repo_id(value: str) -> str:
    """Accept a bare repo id or any huggingface.co URL and return 'org/name'."""
    value = (value or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Enter a model URL or repo id")

    if "://" in value:
        parsed = urlparse(value)
        if parsed.netloc.lower() not in HF_HOSTS:
            raise HTTPException(status_code=400, detail="Only huggingface.co URLs are supported")
        parts = [p for p in parsed.path.split("/") if p]
        if parts and parts[0] in {"models", "datasets", "spaces"}:
            parts = parts[1:]
        for marker in ("tree", "blob", "resolve"):
            if marker in parts:
                parts = parts[: parts.index(marker)]
        value = "/".join(parts[:2])

    segments = value.split("/")
    if len(segments) != 2 or not all(HF_SEGMENT_RE.match(s) for s in segments):
        raise HTTPException(status_code=400, detail=f"Could not parse a repo id from: {value}")
    return value


@dataclass
class Downloader:
    log: EventLog = field(default_factory=lambda: EventLog(2000))
    lock: threading.Lock = field(default_factory=threading.Lock)
    process: subprocess.Popen | None = None
    repo: str | None = None
    started_at: float | None = None
    finished: bool = False
    returncode: int | None = None

    def start(self, repo: str, revision: str | None, include: str | None) -> dict:
        with self.lock:
            if self.process and self.process.poll() is None:
                raise HTTPException(status_code=409, detail="A download is already running")

            argv = [str(VLLM_BIN.parent / "hf"), "download", repo]
            if revision:
                if not HF_REVISION_RE.match(revision):
                    raise HTTPException(status_code=400, detail="Invalid revision")
                argv += ["--revision", revision]
            if include:
                for pattern in shlex.split(include):
                    argv += ["--include", pattern]

            env = os.environ.copy()
            env.setdefault("HF_HOME", "/mnt/vllmdata")
            env["HF_HUB_OFFLINE"] = "0"  # the launcher defaults to offline; downloads need the network
            env["PYTHONUNBUFFERED"] = "1"

            self.log.clear()
            self.log.append(f"$ {shlex.join(argv)}")
            self.log.append(f"HF_HOME={env['HF_HOME']}")
            try:
                process = subprocess.Popen(
                    argv,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                self.log.append(f"Failed to start download: {exc}")
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            self.process = process
            self.repo = repo
            self.started_at = time.time()
            self.finished = False
            self.returncode = None

        threading.Thread(target=self._pump, args=(process,), daemon=True).start()
        return self.status()

    def _pump(self, process: subprocess.Popen):
        try:
            if process.stdout:
                for line in process.stdout:
                    self.log.append(line)
        finally:
            code = process.wait()
            with self.lock:
                self.finished = True
                self.returncode = code
            self.log.append(f"Download finished with code {code}")
            if code == 0:
                discover_models(force=True)
                self.log.append("Model list refreshed.")

    def cancel(self) -> dict:
        with self.lock:
            process = self.process
        if not process or process.poll() is not None:
            self.log.append("No download is running.")
            return self.status()
        self.log.append("Cancelling download...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError) as exc:
            self.log.append(f"Failed to cancel: {exc}")
        return self.status()

    def status(self) -> dict:
        with self.lock:
            process = self.process
            running = bool(process and process.poll() is None)
            return {
                "running": running,
                "repo": self.repo,
                "pid": process.pid if process else None,
                "returncode": self.returncode,
                "elapsed": time.time() - self.started_at if self.started_at else None,
            }


downloader = Downloader()


@app.post("/api/download")
def api_download(body: dict):
    repo = parse_repo_id(body.get("repo", ""))
    return downloader.start(repo, body.get("revision") or None, body.get("include") or None)


@app.post("/api/download/cancel")
def api_download_cancel():
    return downloader.cancel()


@app.get("/api/download/status")
def api_download_status():
    return downloader.status()


@app.get("/api/download/logs")
def api_download_logs():
    return {"logs": downloader.log.snapshot()}


@app.get("/api/download/logs/stream")
def api_download_stream():
    return StreamingResponse(
        downloader.log.stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/api/download/resolve")
def api_download_resolve(body: dict):
    return {"repo": parse_repo_id(body.get("repo", ""))}


# --------------------------------------------------------------------------------------
# Chat proxy to the loaded model
# --------------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    messages: list[dict]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None
    enable_thinking: bool | None = None
    stream: bool = True


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    status = runtime.status()
    if not status["running"]:
        raise HTTPException(status_code=409, detail="No model is loaded")
    models = status.get("endpoint", {}).get("models") or []
    if not models:
        raise HTTPException(status_code=409, detail="The model is still loading")

    payload: dict[str, Any] = {"model": models[0], "messages": req.messages, "stream": req.stream}
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
    ):
        value = getattr(req, key)
        if value is not None:
            payload[key] = value

    extra: dict[str, Any] = {}
    if req.top_k is not None:
        extra["top_k"] = req.top_k
    if req.repetition_penalty is not None:
        extra["repetition_penalty"] = req.repetition_penalty
    if req.enable_thinking is not None:
        extra["chat_template_kwargs"] = {"enable_thinking": req.enable_thinking}
    payload.update(extra)
    if req.stream:
        payload["stream_options"] = {"include_usage": True}

    with runtime.lock:
        port = runtime.spec.port if runtime.spec else None
        api_key = runtime.spec.api_key if runtime.spec else None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    if not req.stream:
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=600)
            return response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    def relay() -> Iterator[str]:
        try:
            with httpx.stream(
                "POST", url, json=payload, headers=headers, timeout=httpx.Timeout(600, connect=10)
            ) as response:
                if response.status_code != 200:
                    detail = response.read().decode("utf-8", "replace")[:400]
                    yield f"data: {json.dumps({'error': detail})}\n\n"
                    return
                for line in response.iter_lines():
                    if line:
                        yield f"{line}\n\n"
        except httpx.HTTPError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )



# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/api/system")
def api_system():
    return system_info()


@app.get("/api/gpus")
def api_gpus():
    return {"gpus": gpu_snapshot()}


@app.get("/api/models")
def api_models(refresh: bool = False, include_missing: bool = False):
    models = discover_models(force=refresh)
    incomplete = sum(1 for m in models if m["download"]["state"] != "ok")
    if not include_missing:
        models = [m for m in models if m["download"]["state"] == "ok"]
    return {"models": models, "incomplete_count": incomplete}


def _cache_root_for(entry: dict) -> Path:
    """The directory to remove: the whole models--org--name repo, or a plain checkpoint dir."""
    path = Path(entry["path"]).resolve()
    for parent in (path, *path.parents):
        if parent.name.startswith("models--"):
            return parent
    return path


@app.post("/api/models/delete")
def api_delete_models(body: dict):
    requested = body.get("models") or []
    if not isinstance(requested, list) or not requested:
        raise HTTPException(status_code=400, detail="No models specified")

    known = {entry["id"]: entry for entry in discover_models()}
    roots = [root.resolve() for root in MODEL_ROOTS if root.is_dir()]
    active = runtime.status()
    deleted, freed = [], 0

    for model_id in requested:
        entry = known.get(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        if active["running"] and active["model"] == model_id:
            raise HTTPException(
                status_code=409, detail=f"{model_id} is currently loaded. Unload it first."
            )
        target = _cache_root_for(entry)
        # Refuse anything that is not strictly inside a configured model root.
        if not any(target.is_relative_to(root) and target != root for root in roots):
            raise HTTPException(status_code=400, detail=f"Refusing to delete outside model roots: {target}")
        if not target.is_dir():
            continue
        freed += _dir_size(target)
        shutil.rmtree(target)
        deleted.append({"id": model_id, "path": str(target)})

    discover_models(force=True)
    return {"deleted": deleted, "freed_bytes": freed}


@app.post("/api/preview")
def api_preview(spec: LaunchSpec):
    argv, env = build_command(spec)
    overrides = {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE") if k in env}
    return {"command": shlex.join(argv), "argv": argv, "env": overrides}


@app.post("/api/launch")
def api_launch(spec: LaunchSpec):
    status = runtime.start(spec)
    _save_profile(spec)  # remember the config that actually got used
    return status


@app.post("/api/stop")
def api_stop():
    return runtime.stop()


@app.get("/api/status")
def api_status():
    return runtime.status()


@app.get("/api/logs")
def api_logs():
    with runtime.lock:
        return {"logs": list(runtime.events)}


@app.get("/api/logs/stream")
def api_logs_stream():
    return StreamingResponse(
        runtime.stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


class Preset(BaseModel):
    id: str
    name: str
    spec: LaunchSpec


def _profile_path(model_id: str) -> Path:
    # Model ids contain '/', so hash them into a flat, traversal-safe filename.
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:24]
    return PROFILE_DIR / f"{digest}.json"


def _save_profile(spec: LaunchSpec) -> None:
    payload = {"model": spec.model, "saved_at": time.time(), "spec": spec.model_dump()}
    _profile_path(spec.model).write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.get("/api/profiles")
def api_profiles():
    profiles = {}
    for path in PROFILE_DIR.glob("*.json"):
        data = _read_json(path)
        if data.get("model") and data.get("spec"):
            profiles[data["model"]] = data
    return {"profiles": profiles}


@app.put("/api/profiles")
def api_save_profile(spec: LaunchSpec):
    _save_profile(spec)
    return {"saved": spec.model}


@app.delete("/api/profiles")
def api_delete_profile(model: str):
    path = _profile_path(model)
    if path.exists():
        path.unlink()
    return {"deleted": model}


def _preset_path(preset_id: str) -> Path:
    if not PRESET_ID_RE.match(preset_id):
        raise HTTPException(status_code=400, detail="Invalid preset id")
    return PRESET_DIR / f"{preset_id}.json"


@app.get("/api/presets")
def api_presets():
    presets = []
    for path in sorted(PRESET_DIR.glob("*.json")):
        data = _read_json(path)
        if data.get("id"):
            presets.append(data)
    return {"presets": presets}


@app.put("/api/presets/{preset_id}")
def api_save_preset(preset_id: str, preset: Preset):
    path = _preset_path(preset_id)
    payload = preset.model_dump()
    payload["id"] = preset_id
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@app.delete("/api/presets/{preset_id}")
def api_delete_preset(preset_id: str):
    path = _preset_path(preset_id)
    if path.exists():
        path.unlink()
    return {"deleted": preset_id}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("VLLM_LAUNCHER_HOST", "0.0.0.0"),
        port=int(os.environ.get("VLLM_LAUNCHER_PORT", "7870")),
        log_level="info",
        # Long-lived SSE log streams never close on their own, so cap graceful shutdown.
        timeout_graceful_shutdown=5,
    )

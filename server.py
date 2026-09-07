#!/usr/bin/env python
"""LAN-only model launcher: discover local checkpoints, configure launch flags, run and monitor
`vllm serve` or `sglang.launch_server`, and chat with whatever is loaded."""

from __future__ import annotations

import asyncio
import ipaddress
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATE_DIR = Path(os.environ.get("VLLM_LAUNCHER_TEMPLATES", APP_DIR / "templates"))
PRESET_DIR = Path(os.environ.get("VLLM_LAUNCHER_PRESETS", APP_DIR / "presets"))
PRESET_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR = Path(os.environ.get("VLLM_LAUNCHER_PROFILES", APP_DIR / "profiles"))
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Engine discovery
# --------------------------------------------------------------------------------------
# Each engine is addressed by the *interpreter* of the environment that has it installed;
# console scripts are optional. Everything (launch argv, version probe, CUDA toolchain,
# PATH) is derived from that interpreter, so an engine that is not installed is reported
# as unavailable instead of falling through to whatever `python` happens to be on PATH.
ENGINE_CHOICES = ("vllm", "sglang")
_ENGINE_MODULES = {"vllm": "vllm", "sglang": "sglang"}
_CONDA_ROOTS = (
    "~/miniconda3", "~/anaconda3", "~/miniforge3", "~/mambaforge", "~/micromamba",
    "/opt/conda", "/opt/miniconda3", "/opt/anaconda3",
)
_VENV_PATTERNS = (
    "~/{name}/.venv", "~/{name}-env", "~/{name}_env", "~/.venvs/{name}", "~/venvs/{name}",
    "~/envs/{name}", "~/.virtualenvs/{name}", "/opt/{name}",
)


def _env_prefix(python: Path) -> Path:
    """`<prefix>/bin/python` -> `<prefix>` (conda envs and venvs share this layout)."""
    return python.resolve().parent.parent


def _site_packages(python: Path) -> list[Path]:
    # Conda ships a `lib/python3.1 -> python3.12` compatibility symlink, so dedupe by
    # resolved path or every lookup below sees each directory twice.
    prefix = _env_prefix(python)
    unique: dict[Path, Path] = {}
    for pattern in ("lib/python3.*/site-packages", "lib64/python3.*/site-packages"):
        for site in sorted(prefix.glob(pattern)):
            unique.setdefault(site.resolve(), site.resolve())
    return list(unique.values())


def _env_has_module(python: Path, module: str) -> bool:
    if python.resolve() == Path(sys.executable).resolve():
        import importlib.util

        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False
    return any((site / module).is_dir() for site in _site_packages(python))


def _python_for_script(script: Path) -> Path | None:
    """Console scripts start with `#!/path/to/python`; that interpreter is the env."""
    sibling = script.parent / "python"
    if sibling.exists():
        return sibling
    try:
        with script.open("rb") as handle:
            first = handle.readline(512).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if first.startswith("#!"):
        interpreter = Path(first[2:].split()[0]) if first[2:].split() else None
        if interpreter and interpreter.exists() and "python" in interpreter.name:
            return interpreter
    return None


@dataclass(frozen=True)
class Engine:
    name: str
    python: Path | None  # interpreter of the environment that has the engine installed
    script: Path | None  # console script (`vllm` / `sglang`) when one exists

    @property
    def available(self) -> bool:
        return self.python is not None

    @property
    def bin_dir(self) -> Path | None:
        return self.python.parent if self.python else None

    def cuda_root(self) -> Path | None:
        """The pip-installed CUDA toolkit (`nvidia/cu13/`) inside *this* engine's env, if any.
        Preferring it over an inherited CUDA_HOME keeps nvcc/libcudart matched to the torch
        build the engine was compiled against."""
        if not self.python:
            return None
        for site in _site_packages(self.python):
            # `cu12`, `cu13`, ... only: `nvidia/cu*` would also match cudnn/cusparselt/cublas.
            roots = [p for p in site.glob("nvidia/cu[0-9]*") if p.name[2:].isdigit()]
            for root in sorted(roots, key=lambda p: int(p.name[2:]), reverse=True):
                if (root / "bin" / "nvcc").exists() or (root / "lib").is_dir():
                    return root
        return None

    def describe(self) -> dict:
        return {
            "available": self.available,
            "python": str(self.python) if self.python else None,
            "bin": str(self.script) if self.script else None,
        }


def _resolve_engine(name: str) -> Engine:
    module = _ENGINE_MODULES[name]
    upper = name.upper()

    explicit_python = os.environ.get(f"{upper}_PYTHON")
    if explicit_python:
        python = Path(explicit_python).expanduser()
        script = python.parent / name
        return Engine(name, python if python.exists() else None, script if script.exists() else None)

    explicit_bin = os.environ.get(f"{upper}_BIN")
    if explicit_bin:
        script = Path(explicit_bin).expanduser()
        if script.exists():
            return Engine(name, _python_for_script(script), script)
        return Engine(name, None, None)

    candidates: list[Path] = [Path(sys.executable)]
    try:
        envs_dir = _env_prefix(Path(sys.executable)).parent  # .../envs
        candidates.append(envs_dir / name / "bin" / "python")
    except (OSError, IndexError):
        pass
    for root in _CONDA_ROOTS:
        candidates.append(Path(root).expanduser() / "envs" / name / "bin" / "python")
    for pattern in _VENV_PATTERNS:
        candidates.append(Path(pattern.format(name=name)).expanduser() / "bin" / "python")
    which = shutil.which(name)
    if which:
        found = _python_for_script(Path(which))
        if found:
            candidates.append(found)

    seen: set[Path] = set()
    for python in candidates:
        try:
            key = python.resolve()
        except OSError:
            continue
        if key in seen or not python.exists():
            continue
        seen.add(key)
        if _env_has_module(python, module):
            script = python.parent / name
            return Engine(name, python, script if script.exists() else None)
    return Engine(name, None, None)


ENGINES: dict[str, Engine] = {name: _resolve_engine(name) for name in ENGINE_CHOICES}


def engine_for(name: str) -> Engine:
    engine = ENGINES[name]
    if not engine.available:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{name} is not installed on this machine. Point {name.upper()}_PYTHON at the "
                f"interpreter of the environment that has it (or {name.upper()}_BIN at the "
                f"`{name}` executable) and restart the launcher."
            ),
        )
    return engine


def _hf_cli() -> list[str]:
    """`hf download` from whichever environment has huggingface_hub: the engines' envs first,
    then the launcher's own interpreter, then PATH."""
    pythons = [e.python for e in ENGINES.values() if e.python] + [Path(sys.executable)]
    for python in pythons:
        for name in ("hf", "huggingface-cli"):
            script = python.parent / name
            if script.exists():
                return [str(script)]
    for name in ("hf", "huggingface-cli"):
        found = shutil.which(name)
        if found:
            return [found]
    for python in pythons:
        if _env_has_module(python, "huggingface_hub"):
            return [str(python), "-m", "huggingface_hub.cli.hf"]
    return ["hf"]

# Hugging Face's own default location when HF_HOME is not set.
HF_HOME = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")

# An OpenAI-compatible server on this port may be owned by another manager. The launcher
# adopts it read-only rather than competing for the GPUs; EXTERNAL_NAME is only used in
# messages that tell the user where to go to stop it.
EXTERNAL_PORT = int(os.environ.get("VLLM_LAUNCHER_EXTERNAL_PORT", "8000"))
EXTERNAL_API_KEY = os.environ.get("VLLM_LAUNCHER_EXTERNAL_API_KEY") or None
EXTERNAL_NAME = os.environ.get("VLLM_LAUNCHER_EXTERNAL_NAME") or "the tool that launched it"

# Directories scanned for models. HF-cache layouts and plain checkpoint dirs are both handled.
MODEL_ROOTS = [
    Path(p).expanduser()
    for p in os.environ.get(
        "VLLM_LAUNCHER_MODEL_ROOTS",
        f"{HF_HOME}/hub:~/models",
    ).split(":")
    if p.strip()
]

LOG_BUFFER = int(os.environ.get("VLLM_LAUNCHER_LOG_LINES", "4000"))
# How many of the most recent log lines to hand to a newly connected SSE
# client (and how many the browser renders). The full buffer stays available
# at /api/logs for debugging.
LOG_STREAM_REPLAY = min(
    LOG_BUFFER, int(os.environ.get("VLLM_LAUNCHER_STREAM_REPLAY", "1000"))
)
PRESET_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


# Host-RAM guards. Loading a model can pull the whole desktop down: kernel JIT (nvcc's
# `cicc` is ~6 GiB *per parallel job*), CPU offload and weight staging all land in
# anonymous memory, and once the box starts swapping it is usually too late for the OOM
# killer to log anything. Three layers, all tunable:
#   1. refuse to launch when MemAvailable is already below MIN_FREE_RAM_GIB;
#   2. derive MAX_JOBS (ninja parallelism used by FlashInfer / SGLang / tvm-ffi JIT) from
#      the RAM that is actually free, never raising an inherited value;
#   3. a watchdog SIGKILLs the engine's process group if MemAvailable drops below
#      RAM_KILL_GIB or the kernel reports the system fully stalled on memory (PSI).
MIN_FREE_RAM_GIB = _env_float("VLLM_LAUNCHER_MIN_FREE_RAM_GIB", 4.0)
RAM_KILL_GIB = _env_float("VLLM_LAUNCHER_RAM_KILL_GIB", 2.0)
RAM_PSI_FULL_KILL = _env_float("VLLM_LAUNCHER_RAM_PSI_FULL", 25.0)  # % of last 10 s stalled
JIT_RAM_PER_JOB_GIB = _env_float("VLLM_LAUNCHER_JIT_RAM_PER_JOB_GIB", 6.0)
JIT_RAM_HEADROOM_GIB = _env_float("VLLM_LAUNCHER_JIT_RAM_HEADROOM_GIB", 6.0)
JIT_MAX_JOBS_CAP = max(1, int(_env_float("VLLM_LAUNCHER_JIT_MAX_JOBS", 4)))
GIB = 1024**3


def ram_snapshot() -> dict:
    """MemAvailable is the kernel's own estimate of what can be handed out without
    swapping (it already discounts reclaimable page cache), which makes it the right
    single number to guard on. PSI is optional (needs CONFIG_PSI / psi=1)."""
    info: dict[str, Any] = {
        "total_bytes": None, "available_bytes": None,
        "swap_total_bytes": None, "swap_free_bytes": None, "psi_full_avg10": None,
    }
    keys = {
        "MemTotal": "total_bytes", "MemAvailable": "available_bytes",
        "SwapTotal": "swap_total_bytes", "SwapFree": "swap_free_bytes",
    }
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                if name in keys:
                    info[keys[name]] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/pressure/memory", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("full"):
                    info["psi_full_avg10"] = float(line.split("avg10=")[1].split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return info


def jit_jobs_for(available_bytes: int | None, inherited: str | None) -> int:
    """Parallel nvcc jobs that fit in the RAM that is free right now, leaving headroom for
    the engine processes themselves. An inherited MAX_JOBS is only ever lowered."""
    cap = JIT_MAX_JOBS_CAP
    if inherited and inherited.isdigit() and int(inherited) > 0:
        cap = min(cap, int(inherited))
    if available_bytes is None:
        return min(cap, 2)
    fits = int((available_bytes / GIB - JIT_RAM_HEADROOM_GIB) // JIT_RAM_PER_JOB_GIB)
    return max(1, min(cap, fits))

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
_version_cache: dict[str, dict] = {}
_version_probe_done = threading.Event()


def _probe_engine_versions(engine: Engine) -> dict:
    """Import the engine inside *its own* interpreter (a heavy import: ~10-20 s)."""
    if not engine.python:
        return {}
    module = _ENGINE_MODULES[engine.name]
    code = (
        "import json,torch\n"
        f"import {module} as m\n"
        "print(json.dumps({'version': getattr(m,'__version__',None),"
        "'torch': torch.__version__, 'cuda': torch.version.cuda}))"
    )
    try:
        out = subprocess.run(
            [str(engine.python), "-c", code],
            capture_output=True, text=True, timeout=240, check=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"},
        ).stdout
        return json.loads(out.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError, IndexError):
        return {}


def _probe_all_versions() -> None:
    threads = []
    for engine in ENGINES.values():
        if not engine.available:
            continue

        def worker(e: Engine = engine) -> None:
            _version_cache[e.name] = _probe_engine_versions(e)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    _version_probe_done.set()


# Kick the probes off at import so they overlap with uvicorn start-up and the first
# browser connection; system_info() just waits for the result.
threading.Thread(target=_probe_all_versions, daemon=True).start()


def system_info() -> dict:
    if not _system_cache:
        _version_probe_done.wait(timeout=300)
        engines = {}
        versions: dict[str, Any] = {"vllm": None, "sglang": None, "torch": None, "cuda": None}
        for name, engine in ENGINES.items():
            probed = _version_cache.get(name, {})
            engines[name] = {**engine.describe(), "version": probed.get("version"),
                             "torch": probed.get("torch"), "cuda": probed.get("cuda")}
            versions[name] = probed.get("version")
            # torch/CUDA shown in the header: whichever engine answered first (vLLM preferred).
            if probed.get("torch") and not versions["torch"]:
                versions["torch"] = probed.get("torch")
                versions["cuda"] = probed.get("cuda")

        gpus = gpu_snapshot()
        caps = [g["compute_cap"] for g in gpus]
        cap_major_minor = min((tuple(int(x) for x in c.split(".")) for c in caps), default=(0, 0))
        _system_cache.update(
            {
                "versions": versions,
                "engines": engines,
                "model_roots": [str(p) for p in MODEL_ROOTS],
                "capability": f"{cap_major_minor[0]}.{cap_major_minor[1]}" if caps else None,
                "capability_int": cap_major_minor[0] * 10 + cap_major_minor[1],
                "gpu_count": len(gpus),
                "external_port": EXTERNAL_PORT,
                "hf_home": HF_HOME,
                "hf_cli": shlex.join(_hf_cli()),
                "ram_guard": {
                    "min_free_gib": MIN_FREE_RAM_GIB,
                    "kill_gib": RAM_KILL_GIB,
                    "psi_full_kill": RAM_PSI_FULL_KILL,
                    "jit_ram_per_job_gib": JIT_RAM_PER_JOB_GIB,
                },
                "access_urls": [
                    f"http://{addr}:{os.environ.get('VLLM_LAUNCHER_PORT', '7870')}"
                    for addr in _local_addresses()
                ],
            }
        )
    info = dict(_system_cache)
    info["gpus"] = gpu_snapshot()
    info["ram"] = ram_snapshot()
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
def _coerce_engine(value: Any) -> str:
    """Normalise the engine name. A *missing* value defaults to vLLM so that saved profiles
    which predate the field still launch; an explicit unknown name is an error rather than
    a silent launch of the wrong engine."""
    engine = str(value or "vllm").strip().lower()
    if engine in ENGINE_CHOICES:
        return engine
    if engine in {"sg", "sgl", "s"}:
        return "sglang"
    raise ValueError(f"Unknown engine {value!r}; expected one of {', '.join(ENGINE_CHOICES)}")


class LaunchSpec(BaseModel):
    model: str
    engine: str = "vllm"
    served_model_name: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None

    @field_validator("engine", mode="before")
    @classmethod
    def _normalise_engine(cls, value: Any) -> str:
        return _coerce_engine(value)

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

    # -- SGLang-specific knobs (ignored when engine == "vllm"). Each field is
    # deliberately named so it can co-exist with the vLLM field of the same
    # concept without a cross-engine collision. -------------------------------
    sglang_max_total_tokens: int | None = None           # --max-total-tokens
    sglang_schedule_policy: str | None = None         # --schedule-policy
    sglang_load_format: str | None = None             # --load-format
    sglang_dp_size: int | None = None                 # --dp-size
    sglang_ep_size: int | None = None                 # --ep-size
    sglang_stream_interval: int | None = None         # --stream-interval
    # Modern CUDA-graph control (`--disable-cuda-graph` is deprecated in 0.5.x).
    sglang_cuda_graph_backend_decode: str | None = None  # --cuda-graph-backend-decode
    sglang_cuda_graph_backend_prefill: str | None = None # --cuda-graph-backend-prefill
    # Hybrid (mamba/linear-attention) models: first-class in SGLang.
    sglang_mamba_backend: str | None = None             # --mamba-backend {triton,flashinfer}
    sglang_mamba_ssm_dtype: str | None = None           # --mamba-ssm-dtype {float32,bfloat16,float16}
    sglang_max_mamba_cache_size: int | None = None      # --max-mamba-cache-size
    sglang_mamba_full_memory_ratio: float | None = None # --mamba-full-memory-ratio
    # Speculative decoding / MTP
    sglang_speculative_algorithm: str | None = None  # --speculative-algorithm
    sglang_speculative_draft_model: str | None = None  # --speculative-draft-model-path
    sglang_speculative_num_steps: int | None = None    # --speculative-num-steps
    sglang_speculative_eagle_topk: int | None = None   # --speculative-eagle-topk
    sglang_speculative_num_draft_tokens: int | None = None  # --speculative-num-draft-tokens
    # Booleans (present in argv only when "on").
    sglang_disable_radix_cache: str | None = None        # --disable-radix-cache
    sglang_disable_overlap_schedule: str | None = None   # --disable-overlap-schedule
    sglang_enable_deterministic_inference: str | None = None  # --enable-deterministic-inference
    sglang_enable_multimodal: str | None = None          # --enable-multimodal
    sglang_enable_dp_attention: str | None = None        # --enable-dp-attention


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


def _validate_common(spec: LaunchSpec) -> None:
    if not 1 <= spec.port <= 65535:
        raise HTTPException(status_code=400, detail="Port must be between 1 and 65535")
    if spec.gpu_memory_utilization is not None and not 0.05 <= spec.gpu_memory_utilization <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="GPU-memory-utilization must be in (0.05, 1.0]. "
                   "For SGLang this value maps to --mem-fraction-static.",
        )


def _pin_served_model_name(spec: LaunchSpec, model_path: str) -> LaunchSpec:
    """Both vLLM and SGLM advertise the path they were handed as the model id. Pinning
    the repo id keeps the client-facing model name stable across machines."""
    if not spec.served_model_name and model_path != spec.model:
        return spec.model_copy(update={"served_model_name": spec.model})
    return spec


def _append_extra_args(argv: list[str], extra_args: str) -> None:
    """Parse & append the user's free-form extra args. A bare --chat-template name that
    matches a file in the launcher's templates dir is expanded to its path so saved
    profiles stay portable; anything else (absolute paths, SGLang's built-in template
    names such as `qwen2-vl`) is passed through untouched."""
    if not extra_args.strip():
        return
    try:
        extra = shlex.split(extra_args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse extra args: {exc}") from exc
    if extra and not extra[0].startswith("-"):
        raise HTTPException(status_code=400, detail="Extra args must start with a flag")
    for index, token in enumerate(extra[:-1]):
        if token == "--chat-template":
            template = Path(extra[index + 1]).expanduser()
            local = TEMPLATE_DIR / template.name
            if not template.is_absolute() and not template.exists() and local.is_file():
                extra[index + 1] = str(local)
    argv += extra


def _resolve_env(spec: LaunchSpec, engine: Engine) -> tuple[dict[str, str], list[str]]:
    """Process environment for the engine, plus human-readable notes about what was derived.

    Everything toolchain-related is taken from the *engine's own* environment: the
    interpreter's bin dir goes first on PATH (ninja, nvcc shims), and the pip-installed
    CUDA toolkit inside that env (nvidia/cu13) provides CUDA_HOME and LD_LIBRARY_PATH.
    Values inherited from the launcher's environment (a systemd EnvironmentFile written
    for one engine) are kept only where the engine env has nothing of its own, so a
    vLLM-specific LD_LIBRARY_PATH never leaks into SGLang's process. The user's own
    `env` overrides are applied last and always win."""
    notes: list[str] = []
    env = os.environ.copy()
    env.setdefault("HF_HOME", HF_HOME)
    env.setdefault("HF_HUB_OFFLINE", "1")
    # vLLM 0.26 pins flashinfer-python==0.6.14 but flashinfer-cubin has no 0.6.14 release.
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # An EnvironmentFile written for one engine typically carries that engine's toolkit
    # paths (~/vllm-launcher.env points CUDA_HOME/LD_LIBRARY_PATH into the vLLM env).
    # Anything that resolves into *another* engine's prefix is dropped before this
    # engine's own toolkit is layered on top.
    foreign_prefixes = [
        _env_prefix(other.python) for other in ENGINES.values()
        if other.python and other.name != engine.name
    ]

    def inside_foreign(path: str) -> bool:
        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        return any(resolved.is_relative_to(prefix) for prefix in foreign_prefixes)

    if env.get("CUDA_HOME") and inside_foreign(env["CUDA_HOME"]):
        notes.append(f"dropped inherited CUDA_HOME={env['CUDA_HOME']} (belongs to another engine's env).")
        env.pop("CUDA_HOME")
    inherited_ld = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    kept_ld = [p for p in inherited_ld if not inside_foreign(p)]
    if len(kept_ld) != len(inherited_ld):
        notes.append("dropped LD_LIBRARY_PATH entries that point into another engine's env.")
    if kept_ld:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(kept_ld)
    else:
        env.pop("LD_LIBRARY_PATH", None)

    cuda_root = engine.cuda_root()
    if cuda_root is not None:
        if (cuda_root / "bin" / "nvcc").exists():
            env["CUDA_HOME"] = str(cuda_root)
            env.pop("CUDA_PATH", None)
        lib_dir = cuda_root / "lib"
        if lib_dir.is_dir():
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [str(lib_dir)] + [p for p in kept_ld if p != str(lib_dir)]
            )
    elif not env.get("CUDA_HOME") and Path("/usr/local/cuda/bin/nvcc").exists():
        env["CUDA_HOME"] = "/usr/local/cuda"

    search_path = [str(engine.bin_dir)]
    nvcc_dir = Path(env["CUDA_HOME"]) / "bin" if env.get("CUDA_HOME") else None
    if nvcc_dir and (nvcc_dir / "nvcc").exists():
        search_path.append(str(nvcc_dir))
    if env.get("PATH"):
        search_path.append(env["PATH"])
    env["PATH"] = os.pathsep.join(search_path)

    if spec.gpu_indices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in sorted(set(spec.gpu_indices)))

    # Kernel JIT parallelism sized to the RAM that is free *now* (see the guard notes at
    # the top of the file). Honoured by FlashInfer, SGLang's kernel JIT and tvm-ffi.
    ram = ram_snapshot()
    jobs = jit_jobs_for(ram["available_bytes"], env.get("MAX_JOBS"))
    env["MAX_JOBS"] = str(jobs)
    env.setdefault("FLASHINFER_NVCC_THREADS", "1")
    available = ram["available_bytes"]
    notes.append(
        f"RAM available {available / GIB:.1f} GiB -> MAX_JOBS={jobs} for kernel JIT "
        f"(~{JIT_RAM_PER_JOB_GIB:g} GiB per nvcc job); watchdog kills the engine below "
        f"{RAM_KILL_GIB:g} GiB free."
        if available is not None
        else f"RAM unknown (/proc/meminfo unreadable) -> MAX_JOBS={jobs}."
    )

    for key, value in spec.env.items():
        if not _ENV_KEY_RE.match(key):
            raise HTTPException(status_code=400, detail=f"Invalid environment variable name: {key}")
        env[key] = str(value)
    return env, notes


def _emit_value_flags(argv: list[str], spec: LaunchSpec, table: list[tuple[str, str]]) -> None:
    for field_name, flag in table:
        value = getattr(spec, field_name)
        if value is None or value == "":
            continue
        argv += [flag, str(value)]


def _build_vllm_argv(spec: LaunchSpec, model_path: str, engine: Engine) -> tuple[list[str], list[str]]:
    if engine.script:
        argv: list[str] = [str(engine.script)]
    else:
        argv = [str(engine.python), "-m", "vllm.entrypoints.cli.main"]
    argv += ["serve", model_path, "--host", spec.host, "--port", str(spec.port)]
    _emit_value_flags(argv, spec, _SIMPLE_FLAGS)
    for field_name, flag in _TRISTATE_FLAGS:
        value = getattr(spec, field_name)
        if value == "on":
            argv.append(f"--{flag}")
        elif value == "off":
            argv.append(f"--no-{flag}")
    _append_extra_args(argv, spec.extra_args)
    return argv, []


# The generic fields (dtype, gpu_memory_utilization, ...) are the *same* values the user
# tunes for vLLM; SGLang just names the flags differently, so a saved config carries over
# when the engine is switched. Flag names verified against `sglang.launch_server --help`
# for 0.5.18 (`--tp-size` not `--tp`, `--context-length`, `--mem-fraction-static`, ...).
_SGLANG_VALUE_FLAGS: list[tuple[str, str]] = [
    ("served_model_name", "--served-model-name"),
    ("api_key", "--api-key"),
    ("dtype", "--dtype"),
    ("quantization", "--quantization"),
    ("tensor_parallel_size", "--tp-size"),
    ("pipeline_parallel_size", "--pp-size"),
    ("max_model_len", "--context-length"),
    ("max_num_seqs", "--max-running-requests"),
    ("gpu_memory_utilization", "--mem-fraction-static"),
    ("kv_cache_dtype", "--kv-cache-dtype"),
    ("max_num_batched_tokens", "--chunked-prefill-size"),
    ("block_size", "--page-size"),
    ("cpu_offload_gb", "--cpu-offload-gb"),
    ("attention_backend", "--attention-backend"),
    ("reasoning_parser", "--reasoning-parser"),
    ("tool_call_parser", "--tool-call-parser"),
    ("sglang_max_total_tokens", "--max-total-tokens"),
    ("sglang_schedule_policy", "--schedule-policy"),
    ("sglang_load_format", "--load-format"),
    ("sglang_dp_size", "--dp-size"),
    ("sglang_ep_size", "--ep-size"),
    ("sglang_stream_interval", "--stream-interval"),
    ("sglang_cuda_graph_backend_decode", "--cuda-graph-backend-decode"),
    ("sglang_cuda_graph_backend_prefill", "--cuda-graph-backend-prefill"),
    ("sglang_mamba_backend", "--mamba-backend"),
    ("sglang_mamba_ssm_dtype", "--mamba-ssm-dtype"),
    ("sglang_max_mamba_cache_size", "--max-mamba-cache-size"),
    ("sglang_mamba_full_memory_ratio", "--mamba-full-memory-ratio"),
    # NEXTN == EAGLE + MTP-verify; for checkpoints with embedded MTP heads the draft
    # model path stays empty.
    ("sglang_speculative_algorithm", "--speculative-algorithm"),
    ("sglang_speculative_draft_model", "--speculative-draft-model-path"),
    ("sglang_speculative_num_steps", "--speculative-num-steps"),
    ("sglang_speculative_eagle_topk", "--speculative-eagle-topk"),
    ("sglang_speculative_num_draft_tokens", "--speculative-num-draft-tokens"),
]

# Store-true flags: "on" includes the flag, "off"/unset omits it (SGLang has no --no- forms).
_SGLANG_BOOL_FLAGS: list[tuple[str, str]] = [
    ("trust_remote_code", "--trust-remote-code"),
    ("sglang_disable_radix_cache", "--disable-radix-cache"),
    ("sglang_disable_overlap_schedule", "--disable-overlap-schedule"),
    ("sglang_enable_deterministic_inference", "--enable-deterministic-inference"),
    ("sglang_enable_multimodal", "--enable-multimodal"),
    ("sglang_enable_dp_attention", "--enable-dp-attention"),
]

# vLLM spellings that SGLang's argparse `choices` reject outright ("invalid choice" -> the
# process exits with code 2 before loading anything). Translate the well-known ones so a
# config saved for vLLM launches unchanged; unknown values pass through and SGLang's own
# error names the flag.
_SGLANG_VALUE_MAP: dict[str, dict[str, str | None]] = {
    "attention_backend": {
        "FLASH_ATTN": "fa3", "FLASHINFER": "flashinfer", "TRITON_ATTN": "triton",
        "TORCH_SDPA": "torch_native", "FLEX_ATTENTION": "flex_attention",
        "FLASHMLA": "flashmla", "CUTLASS_MLA": "cutlass_mla", "TRITON_MLA": "triton",
        "FLASHINFER_MLA": "flashinfer",
    },
    # SGLang has no fp8 alias and no fp16 KV choice ("auto" follows the model dtype).
    "kv_cache_dtype": {"fp8": "fp8_e4m3", "float16": None, "half": None, "fp16": None},
    "tool_call_parser": {
        "qwen3_xml": "qwen3_coder", "llama3_json": "llama3", "llama4_json": "llama3",
        "deepseek_v3": "deepseekv3", "deepseek_v31": "deepseekv31", "openai": "gpt-oss",
    },
    "reasoning_parser": {
        "deepseek_r1": "deepseek-r1", "deepseek_v3": "deepseek-v3",
        "openai_gptoss": "gpt-oss", "hunyuan_a13b": "hunyuan",
    },
}


def _translate_for_sglang(spec: LaunchSpec) -> tuple[LaunchSpec, list[str]]:
    """Map vLLM-shaped field values onto SGLang's vocabulary and fold the vLLM tri-states
    that have an SGLang equivalent into SGLang-only fields (only where the user has not
    set the SGLang field explicitly). Returns the adjusted spec plus notes for the log."""
    notes: list[str] = []
    updates: dict[str, Any] = {}
    for field_name, mapping in _SGLANG_VALUE_MAP.items():
        value = getattr(spec, field_name)
        if value is None or value == "":
            continue
        if value in mapping:
            target = mapping[value]
            updates[field_name] = target
            notes.append(
                f"{field_name}: {value!r} is a vLLM name; "
                + (f"using SGLang's {target!r}." if target else "SGLang has no equivalent, dropped (auto).")
            )
        elif field_name == "attention_backend" and value != value.lower():
            updates[field_name] = value.lower()
            notes.append(f"attention_backend: lower-cased {value!r} for SGLang.")

    if spec.enable_prefix_caching == "off" and not spec.sglang_disable_radix_cache:
        updates["sglang_disable_radix_cache"] = "on"
        notes.append("prefix caching off -> --disable-radix-cache.")
    if spec.enable_chunked_prefill == "off" and spec.max_num_batched_tokens is None:
        updates["max_num_batched_tokens"] = -1
        notes.append("chunked prefill off -> --chunked-prefill-size -1.")
    if spec.enforce_eager == "on":
        for field_name in ("sglang_cuda_graph_backend_decode", "sglang_cuda_graph_backend_prefill"):
            if not getattr(spec, field_name):
                updates[field_name] = "disabled"
        notes.append("enforce eager -> --cuda-graph-backend-{decode,prefill} disabled.")

    ignored = [
        label for label, value in (
            ("linear-backend", spec.linear_backend),
            ("mamba-backend (vLLM field; use the SGLang mamba section)", spec.mamba_backend),
            ("mamba-cache-dtype (vLLM field; use SGLang's mamba SSM dtype)", spec.mamba_cache_dtype),
            ("swap-space", spec.swap_space),
            ("num-gpu-blocks-override", spec.num_gpu_blocks_override),
            ("distributed-executor-backend", spec.distributed_executor_backend),
            ("limit-mm-per-prompt", spec.limit_mm_per_prompt),
            ("enable-auto-tool-choice (implicit in SGLang once a tool-call parser is set)",
             spec.enable_auto_tool_choice),
        ) if value not in (None, "")
    ]
    if ignored:
        notes.append("no SGLang equivalent, ignored: " + "; ".join(ignored) + ".")
    return (spec.model_copy(update=updates) if updates else spec), notes


def _build_sglang_argv(spec: LaunchSpec, model_path: str, engine: Engine) -> tuple[list[str], list[str]]:
    # SGLang 0.5.x takes the model as a *named* argument (`--model-path`); a bare positional
    # is rejected. Run the module through the env's interpreter so a missing console
    # script does not matter.
    spec, notes = _translate_for_sglang(spec)
    argv: list[str] = [
        str(engine.python), "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", spec.host,
        "--port", str(spec.port),
    ]
    _emit_value_flags(argv, spec, _SGLANG_VALUE_FLAGS)
    for field_name, flag in _SGLANG_BOOL_FLAGS:
        if getattr(spec, field_name) == "on":
            argv.append(flag)
    _append_extra_args(argv, spec.extra_args)
    return argv, notes


def build_command(spec: LaunchSpec) -> tuple[list[str], dict[str, str], list[str]]:
    """argv, environment and a list of human-readable notes (translations, RAM sizing)."""
    engine = engine_for(spec.engine)
    model_path = _validate_model(spec)
    spec = _pin_served_model_name(spec, model_path)
    _validate_common(spec)

    if engine.name == "sglang":
        argv, notes = _build_sglang_argv(spec, model_path, engine)
    else:
        argv, notes = _build_vllm_argv(spec, model_path, engine)
    env, env_notes = _resolve_env(spec, engine)
    return argv, env, notes + env_notes


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
        # The sequence stays monotonic: connected SSE clients track the last sequence they
        # saw, so resetting to 0 would make every line of the next run look already-seen.
        with self.condition:
            self.events.clear()

    def snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.events)

    def stream(self) -> Iterator[str]:
        with self.lock:
            last = max(0, self.sequence - LOG_STREAM_REPLAY)
        while True:
            with self.condition:
                pending = [e for e in self.events if e["sequence"] > last]
                if not pending:
                    self.condition.wait(timeout=10)
                    pending = [e for e in self.events if e["sequence"] > last]
                for event in pending:
                    last = event["sequence"]
            if pending:
                for event in pending:
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                yield ": heartbeat\n\n"


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
    adopted: bool = False

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
        # Replay at most the last LOG_STREAM_REPLAY lines: sending the full
        # ring buffer made the page stall on open, and the client only
        # renders the most recent lines anyway.
        with self.lock:
            last = max(0, self.sequence - LOG_STREAM_REPLAY)
        while True:
            with self.condition:
                pending = [e for e in self.events if e["sequence"] > last]
                if not pending:
                    self.condition.wait(timeout=10)
                    pending = [e for e in self.events if e["sequence"] > last]
                for event in pending:
                    last = event["sequence"]
            # Serialize outside the lock — json.dumps on a big replay burst
            # would otherwise stall _pump() and stop() for its duration.
            if pending:
                for event in pending:
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                yield ": heartbeat\n\n"

    # -- lifecycle ----------------------------------------------------------------
    def start(self, spec: LaunchSpec) -> dict:
        spec = spec.model_copy(update={"engine": _coerce_engine(spec.engine)})
        engine = spec.engine
        argv, env, notes = build_command(spec)
        # Probed before locking: another manager may already own this port, and the GPUs
        # cannot hold a second copy of a model even if the bind were to succeed.
        foreign = self._probe(spec.port, spec.api_key or EXTERNAL_API_KEY)
        if foreign["online"]:
            serving = ", ".join(foreign["models"]) or "an unknown model"
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Port {spec.port} is already serving {serving} from a process this launcher "
                    f"does not own. Stop it where it was started ({EXTERNAL_NAME}) first."
                ),
            )
        ram = ram_snapshot()
        available = ram["available_bytes"]
        if available is not None and available < MIN_FREE_RAM_GIB * GIB:
            raise HTTPException(
                status_code=507,
                detail=(
                    f"Only {available / GIB:.1f} GiB of host RAM is available; loading a model "
                    f"needs at least {MIN_FREE_RAM_GIB:g} GiB free or the desktop can lock up. "
                    "Close other programs (or lower VLLM_LAUNCHER_MIN_FREE_RAM_GIB) and retry."
                ),
            )
        with self.lock:
            if self.process and self.process.poll() is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"A {engine} process is already running",
                )
            self.adopted = False
            self.events.clear()  # sequence is deliberately *not* reset: see EventLog.clear
            self._append_locked(f"$ {shlex.join(argv)}")
            visible = env.get("CUDA_VISIBLE_DEVICES", "all")
            self._append_locked(
                f"{engine.upper()}  CUDA_VISIBLE_DEVICES={visible}  HF_HOME={env.get('HF_HOME')}"
                f"  CUDA_HOME={env.get('CUDA_HOME', '-')}  MAX_JOBS={env.get('MAX_JOBS')}"
            )
            for note in notes:
                self._append_locked(f"launcher: {note}")
            try:
                process = subprocess.Popen(
                    argv,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                self._append_locked(f"Failed to start {engine}: {exc}")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            self.process = process
            self.spec = spec
            self.command = argv
            self.started_at = time.time()
            self._append_locked(f"Started {engine} pid={process.pid}")
        threading.Thread(target=self._pump, args=(process,), daemon=True).start()
        threading.Thread(target=self._ram_watchdog, args=(process, engine), daemon=True).start()
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
                    engine = self.spec.engine if self.spec else "engine"
                    if code == -signal.SIGKILL:
                        self._append_locked(
                            f"{engine} was SIGKILLed (exit {code}) - by the RAM watchdog, Stop, "
                            "or the kernel OOM killer."
                        )
                    else:
                        self._append_locked(f"{engine} exited with code {code}")

    def _ram_watchdog(self, process: subprocess.Popen, engine: str):
        """Kill the engine's whole process group before host RAM runs out. SIGKILL rather than
        SIGTERM: a thrashing box cannot afford a graceful shutdown, and the killed processes'
        anonymous memory is released immediately."""
        psi_strikes = 0
        while process.poll() is None:
            time.sleep(0.5)
            ram = ram_snapshot()
            available, psi = ram["available_bytes"], ram["psi_full_avg10"]
            reason = None
            if available is not None and available < RAM_KILL_GIB * GIB:
                reason = f"host RAM critically low ({available / GIB:.2f} GiB available)"
            elif psi is not None and psi >= RAM_PSI_FULL_KILL:
                psi_strikes += 1
                if psi_strikes >= 3:
                    reason = f"host stalled on memory ({psi:.0f}% of the last 10 s, PSI full avg10)"
            else:
                psi_strikes = 0
            if reason is None or process.poll() is not None:
                continue
            with self.lock:
                if process is not self.process:
                    return
                self._append_locked(
                    f"RAM WATCHDOG: {reason}; SIGKILLing the {engine} process group to keep the "
                    "host alive. If this was kernel JIT, retry - finished objects are cached and "
                    "MAX_JOBS is sized from free RAM at each launch."
                )
            _kill_group(process, signal.SIGKILL)
            return

    def stop(self) -> dict:
        with self.lock:
            engine = self.spec.engine if self.spec else "engine"
            process = self.process
            owned = bool(process and process.poll() is None)
            if owned:
                self._append_locked(f"Stopping {engine} pid={process.pid} (SIGTERM)...")
        if not owned:
            current = self.status()
            if current.get("external"):
                self.log(
                    f"The server on port {current['port']} was started outside this launcher; "
                    f"stop it where it was started ({EXTERNAL_NAME})."
                )
            else:
                self.log("No server process is running.")
            return current
        try:
            pgid = os.getpgid(process.pid)
        except (OSError, ProcessLookupError):
            pgid = None
        _kill_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.log("Still alive after SIGTERM; sending SIGKILL...")
            _kill_group(process, signal.SIGKILL)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.log("Main process did not exit after SIGKILL.")
        # The main pid exiting is not the end: SGLang's scheduler/detokenizer workers and
        # vLLM's engine-core processes live in the same group and are what hold GPU memory.
        if pgid is not None and not _wait_group_gone(pgid, timeout=20):
            self.log("Worker processes are still alive; SIGKILLing the process group...")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            if not _wait_group_gone(pgid, timeout=10):
                self.log("Some worker processes could not be killed; GPU memory may still be held.")
        return self.status()

    # -- status -------------------------------------------------------------------
    def _status_locked(self) -> dict:
        process = self.process
        running = bool(process and process.poll() is None)
        return {
            "running": running,
            "engine": self.spec.engine if self.spec else None,
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

        if snapshot["running"]:
            snapshot["endpoint"] = self._probe(port, key) if port else {"online": False}
            snapshot["owned"] = True
            snapshot["external"] = False
            return snapshot

        # Nothing of ours is up, so adopt whatever else is serving the shared port. This keeps
        # the status pill, chat proxy and delete guard honest about a model another manager
        # launched.
        endpoint = self._probe(EXTERNAL_PORT, EXTERNAL_API_KEY)
        snapshot["endpoint"] = endpoint
        snapshot["owned"] = False
        snapshot["external"] = endpoint["online"]
        if endpoint["online"]:
            snapshot.update(
                {
                    "running": True,
                    "engine": None,  # unknown: any OpenAI-compatible server
                    "pid": None,
                    "returncode": None,
                    "uptime": None,
                    "port": EXTERNAL_PORT,
                    "model": (endpoint["models"] or [None])[0],
                    "command": [],
                }
            )

        with self.lock:
            if endpoint["online"] and not self.adopted:
                self.adopted = True
                self._append_locked(
                    f"Adopted an external OpenAI-compatible server on port {EXTERNAL_PORT} serving "
                    f"{', '.join(endpoint['models']) or 'an unknown model'}. It was started "
                    "outside this launcher, so no log output is available here and Stop is "
                    f"disabled. Manage it where it was started ({EXTERNAL_NAME})."
                )
            elif not endpoint["online"]:
                self.adopted = False
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


def _kill_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (OSError, ProcessLookupError):
        # Group already gone (or the leader was reaped): signal the pid as a fallback.
        try:
            process.send_signal(sig)
        except (OSError, ProcessLookupError):
            pass


def _wait_group_gone(pgid: int, timeout: float) -> bool:
    """True once no process in the group exists (signal 0 probes without delivering)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass  # exists but not ours: treat as alive
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


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

            argv = [*_hf_cli(), "download", repo]
            if revision:
                if not HF_REVISION_RE.match(revision):
                    raise HTTPException(status_code=400, detail="Invalid revision")
                argv += ["--revision", revision]
            if include:
                for pattern in shlex.split(include):
                    argv += ["--include", pattern]

            env = os.environ.copy()
            env.setdefault("HF_HOME", HF_HOME)
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
                    encoding="utf-8",
                    errors="replace",
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
        _kill_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _kill_group(process, signal.SIGKILL)
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


def _upstream_error(status_code: int, body: bytes) -> str:
    """Engines answer errors as JSON `{"error": {"message": ...}}` (vLLM) or `{"detail": ...}`
    / a bare `{"message": ...}` (SGLang); fall back to the raw text."""
    text = body.decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:400] or f"upstream returned HTTP {status_code}"
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err.get("detail") or json.dumps(err)[:400])
        return str(data.get("detail") or data.get("message") or err)[:400]
    return text[:400]


_background_tasks: set = set()


async def _abort_upstream(port: int, headers: dict[str, str], rid: str) -> None:
    """SGLang keeps generating after the client hangs up; ask it to drop the request."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(f"http://127.0.0.1:{port}/abort_request", json={"rid": rid}, headers=headers)
    except httpx.HTTPError:
        pass


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    # status() probes the engine over HTTP; run it off the event loop so the SSE log
    # streams and other requests keep flowing while it waits.
    status = await asyncio.to_thread(runtime.status)
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
        owned_key = runtime.spec.api_key if runtime.spec else None
        engine = runtime.spec.engine if runtime.spec else None
    port = status["port"]
    api_key = owned_key if status.get("owned") else EXTERNAL_API_KEY
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    if not req.stream:
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code if response.status_code >= 400 else 502,
                detail=_upstream_error(response.status_code, response.content),
            )
        try:
            return response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Engine returned a non-JSON response") from exc

    async def relay() -> AsyncIterator[str]:
        rid = ""
        done = False
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        detail = _upstream_error(response.status_code, await response.aread())
                        yield f"data: {json.dumps({'error': detail})}\n\n"
                        return
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not rid and line.startswith("data: "):
                            body = line[6:].strip()
                            if body and body != "[DONE]":
                                try:
                                    rid = json.loads(body).get("id") or ""
                                except (json.JSONDecodeError, AttributeError):
                                    pass
                        if line.strip() == "data: [DONE]":
                            done = True
                        yield f"{line}\n\n"
                        if not done and await request.is_disconnected():
                            break
        except httpx.HTTPError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            # Starlette cancels this generator when the browser disconnects, so an await
            # here would itself be cancelled: hand the abort to a detached task instead.
            # (vLLM aborts on its own once the upstream socket closes.)
            if engine == "sglang" and rid and not done:
                task = asyncio.get_running_loop().create_task(_abort_upstream(port, headers, rid))
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

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
    return {"gpus": gpu_snapshot(), "ram": ram_snapshot()}


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
    # Covers both a model we launched and one adopted from another manager, which reports
    # itself by served name rather than by the id discovery assigned it.
    loaded = set()
    if active["running"]:
        loaded.add(active.get("model"))
        loaded.update(active.get("endpoint", {}).get("models") or [])
    loaded.discard(None)
    deleted, freed = [], 0

    for model_id in requested:
        entry = known.get(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        if model_id in loaded or entry["path"] in loaded:
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
    argv, env, notes = build_command(spec)
    shown = ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE", "CUDA_HOME", "MAX_JOBS")
    overrides = {k: env[k] for k in shown if k in env}
    return {"command": shlex.join(argv), "argv": argv, "env": overrides, "notes": notes}


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
    # One file per model holds every engine's saved config; `engine` lives inside
    # so the UI can show "saved for this engine" vs. "saved for the other one".
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:24]
    return PROFILE_DIR / f"{digest}.json"


def _load_profile_file(model_id: str) -> dict:
    """Return the profile file for `model_id` in the *new* shape
    ``{"model", "saved_at", "engine_profiles": {engine: {saved_at, spec}}}``.
    Legacy files (a top-level ``spec`` from the vLLM-only era) are migrated in-memory
    into a single-vllm entry so old data is never lost."""
    data = _read_json(_profile_path(model_id))
    if not data.get("model"):
        return {}
    if isinstance(data.get("engine_profiles"), dict) and data["engine_profiles"]:
        return data
    if not data.get("spec"):
        return {}
    return {
        "model": data["model"],
        "saved_at": data.get("saved_at"),
        "engine_profiles": {
            "vllm": {"saved_at": data.get("saved_at"), "spec": data["spec"]},
        },
    }


def _save_profile(spec: LaunchSpec) -> None:
    engine = _coerce_engine(spec.engine)
    existing = _load_profile_file(spec.model)
    existing.setdefault("engine_profiles", {})
    existing["engine_profiles"][engine] = {
        "saved_at": time.time(),
        "spec": spec.model_dump(),
    }
    existing["model"] = spec.model
    existing["saved_at"] = time.time()
    _profile_path(spec.model).write_text(json.dumps(existing, indent=2), encoding="utf-8")


@app.get("/api/profiles")
def api_profiles():
    """One entry per (model, engine) — keyed as ``{model: {engine: {saved_at, spec}}}``.
    Legacy files are read through the same helper, so a user upgrading from the vLLM-only
    release sees the existing vLLM profile intact without touching the file."""
    profiles = {}
    for path in PROFILE_DIR.glob("*.json"):
        data = _read_json(path)
        model = data.get("model")
        if not model:
            continue
        if isinstance(data.get("engine_profiles"), dict) and data["engine_profiles"]:
            profiles[model] = data["engine_profiles"]
        elif data.get("spec"):
            profiles[model] = {
                "vllm": {"saved_at": data.get("saved_at"), "spec": data["spec"]}
            }
    return {"profiles": profiles}


@app.put("/api/profiles")
def api_save_profile(spec: LaunchSpec):
    _save_profile(spec)
    return {"saved": spec.model, "engine": _coerce_engine(spec.engine)}


@app.delete("/api/profiles")
def api_delete_profile(model: str, engine: str | None = None):
    """Removing a whole profile (``engine`` omitted) unlinks the file; giving an
    engine drops just that engine's slot and leaves the file (and other engines) alone."""
    if engine is not None:
        try:
            engine = _coerce_engine(engine)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = _profile_path(model)
    if not path.exists():
        return {"deleted": model, "engine": engine, "removed": False}
    if engine is None:
        path.unlink()
        return {"deleted": model, "removed": True}
    data = _load_profile_file(model)
    if not data.get("engine_profiles"):
        path.unlink()
        return {"deleted": model, "removed": True}
    data["engine_profiles"].pop(engine, None)
    if data["engine_profiles"]:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        path.unlink()
    return {"deleted": model, "engine": engine, "removed": True}


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
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


class RevalidatingStaticFiles(StaticFiles):
    """Without an explicit Cache-Control, browsers apply heuristic freshness and can
    serve a stale app.js for hours after an update -- fatal when the JS and the API
    shape change together. "no-cache" still allows a cheap 304 via the ETag."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", RevalidatingStaticFiles(directory=STATIC_DIR), name="static")


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

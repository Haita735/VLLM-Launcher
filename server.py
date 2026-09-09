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
    # `pip install -e` leaves only a finder .pth in site-packages; the package sits elsewhere.
    return any(
        (site / module).is_dir() or any(site.glob(f"__editable__.{module}-*.pth"))
        for site in _site_packages(python)
    )


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

# An OpenAI-compatible server on one of these ports may be owned by another manager. The
# launcher adopts it read-only rather than competing for the GPUs; EXTERNAL_NAME is only
# used in messages that tell the user where to go to stop it. Defaults cover vLLM's (8000)
# and SGLang's (30000) stock ports.
EXTERNAL_PORTS = [
    int(p) for p in os.environ.get("VLLM_LAUNCHER_EXTERNAL_PORT", "8000,30000").split(",") if p.strip()
] or [8000]
EXTERNAL_PORT = EXTERNAL_PORTS[0]
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
#      the RAM that is actually free *and* from the cgroup limit the launcher runs under
#      (systemd MemoryMax/MemoryHigh, Docker --memory: the engine inherits it), minus what
#      the engine's own processes need, never raising an inherited value;
#   3. a watchdog SIGKILLs the engine's process group if MemAvailable drops below
#      RAM_KILL_GIB or the kernel reports the system fully stalled on memory (PSI).
MIN_FREE_RAM_GIB = _env_float("VLLM_LAUNCHER_MIN_FREE_RAM_GIB", 4.0)
RAM_KILL_GIB = _env_float("VLLM_LAUNCHER_RAM_KILL_GIB", 2.0)
# PSI `full avg10` (% of the last 10 s the whole host was stalled on memory) that trips the
# watchdog. With a discrete GPU the weights go to VRAM and host memory stays quiet, so 25 % is
# already a thrashing desktop. On unified-memory parts (GB10 / DGX Spark) the model *is* host
# memory: faulting 100+ GiB of weights in stalls the box 25-30 % of the time while it is
# perfectly healthy, so the default moves up and the MemAvailable floor stays the real OOM
# backstop. VLLM_LAUNCHER_RAM_PSI_FULL overrides either; see ram_psi_full_kill().
RAM_PSI_FULL_KILL_DISCRETE = 25.0
RAM_PSI_FULL_KILL_UNIFIED = 80.0
JIT_RAM_PER_JOB_GIB = _env_float("VLLM_LAUNCHER_JIT_RAM_PER_JOB_GIB", 6.0)
# Measured: SGLang's scheduler + tokenizer + detokenizer sit at ~6 GiB of anonymous memory
# while a model loads; vLLM's engine core is similar. The JIT budget is what is left after that.
ENGINE_RAM_RESERVE_GIB = _env_float("VLLM_LAUNCHER_ENGINE_RAM_RESERVE_GIB", 6.0)
JIT_MAX_JOBS_CAP = max(1, int(_env_float("VLLM_LAUNCHER_JIT_MAX_JOBS", 4)))
GIB = 1024**3
# Launcher-owned state that must survive restarts: on-disk engine logs (a host freeze wipes
# the in-memory buffer, and that is exactly when the log matters) and toolchain shims.
STATE_DIR = Path(os.environ.get("VLLM_LAUNCHER_STATE_DIR", "~/.cache/vllm-launcher")).expanduser()
LOG_DIR = STATE_DIR / "logs"
LOG_FILES_KEPT = max(1, int(_env_float("VLLM_LAUNCHER_LOG_FILES_KEPT", 10)))


def _cgroup_memory() -> dict:
    """Memory limits that apply to *this* process through cgroup v2, walking up from the
    leaf so a cap on a parent slice (or a container's root) is seen too. Values are the
    tightest limit found; None means unlimited/unavailable."""
    info: dict[str, Any] = {
        "path": None, "max_bytes": None, "high_bytes": None,
        "current_bytes": None, "anon_bytes": None, "oom_kills": None, "psi_full_avg10": None,
    }
    try:
        with open("/proc/self/cgroup", encoding="ascii") as handle:
            line = next((l for l in handle if l.startswith("0::")), None)
    except OSError:
        return info
    if not line:
        return info  # cgroup v1: no single memory limit to read
    rel = line.strip()[3:]
    leaf = Path("/sys/fs/cgroup" + rel)
    info["path"] = rel or "/"

    def read_int(path: Path) -> int | None:
        try:
            text = path.read_text(encoding="ascii").strip()
        except OSError:
            return None
        return None if text == "max" else int(text.split()[0])

    info["current_bytes"] = read_int(leaf / "memory.current")
    # memory.current counts page cache (mmap'd weights) that the kernel reclaims on demand;
    # anon is what actually has to fit under the limit.
    try:
        for entry in (leaf / "memory.stat").read_text(encoding="ascii").splitlines():
            if entry.startswith("anon "):
                info["anon_bytes"] = int(entry.split()[1])
                break
    except (OSError, ValueError, IndexError):
        pass
    node = leaf
    while str(node).startswith("/sys/fs/cgroup"):
        for key, name in (("max_bytes", "memory.max"), ("high_bytes", "memory.high")):
            value = read_int(node / name)
            if value is not None and (info[key] is None or value < info[key]):
                info[key] = value
        node = node.parent
    try:
        for entry in (leaf / "memory.events").read_text(encoding="ascii").splitlines():
            if entry.startswith("oom_kill "):
                info["oom_kills"] = int(entry.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    try:
        for entry in (leaf / "memory.pressure").read_text(encoding="ascii").splitlines():
            if entry.startswith("full"):
                info["psi_full_avg10"] = float(entry.split("avg10=")[1].split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return info


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
    info["cgroup"] = _cgroup_memory()
    return info


def ram_budget(ram: dict) -> tuple[int | None, str]:
    """Bytes the engine may still allocate before something breaks, and where that bound
    comes from: host MemAvailable, or the launcher's cgroup limit minus what the cgroup
    already holds (the kernel OOM-kills inside the cgroup at memory.max and throttles the
    engine to a crawl above memory.high, long before the host itself is short of RAM)."""
    candidates: list[tuple[int, str]] = []
    if ram.get("available_bytes") is not None:
        candidates.append((ram["available_bytes"], "host MemAvailable"))
    cg = ram.get("cgroup") or {}
    used = cg.get("anon_bytes")
    if used is None:
        used = cg.get("current_bytes") or 0
    for key, label in (("high_bytes", "cgroup memory.high"), ("max_bytes", "cgroup memory.max")):
        limit = cg.get(key)
        if limit is not None:
            candidates.append((max(0, limit - used), label))
    if not candidates:
        return None, "unknown"
    return min(candidates)


def top_ram_consumers(limit: int = 3, min_bytes: int = GIB) -> list[str]:
    """Largest anonymous-memory users on the host, so the launch notes can say *what* is
    occupying the RAM the engine will not get (browsers and IDEs routinely hold 5-10 GiB)."""
    rows: list[tuple[int, str]] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry.name}/status", encoding="ascii", errors="replace") as handle:
                name, anon = "", 0
                for line in handle:
                    if line.startswith("Name:"):
                        name = line.split(None, 1)[1].strip()
                    elif line.startswith("RssAnon:"):
                        anon = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            continue
        if anon >= min_bytes:
            rows.append((anon, name))
    rows.sort(reverse=True)
    return [f"{name} {anon / GIB:.1f} GiB" for anon, name in rows[:limit]]


def jit_jobs_for(ram: dict, inherited: str | None) -> tuple[int, str]:
    """Parallel nvcc jobs that fit in the RAM budget after the engine's own processes are
    accounted for. An inherited MAX_JOBS is only ever lowered. Returns (jobs, reason)."""
    cap = JIT_MAX_JOBS_CAP
    if inherited and inherited.isdigit() and int(inherited) > 0:
        cap = min(cap, int(inherited))
    budget, source = ram_budget(ram)
    if budget is None:
        return 1, "RAM budget unknown"
    free_for_jit = budget / GIB - ENGINE_RAM_RESERVE_GIB
    fits = int(free_for_jit // JIT_RAM_PER_JOB_GIB)
    jobs = max(1, min(cap, fits))
    return jobs, (
        f"{source} {budget / GIB:.1f} GiB - {ENGINE_RAM_RESERVE_GIB:g} GiB engine reserve "
        f"= {max(0.0, free_for_jit):.1f} GiB for JIT at ~{JIT_RAM_PER_JOB_GIB:g} GiB per nvcc job"
    )

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
# Grace Blackwell superchips (GB10: DGX Spark, ASUS GX10, ...) have no dedicated VRAM - CPU and
# GPU share one LPDDR5X pool - and nvidia-smi prints "[N/A]" for every memory field.
_UNIFIED_MEMORY_GPU_RE = re.compile(r"\bGB10\b")


def _smi_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None  # nvidia-smi's "[N/A]"


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
        total = _smi_int(row[2])
        gpus.append(
            {
                "index": int(row[0]),
                "name": row[1],
                "memory_total_mb": total or 0,
                "memory_used_mb": _smi_int(row[3]) or 0,
                "utilization": _smi_int(row[4]) or 0,
                "temperature": _smi_int(row[5]) or 0,
                "compute_cap": row[6],
                # The model lives in host RAM on these parts: the RAM card and guards apply.
                "unified_memory": total is None or bool(_UNIFIED_MEMORY_GPU_RE.search(row[1])),
            }
        )
    return gpus


_unified_memory_host: bool | None = None


def unified_memory_host() -> bool:
    """True when the GPUs share the host's memory (GB10 class); probed once."""
    global _unified_memory_host
    if _unified_memory_host is None:
        gpus = gpu_snapshot()
        _unified_memory_host = bool(gpus) and all(g["unified_memory"] for g in gpus)
    return _unified_memory_host


def ram_psi_full_kill() -> float:
    default = RAM_PSI_FULL_KILL_UNIFIED if unified_memory_host() else RAM_PSI_FULL_KILL_DISCRETE
    return _env_float("VLLM_LAUNCHER_RAM_PSI_FULL", default)


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
        caps = [g["compute_cap"] for g in gpus if re.fullmatch(r"\d+\.\d+", g["compute_cap"])]
        cap_major_minor = min((tuple(int(x) for x in c.split(".")) for c in caps), default=(0, 0))
        _system_cache.update(
            {
                "versions": versions,
                "engines": engines,
                "model_roots": [str(p) for p in MODEL_ROOTS],
                "capability": f"{cap_major_minor[0]}.{cap_major_minor[1]}" if caps else None,
                "capability_int": cap_major_minor[0] * 10 + cap_major_minor[1],
                "gpu_count": len(gpus),
                "unified_memory": unified_memory_host(),
                "external_port": EXTERNAL_PORT,
                "hf_home": HF_HOME,
                "hf_cli": shlex.join(_hf_cli()),
                "log_dir": str(LOG_DIR),
                "ram_guard": {
                    "min_free_gib": MIN_FREE_RAM_GIB,
                    "kill_gib": RAM_KILL_GIB,
                    "psi_full_kill": ram_psi_full_kill(),
                    "jit_ram_per_job_gib": JIT_RAM_PER_JOB_GIB,
                    "engine_reserve_gib": ENGINE_RAM_RESERVE_GIB,
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
                    "text": "No native FP4 tensor cores on this GPU. vLLM: runs NVFP4 weight-only "
                    "(W4A16) through Marlin - pin it with linear-backend=marlin. SGLang: check that "
                    "its NVFP4 kernels support this compute capability before loading.",
                }
            )
    if "fp8" in detail and cap and cap < 89:
        notes.append(
            {
                "level": "info",
                "text": "FP8 weights are dequantised on this GPU (no FP8 tensor cores); compute stays fp16.",
            }
        )
    if quant.get("kv_cache") and "8" in str(quant.get("kv_cache")) and cap and cap < 89:
        notes.append(
            {
                "level": "warn",
                "text": "Checkpoint ships fp8 KV scales, so kv-cache-dtype=auto resolves to fp8, "
                "which needs SM89+. Set kv-cache-dtype explicitly (vLLM: float16; SGLang: bf16).",
            }
        )
    if entry.get("multimodal"):
        notes.append({"level": "info", "text": "Multimodal checkpoint; vLLM: limit-mm-per-prompt applies, SGLang: enable multimodal in its section."})
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


_nvcc_version_cache: dict[str, str | None] = {}


def _nvcc_version(cuda_root: Path) -> str | None:
    """`major.minor` of the toolkit's nvcc, probed once per toolkit."""
    key = str(cuda_root)
    if key not in _nvcc_version_cache:
        version = None
        try:
            out = subprocess.run(
                [str(cuda_root / "bin" / "nvcc"), "--version"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout
            match = re.search(r"release (\d+)\.(\d+)", out)
            version = f"{match.group(1)}.{match.group(2)}" if match else None
        except (OSError, subprocess.SubprocessError):
            pass
        _nvcc_version_cache[key] = version
    return _nvcc_version_cache[key]


def _cudart_header_version(cuda_root: Path) -> str | None:
    """`major.minor` of the CUDA runtime headers (CUDART_VERSION 13000 -> 13.0)."""
    try:
        text = (cuda_root / "include" / "cuda_runtime_api.h").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"#define\s+CUDART_VERSION\s+(\d+)", text)
    if not match:
        return None
    value = int(match.group(1))
    return f"{value // 1000}.{(value % 1000) // 10}"


def _dist_version(python: Path, project: str) -> str | None:
    """Installed version of a pip project in the engine's env, from its .dist-info name."""
    wanted = re.sub(r"[-_.]+", "_", project).lower()
    for site in _site_packages(python):
        for info in site.glob("*.dist-info"):
            name, _, version = info.name[: -len(".dist-info")].partition("-")
            if re.sub(r"[-_.]+", "_", name).lower() == wanted and version:
                return version
    return None


def _flashinfer_version_mismatch(engine: Engine) -> str | None:
    """FlashInfer refuses to start when its companion wheels (`flashinfer-cubin`,
    `flashinfer-jit-cache`) are not the exact same release - which happens whenever an
    engine pins a flashinfer-python version whose companions were never published. Mirrors
    FlashInfer's own check so the bypass is only set when it would otherwise fail."""
    if not engine.python:
        return None
    core = _dist_version(engine.python, "flashinfer-python") or _dist_version(engine.python, "flashinfer")
    if not core:
        return None
    cubin = _dist_version(engine.python, "flashinfer-cubin")
    if cubin and cubin != core:
        return f"flashinfer-python {core} vs flashinfer-cubin {cubin}"
    jit_cache = _dist_version(engine.python, "flashinfer-jit-cache")
    if jit_cache and not jit_cache.startswith(core):
        return f"flashinfer-python {core} vs flashinfer-jit-cache {jit_cache}"
    return None


def _jit_toolchain(cuda_root: Path, engine: Engine, env: dict[str, str]) -> list[str]:
    """Make kernel JIT (FlashInfer, tvm-ffi / sgl-kernel, torch cpp_extension) *link* against
    a pip-installed CUDA toolkit without hand-patching site-packages.

    The `nvidia-cuda-*` wheels lay the toolkit out as `nvidia/cu13/lib/` holding only
    versioned sonames (`libcudart.so.13`), while the JIT builders emit
    `-L$CUDA_HOME/lib64 -lcudart -lcuda` as if it were a system CUDA install. The result
    is `ld: cannot find -lcudart` on the first kernel that needs compiling - hours into a
    load, after the weights are already on the GPU. A launcher-owned directory of
    `libX.so -> <toolkit>/lib/libX.so.N` symlinks placed on LIBRARY_PATH (honoured by gcc
    and clang for every `-l` lookup) and in FLASHINFER_EXTRA_LDFLAGS makes every builder
    resolve them; `libcuda.so` itself comes from the driver package on the system.
    Notes describe what was set."""
    notes: list[str] = []
    lib_dir = cuda_root / "lib"
    if not lib_dir.is_dir():
        return notes
    shim_dir = STATE_DIR / "toolchain" / engine.name / "lib"
    linked: list[str] = []
    try:
        shim_dir.mkdir(parents=True, exist_ok=True)
        for so in sorted(lib_dir.glob("lib*.so.*")):
            # libcudart.so.13 -> libcudart.so; skip libnvrtc-builtins.so.13.0 style names that
            # are never linked by name, and anything the toolkit already ships unversioned.
            match = re.match(r"^(lib[A-Za-z0-9_]+)\.so\.\d+$", so.name)
            if not match or (lib_dir / f"{match.group(1)}.so").exists():
                continue
            link = shim_dir / f"{match.group(1)}.so"
            if link.is_symlink() and link.resolve() == so.resolve():
                linked.append(link.name)
                continue
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(so)
            linked.append(link.name)
    except OSError as exc:
        notes.append(f"could not prepare JIT linker shims in {shim_dir}: {exc}")
        return notes

    search = [str(shim_dir), str(lib_dir)]
    stubs = lib_dir / "stubs"
    if stubs.is_dir():
        search.append(str(stubs))
    inherited = [p for p in env.get("LIBRARY_PATH", "").split(os.pathsep) if p and p not in search]
    env["LIBRARY_PATH"] = os.pathsep.join(search + inherited)
    ldflags = shlex.split(env.get("FLASHINFER_EXTRA_LDFLAGS", ""))
    for path in search:
        if f"-L{path}" not in ldflags:
            ldflags.append(f"-L{path}")
    env["FLASHINFER_EXTRA_LDFLAGS"] = shlex.join(ldflags)
    if linked:
        shown = ", ".join(linked[:3]) + (f" +{len(linked) - 3} more" if len(linked) > 3 else "")
        notes.append(
            f"pip CUDA toolkit has no unversioned .so names for the linker; {len(linked)} shims "
            f"({shown}) in {shim_dir} via LIBRARY_PATH / FLASHINFER_EXTRA_LDFLAGS."
        )

    # The wheels are versioned independently: nvcc 13.3 next to 13.0 runtime headers is
    # normal, and CCCL's compatibility guard aborts the compile on that mismatch.
    nvcc = _nvcc_version(cuda_root)
    headers = _cudart_header_version(cuda_root)
    define = "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
    if nvcc and headers and nvcc != headers:
        # FlashInfer reads its own variable; NVCC_PREPEND_FLAGS is read by nvcc itself, so the
        # define also reaches TileLang, tvm-ffi and torch cpp_extension builds.
        for key in ("FLASHINFER_EXTRA_CUDAFLAGS", "NVCC_PREPEND_FLAGS"):
            flags = shlex.split(env.get(key, ""))
            if define not in flags:
                env[key] = shlex.join(flags + [define])
        notes.append(
            f"nvcc {nvcc} vs CUDA runtime headers {headers}: added {define} for JIT "
            "(FLASHINFER_EXTRA_CUDAFLAGS, NVCC_PREPEND_FLAGS)."
        )
    return notes


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
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    mismatch = _flashinfer_version_mismatch(engine)
    if mismatch and "FLASHINFER_DISABLE_VERSION_CHECK" not in env:
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        notes.append(f"{mismatch}: set FLASHINFER_DISABLE_VERSION_CHECK=1 so FlashInfer starts anyway.")

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
        notes += _jit_toolchain(cuda_root, engine, env)
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

    # Kernel JIT parallelism sized to the RAM budget *now* (see the guard notes at the top
    # of the file). Honoured by FlashInfer, SGLang's kernel JIT and tvm-ffi.
    ram = ram_snapshot()
    jobs, reason = jit_jobs_for(ram, env.get("MAX_JOBS"))
    env["MAX_JOBS"] = str(jobs)
    env.setdefault("FLASHINFER_NVCC_THREADS", "1")
    notes.append(f"MAX_JOBS={jobs}: {reason}.")
    expected = ENGINE_RAM_RESERVE_GIB + jobs * JIT_RAM_PER_JOB_GIB
    budget, _ = ram_budget(ram)
    if budget is not None and expected * GIB > budget:
        notes.append(
            f"WARNING: if this start has to JIT-compile kernels, host RAM may peak at ~{expected:g} GiB "
            f"(engine + {jobs} nvcc job) against a {budget / GIB:.1f} GiB budget - the compile could be "
            "throttled or OOM-killed. Starts with cached kernels are unaffected; close other programs "
            "or raise the cap for a first start."
        )
    cg = ram.get("cgroup") or {}
    if cg.get("max_bytes") is not None or cg.get("high_bytes") is not None:
        limits = ", ".join(
            f"{label} {cg[key] / GIB:.0f} GiB"
            for key, label in (("high_bytes", "high"), ("max_bytes", "max")) if cg.get(key) is not None
        )
        used = cg.get("anon_bytes") if cg.get("anon_bytes") is not None else (cg.get("current_bytes") or 0)
        notes.append(
            f"launcher cgroup ({limits}; {used / GIB:.1f} GiB anon in use) - "
            "the engine inherits this cap: the kernel throttles above high and OOM-kills at max."
        )
    consumers = top_ram_consumers()
    if consumers:
        notes.append("largest other RAM users: " + ", ".join(consumers) + ".")
    notes.append(
        f"watchdog kills the engine below {RAM_KILL_GIB:g} GiB host RAM available or when the host "
        f"is stalled on memory >= {ram_psi_full_kill():g}% of the time"
        + (" (unified memory: the model is host RAM, so the stall trip point is raised)."
           if unified_memory_host() else ".")
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
# Kernel JIT progress
# --------------------------------------------------------------------------------------
# FlashInfer, sgl-kernel and tvm-ffi compile kernels through `ninja -C <dir>` with the output
# captured, and the engine's own progress bar blocks on the result, so a first start looks
# hung for as long as the compile runs (18 min for one CUTLASS FP4 op at MAX_JOBS=1). The
# launcher watches the engine's process group for the build and reports it instead.
_COMPILER_COMMS = {"cicc", "ptxas", "nvcc", "cudafe++", "fatbinary", "nvlink", "cc1plus", "ld", "collect2"}
_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _boot_time() -> float:
    try:
        with open("/proc/stat", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _session_processes(sid: int) -> list[dict]:
    """pid, comm, argv, cwd and start time (epoch) for every live process in the session.
    The engine is started with start_new_session=True, so its session id is its pid and
    everything it spawns inherits it - including ninja's compile jobs, which ninja moves
    into process groups of their own (a plain pgid match misses every `cicc`)."""
    found = []
    boot = _boot_time()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/stat", encoding="ascii", errors="replace") as handle:
                stat = handle.read()
            # comm can contain spaces or parentheses: split on the *last* ')'.
            close = stat.rindex(")")
            fields = stat[close + 2:].split()  # state ppid pgrp session ... starttime@[19]
            if int(fields[3]) != sid:
                continue
            with open(f"/proc/{entry.name}/cmdline", "rb") as handle:
                argv = [a.decode("utf-8", "replace") for a in handle.read().split(b"\0") if a]
            found.append({
                "pid": int(entry.name),
                "comm": stat[stat.index("(") + 1:close],
                "argv": argv,
                "cwd": os.readlink(f"/proc/{entry.name}/cwd"),
                "started": boot + int(fields[19]) / _CLK_TCK,
            })
        except (OSError, ValueError, IndexError):
            continue
    return found


def jit_progress(sid: int) -> dict | None:
    """What the engine's ninja build is doing right now, or None when none is running.
    Progress comes from ninja's own bookkeeping: targets in build.ninja versus .ninja_log
    entries whose output mtime is newer than the ninja process, i.e. finished by *this* run."""
    procs = _session_processes(sid)
    ninjas = sorted((p for p in procs if p["comm"] == "ninja"), key=lambda p: p["started"])
    if not ninjas:
        return None
    ninja = ninjas[0]
    argv = ninja["argv"]
    build_dir = Path(argv[argv.index("-C") + 1]) if "-C" in argv[:-1] else Path(ninja["cwd"])
    if not build_dir.is_absolute():
        build_dir = Path(ninja["cwd"]) / build_dir
    name = build_dir.parent.name if build_dir.name.startswith("build-") else build_dir.name

    total = 0
    try:
        with open(build_dir / "build.ninja", encoding="utf-8", errors="replace") as handle:
            total = sum(1 for line in handle if line.startswith("build "))
    except OSError:
        pass
    done, durations = 0, []
    try:
        with open(build_dir / ".ninja_log", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                start_ms, end_ms, mtime = line.split("\t")[:3]
                mtime_s = int(mtime) / 1e9 if int(mtime) > 1e12 else int(mtime)  # ns (log v5+) or s
                if mtime_s >= ninja["started"] - 1:
                    done += 1
                    durations.append((int(end_ms) - int(start_ms)) / 1000)
    except (OSError, ValueError):
        pass
    return {
        "op": name,
        "dir": str(build_dir),
        "done": done,
        "total": total,
        "compilers": sum(1 for p in procs if p["comm"] in _COMPILER_COMMS),
        "avg_step_s": round(sum(durations) / len(durations), 1) if durations else None,
        "elapsed_s": round(time.time() - ninja["started"]),
    }


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
    log_file: Path | None = None
    _log_handle: Any = None
    _kill_reason: str | None = None  # set by the watchdog / Stop before they SIGKILL
    _oom_kills_at_start: int | None = None
    _max_jobs: int = 1
    _jit: dict | None = None  # current kernel JIT build, for /api/status and the log
    _external_engines: dict = field(default_factory=dict)  # port -> detected engine of an adopted server

    def __post_init__(self):
        self.condition = threading.Condition(self.lock)

    # -- logging ------------------------------------------------------------------
    def _append_locked(self, line: str):
        self.sequence += 1
        line = line.rstrip("\n")
        self.events.append({"sequence": self.sequence, "timestamp": time.time(), "line": line})
        if self._log_handle is not None:
            try:
                self._log_handle.write(line + "\n")
            except OSError:
                self._log_handle = None
        self.condition.notify_all()

    def _note_locked(self, line: str):
        """Launcher-originated line: into the run log *and* stderr, so journald keeps a copy
        of watchdog kills and exit diagnoses even if the host goes down right after."""
        self._append_locked(line)
        print(f"launcher: {line}", file=sys.stderr, flush=True)

    def _open_run_log_locked(self, engine: str) -> None:
        self._close_run_log_locked()
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_file = LOG_DIR / f"{engine}-{stamp}.log"
            self._log_handle = open(self.log_file, "a", encoding="utf-8", buffering=1)
            for stale in sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime)[:-LOG_FILES_KEPT]:
                stale.unlink(missing_ok=True)
        except OSError as exc:
            self.log_file = None
            self._log_handle = None
            self._append_locked(f"launcher: could not open a run log under {LOG_DIR}: {exc}")

    def _close_run_log_locked(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

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
            self._kill_reason = None
            self._jit = None
            self._max_jobs = int(env.get("MAX_JOBS") or 1)
            self._oom_kills_at_start = (ram.get("cgroup") or {}).get("oom_kills")
            self.events.clear()  # sequence is deliberately *not* reset: see EventLog.clear
            self._open_run_log_locked(engine)
            self._append_locked(f"$ {shlex.join(argv)}")
            visible = env.get("CUDA_VISIBLE_DEVICES", "all")
            self._append_locked(
                f"{engine.upper()}  CUDA_VISIBLE_DEVICES={visible}  HF_HOME={env.get('HF_HOME')}"
                f"  CUDA_HOME={env.get('CUDA_HOME', '-')}  MAX_JOBS={env.get('MAX_JOBS')}"
            )
            for note in notes:
                self._append_locked(f"launcher: {note}")
            if self.log_file is not None:
                self._append_locked(f"launcher: this run's output is also written to {self.log_file}")
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
                self._note_locked(f"Failed to start {engine}: {exc}")
                self._close_run_log_locked()
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            self.process = process
            self.spec = spec
            self.command = argv
            self.started_at = time.time()
            self._note_locked(f"Started {engine} pid={process.pid}")
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
                        self._note_locked(f"{engine} exited with SIGKILL: {self._explain_sigkill_locked(engine)}")
                    else:
                        self._note_locked(f"{engine} exited with code {code}")
                    self._close_run_log_locked()

    def _explain_sigkill_locked(self, engine: str) -> str:
        """Exit -9 has four very different causes; the log line has to name the right one."""
        if self._kill_reason:
            return self._kill_reason
        cg = _cgroup_memory()
        before, now = self._oom_kills_at_start, cg.get("oom_kills")
        if before is not None and now is not None and now > before:
            cap = cg.get("max_bytes")
            return (
                f"the kernel OOM killer fired inside the launcher's cgroup ({now - before}x, "
                f"memory.max={cap / GIB:.0f} GiB) - the engine's host RAM outgrew the cap. "
                "Lower MAX_JOBS via the env box, close other programs, or raise the unit's MemoryMax."
                if cap else
                f"the kernel OOM killer fired inside the launcher's cgroup ({now - before}x)."
            )
        if engine == "sglang":
            return (
                "not the launcher. SGLang SIGKILLs its own process tree after a fatal error in a "
                "worker - the traceback above is the actual cause."
            )
        return "not the launcher; killed externally or by the kernel (check `journalctl -k`)."

    def _ram_watchdog(self, process: subprocess.Popen, engine: str):
        """Kill the engine's whole process group before host RAM runs out. SIGKILL rather than
        SIGTERM: a thrashing box cannot afford a graceful shutdown, and the killed processes'
        anonymous memory is released immediately. The same loop reports kernel JIT builds,
        which are what the engine is silently doing whenever RAM climbs during a load."""
        psi_strikes = 0
        psi_kill = ram_psi_full_kill()
        swap_warned = False
        ticks = 0
        while process.poll() is None:
            time.sleep(0.5)
            ticks += 1
            if ticks % 4 == 0:
                self._track_jit(process)
            ram = ram_snapshot()
            available, psi = ram["available_bytes"], ram["psi_full_avg10"]
            swap_total, swap_free = ram["swap_total_bytes"], ram["swap_free_bytes"]
            if (
                not swap_warned and swap_total and swap_free is not None
                and swap_free < 0.1 * swap_total
            ):
                # Not a kill: the kernel has pushed idle programs out to make room, which is
                # exactly what makes the desktop feel frozen. Say why while it is happening.
                swap_warned = True
                with self.lock:
                    if process is self.process:
                        self._note_locked(
                            f"RAM WATCHDOG: swap is {100 * (1 - swap_free / swap_total):.0f}% full "
                            f"({(swap_total - swap_free) / GIB:.1f} GiB); the host is under memory "
                            "pressure and other programs are being swapped out. Largest users: "
                            + (", ".join(top_ram_consumers()) or "n/a") + "."
                        )
            reason = None
            if available is not None and available < RAM_KILL_GIB * GIB:
                reason = f"host RAM critically low ({available / GIB:.2f} GiB available)"
            elif psi is not None and psi >= psi_kill:
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
                self._kill_reason = f"RAM watchdog ({reason})"
                self._note_locked(
                    f"RAM WATCHDOG: {reason}; SIGKILLing the {engine} process group to keep the "
                    "host alive. If this was kernel JIT, retry - finished objects are cached and "
                    "MAX_JOBS is sized from the RAM budget at each launch."
                )
            _kill_group(process, signal.SIGKILL)
            return

    def _track_jit(self, process: subprocess.Popen) -> None:
        try:
            current = jit_progress(process.pid)  # start_new_session=True: session id == pid
        except OSError:
            return
        with self.lock:
            if process is not self.process:
                return
            previous = self._jit
            self._jit = current
            if current is None:
                if previous is not None:
                    self._note_locked(
                        f"kernel JIT finished: {previous['op']} "
                        f"({previous['done']}/{previous['total']} steps, {_fmt_duration(previous['elapsed_s'])}); "
                        "cached for later starts."
                    )
                return
            if previous is None or previous["op"] != current["op"]:
                self._note_locked(
                    f"kernel JIT started: {current['op']} - {current['total']} compile steps, "
                    f"{self._max_jobs} at a time (MAX_JOBS). The engine's own output stays quiet until "
                    "this finishes; each step is cached, so a retry resumes where it stopped."
                )
                return
            if current["done"] != previous["done"] and current["total"]:
                remaining = ""
                if current["avg_step_s"]:
                    left = (current["total"] - current["done"]) * current["avg_step_s"] / max(1, self._max_jobs)
                    remaining = f", about {_fmt_duration(left)} left"
                self._note_locked(
                    f"kernel JIT {current['op']}: {current['done']}/{current['total']} steps"
                    + (f", ~{current['avg_step_s']:.0f} s per step" if current["avg_step_s"] else "")
                    + f", {current['compilers']} compiler process(es) running{remaining}."
                )

    def stop(self) -> dict:
        with self.lock:
            engine = self.spec.engine if self.spec else "engine"
            process = self.process
            owned = bool(process and process.poll() is None)
            if owned:
                self._kill_reason = "Stop requested from the launcher"
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
        sid = process.pid  # start_new_session=True: the engine leads its own session
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
        # vLLM's engine-core processes live in the same session and are what hold GPU memory.
        if not _wait_group_gone(sid, timeout=20):
            self.log("Worker processes are still alive; SIGKILLing the whole session...")
            _kill_group(process, signal.SIGKILL)
            if not _wait_group_gone(sid, timeout=10):
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
            "log_file": str(self.log_file) if self.log_file else None,
            "jit": self._jit if running else None,
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

        # Nothing of ours is up, so adopt whatever else is serving one of the shared ports.
        # This keeps the status pill, chat proxy and delete guard honest about a model
        # another manager launched.
        endpoint, port = {"online": False, "models": [], "error": None}, EXTERNAL_PORT
        for candidate in EXTERNAL_PORTS:
            probed = self._probe(candidate, EXTERNAL_API_KEY)
            if probed["online"]:
                endpoint, port = probed, candidate
                break
        snapshot["endpoint"] = endpoint
        snapshot["owned"] = False
        snapshot["external"] = endpoint["online"]
        if endpoint["online"]:
            snapshot.update(
                {
                    "running": True,
                    "engine": self._external_engine(port, EXTERNAL_API_KEY),
                    "pid": None,
                    "returncode": None,
                    "uptime": None,
                    "port": port,
                    "model": (endpoint["models"] or [None])[0],
                    "command": [],
                }
            )

        with self.lock:
            if endpoint["online"] and not self.adopted:
                self.adopted = True
                self._append_locked(
                    f"Adopted an external {snapshot['engine'] or 'OpenAI-compatible'} server on "
                    f"port {port} serving {', '.join(endpoint['models']) or 'an unknown model'}. "
                    "It was started outside this launcher, so no log output is available here "
                    f"and Stop is disabled. Manage it where it was started ({EXTERNAL_NAME})."
                )
            elif not endpoint["online"]:
                self.adopted = False
                self._external_engines.clear()
        return snapshot

    def _external_engine(self, port: int, api_key: str | None) -> str | None:
        """Which engine an adopted server is, from endpoints only one of them has: vLLM
        answers /version, SGLang answers /get_model_info. Cached while it stays online."""
        if port in self._external_engines:
            return self._external_engines[port]
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        engine = None
        for path, key, name in (("/get_model_info", "model_path", "sglang"), ("/version", "version", "vllm")):
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{port}{path}", headers=headers,
                    timeout=httpx.Timeout(1.5, connect=0.4),
                )
                if response.status_code == 200 and key in response.json():
                    engine = name
                    break
            except (httpx.HTTPError, ValueError):
                continue
        self._external_engines[port] = engine
        return engine

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
    """Signal the engine's whole *session*: the leader's process group plus every process
    that re-grouped itself (ninja does this for each nvcc/cicc job, and those are the 6 GiB
    processes a RAM kill exists to remove)."""
    pids = {p["pid"] for p in _session_processes(process.pid)}
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (OSError, ProcessLookupError):
        pass
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (OSError, ProcessLookupError):
            pass
    if not pids:
        try:
            process.send_signal(sig)
        except (OSError, ProcessLookupError):
            pass


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {(seconds % 3600) // 60:02d} min"


def _wait_group_gone(sid: int, timeout: float) -> bool:
    """True once no process of the engine's session exists (leader, workers, compile jobs)."""
    deadline = time.monotonic() + timeout
    while True:
        alive = _session_processes(sid)
        if not alive:
            try:
                os.killpg(sid, 0)  # session leader's group, in case /proc was unreadable
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
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
    """Ask SGLang to drop a request. Must reach it while the streaming connection for that
    request is still open: on client disconnect SGLang discards the request's state without
    aborting the scheduler, after which an abort for that rid is ignored and decoding runs
    on to max_tokens."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(f"http://127.0.0.1:{port}/abort_request", json={"rid": rid}, headers=headers)
    except httpx.HTTPError:
        pass


async def _pump_upstream(
    url: str, payload: dict, headers: dict[str, str], engine: str | None, port: int,
    queue: "asyncio.Queue[str | None]", stop: asyncio.Event,
) -> None:
    """Read the engine's SSE stream in a task of its own. The browser disconnecting cancels
    the response generator; had that cancellation landed inside the httpx read, httpcore
    would close the upstream socket immediately and SGLang would drop the request *without*
    aborting it. Here the generator only sets `stop`; this task sends the abort while the
    socket is still open, then closes it."""
    rid = ""
    done = False
    stop_wait = asyncio.ensure_future(stop.wait())
    next_line: asyncio.Future | None = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    detail = _upstream_error(response.status_code, await response.aread())
                    await queue.put(f"data: {json.dumps({'error': detail})}\n\n")
                    return
                lines = response.aiter_lines()
                while not done:
                    next_line = asyncio.ensure_future(lines.__anext__())
                    # SGLang can only abort a request it has already named: after Stop, keep
                    # reading (bounded) until the first chunk carries the id.
                    need_rid = stop.is_set() and engine == "sglang" and not rid
                    await asyncio.wait(
                        {next_line} if need_rid else {next_line, stop_wait},
                        timeout=30 if need_rid else None,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not next_line.done():
                        if engine == "sglang" and rid:
                            await _abort_upstream(port, headers, rid)
                        # Cancelling the read closes the socket (httpcore); vLLM aborts on that.
                        next_line.cancel()
                        await asyncio.gather(next_line, return_exceptions=True)
                        next_line = None
                        break
                    try:
                        line = next_line.result()
                    except StopAsyncIteration:
                        break
                    next_line = None
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
                    if not stop.is_set():
                        await queue.put(f"{line}\n\n")
                    elif not done:
                        if engine == "sglang" and rid:
                            await _abort_upstream(port, headers, rid)
                        break  # leaving the context closes the socket
    except httpx.HTTPError as exc:
        await queue.put(f"data: {json.dumps({'error': str(exc)})}\n\n")
    finally:
        for fut in (next_line, stop_wait):
            if fut is not None and not fut.done():
                fut.cancel()
        if next_line is not None:
            await asyncio.gather(next_line, return_exceptions=True)
        await queue.put(None)


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
    if not any(m.get("role") != "system" for m in req.messages):
        raise HTTPException(status_code=400, detail="Chat history has no user message to send")

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
    # status() names the engine for both a process we launched and an adopted external one;
    # the port likewise follows whichever server is actually up (vLLM 8000, SGLang 30000...).
    engine = status.get("engine")
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
        # Starlette cancels this generator when the browser disconnects (Stop). The upstream
        # read lives in _pump_upstream so that cancellation never lands inside httpx: only
        # `stop` is set here, and the pump aborts the engine request before closing.
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stop = asyncio.Event()
        pump = asyncio.get_running_loop().create_task(
            _pump_upstream(url, payload, headers, engine, port, queue, stop)
        )
        _background_tasks.add(pump)
        pump.add_done_callback(_background_tasks.discard)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
                if await request.is_disconnected():
                    break
        finally:
            stop.set()

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
    shown = (
        "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE", "CUDA_HOME", "MAX_JOBS",
        "NVCC_PREPEND_FLAGS", "FLASHINFER_EXTRA_CUDAFLAGS", "FLASHINFER_DISABLE_VERSION_CHECK",
    )
    overrides = {k: env[k] for k in shown if k in env}
    overrides.update(spec.env)
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

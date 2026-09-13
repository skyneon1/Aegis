"""
aegis.hostinfo — what the inference host is actually doing right now.

WHY THIS BELONGS IN THE PACKAGE AND NOT IN THE UI
-------------------------------------------------
When inference is remote, the host is somebody else's problem: you send
tokens, you get tokens, and the only observable is latency. When inference
is LOCAL, the host is the single most important variable in the system, and
it is invisible from inside the OpenAI dialect. A 4GB card that silently
spills 41% of a model to the CPU returns byte-identical responses to one
that did not - three times slower.

So "is the model fully resident on the GPU?" is not a UI garnish. It is
the difference between a 4-second turn and a 25-second turn, and it is
the first thing to check when a run feels slow. Recording it alongside a
transcript is a reproducibility question too: a run that took 90 seconds
because another process was holding VRAM is not evidence about your
prompt.

DESIGN RULES
------------
* Read-only. Nothing here starts, stops, or reconfigures anything.
* Never raises. Observability that can crash the thing it observes is a
  liability; every probe degrades to "unknown" instead. A missing GPU, no
  nvidia-smi, a stopped Ollama - all are normal conditions, not errors.
* No os.environ reads (config.py owns that). The Ollama address is passed
  in from Settings.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

_TIMEOUT_S = 2.0  # a probe that blocks the UI is worse than no probe


@dataclass
class GpuInfo:
    available: bool = False
    name: str = ""
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    utilisation_pct: int = 0
    temperature_c: int = 0
    detail: str = ""          # why it is unavailable, when it is

    @property
    def mem_free_mb(self) -> int:
        return max(0, self.mem_total_mb - self.mem_used_mb)

    @property
    def mem_pct(self) -> float:
        if self.mem_total_mb <= 0:
            return 0.0
        return 100.0 * self.mem_used_mb / self.mem_total_mb


@dataclass
class LoadedModel:
    name: str
    size_mb: int = 0
    vram_mb: int = 0
    context_length: int = 0
    quantisation: str = ""
    parameter_size: str = ""
    expires_at: str = ""

    @property
    def gpu_fraction(self) -> float:
        """
        Share of this model's weights actually sitting in VRAM.

        1.0 means fully GPU-resident. Anything less means Ollama split the
        model, and the CPU-side layers dominate generation time - this is
        the number that explains an unexpectedly slow run. It is derived
        from Ollama's own `size` vs `size_vram`, not estimated.
        """
        if self.size_mb <= 0:
            return 0.0
        return min(1.0, self.vram_mb / self.size_mb)

    @property
    def placement(self) -> str:
        frac = self.gpu_fraction
        if frac >= 0.999:
            return "100% GPU"
        if frac <= 0.001:
            return "100% CPU"
        return f"{round(frac * 100)}% GPU / {round((1 - frac) * 100)}% CPU"


@dataclass
class HostSnapshot:
    gpu: GpuInfo = field(default_factory=GpuInfo)
    loaded: list[LoadedModel] = field(default_factory=list)
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ollama_reachable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": vars(self.gpu),
            "loaded": [vars(m) for m in self.loaded],
            "ram_used_gb": self.ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
            "ollama_reachable": self.ollama_reachable,
        }


def gpu_info() -> GpuInfo:
    """NVIDIA GPU state via nvidia-smi. Absent GPU is not an error."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return GpuInfo(detail="nvidia-smi not found")
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
                  "temperature.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=_TIMEOUT_S, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        if not out:
            return GpuInfo(detail="nvidia-smi returned nothing")
        name, used, total, util, temp = [f.strip() for f in out[0].split(",")]
        return GpuInfo(
            available=True, name=name,
            mem_used_mb=int(float(used)), mem_total_mb=int(float(total)),
            utilisation_pct=int(float(util)), temperature_c=int(float(temp)),
        )
    except Exception as exc:                      # noqa: BLE001 - see DESIGN RULES
        return GpuInfo(detail=f"{type(exc).__name__}: {exc}")


def ram_info() -> tuple[float, float]:
    """(used_gb, total_gb) from /proc/meminfo. (0, 0) if unreadable."""
    try:
        fields: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                fields[key] = int(rest.strip().split()[0])  # kB
        total = fields.get("MemTotal", 0) / 1024 / 1024
        avail = fields.get("MemAvailable", 0) / 1024 / 1024
        return round(total - avail, 1), round(total, 1)
    except Exception:                             # noqa: BLE001
        return 0.0, 0.0


def _api_root(base_url: str) -> str:
    """
    Ollama's native API sits beside its OpenAI-compatible shim:
    http://host:11434/v1  ->  http://host:11434
    """
    root = (base_url or "http://localhost:11434/v1").rstrip("/")
    return root[:-3].rstrip("/") if root.endswith("/v1") else root


def loaded_models(base_url: str = "") -> tuple[list[LoadedModel], bool]:
    """
    ([models currently in memory], ollama_reachable).

    Uses Ollama's /api/ps rather than inferring residency from nvidia-smi,
    because the GPU only reports a total - it cannot tell you WHICH model
    the bytes belong to, or how much of that model got left on the CPU.
    """
    url = _api_root(base_url) + "/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return [], False

    models: list[LoadedModel] = []
    for entry in payload.get("models", []):
        details = entry.get("details") or {}
        models.append(LoadedModel(
            name=entry.get("name", "?"),
            size_mb=int(entry.get("size", 0)) // (1024 * 1024),
            vram_mb=int(entry.get("size_vram", 0)) // (1024 * 1024),
            context_length=int(entry.get("context_length", 0) or 0),
            quantisation=details.get("quantization_level", ""),
            parameter_size=details.get("parameter_size", ""),
            expires_at=entry.get("expires_at", ""),
        ))
    return models, True


def available_models(base_url: str = "") -> list[str]:
    """Everything pulled locally, whether loaded or not. For UI pickers."""
    url = _api_root(base_url) + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []
    return sorted(m.get("name", "") for m in payload.get("models", []) if m.get("name"))


def model_capabilities(model: str, base_url: str = "") -> list[str]:
    """
    What a locally-installed model can do, as Ollama reports it.

    The one that matters here is `thinking`. A reasoning model spends tokens
    on an internal monologue that is billed against `max_tokens` but does NOT
    appear in `message.content` - so a budget tuned for a non-reasoning model
    gets consumed entirely by thought, and the call returns an empty string
    while cheerfully reporting that it generated hundreds of tokens.

    That failure is nearly invisible from the outside: no error, no warning,
    just an agent that says nothing. Asking the server what kind of model this
    is beats inferring it from a name (`qwen3.5:4b` does not say "reasoning"
    anywhere) or discovering it from the wreckage.
    """
    url = _api_root(base_url) + "/api/show"
    try:
        req = urllib.request.Request(
            url, data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []
    return list(payload.get("capabilities") or [])


def is_reasoning_model(model: str, base_url: str = "") -> bool:
    """True if this model thinks before it answers. See model_capabilities."""
    return "thinking" in model_capabilities(model, base_url)


def reasoning_models(models: list[str], base_url: str = "") -> set[str]:
    """Which of these models advertise a thinking capability."""
    return {m for m in models if is_reasoning_model(m, base_url)}


def snapshot(base_url: str = "") -> HostSnapshot:
    """One read of everything, for a UI panel or a transcript footer."""
    models, reachable = loaded_models(base_url)
    used, total = ram_info()
    return HostSnapshot(
        gpu=gpu_info(), loaded=models,
        ram_used_gb=used, ram_total_gb=total,
        ollama_reachable=reachable,
    )


def fits_in_vram(model_size_mb: int, gpu: GpuInfo | None = None) -> bool:
    """
    Would a model of this size stay fully GPU-resident right now?

    Deliberately compares against FREE memory, not total: your desktop
    session is holding VRAM too, and the arithmetic that matters is what
    is left, not what the box was sold with. Advisory only - Ollama makes
    the real decision, and it accounts for the KV cache as well.
    """
    gpu = gpu or gpu_info()
    if not gpu.available:
        return False
    return model_size_mb < gpu.mem_free_mb

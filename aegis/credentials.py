"""
aegis.credentials — API keys that did not come from the environment.

WHY THIS EXISTS
---------------
config.py reads credentials from os.environ, which is correct and which is
also the whole problem: an environment variable can only be set before the
process starts. A user who wants to try Groq has to stop the server, edit
.env, and start it again — three steps, two of which are invisible from
inside the app that is asking for the key.

So this module adds ONE more place a key may come from: a small JSON file
the UI can write at runtime. It is deliberately the LOWEST-priority source
(see config.load_settings), because a key in the environment is an explicit,
per-session statement and a key in a file is a remembered convenience. When
the two disagree, the explicit one wins.

WHERE THE FILE LIVES, AND WHY NOT IN THE PROJECT
------------------------------------------------
~/.aegis/keys.json, not ./.aegis-keys.json. A secrets file inside the working
tree is one `git add -A` away from a public repository, and .gitignore only
protects you until somebody force-adds or copies the directory. Keeping it
outside the tree means the mistake is not available to make. AEGIS_KEYSTORE
overrides the location for anyone who wants it elsewhere.

DESIGN RULES
------------
* Never raises. A missing, unreadable, or corrupt store is a normal
  condition — it means "no saved keys", not "crash the app".
* No os.environ reads. config.py owns that; the path is passed in.
* 0600 on write. A key file the rest of the machine can read is not storage,
  it is disclosure with extra steps.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".aegis" / "keys.json"


def store_path(override: str = "") -> Path:
    """Where the key store lives. `override` normally comes from config."""
    return Path(override).expanduser() if override else DEFAULT_PATH


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def all_keys(path: str | Path = "") -> dict[str, str]:
    """
    {provider: key} for everything saved. Values are the real keys — this is
    the one function that returns them, so grep finds every call site.
    """
    target = path if isinstance(path, Path) else store_path(str(path))
    keys = _read(target).get("keys")
    if not isinstance(keys, dict):
        return {}
    return {str(k): str(v) for k, v in keys.items() if v}


def get_key(provider: str, path: str | Path = "") -> str:
    return all_keys(path).get(provider, "")


def saved_providers(path: str | Path = "") -> list[str]:
    """Which providers have a remembered key. Safe to display."""
    return sorted(all_keys(path))


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def set_key(provider: str, key: str, path: str | Path = "") -> bool:
    """
    Remember `key` for `provider`. An empty key deletes the entry rather
    than storing "", so "saved but blank" is not a state that can exist.

    Returns True on success. Never raises: the caller is a UI, and a UI that
    dies because a home directory is read-only is worse than one that says
    it could not save.
    """
    target = path if isinstance(path, Path) else store_path(str(path))
    data = _read(target)
    keys = data.get("keys")
    if not isinstance(keys, dict):
        keys = {}

    key = (key or "").strip()
    if key:
        keys[provider] = key
    else:
        keys.pop(provider, None)
    data["keys"] = keys

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a half-written key file would take every saved
        # credential with it, and the failure mode (truncated JSON) is
        # indistinguishable from "no keys" at read time.
        tmp = target.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600, before it is visible
        tmp.replace(target)
        return True
    except OSError:
        return False


def forget_key(provider: str, path: str | Path = "") -> bool:
    return set_key(provider, "", path)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def fingerprint(key: str) -> str:
    """
    A key rendered safe to put on screen: `sk-or…8f2c`.

    Enough to answer "is this the key I think it is?" and not enough to use.
    Short keys collapse to a bare mask rather than leaking most of
    themselves — the redaction has to hold for the worst case, not the
    typical one.
    """
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) < 12:
        return "•" * len(key)
    return f"{key[:5]}…{key[-4:]}"

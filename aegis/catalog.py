"""
aegis.catalog — asking an endpoint what it can actually serve.

WHY NOT JUST HARD-CODE THE MODEL LISTS
--------------------------------------
providers.py carries a curated list per provider, and that list is wrong the
week after it is written. Model ids are the fastest-moving strings in this
whole system: they get renamed, dated, deprecated and superseded, and the
failure they produce is a 404 in the middle of round two — after you have
typed a topic and waited.

Every provider here speaks the OpenAI dialect, and that dialect includes
GET /models. So the authoritative list is one HTTP call away, and the
curated list becomes what it should have been all along: a fallback for when
the endpoint is unreachable or has not been asked yet.

DESIGN RULES (same as hostinfo, for the same reason)
----------------------------------------------------
* Never raises. A provider being down, slow, or rude is a normal condition.
  Every probe degrades to "unknown" and the UI keeps working.
* No os.environ. config.py owns that; keys and URLs are passed in.
* urllib, not the openai SDK. This module has to work before a client is
  successfully constructed — that is precisely when you need it most, since
  "is my key any good?" is a question asked by people whose key is not.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# Short. This runs inside a UI render, and a provider that takes eight
# seconds to list its models has already failed at being useful.
_TIMEOUT_S = 8.0


@dataclass
class Probe:
    """
    The result of asking an endpoint whether it will talk to us.

    `ok` and `models` are separate because they answer different questions
    and can disagree: a key can be valid against a provider that does not
    implement /models at all, which is a working setup with an empty list.
    Collapsing the two would report that working setup as broken.
    """

    ok: bool = False
    status: int = 0
    detail: str = ""
    models: list[str] = field(default_factory=list)

    @property
    def unauthorised(self) -> bool:
        return self.status in (401, 403)


def _models_url(base_url: str) -> str:
    return (base_url or "").rstrip("/") + "/models"


def _request(url: str, api_key: str, extra_headers: dict[str, str] | None = None):
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra_headers or {})
    return urllib.request.Request(url, headers=headers, method="GET")


def _parse_models(payload: object) -> list[str]:
    """
    Pull ids out of a /models response.

    Providers agree on the shape {"data": [{"id": ...}]} and then do not:
    some return a bare list, some nest the id under "name", and Google
    prefixes every id with "models/". Normalising here rather than at each
    call site means the UI receives one shape from fourteen endpoints.
    """
    rows: list = []
    if isinstance(payload, dict):
        data = payload.get("data")
        rows = data if isinstance(data, list) else []
        if not rows and isinstance(payload.get("models"), list):
            rows = payload["models"]
    elif isinstance(payload, list):
        rows = payload

    names: list[str] = []
    for row in rows:
        if isinstance(row, str):
            name = row
        elif isinstance(row, dict):
            name = str(row.get("id") or row.get("name") or "")
        else:
            continue
        # Google's OpenAI-compatible listing returns "models/gemini-2.5-flash"
        # while its chat endpoint wants "gemini-2.5-flash". Handing the UI an
        # id the completion call will reject is worse than not listing it.
        if name.startswith("models/"):
            name = name[len("models/"):]
        if name:
            names.append(name)
    return sorted(set(names))


def probe(base_url: str, api_key: str = "",
          extra_headers: dict[str, str] | None = None,
          timeout: float = _TIMEOUT_S) -> Probe:
    """
    One round trip that answers both "does this key work?" and "what models
    are there?".

    The two questions share an answer because they share a request, and
    asking them separately would double the latency of every provider switch
    for no extra information.
    """
    if not base_url:
        return Probe(detail="No base URL set for this provider.")

    try:
        with urllib.request.urlopen(
            _request(_models_url(base_url), api_key, extra_headers),
            timeout=timeout,
        ) as response:
            payload = json.loads(response.read())
            return Probe(ok=True, status=getattr(response, "status", 200),
                         models=_parse_models(payload))

    except urllib.error.HTTPError as exc:
        # An HTTP error is a REPLY, not a failure to reach anyone, and the
        # status code is the single most useful thing the user can be told:
        # 401 means fix the key, 404 means fix the URL, 429 means wait. A
        # generic "could not connect" throws all of that away.
        detail = {
            401: "Key rejected (401). Check it was copied whole.",
            403: "Key refused (403). It may lack permission, or need billing enabled.",
            404: "No /models endpoint here (404). The base URL is probably wrong.",
            429: "Rate limited (429). The key works — you are just out of quota.",
        }.get(exc.code, f"HTTP {exc.code} from the provider.")
        return Probe(ok=False, status=exc.code, detail=detail)

    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Probe(detail=f"Could not reach {base_url} ({exc}).")
    except ValueError:
        return Probe(ok=True, status=200,
                     detail="Endpoint answered, but not with JSON model data.")


def remote_models(base_url: str, api_key: str = "",
                  extra_headers: dict[str, str] | None = None) -> list[str]:
    """Just the model ids. Empty list on any problem — never raises."""
    return probe(base_url, api_key, extra_headers).models


def rank_models(models: list[str], *, prefer: str = "") -> list[str]:
    """
    Order a model list so a human can find something in it.

    A raw /models listing is alphabetical and can run to several hundred
    entries — OpenRouter alone serves over 300 — which makes the picker a
    scrolling exercise rather than a choice. Free models first (they are why
    most people are here), then the currently-selected one so it never
    vanishes from view, then everything else alphabetically.
    """
    from .providers import is_free_model

    def key(name: str) -> tuple:
        return (name != prefer, not is_free_model(name), name.lower())

    return sorted(set(models), key=key)

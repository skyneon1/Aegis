#!/usr/bin/env bash
#
# Aegis dev launcher.
#
#   ./dev.sh              start the UI (default)
#   ./dev.sh ui           same
#   ./dev.sh cli "topic"  run one debate in the terminal
#   ./dev.sh test         hermetic suite (fast, no GPU)
#   ./dev.sh test-live    calibration tests against the local model
#   ./dev.sh eval         batch eval over the built-in topics
#   ./dev.sh calibrate    is the Critic a working gate? (judge the judge)
#   ./dev.sh doctor       what is installed, loaded, and reachable
#   ./dev.sh stop         stop the UI
#   ./dev.sh restart      restart it (required for .env changes to apply)
#
# WHY A SCRIPT AND NOT A README LINE
# ----------------------------------
# Three things have to be true before a debate can run: the venv exists,
# Ollama is up, and the configured model is pulled. Each fails differently
# and only one of them fails loudly. A missing model surfaces as a 404
# mid-run, after you have already typed a topic and waited; Ollama being
# down surfaces as a connection error inside the first agent turn. Checking
# up front turns three confusing runtime failures into one clear message
# before anything starts.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="./.venv"
PY="$VENV/bin/python"
PORT="${AEGIS_PORT:-8899}"
PIDFILE=".aegis-ui.pid"

c() { printf '\033[%sm%s\033[0m\n' "$1" "$2"; }
info() { c "36" "  $1"; }
ok()   { c "32" "  ✓ $1"; }
warn() { c "33" "  ! $1"; }
die()  { c "31" "  ✗ $1"; exit 1; }

# --- preflight -------------------------------------------------------------
# Ordered cheapest-first: no point starting Ollama checks if there is no venv.
need_venv() {
  [ -x "$PY" ] || die "no virtualenv at $VENV — run: python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt"
}

# Reads the resolved provider/model from config rather than re-parsing .env.
# config.py is the only thing allowed to know how settings are merged, and a
# launcher that guesses at that will eventually disagree with the app it launches.
settings_field() {
  "$PY" -c "from aegis import load_settings; s=load_settings(); print(getattr(s,'$1'))" 2>/dev/null || echo ""
}

need_ollama() {
  local provider; provider="$(settings_field provider)"
  [ "$provider" = "ollama" ] || { info "provider is '$provider' — skipping local model checks"; return 0; }

  curl -sf -m 3 http://localhost:11434/api/tags >/dev/null 2>&1 || {
    warn "Ollama is not responding on :11434"
    info "start it with:  ollama serve"
    die "cannot run locally without it (or set AEGIS_PROVIDER=fake in .env)"
  }
  ok "ollama up"

  local missing=0
  for role in proposer_model critic_model; do
    local m; m="$(settings_field "$role")"
    [ -n "$m" ] || continue
    if curl -sf -m 3 http://localhost:11434/api/tags | grep -q "\"$m\""; then
      ok "$role: $m"
    else
      warn "$role '$m' is not pulled — run:  ollama pull $m"
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || die "pull the missing model(s) above first"
}

cmd_doctor() {
  need_venv
  c "1;36" "AEGIS DOCTOR"
  info "python:   $("$PY" --version 2>&1)"
  info "provider: $(settings_field provider)"
  info "proposer: $(settings_field proposer_model)"
  info "critic:   $(settings_field critic_model)"
  info "guards:   $(settings_field max_rounds) rounds / $(settings_field max_seconds)s / \$$(settings_field max_cost_usd)"
  echo
  "$PY" - <<'PYEOF'
from aegis import catalog, credentials, hostinfo, keystore_path, load_settings
from aegis.providers import PROVIDERS

s = load_settings()

# CREDENTIAL INVENTORY, FIRST.
#
# "which providers can I actually run right now" is the question a
# fourteen-entry table creates and cannot itself answer. Printing it before
# the GPU section because for a cloud provider the GPU section is irrelevant,
# and for a local one this part is a single line.
env_keyed, saved = [], set(credentials.saved_providers(keystore_path()))
import os
for name, preset in PROVIDERS.items():
    if preset["key_env"] and os.getenv(preset["key_env"]) and preset["requires_key"]:
        env_keyed.append(name)
ready = sorted(set(env_keyed) | saved | {"fake", "ollama"})
print(f"  keystore: {keystore_path()}")
print(f"  usable:   {', '.join(ready)}")
if env_keyed:
    print(f"    from env:   {', '.join(sorted(env_keyed))}")
if saved:
    print(f"    saved keys: {', '.join(sorted(saved))}")

if not s.local and not s.is_fake:
    # A cloud provider has no host state worth printing — one round trip to
    # /models answers everything the GPU panel answers locally.
    print()
    print(f"  endpoint: {s.base_url}")
    print(f"  rate:     ${s.cost_in_per_1m:.2f} in / ${s.cost_out_per_1m:.2f} out per 1M")
    probe = catalog.probe(s.base_url, s.api_key, s.extra_headers)
    if probe.ok:
        print(f"  reachable: yes, {len(probe.models)} models listed")
    else:
        print(f"  reachable: NO — {probe.detail}")

    # Does the configured model actually exist here? Checked against the live
    # listing when there is one and the curated list otherwise, because the
    # case that most needs the warning is the case where the probe failed:
    # a .env written for Ollama pins AEGIS_PROPOSER_MODEL, and overriding
    # only AEGIS_PROVIDER on the command line carries those local model names
    # onto a cloud endpoint that has never heard of them.
    known = probe.models or PROVIDERS[s.provider].get("models") or []
    if known:
        source = "this provider's model list" if probe.models else "the known-good list"
        for role, m in (("proposer", s.proposer_model), ("critic", s.critic_model)):
            if m not in known:
                print(f"  !! {role} '{m}' is not in {source}")
                print(f"     unset AEGIS_{role.upper()}_MODEL, or pick one in the sidebar")
    raise SystemExit(0)

snap = hostinfo.snapshot(s.base_url)
g = snap.gpu
if g.available:
    print(f"  GPU:  {g.name}  {g.mem_used_mb}/{g.mem_total_mb} MB  "
          f"({g.mem_pct:.0f}% used)  {g.utilisation_pct}% util  {g.temperature_c}°C")
else:
    print(f"  GPU:  unavailable ({g.detail})")
print(f"  RAM:  {snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB")
print(f"  ollama reachable: {snap.ollama_reachable}")
if snap.loaded:
    for m in snap.loaded:
        flag = "OK " if m.gpu_fraction >= 0.999 else "!! "
        print(f"  {flag}loaded {m.name}: {m.size_mb} MB, {m.placement}, ctx {m.context_length}")
else:
    print("  no model resident (the first turn will load one)")
avail = hostinfo.available_models(s.base_url)
if avail:
    thinking = hostinfo.reasoning_models(avail, s.base_url)
    print("  installed:")
    for m in avail:
        tag = "  [thinking]" if m in thinking else ""
        print(f"    - {m}{tag}")
if thinking:
    print()
    print(f"  note: {', '.join(sorted(thinking))} reason before answering.")
    print(f"        Their thinking is billed against the token budget and does")
    print(f"        not appear in the reply, so they run on separate dials:")
    print(f"          budget  {s.reasoning_max_tokens} tokens (vs {s.max_tokens})")
    print(f"          context {s.reasoning_num_ctx} tokens (vs {s.num_ctx})")
    print(f"        Measured: ~2450 output tokens and ~260s for one critic")
    print(f"        turn on this 4GB card. Correct, but slow.")
PYEOF
}

cmd_ui() {
  need_venv
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    ok "UI already running (pid $(cat "$PIDFILE")) → http://localhost:$PORT"
    exit 0
  fi
  info "starting UI on :$PORT"
  exec "$PY" -m streamlit run app.py --server.port "$PORT" --server.headless true
}

# PID of whatever this project has running on $PORT, if anything.
ui_pid() { pgrep -f "streamlit run app.py" | head -1; }

port_busy() { curl -sf -m 2 -o /dev/null "http://localhost:$PORT"; }

# Stop any server we own on this port and wait for the socket to clear.
ui_kill() {
  local pid; pid="$(ui_pid)"
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
  [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null
  rm -f "$PIDFILE"
  for _ in 1 2 3 4 5 6 7 8 9 10; do port_busy || return 0; sleep 0.5; done
  return 1
}

cmd_ui_bg() {
  need_venv

  # ALWAYS restart rather than skipping when the port is busy.
  #
  # This used to fire-and-forget a `nohup streamlit &` and print "UI running
  # (pid N)" with the shell's pid, never checking that it bound. A stale
  # server from an earlier session kept the port, the new process died
  # immediately, and the script reported success while the browser served
  # OLD CODE with an OLD .env - the sidebar showed models that had been
  # changed hours earlier. Reporting success without verifying it is worse
  # than failing.
  #
  # Restarting is also the only way a config change takes effect: config.py
  # calls load_dotenv() at import, so a running server holds whatever .env
  # said when it started.
  if port_busy; then
    info "port $PORT already serving — restarting so .env changes take effect"
    ui_kill || die "could not free port $PORT; something else is bound to it"
  fi

  nohup "$PY" -m streamlit run app.py --server.port "$PORT" --server.headless true \
      > .aegis-ui.log 2>&1 &
  echo $! > "$PIDFILE"

  # Verify it actually came up. Poll rather than sleep: a fixed sleep is
  # either too short (false failure) or too slow (wasted seconds every time).
  for _ in $(seq 1 40); do
    port_busy && {
      ok "UI running (pid $(ui_pid)) → http://localhost:$PORT   [logs: .aegis-ui.log]"
      "$PY" -c "from aegis import load_settings; s=load_settings(); print(f'    serving {s.proposer_model} vs {s.critic_model}')"
      return 0
    }
    sleep 0.5
  done
  warn "UI did not come up within 20s. Last log lines:"
  tail -15 .aegis-ui.log
  die "start failed"
}

cmd_stop() {
  # Match on the command line, not just the pidfile: a server survives the
  # session that started it, and the pidfile does not survive a `rm -rf` of
  # the working tree or a machine restart. The pidfile is a hint, not the
  # source of truth.
  if [ -z "$(ui_pid)" ] && [ ! -f "$PIDFILE" ]; then
    warn "nothing running on port $PORT"; rm -f "$PIDFILE"; return 0
  fi
  ui_kill && ok "stopped" || warn "port $PORT still busy"
}

case "${1:-ui}" in
  ui)        cmd_ui ;;
  ui-bg)     cmd_ui_bg ;;
  stop)      cmd_stop ;;
  restart)   cmd_ui_bg ;;
  doctor)    cmd_doctor ;;
  cli)       need_venv; need_ollama; shift; exec "$PY" cli.py "$@" ;;
  test)      need_venv; exec "$VENV/bin/pytest" tests/ -q ;;
  test-live) need_venv; need_ollama; exec "$VENV/bin/pytest" -m live tests/ -v ;;
  eval)      need_venv; need_ollama; shift; exec "$PY" evaluate.py "$@" ;;
  calibrate) need_venv; need_ollama; shift; exec "$PY" calibrate.py "$@" ;;
  *)         sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//' ; exit 1 ;;
esac

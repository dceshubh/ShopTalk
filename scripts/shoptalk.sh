#!/usr/bin/env bash
# scripts/shoptalk.sh — start/stop the local ShopTalk stack idempotently, in one command.
#
# Usage (from anywhere):
#   ./scripts/shoptalk.sh            # (re)start: stop any stale ShopTalk API/UI processes,
#                                    #   make sure Redis is up, boot both fresh in the
#                                    #   background. Safe to run when nothing is running,
#                                    #   when it's already running, or after a crash —
#                                    #   "first time / already up / restart" all converge
#                                    #   to the same end state: API + UI up, nothing duped.
#   ./scripts/shoptalk.sh down       # stop: kill ShopTalk's API + UI processes only — this
#                                    #   is what frees the ~2-3 GB of loaded models (Groq
#                                    #   client, bge-base-en-v1.5 encoder, Chroma, Whisper,
#                                    #   Piper) when you're not actively using the app.
#   ./scripts/shoptalk.sh status     # show what's running and the URLs to open
#
# Logs land in /tmp/shoptalk_{api,ui}.log — `tail -f` them while the stack boots.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR=".venv-shoptalk"
API_LOG="/tmp/shoptalk_api.log"
UI_LOG="/tmp/shoptalk_ui.log"

# Matched by FULL COMMAND LINE, scoped to THIS project's venv path — not just "uvicorn" or
# a port number, either of which can collide with unrelated projects on the same machine
# (this one, for instance, also runs a separate `Finances Tracker` API on :8000 — why
# ShopTalk's API defaults to :8010, see configs/config.yaml). Two patterns, unioned and
# de-duped, rather than one alternation regex — keeps this portable across the BSD pgrep
# on macOS and the GNU pgrep elsewhere.
_pids() {
  {
    pgrep -f "${VENV_DIR}/bin/uvicorn.*src.api.main" || true
    pgrep -f "${VENV_DIR}/bin/streamlit.*src/ui/app" || true
  } | sort -un | tr '\n' ' '
}

_kill_existing() {
  local pids
  pids="$(_pids)"
  if [[ -n "${pids// /}" ]]; then
    echo "Stopping existing ShopTalk processes: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 2
    pids="$(_pids)"
    # shellcheck disable=SC2086
    [[ -n "${pids// /}" ]] && kill -9 $pids 2>/dev/null || true
  fi
}

_redis_up() { redis-cli ping >/dev/null 2>&1; }

_print_urls() {
  echo "  API:   http://localhost:8010/health   (logs: $API_LOG)"
  echo "  UI:    http://localhost:8501           (logs: $UI_LOG)"
}

_status() {
  local pids
  pids="$(_pids)"
  if [[ -n "${pids// /}" ]]; then
    echo "Running (pid(s): $pids):"
    # `ps -p` wants a comma-separated list, not space-separated — join before passing.
    ps -p "$(echo "$pids" | tr -s ' ' ',' | sed 's/^,\|,$//g')" -o pid,etime,command | tail -n +2
    echo
    _print_urls
  else
    echo "Nothing running."
  fi
  if _redis_up; then
    echo "Redis: up   (redis://localhost:6379)"
  else
    echo "Redis: DOWN — \`brew services start redis\` (or let \`./scripts/shoptalk.sh\` start it for you)"
  fi
}

case "${1:-up}" in
  down)
    _kill_existing
    echo "Stopped — models unloaded, memory freed."
    echo "(Redis left running — it's lightweight, ~1MB resident; \`brew services stop redis\` to also free it.)"
    ;;
  status)
    _status
    ;;
  up | "")
    _kill_existing

    if ! _redis_up; then
      echo "Redis isn't running — starting it (\`brew services start redis\`)…"
      brew services start redis >/dev/null
      for _ in $(seq 1 10); do _redis_up && break; sleep 1; done
    fi

    echo "Starting API…"
    nohup "${VENV_DIR}/bin/uvicorn" src.api.main:app --host 0.0.0.0 --port 8010 >"$API_LOG" 2>&1 &
    disown

    echo "Starting UI…"
    nohup "${VENV_DIR}/bin/streamlit" run src/ui/app.py >"$UI_LOG" 2>&1 &
    disown

    echo
    echo "Booting — the API takes ~15-20s on its first request (every model loads exactly"
    echo "once, at startup: Groq client, bge-base-en-v1.5 encoder, Chroma index, Whisper,"
    echo "Piper). Tail the logs to watch it come up:"
    echo "  tail -f $API_LOG $UI_LOG"
    echo
    _print_urls
    echo
    echo "When you're done and want the memory back: ./scripts/shoptalk.sh down"
    ;;
  *)
    echo "Usage: $0 [up|down|status]" >&2
    exit 1
    ;;
esac

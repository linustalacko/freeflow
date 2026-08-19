#!/bin/bash
# launchd wrapper for `ollama serve` (see launchd/com.freeflow.ollama.plist).
#
# The Ollama *desktop app* also runs `ollama serve` on 127.0.0.1:11434 when it is
# open / set to launch at login. If it got there first, a bare `ollama serve` in
# this job dies with "address already in use" and launchd's KeepAlive respawns it
# forever (18MB of the same line in ollama.log). So: if something already
# listens on the port, exit 0 and let launchd leave us alone (the plist uses
# KeepAlive.SuccessfulExit=false, i.e. only restart on failure). Otherwise exec
# the server with the memory/keep-alive settings from the plist environment.
#
# Note: when the desktop app owns the server, the plist's OLLAMA_* caps do NOT
# apply — quit the app (and untick it in System Settings ▸ General ▸ Login Items)
# if you want this job's settings to be the ones in effect.
set -u
HOST_PORT="${OLLAMA_HOST:-127.0.0.1:11434}"
PORT="${HOST_PORT##*:}"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "$(date '+%H:%M:%S') ollama already listening on :$PORT (desktop app?) — not starting a second server" >&2
  exit 0
fi

OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo /opt/homebrew/bin/ollama)}"
exec "$OLLAMA_BIN" serve

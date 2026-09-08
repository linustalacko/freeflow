#!/bin/bash
set -euo pipefail

# Explicit optional setup. No login service, model prewarming, or vision download.
if ! command -v ollama >/dev/null; then brew install ollama; fi
journal_ollama="$(command -v ollama)"
journal_host=127.0.0.1:11436
journal_plist="$HOME/Library/LaunchAgents/com.freeflow.journal-model.plist"

# Remove only the old journal-specific login service created by this installer.
if [ -f "$journal_plist" ]; then
  python3 - "$journal_plist" <<'PY'
import plistlib, sys
with open(sys.argv[1], 'rb') as stream:
    service = plistlib.load(stream)
if (service.get('Label') != 'com.freeflow.journal-model' or
    service.get('EnvironmentVariables', {}).get('OLLAMA_HOST') != '127.0.0.1:11436'):
    raise SystemExit('Unrecognized journal service file; left unchanged.')
PY
  launchctl bootout "gui/$(id -u)/com.freeflow.journal-model" 2>/dev/null || true
  rm "$journal_plist"
fi
if curl --silent --fail --max-time 1 "http://$journal_host/api/version" >/dev/null; then
  echo 'Stop the running activity journal before installing its model.' >&2
  exit 1
fi

journal_pid=''
cleanup() {
  if [ -n "$journal_pid" ]; then
    kill "$journal_pid" 2>/dev/null || true
    wait "$journal_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
  OLLAMA_HOST="$journal_host" OLLAMA_NO_CLOUD=1 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=0 OLLAMA_CONTEXT_LENGTH=2048 \
  "$journal_ollama" serve >/dev/null 2>&1 &
journal_pid=$!
journal_ready=false
for journal_attempt in {1..30}; do
  kill -0 "$journal_pid" 2>/dev/null || { echo 'Local model runtime failed to start.' >&2; exit 1; }
  if curl --silent --fail --max-time 1 "http://$journal_host/api/version" >/dev/null; then journal_ready=true; break; fi
  sleep 0.2
done
$journal_ready || { echo 'Local model runtime did not become ready.' >&2; exit 1; }
OLLAMA_HOST="$journal_host" "$journal_ollama" pull qwen2.5:0.5b
echo 'Optional journal model installed. Start it from FreeFlow → Activity Journal → Start Journal.'

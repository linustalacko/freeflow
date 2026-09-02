#!/usr/bin/env bash
# Deploy the local pipeline from the repo and prove what is running.
#
#   local-setup/deploy.sh           # test → copy → restart → verify
#   local-setup/deploy.sh --check   # only compare the running services with the repo
#
# The repo is the source of truth; ~/.freeflow-stt/ holds the copies launchd runs.
# Each service reports the sha of the file it loaded (/health, /v1/status), so a
# copy that drifted from the checkout — an old file re-copied, an edit made in
# place, a plist pointing elsewhere — shows up here instead of as a regression
# nobody can explain.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="$HOME/.freeflow-stt"
FILES=(router.py detclean.py stt_server.py)
VENV_PY="$HOME/.freeflow-ft/venv/bin/python"

sha() { shasum -a 256 "$1" | cut -c1-12; }

verify() {
  local ok=0
  local want_router want_stt have_router have_stt
  want_router=$(sha local-setup/router.py); want_stt=$(sha local-setup/stt_server.py)
  have_router=$(curl -s -m 3 127.0.0.1:11435/v1/status | python3 -c 'import json,sys; print(json.load(sys.stdin).get("source_sha","?"))' 2>/dev/null || echo "down")
  have_stt=$(curl -s -m 3 127.0.0.1:8082/health | python3 -c 'import json,sys; print(json.load(sys.stdin).get("source_sha","?"))' 2>/dev/null || echo "down")
  for f in "${FILES[@]}"; do
    if ! cmp -s "local-setup/$f" "$DEST/$f"; then echo "DRIFT: $DEST/$f differs from the repo"; ok=1; fi
  done
  [ "$have_router" = "$want_router" ] && echo "router  running $have_router = repo" || { echo "router  running $have_router ≠ repo $want_router"; ok=1; }
  [ "$have_stt" = "$want_stt" ] && echo "stt     running $have_stt = repo" || { echo "stt     running $have_stt ≠ repo $want_stt"; ok=1; }
  echo "router policy: $(curl -s -m 3 127.0.0.1:11435/v1/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print("force_local=%s precache_llm=%s heartbeat=%ss local_warm=%s" % (d["force_local"], d["precache_llm"], d["local_heartbeat_s"], d["local_warm"]))' 2>/dev/null || echo unavailable)"
  echo "stt memory:    $(curl -s -m 3 127.0.0.1:8082/health | python3 -c 'import json,sys; m=json.load(sys.stdin).get("memory",{}); print("active %sMB cache %sMB" % (m.get("active_mb","?"), m.get("cache_mb","?")))' 2>/dev/null || echo unavailable)"
  local auth
  auth=$(codesign -dvv "build/FreeFlow Dev.app" 2>&1 | grep -o "Authority=Apple Development[^)]*)" | tail -n 1)
  echo "app:           ${auth:-not built} — $(git rev-parse --short HEAD) $(git diff --quiet && echo clean || echo 'with uncommitted changes')"
  return $ok
}

if [ "${1:-}" = "--check" ]; then verify; exit $?; fi

echo "==> tests"
python3 local-setup/test_router.py >/dev/null 2>&1 || { echo "router tests FAILED — not deploying"; exit 1; }
python3 local-setup/test_detclean.py >/dev/null 2>&1 || { echo "detclean tests FAILED — not deploying"; exit 1; }
"$VENV_PY" local-setup/test_stt_server.py >/dev/null 2>&1 || { echo "stt tests FAILED — not deploying"; exit 1; }
echo "    all passing"

echo "==> copying to $DEST"
mkdir -p "$DEST"
for f in "${FILES[@]}"; do cp "local-setup/$f" "$DEST/$f"; done

echo "==> restarting services (kickstart keeps the plist env; use bootout/bootstrap after env changes)"
launchctl kickstart -k "gui/$(id -u)/com.freeflow.router"
launchctl kickstart -k "gui/$(id -u)/com.freeflow.stt"
for _ in $(seq 1 60); do
  sleep 1
  curl -s -m 2 127.0.0.1:8082/health 2>/dev/null | grep -q '"ok": true' && curl -s -m 2 127.0.0.1:11435/v1/status >/dev/null 2>&1 && break
done

echo "==> verify"
verify

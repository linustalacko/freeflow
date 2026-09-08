#!/usr/bin/env bash
# Rebuild FreeFlow (dev), ad-hoc sign it, and relaunch — in one command.
# Use this after any code change. `make run` doesn't work here because the
# Makefile's codesign step wants a "FreeFlow Dev" identity that doesn't exist;
# Apple Silicon auto-ad-hoc-signs the binary, so we just ad-hoc sign the bundle
# and run it.
set -euo pipefail
cd "$(dirname "$0")"
APP="build/FreeFlow Dev.app"

echo "==> building (arm64)…"
# The codesign step inside `make` will fail (no 'FreeFlow Dev' identity) — that's
# expected and harmless; the swiftc build already produced the binary.
make ARCH=arm64 >/tmp/freeflow-build.log 2>&1 || true
if ! grep -q "FreeFlow Dev" "build/FreeFlow Dev.app/Contents/MacOS/FreeFlow Dev" 2>/dev/null \
   && [ ! -f "build/FreeFlow Dev.app/Contents/MacOS/FreeFlow Dev" ]; then
  echo "build failed — see /tmp/freeflow-build.log"; tail -20 /tmp/freeflow-build.log; exit 1
fi
if grep -qiE "error:" /tmp/freeflow-build.log; then
  echo "COMPILE ERRORS:"; grep -iE "error:" /tmp/freeflow-build.log | head; exit 1
fi

echo "==> signing…"
# Sign with your stable Apple Development cert so macOS keeps the app's
# Accessibility/Input-Monitoring/Mic grants across rebuilds. Ad-hoc signing
# changes the signature every build and silently drops those permissions
# (which makes the app feel "not running" — the hotkey goes dead).
DEV_ID=$(security find-identity -v -p codesigning 2>/dev/null | grep "Apple Development" | head -1 | awk '{print $2}')
if [ -n "$DEV_ID" ]; then
  codesign --force --deep --sign "$DEV_ID" --entitlements FreeFlow.entitlements "$APP" >/dev/null 2>&1 \
    && echo "    signed (stable dev cert ${DEV_ID:0:10}…) — permissions persist"
else
  echo "    no Apple Development cert — falling back to ad-hoc (permissions reset each rebuild)"
  codesign --force --deep --sign - --entitlements FreeFlow.entitlements "$APP" >/dev/null 2>&1
fi
codesign --verify "$APP" && echo "    signature OK"

echo "==> relaunching…"
pkill -f "FreeFlow Dev.app/Contents/MacOS/FreeFlow Dev" 2>/dev/null || true
for _ in 1 2 3 4; do pgrep -f "FreeFlow Dev.app/Contents/MacOS" >/dev/null || break; sleep 1; done
# Relaunch via the login agent if installed (reliable GUI session), else plain open.
AGENT="gui/$(id -u)/com.freeflow.dev.autostart"
if launchctl print "$AGENT" >/dev/null 2>&1; then
  launchctl kickstart -k "$AGENT"
else
  open "$APP"
fi
echo "==> done. If the hotkey / typing / mic stop working, re-grant permissions"
echo "    (System Settings ▸ Privacy & Security ▸ Accessibility / Microphone /"
echo "     Input Monitoring) — ad-hoc rebuilds change the signature and macOS"
echo "     sometimes drops the grants."

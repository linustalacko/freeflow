#!/bin/bash
# Native Accessibility smoke test. Creates only a synthetic WebKit email view;
# never reads other applications, changes permissions, or calls an AI provider.
set -euo pipefail
cd "$(dirname "$0")/../.."
writing_scratch=$(mktemp -d "${TMPDIR:-/tmp}/freeflow-writing-smoke.XXXXXX")
trap 'rm -rf "$writing_scratch"' EXIT
swiftc -parse-as-library Sources/VoiceWritingCore.swift Sources/VoiceWritingTargetService.swift \
  local-setup/voice-writing-smoke/Probe.swift -o "$writing_scratch/probe"
swiftc -parse-as-library local-setup/voice-writing-smoke/BrowserFixture.swift -o "$writing_scratch/fixture"
"$writing_scratch/fixture" "$writing_scratch/probe"

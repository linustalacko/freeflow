#!/usr/bin/env python3
"""Build the optional Core ML benchmark worker outside the FreeFlow project."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from benchmark import HERE, command

REVISION = "5c19d5e12320e22bbfb7a1877b089d2665a69add"


def publish(binary, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, output)
    for bundle in binary.parent.glob("*.bundle"):
        destination = output.parent / bundle.name
        if not destination.exists():
            shutil.copytree(bundle, destination)
    output.with_suffix(".json").write_text(json.dumps({
        "fluid_audio_revision": REVISION, "model": "parakeet-tdt-0.6b-v2-coreml",
        "library_logging_disabled": True,
        "worker_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "worker_source_sha256": hashlib.sha256((HERE / "CoreMLWorker.swift").read_bytes()).hexdigest(),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("worker and metadata output must be new")
    with tempfile.TemporaryDirectory(prefix="freeflow-coreml-build-") as temporary:
        root = Path(temporary)
        dependency = root / "FluidAudio"
        print("Fetching pinned FluidAudio benchmark dependency...", flush=True)
        command(["git", "init", str(dependency)])
        command(["git", "-C", str(dependency), "remote", "add", "origin", "https://github.com/FluidInference/FluidAudio.git"])
        command(["git", "-C", str(dependency), "fetch", "--depth", "1", "origin", REVISION])
        command(["git", "-C", str(dependency), "checkout", "--detach", "FETCH_HEAD"])
        logger = dependency / "Sources/FluidAudio/Shared/AppLogger.swift"
        content = logger.read_text()
        needle = "    private func log(_ level: Level, _ message: String) {\n"
        if content.count(needle) != 1:
            raise ValueError("dependency logging boundary changed")
        logger.write_text(content.replace(needle, needle +
            '        if ProcessInfo.processInfo.environment["FREEFLOW_BENCHMARK_SILENT"] == "1" { return }\n'))
        package = root / "Worker"
        (package / "Sources").mkdir(parents=True)
        (package / "Package.swift").write_text('''// swift-tools-version: 6.0
import PackageDescription
let package = Package(name: "FreeFlowCoreML", platforms: [.macOS(.v14)], dependencies: [.package(path: "../FluidAudio")], targets: [.executableTarget(name: "FreeFlowCoreML", dependencies: [.product(name: "FluidAudio", package: "FluidAudio")], path: "Sources")])
''')
        shutil.copyfile(HERE / "CoreMLWorker.swift", package / "Sources/CoreMLWorker.swift")
        print("Building the isolated release worker...", flush=True)
        command(["swift", "build", "-c", "release", "--product", "FreeFlowCoreML", "-j", "6"], cwd=package)
        binary_dir = command(["swift", "build", "-c", "release", "--show-bin-path"], cwd=package).stdout.decode().strip()
        publish(Path(binary_dir) / "FreeFlowCoreML", args.output)
    print("Worker saved; temporary checkout and build files removed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print("Core ML worker build failed: " + type(error).__name__, file=sys.stderr)
        sys.exit(2)

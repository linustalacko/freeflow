#!/usr/bin/env python3
"""Start isolated local engines, benchmark, then stop them and remove scratch files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from benchmark import HERE, command


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(process, port):
    deadline = time.monotonic() + 120
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("an isolated server exited during startup")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(.2)
    raise RuntimeError("isolated server startup timed out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stt-python", type=Path, default=Path.home() / ".freeflow-ft/venv/bin/python")
    parser.add_argument("--model-dir", type=Path, default=Path.home() / ".cache/freeflow-benchmark/models")
    args = parser.parse_args()
    model_names = ["ggml-large-v3-turbo-q5_0.bin", "ggml-base.en.bin"]
    if not args.stt_python.is_file() or not shutil.which("whisper-server"):
        parser.error("requires the local FreeFlow Python runtime and whisper-server; see README.md")
    if not all((args.model_dir / name).is_file() for name in model_names):
        parser.error("Whisper model files are missing; see README.md")
    if args.output.exists() or args.output.with_suffix(".md").exists() or args.repeats < 1:
        parser.error("output files must be new and repeats positive")
    processes = []
    try:
        with tempfile.TemporaryDirectory(prefix="freeflow-engines-") as tmp:
            temp = Path(tmp)
            ports = []
            while len(ports) < 3:
                port = free_port()
                if port not in ports:
                    ports.append(port)
            config = json.loads((HERE / "engines.json").read_text())
            stt_env = {**os.environ, "STT_PORT": str(ports[0]), "ROUTER_PRECACHE_URL": "",
                       "STT_MODEL": "mlx-community/parakeet-tdt-0.6b-v2", "STT_HEARTBEAT_S": "0",
                       "STT_PARTIAL_EVERY_S": "2", "STT_LEFT_CONTEXT_S": "10",
                       "STT_SETTLE_HORIZON_S": "1", "STT_SETTLE_GAP_S": "0",
                       "STT_MIN_SETTLE_S": "1", "STT_CACHE_MB": "512", "STT_WIRED_MB": "2048",
                       "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            # Preserve the virtualenv entry point; resolving its symlink would
            # launch the base interpreter without the installed STT packages.
            commands = [[str(args.stt_python.absolute()), str(HERE.parent / "stt_server.py")]]
            for index, name in enumerate(model_names, 1):
                commands.append(["whisper-server", "--model", str((args.model_dir / name).resolve()),
                                 "--host", "127.0.0.1", "--port", str(ports[index]),
                                 "--threads", "4", "--language", "en", "--public", str(temp)])
            for index, argv in enumerate(commands):
                print(f"Starting isolated engine {index + 1}/3", flush=True)
                process = subprocess.Popen(argv, cwd=temp, env=stt_env if index == 0 else None,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                processes.append(process)
                wait_ready(process, ports[index])
            for engine, port, path in zip(config["engines"],
                    [ports[0], ports[0], ports[1], ports[2]],
                    ["/v1", "/v1/audio/transcriptions", "/inference", "/inference"]):
                engine["url"] = f"http://127.0.0.1:{port}{path}"
            config_path = temp / "engines.json"
            config_path.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(HERE / "benchmark.py"),
                                     "--manifest", str(args.manifest.resolve()), "--config", str(config_path),
                                     "--output", str(args.output.resolve()), "--repeats", str(args.repeats)])
            if args.output.is_file():
                report = json.loads(args.output.read_text())
                versions = command([str(args.stt_python), "-c",
                    "import importlib.metadata,json; print(json.dumps({x:importlib.metadata.version(x) "
                    "for x in ['parakeet-mlx','mlx','scipy','aiohttp']}))"])
                report["local_setup"] = {
                    "stt_source_sha256": hashlib.sha256((HERE.parent / "stt_server.py").read_bytes()).hexdigest(),
                    "whisper_binary_sha256": hashlib.sha256(Path(shutil.which("whisper-server")).read_bytes()).hexdigest(),
                    "runtime_versions": json.loads(versions.stdout),
                    "whisper_models_sha256": {name: hashlib.sha256((args.model_dir / name).read_bytes()).hexdigest()
                                               for name in model_names},
                    "whisper_threads": 4, "whisper_vad": False, "cleanup_precache": False,
                    "cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.decode().strip(),
                    "memory_bytes": int(command(["sysctl", "-n", "hw.memsize"]).stdout),
                    "load_average_at_finish": list(os.getloadavg()),
                }
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            return result.returncode
    except (OSError, RuntimeError, subprocess.SubprocessError):
        print("Local benchmark could not start or finish. Check installed runtimes and cached models; no app settings were changed.", file=sys.stderr)
        return 2
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if processes:
            print("Stopped all isolated benchmark servers.", flush=True)


if __name__ == "__main__":
    sys.exit(main())

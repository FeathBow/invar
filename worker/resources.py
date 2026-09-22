import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

FORMAT = "invar-process-resources-v1"
KIBIBYTE = 1024
SIGNAL_EXIT_BASE = 128


def rss_unit(system):
    if system == "darwin":
        return "bytes", 1
    if system == "linux":
        return "KiB", KIBIBYTE
    raise ValueError("Process peak RSS units are not defined for: " + system)


def artifact(path):
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def execute(command, *, directory, input_path=None):
    arguments = tuple(map(str, command))
    if not arguments:
        raise ValueError("A measured command is required")
    unit, multiplier = rss_unit(sys.platform)
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    working = str(Path.cwd())
    incoming = nullcontext(None) if input_path is None else Path(input_path).open("rb")
    with incoming as source, (destination / "stdout").open("xb") as output, (destination / "stderr").open("xb") as error:
        supplied = None
        if source is not None:
            supplied = {"path": str(Path(input_path).resolve()), "bytes": os.fstat(source.fileno()).st_size,
                        "sha256": hashlib.file_digest(source, "sha256").hexdigest()}
            source.seek(0)
        started = time.perf_counter()
        process = subprocess.Popen(arguments, stdin=source, stdout=output, stderr=error)
        try:
            # Reap this process exactly once. Popen.poll/wait would discard the
            # per-child rusage, and RUSAGE_CHILDREN would include earlier runs.
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        except BaseException:
            process.kill()
            process.wait()
            raise
        elapsed = time.perf_counter() - started
    result = {
        "format": FORMAT, "command": arguments, "cwd": working, "pid": process.pid,
        "returncode": process.returncode, "signal": -process.returncode if process.returncode < 0 else None,
        "elapsed_seconds": elapsed, "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
        "peak_rss": {"raw": usage.ru_maxrss, "unit": unit, "bytes": usage.ru_maxrss * multiplier},
        "scope": "OS wait4 rusage for this terminated command, including OS-accounted waited descendants; not simultaneous process-tree memory",
        "interval": "process launch through termination, including parsing, computation, output encoding and writing",
        "platform": {"system": sys.platform, "release": platform.release(), "machine": platform.machine()},
        "observer": {"python": sys.version, "source_sha256": artifact(Path(__file__))["sha256"]},
        "stdin": supplied,
        "stdout": artifact(destination / "stdout"), "stderr": artifact(destination / "stderr"),
        "budget_acceptance": "not_evaluated",
    }
    with (destination / "resources.json").open("x") as record:
        json.dump(result, record, indent=2, allow_nan=False)
        record.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for stdout, stderr and resources.json")
    parser.add_argument("--stdin", type=Path, help="Optional command input file; otherwise inherit stdin")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --; passed directly, without a shell")
    options = parser.parse_args()
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    if not command:
        parser.error("a command after -- is required")
    result = execute(command, directory=options.output, input_path=options.stdin)
    print(json.dumps(result, allow_nan=False), flush=True)
    code = result["returncode"]
    raise SystemExit(SIGNAL_EXIT_BASE - code if code < 0 else code)


if __name__ == "__main__":
    main()

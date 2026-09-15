import json
from pathlib import Path
import subprocess
import sys

from worker import core

ENTRY = Path(__file__).with_name("fixture.py").resolve()
CHILD_SECONDS = 45
BOOTSTRAP_BINDING = 7


def flags(values):
    return [str(item) for key, value in values.items() for item in ("--" + key, value)]


def seal(root, initial, *, observed, configuration, executable):
    inputs = {"digest": observed["policy"],
              **{name + "-digest": observed[name] for name in ("tokenizer", "base", "assembly")},
              "prompt": "one two", "tokens": 2, "temperature": 0.8, "seed": 17,
              **{name: BOOTSTRAP_BINDING for name in ("call", "attempt", "instance")}}
    command = [executable, "infer", *flags({**inputs, "python": sys.executable, "worker": ENTRY,
               "worker-config": configuration, "cache": root, "adapter": initial / "adapter.safetensors"})]
    (root / "bootstrap-command.json").write_text(json.dumps(command) + "\n")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=CHILD_SECONDS)
    log = root / "bootstrap.jsonl"
    log.write_text(completed.stdout)
    (root / "bootstrap.stderr").write_text(completed.stderr)
    (root / "bootstrap-status.json").write_text(json.dumps({"exit_code": completed.returncode}) + "\n")
    completed.check_returncode()
    command = ["policy", *flags({**inputs, "checkpoint": initial, "log": log, "exit-code": completed.returncode})]
    (root / "policy-command.json").write_text(json.dumps([executable, *command]) + "\n")
    return core.invoke(command, executable=executable)

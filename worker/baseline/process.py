from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess

from worker import core
from worker.baseline.measure import flags, write


class Client:
    def __init__(self, process, stream):
        self.process = process
        self.stream = stream

    def receive(self, stage):
        while True:
            raw = self.process.stdout.readline()
            if not raw:
                raise RuntimeError("Native inference process ended before " + stage)
            self.stream.write(raw)
            self.stream.flush()
            value = core.decode(raw)
            if not isinstance(value, dict) or not isinstance(value.get("stage"), str):
                raise ValueError("Expected a native inference stage record")
            if value["stage"] == stage:
                return value

    def exchange(self, operation, fields):
        self.process.stdin.write((json.dumps({"operation": operation, **fields}, allow_nan=False) + "\n").encode())
        self.process.stdin.flush()
        return self.receive("completed")


@contextmanager
def open_owner(options, *, settings):
    if options.inference_python is None:
        raise ValueError("CUDA composition requires its existing native inference Python")
    declaration = options.output / "native-settings.json"
    write(declaration, settings)
    entry = options.inference_entry if options.inference_entry is not None else Path(__file__).resolve().parents[2] / "entries" / "baselineinference.py"
    command = [options.inference_python, "-B", str(entry),
               *flags({"cache": options.cache, "adapter": options.initial / "adapter.safetensors",
                       "config": options.configuration, "settings": declaration})]
    write(options.output / "native-command.json", command)
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    with (options.output / "native.stdout.jsonl").open("xb") as output, (options.output / "native.stderr.log").open("xb") as error:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error, env=environment)
        client = Client(process, output)
        drained = False
        try:
            ready = client.receive("ready")
            yield client, ready
            client.exchange("close", {})
            process.stdin.close()
            process.stdin = None
            trailing, _ = process.communicate()
            drained = True
            output.write(trailing)
            if trailing or process.returncode:
                raise RuntimeError("Native inference process did not close cleanly")
        except BaseException:
            if process.poll() is None:
                process.kill()
            if not drained:
                trailing, _ = process.communicate()
                output.write(trailing)
            raise
        finally:
            write(options.output / "native-status.json", {"pid": process.pid, "exit_code": process.returncode})

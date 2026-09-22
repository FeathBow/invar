import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from worker import resources

ALLOCATION_BYTES = 64 * 1024 * 1024
OUTPUT_BYTES = 2 * 1024 * 1024
CHILD_TIMEOUT = 15


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="invar-process-resources-"))

    def invoke(self, name, program, *, input_path=None):
        command = [sys.executable, "-B", "-m", "worker.resources", "--output", str(self.root / name)]
        if input_path is not None:
            command += ["--stdin", str(input_path)]
        return subprocess.run([*command, "--", sys.executable, "-B", "-c", program],
                              capture_output=True, text=True, timeout=CHILD_TIMEOUT)

    def test_peak_includes_late_touched_allocation_and_output_encoding(self):
        program = f'''import json, resource
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
payload = bytearray(b"x") * {ALLOCATION_BYTES}
encoded = json.dumps({{"text": payload.decode()}})
print(json.dumps({{"before": before, "encoded_bytes": len(encoded), "last": payload[-1]}}))'''
        completed = self.invoke("memory", program)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        actual = json.loads(Path(result["stdout"]["path"]).read_text())
        self.assertEqual(actual["last"], ord("x"))
        self.assertGreater(actual["encoded_bytes"], ALLOCATION_BYTES)
        self.assertGreater(result["peak_rss"]["raw"], actual["before"])
        self.assertGreaterEqual(result["peak_rss"]["bytes"], ALLOCATION_BYTES)
        self.assertLess(result["peak_rss"]["bytes"], ALLOCATION_BYTES * 16)
        self.assertGreater(result["elapsed_seconds"], 0)
        self.assertGreater(result["user_seconds"] + result["system_seconds"], 0)
        self.assertEqual(result["budget_acceptance"], "not_evaluated")

    def test_input_and_large_outputs_are_retained_exactly(self):
        incoming = self.root / "input"
        incoming.write_bytes("输入\n".encode())
        program = f'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read() + b"z" * {OUTPUT_BYTES}); sys.stderr.buffer.write(b"diagnostic\\n")'
        completed = self.invoke("io", program, input_path=incoming)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        expected = incoming.read_bytes() + b"z" * OUTPUT_BYTES
        self.assertEqual(Path(result["stdout"]["path"]).read_bytes(), expected)
        self.assertEqual(result["stdout"]["bytes"], len(expected))
        self.assertEqual(result["stdout"]["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(result["stdin"]["sha256"], hashlib.sha256(incoming.read_bytes()).hexdigest())
        self.assertEqual(Path(result["stderr"]["path"]).read_bytes(), b"diagnostic\n")
        self.assertEqual(json.loads((self.root / "io/resources.json").read_text()), result)

    def test_nonzero_exit_and_signal_are_not_hidden(self):
        failed = self.invoke("failure", 'import sys; print("partial"); sys.exit(7)')
        self.assertEqual(failed.returncode, 7)
        self.assertEqual(json.loads(failed.stdout)["returncode"], 7)
        killed = self.invoke("signal", 'import os,signal; os.kill(os.getpid(),signal.SIGTERM)')
        self.assertEqual(killed.returncode, resources.SIGNAL_EXIT_BASE + signal.SIGTERM)
        result = json.loads(killed.stdout)
        self.assertEqual(result["returncode"], -signal.SIGTERM)
        self.assertEqual(result["signal"], signal.SIGTERM)

    def test_existing_output_and_missing_program_cannot_produce_success(self):
        existing = self.root / "existing"
        existing.mkdir()
        marker = self.root / "should-not-exist"
        program = f'from pathlib import Path; Path({str(marker)!r}).write_text("executed")'
        completed = self.invoke("existing", program)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(marker.exists())
        with self.assertRaises(FileNotFoundError):
            resources.execute([str(self.root / "missing-program")], directory=self.root / "missing")
        self.assertFalse((self.root / "missing/resources.json").exists())

    def test_os_units_are_explicit_and_unknown_units_fail(self):
        self.assertEqual(resources.rss_unit("darwin"), ("bytes", 1))
        self.assertEqual(resources.rss_unit("linux"), ("KiB", 1024))
        with self.assertRaisesRegex(ValueError, "not defined"):
            resources.rss_unit("unsupported")
        with self.assertRaisesRegex(ValueError, "required"):
            resources.execute([], directory=self.root / "empty")

    def test_each_waited_child_has_its_own_peak(self):
        large = resources.execute([sys.executable, "-B", "-c", f'payload = bytearray(b"x") * {ALLOCATION_BYTES}'],
                                  directory=self.root / "large")
        small = resources.execute([sys.executable, "-B", "-c", "pass"], directory=self.root / "small")
        self.assertGreater(large["peak_rss"]["bytes"], small["peak_rss"]["bytes"])
        self.assertNotEqual(large["pid"], small["pid"])


if __name__ == "__main__":
    unittest.main()

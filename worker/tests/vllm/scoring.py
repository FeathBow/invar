import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
CHILD_TIMEOUT_SECONDS = 20


def invoke(arguments):
    return subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=CHILD_TIMEOUT_SECONDS)


class ScoringEntryTests(unittest.TestCase):
    def test_import_does_not_initialize_the_native_logger(self):
        result = invoke(["-c", "import json,sys; import worker.vllm.scoring; print(json.dumps({'native_imported':'vllm' in sys.modules}))"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"native_imported": False})

    def test_help_does_not_require_native_engine_initialization(self):
        result = invoke([str(ROOT / "entries/vllmscore.py"), "--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)
        self.assertIn("--adapter", result.stdout)


if __name__ == "__main__":
    unittest.main()

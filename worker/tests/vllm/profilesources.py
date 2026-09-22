import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from worker.vllm.profile import declared_sources

ROOT = Path(__file__).resolve().parents[3]
TARGET = "worker.probepacked"
CHILD_SECONDS = 20


class ProfileSources(unittest.TestCase):
    def test_independent_interpreter_resolves_the_declared_source(self):
        from worker import probepacked

        code = (
            "import hashlib,json,sys; from pathlib import Path; "
            "from worker.vllm.profile import declared_sources; "
            "name='worker.probepacked'; before=name in sys.modules; "
            "source=declared_sources((name,))[name]; "
            "print(json.dumps({'before':before,'same':source is sys.modules[name],"
            "'sha256':hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()}))"
        )
        for _ in range(2):
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                    capture_output=True, text=True, timeout=CHILD_SECONDS)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {
                "before": False, "same": True,
                "sha256": hashlib.sha256(Path(probepacked.__file__).read_bytes()).hexdigest(),
            })

    def test_existing_module_identity_is_preserved(self):
        from worker import probepacked

        self.assertIs(declared_sources((TARGET,))[TARGET], probepacked)

    def test_unavailable_declared_source_remains_an_explicit_error(self):
        with self.assertRaises(ModuleNotFoundError):
            declared_sources(("worker.invar_missing_declared_source",))


if __name__ == "__main__":
    unittest.main()

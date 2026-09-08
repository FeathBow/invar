import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

VALUE = 1.0
NEGATIVE_ZERO = -0.0
TIMEOUT_SECONDS = 20


class ArtifactReaderTests(unittest.TestCase):
    def test_actual_artifact_validation_does_not_import_model_loaders(self):
        directory = Path(tempfile.mkdtemp(prefix="invar-artifact-reader-"))
        path = directory / "adapter.safetensors"
        save_file({"adapter": torch.tensor([VALUE, NEGATIVE_ZERO])}, path)
        metadata = json.dumps(["adapter", "torch.float32", [2]]).encode()
        expected = hashlib.sha256(metadata + struct.pack("<ff", VALUE, NEGATIVE_ZERO)).hexdigest()
        program = """
import sys
import torch
from policy import read_adapter
from states import validate_rng
state = read_adapter(sys.argv[1], sys.argv[2])
validate_rng(torch.get_rng_state())
assert list(state) == ["adapter"]
loaded = sorted(set(sys.modules) & {"peft", "transformers", "bitsandbytes", "accelerate"})
assert not loaded, f"Artifact validation imported model loaders: {loaded}"
"""
        result = subprocess.run([sys.executable, "-B", "-c", program, str(path), expected],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                                check=False, timeout=TIMEOUT_SECONDS)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

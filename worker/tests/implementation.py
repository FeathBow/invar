import ast
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import implementation

ENTRY_POINTS = ("initialize", "infer", "session", "step", "probe", "experiment")
TIMEOUT_SECONDS = 60
TEST_THREADS = 2
ADAPTER_SHIFT = 1


def local_imports(path, available):
    nodes = ast.walk(ast.parse(path.read_text()))
    names = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names & available


def copied_worker():
    original = Path(__file__).resolve().parents[1]
    copied = Path(tempfile.mkdtemp(prefix="invar-implementation-"))
    for path in original.glob("*.py"):
        shutil.copyfile(path, copied / path.name)
    shutil.copytree(original / "tests", copied / "tests")
    return copied


def exercise_boundaries():
    import torch
    import assembly
    import frozen
    from learning import update
    from .learning import batch
    from policy import activate
    from probe import adapter_state, assert_equal, restore
    from step import checkpoint_update
    from .step import prepared
    from .successor import observed

    torch.set_num_threads(TEST_THREADS)
    check = unittest.TestCase()
    learner, _, request, options = prepared()
    state = adapter_state(learner.model)
    base, binding = frozen.digest(learner.model), assembly.digest(learner.model)
    activate(learner.model, state, base=base, assembly=binding)
    summary = update(learner, batch()).summary
    path = Path(__file__).resolve().parents[1] / "objective.py"
    path.write_bytes(path.read_bytes() + b"\n# Changed implementation artifact.\n")
    check.assertNotEqual(binding, assembly.digest(learner.model))
    before = observed(learner, options.tokenizer)
    shifted = {name: value + ADAPTER_SHIFT for name, value in state.items()}
    with check.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
        activate(learner.model, shifted, base=base, assembly=binding)
    assert_equal(before, observed(learner, options.tokenizer))
    with check.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
        restore(learner.model, learner.optimizer, options.checkpoint, tokenizer=options.tokenizer)
    assert_equal(before, observed(learner, options.tokenizer))
    output = Path(tempfile.mkdtemp(prefix="invar-implementation-successor-"))
    with check.assertRaisesRegex(RuntimeError, "Checkpoint identity differs"):
        checkpoint_update(learner, request, output, tokenizer=options.tokenizer, summary=summary)
    check.assertEqual(list(output.iterdir()), [])
    assert_equal(before, observed(learner, options.tokenizer))


class ImplementationTests(unittest.TestCase):
    def test_inventory_closes_actual_worker_imports(self):
        root = Path(__file__).resolve().parents[1]
        available = {path.stem for path in root.glob("*.py")}
        pending, visited = set(ENTRY_POINTS), set()
        while pending:
            name = pending.pop()
            visited.add(name)
            pending.update(local_imports(root / f"{name}.py", available) - visited)
        self.assertEqual({f"{name}.py" for name in visited}, set(implementation.SOURCE_FILES))
        self.assertEqual(len(implementation.SOURCE_FILES), len(visited))

    def test_actual_bytes_are_bound_independently_of_directory(self):
        root = copied_worker()
        description = implementation.description(root)
        self.assertEqual(description, implementation.current())
        expected = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in implementation.SOURCE_FILES}
        self.assertEqual(description["files"], expected)
        path = root / "rollout.py"
        path.write_bytes(path.read_bytes() + b"\n# Different source bytes.\n")
        actual = implementation.description(root)
        self.assertEqual({name for name in expected if actual["files"][name] != expected[name]}, {"rollout.py"})
        self.assertEqual(description, implementation.current())

    def test_missing_source_is_an_error(self):
        root = Path(tempfile.mkdtemp(prefix="invar-implementation-missing-"))
        with self.assertRaises(FileNotFoundError):
            implementation.description(root)

    def test_changed_source_prevents_actual_activation_restore_and_successor(self):
        root = copied_worker()
        result = subprocess.run([sys.executable, "-B", "-m", "tests.implementation", "--exercise-copy"],
                                cwd=root, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["--exercise-copy"]:
        exercise_boundaries()
    else:
        unittest.main()

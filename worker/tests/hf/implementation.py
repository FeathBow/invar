import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import ast
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from worker import implementation
from worker.implementation import INFERENCE, LEARNING

ENTRY_POINTS = {implementation.INFERENCE: ("worker.hf.infer", "worker.hf.session", "worker.hf.cohort"),
                implementation.LEARNING: ("worker.hf.initialize", "worker.hf.step", "worker.hf.resident")}
TIMEOUT_SECONDS = 60
TEST_THREADS = 2
ADAPTER_SHIFT = 1


def local_imports(path, available):
    nodes = ast.walk(ast.parse(path.read_text()))
    names = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names & available


def package_modules(root):
    return {".".join(path.relative_to(root.parent).with_suffix("").parts): path
            for path in root.rglob("*.py") if "tests" not in path.relative_to(root).parts and path.stem != "__init__"}


def defined(path):
    return {node.name for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def referenced(paths):
    names = set()
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name.rsplit(".", 1)[-1])
    return names


def crossed(root, manifest, closures):
    used = {role: referenced(root / name for name in closures[role] - set(manifest.IRRELEVANT[role])) for role in closures}
    return {(role, name, symbol) for role, bound in manifest.ROLES.items() for name in bound for symbol in defined(root / name)
            if symbol not in used[role] and any(symbol in names for other, names in used.items() if other != role)}


def copied_worker():
    original = Path(__file__).resolve().parents[2]
    copied = Path(tempfile.mkdtemp(prefix="invar-implementation-")) / "worker"
    shutil.copytree(original, copied, ignore=shutil.ignore_patterns("__pycache__"))
    return copied


def exercise_boundaries():
    import torch
    from worker.hf import assembly
    from worker.hf import frozen
    from worker.hf.learning import update
    from worker.tests.hf.learning import batch
    from worker.hf.policy import activate
    from worker.hf.model import adapter_state
    from worker.hf.tensors import assert_equal
    from worker.hf.checkpoint import restore
    from worker.hf.step import checkpoint_update
    from worker.tests.hf.step import prepared
    from worker.tests.hf.successor import observed

    torch.set_num_threads(TEST_THREADS)
    check = unittest.TestCase()
    learner, _, request, options = prepared()
    state = adapter_state(learner.model)
    base, binding = frozen.digest(learner.model), assembly.digest(learner.model, LEARNING)
    inference = assembly.digest(learner.model, INFERENCE)
    activate(learner.model, state, base=base, assembly=binding, role=LEARNING)
    summary = update(learner, batch()).summary
    path = Path(__file__).resolve().parents[2] / "hf" / "learning.py"
    path.write_bytes(path.read_bytes() + b"\n# Changed implementation artifact.\n")
    check.assertNotEqual(binding, assembly.digest(learner.model, LEARNING))
    check.assertEqual(inference, assembly.digest(learner.model, INFERENCE))
    before = observed(learner, options.tokenizer)
    shifted = {name: value + ADAPTER_SHIFT for name, value in state.items()}
    with check.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
        activate(learner.model, shifted, base=base, assembly=binding, role=LEARNING)
    assert_equal(before, observed(learner, options.tokenizer))
    with check.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
        restore(learner.model, learner.optimizer, options.checkpoint, tokenizer=options.tokenizer)
    assert_equal(before, observed(learner, options.tokenizer))
    output = Path(tempfile.mkdtemp(prefix="invar-implementation-successor-"))
    with check.assertRaisesRegex(RuntimeError, "Checkpoint identity differs"):
        checkpoint_update(learner, request, output, tokenizer=options.tokenizer, summary=summary)
    check.assertEqual(list(output.iterdir()), [])
    assert_equal(before, observed(learner, options.tokenizer))


def closure(role):
    root = Path(__file__).resolve().parents[2]
    modules = package_modules(root)
    pending, visited = set(ENTRY_POINTS[role]), set()
    while pending:
        name = pending.pop()
        visited.add(name)
        pending.update(local_imports(modules[name], modules.keys()) - visited)
    return {str(modules[name].relative_to(root)) for name in visited}


def mutated(root, name):
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n# Different source bytes.\n")


class ImplementationTests(unittest.TestCase):
    def test_every_imported_module_is_bound_to_its_role_or_excluded_with_a_reason(self):
        for role, bound in implementation.ROLES.items():
            with self.subTest(role=role):
                excluded = set(implementation.NEUTRAL) | set(implementation.IRRELEVANT[role])
                self.assertEqual(closure(role) - excluded, set(bound))
                self.assertFalse(set(bound) & excluded)

    def test_bound_files_hold_no_code_used_only_by_the_other_role(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(crossed(root, implementation, {role: closure(role) for role in implementation.ROLES}), set())

    def test_learner_only_changes_keep_the_inference_identity(self):
        root = copied_worker()
        before = {role: implementation.description(root, role) for role in implementation.ROLES}
        learner = set(implementation.ROLES[implementation.LEARNING]) - set(implementation.ROLES[implementation.INFERENCE])
        for name in sorted(learner):
            mutated(root, name)
        self.assertEqual(implementation.description(root, implementation.INFERENCE), before[implementation.INFERENCE])
        self.assertNotEqual(implementation.description(root, implementation.LEARNING), before[implementation.LEARNING])

    def test_each_inference_dependency_changes_the_inference_identity(self):
        for name in implementation.ROLES[implementation.INFERENCE]:
            with self.subTest(name=name):
                root = copied_worker()
                before = implementation.description(root, implementation.INFERENCE)
                mutated(root, name)
                self.assertNotEqual(implementation.description(root, implementation.INFERENCE), before)

    def test_the_step_plan_mapping_is_part_of_the_learning_identity(self):
        root = copied_worker()
        before = implementation.description(root, LEARNING)
        mutated(root, "logical.py")
        self.assertNotEqual(implementation.description(root, LEARNING), before)

    def test_neutral_changes_keep_both_identities(self):
        root = copied_worker()
        before = {role: implementation.description(root, role) for role in implementation.ROLES}
        for name in implementation.NEUTRAL:
            mutated(root, name)
        self.assertEqual({role: implementation.description(root, role) for role in implementation.ROLES}, before)
        for role, irrelevant in implementation.IRRELEVANT.items():
            for name in irrelevant:
                mutated(root, name)
            self.assertEqual(implementation.description(root, role), before[role])

    def test_actual_bytes_are_bound_independently_of_directory(self):
        root = copied_worker()
        for role in implementation.ROLES:
            description = implementation.description(root, role)
            self.assertEqual(description, implementation.current(role))
            expected = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in implementation.ROLES[role]}
            self.assertEqual(description["files"], expected)

    def test_missing_source_is_an_error(self):
        root = Path(tempfile.mkdtemp(prefix="invar-implementation-missing-"))
        with self.assertRaises(FileNotFoundError):
            implementation.description(root, implementation.INFERENCE)

    def test_changed_source_prevents_actual_activation_restore_and_successor(self):
        root = copied_worker()
        result = subprocess.run([sys.executable, "-B", "-m", "worker.tests.hf.implementation", "--exercise-copy"],
                                cwd=root.parent, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["--exercise-copy"]:
        exercise_boundaries()
    else:
        unittest.main()

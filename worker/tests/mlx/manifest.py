from pathlib import Path
import unittest

from worker.implementation import INFERENCE, LEARNING
from worker.mlx import implementation
from worker.tests.hf.implementation import copied_worker, crossed, local_imports, mutated, package_modules

ENTRY_POINTS = {INFERENCE: ("worker.mlx.infer", "worker.mlx.batch", "worker.mlx.scoring", "worker.mlx.cohort"),
                LEARNING: ("worker.mlx.initialize", "worker.mlx.learner", "worker.mlx.step")}


def closure(role):
    root = Path(__file__).resolve().parents[2]
    modules = package_modules(root)
    pending, visited = set(ENTRY_POINTS[role]), set()
    while pending:
        name = pending.pop()
        visited.add(name)
        pending.update(local_imports(modules[name], modules.keys()) - visited)
    return {str(modules[name].relative_to(root)) for name in visited}


class ManifestTests(unittest.TestCase):
    def test_every_imported_module_is_bound_to_its_role_or_excluded_with_a_reason(self):
        for role, bound in implementation.ROLES.items():
            with self.subTest(role=role):
                excluded = set(implementation.NEUTRAL) | set(implementation.UNREACHED) | set(implementation.IRRELEVANT[role])
                self.assertEqual(closure(role) - excluded, set(bound))
                self.assertFalse(set(bound) & excluded)

    def test_bound_files_hold_no_code_used_only_by_the_other_role(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(crossed(root, implementation, {role: closure(role) for role in implementation.ROLES}), set())

    def test_learner_only_changes_keep_the_inference_identity(self):
        root = copied_worker()
        before = {role: implementation.description(root, role) for role in implementation.ROLES}
        for name in sorted(set(implementation.ROLES[LEARNING]) - set(implementation.ROLES[INFERENCE])):
            mutated(root, name)
        self.assertEqual(implementation.description(root, INFERENCE), before[INFERENCE])
        self.assertNotEqual(implementation.description(root, LEARNING), before[LEARNING])

    def test_each_inference_dependency_changes_the_inference_identity(self):
        for name in implementation.ROLES[INFERENCE]:
            with self.subTest(name=name):
                root = copied_worker()
                before = implementation.description(root, INFERENCE)
                mutated(root, name)
                self.assertNotEqual(implementation.description(root, INFERENCE), before)

    def test_the_step_plan_mapping_is_part_of_the_learning_identity(self):
        root = copied_worker()
        before = implementation.description(root, LEARNING)
        mutated(root, "logical.py")
        self.assertNotEqual(implementation.description(root, LEARNING), before)

    def test_neutral_and_irrelevant_changes_keep_the_identity(self):
        root = copied_worker()
        before = {role: implementation.description(root, role) for role in implementation.ROLES}
        for name in {**implementation.NEUTRAL, **implementation.UNREACHED}:
            mutated(root, name)
        self.assertEqual({role: implementation.description(root, role) for role in implementation.ROLES}, before)
        for name in implementation.IRRELEVANT[LEARNING]:
            mutated(root, name)
        self.assertEqual(implementation.description(root, LEARNING), before[LEARNING])


if __name__ == "__main__":
    unittest.main()

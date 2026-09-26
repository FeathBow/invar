from pathlib import Path
from types import SimpleNamespace
import unittest

from worker.tests.hf.implementation import local_imports, package_modules
from worker.vllm import profile

ENTRY_POINTS = ("worker.vllm.infer", "worker.vllm.session", "worker.vllm.entry", "worker.vllm.resident",
                "worker.vllm.scoring", "worker.vllm.inspect")


def closure():
    root = Path(__file__).resolve().parents[2]
    modules = package_modules(root)
    pending, visited = set(ENTRY_POINTS), set()
    while pending:
        name = pending.pop()
        visited.add(name)
        pending.update(local_imports(modules[name], modules.keys()) - visited)
    return {str(modules[name].relative_to(root)) for name in visited}


class ManifestTests(unittest.TestCase):
    def test_every_imported_worker_module_is_bound_or_excluded_with_a_reason(self):
        bound = {name.removeprefix("worker.").replace(".", "/") + ".py" for name in profile.NUMERICAL if name.startswith("worker.")}
        excluded = set(profile.NEUTRAL) | set(profile.UNREACHED)
        self.assertEqual(closure() - excluded, bound)
        self.assertFalse(bound & excluded)

    def test_the_source_learner_assembly_does_not_define_the_inference_identity(self):
        package = SimpleNamespace(configuration='{"r": 8}', source={"base": "b" * 64, "assembly": "a" * 64, "tokenizer": "t" * 64})
        relearned = SimpleNamespace(configuration=package.configuration, source={**package.source, "assembly": "c" * 64})
        rebased = SimpleNamespace(configuration=package.configuration, source={**package.source, "base": "d" * 64})
        self.assertEqual(profile.transformed(package, ()), profile.transformed(relearned, ()))
        self.assertNotEqual(profile.transformed(package, ()), profile.transformed(rebased, ()))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
    import vllm  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from safetensors.torch import save_file

from worker.hf import operation
from worker.hf.tensors import digest
from worker.tests.hf.tokenization import make_tokenizer
from worker.trajectory import Request
from worker.vllm.configuration import BASE_FORMAT, configuration
from worker.vllm.execution import delivered
from worker.vllm.mapping import UNADAPTED, selection
from worker.vllm.rollout import LORA_FIELDS
from worker.vllm.profile import Loaded
from worker.vllm.runtime import observed, unadapted

CONFIG = configuration({"format": BASE_FORMAT, "model": "invar/local-fixture", "revision": "1" * 40,
                        "engine": {"dtype": "bfloat16"}})
LOADED = Loaded(adapter_id=UNADAPTED, base="b" * 64, assembly="a" * 64)


class Engine:
    def __init__(self):
        self.calls = []

    def collective_rpc(self, method, **_):
        self.calls.append(method.__name__)
        return [{"model": {"adapter_id": LOADED.adapter_id, "base": LOADED.base, "assembly": LOADED.assembly}, "profile": {}}]


def adapter(directory, tensors):
    path = Path(directory) / "teacher.safetensors"
    save_file(tensors, path)
    return path


class UnadaptedTests(unittest.TestCase):
    def test_a_base_launch_loads_an_empty_adapter_file_as_the_empty_policy(self):
        tokenizer = make_tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = unadapted(Engine(), tokenizer, adapter=adapter(directory, {}), config=CONFIG)
        self.assertIsNone(runtime.lora)
        self.assertEqual(dict(runtime.identities), {"adapter": digest({}), "tokenizer": operation.digest(tokenizer),
                                                    "base": LOADED.base, "assembly": LOADED.assembly})

    def test_a_base_launch_refuses_an_adapter_file_with_tensors(self):
        engine = Engine()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "no tensors"):
            unadapted(engine, make_tokenizer(), adapter=adapter(directory, {"a": torch.zeros(1)}), config=CONFIG)
        self.assertEqual(engine.calls, [])

    def test_a_base_execution_reports_the_empty_policy_only_without_an_adapter(self):
        runtime = SimpleNamespace(lora=None)
        actual = observed(SimpleNamespace(resident=((),), models=((LOADED,),)), runtime, tokenizer="t" * 64)
        self.assertEqual(actual, {"adapter": digest({}), "tokenizer": "t" * 64, "base": LOADED.base, "assembly": LOADED.assembly})
        adapted = Loaded(adapter_id=1, base=LOADED.base, assembly=LOADED.assembly)
        for prepared in (SimpleNamespace(resident=((object(),),), models=((LOADED,),)),
                         SimpleNamespace(resident=((),), models=((adapted,),))):
            with self.subTest(prepared=prepared), self.assertRaisesRegex(ValueError, "adapter selection"):
                observed(prepared, runtime, tokenizer="t" * 64)


    def test_a_base_generation_delivers_only_an_output_without_an_adapter(self):
        tokenizer = make_tokenizer()
        eos = tokenizer.eos_token_id
        request = Request(sample="s0", group="g", prompt="p", seed=17, limit=2, temperature=0.8)
        completion = SimpleNamespace(index=0, token_ids=[3, eos], finish_reason="stop", stop_reason=eos,
                                     logprobs=[{token: SimpleNamespace(logprob=-0.5)} for token in (3, eos)])
        bound = SimpleNamespace(selection=selection(None), request=request, prompt=(2,))

        def output(lora):
            return SimpleNamespace(finished=True, outputs=[completion], prompt_token_ids=[2], lora_request=lora)

        actual = delivered(output(None), tokenizer, bound, vocabulary=len(tokenizer))
        self.assertEqual(actual.tokens.tolist(), [[2, 3, eos]])
        with self.assertRaisesRegex(ValueError, "LoRA selection"):
            delivered(output(SimpleNamespace(**{name: 1 for name in LORA_FIELDS})), tokenizer, bound, vocabulary=len(tokenizer))


if __name__ == "__main__":
    unittest.main()

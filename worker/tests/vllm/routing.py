from types import SimpleNamespace
import unittest

try:
    import torch
    import vllm  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from worker.trajectory import Request
from worker.vllm.mapping import NO_ADAPTER, UNADAPTED, Adapted, Row, Selection, Unadapted, selection
from worker.vllm.state import Binding

BASE = Selection(adapter=UNADAPTED, description=None)


def runner(*, lora_config=None, lora_request=None, mapped=UNADAPTED, wrapped=False):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    if wrapped:
        model[0].punica_wrapper = object()
    return SimpleNamespace(lora_config=lora_config, get_model=lambda: model, device=torch.device("cpu"),
                           requests={"r0": SimpleNamespace(req_id="r0", lora_request=lora_request)},
                           input_batch=SimpleNamespace(req_ids=["r0"], num_reqs=1, request_lora_mapping=[mapped]),
                           num_scheduled_tokens=SimpleNamespace(cpu=torch.tensor([3])),
                           query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 3], dtype=torch.int32)))


def bound(chosen):
    request = Request(sample="s0", group="test/0", prompt="p", seed=17, limit=8, temperature=0.8)
    return Binding(internal="r0", external="e0", request=request, prompt=(1, 2, 3), eos=4, selection=chosen, receipt=None)


class RoutingTests(unittest.TestCase):
    def test_a_request_without_a_lora_selects_the_base(self):
        self.assertEqual(selection(None), BASE)

    def test_a_base_engine_schedules_rows_without_adapters(self):
        routing = Unadapted(runner())
        self.assertEqual(routing.resident(), ())
        self.assertEqual(routing.rows({"r0": BASE}, token_count=3), (Row(request="r0", adapter=UNADAPTED, slot=NO_ADAPTER, tokens=3),))

    def test_a_base_engine_refuses_lora_layers(self):
        for engine in (runner(lora_config=object()), runner(wrapped=True)):
            with self.subTest(engine=engine), self.assertRaisesRegex(ValueError, "without LoRA"):
                Unadapted(engine)

    def test_a_base_engine_refuses_a_request_routed_to_a_lora(self):
        cases = ((runner(lora_request=object()), BASE), (runner(mapped=1), BASE),
                 (runner(), Selection(adapter=1, description="d")))
        for engine, chosen in cases:
            with self.subTest(chosen=chosen), self.assertRaisesRegex(ValueError, "LoRA"):
                Unadapted(engine).rows({"r0": chosen}, token_count=3)

    def test_a_lora_engine_refuses_a_request_without_its_adapter(self):
        native = SimpleNamespace(model=None, lora_index_to_id=[1, None])
        with self.assertRaisesRegex(ValueError, "prepared adapter"):
            Adapted(runner(), native, bindings=(bound(BASE),), verify=None)


if __name__ == "__main__":
    unittest.main()

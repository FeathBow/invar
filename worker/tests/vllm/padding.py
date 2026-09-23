import unittest

try:
    import torch
    import vllm  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
    raise unittest.SkipTest("the physical padding check requires the selected SM90 device")

from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

from worker.vllm.mapping import kernel

PHYSICAL_ROWS = 128
HIDDEN = 67
RANK = 8
WIDTHS = (65, 71)
ADAPTERS = 2
BASE = 0.125
LIVE_COUNTS = (73, 9, 127, 128, 33)


def fixture():
    generator = torch.Generator().manual_seed(20260914)
    x = torch.randn((PHYSICAL_ROWS, HIDDEN), generator=generator).bfloat16().cuda()
    a = [torch.randn((ADAPTERS, 1, RANK, HIDDEN), generator=generator).cuda() for _ in WIDTHS]
    b = [torch.randn((ADAPTERS, 1, width, RANK), generator=generator).cuda() for width in WIDTHS]
    middle = torch.empty((len(WIDTHS), PHYSICAL_ROWS, RANK), device="cuda")
    output = torch.empty((PHYSICAL_ROWS, sum(WIDTHS)), dtype=torch.bfloat16, device="cuda")
    return x, a, b, middle, output


def prepare(meta, count):
    meta.token_lora_mapping.fill_(ADAPTERS - 1)
    meta.token_indices_sorted_by_lora_ids.fill_(PHYSICAL_ROWS - 1)
    slots = torch.arange(count, device="cuda") % ADAPTERS
    meta.prepare_tensors(slots)
    return slots


def execute(fixture, meta):
    x, a, b, middle, output = fixture
    output.fill_(BASE)
    routing = meta.meta_args(x.shape[0], specialize_active_lora=True)
    lora_shrink(x, a, middle, *routing, 1.0)
    lora_expand(middle, b, output, *routing, offset_start=0, add_inputs=True)


class PaddingTests(unittest.TestCase):
    def check_outputs(self, full, slots):
        count = slots.numel()
        x, a, b, middle, output = full
        reference = (x[:count].contiguous(), a, b,
                     torch.empty((len(WIDTHS), count, RANK), device="cuda"),
                     torch.empty((count, sum(WIDTHS)), dtype=torch.bfloat16, device="cuda"))
        meta = LoRAKernelMeta.make(ADAPTERS, count, "cuda", [1, 2])
        meta.prepare_tensors(slots)
        execute(reference, meta)
        self.assertTrue(torch.equal(middle[:, :count], reference[3]), "Live shrink rows changed")
        self.assertTrue(torch.equal(output[:count], reference[4]), "Live expand rows changed")
        self.assertEqual(torch.count_nonzero(middle[:, count:]).item(), 0, "Padding shrink rows were written")
        self.assertTrue(torch.equal(output[count:], torch.full_like(output[count:], BASE)), "Padding output was written")

    def test_eager_padding(self):
        full = fixture()
        meta = LoRAKernelMeta.make(ADAPTERS, PHYSICAL_ROWS, "cuda", [1, 2])
        slots = prepare(meta, LIVE_COUNTS[0])
        execute(full, meta)
        self.check_outputs(full, slots)

    def test_graph_replays_use_current_live_counts(self):
        full = fixture()
        meta = LoRAKernelMeta.make(ADAPTERS, PHYSICAL_ROWS, "cuda", [1, 2])
        prepare(meta, LIVE_COUNTS[0])
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            execute(full, meta)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute(full, meta)
        for count in LIVE_COUNTS:
            with self.subTest(live_tokens=count):
                slots = prepare(meta, count)
                graph.replay()
                self.check_outputs(full, slots)

    def test_observer_checks_live_routes_and_all_consumed_counts(self):
        meta = LoRAKernelMeta.make(ADAPTERS, PHYSICAL_ROWS, "cuda", [1, 2])
        slots = tuple(prepare(meta, LIVE_COUNTS[0]).cpu().tolist())
        options = dict(max_loras=ADAPTERS, specialized=True, device="cuda", token_count=PHYSICAL_ROWS)
        kernel(meta, slots, **options)
        for name in ("token_lora_mapping", "token_indices_sorted_by_lora_ids", "num_tokens_per_lora",
                     "lora_token_start_loc", "active_lora_ids", "no_lora_flag_cpu", "num_active_loras_cpu"):
            with self.subTest(field=name):
                value = getattr(meta, name)
                saved = value.clone()
                value[0] = not value[0].item() if value.dtype == torch.bool else value[0] + 1
                with self.assertRaises(ValueError):
                    kernel(meta, slots, **options)
                value.copy_(saved)



if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

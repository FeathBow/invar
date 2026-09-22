import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer

from worker.vllm.lora import layer_settings, qwen_mlp_targets
from worker.tests.hf.decoding import hybrid


class NativeLoRASettingsTests(unittest.TestCase):
    def test_resolved_rank_alpha_and_rslora_match_actual_peft_layers(self):
        targets = qwen_mlp_targets(layers=2, source_prefix="model.language_model.layers",
                                   runtime_prefix="language_model.model.layers")
        names = [target.source.removeprefix("base_model.model.") for target in targets]
        for stabilized in (False, True):
            with self.subTest(rslora=stabilized):
                base = hybrid().unload()
                configured = LoraConfig(r=8, lora_alpha=16, target_modules=names, use_rslora=stabilized,
                                         rank_pattern={"layers.0.mlp.gate_proj": 4},
                                         alpha_pattern={"layers.0.mlp.gate_proj": 6,
                                                        "layers.1.mlp.down_proj": 7})
                model = get_peft_model(base, configured)
                for target in targets:
                    actual = model.get_submodule(target.source)
                    self.assertIsInstance(actual, LoraLayer)
                    self.assertEqual(layer_settings(configured.to_dict(), target),
                                     (actual.r["default"], actual.lora_alpha["default"], actual.scaling["default"]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

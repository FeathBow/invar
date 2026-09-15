import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import check_target_module_exists
from peft.utils.other import get_pattern_key
from safetensors.torch import load_file, save_file

from worker.hf import assembly
from worker.hf import operation
from worker.implementation import file_digest
from worker.hf.artifact import CONFIG, FORMAT, WEIGHTS
from worker.hf.policy import validate_state, verify
from worker.hf.probe import CheckpointIdentity, adapter_state, assert_equal

ADAPTER = "default"


def configuration(model):
    if set(model.peft_config) != {ADAPTER}:
        raise ValueError("The current handoff requires the single default adapter")
    config = model.peft_config[ADAPTER]
    modules = dict(model.get_base_model().named_modules())
    declared = {name for name in modules if check_target_module_exists(config, name)}
    actual = {name for name, layer in modules.items() if isinstance(layer, LoraLayer)}
    if not actual or declared != actual:
        raise ValueError("PEFT configuration differs from actual LoRA targets")
    for name, layer in modules.items():
        if not isinstance(layer, LoraLayer):
            continue
        if layer.disable_adapters or layer.merged_adapters or layer.active_adapters != [ADAPTER]:
            raise ValueError(f"Handoff requires an active, unmerged adapter: {name}")
        rank = config.rank_pattern.get(get_pattern_key(config.rank_pattern, name), config.r)
        alpha = config.alpha_pattern.get(get_pattern_key(config.alpha_pattern, name), config.lora_alpha)
        scaling = alpha / (math.sqrt(rank) if config.use_rslora else rank)
        if (layer.r[ADAPTER], layer.lora_alpha[ADAPTER], layer.scaling[ADAPTER]) != (rank, alpha, scaling):
            raise ValueError(f"PEFT configuration differs from actual LoRA scaling: {name}")
    return json.dumps(config.to_dict(), default=assembly.encode,
                      sort_keys=True, indent=2, allow_nan=False) + "\n"


def export(model, output, *, tokenizer, expected):
    operation.verify(tokenizer, expected.tokenizer)
    verify(model, expected.adapter, base=expected.base, assembly=expected.assembly)
    configured = configuration(model)
    state = adapter_state(model)
    validate_state(state, expected.adapter)
    output.mkdir(exist_ok=False)
    save_file(state, output / WEIGHTS)
    with (output / CONFIG).open("x") as stream:
        stream.write(configured)
    assert_equal(state, load_file(output / WEIGHTS))
    if (output / CONFIG).read_text() != configured:
        raise RuntimeError("Exported PEFT configuration differs from the source")
    receipt = {"format": FORMAT, "source": asdict(expected),
               "producer": file_digest(Path(__file__).resolve()),
               "files": {name: file_digest(output / name) for name in (WEIGHTS, CONFIG)},
               "scope": "Unmerged FP32 adapter export; not publication or engine qualification"}
    with (output / "handoff.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("adapter", "base", "assembly", "tokenizer"):
        parser.add_argument(f"--{name}-digest", required=True)
    return parser.parse_args()


def main():
    from worker.hf.inference import load

    options = arguments()
    expected = CheckpointIdentity(adapter=options.adapter_digest, base=options.base_digest,
                                  assembly=options.assembly_digest, tokenizer=options.tokenizer_digest)
    runtime = load(options.cache, options.adapter, expected=asdict(expected))
    receipt = export(runtime.model, options.output, tokenizer=runtime.tokenizer, expected=expected)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

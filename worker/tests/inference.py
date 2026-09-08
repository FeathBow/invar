import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

import assembly
import frozen
import session
import inference
import invocation
import registry
import tensors
import operation
from policy import read_adapter
from probe import adapter_state
from .tokenization import WORDS, make_tokenizer

MODEL_SEED = 41
HIDDEN_SIZE = 16
INTERMEDIATE_SIZE = 32
TEST_THREADS = 2
TOKEN_LIMIT = 4
IDENTITY = ("local-test/Llama-tiny", "deterministic-cpu-fixture")


def model():
    with torch.random.fork_rng():
        torch.manual_seed(MODEL_SEED)
        config = LlamaConfig(vocab_size=len(WORDS), hidden_size=HIDDEN_SIZE,
                             intermediate_size=INTERMEDIATE_SIZE, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=2)
        base = LlamaForCausalLM(config)
        result = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"]))
    return result.eval()


def fixture():
    loaded = model()
    tokenizer = make_tokenizer()
    adapter = Path(tempfile.mkdtemp(prefix="invar-batch-")) / "adapter.safetensors"
    state = adapter_state(loaded)
    save_file(state, adapter)
    identities = {"adapter": tensors.digest(state), "base": frozen.digest(loaded),
                  "assembly": assembly.digest(loaded), "tokenizer": operation.digest(tokenizer)}
    runtime = inference.Runtime(model=loaded, tokenizer=tokenizer, adapter=adapter, device="cpu", identity=IDENTITY)
    return runtime, identities


def envelope(identities, index):
    loading = {'binding': {name: index for name in ('call', 'attempt', 'instance')},
               'program': 'explicit load protocol fixture; CPU numerical test'}
    return {**identities, "binding": {name: index for name in ("call", "attempt", "instance")},
            'load': loading,
            "program": "explicit protocol fixture; CPU numerical test",
            "request": {"prompt": "Compute the answer.", "tokens": TOKEN_LIMIT,
                        "temperature": 0.8, "seed": MODEL_SEED + index}}


def stream(requests):
    lines = []
    for request in requests:
        lines.extend([request, {key: request[key] for key in ("binding", "program")}])
    return io.StringIO("".join(json.dumps(line) + "\n" for line in lines))


def measured(stage, action):
    started = time.perf_counter()
    result = action()
    print(json.dumps({"stage": stage, "cpu_seconds": time.perf_counter() - started}))
    return result


def run(runtime, source, *, loader=None):
    def loaded(cache, adapter, *, expected):
        read_adapter(adapter, expected["adapter"])
        return measured("load", lambda: runtime)

    options = SimpleNamespace(cache=runtime.adapter.parent, adapter=runtime.adapter)
    session.serve(options, source=source, loader=loaded if loader is None else loader,
                      execute=partial(inference.execute, measure=measured),
                      permission=partial(invocation.approve, source=source))


def records(output, stage):
    return [value for line in output.getvalue().splitlines()
            if (value := json.loads(line))["stage"] == stage]


class InferenceTests(unittest.TestCase):
    def test_reused_real_model_matches_fresh_execution_exactly(self):
        runtime, identities = fixture()
        requests = [envelope(identities, index) for index in (0, 1, 2)]
        output = io.StringIO()
        loads = []

        def loader(cache, adapter, *, expected):
            read_adapter(adapter, expected["adapter"])
            loaded = measured("load", model)
            loads.append(loaded)
            return inference.Runtime(model=loaded, tokenizer=make_tokenizer(), adapter=adapter,
                                     device="cpu", identity=IDENTITY)

        with redirect_stdout(output):
            run(runtime, stream(requests), loader=loader)
        self.assertEqual(len(loads), 1)
        self.assertEqual(len(records(output, "load")), 1)
        self.assertEqual(records(output, 'unloaded_adapter'),
                         [{'stage': 'unloaded_adapter', **request['load']} for request in requests[:-1]])
        loaded = records(output, 'loaded_adapter')
        self.assertEqual([event['load'] for event in loaded], [request['load'] for request in requests])
        self.assertEqual([event['image'] for event in loaded], [registry.image(identities)] * len(requests))
        actual = records(output, "result")
        self.assertEqual(len(actual), len(requests))
        for request, result in zip(requests, actual, strict=True):
            fresh = inference.Runtime(model=model(), tokenizer=make_tokenizer(), adapter=runtime.adapter,
                                      device="cpu", identity=IDENTITY)
            independent = io.StringIO()
            with redirect_stdout(independent):
                run(fresh, stream([request]))
            self.assertEqual(records(independent, "result"), [result])
            self.assertTrue(result["behavior_bits"])

    def test_permission_mismatch_prevents_generation_and_later_dispatch(self):
        runtime, identities = fixture()
        source = stream([envelope(identities, index) for index in (0, 1, 2)])
        lines = source.getvalue().splitlines()
        wrong = json.loads(lines[3])
        lines[3] = json.dumps({**wrong, "program": "different"})
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, "permission differs"):
            run(runtime, io.StringIO("\n".join(lines) + "\n"))
        self.assertEqual(len(records(output, "consumed")), 2)
        self.assertEqual(len(records(output, "result")), 1)

    def test_each_request_rechecks_actual_frozen_tensors(self):
        runtime, identities = fixture()
        source = stream([envelope(identities, index) for index in (0, 1)])
        output = io.StringIO()

        def permission(bound):
            invocation.approve(bound, source=source)
            with torch.no_grad():
                next(value for value in runtime.model.parameters() if not value.requires_grad).add_(1)

        options = SimpleNamespace(cache=runtime.adapter.parent, adapter=runtime.adapter)
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "frozen.*binding mismatch"):
            session.serve(options, source=source,
                              loader=lambda cache, adapter, **keywords: runtime,
                              execute=partial(inference.execute, measure=measured), permission=permission)
        self.assertEqual(len(records(output, "consumed")), 1)
        self.assertEqual(len(records(output, "result")), 1)

    def test_duplicate_identities_stop_before_second_consumption(self):
        runtime, identities = fixture()
        for key in ("call", "attempt", "instance"):
            first, second = [envelope(identities, index) for index in (0, 1)]
            second["binding"][key] = first["binding"][key]
            second['load']['binding'][key] = first['load']['binding'][key]
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaisesRegex(ValueError, "reuses"):
                run(runtime, stream([first, second]))
            self.assertEqual(len(records(output, "result")), 1)

    def test_changed_adapter_file_prevents_second_consumption(self):
        runtime, identities = fixture()
        source = stream([envelope(identities, index) for index in (0, 1)])
        output = io.StringIO()

        def permission(bound):
            invocation.approve(bound, source=source)
            changed = {name: value + 1 for name, value in adapter_state(runtime.model).items()}
            save_file(changed, runtime.adapter)

        options = SimpleNamespace(cache=runtime.adapter.parent, adapter=runtime.adapter)
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, "contents do not match"):
            session.serve(options, source=source,
                              loader=lambda cache, adapter, **keywords: runtime,
                              execute=partial(inference.execute, measure=measured), permission=permission)
        self.assertEqual(len(records(output, "consumed")), 1)
        self.assertEqual(len(records(output, "result")), 1)

    def test_first_adapter_mismatch_prevents_actual_model_loading(self):
        runtime, identities = fixture()
        with self.assertRaisesRegex(ValueError, "contents do not match"):
            inference.load(runtime.adapter.parent, runtime.adapter,
                           expected={**identities, "adapter": "0" * 64})

    def test_malformed_request_fails_before_loader(self):
        runtime, identities = fixture()
        value = envelope(identities, 0)

        def forbidden(cache, adapter, *, expected):
            self.fail("A malformed envelope reached the model loader")

        for change in ({"tokens": True}, {"temperature": float("nan")}, {"seed": False}):
            invalid = {**value, "request": {**value["request"], **change}}
            with self.assertRaises(ValueError):
                run(runtime, stream([invalid]), loader=forbidden)
        duplicate = json.dumps(value).replace('"tokens": 4', '"tokens": 4, "tokens": 4')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            run(runtime, io.StringIO(duplicate + "\n"), loader=forbidden)

    def test_load_envelope_must_match_inference_before_loading(self):
        runtime, identities = fixture()
        value = envelope(identities, 0)

        def forbidden(cache, adapter, *, expected):
            self.fail('An invalid load declaration reached the model loader')

        wrong_binding = {**value['load'], 'binding': {**value['binding'], 'instance': 1}}
        for loading in ({}, wrong_binding, {**value['load'], 'program': ''}):
            with self.assertRaises(ValueError):
                run(runtime, stream([{**value, 'load': loading}]), loader=forbidden)


if __name__ == "__main__":
    torch.set_num_threads(TEST_THREADS)
    unittest.main()

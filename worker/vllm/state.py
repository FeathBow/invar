from contextlib import ExitStack
from dataclasses import asdict, dataclass
from functools import partial
from itertools import accumulate
from types import MappingProxyType

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from worker.hf.rollout import Request
from worker.hf.tensors import assert_equal
from worker.vllm.mapping import Selection, observe
from worker.vllm.rollout import parameters

SAMPLING_FIELDS = ("n", "temperature", "seed", "max_tokens", "min_tokens", "top_k", "top_p", "min_p",
                   "presence_penalty", "frequency_penalty", "repetition_penalty", "stop", "stop_token_ids",
                   "ignore_eos", "logprobs", "prompt_logprobs", "detokenize", "output_kind")


@dataclass(frozen=True, kw_only=True)
class Binding:
    internal: str
    external: str
    request: Request
    prompt: tuple[int, ...]
    eos: int
    selection: Selection
    receipt: str


@dataclass(frozen=True, kw_only=True)
class Step:
    rows: tuple
    model_tokens: int
    logits_indices: tuple[int, ...]


def binding(value):
    return Binding(**{**value, "request": Request(**value["request"]),
                       "selection": Selection(**value["selection"]), "prompt": tuple(value["prompt"])})


def sampling(state, bound):
    expected = parameters(bound.request, eos=bound.eos)
    if any(getattr(state.sampling_params, name) != getattr(expected, name) for name in SAMPLING_FIELDS):
        raise ValueError("Native cached sampling parameters differ from the approved request")
    if tuple(state.prompt_token_ids or ()) != bound.prompt:
        raise ValueError("Native cached prompt differs from the approved request")


def model_inputs(runner, step, *, bindings, tokens, arguments):
    expected_tokens, expected_positions, sampling_requests = [], [], []
    for row in step.rows:
        bound, state = bindings[row.request], runner.requests[row.request]
        sampling(state, bound)
        known = (*bound.prompt, *tokens[row.request])
        start, end = state.num_computed_tokens, state.num_computed_tokens + row.tokens
        if not 0 <= start < end <= len(known):
            raise ValueError("Scheduled model input extends outside the observed logical token sequence")
        expected_tokens.extend(known[start:end])
        expected_positions.extend(range(start, end))
        if end == len(known):
            sampling_requests.append(row.request)
    actual_tokens = arguments["input_ids"]
    if actual_tokens.shape != (step.model_tokens,):
        raise ValueError("Native model token IDs have a different physical shape")
    logical_tokens = len(expected_tokens)
    descriptor = get_forward_context().batch_descriptor
    if step.model_tokens != logical_tokens and (descriptor is None or descriptor.num_tokens != step.model_tokens):
        raise ValueError("Native model padding differs from its actual batch descriptor")
    assert_equal(actual_tokens[:logical_tokens], torch.tensor(expected_tokens, dtype=torch.int32))
    positions = arguments["positions"]
    if positions.ndim not in (1, 2) or positions.shape[-1] != step.model_tokens:
        raise ValueError("Native model positions have a different physical shape")
    expected = torch.tensor(expected_positions, dtype=torch.int64)
    if positions.ndim == 2:
        expected = expected.expand(positions.shape[0], -1)
    assert_equal(positions[..., :logical_tokens], expected)
    return tuple(sampling_requests)


def sampler_controls(metadata, temperatures):
    if not metadata.all_random or metadata.all_greedy or not metadata.no_penalties:
        raise ValueError("Native sampler changed the requested random sampling controls")
    if metadata.top_k is not None or metadata.top_p is not None or metadata.allowed_token_ids_mask is not None:
        raise ValueError("Native sampler installed a token filter outside the approved request")
    if metadata.bad_words_token_ids or metadata.max_num_logprobs != 0:
        raise ValueError("Native sampler changed the requested token or probability observations")
    assert_equal(metadata.temperature, torch.tensor(temperatures, dtype=torch.float32))


def observed_forward(inner, monitor, **arguments):
    monitor.before_model(arguments)
    output = inner(**arguments)
    monitor.after_model(output)
    return output


def before_logits(module, args, kwargs, *, monitor):
    hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
    monitor.before_logits(hidden)


def before_sampler(module, args, kwargs, *, monitor):
    monitor.before_sampler(kwargs["sampling_metadata"])


def after_sampler(module, args, output, *, monitor):
    monitor.after_sampler(output)


class Monitor:
    def __init__(self, runner, native, *, bindings, verify, observe_model, models=None):
        self.runner, self.native, self.verify = runner, native, verify
        self.observe_model = observe_model
        self.bindings = MappingProxyType({value.internal: value for value in bindings})
        if not self.bindings or len(self.bindings) != len(bindings):
            raise ValueError("Native execution requires distinct owned request identities")
        self.selections = MappingProxyType({key: value.selection for key, value in self.bindings.items()})
        self.policies = MappingProxyType({value.selection.adapter: value.receipt for value in bindings})
        if any(self.policies[value.selection.adapter] != value.receipt for value in bindings):
            raise ValueError("One native adapter ID cannot bind different approved packages")
        self.tokens = {key: [] for key in self.bindings}
        self.behavior = {key: [] for key in self.bindings}
        self.generators_seen = set()
        self.steps = []
        self.pending = None
        self.sampling_requests = ()
        self.model_finished = False
        self.hidden_states = None
        self.logits_finished = False
        self.permitted = False
        self.hooks = ExitStack()
        self.resident = self.verify_policies()
        self.models = self.observe_model() if models is None else models
        self.slot_layout = tuple(native.lora_index_to_id)

    def verify_policies(self):
        return tuple(self.verify(receipt=receipt, adapter_id=adapter) for adapter, receipt in self.policies.items())

    def verify_model(self):
        if self.observe_model() != self.models:
            raise RuntimeError("Native base or assembly changed after it was observed")

    def allow(self):
        if self.permitted:
            raise RuntimeError("Native execution permission was already consumed")
        self.verify_policies()
        self.permitted = True

    def attach(self):
        with ExitStack() as hooks:
            processors = [module for module in self.native.model.modules() if isinstance(module, LogitsProcessor)]
            if len(processors) != 1:
                raise ValueError("Native model must expose one actual logits row consumer")
            handle = processors[0].register_forward_pre_hook(partial(before_logits, monitor=self), with_kwargs=True)
            hooks.callback(handle.remove)
            handle = self.runner.sampler.register_forward_pre_hook(partial(before_sampler, monitor=self), with_kwargs=True)
            hooks.callback(handle.remove)
            handle = self.runner.sampler.register_forward_hook(partial(after_sampler, monitor=self))
            hooks.callback(handle.remove)
            # Full CUDA graph replays never call the model module, so the step
            # is observed at the runner's own forward helper.
            self.runner._model_forward = partial(observed_forward, self.runner._model_forward, self)
            hooks.callback(delattr, self.runner, "_model_forward")
            self.hooks = hooks.pop_all()

    def before_model(self, arguments):
        if not self.permitted:
            raise RuntimeError("Native request execution has not been approved")
        if self.pending is not None:
            raise RuntimeError("Native model execution overlaps another owned step")
        if arguments["input_ids"] is None or arguments.get("inputs_embeds") is not None:
            raise ValueError("Approved native text requests require their actual token IDs")
        layout = tuple(self.native.lora_index_to_id)
        if layout != self.slot_layout:
            self.verify_policies()
            self.slot_layout = layout
        rows = observe(self.runner, self.native, expected=self.selections, token_count=arguments["input_ids"].numel())
        step = Step(rows=rows, model_tokens=arguments["input_ids"].numel(),
                    logits_indices=tuple(end - 1 for end in accumulate(row.tokens for row in rows)))
        self.sampling_requests = model_inputs(self.runner, step, bindings=self.bindings,
                                               tokens=self.tokens, arguments=arguments)
        self.pending = step
        self.model_finished = False
        self.logits_finished = False

    def after_model(self, output):
        if self.pending is None or not isinstance(output, torch.Tensor) or output.shape[0] != self.pending.model_tokens:
            raise ValueError("Native model output differs from its owned physical rows")
        self.hidden_states = output
        self.model_finished = True

    def before_logits(self, hidden):
        if self.pending is None or not self.model_finished or self.logits_finished:
            raise RuntimeError("Native logits have no unconsumed owned model step")
        indices = torch.tensor(self.pending.logits_indices, device=self.hidden_states.device)
        assert_equal(hidden, self.hidden_states[indices])
        self.hidden_states = None
        self.logits_finished = True

    def before_sampler(self, metadata):
        if self.pending is None or not self.model_finished or not self.logits_finished:
            raise RuntimeError("Native sampler has no completed owned model step")
        requests = tuple(row.request for row in self.pending.rows)
        if tuple(self.runner.input_batch.req_ids) != requests or set(metadata.generators) != set(range(len(requests))):
            raise ValueError("Native sampler rows differ from the observed model requests")
        sampler_controls(metadata, [self.bindings[key].request.temperature for key in requests])
        for index, key in enumerate(requests):
            self.generator(key, metadata.generators[index])

    def generator(self, key, generator):
        bound = self.bindings[key]
        if generator is not self.runner.requests[key].generator or generator.initial_seed() != bound.request.seed:
            raise ValueError("Native sampler uses a generator outside the approved request")
        if key not in self.generators_seen:
            initial = torch.Generator(device=generator.device).manual_seed(bound.request.seed)
            assert_equal(generator.get_state(), initial.get_state())
            self.generators_seen.add(key)

    def after_sampler(self, output):
        count = len(self.pending.rows)
        token_ids, logprobs = output.sampled_token_ids, output.logprobs_tensors.logprobs
        if token_ids.shape != (count, 1) or logprobs.shape != (count, 1) or logprobs.dtype != torch.float32:
            raise ValueError("Native sampler observations have a different row or probability representation")
        if not logprobs.isfinite().all() or (logprobs > 0).any():
            raise ValueError("Native sampler returned invalid behavior probabilities")
        for row, token, probability in zip(self.pending.rows, token_ids.flatten().tolist(), logprobs.flatten().tolist(), strict=True):
            if row.request in self.sampling_requests:
                self.tokens[row.request].append(token)
                self.behavior[row.request].append(probability)
        self.steps.append(asdict(self.pending))
        self.pending = None
        self.model_finished = False
        self.logits_finished = False

    def close(self, *, completed):
        self.hooks.close()
        if completed:
            if self.pending is not None or not all(self.tokens.values()):
                raise ValueError("Native execution did not complete all owned sampling observations")
            self.verify_policies()
            self.verify_model()
        return {"requests": {key: {"tokens": self.tokens[key], "behavior": self.behavior[key]}
                              for key in self.bindings}, "steps": self.steps}

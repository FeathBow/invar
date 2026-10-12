from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from types import MappingProxyType

import torch

from worker.vllm.rollout import lora_selection

NO_ADAPTER = -1
UNADAPTED = 0


@dataclass(frozen=True, kw_only=True)
class Selection:
    adapter: int
    description: str | None


def selection(lora):
    if lora is None:
        return Selection(adapter=UNADAPTED, description=None)
    if type(lora.lora_int_id) is not int or lora.lora_int_id <= 0:
        raise ValueError("Native execution requires a positive prepared LoRA identity")
    return Selection(adapter=lora.lora_int_id, description=lora_selection(lora))


@dataclass(frozen=True, kw_only=True)
class Row:
    request: str
    adapter: int
    slot: int
    tokens: int


def tensor(value, expected, *, dtype, device, label):
    if value.dtype != dtype or value.device.type != device or value.shape != (len(expected),):
        raise ValueError(f"Native LoRA routing tensor representation mismatch: {label}")
    if tuple(value.detach().cpu().tolist()) != tuple(expected):
        raise ValueError(f"Native LoRA routing tensor values mismatch: {label}")


def kernel_expected(slots, *, capacity):
    counts = Counter(slots)
    active = sorted(counts)
    padding = capacity - len(active)
    if not slots or padding < 0 or any(slot < 0 or slot >= capacity - 1 for slot in active):
        raise ValueError("Native LoRA routing requires a resident adapter for every token")
    sizes = [counts[slot] for slot in active]
    return (tuple(slots), tuple(sorted(range(len(slots)), key=lambda index: (slots[index], index))),
            (*sizes, *([0] * padding)), (0, *accumulate(sizes), *([0] * padding)),
            (*active, *([NO_ADAPTER] * padding)))


def kernel(meta, slots, *, max_loras, specialized, device, token_count):
    capacity = max_loras + 1
    expected = kernel_expected(slots, capacity=capacity)
    *routing, no_lora, num_active = meta.meta_args(token_count, specialized)
    labels = ("token slots", "sorted token indices", "tokens per slot", "slot starts", "active slots")
    for index, (value, wanted, label) in enumerate(zip(routing, expected, labels, strict=True)):
        if index < 2:
            if value.shape != (token_count,):
                raise ValueError(f"Native LoRA routing physical shape mismatch: {label}")
            # Exact counts and starts below bound every consumed sorted index.
            value = value[:len(slots)]
        tensor(value, wanted, dtype=torch.int32, device=device, label=label)
    active_count = len(set(slots))
    if specialized:
        candidates = [count for count in meta.captured_lora_counts if count >= active_count]
        count = min(candidates) if candidates else active_count
    else:
        count = capacity
    tensor(no_lora, (False,), dtype=torch.bool, device="cpu", label="no-LoRA flag")
    tensor(num_active, (count,), dtype=torch.int32, device="cpu", label="active slot count")


def slot(native, adapter):
    matches = [index for index, identity in enumerate(native.lora_index_to_id) if identity == adapter]
    if len(matches) != 1 or adapter not in native._active_adapters:
        raise ValueError("Scheduled LoRA must occupy exactly one active native slot")
    return matches[0]


def request_row(runner, native, *, request, index, count, selected):
    state = runner.requests[request]
    cached = runner.input_batch.lora_id_to_lora_request.get(selected.adapter)
    if lora_selection(state.lora_request) != selected.description or lora_selection(cached) != selected.description:
        raise ValueError("Scheduled native request selects a different LoRA")
    if state.req_id != request or int(runner.input_batch.request_lora_mapping[index]) != selected.adapter:
        raise ValueError("Native input batch maps the request to a different LoRA")
    if count <= 0:
        raise ValueError("Scheduled native request must contain model input tokens")
    return Row(request=request, adapter=selected.adapter, slot=slot(native, selected.adapter), tokens=count)


def unadapted_row(runner, *, request, index, count, selected):
    state = runner.requests[request]
    if selected.adapter != UNADAPTED or state.lora_request is not None:
        raise ValueError("Native base execution scheduled a request with a LoRA selection")
    if state.req_id != request or int(runner.input_batch.request_lora_mapping[index]) != UNADAPTED:
        raise ValueError("Native input batch maps a base request to a LoRA")
    if count <= 0:
        raise ValueError("Scheduled native request must contain model input tokens")
    return Row(request=request, adapter=UNADAPTED, slot=NO_ADAPTER, tokens=count)


def rows(runner, *, expected, row, device):
    requests = tuple(runner.input_batch.req_ids)
    if len(requests) != runner.input_batch.num_reqs or len(set(requests)) != len(requests):
        raise ValueError("Native input batch has repeated or inconsistent request identities")
    if not requests or not set(requests) <= expected.keys():
        raise ValueError("Native input batch contains a request outside this execution")
    counts = runner.num_scheduled_tokens.cpu[:len(requests)].tolist()
    result = tuple(row(request=request, index=index, count=count, selected=expected[request])
                   for index, (request, count) in enumerate(zip(requests, counts, strict=True)))
    starts = (0, *accumulate(counts))
    tensor(runner.query_start_loc.gpu[:len(starts)], starts, dtype=torch.int32,
           device=device, label="scheduled request starts")
    return result


def wrappers(native):
    result = {}
    for name, layer in native.modules.items():
        wrapper = layer.punica_wrapper
        if native.model.get_submodule(name) is not layer or native._get_punica_wrapper(name) is not wrapper:
            raise ValueError("Native LoRA module and manager do not use the same routing object")
        result[id(wrapper)] = wrapper
    if not result:
        raise ValueError("Native execution has no LoRA routing consumers")
    return tuple(result.values())


def observe(runner, native, *, expected, token_count):
    scheduled = rows(runner, expected=expected, row=partial(request_row, runner, native), device=native.device.type)
    slots = tuple(row.slot for row in scheduled for _ in range(row.tokens))
    if len(slots) > token_count:
        raise ValueError("Native model input rows omit scheduled LoRA tokens")
    for wrapper in wrappers(native):
        tensor(wrapper.token_lora_indices, slots, dtype=torch.int64,
               device=native.device.type, label="native token slot indices")
        kernel(wrapper.token_mapping_meta, slots, max_loras=native.lora_slots,
               specialized=native.lora_config.specialize_active_lora, device=native.device.type,
               token_count=token_count)
    return scheduled


def unadapted(runner):
    model = runner.get_model()
    if runner.lora_config is not None or any(hasattr(module, "punica_wrapper") for module in model.modules()):
        raise ValueError("Native base execution requires an engine without LoRA layers")
    return model


def approved_policies(bindings):
    policies = {value.selection.adapter: value.receipt for value in bindings}
    if any(adapter <= 0 for adapter in policies):
        raise ValueError("Native LoRA execution requires a prepared adapter for every request")
    if any(policies[value.selection.adapter] != value.receipt for value in bindings):
        raise ValueError("One native adapter ID cannot bind different approved packages")
    return MappingProxyType(policies)


class Adapted:
    def __init__(self, runner, native, *, bindings, verify):
        self.runner, self.native, self.verify = runner, native, verify
        self.model = native.model
        self.policies = approved_policies(bindings)
        self.slots = tuple(native.lora_index_to_id)

    def resident(self):
        return tuple(self.verify(receipt=receipt, adapter_id=adapter) for adapter, receipt in self.policies.items())

    def rows(self, expected, *, token_count):
        slots = tuple(self.native.lora_index_to_id)
        if slots != self.slots:
            self.resident()
            self.slots = slots
        return observe(self.runner, self.native, expected=expected, token_count=token_count)


class Unadapted:
    def __init__(self, runner):
        self.runner = runner
        self.model = unadapted(runner)

    def resident(self):
        if unadapted(self.runner) is not self.model:
            raise ValueError("Native base execution changed its model")
        return ()

    def rows(self, expected, *, token_count):
        return rows(self.runner, expected=expected, row=partial(unadapted_row, self.runner), device=self.runner.device.type)

import json
import time
from collections import Counter

import torch


def report(stage, values):
    print(json.dumps({"stage": stage, **values}, allow_nan=False), flush=True)


def measure(stage, operation, *, memory=True, emit=report):
    torch.cuda.synchronize()
    if memory:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    values = {"seconds": time.perf_counter() - started}
    if memory:
        values.update(peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved())
    emit(stage, values)
    return result


def update_profile(gradients, optimizer):
    states = {}
    for state in optimizer.state.values():
        for name, value in state.items():
            dtype = str(value.dtype) if isinstance(value, torch.Tensor) else type(value).__name__
            states.setdefault(name, Counter())[dtype] += 1
    return {"gradients": dict(Counter(str(value.dtype) for value in gradients)),
            "optimizer": {name: dict(counts) for name, counts in sorted(states.items())}}

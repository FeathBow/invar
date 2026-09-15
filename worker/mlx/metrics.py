import json
import time

import mlx.core as mx


def report(stage, values):
    print(json.dumps({"stage": stage, **values}, allow_nan=False), flush=True)


def measure(stage, operation, *, emit=report):
    mx.synchronize()
    initial = mx.get_active_memory()
    mx.reset_peak_memory()
    started = time.perf_counter()
    result = operation()
    mx.synchronize()
    emit(stage, {"seconds": time.perf_counter() - started, "allocator": "mlx",
                 "peak_active": max(initial, mx.get_peak_memory(), mx.get_active_memory()),
                 "cache_end": mx.get_cache_memory()})
    return result

import io
import json

import mlx.core as mx
from mlx.utils import tree_flatten

from worker import core

FORMAT = "invar-mlx-learner/v1"
METADATA = "invar_learner"
OPTIMIZER_PREFIX = "optimizer/"
RNG_PREFIX = "rng/"


def observe(optimizer, *, identities, parameters):
    return {"format": FORMAT, **identities, "parameters": sorted(parameters),
            "optimizer": {"configuration": {"betas": list(optimizer.betas), "epsilon": optimizer.eps,
                                              "weight_decay": optimizer.weight_decay,
                                              "bias_correction": optimizer.bias_correction},
                          "state": dict(tree_flatten(optimizer.state))},
            "rng": list(mx.random.state)}


def save(path, snapshot):
    state = snapshot["optimizer"]["state"]
    tensors = {OPTIMIZER_PREFIX + name: value for name, value in state.items()}
    tensors.update({RNG_PREFIX + str(index): value for index, value in enumerate(snapshot["rng"])})
    metadata = {name: value for name, value in snapshot.items() if name not in ("optimizer", "rng")}
    metadata.update(optimizer=snapshot["optimizer"]["configuration"], rng_count=len(snapshot["rng"]))
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("xb") as target:
        mx.save_safetensors(target, tensors, metadata={METADATA: encoded})


def load(encoded):
    tensors, metadata = mx.load(io.BytesIO(encoded), format="safetensors", return_metadata=True)
    if set(metadata) != {METADATA}:
        raise ValueError("Expected the native MLX learner container")
    fields = core.decode(metadata[METADATA])
    expected = {"format", "adapter", "base", "assembly", "tokenizer", "parameters", "optimizer", "rng_count"}
    if type(fields) is not dict or set(fields) != expected:
        raise ValueError("Unexpected native MLX learner container fields")
    count = fields["rng_count"]
    if type(count) is not int or count < 0:
        raise ValueError("Expected a nonnegative native RNG inventory")
    state = {name.removeprefix(OPTIMIZER_PREFIX): value for name, value in tensors.items()
             if name.startswith(OPTIMIZER_PREFIX)}
    random = {name for name in tensors if name.startswith(RNG_PREFIX)}
    if len(random) != count or random != {RNG_PREFIX + str(index) for index in range(count)} or len(state) + len(random) != len(tensors):
        raise ValueError("Native learner tensor names differ from the saved container")
    values = {name: value for name, value in fields.items() if name not in ("optimizer", "rng_count")}
    return {**values, "optimizer": {"configuration": fields["optimizer"], "state": state},
            "rng": [tensors[RNG_PREFIX + str(index)] for index in range(count)]}

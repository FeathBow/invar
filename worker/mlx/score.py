import argparse
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import mlx.core as mx

from worker import core
from worker import probeoutput
from worker.cohort import identity
from worker.distribution import FP32_BYTES, PROBE_FORMAT, Probe
from worker.invocation import request as request_value
from worker.mlx import model as mlx_model
from worker.mlx import numerics as mlx_numerics
from worker.mlx import tensors as mlx_tensors
from worker.mlx.crossscore import score, validate
from worker.mlx.metrics import measure
from worker.scoring import Source, provenance, source
from worker.probestore import Store
from worker.implementation import INFERENCE

FORMAT = "invar-cached-path-score-v1"
IMPLEMENTATIONS = ("worker.scoring", "worker.mlx.score", "worker.mlx.crossscore", "worker.mlx.rollout",
                   "worker.probestore", "worker.probeoutput", "worker.resident", "mlx_lm.generate")


def implementation(modules=IMPLEMENTATIONS):
    return provenance(modules, packages=("mlx", "mlx-lm", "transformers"))


def observe(loaded, selected, *, expected, sampling, measured, probe=None, store=None):
    if selected.tokenizer != expected["tokenizer"]:
        raise ValueError("Scoring target tokenizer differs from the source operation")
    validate(selected.request, selected.path, loaded.tokenizer)
    if selected.truncated != (selected.path.response[-1] != loaded.tokenizer.eos_token_id):
        raise ValueError("Source stopping report differs from the actual target EOS")
    actual = measured("verify_before", lambda: mlx_model.verify(loaded, expected, INFERENCE))
    before = {str(index): mx.array(value) for index, value in enumerate(mx.random.state)}
    observed, = measured("cross_score", lambda: score(loaded.model, loaded.tokenizer, (selected.request,),
                                                      paths=(selected.path,), sampling=sampling, probes=(probe,), stores=(store,)))
    if not mlx_tensors.equal(before, {str(index): value for index, value in enumerate(mx.random.state)}):
        raise RuntimeError("Cached scoring advanced the learner PRNG state")
    measured("verify_after", lambda: mlx_model.verify(loaded, actual, INFERENCE))
    result = {"format": FORMAT, "role": "cached_behavior_cross_score", "use_admission": "not_evaluated",
            "source_inspection_sha256": hashlib.sha256(selected.inspection).hexdigest(),
            "source": core.decode(selected.inspection),
            "target": {**actual, "model": loaded.identity[0], "revision": loaded.identity[1],
                       "numerics": loaded.numerics.name},
            "request": request_value(observed.request), "prefix_tokens": list(observed.path.prefix),
            "response_tokens": list(observed.path.response), "log_probability_bits": list(observed.log_probability_bits),
            "probability": {"role": "behavior", "log_base": "e", "representation": "F32 words",
                            "zero_support_word": 0xff800000, "temperature": observed.request.temperature,
                            "mask": "none", "top_k": "disabled", "top_p": "disabled"},
            "execution": {"engine": "mlx_lm.generate.BatchGenerator", "sampling": asdict(sampling),
                          "cache_origin": "fresh native caches; no reference cache input",
                          "unused_native_lookahead_draws": observed.lookahead_draws,
                          "truncated": observed.truncated},
            "implementation": implementation()}
    if observed.probe is not None:
        vectors = observed.probe
        result.update(format=PROBE_FORMAT, role="cached_behavior_full_vocabulary",
                      source_inspection=selected.inspection.decode("utf-8"),
                      full_vocabulary={"steps": list(vectors.probe.steps), "vocabulary": vectors.vocabulary,
                                       "coordinates": "output token ids 0..vocabulary-1", "representation": "F32 probability words",
                                       "snapshots": ([asdict(value) for value in vectors.snapshots] if store is None
                                                     else probeoutput.Rows(snapshots=vectors.snapshots)),
                                       "raw_payload_bytes": len(vectors.snapshots) * vectors.vocabulary * FP32_BYTES},
                      implementation=implementation((*IMPLEMENTATIONS, "worker.distribution", "worker.mlx.distribution")))
    return result


def run(options, *, loader=mlx_model.load):
    selected = source(options.path.read_bytes())
    probe = None if options.probe_steps is None else probe_selection(options.probe_steps, selected)
    expected = {"adapter": identity(options.digest), "tokenizer": identity(options.tokenizer_digest),
                "base": identity(options.base_digest), "assembly": identity(options.assembly_digest)}
    if selected.tokenizer != expected["tokenizer"]:
        raise ValueError("Scoring target tokenizer differs from the source operation")
    configuration = mlx_model.configuration(options.config)
    numerics = {"primary": mlx_numerics.PRIMARY, "native": mlx_numerics.NATIVE}[options.numerics]
    records = []

    def emit(stage, values):
        records.append({"stage": stage, **values})

    measured = partial(measure, emit=emit)
    with ExitStack() as storage:
        store = None if probe is None else Store(storage.enter_context(tempfile.TemporaryFile()), probe=probe)
        with redirect_stdout(sys.stderr), ExitStack() as scope:
            loaded = loader(options.cache, scope=scope, configuration=configuration, measure=measured,
                            emit=emit, initial=(options.adapter, expected), numerics=numerics)
            result = observe(loaded, selected, expected=expected, sampling=configuration.sampling(),
                             measured=measured, probe=probe, store=store)
        value = {**result, "measurements": records}
        if probe is None:
            print(json.dumps(value, allow_nan=False), flush=True)
        else:
            probeoutput.write(value, output=sys.stdout)


def probe_selection(encoded, selected):
    value = core.decode(encoded)
    if not isinstance(value, list):
        raise ValueError("Probe step selection must be a JSON array")
    probe = Probe(steps=tuple(value))
    probe.validate(len(selected.path.response))
    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Complete output of invar inspect inference")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--numerics", choices=("primary", "native"), default="primary")
    parser.add_argument("--probe-steps", help="Preselected zero-based response steps as a JSON array; retain full behavior vectors")
    for name in ("digest", "tokenizer-digest", "base-digest", "assembly-digest"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

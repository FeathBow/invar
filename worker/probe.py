import argparse
from dataclasses import dataclass
import hashlib
from importlib.util import find_spec
import json
import math
from pathlib import Path
import platform
import sys
import time

from worker import core, scalar
from worker.cohort import fields, identity
from worker.distribution import FP32_BYTES, PROBE_FORMAT, Observed, Probe, Snapshot
from worker.invocation import request as request_value
from worker.probeschema import Cost, Relation, cost, execution
from worker.scoring import Source, source

NEGATIVE_INFINITY = 0xff800000
SERIES_RADIUS = 1 / 8
SERIES_TERMS = 24
REDUCTION = "binary64 normalized FP32 masses; nonnegative generalized-KL summands; math.fsum; near-equality series/v1"


@dataclass(frozen=True, kw_only=True)
class Artifact:
    encoded: bytes
    source: Source
    target: tuple[tuple[str, str], ...]
    vectors: Observed
    cost: Cost

    @property
    def digest(self):
        return hashlib.sha256(self.encoded).hexdigest()

    @property
    def byte_count(self):
        return len(self.encoded)


def read(encoded):
    value = core.decode(encoded)
    selected, target, relation = metadata(value)
    observed = vectors(value["full_vocabulary"], selected)
    selected_words(value["log_probability_bits"], selected, observed, relation=relation)
    return Artifact(encoded=bytes(encoded), source=selected, target=target, vectors=observed,
                    cost=cost(value["measurements"], relation))


def metadata(value):
    fields(value, "format role use_admission source_inspection_sha256 source_inspection source target request prefix_tokens response_tokens log_probability_bits probability execution implementation full_vocabulary measurements")
    if value["format"] != PROBE_FORMAT or value["role"] != "cached_behavior_full_vocabulary" or value["use_admission"] != "not_evaluated":
        raise ValueError("Expected an external full-vocabulary probe observation")
    selected = source_inspection(value)
    prescribed(value, selected)
    request_fields(value["request"], selected)
    target = target_fields(value["target"], selected)
    relation = semantics(value, selected)
    return selected, target, relation


def source_inspection(value):
    if not isinstance(value["source_inspection"], str):
        raise ValueError("A probe must retain the exact source inspection")
    selected = source(value["source_inspection"].encode("utf-8"))
    # Python equality equates bools with 0/1; validate the displayed source's
    # schema as well as comparing it with the exact source inspection bytes.
    source(json.dumps(value["source"], allow_nan=False).encode("utf-8"))
    if value["source_inspection_sha256"] != hashlib.sha256(selected.inspection).hexdigest() or value["source"] != core.decode(selected.inspection):
        raise ValueError("Probe source inspection identity differs from its actual bytes")
    return selected


def prescribed(value, selected):
    for name in ("prefix_tokens", "response_tokens"):
        if not isinstance(value[name], list) or any(type(token) is not int or token < 0 for token in value[name]):
            raise ValueError("Probe prefixes and responses must be nonnegative token arrays")
    if value["prefix_tokens"] != list(selected.path.prefix) or value["response_tokens"] != list(selected.path.response):
        raise ValueError("Probe prefixes and response must match the complete prescribed source path")


def request_fields(value, selected):
    requested = fields(value, "prompt seed temperature tokens")
    if type(requested["tokens"]) is not int or type(requested["seed"]) is not int or type(requested["temperature"]) not in (int, float):
        raise ValueError("Probe request numerical fields must have their declared types")
    if requested != request_value(selected.request):
        raise ValueError("Probe request differs from its source")


def target_fields(value, selected):
    fields(value, "adapter tokenizer base assembly model revision numerics")
    for name in ("adapter", "tokenizer", "base", "assembly"):
        identity(value[name])
    if value["tokenizer"] != selected.tokenizer:
        raise ValueError("Probe target tokenizer differs from the source")
    if any(not isinstance(value[name], str) or not value[name] for name in ("model", "revision", "numerics")):
        raise ValueError("Probe target requires actual model and numerical profile descriptions")
    return tuple(sorted(value.items()))


def semantics(value, selected):
    expected = {"role": "behavior", "log_base": "e", "representation": "F32 words",
                "zero_support_word": NEGATIVE_INFINITY, "temperature": selected.request.temperature,
                "mask": "none", "top_k": "disabled", "top_p": "disabled"}
    probability = fields(value["probability"], "role log_base representation zero_support_word temperature mask top_k top_p")
    if type(probability["temperature"]) not in (int, float) or probability != expected:
        raise ValueError("Probe must measure the prescribed behavior probability semantics")
    relation = execution(value["execution"], selected)
    implementation_fields(value["implementation"])
    return relation


def implementation_fields(value):
    implementation = fields(value, "sources_sha256 packages")
    for name in ("sources_sha256", "packages"):
        entries = implementation[name]
        if not isinstance(entries, dict) or not entries or any(not isinstance(key, str) or not key or not isinstance(item, str) or not item for key, item in entries.items()):
            raise ValueError("Probe implementation requires named source identities and versions")
    for digest in implementation["sources_sha256"].values():
        identity(digest)


def vector_plan(value, selected):
    fields(value, "steps vocabulary coordinates representation snapshots raw_payload_bytes")
    if value["coordinates"] != "output token ids 0..vocabulary-1" or value["representation"] != "F32 probability words":
        raise ValueError("Probe vocabulary coordinates or probability representation differs")
    if not isinstance(value["steps"], list) or not isinstance(value["snapshots"], list):
        raise ValueError("Probe steps and snapshots must be explicit arrays")
    planned = Probe(steps=tuple(value["steps"]))
    planned.validate(len(selected.path.response))
    if type(value["vocabulary"]) is not int or value["vocabulary"] <= 0:
        raise ValueError("Probe vocabulary must have a positive integer size")
    if len(value["snapshots"]) != len(planned.steps):
        raise ValueError("Probe snapshot inventory differs from the selected steps")
    if type(value["raw_payload_bytes"]) is not int or value["raw_payload_bytes"] != len(planned.steps) * value["vocabulary"] * FP32_BYTES:
        raise ValueError("Probe raw vector byte count differs from its inventory")
    return planned


def snapshot(value):
    fields(value, "step probability_bits")
    if not isinstance(value["probability_bits"], list):
        raise ValueError("Probe probability words must be an array")
    return Snapshot(step=value["step"], probability_bits=tuple(value["probability_bits"]))


def vectors(value, selected):
    planned = vector_plan(value, selected)
    snapshots = tuple(map(snapshot, value["snapshots"]))
    result = Observed(probe=planned, snapshots=snapshots)
    if type(value["vocabulary"]) is not int or value["vocabulary"] != result.vocabulary:
        raise ValueError("Probe vocabulary size differs from its full vectors")
    return result


def path_words(encoded, selected):
    if not isinstance(encoded, list) or len(encoded) != len(selected.path.response):
        raise ValueError("Probe is missing complete-path log probabilities")
    for word in encoded:
        if type(word) is not int or word != NEGATIVE_INFINITY:
            scalar.probability(word)


def selected_support(snapshot, selected, vocabulary, *, encoded, relation):
    token = selected.path.response[snapshot.step]
    if token >= vocabulary:
        raise ValueError("Selected source token is outside the captured vocabulary")
    zero = scalar.number(snapshot.probability_bits[token]) == 0
    if relation is Relation.LOG_OF_MASS and zero != (encoded[snapshot.step] == NEGATIVE_INFINITY):
        raise ValueError("Selected log probability disagrees with captured zero support")


def selected_words(encoded, selected, observed, *, relation):
    path_words(encoded, selected)
    for snapshot in observed.snapshots:
        selected_support(snapshot, selected, observed.vocabulary, encoded=encoded, relation=relation)


def normalize(snapshot):
    values = tuple(map(scalar.number, snapshot.probability_bits))
    total = math.fsum(values)
    return tuple(value / total for value in values), total


def entropy_term(p, q):
    if p == 0:
        return q
    if q == 0:
        return math.inf
    x = (p - q) / q
    if abs(x) > SERIES_RADIUS:
        return p * math.log(p / q) + q - p
    # (1+x) log(1+x) - x has no linear term. Direct subtraction loses
    # close-distribution differences; the convergent series preserves them.
    power = x * x
    terms = []
    for order in range(2, SERIES_TERMS + 1):
        terms.append(power / (order * (order - 1)))
        power *= -x
    return q * math.fsum(terms)


def kl(p, q):
    terms = tuple(entropy_term(left, right) for left, right in zip(p, q, strict=True))
    if any(value == math.inf for value in terms):
        return math.inf
    result = math.fsum(terms)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Full-vocabulary KL reduction produced an invalid numerical result")
    return result


def metric(value):
    return {"kind": "positive_infinity"} if value == math.inf else {"kind": "finite", "value": value}


def compare(reference, candidate):
    if reference.source.inspection != candidate.source.inspection:
        raise ValueError("Distribution probes must share the exact source path inspection")
    if reference.vectors.probe != candidate.vectors.probe or reference.vectors.vocabulary != candidate.vectors.vocabulary:
        raise ValueError("Distribution probes must share frozen steps and vocabulary coordinates")
    rows = []
    for left, right in zip(reference.vectors.snapshots, candidate.vectors.snapshots, strict=True):
        p, p_mass = normalize(left)
        q, q_mass = normalize(right)
        rows.append({"step": left.step, "prefix_length": len(reference.source.path.prefix) + left.step,
                     "kl_reference_candidate": metric(kl(p, q)), "kl_candidate_reference": metric(kl(q, p)),
                     "total_variation": math.fsum(abs(a - b) for a, b in zip(p, q, strict=True)) / 2,
                     "reference_raw_mass": p_mass, "candidate_raw_mass": q_mass})
    return {"format": "invar-finite-distribution-comparison-v1", "strength": "external_finite_measurement",
            "use_admission": "not_evaluated", "source_inspection_sha256": hashlib.sha256(reference.source.inspection).hexdigest(),
            "steps": list(reference.vectors.probe.steps), "vocabulary": reference.vectors.vocabulary,
            "reference": description(reference), "candidate": description(candidate),
            "normalization": "math.fsum of raw FP32 masses followed by binary64 division",
            "reduction": REDUCTION, "log_base": "e", "probability_role": "behavior",
            "observations": rows, "summary": summary(rows),
            "unestablished": ["source-inspection authenticity", "actual probability measurement", "own-cache lineage",
                              "general numerical reduction error bound", "population guarantee", "use admission"]}


def description(artifact):
    return {"artifact_sha256": artifact.digest, "target": dict(artifact.target),
            "artifact_bytes": artifact.byte_count,
            "raw_vector_bytes": len(artifact.vectors.probe.steps) * artifact.vectors.vocabulary * FP32_BYTES,
            **artifact.cost.describe()}


def summary(rows):
    result = {}
    for direction in ("kl_reference_candidate", "kl_candidate_reference"):
        infinite = [row["step"] for row in rows if row[direction]["kind"] == "positive_infinity"]
        values = [row[direction]["value"] for row in rows if row[direction]["kind"] == "finite"]
        result[direction] = {"mean": metric(math.inf if infinite else math.fsum(values) / len(values)),
                             "maximum": metric(math.inf if infinite else max(values)), "positive_infinity_steps": infinite}
    return result


def main():
    from worker.probefile import open_probe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    options = parser.parse_args()
    with open_probe(options.reference) as reference, open_probe(options.candidate) as candidate:
        started = time.perf_counter()
        result = compare(reference, candidate)
        elapsed = time.perf_counter() - started
    print(json.dumps({**result, "reduction_seconds": elapsed, "implementation": implementation()}, allow_nan=False), flush=True)


def implementation():
    sources = {}
    for name in ("worker.probe", "worker.probefile", "worker.probejson", "worker.probeschema",
                 "worker.distribution", "worker.scoring", "worker.scalar"):
        selected = find_spec(name)
        if selected is None or selected.origin is None:
            raise RuntimeError("Probe reduction source is unavailable: " + name)
        sources[name] = hashlib.sha256(Path(selected.origin).read_bytes()).hexdigest()
    return {"sources_sha256": sources, "python": sys.version, "platform": platform.platform()}


if __name__ == "__main__":
    main()

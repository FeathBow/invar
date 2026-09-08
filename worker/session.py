import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from cohort import fields, identity, number
from invocation import Invocation, approve
from invocation import decode as invocation
from rollout import Request


@dataclass(frozen=True, kw_only=True)
class Call:
    invocation: Invocation
    load: Invocation
    request: Request
    identities: dict[str, str]


def decode(value):
    fields(value, "binding program load adapter tokenizer base assembly request")
    bound = invocation({key: value[key] for key in ("binding", "program")})
    loading = invocation(value['load'])
    if loading.binding() != bound.binding():
        raise ValueError('Policy load and inference correlation bindings differ')
    identities = {name: identity(value[name]) for name in ("adapter", "tokenizer", "base", "assembly")}
    request = value["request"]
    fields(request, "prompt tokens temperature seed")
    if not isinstance(request["prompt"], str) or type(request["seed"]) is not int:
        raise ValueError("Expected a textual prompt and integer sampling seed")
    if type(request["tokens"]) is not int or request["tokens"] <= 0:
        raise ValueError("Expected a positive token limit")
    thermal = number(request["temperature"])
    if thermal <= 0 or not math.isfinite(thermal):
        raise ValueError("Expected a finite positive temperature")
    requested = Request(sample="inference", group="inference", prompt=request["prompt"],
                        seed=request["seed"], limit=request["tokens"], temperature=thermal)
    return Call(invocation=bound, load=loading, request=requested, identities=identities)


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate inference request field")
        value[key] = item
    return value


def serve(options, *, source, loader, execute, permission):
    line = source.readline()
    if not line:
        raise ValueError("An inference batch requires at least one request")
    call = decode(json.loads(line, object_pairs_hook=unique))
    runtime = loader(options.cache, options.adapter, expected=call.identities)
    calls, attempts, instances = set(), set(), set()
    previous = None
    while True:
        bound = call.invocation
        if bound.call in calls or bound.attempt in attempts or bound.instance in instances:
            raise ValueError("Inference batch reuses a call, attempt or activation instance")
        calls.add(bound.call)
        attempts.add(bound.attempt)
        instances.add(bound.instance)
        execute(runtime, call, approve=permission, previous=previous)
        previous = call.load
        line = source.readline()
        if not line:
            return
        call = decode(json.loads(line, object_pairs_hook=unique))


def main():
    from functools import partial
    from inference import execute, load
    from metrics import measure

    parser = argparse.ArgumentParser(description="Execute a bound inference batch with one model load")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    serve(parser.parse_args(), source=sys.stdin, loader=load,
          execute=partial(execute, measure=measure), permission=approve)


if __name__ == "__main__":
    main()

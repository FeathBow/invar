from functools import partial
from types import MappingProxyType

import torch

from worker.hf.tensors import assert_equal
from worker.vllm.probes import Probes


def sampled(inner, logits, sampling_metadata, *, control, monitor, **kwargs):
    if not monitor.permitted or monitor.pending is None or not monitor.logits_finished:
        raise RuntimeError("Prescribed sampling has no permitted native model step")
    if kwargs.get("logprobs_mode_override") not in (None, "processed_logprobs"):
        raise ValueError("Prescribed sampling requires native processed log probabilities")
    if sampling_metadata.logprob_token_ids:
        raise ValueError("Prescribed sampling requires the selected-token probability column")
    if control.pending is not None:
        raise RuntimeError("A prescribed selection has not reached the native sampler output")
    operation = partial(inner, logits, sampling_metadata, **kwargs)
    tokens, probabilities = (operation() if control.probes is None
                             else control.probes.sample(operation, logits=logits, monitor=monitor))
    selected = control.select(tokens, probabilities, monitor=monitor)
    return selected, probabilities


def response(path, bound):
    if not path or len(path) > bound.request.limit or any(type(token) is not int or token < 0 for token in path):
        raise ValueError("Prescribed response requires valid token IDs within the request horizon")
    if bound.eos in path[:-1] or (path[-1] != bound.eos and len(path) != bound.request.limit):
        raise ValueError("Prescribed response does not end at the approved EOS or horizon")


def native_distribution(tokens, probabilities, count):
    if tokens.shape != (count,) or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Native selection does not contain one token per owned row")
    if probabilities is None or probabilities.ndim != 2 or probabilities.shape[0] != count or probabilities.dtype != torch.float32:
        raise ValueError("Native sampling did not return its processed FP32 probability matrix")
    if probabilities.device != tokens.device:
        raise ValueError("Native token and probability devices differ")


class Prescribed:
    def __init__(self, bindings, paths, *, probes=None):
        if set(paths) != set(bindings):
            raise ValueError("Prescribed paths differ from the owned native request set")
        selected = {key: tuple(path) for key, path in paths.items()}
        for key, path in selected.items():
            response(path, bindings[key])
        self.paths = MappingProxyType(selected)
        self.probes = None if probes is None else Probes(selected, probes)
        self.rows = {key: 0 for key in selected}
        self.ignored = {key: 0 for key in selected}
        self.pending = None

    def attach(self, monitor, hooks):
        sampler = monitor.runner.sampler
        original = sampler.sample
        instance_owned = "sample" in vars(sampler)
        sampler.sample = partial(sampled, original, control=self, monitor=monitor)
        if instance_owned:
            hooks.callback(setattr, sampler, "sample", original)
        else:
            hooks.callback(delattr, sampler, "sample")

    def select(self, tokens, probabilities, *, monitor):
        requests = tuple(row.request for row in monitor.pending.rows)
        native_distribution(tokens, probabilities, len(requests))
        selected = tokens.clone()
        for index, key in enumerate(requests):
            self.rows[key] += 1
            if key not in monitor.sampling_requests:
                self.ignored[key] += 1
                continue
            selected[index] = self.next_token(key, history=monitor.tokens[key], vocabulary=probabilities.shape[1])
        # Keep only selected coordinates while forward performs its own native gather.
        values = probabilities.gather(-1, selected.long().unsqueeze(-1))
        self.pending = selected.to(torch.int32).unsqueeze(-1), values
        return selected

    def next_token(self, key, *, history, vocabulary):
        position, path = len(history), self.paths[key]
        if position >= len(path) or tuple(history) != path[:position]:
            raise ValueError("Native cache history differs from the prescribed response prefix")
        token = path[position]
        if token >= vocabulary:
            raise ValueError("Prescribed token is outside the target output vocabulary")
        return token

    def observe(self, tokens, probabilities):
        if self.pending is None:
            raise RuntimeError("Native sampler output has no prescribed selection")
        expected_tokens, expected_probabilities = self.pending
        assert_equal(tokens, expected_tokens)
        assert_equal(probabilities, expected_probabilities)
        self.pending = None

    def completed(self, tokens):
        if self.pending is not None or any(tuple(tokens[key]) != path for key, path in self.paths.items()):
            raise ValueError("Native scoring did not consume the complete prescribed response")
        if any(self.rows[key] - self.ignored[key] != len(path) for key, path in self.paths.items()):
            raise ValueError("Native scoring row counts differ from the prescribed response")
        if self.probes is not None:
            self.probes.completed()

    def accounting(self, key, *, completed=False):
        result = {"native_sample_rows": self.rows[key], "ignored_prefill_rows": self.ignored[key]}
        if completed and self.probes is not None:
            observed = self.probes.observation(key)
            if observed is not None:
                result["probe"] = observed
        return result

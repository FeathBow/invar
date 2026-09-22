import copy
import hashlib
import json
from fractions import Fraction
import struct
import sys
import unittest

from worker.tests.mlx.scoring import BoundScoreFixture, SCORE_ENTRY, SOURCE_ENTRY, flags

FIXTURE_PATH_BUDGET = 100
REFERENCE_CALL = 81
CANDIDATE_CALL = 82
CANDIDATE_GENERATION_CALL = 8


def exact_word(word):
    return Fraction(struct.unpack("<f", struct.pack("<I", word))[0])


class ScoredProbeTests(BoundScoreFixture, unittest.TestCase):
    def test_complete_probed_scores_keep_explicit_directions_and_call_plans(self):
        original, sources = self.pair()
        selections = {"reference": [0, len(sources[0]["behavior_bits"]) - 1], "candidate": [0]}
        self.assertGreater(selections["reference"][-1], 0)
        scores = self.cross_scores(original, selections=selections)
        observations = []
        for side, source in zip(("reference", "candidate"), sources, strict=True):
            options = {**original, **scores, "relation": side + "-path-log-ratio", "budget": FIXTURE_PATH_BUDGET}
            result = self.command("full-score-" + side, ["compare", "numerical", *flags(options)])
            actual = json.loads(result.stdout)
            observed = actual["observation"]
            self.assertEqual(actual["finding"]["judgement"]["status"], "accept")
            self.assertEqual(actual["finding"]["use_admission"], "not_evaluated")
            self.assertEqual(len(actual["finding"]["judgement"]["assumptions"]), 14)
            self.assertEqual(observed["full_vocabulary"], [])
            self.assertEqual(observed["scope"]["full_vocabulary_probes"], [])
            self.assertEqual({item["path_source"] for item in observed["scored_paths"]}, {"Reference", "Candidate"})
            self.check_score(observed, scores=scores, side=side, source=source, steps=selections[side])
            rejected = self.command("full-score-zero-" + side, ["compare", "numerical", *flags({**options, "budget": 0})])
            self.assertEqual(json.loads(rejected.stdout)["finding"]["judgement"]["status"], "refute")
            self.check_missing_and_invalid(original, scores=scores, side=side)
            observations.append(observed)
        self.assertEqual(observations[0], observations[1])
        self.check_probe_is_not_score(original, scores=scores)
        missing_kl = self.command("score-is-not-kl", ["compare", "numerical", *flags({**original, **scores,
                                 "relation": "kl-reference-candidate", "probe-path": "reference", "budget": 0})])
        self.assertIn("MissingFullVocabulary", json.loads(missing_kl.stdout)["finding"]["judgement"]["reason"])
        self.check_use(original, scores=scores, numerical=observations[0])
        print(json.dumps({"actual_scored_probe_fixture": str(self.root), "steps": selections}))

    def pair(self):
        cache, checkpoint, identity = self.targets[1]
        inputs = {**self.source_inputs, "digest": identity["adapter"],
                  **{key + "-digest": identity[key] for key in ("tokenizer", "base", "assembly")},
                  **{key: CANDIDATE_GENERATION_CALL for key in ("call", "attempt", "instance")}}
        result = self.command("candidate", ["infer", *flags({**inputs, "python": sys.executable,
                              "worker": SOURCE_ENTRY, "cache": cache, "adapter": checkpoint, "worker-config": self.config})])
        candidate = json.loads(result.stdout.splitlines()[-1])
        self.assertNotEqual(candidate["tokens"], self.source_result["tokens"])
        reference = {**self.source_inputs, "log": self.source_log, "exit-code": 0}
        other = {**inputs, "log": self.root / "candidate.stdout", "exit-code": result.returncode}
        paired = {**{"reference-" + key: value for key, value in reference.items()},
                  **{"candidate-" + key: value for key, value in other.items()}}
        return paired, (self.source_result, candidate)

    def cross_scores(self, original, *, selections):
        result = {}
        for side, target, call in (("reference", 1, REFERENCE_CALL), ("candidate", 0, CANDIDATE_CALL)):
            cache, checkpoint, identity = self.targets[target]
            source = {"source-" + key.removeprefix(side + "-"): value
                      for key, value in original.items() if key.startswith(side + "-")}
            selected = {**source, "target-digest": identity["adapter"],
                        **{"target-" + key + "-digest": identity[key] for key in ("tokenizer", "base", "assembly")},
                        "call": call, "attempt": call, "instance": call, "probe-steps": json.dumps(selections[side])}
            scored = self.command("score-" + side, ["score", *flags({**selected, "python": sys.executable,
                                  "worker": SCORE_ENTRY, "cache": cache, "adapter": checkpoint, "worker-config": self.config})])
            prefix = side + "-score-"
            result.update({prefix + "log": self.root / ("score-" + side + ".stdout"),
                           prefix + "exit-code": scored.returncode, prefix + "probe-steps": selected["probe-steps"],
                           **{prefix + key: call for key in ("call", "attempt", "instance")}})
        return result

    def check_score(self, observed, *, scores, side, source, steps):
        body = json.loads(scores[side + "-score-log"].read_text().splitlines()[-1])["observation"]
        selected = next(item for item in observed["scored_paths"] if item["path_source"] == side.title())
        self.assertEqual(json.loads(selected["checked_observation_json"]), body)
        self.assertEqual(body["full_vocabulary"]["steps"], steps)
        self.assertEqual(body["prefix_tokens"] + body["response_tokens"], source["tokens"])
        ratio = sum((exact_word(p) - exact_word(q) for p, q in
                     zip(source["behavior_bits"], body["log_probability_bits"], strict=True)), Fraction())
        self.assertNotEqual(ratio, 0)
        self.assertEqual(selected["source_minus_target_log_ratio"],
                         {"kind": "finite", "numerator": ratio.numerator, "denominator": ratio.denominator})

    def check_missing_and_invalid(self, original, *, scores, side):
        prefix = side + "-score-"
        relation = {"relation": side + "-path-log-ratio", "budget": FIXTURE_PATH_BUDGET}
        missing = {key: value for key, value in scores.items() if not key.startswith(prefix)}
        result = self.command("missing-direction-" + side, ["compare", "numerical", *flags({**original, **missing, **relation})])
        self.assertIn("MissingScoredPath " + side.title(), json.loads(result.stdout)["finding"]["judgement"]["reason"])
        for index, steps in enumerate(("[1]", "[]", "[0,0]", "[1,0]", "[-1]", "[true]", "[0.5]", "null", "not-json")):
            self.command(side + "-wrong-steps-" + str(index), ["compare", "numerical", *flags({**original, **scores,
                         **relation, prefix + "probe-steps": steps})], success=False)
        undeclared = {key: value for key, value in scores.items() if key != prefix + "probe-steps"}
        self.command(side + "-undeclared-probe", ["compare", "numerical", *flags({**original, **undeclared, **relation})], success=False)
        orphan = {**original, prefix + "probe-steps": "[0]", "relation": "tokens"}
        result = self.command(side + "-orphan-steps", ["compare", "numerical", *flags(orphan)], success=False)
        self.assertIn("Missing " + prefix + "log", result.stderr)
        for key in ("call", "attempt", "instance", "exit-code"):
            self.command(side + "-wrong-" + key, ["compare", "numerical", *flags({**original, **scores, **relation,
                         prefix + key: scores[prefix + key] + 1})], success=False)
        other = "candidate" if side == "reference" else "reference"
        swapped = {prefix + key: scores[other + "-score-" + key]
                   for key in ("log", "call", "attempt", "instance", "exit-code", "probe-steps")}
        self.command(side + "-wrong-path", ["compare", "numerical", *flags({**original, **scores, **relation, **swapped})], success=False)
        self.check_changed_record(original, scores=scores, side=side, relation=relation)

    def check_changed_record(self, original, *, scores, side, relation):
        prefix = side + "-score-"
        events = [json.loads(line) for line in scores[prefix + "log"].read_text().splitlines()]
        mutations = (
            lambda values: next(row for row in values if row["stage"] == "consumed").update(probe_steps=[1]),
            lambda values: values[-1]["observation"]["full_vocabulary"].update(steps=[1]),
            lambda values: values[-1]["observation"].update(format="invar-cached-path-score-v1"),
        )
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(events)
            mutate(changed)
            self.assertNotEqual(changed, events)
            path = self.root / (side + "-changed-" + str(index) + ".jsonl")
            path.write_text("\n".join(map(json.dumps, changed)) + "\n")
            self.command(side + "-changed-" + str(index), ["compare", "numerical", *flags({**original, **scores,
                         **relation, prefix + "log": path})], success=False)

    def check_probe_is_not_score(self, original, *, scores):
        probes = {"candidate-probe-" + key: scores["reference-score-" + key]
                  for key in ("log", "call", "attempt", "instance", "exit-code")}
        options = {**original, **probes, "probe-path": "reference",
                   "probe-steps": scores["reference-score-probe-steps"],
                   "relation": "reference-path-log-ratio", "budget": FIXTURE_PATH_BUDGET}
        result = self.command("probe-is-not-score", ["compare", "numerical", *flags(options)])
        actual = json.loads(result.stdout)
        self.assertEqual(actual["observation"]["scored_paths"], [])
        self.assertIn("MissingScoredPath Reference", actual["finding"]["judgement"]["reason"])

    def check_use(self, original, *, scores, numerical):
        domain = {"name": "Native probe-enabled score fixture", "provenance_hex": hashlib.sha256(self.source_log.read_bytes()).hexdigest(),
                  "unit_definition": "One generated fixture pair; no independent sampling assertion", "inputs": [
                      {"cohort": 0, "key": "pair", "unit": "fixture", "parameters": {},
                       **{key: self.source_inputs[key] for key in ("prompt", "seed", "temperature", "tokens")}}]}
        policies = {side: {"format": "invar-policy-v1", **{key: numerical["scope"][side][key]
                    for key in ("model", "revision", "adapter", "tokenizer", "base", "assembly")}}
                    for side in ("reference", "candidate")}
        contract = {"format": "invar-use-contract", "purpose": "Native fixture plumbing; no release claim",
                    "domain": domain, "measurement": None, **policies,
                    "maximum_context": self.source_result["prompt_length"] + self.source_inputs["tokens"],
                    "criterion": {"invariance": [], "loss": None, "numerical": [{"relation": {"kind": "reference-path-log-ratio",
                         "budget": {"numerator": FIXTURE_PATH_BUDGET, "denominator": 1}}, "probe_steps": [],
                         "rationale": "Fixed interface fixture budget; not a replacement threshold"}]},
                    "protocols": {"freeze": "Fixture only", "isolation": "Exposed fixture", "selection": "No release selection"},
                    "reliance": []}
        contract_path, runs_path = self.root / "use-contract.json", self.root / "use-runs.json"
        contract_path.write_text(json.dumps(contract))
        runs_path.write_text(json.dumps([[0, "pair", flags({**original, **scores}), []]]))
        arguments = ["--contract", str(contract_path), "--runs", str(runs_path)]
        inspected = self.command("use-score-inspect", ["use", "inspect", *arguments])
        self.assertEqual(json.loads(inspected.stdout)["samples"][0]["numerical"], numerical)
        admitted = self.command("use-score-missing-reliance", ["use", "admit", *arguments])
        self.assertEqual(json.loads(admitted.stdout)["decision"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()

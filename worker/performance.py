import argparse
import json
import statistics
from pathlib import Path

import direct
import evaluation
import evidence

REQUIRED_REPEATS = 2


def durations(values):
    return {"total": sum(values), "minimum": min(values), "median": statistics.median(values), "maximum": max(values)}


def summarize(run, observed):
    samples = observed["measurements"]
    loads = observed["loads"]
    tokens = sum(row["response_tokens"] for row in samples)
    load = durations(tuple(row["seconds"] for row in loads))
    infer = durations(tuple(row["inference"]["seconds"] for row in samples))
    elapsed = evidence.positive(run["elapsed_seconds"])
    inference_seconds = evidence.positive(infer["total"])
    peaks = {key: max([row["inference"][key] for row in samples] + [row[key] for row in loads])
             for key in ("peak_allocated", "peak_reserved")}
    critical = observed["critical_path_seconds"]
    result = {"name": run["name"], "route": run["route"], "elapsed_seconds": elapsed, "calls": len(samples),
              "cohorts": observed["cohorts"], "sessions_per_cohort": list(observed["sessions_per_cohort"]),
              "concurrent": observed["concurrent"], "response_tokens": tokens, "equal_results": observed["equal_results"],
              "load_seconds": load, "inference_seconds": infer, **peaks,
              "worker_seconds_total": load["total"] + infer["total"],
              "worker_critical_path_seconds": critical,
              "seconds_outside_worker_critical_path": elapsed - critical,
              "response_tokens_per_elapsed_second": tokens / elapsed,
              "response_tokens_per_inference_second": tokens / inference_seconds,
              "log_sha256": observed["log_sha256"]}
    if run["route"] == "direct":
        result.update(completion_sha256=observed["completion_sha256"], campaign_wall_seconds=observed["campaign_wall_seconds"],
                      process_seconds=observed["process_seconds"])
    return result


def comparison(rows):
    routes = {route: tuple(row for row in rows if row["route"] == route) for route in ("invar", "direct")}
    evaluation.require(all(len(values) >= REQUIRED_REPEATS for values in routes.values()), "Repeated runs required for both routes")
    evaluation.require(len({(tuple(row["sessions_per_cohort"]), row["concurrent"]) for row in rows}) == 1,
                       "Matched measurements require the same number of model loads per cohort and the same session schedule on every run")
    elapsed = {route: statistics.mean(row["elapsed_seconds"] for row in values) for route, values in routes.items()}
    residual = {route: statistics.mean(row["seconds_outside_worker_critical_path"] for row in values)
                for route, values in routes.items()}
    return {"repeats": {route: len(values) for route, values in routes.items()}, "cohorts": rows[0]["cohorts"],
            "sessions_per_cohort": rows[0]["sessions_per_cohort"], "concurrent": rows[0]["concurrent"], "mean_elapsed_seconds": elapsed,
            "invar_minus_direct_elapsed_seconds": elapsed["invar"] - elapsed["direct"],
            "invar_minus_direct_residual_seconds": residual["invar"] - residual["direct"],
            "all_results_equal_to_reference": all(row["equal_results"] == row["calls"] for row in rows)}


def matching_profiles(observed, reference):
    actual = tuple(row["profile_sha256"] for row in observed["measurements"])
    expected = tuple(row["profile_sha256"] for row in reference["measurements"])
    evaluation.require(actual == expected, "Measured numerical profile or model revision differs from the reference")


def report(manifest, tasks, policy):
    manifest_digest, supplied = evidence.manifest(manifest)
    reference = evidence.Reference(tasks=tasks, policy=policy, reference_log=Path(supplied["reference_log"]),
                                   reference_exit_code=supplied["reference_exit_code"])
    reference_digest, tasks_digest, calls = direct.calls(reference)
    evaluation.require(all(call.consumed["adapter"] == policy for call in calls), "Reference policy mismatch")
    initial = evidence.invar({"path": str(reference.reference_log), "exit_code": reference.reference_exit_code}, reference, calls)
    rows = []
    for run in supplied["runs"]:
        if run["route"] == "invar":
            observed = evidence.invar(run, reference, calls)
        else:
            observed = evidence.direct_run(run, reference, calls, reference_digest=reference_digest, tasks_digest=tasks_digest)
        matching_profiles(observed, initial)
        rows.append(summarize(run, observed))
    return {"manifest_sha256": manifest_digest, "tasks_sha256": tasks_digest, "reference_log_sha256": reference_digest,
            "policy": policy, "runs": rows, "comparison": comparison(rows),
            "scope": "repeated reported inference measurements with supplied successful exit statuses and elapsed durations; the residual is elapsed time minus the worker critical path (per cohort the longest session when the evaluation declared several concurrent sessions, otherwise the sum, added over cohorts; direct replays are serial by construction); route differences include worker setup, validation, logging and environmental variation, not isolated causal control overhead"}


def main():
    parser = argparse.ArgumentParser(description="Summarize complete repeated Invar and direct inference measurements")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    options = parser.parse_args()
    print(json.dumps(report(options.manifest, options.tasks, options.policy), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

import hashlib
import json

FORMAT = "invar-probabilities-v5"


def document(order, update, *, invocation, request):
    learner_source = request["reference_source"] == "learner"
    samples = [{"sample": name, "dtype": "F32", "proximal": list(update.proximal[name]),
                **({"reference": list(update.reference[name])} if learner_source else {}),
                "steps": [{"step": step, "current": list(current)} for step, sample, current in update.currents if sample == name]}
               for name in order]
    return {"format": FORMAT, "invocation": invocation, "request": request, "samples": samples}


def save(path, order, update, *, invocation, request):
    encoded = json.dumps(document(order, update, invocation=invocation, request=request), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    with path.open("xb") as output:
        output.write(encoded)
    return hashlib.sha256(encoded).hexdigest()

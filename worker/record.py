import hashlib
import json

FORMAT = "invar-probabilities-v4"


def document(order, update, *, invocation, request):
    samples = [{"sample": name, "dtype": "F32", "proximal": list(update.proximal[name]),
                "steps": [{"step": step, "current": list(current)} for step, sample, current in update.currents if sample == name]}
               for name in order]
    return {"format": FORMAT, "invocation": invocation, "request": request, "samples": samples}


def save(path, order, update, *, invocation, request):
    encoded = json.dumps(document(order, update, invocation=invocation, request=request), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    with path.open("xb") as output:
        output.write(encoded)
    return hashlib.sha256(encoded).hexdigest()

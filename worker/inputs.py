from dataclasses import dataclass
from pathlib import Path

from worker.cohort import fields
from worker import resident
from worker.update import decode as update

FORMAT = "invar-learning-resident-v1"


@dataclass(frozen=True, kw_only=True)
class Paths:
    cache: Path
    reference: Path
    checkpoint: Path
    output: Path


def decode(value, options):
    fields(value, "format checkpoint output call")
    if value["format"] != FORMAT or any(not isinstance(value[name], str) or not value[name] for name in ("checkpoint", "output", "call")):
        raise ValueError("Expected a resident update with checkpoint, output and original call bytes")
    return update(resident.decode(value["call"])), Paths(cache=options.cache, reference=options.reference,
                                                       checkpoint=Path(value["checkpoint"]), output=Path(value["output"]))

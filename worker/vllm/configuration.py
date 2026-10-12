from dataclasses import dataclass, replace
import json
from pathlib import Path
import re

from worker.cohort import fields, identity
from worker.core import decode

FORMAT = "invar-vllm-launch-v1"
CHECKPOINT_FORMAT = "invar-vllm-checkpoint-launch-v1"
BASE_FORMAT = "invar-vllm-base-launch-v1"
FIELDS = {FORMAT: "format model revision handoff engine",
          CHECKPOINT_FORMAT: "format model revision handoff template engine",
          BASE_FORMAT: "format model revision engine"}
MODEL_ARGUMENTS = frozenset(("model", "model_weights", "revision", "tokenizer", "tokenizer_revision", "download_dir"))


@dataclass(frozen=True, kw_only=True)
class Configuration:
    model: str
    revision: str
    handoff: str | None
    engine: str
    template: str | None

    def arguments(self):
        return decode(self.engine)


def adapter_template(value):
    if not isinstance(value, dict) or value.get("format") not in FIELDS:
        raise ValueError("Unsupported native launch configuration")
    fields(value, FIELDS[value["format"]])
    if value["format"] != CHECKPOINT_FORMAT:
        return None
    template = value["template"]
    if not isinstance(template, str) or not template:
        raise ValueError("Native checkpoint launch requires a PEFT configuration template directory")
    return template


def configuration(value):
    template = adapter_template(value)
    if not isinstance(value["model"], str) or not value["model"]:
        raise ValueError("Native launch requires a model repository name")
    if not isinstance(value["revision"], str) or re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is None:
        raise ValueError("Native launch requires a pinned model commit")
    engine = value["engine"]
    if not isinstance(engine, dict) or MODEL_ARGUMENTS.intersection(engine):
        raise ValueError("Native engine arguments cannot override the pinned model or tokenizer location")
    handoff = None if value["format"] == BASE_FORMAT else identity(value["handoff"])
    return Configuration(model=value["model"], revision=value["revision"], handoff=handoff,
                         engine=json.dumps(engine, sort_keys=True, separators=(",", ":"), allow_nan=False),
                         template=template)


def read(path):
    parsed = configuration(decode(path.read_text()))
    if parsed.template is None:
        return parsed
    return replace(parsed, template=str((path.parent / Path(parsed.template)).resolve()))


def snapshot(config, cache, *, resolve):
    return resolve(config.model, revision=config.revision, cache_dir=cache, local_files_only=True)

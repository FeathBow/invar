from functools import partial

from worker.batch import approve, decode as batch
from worker import resident


def execute_group(group, runtime, *, source, owner, transcript, execute, release, measure):
    execute(runtime, group.calls, approve=partial(approve, source=source))
    resident.release(owner, tuple(call.load for call in group.calls), source=source, transcript=transcript,
                     operation=partial(release, runtime), measure=measure)


def serve(owner, *, source, transcript, load, activate, execute, release, close, measure):
    runtime = None
    history = resident.History()
    while True:
        raw = source.readline()
        value = resident.decode(raw)
        if isinstance(value, dict) and value.get("format") == resident.FORMAT:
            resident.closing(owner, value)
            runtime = None
            resident.close(owner, history.groups, source=source, transcript=transcript, operation=close, measure=measure)
            return
        group = batch(raw)
        next_history = history.advance(group.calls)
        expected = group.calls[0].identities
        if any(call.identities != expected for call in group.calls):
            raise ValueError("Resident group members require the same materialization")
        transcript.begin()
        if runtime is None:
            runtime = load(group.adapter, expected=expected)
        else:
            runtime = measure("activation", partial(activate, runtime, group.adapter, expected=expected), emit=transcript.emit)
        execute_group(group, runtime, source=source, owner=owner, transcript=transcript,
                      execute=execute, release=release, measure=measure)
        history = next_history

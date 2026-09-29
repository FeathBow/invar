from functools import partial

from worker.invocation import approve
from worker.hf import runtime as learner_runtime
from worker import resident
from worker.inputs import FORMAT, Paths, decode


def serve(owner, options, *, source, transcript, loader, measure, evaluate, close):
    runtime = None
    history = resident.History()
    observed = partial(measure, emit=transcript.emit)

    def shutdown():
        nonlocal runtime
        runtime = None
        close()

    while True:
        value = resident.decode(source.readline())
        if isinstance(value, dict) and value.get("format") == resident.FORMAT:
            resident.closing(owner, value)
            resident.close(owner, history.groups, source=source, transcript=transcript, operation=shutdown, measure=measure)
            return
        call, paths = decode(value, options)
        next_history = history.advance((call,))
        paths.output.mkdir(exist_ok=False)
        transcript.begin()
        if runtime is None:
            runtime, loaded = learner_runtime.initialize(paths, call.request, loader=loader,
                                                         measure=observed, evaluate=evaluate)
        else:
            loaded = observed("activation", partial(learner_runtime.activate, runtime, call.request))
        runtime = learner_runtime.execute(runtime, call, paths.output, loaded=loaded, measure=observed,
                                           permission=partial(approve, source=source), emit=transcript.emit,
                                           receive=source.readline)
        resident.release(owner, (call.load,), source=source, transcript=transcript,
                         operation=partial(learner_runtime.release, runtime), measure=measure)
        history = next_history

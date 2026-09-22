from dataclasses import replace
import json
import os
import subprocess
import sys
import time
from unittest.mock import patch

from worker import core, cycle, direct
from worker.tests.mlx import lifecycle


class ExecutionCycleChecks(lifecycle.LifecycleTests):
    def cycle(self, root, initial, trace, configuration):
        super().cycle(root, initial, trace, configuration)
        declaration = root / "cycle-reference.json"
        declaration.write_text(json.dumps({name: str(value) for name, value in trace.items()}))
        options = cycle.Options(core=lifecycle.CORE, python=sys.executable, inference_python=sys.executable,
                                inference=lifecycle.ENTRY, learning=lifecycle.ENTRY, cache=root, initial=initial,
                                reference=initial / "adapter.safetensors", trace=declaration,
                                inference_config=configuration, output=root / "execute-cycle")
        services = direct.Services(run=subprocess.run, spawn=subprocess.Popen, clock=time.perf_counter,
                                   environment=dict(os.environ))
        invoke = core.invoke

        def online(arguments, **kwargs):
            self.assertNotEqual(arguments[:2], ["replay", "inspect"])
            with (root / "online-core-commands.jsonl").open("a") as record:
                record.write(json.dumps(list(map(str, arguments))) + "\n")
            return invoke(arguments, **kwargs)

        with patch("worker.cycle.core.invoke", side_effect=online):
            executed = cycle.execute(options, services)
        self.assertEqual(executed, json.loads((options.output / "execution.json").read_text()))
        self.assertEqual((executed["sessions"], executed["publications"], len(executed["cycles"])), (1, 2, 2))
        self.assertEqual(json.loads((options.output / "trace.json").read_text()),
                         json.loads(declaration.read_text()))
        for name in ("equal", "comparison_seconds"):
            self.assertNotIn(name, executed)
        for name in ("complete.json", "observations.json"):
            self.assertFalse((options.output / name).exists())
        observed = invoke(["replay", "inspect", "--initial", initial, *cycle.flags(trace),
                           *cycle.flags({"replay-output": options.output / "checkpoints",
                                         "replay-log": options.output / "training.jsonl", "replay-exit-code": 0})],
                          executable=lifecycle.CORE)
        (root / "execution-offline-observation.json").write_text(json.dumps(observed) + "\n")
        self.assertTrue(observed["equal"])
        values = {"core": lifecycle.CORE, "python": sys.executable, "inference-python": sys.executable,
                  "inference": lifecycle.ENTRY, "learning": lifecycle.ENTRY, "cache": root, "initial": initial,
                  "reference": options.reference, "trace": declaration, "inference-config": configuration,
                  "output": root / "execute-cycle-cli"}
        entry = lifecycle.ENTRY.parents[3] / "entries" / "cycleexecute.py"
        command = [sys.executable, "-B", str(entry), *cycle.flags(values)]
        (root / "cycle-execute-command.json").write_text(json.dumps(command) + "\n")
        result, = self.command(command, root / "execute-cycle-cli.jsonl")
        self.assertEqual(result, json.loads((values["output"] / "execution.json").read_text()))
        self.assertFalse((values["output"] / "complete.json").exists())
        self.assertNotIn("equal", result)
        failed = root / "failed-worker.py"
        failed.write_text("import sys\nprint('intentional child exit', file=sys.stderr)\nraise SystemExit(23)\n")
        selected = replace(options, inference=failed, learning=failed, output=root / "execution-failed")
        with self.assertRaisesRegex(ValueError, "incomplete exchange"):
            cycle.execute(selected, services)
        self.assertFalse((selected.output / "execution.json").exists())
        self.assertFalse((selected.output / "complete.json").exists())
        statuses = [json.loads(path.read_text()) for path in selected.output.glob("*/*.status.json")]
        self.assertEqual([row["exit_code"] for row in statuses], [23])
        for status in statuses:
            with self.assertRaises(ProcessLookupError):
                os.kill(status["pid"], 0)

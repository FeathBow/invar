import unittest

try:
    import mlx  # noqa: F401
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from contextlib import redirect_stdout
from functools import partial
import io
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from worker import core
from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import initialize as mlx_initialize
from worker.baseline import mlx as mlx_product
from worker.mlx import tokenization as mlx_tokenization
from worker.tests.mlx.fixture import load
from worker.tests.mlx.publication import seal
from worker.tests.mlx.rollout import tokenizer
from worker.baseline import measure as product

CORE = os.environ.get("INVAR_CORE", "invar")


class ProductTests(unittest.TestCase):
    def test_native_shared_model_retains_optimizer_and_consumes_two_publications(self):
        root = Path(tempfile.mkdtemp(prefix="invar-mlx-product-"))
        configuration = root / "configuration.json"
        product.write(configuration, {"format": "invar-mlx-runtime-v1", "batch_size": 2,
                                      "prefill_step": 16, "cache_bytes": 1048576})
        initial = root / "initial"
        captured = io.StringIO()
        with redirect_stdout(captured):
            mlx_initialize.run(SimpleNamespace(cache=root, output=initial, config=configuration, seed=17,
                               tokenizer_digest=mlx_tokenization.digest(tokenizer())), loader=load)
        (root / "initial.jsonl").write_text(captured.getvalue())
        saved = [core.decode(line) for line in captured.getvalue().splitlines()][-1]
        selected = seal(root, initial, observed=saved, configuration=configuration, executable=CORE)
        settings = {"policy": saved["policy"], "learner": saved["learner"], "reference-digest": saved["policy"],
                    **{name + "-digest": saved[name] for name in ("tokenizer", "base", "assembly")},
                    **{"behavior-" + name + "-digest": saved[name] for name in ("base", "assembly")},
                    "clip": 0.2, "penalty": 0, "delta": 0.0001, "rate": 0.0001,
                    "beta1": 0.9, "beta2": 0.999, "optimizer-epsilon": 1e-8, "decay": 0}
        tasks = [{"tasks": [{"name": str(index), "group": "group", "prompt": "one two", "tokens": 2,
                              "seed": 17 + index, "temperature": 0.8, "answer": "#### 1"} for index in range(2)],
                  "order": [1, 0], "delivery": [0, 1]} for _ in range(2)]
        product.write(root / "tasks.json", tasks)
        product.write(root / "settings.json", settings)
        options = product.Options(backend="mlx", core=CORE, cache=root, initial=initial,
                                  reference=initial / "adapter.safetensors", tasks=root / "tasks.json",
                                  settings=root / "settings.json", configuration=configuration,
                                  output=root / "product", publication="reference", gradient_observation="objective-and-reward")
        result = product.run(options, product.Services(backend=partial(mlx_product.load, loader=load),
                                                       invoke=core.invoke, clock=time.perf_counter))
        self.assertEqual((result["publications"], len(result["cycles"])), (2, 2))
        self.assertEqual(result["final"]["policy"], settings["policy"])
        self.assertTrue(all(value["update"]["reward_gradient_norm"] == 0 for value in result["cycles"]))
        state = mlx_checkpoint.load((options.output / "checkpoints/generation2/learner.pt").read_bytes())
        self.assertEqual(state["optimizer"]["state"]["step"].item(), 2)
        events = [core.decode(line) for line in (options.output / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(value["stage"] == "load" for value in events), 1)
        self.assertEqual(sum(value["stage"] == "restore" for value in events), 1)
        self.assertEqual(sum(value["stage"] == "published_activation" for value in events), 2)
        self.assertTrue(all((options.output / "checkpoints" / f"generation{index}").is_symlink() for index in (1, 2)))
        for cycle in result["cycles"]:
            description = core.invoke(["policy", "inspect", "--checkpoint", cycle["checkpoint"]], executable=CORE)
            self.assertEqual(description, {**selected, "adapter": cycle["update"]["policy"]})


if __name__ == "__main__":
    unittest.main()

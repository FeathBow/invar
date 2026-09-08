import contextlib
import io
import json

import torch

from probe import assert_equal, check_loading


def rejects(operation):
    try:
        operation()
    except RuntimeError:
        return
    raise AssertionError("Expected an explicit mismatch")


def verify():
    clean = {"missing_keys": set(), "unexpected_keys": set(),
             "mismatched_keys": set(), "error_msgs": []}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        check_loading(clean)
    assert json.loads(output.getvalue()) == {
        "stage": "loading", "missing_keys": [], "unexpected_keys": [],
        "mismatched_keys": [], "error_msgs": [],
    }
    for key, value in (("missing_keys", {"weight"}),
                       ("unexpected_keys", {"extra"}),
                       ("mismatched_keys", {("weight", (2,), (3,))}),
                       ("error_msgs", ["conversion failed"])):
        rejects(lambda: check_loading({**clean, key: value}))
    state = {"step": torch.tensor(1.0), "nested": [torch.tensor([0.0, -0.0])]}
    assert_equal(state, {"step": torch.tensor(1.0), "nested": [torch.tensor([0.0, -0.0])]})
    rejects(lambda: assert_equal(torch.tensor(0.0), torch.tensor(-0.0)))
    rejects(lambda: assert_equal(torch.tensor([1.0]), torch.tensor([[1.0]])))
    rejects(lambda: assert_equal(torch.tensor(1.0), torch.tensor(1.0, dtype=torch.float64)))
    rejects(lambda: assert_equal(state, {"step": torch.tensor(2.0), "nested": state["nested"]}))
    print("Probe diagnostic and checkpoint comparator checks passed")


if __name__ == "__main__":
    verify()

import io

import torch

from worker import transport as native_codec
from worker.transport import write


def load(encoded):
    return torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)


def describe(value):
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "size": value.numel() * value.element_size(), "layout": str(value.layout)}
    return None


def view(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    little = value.astype(value.dtype.newbyteorder("<"), copy=False)
    return memoryview(little).cast("B")


class Session(native_codec.Session):
    def __init__(self):
        super().__init__(load=load, describe=describe, view=view)


def main():
    native_codec.run(Session())


if __name__ == "__main__":
    main()

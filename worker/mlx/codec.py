from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import tensors as mlx_tensors
from worker import transport as native_codec


class Session(native_codec.Session):
    def __init__(self):
        super().__init__(load=mlx_checkpoint.load, describe=mlx_tensors.describe, view=mlx_tensors.view)


def main():
    native_codec.run(Session())


if __name__ == "__main__":
    main()

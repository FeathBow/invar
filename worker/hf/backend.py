import torch

FORMAT = "invar-torch-numerical-settings-v1"
MATMUL_SETTINGS = ("allow_tf32", "allow_fp16_reduced_precision_reduction",
                   "allow_bf16_reduced_precision_reduction", "allow_fp16_accumulation")
CUDNN_SETTINGS = ("enabled", "benchmark", "deterministic", "allow_tf32")
AUTOCAST_DEVICES = ("cpu", "cuda")


def description():
    return {"format": FORMAT, "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul": {name: getattr(torch.backends.cuda.matmul, name) for name in MATMUL_SETTINGS},
            "blas": str(torch.backends.cuda.preferred_blas_library()),
            "cudnn": {name: getattr(torch.backends.cudnn, name) for name in CUDNN_SETTINGS},
            "deterministic": {"enabled": torch.are_deterministic_algorithms_enabled(),
                              "warn_only": torch.is_deterministic_algorithms_warn_only_enabled()},
            "autocast": {device: {"enabled": torch.is_autocast_enabled(device),
                                  "dtype": str(torch.get_autocast_dtype(device))}
                         for device in AUTOCAST_DEVICES}}

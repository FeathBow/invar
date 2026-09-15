import hashlib
from pathlib import Path
import sys

from worker.vllm.identity import kind

LINEAR_METHOD = "vllm_bnb_plugin.quantization.linear.BitsAndBytesLinearMethod"


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sources(module):
    root = Path(module.__file__).resolve().parent
    return {str(path.relative_to(root)): file_digest(path) for path in sorted(root.rglob("*.py"))}


def observe(model):
    consumers = tuple(name for name, module in model.named_modules(remove_duplicate=False)
                      if kind(getattr(module, "quant_method", None)) == LINEAR_METHOD)
    if not consumers:
        return None
    package = sys.modules["bitsandbytes"]
    library = sys.modules["bitsandbytes.cextension"].lib
    if not library.compiled_with_cuda:
        raise RuntimeError("Native BitsAndBytes consumers require the actual CUDA library")
    binary = Path(library._lib._name)
    return {"consumers": consumers, "bitsandbytes": package.__version__,
            "sources": {"bitsandbytes": sources(package), "vllm_bnb_plugin": sources(sys.modules["vllm_bnb_plugin"])},
            "binary": {"class": kind(library), "name": binary.name, "sha256": file_digest(binary)}}

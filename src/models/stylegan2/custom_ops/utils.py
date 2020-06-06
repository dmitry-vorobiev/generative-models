import importlib
import os

from torch.utils import cpp_extension
from typing import Any, Iterable


def load_extension(ext_name: str, files: Iterable[str]) -> Any:
    try:
        module = importlib.import_module(ext_name)
    except ImportError:
        module_dir = os.path.dirname(__file__)
        sources = [os.path.join(module_dir, f) for f in files]
        module = cpp_extension.load(ext_name, sources)
    return module

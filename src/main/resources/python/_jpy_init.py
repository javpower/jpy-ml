"""jpy-ml bootstrap: check ultralytics version, import modules, GIL safety."""
import sys
import os

# Force GIL on for Python 3.13+ when running under JEP with PyTorch
# This must happen before torch is imported
if sys.version_info >= (3, 13):
    os.environ.setdefault("PYTHON_GIL", "1")

try:
    import ultralytics
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics")

import numpy as np
import json
import threading

_jpy_models = {}
_jpy_model_counter = 0
_jpy_init_lock = threading.Lock()

def jpy_version():
    return {'ultralytics': ultralytics.__version__, 'python': sys.version}

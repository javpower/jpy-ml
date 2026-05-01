"""jpy-ml bootstrap: check ultralytics version, import modules."""
try:
    import ultralytics
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics")

import numpy as np
import json

_jpy_models = {}
_jpy_model_counter = 0

def jpy_version():
    return {'ultralytics': ultralytics.__version__}

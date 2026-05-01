"""Model export helper."""
from ultralytics import YOLO
import os

def jpy_export(model_path, fmt, **kwargs):
    model = YOLO(model_path)
    output = model.export(format=fmt, **kwargs)
    return {
        'path': str(output),
        'size': os.path.getsize(output) if os.path.exists(str(output)) else 0,
        'format': fmt,
    }

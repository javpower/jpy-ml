"""Model validation helper."""
from ultralytics import YOLO

def jpy_validate(model_path, **kwargs):
    model = YOLO(model_path)
    metrics = model.val(**kwargs)
    result = {
        'map50': 0.0,
        'map50_95': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'speed': dict(metrics.speed) if hasattr(metrics, 'speed') else {},
        'per_class': [],
    }
    # Use the appropriate metrics sub-object based on what's available
    if hasattr(metrics, 'box') and metrics.box is not None:
        m = metrics.box
    elif hasattr(metrics, 'seg') and metrics.seg is not None:
        m = metrics.seg
    elif hasattr(metrics, 'cls') and metrics.cls is not None:
        m = metrics.cls
    else:
        m = None

    if m is not None:
        result['map50'] = float(m.map50)
        result['map50_95'] = float(m.map)
        result['precision'] = float(m.mp)
        result['recall'] = float(m.mr)

    if hasattr(metrics, 'names') and m is not None:
        for i, name in metrics.names.items():
            ap = float(m.maps[i]) if i < len(m.maps) else 0
            result['per_class'].append({
                'class_id': i, 'class_name': name, 'map50_95': ap
            })
    return result

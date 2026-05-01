"""Model validation helper."""
from ultralytics import YOLO

def jpy_validate(model_path, **kwargs):
    model = YOLO(model_path)
    metrics = model.val(**kwargs)
    result = {
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'speed': dict(metrics.speed) if hasattr(metrics, 'speed') else {},
        'per_class': [],
    }
    if hasattr(metrics, 'names'):
        for i, name in metrics.names.items():
            ap = float(metrics.box.maps[i]) if i < len(metrics.box.maps) else 0
            result['per_class'].append({
                'class_id': i, 'class_name': name, 'map50_95': ap
            })
    return result

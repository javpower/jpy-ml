"""Inference helper functions for jpy-ml."""
from ultralytics import YOLO, RTDETR, SAM
import numpy as np
import torch

def jpy_load_model(model_path, task=None):
    """Load a model. If task is specified, force that task type regardless of filename."""
    global _jpy_model_counter
    var_name = f"_jpy_m{_jpy_model_counter}"
    _jpy_model_counter += 1

    if task is not None:
        # Explicit task: use YOLO which supports all task types
        model = YOLO(model_path)
        # Override task if model's auto-detection disagrees
        if hasattr(model, 'task') and model.task != task:
            model.task = task
    else:
        # Auto-detect from filename
        path_lower = model_path.lower()
        if "rtdetr" in path_lower:
            model = RTDETR(model_path)
        elif "sam" in path_lower:
            model = SAM(model_path)
        else:
            model = YOLO(model_path)

    _jpy_models[var_name] = model
    task_type = task if task is not None else (getattr(model, 'task', 'detect') or 'detect')
    names = dict(model.names) if hasattr(model, 'names') and model.names else {}
    return {
        'var': var_name,
        'task': task_type,
        'names': names,
        'num_classes': len(names),
    }

def jpy_predict(var_name, source, **kwargs):
    """Run prediction. Returns raw results list."""
    model = _jpy_models[var_name]
    return model(source, **kwargs)

def jpy_extract_result(result, task_type):
    """Extract all data from a single result as plain Python dicts/lists."""
    data = {
        'task': task_type,
        'speed': {
            'preprocess': result.speed.get('preprocess', 0),
            'inference': result.speed.get('inference', 0),
            'postprocess': result.speed.get('postprocess', 0),
        },
        'orig_shape': list(result.orig_shape) if result.orig_shape else [0, 0],
        'path': result.path or '',
        'names': dict(result.names) if result.names else {},
    }

    # Boxes (common to detect, segment, pose, obb)
    if result.boxes is not None:
        boxes = result.boxes
        data['boxes'] = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().tolist()
            data['boxes'].append({
                'x1': xyxy[0], 'y1': xyxy[1], 'x2': xyxy[2], 'y2': xyxy[3],
                'confidence': float(boxes.conf[i].cpu()),
                'class_id': int(boxes.cls[i].cpu()),
            })

    # Masks (segmentation)
    if result.masks is not None:
        data['masks'] = []
        for i in range(len(result.masks)):
            polygon = result.masks.xy[i].tolist() if hasattr(result.masks, 'xy') else []
            data['masks'].append({'polygon': polygon})

    # Keypoints (pose)
    if result.keypoints is not None:
        data['keypoints'] = []
        kpts = result.keypoints
        for i in range(len(kpts)):
            xy = kpts.xy[i].cpu().tolist()
            conf = kpts.conf[i].cpu().tolist() if kpts.conf is not None else [1.0] * len(xy)
            data['keypoints'].append({'xy': xy, 'conf': conf})

    # Classification (probs)
    if result.probs is not None:
        probs = result.probs
        top5 = probs.top5 if hasattr(probs, 'top5') else list(range(5))
        top5conf = probs.top5conf.cpu().tolist() if hasattr(probs, 'top5conf') else [float(probs.data[i]) for i in top5]
        data['classification'] = [
            {'class_id': int(idx), 'confidence': float(conf)}
            for idx, conf in zip(top5, top5conf)
        ]

    # OBB (oriented bounding boxes)
    if result.obb is not None:
        data['obb'] = []
        obb = result.obb
        for i in range(len(obb)):
            xywhr = obb.xywhr[i].cpu().tolist()
            data['obb'].append({
                'cx': xywhr[0], 'cy': xywhr[1], 'w': xywhr[2], 'h': xywhr[3],
                'angle': xywhr[4],
                'confidence': float(obb.conf[i].cpu()),
                'class_id': int(obb.cls[i].cpu()),
            })

    return data

def jpy_model_info(var_name):
    """Get model metadata."""
    model = _jpy_models[var_name]
    info = {}
    info['task'] = getattr(model, 'task', 'detect') or 'detect'
    info['names'] = dict(model.names) if hasattr(model, 'names') and model.names else {}
    if hasattr(model, 'model') and model.model is not None:
        n_params = sum(p.numel() for p in model.model.parameters())
        n_layers = len(list(model.model.modules()))
        info['parameters'] = n_params
        info['layers'] = n_layers
    return info

def jpy_cleanup(var_name):
    """Remove model from cache."""
    if var_name in _jpy_models:
        del _jpy_models[var_name]

def jpy_annotate(result, output_path=None):
    """Get annotated image. If output_path given, save to file."""
    annotated = result.plot()
    if output_path:
        import cv2
        cv2.imwrite(output_path, annotated)
        return output_path
    else:
        import cv2
        _, buf = cv2.imencode('.png', annotated)
        return buf.tobytes()

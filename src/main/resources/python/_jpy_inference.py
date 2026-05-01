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

def jpy_batch_extract(results, task_type):
    """Extract all results from a batch as a list of dicts."""
    return [jpy_extract_result(r, task_type) for r in results]

def jpy_batch_predict(var_name, sources, task_type, kwargs):
    """Run prediction on multiple images. Returns list of extracted result dicts."""
    model = _jpy_models[var_name]
    results = []
    for src in sources:
        raw = model(src, **kwargs)
        for r in raw:
            results.append(jpy_extract_result(r, task_type))
    return results

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


def jpy_extract_result_ndarray(result, task_type, buffers=None):
    """Extract result data using NDArray for zero-copy transfer.

    Args:
        result: Ultralytics inference result
        task_type: Task type string (detect, segment, classify, pose, obb)
        buffers: Optional dict of DirectNDArray buffers for zero-copy write

    Returns:
        dict with NDArray or list data
    """
    from jep import NDArray

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
        n = len(boxes)

        if buffers is not None and 'boxes_xyxy' in buffers:
            # Zero-copy mode: write to pre-allocated DirectNDArray buffers
            buf_xyxy = buffers['boxes_xyxy']
            buf_conf = buffers['boxes_conf']
            buf_cls = buffers['boxes_cls']

            # Get numpy arrays (shared memory with CPU tensor)
            xyxy_np = boxes.xyxy.cpu().numpy().astype(np.float32)
            conf_np = boxes.conf.cpu().numpy().astype(np.float32)
            cls_np = boxes.cls.cpu().numpy().astype(np.int32)

            # Get the underlying Java buffers
            xyxy_buf = buf_xyxy.getData()
            conf_buf = buf_conf.getData()
            cls_buf = buf_cls.getData()

            # Write data to buffers using struct bulk copy
            import struct
            xyxy_bytes = xyxy_np.tobytes()
            conf_bytes = conf_np.tobytes()
            cls_bytes = cls_np.tobytes()

            # Bulk write: pack as float/int arrays directly into buffer
            struct.pack_into(f'<{len(xyxy_np.flat)}f', xyxy_buf, 0, *xyxy_np.flat)
            struct.pack_into(f'<{len(conf_np)}f', conf_buf, 0, *conf_np)
            struct.pack_into(f'<{len(cls_np)}i', cls_buf, 0, *cls_np)

            data['boxes_count'] = n
            data['boxes_ndarray'] = True
        else:
            # Standard NDArray mode
            xyxy = boxes.xyxy.cpu().numpy().flatten().tolist()
            conf = boxes.conf.cpu().numpy().flatten().tolist()
            cls = boxes.cls.cpu().numpy().flatten().tolist()

            data['boxes_xyxy'] = NDArray(xyxy, n, 4)
            data['boxes_conf'] = NDArray(conf, n)
            data['boxes_cls'] = NDArray(cls, n)
            data['boxes_count'] = n
            data['boxes_ndarray'] = True

    # Masks (segmentation)
    if result.masks is not None:
        all_points = []
        all_sizes = []
        for mask in result.masks.xy:
            points = mask.tolist()
            all_points.extend(points)
            all_sizes.append(len(points))
        data['masks_points'] = NDArray([p for points in all_points for p in points])
        data['masks_sizes'] = all_sizes

    # Keypoints (pose)
    if result.keypoints is not None:
        kpts = result.keypoints
        n_kpts = len(kpts)
        xy = kpts.xy.cpu().numpy().flatten().tolist()
        data['keypoints_xy'] = NDArray(xy, n_kpts, 17, 2)
        if kpts.conf is not None:
            conf = kpts.conf.cpu().numpy().flatten().tolist()
            data['keypoints_conf'] = NDArray(conf, n_kpts, 17)

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
        obb = result.obb
        n_obb = len(obb)
        xywhr = obb.xywhr.cpu().numpy().flatten().tolist()
        conf = obb.conf.cpu().numpy().flatten().tolist()
        cls = obb.cls.cpu().numpy().flatten().tolist()
        data['obb_xywhr'] = NDArray(xywhr, n_obb, 5)
        data['obb_conf'] = NDArray(conf, n_obb)
        data['obb_cls'] = NDArray(cls, n_obb)
        data['obb_count'] = n_obb

    return data


def jpy_decode_image(raw_bytes):
    """Decode raw image bytes to a numpy array suitable for model inference.

    Args:
        raw_bytes: Python bytes or jep.PyJArray (Java byte[])

    Returns:
        numpy ndarray (H, W, 3) in BGR format
    """
    import cv2
    # Jep converts Java byte[] (signed -128..127) to PyJArray
    # Use numpy to handle the conversion correctly
    arr = np.array(raw_bytes, dtype=np.int8).astype(np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

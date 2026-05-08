"""Inference helper functions for jpy-ml."""
from ultralytics import YOLO, RTDETR, SAM
import numpy as np
import threading

_jpy_inference_lock = threading.Lock()

def jpy_load_model(model_path, task=None):
    """Load a model. If task is specified, force that task type regardless of filename."""
    global _jpy_model_counter
    with _jpy_inference_lock:
        var_name = f"_jpy_m{_jpy_model_counter}"
        _jpy_model_counter += 1

    if task is not None:
        model = YOLO(model_path)
        if hasattr(model, 'task') and model.task != task:
            model.task = task
    else:
        path_lower = model_path.lower()
        if "rtdetr" in path_lower:
            model = RTDETR(model_path)
        elif "sam" in path_lower:
            model = SAM(model_path)
        else:
            model = YOLO(model_path)

    with _jpy_inference_lock:
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
    if var_name not in _jpy_models:
        raise KeyError(f"Model '{var_name}' not found. Was it loaded?")
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
        xyxy_all = boxes.xyxy.cpu().numpy()
        conf_all = boxes.conf.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy()
        data['boxes'] = []
        for i in range(len(boxes)):
            data['boxes'].append({
                'x1': float(xyxy_all[i, 0]), 'y1': float(xyxy_all[i, 1]),
                'x2': float(xyxy_all[i, 2]), 'y2': float(xyxy_all[i, 3]),
                'confidence': float(conf_all[i]),
                'class_id': int(cls_all[i]),
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
        xy_all = kpts.xy.cpu().numpy()
        conf_all = kpts.conf.cpu().numpy() if kpts.conf is not None else None
        for i in range(len(kpts)):
            xy = xy_all[i].tolist()
            conf = conf_all[i].tolist() if conf_all is not None else [1.0] * len(xy)
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
        xywhr_all = obb.xywhr.cpu().numpy()
        conf_all = obb.conf.cpu().numpy()
        cls_all = obb.cls.cpu().numpy()
        for i in range(len(obb)):
            data['obb'].append({
                'cx': float(xywhr_all[i, 0]), 'cy': float(xywhr_all[i, 1]),
                'w': float(xywhr_all[i, 2]), 'h': float(xywhr_all[i, 3]),
                'angle': float(xywhr_all[i, 4]),
                'confidence': float(conf_all[i]),
                'class_id': int(cls_all[i]),
            })

    return data

def jpy_batch_extract(results, task_type):
    """Extract all results from a batch as a list of dicts."""
    return [jpy_extract_result(r, task_type) for r in results]

def jpy_batch_predict(model, sources, task_type, kwargs):
    """Run prediction on multiple images using native batch inference.

    Converts all sources (paths, URLs, bytes, numpy arrays) to numpy arrays
    first, then passes the list to Ultralytics for GPU-batched inference.
    Falls back to per-image processing if batch fails.
    """
    import cv2
    import numpy as np

    images = []
    for src in sources:
        img = _decode_source(src)
        if img is not None:
            images.append(img)

    if not images:
        return []

    results = []
    try:
        raw_results = model(images, **kwargs)
        for r in raw_results:
            results.append(jpy_extract_result(r, task_type))
    except (TypeError, ValueError):
        for img in images:
            raw = model(img, **kwargs)
            for r in raw:
                results.append(jpy_extract_result(r, task_type))
    return results


def _decode_source(src):
    """Decode a source (path, URL, bytes, numpy array) to a BGR numpy array."""
    import cv2
    import numpy as np

    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, str):
        if src.startswith(('http://', 'https://')):
            import urllib.request
            resp = urllib.request.urlopen(src)
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.imread(src)
    # bytes-like (Java byte[] from Jep)
    arr = np.frombuffer(bytes(src), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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
    """Remove model from cache and release GPU memory."""
    with _jpy_inference_lock:
        if var_name in _jpy_models:
            del _jpy_models[var_name]
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def jpy_annotate(result, output_path=None):
    """Get annotated image. If output_path given, save to file."""
    import cv2
    annotated = result.plot()
    if output_path:
        cv2.imwrite(output_path, annotated)
        return output_path
    else:
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

            # Get numpy arrays
            xyxy_np = boxes.xyxy.cpu().numpy().astype(np.float32)
            conf_np = boxes.conf.cpu().numpy().astype(np.float32)
            cls_np = boxes.cls.cpu().numpy().astype(np.int32)

            # Get the underlying Java buffers and copy data directly
            xyxy_buf = buf_xyxy.getData()
            conf_buf = buf_conf.getData()
            cls_buf = buf_cls.getData()

            xyxy_bytes = xyxy_np.tobytes()
            conf_bytes = conf_np.tobytes()
            cls_bytes = cls_np.tobytes()

            xyxy_buf[:len(xyxy_bytes)] = xyxy_bytes
            conf_buf[:len(conf_bytes)] = conf_bytes
            cls_buf[:len(cls_bytes)] = cls_bytes

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
        data['masks_points'] = NDArray(all_points)
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

    Raises:
        ValueError: if the image cannot be decoded
    """
    import cv2
    arr = np.array(raw_bytes, dtype=np.int8).astype(np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from raw bytes")
    return img

"""SAM 2 (Segment Anything Model 2) integration for jpy-ml."""

from ultralytics import SAM

# Global model storage
_jpy_sam2_models = {}


def jpy_sam2_load(model_path, var_name):
    """Load a SAM 2 model.

    Args:
        model_path: Path to the SAM 2 model file (e.g., 'sam2_b.pt')
        var_name: Variable name for storage

    Returns:
        dict with model info
    """
    model = SAM(model_path)
    _jpy_sam2_models[var_name] = model
    return {'var': var_name, 'task': 'segment'}


def jpy_sam2_predict(var_name, image_path, prompts):
    """Run SAM 2 prediction with the given prompts.

    Args:
        var_name: Variable name of the loaded model
        image_path: Path to the image file
        prompts: List of prompt dicts with 'type' key ('point', 'box', 'text')

    Returns:
        dict with masks and scores
    """
    model = _jpy_sam2_models[var_name]

    # Convert prompts to Python dicts (handles Java Maps from Jep)
    prompts = [dict(p) for p in prompts]

    # Convert prompts to the format expected by Ultralytics SAM
    points = []
    boxes = []
    labels = []

    for p in prompts:
        if p['type'] == 'point':
            points.append([p['x'], p['y']])
            labels.append(p.get('label', 1))
        elif p['type'] == 'box':
            boxes.append([p['x1'], p['y1'], p['x2'], p['y2']])

    # Build prediction kwargs
    kwargs = {}
    if points:
        kwargs['points'] = points
    if labels:
        kwargs['labels'] = labels
    if boxes:
        kwargs['bboxes'] = boxes

    # Run prediction
    results = model(image_path, **kwargs)

    # Extract results
    if not results:
        return {'masks': [], 'path': image_path}

    result = results[0]
    masks_data = []

    if result.masks is not None:
        for i in range(len(result.masks)):
            # Get polygon representation
            polygon = result.masks.xy[i].tolist() if hasattr(result.masks, 'xy') else []
            score = float(result.boxes.conf[i].cpu()) if result.boxes is not None and i < len(result.boxes) else 1.0
            masks_data.append({
                'polygon': polygon,
                'score': score,
            })

    return {
        'masks': masks_data,
        'path': image_path,
    }

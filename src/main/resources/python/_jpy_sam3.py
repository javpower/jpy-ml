"""SAM 3 (Segment Anything Model 3) integration for jpy-ml."""

from ultralytics import SAM

# Global model storage
_jpy_sam3_models = {}


def jpy_sam3_load(model_path, var_name):
    """Load a SAM 3 model.

    Args:
        model_path: Path to the SAM 3 model file
        var_name: Variable name for storage

    Returns:
        dict with model info
    """
    model = SAM(model_path)
    _jpy_sam3_models[var_name] = model
    return {'var': var_name, 'task': 'segment'}


def jpy_sam3_predict_text(var_name, image_path, text_prompts):
    """Run SAM 3 prediction with text prompts.

    Args:
        var_name: Variable name of the loaded model
        image_path: Path to the image file
        text_prompts: List of text prompts (e.g., ['person', 'car'])

    Returns:
        dict with masks and scores
    """
    model = _jpy_sam3_models[var_name]
    results = model(image_path, texts=list(text_prompts))

    if not results:
        return {'masks': [], 'path': image_path}

    result = results[0]
    masks_data = []

    if result.masks is not None:
        for i in range(len(result.masks)):
            polygon = result.masks.xy[i].tolist() if hasattr(result.masks, 'xy') else []
            score = float(result.boxes.conf[i].cpu()) if result.boxes is not None and i < len(result.boxes) else 1.0
            class_id = int(result.boxes.cls[i].cpu()) if result.boxes is not None and i < len(result.boxes) else 0
            masks_data.append({
                'polygon': polygon,
                'score': score,
                'class_id': class_id,
            })

    return {
        'masks': masks_data,
        'path': image_path,
    }


def jpy_sam3_predict_exemplar(var_name, image_path, exemplar_path, exemplar_box):
    """Run SAM 3 prediction with an image exemplar.

    Args:
        var_name: Variable name of the loaded model
        image_path: Path to the target image
        exemplar_path: Path to the exemplar image
        exemplar_box: Bounding box in exemplar image [x1, y1, x2, y2]

    Returns:
        dict with masks and scores
    """
    model = _jpy_sam3_models[var_name]

    # First, segment the exemplar to get a reference mask
    exemplar_results = model(exemplar_path, bboxes=[list(exemplar_box)])

    if not exemplar_results or exemplar_results[0].masks is None:
        return {'masks': [], 'path': image_path}

    # Use the exemplar mask as a prompt for the target image
    exemplar_mask = exemplar_results[0].masks.xy[0].tolist()

    results = model(image_path, masks=[exemplar_mask])

    if not results:
        return {'masks': [], 'path': image_path}

    result = results[0]
    masks_data = []

    if result.masks is not None:
        for i in range(len(result.masks)):
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

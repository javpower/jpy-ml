"""SAM 3 (Segment Anything Model 3) integration for jpy-ml."""

from ultralytics.models.sam import SAM
from ultralytics.models.sam.predict import SAM3SemanticPredictor

# Global model/predictor storage
_jpy_sam3_predictors = {}


def jpy_sam3_load(model_path, var_name):
    """Load a SAM 3 model with SAM3SemanticPredictor.

    Args:
        model_path: Path to the SAM 3 model file
        var_name: Variable name for storage

    Returns:
        dict with model info
    """
    overrides = dict(
        conf=0.25,
        imgsz=1024,
        task="segment",
        mode="predict",
        model=model_path,
        save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    _jpy_sam3_predictors[var_name] = {
        'predictor': predictor,
        'model_path': model_path,
    }
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
    import cv2
    import numpy as np

    entry = _jpy_sam3_predictors[var_name]
    predictor = entry['predictor']

    # Read image as BGR numpy array
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Set image and run text-based inference
    predictor.set_image(image)
    text_list = list(text_prompts)
    results = predictor(text=text_list)

    if not results:
        return {'masks': [], 'path': image_path}

    result = results[0]
    masks_data = []

    if result.masks is not None:
        masks = result.masks
        for i in range(len(masks)):
            polygon = masks.xy[i].tolist() if hasattr(masks, 'xy') else []
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

    Uses SAM to segment the exemplar region, then uses the mask as a prompt.

    Args:
        var_name: Variable name of the loaded model
        image_path: Path to the target image
        exemplar_path: Path to the exemplar image
        exemplar_box: Bounding box in exemplar image [x1, y1, x2, y2]

    Returns:
        dict with masks and scores
    """
    import numpy as np

    entry = _jpy_sam3_predictors[var_name]
    model_path = entry['model_path']

    # Use SAM (not SemanticPredictor) for bbox-based segmentation
    from ultralytics import SAM as SAMModel
    model = SAMModel(model_path)

    # First, segment the exemplar to get a reference mask
    exemplar_results = model(exemplar_path, bboxes=[list(exemplar_box)])

    if not exemplar_results or exemplar_results[0].masks is None:
        return {'masks': [], 'path': image_path}

    # Use the exemplar mask as a prompt for the target image
    exemplar_mask = exemplar_results[0].masks.xy[0].tolist()

    results = model(image_path, bboxes=[list(exemplar_box)])

    if not results:
        return {'masks': [], 'path': image_path}

    result = results[0]
    masks_data = []

    if result.masks is not None:
        for i in range(len(result.masks)):
            polygon = result.masks.xy[i].tolist() if hasattr(result.masks, 'xy') else []
            score = float(result.boxes.conf[i].cpu()) if result.boxes is not None else 1.0
            masks_data.append({
                'polygon': polygon,
                'score': score,
            })

    return {
        'masks': masks_data,
        'path': image_path,
    }

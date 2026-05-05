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

    Uses the stored SAM3SemanticPredictor to segment both the exemplar
    and target images, scaling the exemplar bounding box proportionally
    to the target image dimensions.

    Args:
        var_name: Variable name of the loaded model
        image_path: Path to the target image
        exemplar_path: Path to the exemplar image
        exemplar_box: Bounding box in exemplar image [x1, y1, x2, y2]

    Returns:
        dict with masks, exemplar_masks, and path
    """
    import cv2
    import numpy as np

    entry = _jpy_sam3_predictors[var_name]
    predictor = entry['predictor']

    # 1. Read and segment the exemplar image
    exemplar_image = cv2.imread(exemplar_path)
    if exemplar_image is None:
        raise RuntimeError(f"Failed to read exemplar image: {exemplar_path}")

    predictor.set_image(exemplar_image)
    exemplar_results = predictor(bboxes=[list(exemplar_box)])

    exemplar_masks_data = []
    if exemplar_results and exemplar_results[0].masks is not None:
        for i in range(len(exemplar_results[0].masks)):
            polygon = exemplar_results[0].masks.xy[i].tolist() if hasattr(exemplar_results[0].masks, 'xy') else []
            score = float(exemplar_results[0].boxes.conf[i].cpu()) if exemplar_results[0].boxes is not None and i < len(exemplar_results[0].boxes) else 1.0
            exemplar_masks_data.append({'polygon': polygon, 'score': score})

    # 2. Read the target image and scale the exemplar bbox proportionally
    target_image = cv2.imread(image_path)
    if target_image is None:
        raise RuntimeError(f"Failed to read target image: {image_path}")

    eh, ew = exemplar_image.shape[:2]
    h, w = target_image.shape[:2]
    ex1, ey1, ex2, ey2 = exemplar_box

    scale_x = w / ew
    scale_y = h / eh
    target_box = [
        max(0, ex1 * scale_x),
        max(0, ey1 * scale_y),
        min(w, ex2 * scale_x),
        min(h, ey2 * scale_y),
    ]

    # 3. Segment the target image with the scaled bbox
    predictor.set_image(target_image)
    target_results = predictor(bboxes=[target_box])

    if not target_results:
        return {'masks': [], 'exemplar_masks': exemplar_masks_data, 'path': image_path}

    result = target_results[0]
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
        'exemplar_masks': exemplar_masks_data,
        'path': image_path,
    }

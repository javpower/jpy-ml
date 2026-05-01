"""OpenCV integration for jpy-ml."""

import cv2
import numpy as np


def jpy_imread(image_path):
    """Read an image from file.

    Args:
        image_path: Path to the image file

    Returns:
        dict with image info (width, height, channels)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    return {
        'width': w,
        'height': h,
        'channels': c,
        'path': image_path,
    }


def jpy_imwrite(image_path, output_path):
    """Write an image to file.

    Args:
        image_path: Path to the source image
        output_path: Path to save the image

    Returns:
        output_path on success
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    cv2.imwrite(output_path, img)
    return output_path


def jpy_cvtColor(image_path, output_path, conversion):
    """Convert image color space.

    Args:
        image_path: Path to the source image
        output_path: Path to save the converted image
        conversion: Color conversion code (e.g., 'BGR2GRAY', 'BGR2RGB')

    Returns:
        output_path on success
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Get conversion code
    code = getattr(cv2, f'COLOR_{conversion}', None)
    if code is None:
        raise ValueError(f"Unknown color conversion: {conversion}")

    converted = cv2.cvtColor(img, code)
    cv2.imwrite(output_path, converted)
    return output_path


def jpy_resize(image_path, output_path, width, height, interpolation='INTER_LINEAR'):
    """Resize an image.

    Args:
        image_path: Path to the source image
        output_path: Path to save the resized image
        width: Target width
        height: Target height
        interpolation: Interpolation method

    Returns:
        output_path on success
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Get interpolation method
    interp = getattr(cv2, interpolation, cv2.INTER_LINEAR)

    resized = cv2.resize(img, (width, height), interpolation=interp)
    cv2.imwrite(output_path, resized)
    return output_path


def jpy_blur(image_path, output_path, ksize_x, ksize_y):
    """Apply Gaussian blur to an image.

    Args:
        image_path: Path to the source image
        output_path: Path to save the blurred image
        ksize_x: Kernel width
        ksize_y: Kernel height

    Returns:
        output_path on success
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    blurred = cv2.GaussianBlur(img, (ksize_x, ksize_y), 0)
    cv2.imwrite(output_path, blurred)
    return output_path


def jpy_canny(image_path, output_path, threshold1, threshold2):
    """Apply Canny edge detection.

    Args:
        image_path: Path to the source image
        output_path: Path to save the edge image
        threshold1: First threshold for hysteresis
        threshold2: Second threshold for hysteresis

    Returns:
        output_path on success
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    cv2.imwrite(output_path, edges)
    return output_path


def jpy_threshold(image_path, output_path, thresh, maxval, thresh_type='THRESH_BINARY'):
    """Apply threshold to an image.

    Args:
        image_path: Path to the source image
        output_path: Path to save the thresholded image
        thresh: Threshold value
        maxval: Maximum value
        thresh_type: Threshold type

    Returns:
        output_path on success
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get threshold type
    ttype = getattr(cv2, thresh_type, cv2.THRESH_BINARY)

    _, thresholded = cv2.threshold(gray, thresh, maxval, ttype)
    cv2.imwrite(output_path, thresholded)
    return output_path


def jpy_find_contours(image_path, output_path=None):
    """Find contours in an image.

    Args:
        image_path: Path to the source image
        output_path: Optional path to save the contour visualization

    Returns:
        dict with contour data
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to list of points
    contour_data = []
    for contour in contours:
        points = contour.reshape(-1, 2).tolist()
        contour_data.append({
            'points': points,
            'area': cv2.contourArea(contour),
            'length': cv2.arcLength(contour, True),
        })

    # Draw contours if output path specified
    if output_path:
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(output_path, img)

    return {
        'contours': contour_data,
        'count': len(contours),
    }


def jpy_morphology(image_path, output_path, operation, kernel_size, iterations=1):
    """Apply morphological operation.

    Args:
        image_path: Path to the source image
        output_path: Path to save the result
        operation: Morphological operation (ERODE, DILATE, OPEN, CLOSE, etc.)
        kernel_size: Kernel size
        iterations: Number of iterations

    Returns:
        output_path on success
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get morphological operation
    op = getattr(cv2, f'MORPH_{operation}', cv2.MORPH_ERODE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    result = cv2.morphologyEx(gray, op, kernel, iterations=iterations)
    cv2.imwrite(output_path, result)
    return output_path

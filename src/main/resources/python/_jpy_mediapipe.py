"""MediaPipe integration for jpy-ml (v0.10+ tasks API)."""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Model paths - will be downloaded on first use
_HAND_MODEL = "hand_landmarker.task"
_FACE_MODEL = "face_landmarker.task"
_POSE_MODEL = "pose_landmarker.task"


def _download_model(model_name, url):
    """Download model if not exists."""
    import os
    if not os.path.exists(model_name):
        import urllib.request
        print(f"Downloading {model_name}...")
        urllib.request.urlretrieve(url, model_name)
    return model_name


def jpy_mp_init():
    """Initialize MediaPipe modules.

    Returns:
        dict with initialized modules
    """
    # Download models
    hand_model = _download_model(_HAND_MODEL,
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    face_model = _download_model(_FACE_MODEL,
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")

    # Create detectors
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=hand_model),
        num_hands=2,
        min_hand_detection_confidence=0.5
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=face_model),
        num_faces=1,
        min_face_detection_confidence=0.5
    )

    result = {
        'hands': vision.HandLandmarker.create_from_options(hand_options),
        'face': vision.FaceLandmarker.create_from_options(face_options),
    }

    # Pose model may not be available
    try:
        pose_model = _download_model(_POSE_MODEL,
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/latest/pose_landmarker.task")
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model),
            min_pose_detection_confidence=0.5
        )
        result['pose'] = vision.PoseLandmarker.create_from_options(pose_options)
    except Exception as e:
        print(f"Warning: Pose model not available: {e}")
        result['pose'] = None

    return result


def jpy_mp_detect_hands(modules, image_path):
    """Detect hands in an image.

    Args:
        modules: dict with initialized modules
        image_path: Path to the image file

    Returns:
        dict with hand detection results
    """
    image = mp.Image.create_from_file(image_path)
    result = modules['hands'].detect(image)

    hands = []
    for hand_landmarks in result.hand_landmarks:
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
            })
        hands.append({
            'landmarks': landmarks,
            'count': len(landmarks),
        })

    return {
        'hands': hands,
        'count': len(hands),
        'path': image_path,
    }


def jpy_mp_detect_face(modules, image_path):
    """Detect face mesh in an image.

    Args:
        modules: dict with initialized modules
        image_path: Path to the image file

    Returns:
        dict with face detection results
    """
    image = mp.Image.create_from_file(image_path)
    result = modules['face'].detect(image)

    faces = []
    for face_landmarks in result.face_landmarks:
        landmarks = []
        for lm in face_landmarks:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
            })
        faces.append({
            'landmarks': landmarks,
            'count': len(landmarks),
        })

    return {
        'faces': faces,
        'count': len(faces),
        'path': image_path,
    }


def jpy_mp_detect_pose(modules, image_path):
    """Detect pose in an image.

    Args:
        modules: dict with initialized modules
        image_path: Path to the image file

    Returns:
        dict with pose detection results
    """
    image = mp.Image.create_from_file(image_path)
    result = modules['pose'].detect(image)

    landmarks = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks[0]:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility,
            })

    return {
        'landmarks': landmarks,
        'count': len(landmarks),
        'path': image_path,
    }

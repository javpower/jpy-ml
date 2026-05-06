"""SAM 2 video tracking integration for jpy-ml."""


def jpy_sam2_video_start(model, video_path, prompts):
    """Initialize video tracking with SAM 2."""
    points = []
    boxes = []
    labels = []

    for p in prompts:
        if p['type'] == 'point':
            points.append([p['x'], p['y']])
            labels.append(p.get('label', 1))
        elif p['type'] == 'box':
            boxes.append([p['x1'], p['y1'], p['x2'], p['y2']])

    tracker = {
        'model': model,
        'video_path': video_path,
        'points': points,
        'boxes': boxes,
        'labels': labels,
        'frame_prompts': {},
    }

    return tracker


def jpy_sam2_video_add_prompt(tracker, frame_index, prompt):
    """Add a prompt to a specific frame."""
    if frame_index not in tracker['frame_prompts']:
        tracker['frame_prompts'][frame_index] = []
    tracker['frame_prompts'][frame_index].append(dict(prompt))


def jpy_sam2_video_propagate(tracker):
    """Propagate tracking through all frames.

    Uses fresh prompts per frame (merged from initial + frame-specific)
    to avoid state accumulation issues in Jep environment.
    """
    import cv2
    import numpy as np

    model = tracker['model']
    video_path = tracker['video_path']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_results = []

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Merge initial prompts with frame-specific prompts
            points = list(tracker['points'])
            boxes = list(tracker['boxes'])
            labels = list(tracker['labels'])

            if frame_idx in tracker['frame_prompts']:
                for p in tracker['frame_prompts'][frame_idx]:
                    if p['type'] == 'point':
                        points.append([p['x'], p['y']])
                        labels.append(p.get('label', 1))
                    elif p['type'] == 'box':
                        boxes.append([p['x1'], p['y1'], p['x2'], p['y2']])

            # Build kwargs — only include non-empty lists
            kwargs = {}
            if points:
                kwargs['points'] = np.array(points, dtype=np.float32)
                kwargs['labels'] = np.array(labels, dtype=np.int32)
            if boxes:
                kwargs['bboxes'] = np.array(boxes, dtype=np.float32)

            results = model(frame, **kwargs)

            if results and results[0].masks is not None:
                conf_all = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else None
                for i in range(len(results[0].masks)):
                    polygon = results[0].masks.xy[i].tolist() if hasattr(results[0].masks, 'xy') else []
                    score = float(conf_all[i]) if conf_all is not None else 1.0
                    frame_results.append({
                        'frame_index': frame_idx,
                        'polygon': polygon,
                        'score': score,
                    })

            frame_idx += 1
    finally:
        cap.release()

    return {
        'total_frames': total_frames,
        'frames': frame_results,
    }

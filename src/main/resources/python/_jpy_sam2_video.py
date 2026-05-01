"""SAM 2 video tracking integration for jpy-ml."""


def jpy_sam2_video_start(model, video_path, prompts):
    """Initialize video tracking with SAM 2.

    Args:
        model: Loaded SAM model
        video_path: Path to the video file
        prompts: Initial prompts for the first frame

    Returns:
        Tracker object
    """
    # Convert prompts
    points = []
    boxes = []
    labels = []

    for p in prompts:
        if p['type'] == 'point':
            points.append([p['x'], p['y']])
            labels.append(p.get('label', 1))
        elif p['type'] == 'box':
            boxes.append([p['x1'], p['y1'], p['x2'], p['y2']])

    # Create tracker
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
    """Add a prompt to a specific frame.

    Args:
        tracker: Tracker object
        frame_index: Frame index (0-based)
        prompt: Prompt dict
    """
    if frame_index not in tracker['frame_prompts']:
        tracker['frame_prompts'][frame_index] = []

    tracker['frame_prompts'][frame_index].append(prompt)


def jpy_sam2_video_propagate(tracker):
    """Propagate tracking through all frames.

    Args:
        tracker: Tracker object

    Returns:
        dict with tracking results for all frames
    """
    import cv2

    model = tracker['model']
    video_path = tracker['video_path']

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_results = []

    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check for frame-specific prompts
        if frame_idx in tracker['frame_prompts']:
            for p in tracker['frame_prompts'][frame_idx]:
                if p['type'] == 'point':
                    tracker['points'].append([p['x'], p['y']])
                    tracker['labels'].append(p.get('label', 1))
                elif p['type'] == 'box':
                    tracker['boxes'].append([p['x1'], p['y1'], p['x2'], p['y2']])

        # Run prediction on this frame
        kwargs = {}
        if tracker['points']:
            kwargs['points'] = tracker['points']
        if tracker['labels']:
            kwargs['labels'] = tracker['labels']
        if tracker['boxes']:
            kwargs['bboxes'] = tracker['boxes']

        results = model(frame, **kwargs)

        if results and results[0].masks is not None:
            for i in range(len(results[0].masks)):
                polygon = results[0].masks.xy[i].tolist() if hasattr(results[0].masks, 'xy') else []
                score = float(results[0].boxes.conf[i].cpu()) if results[0].boxes is not None else 1.0
                frame_results.append({
                    'frame_index': frame_idx,
                    'polygon': polygon,
                    'score': score,
                })

        frame_idx += 1

    cap.release()

    return {
        'total_frames': total_frames,
        'frames': frame_results,
    }

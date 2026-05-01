"""Streaming inference helper for jpy-ml — chunk-based video/webcam inference."""

_jpy_stream_results = None
_jpy_stream_exhausted = False
_jpy_stream_frame_index = 0


def jpy_stream_start(model_obj, source, kwargs):
    """Start streaming prediction. model_obj is the Ultralytics model object."""
    global _jpy_stream_results, _jpy_stream_exhausted, _jpy_stream_frame_index
    _jpy_stream_exhausted = False
    _jpy_stream_frame_index = 0
    _jpy_stream_results = model_obj(source, stream=True, **kwargs)


def jpy_stream_next(task_type, chunk_size=10, annotate=False):
    """Get next chunk of frames from the stream generator.

    Args:
        task_type: Task type string for result extraction.
        chunk_size: Number of frames per chunk.
        annotate: If True, include annotated JPEG image bytes.

    Returns a list of dicts. Each dict has 'result' (extracted result dict)
    and optionally 'image' (JPEG bytes) and 'frame_index' (int).
    Empty list means stream is exhausted.
    """
    global _jpy_stream_exhausted, _jpy_stream_frame_index
    chunk = []
    if _jpy_stream_results is None or _jpy_stream_exhausted:
        return chunk
    try:
        for r in _jpy_stream_results:
            entry = {
                'result': jpy_extract_result(r, task_type),
                'frame_index': _jpy_stream_frame_index,
            }
            if annotate:
                import cv2
                annotated = r.plot()  # numpy BGR image
                _, buf = cv2.imencode('.jpg', annotated)
                entry['image'] = buf.tobytes()
            _jpy_stream_frame_index += 1
            chunk.append(entry)
            if len(chunk) >= chunk_size:
                break
    except StopIteration:
        pass
    if len(chunk) < chunk_size:
        _jpy_stream_exhausted = True
    return chunk


def jpy_stream_cleanup():
    """Clean up streaming state."""
    global _jpy_stream_results, _jpy_stream_exhausted, _jpy_stream_frame_index
    _jpy_stream_results = None
    _jpy_stream_exhausted = False
    _jpy_stream_frame_index = 0

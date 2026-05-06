"""Progress file bridge for real-time training callbacks.

Provides a file-based communication channel from Python training callbacks
to Java. Python writes JSON lines after each epoch; Java polls the file
on a separate thread and fires callbacks in real-time.

Used by YOLO training (_jpy_training.py) and future LLM fine-tuning.
"""
import json
import os
import threading

_progress_file = None
_cancel_file = None
_progress_lock = threading.Lock()


def jpy_progress_init(progress_file, cancel_file=None):
    """Initialize progress writer. Creates/truncates the progress file.

    Args:
        progress_file: Path to the JSONL progress file.
        cancel_file: Path to the cancel signal file (optional).
    """
    global _progress_file, _cancel_file
    with _progress_lock:
        _progress_file = progress_file
        _cancel_file = cancel_file
    if progress_file:
        with open(progress_file, 'w') as f:
            f.write('')


def jpy_progress_write(event_type, data):
    """Append a JSON line to the progress file.

    Args:
        event_type: String like "epoch", "done".
        data: Dict of event-specific data.
    """
    global _progress_file
    if _progress_file is None:
        return
    entry = {'type': event_type}
    entry.update(data)
    line = json.dumps(entry, separators=(',', ':'))
    with _progress_lock:
        try:
            with open(_progress_file, 'a') as f:
                f.write(line + '\n')
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass


def jpy_progress_done(error=None):
    """Write the terminal 'done' event.

    Args:
        error: Error message string, or None if training completed successfully.
    """
    jpy_progress_write('done', {'error': error or ''})


def jpy_progress_check_cancel():
    """Check if cancellation has been requested via the cancel signal file.

    Returns True if the cancel marker file exists.
    """
    global _cancel_file
    if _cancel_file is None:
        return False
    return os.path.exists(_cancel_file)

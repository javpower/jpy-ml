"""LLM model download from HuggingFace Hub with progress reporting."""
import os
import json

_download_progress_file = None

def jpy_llm_download_init(progress_file):
    global _download_progress_file
    _download_progress_file = progress_file

def jpy_llm_download(model_id, cache_dir=None, progress_file=None):
    """Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
        cache_dir: Local cache directory. Defaults to ~/.jpy-ml/llm-models.
        progress_file: Path to JSONL progress file for download progress.

    Returns:
        dict with 'path' pointing to the local model directory.
    """
    global _download_progress_file
    _download_progress_file = progress_file
    if progress_file:
        jpy_progress_init(progress_file)

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.jpy-ml/llm-models")

    # Check if already cached (snapshot_download uses models--<id> directory)
    local_path = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
    if os.path.isdir(local_path):
        # Find the latest snapshot
        snapshots_dir = os.path.join(local_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            versions = os.listdir(snapshots_dir)
            if versions:
                latest = os.path.join(snapshots_dir, versions[-1])
                if os.path.exists(os.path.join(latest, "config.json")):
                    jpy_progress_write("download_done", {"path": latest, "cached": True})
                    return {"path": latest, "cached": True}

    from huggingface_hub import snapshot_download

    jpy_progress_write("download", {"model_id": model_id, "status": "starting"})

    path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
    )

    jpy_progress_write("download_done", {"path": path, "cached": False})
    return {"path": path, "cached": False}

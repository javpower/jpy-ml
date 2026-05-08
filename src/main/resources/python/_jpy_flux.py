"""FLUX.1 image generation via diffusers."""
import os
import gc
import torch
import threading
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

_jpy_flux_cache = {}
_jpy_flux_lock = threading.Lock()


def _detect_device(device):
    if device == "auto" or device is None:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def _get_dtype(device, dtype_str):
    if dtype_str and dtype_str != "auto":
        return getattr(torch, dtype_str)
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


def jpy_flux_load(model_id, device="auto", dtype="auto", variant=None):
    """Load FLUX.1 pipeline.

    Args:
        model_id: HuggingFace model ID or local path.
            - "black-forest-labs/FLUX.1-dev"
            - "black-forest-labs/FLUX.1-schnell"
        device: "auto", "cuda", "mps", "cpu"
        dtype: "auto", "float16", "bfloat16", "float32"
        variant: "fp16", None

    Returns:
        dict with model info.
    """
    from diffusers import FluxPipeline

    device = _detect_device(device)
    torch_dtype = _get_dtype(device, dtype)

    cache_key = (model_id, device, str(torch_dtype))
    with _jpy_flux_lock:
        if cache_key in _jpy_flux_cache:
            return {"status": "cached", "model_id": model_id, "device": device}

    kwargs = {"torch_dtype": torch_dtype}
    if variant:
        kwargs["variant"] = variant

    if device == "mps":
        pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
        pipe = pipe.to("mps")
    elif device == "cpu":
        pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
        pipe = pipe.to("cpu")
    else:
        pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
        pipe = pipe.to("cuda")

    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass

    with _jpy_flux_lock:
        _jpy_flux_cache[cache_key] = pipe

    return {"status": "loaded", "model_id": model_id, "device": device}


def jpy_flux_generate(model_id, prompt, output_path,
                       device="auto", dtype="auto", variant=None,
                       width=1024, height=1024,
                       steps=20, guidance=3.5, seed=-1,
                       negative_prompt=None, num_images=1):
    """Generate image(s) from text prompt.

    Args:
        model_id: HuggingFace model ID or local path.
        prompt: Text prompt.
        output_path: Path to save generated image.
        device: "auto", "cuda", "mps", "cpu"
        dtype: "auto", "float16", "bfloat16", "float32"
        variant: "fp16", None
        width: Image width (default 1024).
        height: Image height (default 1024).
        steps: Number of inference steps (default 20).
        guidance: Guidance scale (default 3.5).
        seed: Random seed (-1 for random).
        negative_prompt: Negative prompt (optional).
        num_images: Number of images to generate.

    Returns:
        dict with output paths and metadata.
    """
    from diffusers import FluxPipeline
    import time

    device = _detect_device(device)
    torch_dtype = _get_dtype(device, dtype)

    cache_key = (model_id, device, str(torch_dtype))
    with _jpy_flux_lock:
        pipe = _jpy_flux_cache.get(cache_key)

    if pipe is None:
        load_result = jpy_flux_load(model_id, device, dtype, variant)
        with _jpy_flux_lock:
            pipe = _jpy_flux_cache.get(cache_key)

    if pipe is None:
        return {"error": "Failed to load model"}

    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    start_time = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        num_images_per_prompt=num_images,
    )

    elapsed = time.time() - start_time

    output_paths = []
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    ext = os.path.splitext(output_path)[1] or ".png"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, image in enumerate(result.images):
        if num_images == 1:
            path = output_path
        else:
            path = os.path.join(output_dir, f"{base_name}_{i}{ext}")
        image.save(path)
        output_paths.append(path)

    return {
        "status": "success",
        "output_paths": output_paths,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        "seed": seed if seed >= 0 else None,
        "elapsed_seconds": round(elapsed, 2),
        "model_id": model_id,
        "prompt": prompt,
    }


def jpy_flux_img2img(model_id, prompt, input_image_path, output_path,
                       device="auto", dtype="auto", variant=None,
                       strength=0.75, steps=20, guidance=3.5, seed=-1,
                       negative_prompt=None):
    """Image-to-image generation.

    Args:
        model_id: HuggingFace model ID or local path.
        prompt: Text prompt.
        input_image_path: Path to input image.
        output_path: Path to save generated image.
        device: "auto", "cuda", "mps", "cpu"
        dtype: "auto", "float16", "bfloat16", "float32"
        variant: "fp16", None
        strength: Transformation strength (0-1, default 0.75).
        steps: Number of inference steps (default 20).
        guidance: Guidance scale (default 3.5).
        seed: Random seed (-1 for random).
        negative_prompt: Negative prompt (optional).

    Returns:
        dict with output path and metadata.
    """
    from diffusers import FluxImg2ImgPipeline
    from PIL import Image
    import time

    device = _detect_device(device)
    torch_dtype = _get_dtype(device, dtype)

    pipe = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    if device == "mps":
        pipe = pipe.to("mps")
    elif device == "cuda":
        pipe = pipe.to("cuda")
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    input_image = Image.open(input_image_path).convert("RGB")

    start_time = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    elapsed = time.time() - start_time

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result.images[0].save(output_path)

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "success",
        "output_path": output_path,
        "strength": strength,
        "steps": steps,
        "guidance": guidance,
        "elapsed_seconds": round(elapsed, 2),
        "model_id": model_id,
        "prompt": prompt,
    }


def jpy_flux_unload(model_id=None, device="auto"):
    """Unload FLUX model from memory.

    Args:
        model_id: If specified, only unload that model. If None, unload all.
        device: Device hint.
    """
    import gc
    with _jpy_flux_lock:
        if model_id:
            keys = [k for k in _jpy_flux_cache if k[0] == model_id]
        else:
            keys = list(_jpy_flux_cache.keys())

        for k in keys:
            pipe = _jpy_flux_cache.pop(k, None)
            if pipe:
                del pipe

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def jpy_flux_list_models():
    """List available FLUX models."""
    return {
        "models": [
            {
                "id": "black-forest-labs/FLUX.1-dev",
                "name": "FLUX.1 Dev",
                "description": "高质量版本，需要授权",
                "steps": 20,
                "guidance": 3.5,
            },
            {
                "id": "black-forest-labs/FLUX.1-schnell",
                "name": "FLUX.1 Schnell",
                "description": "快速版本，Apache 2.0 许可",
                "steps": 4,
                "guidance": 0.0,
            },
        ]
    }

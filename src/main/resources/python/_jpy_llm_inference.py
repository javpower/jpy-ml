"""LLM inference (chat completion) for fine-tuned models."""
import os
import sys
import torch
import platform
import threading

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

_jpy_llm_cache = {}
_jpy_llm_cache_lock = threading.Lock()

def _detect_device(device):
    if device == "auto" or device is None:
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def jpy_llm_chat(model_path, adapter_path, messages, gen_kwargs, quantization=None):
    """Synchronous chat completion.

    Args:
        model_path: Path to base model.
        adapter_path: Path to LoRA adapter, or None.
        messages: List of {role, content} dicts.
        gen_kwargs: Generation config dict.
        quantization: "nf4", "int8", or None.

    Returns:
        dict with content, finish_reason, prompt_tokens, completion_tokens.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _detect_device(gen_kwargs.get("device"))
    gen_kwargs = dict(gen_kwargs)
    gen_kwargs.pop("device", None)

    cache_key = (model_path, adapter_path or "", quantization or "")
    with _jpy_llm_cache_lock:
        cached = _jpy_llm_cache.get(cache_key)

    if cached is not None:
        model = cached["model"]
        tokenizer = cached["tokenizer"]
    else:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        if device == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}
        elif device == "mps":
            model_kwargs["device_map"] = {"": "cpu"}
        else:
            model_kwargs["device_map"] = "auto"

        if quantization and quantization != "none":
            from transformers import BitsAndBytesConfig
            if quantization == "nf4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif quantization == "int8":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        with suppress_stderr_fd():
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        if device == "mps":
            model = model.to("mps")

        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        with _jpy_llm_cache_lock:
            _jpy_llm_cache[cache_key] = {"model": model, "tokenizer": tokenizer}

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_kwargs.get("max_new_tokens", 512),
            temperature=gen_kwargs.get("temperature", 0.7),
            top_p=gen_kwargs.get("top_p", 0.9),
            do_sample=gen_kwargs.get("do_sample", True),
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.1),
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "content": response,
        "finish_reason": "stop",
        "prompt_tokens": inputs["input_ids"].shape[1],
        "completion_tokens": len(new_tokens),
    }


def jpy_llm_unload(model_var):
    """Unload model from GPU memory."""
    import gc
    if model_var is not None:
        del model_var
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def jpy_llm_unload_cache(model_path=None):
    """Unload cached LLM models to free GPU memory.

    Args:
        model_path: If specified, only unload that model. If None, unload all.
    """
    import gc
    with _jpy_llm_cache_lock:
        if model_path is not None:
            keys_to_remove = [k for k in _jpy_llm_cache if k[0] == model_path]
        else:
            keys_to_remove = list(_jpy_llm_cache.keys())

        for k in keys_to_remove:
            entry = _jpy_llm_cache.pop(k, None)
            if entry is not None:
                del entry["model"]
                del entry["tokenizer"]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

"""Merge LoRA adapter back into base model for deployment."""
import os
import sys

def jpy_llm_merge(model_path, adapter_path, output_path=None):
    """Merge LoRA adapter weights into the base model.

    Args:
        model_path: Path to base model.
        adapter_path: Path to LoRA adapter directory.
        output_path: Where to save merged model. Defaults to adapter_dir/merged.

    Returns:
        dict with merged_path.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Suppress harmless JVM error from safetensors Rust fork on macOS
    old_stderr = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(adapter_path), "merged")
    os.makedirs(output_path, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return {"merged_path": output_path}

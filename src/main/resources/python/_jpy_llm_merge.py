"""Merge LoRA adapter back into base model for deployment."""
import os

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

    with suppress_stderr_fd():
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(adapter_path), "merged")
    os.makedirs(output_path, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return {"merged_path": output_path}

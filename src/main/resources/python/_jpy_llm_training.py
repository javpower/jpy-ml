"""LLM fine-tuning via QLoRA/LoRA using HuggingFace ecosystem."""
import os
import sys
import json
import platform
import inspect
import torch

# --- Platform detection ---

def _detect_device_and_quantization(device, quantization):
    """Auto-detect best device and quantization based on platform."""
    system = platform.system()
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    if quantization == "auto":
        if has_cuda:
            quantization = "nf4"
        else:
            quantization = None

    if device == "auto" or device is None:
        if has_cuda:
            device = "cuda:0"
        elif has_mps:
            device = "mps"
        else:
            device = "cpu"

    return device, quantization


# --- LoRA target modules ---

_TARGET_MODULES_MAP = {
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama": ["q_proj", "v_proj"],
    "gemma": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "deepseek": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "phi": ["q_proj", "v_proj"],
    "yi": ["q_proj", "v_proj"],
    "internlm": ["q_proj", "v_proj"],
}

def _detect_target_modules(model_path):
    """Detect LoRA target modules from model path or config."""
    name = model_path.lower()
    for key, modules in _TARGET_MODULES_MAP.items():
        if key in name:
            return modules
    # Fallback: try to read model config
    try:
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", "")
            for key, modules in _TARGET_MODULES_MAP.items():
                if key in model_type:
                    return modules
    except Exception:
        pass
    return ["q_proj", "v_proj"]


# --- Data formatting ---

def _format_conversation(example, tokenizer):
    """Convert dataset example to text using chat template."""
    if "messages" in example:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    elif "instruction" in example:
        prompt = f"### Instruction:\n{example['instruction']}\n"
        if example.get("input"):
            prompt += f"### Input:\n{example['input']}\n"
        prompt += f"### Response:\n{example['output']}"
        return {"text": prompt}
    return {"text": example.get("text", "")}


# --- Training ---

_train_log = []


def _build_bnb_config(quantization):
    """Build BitsAndBytes config from quantization string."""
    if quantization is None or quantization == "none":
        return None
    from transformers import BitsAndBytesConfig
    if quantization == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def jpy_llm_train(model_path, dataset_path, lora_kwargs, train_kwargs,
                  progress_file=None, cancel_file=None):
    """Run LLM fine-tuning with QLoRA/LoRA.

    Args:
        model_path: Path to base model (local or HuggingFace cache).
        dataset_path: Path to training data (JSONL).
        lora_kwargs: LoRA config dict (rank, alpha, target_modules, dropout, bias).
        train_kwargs: Training hyperparameters dict.
        progress_file: Path to JSONL progress file for real-time callbacks.
        cancel_file: Path to cancel signal file.

    Returns:
        dict with adapter_path, log, final_loss.
    """
    global _train_log
    _train_log = []

    if progress_file:
        jpy_progress_init(progress_file, cancel_file)

    # Ensure Java Maps from Jep are fully converted to native Python dicts
    lora_kwargs = dict(lora_kwargs) if lora_kwargs else {}
    train_kwargs = dict(train_kwargs) if train_kwargs else {}

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainerCallback
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    # --- Device & quantization ---
    device = train_kwargs.get("device")
    quantization = train_kwargs.get("quantization")
    device, quantization = _detect_device_and_quantization(device, quantization)
    train_kwargs = dict(train_kwargs)  # copy to avoid mutation

    jpy_progress_write("info", {
        "device": device,
        "quantization": str(quantization),
        "status": "loading_model"
    })

    # --- Load model ---
    bnb_config = _build_bnb_config(quantization)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if device == "cpu":
        model_kwargs["device_map"] = {"": "cpu"}
    elif device == "mps":
        # MPS doesn't support device_map="auto", load to CPU first then move
        model_kwargs["device_map"] = {"": "cpu"}
    else:
        model_kwargs["device_map"] = "auto"
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    # Suppress harmless JVM error from safetensors Rust fork on macOS
    old_stderr = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

    # Move model to MPS device after loading
    if device == "mps":
        model = model.to("mps")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- LoRA ---
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    target_modules = lora_kwargs.get("target_modules")
    if target_modules is None:
        target_modules = _detect_target_modules(model_path)

    lora_config = LoraConfig(
        r=lora_kwargs.get("rank", 16),
        lora_alpha=lora_kwargs.get("alpha", 32),
        target_modules=target_modules,
        lora_dropout=lora_kwargs.get("dropout", 0.05),
        bias=lora_kwargs.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = model.print_trainable_parameters()

    jpy_progress_write("info", {"status": "model_loaded", "trainable_params": str(trainable)})

    error_msg = None
    try:
        # --- Dataset ---
        dataset = load_dataset("json", data_files=dataset_path)
        dataset = dataset.map(lambda x: _format_conversation(x, tokenizer))

        # --- Training args ---
        output_dir = train_kwargs.pop("output_dir", os.path.expanduser("~/.jpy-ml/llm-output"))
        os.makedirs(output_dir, exist_ok=True)

        # Remove non-TrainingArguments keys
        train_kwargs.pop("device", None)
        train_kwargs.pop("quantization", None)

        # SFTConfig extends TrainingArguments with SFT-specific params
        sft_config_params = set(inspect.signature(SFTConfig.__init__).parameters)

        _config_kwargs = dict(
            output_dir=output_dir,
            num_train_epochs=train_kwargs.get("epochs", 3),
            per_device_train_batch_size=train_kwargs.get("batch_size", 4),
            gradient_accumulation_steps=train_kwargs.get("gradient_accumulation", 4),
            learning_rate=train_kwargs.get("learning_rate", 2e-4),
            lr_scheduler_type=train_kwargs.get("lr_scheduler", "cosine"),
            warmup_steps=train_kwargs.get("warmup_steps", 100),
            max_grad_norm=1.0,
            logging_steps=train_kwargs.get("logging_steps", 10),
            save_steps=train_kwargs.get("save_steps", 500),
            save_total_limit=2,
            bf16=(device != "cpu" and not bnb_config),
            fp16=False,
            gradient_checkpointing=train_kwargs.get("gradient_checkpointing", True),
            seed=train_kwargs.get("seed", 42),
            report_to="none",
            optim="adamw_torch",
        )
        # max_seq_length -> max_length for trl >= 1.0 (SFTConfig)
        if "max_length" in sft_config_params:
            _config_kwargs["max_length"] = train_kwargs.get("max_seq_length", 2048)

        training_args = SFTConfig(**_config_kwargs)

        # Build callback that implements both progress writing and cancellation
        class _ProgressCB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                global _train_log
                if logs is None:
                    return
                entry = {
                    "step": state.global_step,
                    "epoch": round(state.epoch, 2),
                    "loss": logs.get("loss"),
                    "learning_rate": logs.get("learning_rate"),
                }
                _train_log.append(entry)
                jpy_progress_write("step", entry)
                if jpy_progress_check_cancel():
                    control.should_training_stop = True

        # SFTTrainer: trl >= 0.12 uses 'processing_class' instead of 'tokenizer'
        _sft_params = set(inspect.signature(SFTTrainer.__init__).parameters)
        _trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            callbacks=[_ProgressCB()],
        )
        if "processing_class" in _sft_params:
            _trainer_kwargs["processing_class"] = tokenizer
        else:
            _trainer_kwargs["tokenizer"] = tokenizer

        trainer = SFTTrainer(**_trainer_kwargs)

        jpy_progress_write("info", {"status": "training_started"})
        trainer.train()
    except Exception as e:
        error_msg = str(e)
        raise
    finally:
        if progress_file:
            jpy_progress_done(error_msg)

    # --- Save adapter ---
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    final_loss = _train_log[-1].get("loss") if _train_log else None
    return {
        "adapter_path": adapter_path,
        "log": list(_train_log),
        "final_loss": final_loss,
    }

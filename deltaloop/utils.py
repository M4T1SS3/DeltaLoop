"""Common utility functions for DeltaLoop."""

import os
from pathlib import Path
from typing import Optional


def get_dir_size(path: str) -> int:
    """Calculate the total size of a directory in bytes.

    Args:
        path: Directory path

    Returns:
        Total size in bytes
    """
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except PermissionError:
        pass
    return total


def ensure_dir(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_model(base_model: str, adapter: Optional[str] = None, device: str = "auto", **kwargs):
    """Load a model with optional LoRA adapter for inference.

    This function handles loading base models and merging LoRA adapters,
    providing a simple interface for using fine-tuned models in production.

    Args:
        base_model: Name or path to base model (e.g., "unsloth/mistral-7b-bnb-4bit")
        adapter: Optional path to LoRA adapter weights directory
        device: Device to load on ("auto", "cuda", "cpu", etc.)
        **kwargs: Additional arguments passed to model loading

    Returns:
        Tuple of (model, tokenizer) ready for inference

    Raises:
        ImportError: If required dependencies are not installed
        FileNotFoundError: If adapter path doesn't exist

    Example:
        >>> # Load base model only
        >>> model, tokenizer = load_model("mistralai/Mistral-7B-v0.1")
        >>>
        >>> # Load base model + adapter
        >>> model, tokenizer = load_model(
        ...     "unsloth/mistral-7b-bnb-4bit",
        ...     adapter="data/models/v1"
        ... )
        >>>
        >>> # Generate text
        >>> inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_length=100)
        >>> print(tokenizer.decode(outputs[0]))
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "Model loading requires transformers and torch.\n"
            "Install with: pip install transformers torch"
        )

    print(f"[DeltaLoop] Loading base model: {base_model}")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model_kwargs = {
        "device_map": device if device != "cpu" else None,
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        **kwargs
    }

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Load and merge adapter if provided
    if adapter:
        if not Path(adapter).exists():
            raise FileNotFoundError(f"Adapter not found: {adapter}")

        print(f"[DeltaLoop] Loading LoRA adapter: {adapter}")

        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "LoRA adapter loading requires PEFT.\n"
                "Install with: pip install peft"
            )

        # Load adapter
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[DeltaLoop] LoRA adapter loaded successfully")

        # Optional: merge adapter weights for faster inference
        # Uncomment if you want permanent merging:
        # print("[DeltaLoop] Merging adapter weights...")
        # model = model.merge_and_unload()

    print(f"[DeltaLoop] Model ready on device: {model.device}")

    return model, tokenizer


def load_model_for_training(base_model: str, **kwargs):
    """Load a model specifically configured for training.

    This is a convenience wrapper used internally by the training pipeline.
    For inference, use load_model() instead.

    Args:
        base_model: Name or path to base model
        **kwargs: Arguments passed to model loading

    Returns:
        Tuple of (model, tokenizer) configured for training
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Training requires Unsloth.\n"
            "Install with: pip install unsloth"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        **kwargs
    )

    return model, tokenizer

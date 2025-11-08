"""Training orchestration for DeltaLoop.

This module coordinates fine-tuning across different backends (Unsloth, Axolotl, etc.)
and provides a unified interface for training LoRA adapters.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning.

    Args:
        dataset_path: Path to training dataset (JSONL)
        base_model: Base model name or path (e.g., "mistralai/Mistral-7B-v0.1")
        output_dir: Directory to save adapter weights
        backend: Training backend ("unsloth", "axolotl", or "peft")

        # LoRA parameters
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha scaling (default: 32, typically 2x rank)
        lora_dropout: Dropout for LoRA layers (default: 0.05)

        # Training parameters
        max_steps: Maximum training steps (default: 500)
        learning_rate: Learning rate (default: 2e-4)
        per_device_batch_size: Batch size per device (default: 2)
        gradient_accumulation_steps: Gradient accumulation (default: 4)
        warmup_steps: Warmup steps (default: 10)

        # Model parameters
        max_seq_length: Maximum sequence length (default: 2048)
        load_in_4bit: Use 4-bit quantization (default: True)

        # Logging
        logging_steps: Steps between logs (default: 10)
        save_steps: Steps between checkpoints (default: 100)
    """

    # Required
    dataset_path: str
    base_model: str
    output_dir: str
    backend: str = "unsloth"  # "unsloth", "transformers", or "dpo"

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training parameters
    max_steps: int = 500
    learning_rate: float = 2e-4
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10

    # Model parameters
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "dataset_path": self.dataset_path,
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "backend": self.backend,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
        }

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TrainingResult:
    """Results from a training run."""

    output_dir: str
    adapter_size_mb: float
    steps_completed: int
    final_loss: Optional[float] = None
    training_time_seconds: Optional[float] = None
    config: Optional[Dict[str, Any]] = None


class Trainer:
    """Orchestrates model fine-tuning across different backends.

    Args:
        config: Training configuration
        verbose: Print progress information (default: True)
    """

    def __init__(self, config: TrainingConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._backend = None

    def train(self) -> TrainingResult:
        """Run fine-tuning with configured backend.

        Returns:
            TrainingResult with output directory and metrics

        Raises:
            ValueError: If backend is not supported
            ImportError: If backend dependencies are not installed
        """
        if self.verbose:
            print("=" * 60)
            print(f"DeltaLoop Training")
            print("=" * 60)
            print(f"Backend:      {self.config.backend}")
            print(f"Base Model:   {self.config.base_model}")
            print(f"Dataset:      {self.config.dataset_path}")
            print(f"Output:       {self.config.output_dir}")
            print(f"Max Steps:    {self.config.max_steps}")
            print(f"LoRA Rank:    {self.config.lora_r}")
            print("=" * 60)
            print()

        # Validate dataset exists
        if not Path(self.config.dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")

        # Load backend (unless already set, e.g., for testing)
        if self._backend is None:
            backend = self._load_backend(self.config.backend)
        else:
            backend = self._backend

        # Run training
        result = backend.train(self.config)

        # Save config alongside model
        config_path = Path(self.config.output_dir) / "training_config.json"
        self.config.save(str(config_path))

        if self.verbose:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
            print(f"Output Dir:   {result.output_dir}")
            print(f"Adapter Size: {result.adapter_size_mb:.2f} MB")
            print(f"Steps:        {result.steps_completed}")
            if result.final_loss:
                print(f"Final Loss:   {result.final_loss:.4f}")
            if result.training_time_seconds:
                print(f"Time:         {result.training_time_seconds:.1f}s")
            print("=" * 60)

        return result

    def _load_backend(self, backend_name: str):
        """Load the appropriate training backend."""
        if backend_name == "unsloth":
            # Check if unsloth is available before trying to use it
            try:
                import unsloth
                from backends.unsloth_backend import UnslothBackend
                return UnslothBackend(verbose=self.verbose)
            except ImportError as e:
                # Auto-fallback to transformers backend
                if self.verbose:
                    print(f"[Note] Unsloth not available, falling back to transformers backend")
                from backends.transformers_backend import TransformersBackend
                return TransformersBackend(verbose=self.verbose)

        elif backend_name == "transformers":
            from backends.transformers_backend import TransformersBackend
            return TransformersBackend(verbose=self.verbose)

        elif backend_name == "dpo":
            from backends.dpo_backend import DPOBackend
            return DPOBackend(verbose=self.verbose)

        elif backend_name == "axolotl":
            raise NotImplementedError(
                "Axolotl backend coming soon! Use 'unsloth' or 'transformers' for now."
            )

        elif backend_name == "peft":
            raise NotImplementedError(
                "Raw PEFT backend coming soon! Use 'unsloth' or 'transformers' for now."
            )

        else:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Supported: 'unsloth', 'transformers', 'dpo', 'axolotl', 'peft'"
            )


def train(
    dataset_path: str,
    base_model: str = "unsloth/mistral-7b-bnb-4bit",
    output_dir: str = "data/models/run_1",
    backend: str = "unsloth",  # "unsloth", "transformers", or "dpo"
    max_steps: int = 500,
    lora_r: int = 16,
    learning_rate: float = 2e-4,
    verbose: bool = True,
    **kwargs
) -> TrainingResult:
    """Convenience function to train a model.

    Args:
        dataset_path: Path to training dataset (JSONL)
        base_model: Base model name (default: unsloth/mistral-7b-bnb-4bit)
        output_dir: Output directory for adapter
        backend: Training backend (default: unsloth)
        max_steps: Training steps (default: 500)
        lora_r: LoRA rank (default: 16)
        learning_rate: Learning rate (default: 2e-4)
        verbose: Print progress (default: True)
        **kwargs: Additional TrainingConfig parameters

    Returns:
        TrainingResult with output directory and metrics

    Example:
        >>> from deltaloop.train import train
        >>> result = train(
        ...     dataset_path="data/datasets/train.jsonl",
        ...     base_model="unsloth/mistral-7b-bnb-4bit",
        ...     max_steps=500
        ... )
    """
    config = TrainingConfig(
        dataset_path=dataset_path,
        base_model=base_model,
        output_dir=output_dir,
        backend=backend,
        max_steps=max_steps,
        lora_r=lora_r,
        learning_rate=learning_rate,
        **kwargs
    )

    trainer = Trainer(config, verbose=verbose)
    return trainer.train()

"""Training backends for DeltaLoop.

This package contains different backend implementations for fine-tuning:
    - unsloth_backend: Fast training with Unsloth (2x faster, 50% less memory)
    - axolotl_backend: Configuration-based training with Axolotl (coming soon)
    - peft_backend: Raw PEFT/LoRA implementation (coming soon)

Each backend implements a standard interface for training LoRA adapters.
"""

__all__ = []

# Backends are imported lazily to avoid requiring all dependencies

try:
    from .unsloth_backend import UnslothBackend
    __all__.append("UnslothBackend")
except ImportError:
    pass

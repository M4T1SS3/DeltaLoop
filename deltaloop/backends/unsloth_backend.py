"""Unsloth training backend for DeltaLoop.

This backend uses Unsloth for 2x faster fine-tuning with 50% less memory.

Unsloth optimizations:
    - Optimized attention kernels
    - 4-bit quantization support
    - Gradient checkpointing
    - Memory-efficient LoRA

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install transformers datasets trl
"""

import os
import time
from pathlib import Path
from typing import Optional
from deltaloop.train import TrainingConfig, TrainingResult
from deltaloop.utils import get_dir_size


class UnslothBackend:
    """Unsloth training backend.

    Args:
        verbose: Print progress information (default: True)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train model using Unsloth.

        Args:
            config: Training configuration

        Returns:
            TrainingResult with metrics

        Raises:
            ImportError: If Unsloth dependencies are not installed
        """
        start_time = time.time()

        # Import dependencies (fail early if not installed)
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from datasets import load_dataset
            import torch
        except ImportError as e:
            raise ImportError(
                f"Unsloth backend dependencies not installed.\n"
                f"Install with: pip install unsloth transformers datasets trl\n"
                f"Error: {e}"
            )

        if self.verbose:
            print(f"[Unsloth] Loading base model: {config.base_model}")

        # Load model with 4-bit quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            dtype=None,  # Auto-detect best dtype
            load_in_4bit=config.load_in_4bit,
        )

        if self.verbose:
            print(f"[Unsloth] Adding LoRA adapters (rank={config.lora_r})")

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
            random_state=42,
        )

        if self.verbose:
            print(f"[Unsloth] Loading dataset: {config.dataset_path}")

        # Load dataset
        dataset = load_dataset(
            "json",
            data_files=config.dataset_path,
            split="train"
        )

        if self.verbose:
            print(f"[Unsloth] Dataset size: {len(dataset)} examples")

        # Format dataset for training
        def format_prompt(example):
            """Format example for training."""
            # Alpaca format
            if "instruction" in example:
                if example.get("input", "").strip():
                    text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
                else:
                    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

            # ChatML format
            elif "messages" in example:
                messages = example["messages"]
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|{role}|>\n{content}\n"

            # Raw format
            elif "prompt" in example:
                text = f"{example['prompt']}\n\n{example['completion']}"

            else:
                raise ValueError(f"Unknown dataset format. Keys: {example.keys()}")

            return {"text": text}

        dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

        if self.verbose:
            print(f"[Unsloth] Starting training...")
            print(f"  Max steps: {config.max_steps}")
            print(f"  Batch size: {config.per_device_batch_size}")
            print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
            print(f"  Effective batch size: {config.per_device_batch_size * config.gradient_accumulation_steps}")
            print(f"  Learning rate: {config.learning_rate}")
            print()

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config.logging_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=3,  # Keep only last 3 checkpoints
            optim="adamw_8bit",  # Memory-efficient optimizer
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            args=training_args,
        )

        # Train!
        train_result = trainer.train()

        # Extract final loss
        final_loss = None
        if hasattr(train_result, "training_loss"):
            final_loss = train_result.training_loss

        if self.verbose:
            print(f"\n[Unsloth] Training complete!")
            print(f"[Unsloth] Saving model to {config.output_dir}")

        # Save model and tokenizer
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

        # Calculate metrics
        training_time = time.time() - start_time
        adapter_size = get_dir_size(config.output_dir) / (1024 * 1024)  # MB

        if self.verbose:
            print(f"[Unsloth] Adapter size: {adapter_size:.2f} MB")
            print(f"[Unsloth] Training time: {training_time:.1f}s")

        return TrainingResult(
            output_dir=config.output_dir,
            adapter_size_mb=adapter_size,
            steps_completed=config.max_steps,
            final_loss=final_loss,
            training_time_seconds=training_time,
            config=config.to_dict()
        )


class MockUnslothBackend:
    """Mock backend for testing without GPU.

    This allows testing the training pipeline without actually
    fine-tuning a model. Useful for CI/CD and development.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Mock training that doesn't require GPU."""
        if self.verbose:
            print("[Mock Unsloth] Running mock training...")
            print("[Mock Unsloth] (No actual training performed)")

        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create mock adapter files
        (output_dir / "adapter_model.bin").write_text("mock adapter weights")
        (output_dir / "adapter_config.json").write_text('{"mock": true}')

        # Simulate training time
        import time
        time.sleep(2)

        if self.verbose:
            print(f"[Mock Unsloth] Created mock adapter at {config.output_dir}")

        return TrainingResult(
            output_dir=config.output_dir,
            adapter_size_mb=0.001,  # Very small mock file
            steps_completed=config.max_steps,
            final_loss=0.5,
            training_time_seconds=2.0,
            config=config.to_dict()
        )

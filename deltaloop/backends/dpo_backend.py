"""DPO (Direct Preference Optimization) training backend for DeltaLoop.

This backend uses TRL's DPOTrainer for state-of-the-art preference learning.

DPO advantages over standard SFT:
    - Learns from both success AND failure examples
    - Understands relative quality (what's better vs worse)
    - More robust to noisy data
    - Better alignment with desired behavior

Requirements:
    pip install transformers torch peft trl datasets
"""

import os
import time
from pathlib import Path
from typing import Optional
from deltaloop.train import TrainingConfig, TrainingResult
from deltaloop.utils import get_dir_size


class DPOBackend:
    """DPO training backend using TRL.

    Args:
        verbose: Print progress information (default: True)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train model using DPO (Direct Preference Optimization).

        Args:
            config: Training configuration

        Returns:
            TrainingResult with metrics

        Raises:
            ImportError: If dependencies are not installed
        """
        start_time = time.time()

        # Import dependencies
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments
            )
            from peft import get_peft_model, LoraConfig, TaskType
            from trl import DPOTrainer, DPOConfig
            from datasets import load_dataset
            import torch
        except ImportError as e:
            raise ImportError(
                f"DPO backend dependencies not installed.\n"
                f"Install with: pip install transformers torch peft trl datasets\n"
                f"Error: {e}"
            )

        if self.verbose:
            print(f"[DPO] Loading base model: {config.base_model}")

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        if self.verbose:
            print(f"[DPO] Using device: {device}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {}
        if config.load_in_4bit and device == "cuda":
            # 4-bit only works on CUDA
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            # For MPS/CPU
            model_kwargs["torch_dtype"] = torch.float32 if device == "cpu" else torch.float16
            if device == "mps":
                model_kwargs["low_cpu_mem_usage"] = True
                model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            **model_kwargs
        )

        # For MPS, move model explicitly
        if device == "mps":
            model = model.to("mps")

        # Create reference model for DPO (frozen copy)
        if self.verbose:
            print(f"[DPO] Creating reference model (frozen copy)")

        ref_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            **model_kwargs
        )
        if device == "mps":
            ref_model = ref_model.to("mps")
        ref_model.eval()  # Freeze reference model

        if self.verbose:
            print(f"[DPO] Adding LoRA adapters (rank={config.lora_r})")

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )

        # Add LoRA to trainable model only
        model = get_peft_model(model, peft_config)

        if self.verbose:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"[DPO] Trainable params: {trainable_params:,} / {total_params:,} "
                  f"({100 * trainable_params / total_params:.2f}%)")

        # Load dataset
        if self.verbose:
            print(f"[DPO] Loading preference pairs: {config.dataset_path}")

        dataset = load_dataset('json', data_files=config.dataset_path, split='train')

        if self.verbose:
            print(f"[DPO] Loaded {len(dataset)} preference pairs")
            if len(dataset) > 0:
                print(f"[DPO] Example pair keys: {list(dataset[0].keys())}")

        # DPO Training Configuration
        training_args = DPOConfig(
            output_dir=config.output_dir,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=1,
            warmup_steps=config.warmup_steps,
            fp16=device == "cuda",  # Use fp16 only on CUDA
            remove_unused_columns=False,
            report_to="none",  # Disable wandb, etc.
            beta=0.1,  # DPO temperature parameter
            max_length=config.max_seq_length,
            max_prompt_length=config.max_seq_length // 2,
        )

        # Create DPO trainer
        # Note: peft_config is not passed because we already applied PEFT to the model
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,  # TRL 0.24+ uses processing_class instead of tokenizer
        )

        if self.verbose:
            print(f"[DPO] Starting preference learning for {config.max_steps} steps...")
            print(f"[DPO] Learning from chosen vs rejected responses...")

        # Train
        train_result = trainer.train()

        # Save adapter
        if self.verbose:
            print(f"[DPO] Saving adapter to {config.output_dir}")

        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

        # Calculate metrics
        training_time = time.time() - start_time
        adapter_size_mb = get_dir_size(config.output_dir)

        return TrainingResult(
            output_dir=config.output_dir,
            adapter_size_mb=adapter_size_mb,
            steps_completed=config.max_steps,
            final_loss=train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            training_time_seconds=training_time
        )

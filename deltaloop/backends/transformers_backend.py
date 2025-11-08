"""Standard Transformers training backend for DeltaLoop.

This backend uses standard Transformers + PEFT for fine-tuning.
Works with any HuggingFace model without requiring Unsloth.

Requirements:
    pip install transformers torch peft trl datasets
"""

import os
import time
from pathlib import Path
from typing import Optional
from deltaloop.train import TrainingConfig, TrainingResult
from deltaloop.utils import get_dir_size


class TransformersBackend:
    """Standard Transformers training backend.

    Args:
        verbose: Print progress information (default: True)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train model using Transformers + PEFT.

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
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling
            )
            from peft import get_peft_model, LoraConfig, TaskType
            from datasets import load_dataset
            import torch
        except ImportError as e:
            raise ImportError(
                f"Transformers backend dependencies not installed.\n"
                f"Install with: pip install transformers torch peft trl datasets\n"
                f"Error: {e}"
            )

        if self.verbose:
            print(f"[Transformers] Loading base model: {config.base_model}")

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        if self.verbose:
            print(f"[Transformers] Using device: {device}")

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
            # For MPS/CPU, use float32 or float16
            # On MPS, use lower memory with flash attention disabled
            model_kwargs["torch_dtype"] = torch.float32 if device == "cpu" else torch.float16
            if device == "mps":
                model_kwargs["low_cpu_mem_usage"] = True
                model_kwargs["attn_implementation"] = "eager"  # Avoid memory-intensive SDPA

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            **model_kwargs
        )

        # For MPS, move model explicitly
        if device == "mps":
            model = model.to("mps")

        if self.verbose:
            print(f"[Transformers] Adding LoRA adapters (rank={config.lora_r})")

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common targets
            bias="none"
        )

        # Add LoRA to model
        model = get_peft_model(model, peft_config)

        if self.verbose:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"[Transformers] Trainable params: {trainable_params:,} / {total_params:,} "
                  f"({100 * trainable_params / total_params:.2f}%)")

        # Load dataset
        if self.verbose:
            print(f"[Transformers] Loading dataset: {config.dataset_path}")

        dataset = load_dataset('json', data_files=config.dataset_path, split='train')

        # Tokenize dataset
        def tokenize_function(examples):
            # Extract text from messages format
            texts = []
            for messages in examples['messages']:
                # Convert messages to text
                text = ""
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    text += f"{role}: {content}\n"
                texts.append(text)

            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=config.max_seq_length,
                return_tensors="pt"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Training arguments
        training_args = TrainingArguments(
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
            optim="adamw_torch",
            report_to="none",  # Disable wandb, etc.
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        if self.verbose:
            print(f"[Transformers] Starting training for {config.max_steps} steps...")

        # Train
        train_result = trainer.train()

        # Save adapter
        if self.verbose:
            print(f"[Transformers] Saving adapter to {config.output_dir}")

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

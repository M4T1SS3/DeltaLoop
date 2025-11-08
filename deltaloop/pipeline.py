"""Complete pipeline orchestration for DeltaLoop.

This module provides a high-level Pipeline class that runs the complete
workflow: logs → distillation → training → evaluation.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import time
from datetime import datetime

from .schema import AgentTrace
from .distill import distill_dataset, DistillationStats
from .train import train, TrainingResult
from .eval import evaluate, EvalSummary, get_default_tasks


@dataclass
class PipelineConfig:
    """Configuration for complete pipeline run.

    Args:
        # Input/Output
        logs_path: Path to agent logs (JSONL with AgentTrace objects)
        output_dir: Directory for all outputs (default: data/pipeline_runs)
        run_name: Name for this run (default: auto-generated timestamp)

        # Training
        base_model: Base model to fine-tune (default: TinyLlama-1.1B)
        training_steps: Number of training steps (default: 100)
        training_method: Training method - 'sft' or 'dpo' (default: sft)
        lora_r: LoRA rank (default: 16)
        learning_rate: Learning rate (default: 2e-4)

        # Evaluation
        eval_device: Device for evaluation (default: auto)

        # Options
        verbose: Print progress (default: True)
        save_results: Save results to JSON (default: True)
    """

    # Input/Output
    logs_path: str
    output_dir: str = "data/pipeline_runs"
    run_name: Optional[str] = None

    # Training
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    training_steps: int = 100
    training_method: str = "sft"  # "sft" (supervised) or "dpo" (preference learning)
    lora_r: int = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 512  # Reduced for MPS compatibility
    per_device_batch_size: int = 1  # Reduced for MPS compatibility

    # Evaluation
    eval_device: str = "auto"

    # Options
    verbose: bool = True
    save_results: bool = True


@dataclass
class PipelineResult:
    """Results from complete pipeline run."""

    # Metadata
    run_name: str
    timestamp: str
    config: Dict[str, Any]

    # Stats
    distillation_stats: Dict[str, Any]
    training_result: Dict[str, Any]
    eval_summary: Dict[str, Any]

    # Timing
    total_time_seconds: float

    # Paths
    dataset_path: str
    adapter_path: str
    results_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Pipeline:
    """Orchestrates complete DeltaLoop workflow.

    The Pipeline class runs the complete workflow:
    1. Load/generate agent logs
    2. Distill into training data
    3. Train LoRA adapter
    4. Evaluate baseline vs adapted model
    5. Return comprehensive results

    Example:
        >>> from deltaloop.pipeline import Pipeline, PipelineConfig
        >>>
        >>> config = PipelineConfig(
        ...     logs_path="data/raw_logs/agent.jsonl",
        ...     base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ...     training_steps=100
        ... )
        >>>
        >>> pipeline = Pipeline()
        >>> result = pipeline.run(config)
        >>> print(f"Improvement: {result.eval_summary['improvement_percent']:.1f}%")
    """

    def __init__(self, verbose: bool = True):
        """Initialize pipeline.

        Args:
            verbose: Print progress information
        """
        self.verbose = verbose

    def run(self, config: PipelineConfig) -> PipelineResult:
        """Run complete pipeline.

        Args:
            config: Pipeline configuration

        Returns:
            PipelineResult with all outputs and metrics

        Raises:
            FileNotFoundError: If logs_path doesn't exist
            ValueError: If configuration is invalid
        """
        start_time = time.time()

        # Generate run name if not provided
        if config.run_name is None:
            config.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create output directory
        run_dir = Path(config.output_dir) / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print("=" * 70)
            print(f"DeltaLoop Pipeline: {config.run_name}")
            print("=" * 70)
            print(f"Base Model: {config.base_model}")
            print(f"Training Steps: {config.training_steps}")
            print(f"Output Directory: {run_dir}")
            print("=" * 70)
            print()

        # Step 1: Validate logs exist
        if not Path(config.logs_path).exists():
            raise FileNotFoundError(f"Logs not found: {config.logs_path}")

        # Step 2: Distill training data
        if self.verbose:
            method_name = "DPO preference pairs" if config.training_method == "dpo" else "training data"
            print(f"[Step 1/4] Distilling {method_name}...")

        dataset_path = str(run_dir / "train.jsonl")

        # Use DPO format if requested, otherwise use ChatML for SFT
        dataset_format = "dpo" if config.training_method == "dpo" else "chatml"

        distill_stats = distill_dataset(
            input_file=config.logs_path,
            output_file=dataset_path,
            filter_success=(config.training_method != "dpo"),  # DPO needs both success/fail
            format=dataset_format,
            verbose=config.verbose
        )

        if self.verbose:
            print(f"  ✓ Created {distill_stats.training_examples} training examples")
            print()

        # Step 3: Train adapter
        if self.verbose:
            method_desc = "DPO (Preference Learning)" if config.training_method == "dpo" else "SFT (Supervised)"
            print(f"[Step 2/4] Training LoRA adapter using {method_desc}...")

        adapter_path = str(run_dir / "adapter")

        # Select backend based on training method
        backend = "dpo" if config.training_method == "dpo" else "unsloth"

        training_result = train(
            dataset_path=dataset_path,
            base_model=config.base_model,
            output_dir=adapter_path,
            backend=backend,
            max_steps=config.training_steps,
            lora_r=config.lora_r,
            learning_rate=config.learning_rate,
            max_seq_length=config.max_seq_length,
            per_device_batch_size=config.per_device_batch_size,
            verbose=config.verbose
        )

        if self.verbose:
            print(f"  ✓ Adapter saved to: {adapter_path}")
            print(f"  ✓ Size: {training_result.adapter_size_mb:.2f} MB")
            print()

        # Step 4: Evaluate
        if self.verbose:
            print("[Step 3/4] Evaluating models...")

        eval_summary = evaluate(
            base_model=config.base_model,
            adapter_path=adapter_path,
            device=config.eval_device,
            verbose=config.verbose
        )

        if self.verbose:
            print(f"  ✓ Improvement: {eval_summary.improvement_percent:+.1f}%")
            print()

        # Step 5: Save results
        total_time = time.time() - start_time

        result = PipelineResult(
            run_name=config.run_name,
            timestamp=datetime.now().isoformat(),
            config=asdict(config),
            distillation_stats=asdict(distill_stats),
            training_result=asdict(training_result),
            eval_summary=eval_summary.to_dict(),
            total_time_seconds=total_time,
            dataset_path=dataset_path,
            adapter_path=adapter_path
        )

        if config.save_results:
            if self.verbose:
                print("[Step 4/4] Saving results...")

            results_path = str(run_dir / "results.json")
            result.save(results_path)
            result.results_path = results_path

            if self.verbose:
                print(f"  ✓ Results saved to: {results_path}")
                print()

        if self.verbose:
            print("=" * 70)
            print("Pipeline Complete")
            print("=" * 70)
            print(f"Total Time: {total_time:.1f}s")
            print(f"Improvement: {eval_summary.improvement_percent:+.1f}%")
            print(f"Output: {run_dir}")
            print("=" * 70)

        return result


def run_pipeline(
    logs_path: str,
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    training_steps: int = 100,
    output_dir: str = "data/pipeline_runs",
    verbose: bool = True,
    **kwargs
) -> PipelineResult:
    """Convenience function to run pipeline.

    Args:
        logs_path: Path to agent logs (JSONL)
        base_model: Base model to fine-tune
        training_steps: Number of training steps
        output_dir: Output directory
        verbose: Print progress
        **kwargs: Additional PipelineConfig parameters

    Returns:
        PipelineResult with all outputs and metrics

    Example:
        >>> from deltaloop.pipeline import run_pipeline
        >>> result = run_pipeline(
        ...     logs_path="data/raw_logs/agent.jsonl",
        ...     training_steps=100
        ... )
        >>> print(f"Improvement: {result.eval_summary['improvement_percent']:.1f}%")
    """
    config = PipelineConfig(
        logs_path=logs_path,
        base_model=base_model,
        training_steps=training_steps,
        output_dir=output_dir,
        verbose=verbose,
        **kwargs
    )

    pipeline = Pipeline(verbose=verbose)
    return pipeline.run(config)

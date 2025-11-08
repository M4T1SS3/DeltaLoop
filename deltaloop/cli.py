"""Command-line interface for DeltaLoop.

DeltaLoop CLI provides commands for the complete fine-tuning workflow:
    - distill: Convert agent logs to training datasets
    - train: Fine-tune models from datasets (coming soon)
    - eval: Evaluate adapted models (coming soon)
    - deploy: Deploy adapters to production (coming soon)
"""

import click
from pathlib import Path
from deltaloop import __version__
from deltaloop.distill import distill_dataset


@click.group()
@click.version_option(version=__version__)
def cli():
    """DeltaLoop: Stop optimizing prompts. Start optimizing models.

    Continuous fine-tuning layer for AI agents.
    """
    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    required=True,
    help="Input JSONL file with agent logs (AgentTrace format)"
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output JSONL file for training data"
)
@click.option(
    "--format",
    "-f",
    default="alpaca",
    type=click.Choice(["alpaca", "chatml", "raw"], case_sensitive=False),
    help="Training format (default: alpaca)"
)
@click.option(
    "--filter-success/--no-filter",
    default=True,
    help="Only include successful runs (default: enabled)"
)
@click.option(
    "--min-length",
    default=10,
    type=int,
    help="Minimum output length (default: 10)"
)
@click.option(
    "--max-examples",
    default=None,
    type=int,
    help="Maximum number of examples to include (optional)"
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress progress output"
)
def distill(input, output, format, filter_success, min_length, max_examples, quiet):
    """Process agent logs into training dataset.

    Converts raw agent execution logs (JSONL) into high-quality
    training datasets for fine-tuning.

    Example:

        deltaloop distill \\
          --input data/raw_logs/traces.jsonl \\
          --output data/datasets/train.jsonl \\
          --format alpaca
    """
    try:
        # Validate input file exists
        if not Path(input).exists():
            click.echo(f"❌ Error: Input file not found: {input}", err=True)
            raise click.Abort()

        # Run distillation
        stats = distill_dataset(
            input_file=input,
            output_file=output,
            filter_success=filter_success,
            format=format,
            min_output_length=min_length,
            max_examples=max_examples,
            verbose=not quiet
        )

        if not quiet:
            # Print success message
            click.echo(f"\n✓ Created {stats.training_examples} training examples")
            click.echo(f"  Output: {output}")

            # Show next steps
            click.echo("\nNext steps:")
            click.echo(f"  1. Inspect data: head -n 1 {output} | python3 -m json.tool")
            click.echo(f"  2. Train model: deltaloop train --dataset {output}")

    except Exception as e:
        click.echo(f"\n❌ Error during distillation: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--dataset", required=True, help="Training dataset (JSONL)")
@click.option("--model", default="unsloth/mistral-7b-bnb-4bit", help="Base model")
@click.option("--backend", default="unsloth", help="Training backend (unsloth)")
@click.option("--steps", default=500, type=int, help="Training steps (default: 500)")
@click.option("--output", default="data/models/run_1", help="Output directory")
@click.option("--lora-r", default=16, type=int, help="LoRA rank (default: 16)")
@click.option("--learning-rate", default=2e-4, type=float, help="Learning rate (default: 2e-4)")
@click.option("--batch-size", default=2, type=int, help="Batch size per device (default: 2)")
@click.option("--mock", is_flag=True, help="Use mock backend for testing (no GPU needed)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def train_cmd(dataset, model, backend, steps, output, lora_r, learning_rate, batch_size, mock, quiet):
    """Fine-tune model from dataset.

    Uses LoRA adapters for efficient fine-tuning with minimal GPU memory.

    Example:

        deltaloop train \\
          --dataset data/datasets/train.jsonl \\
          --model unsloth/mistral-7b-bnb-4bit \\
          --steps 500 \\
          --output data/models/v1
    """
    try:
        from deltaloop.train import train as train_model
        from deltaloop.train import TrainingConfig, Trainer

        # Validate dataset exists
        if not Path(dataset).exists():
            click.echo(f"❌ Error: Dataset not found: {dataset}", err=True)
            raise click.Abort()

        # Use mock backend if requested
        if mock:
            if not quiet:
                click.echo("🧪 Using mock backend (no actual training)")
            from deltaloop.backends.unsloth_backend import MockUnslothBackend
            config = TrainingConfig(
                dataset_path=dataset,
                base_model=model,
                output_dir=output,
                backend=backend,
                max_steps=steps,
                lora_r=lora_r,
                learning_rate=learning_rate,
                per_device_batch_size=batch_size,
            )
            trainer = Trainer(config, verbose=not quiet)
            trainer._backend = MockUnslothBackend(verbose=not quiet)
            result = trainer.train()
        else:
            # Real training
            result = train_model(
                dataset_path=dataset,
                base_model=model,
                output_dir=output,
                backend=backend,
                max_steps=steps,
                lora_r=lora_r,
                learning_rate=learning_rate,
                per_device_batch_size=batch_size,
                verbose=not quiet
            )

        if not quiet:
            # Print success message
            click.echo(f"\n✓ Training complete!")
            click.echo(f"  Adapter saved to: {result.output_dir}")
            click.echo(f"  Adapter size: {result.adapter_size_mb:.2f} MB")

            # Show next steps
            click.echo("\nNext steps:")
            click.echo(f"  1. Evaluate: deltaloop eval --adapter {result.output_dir}")
            click.echo(f"  2. Deploy: deltaloop deploy --adapter {result.output_dir}")

    except ImportError as e:
        click.echo(f"\n❌ Error: Missing dependencies", err=True)
        click.echo(f"   {e}", err=True)
        click.echo("\nInstall training dependencies:")
        click.echo("  pip install unsloth transformers datasets trl")
        raise click.Abort()
    except Exception as e:
        click.echo(f"\n❌ Error during training: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--adapter", required=True, help="Path to LoRA adapter")
@click.option("--model", default="unsloth/mistral-7b-bnb-4bit", help="Base model")
@click.option("--device", default="auto", help="Device (auto, cuda, cpu)")
@click.option("--max-length", default=256, type=int, help="Max generation length")
@click.option("--output", default=None, help="Save results to JSON file")
@click.option("--mock", is_flag=True, help="Use mock evaluation (no GPU needed)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def eval_cmd(adapter, model, device, max_length, output, mock, quiet):
    """Evaluate adapted model vs baseline.

    Runs evaluation tasks comparing your fine-tuned adapter
    against the base model to measure improvement.

    Example:

        deltaloop eval \\
          --adapter data/models/v1 \\
          --model unsloth/mistral-7b-bnb-4bit
    """
    try:
        from deltaloop.eval import evaluate, Evaluator, get_default_tasks, MockEvalTask

        # Validate adapter exists
        if not Path(adapter).exists():
            click.echo(f"❌ Error: Adapter not found: {adapter}", err=True)
            raise click.Abort()

        if mock:
            # Use mock evaluation (no GPU/transformers needed)
            if not quiet:
                click.echo("🧪 Using mock evaluation (no actual model loading)")

            # Create mock tasks
            mock_tasks = [
                MockEvalTask("tool_use", "What's the weather?", 0.6, 0.85),
                MockEvalTask("reasoning", "Why is sky blue?", 0.55, 0.82),
                MockEvalTask("instruction_following", "List 3 items", 0.7, 0.90),
                MockEvalTask("direct_answer", "What is 2+2?", 0.65, 0.88),
                MockEvalTask("domain_knowledge", "Capital of France?", 0.75, 0.92),
            ]

            # Create mock models
            class MockModel:
                def __init__(self, is_adapted=False):
                    self._is_adapted = is_adapted
                    self.device = "cpu"

            class MockTokenizer:
                eos_token_id = 0
                def __call__(self, text, **kwargs):
                    return {"input_ids": [[1, 2, 3]]}
                def decode(self, tokens, **kwargs):
                    return "Mock output"

            # Run evaluation with mock
            evaluator = Evaluator(mock_tasks, verbose=not quiet)

            # Simulate evaluation manually
            from datetime import datetime
            results = []
            for task in mock_tasks:
                baseline_score, baseline_out = task.evaluate(MockModel(False), MockTokenizer())
                adapted_score, adapted_out = task.evaluate(MockModel(True), MockTokenizer())

                improvement = ((adapted_score - baseline_score) / baseline_score) * 100
                from deltaloop.eval import EvalResult
                results.append(EvalResult(
                    task_name=task.name,
                    baseline_score=baseline_score,
                    adapted_score=adapted_score,
                    improvement_percent=improvement,
                    baseline_output=baseline_out,
                    adapted_output=adapted_out,
                    prompt=task.prompt
                ))

            # Calculate summary
            baseline_scores = [r.baseline_score for r in results]
            adapted_scores = [r.adapted_score for r in results]
            baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
            adapted_avg = sum(adapted_scores) / len(adapted_scores) if adapted_scores else 0.0
            improvement = ((adapted_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0.0

            from deltaloop.eval import EvalSummary
            summary = EvalSummary(
                total_tasks=len(results),
                baseline_avg=baseline_avg,
                adapted_avg=adapted_avg,
                improvement_percent=improvement,
                tasks_improved=sum(1 for r in results if r.adapted_score > r.baseline_score),
                tasks_regressed=sum(1 for r in results if r.adapted_score < r.baseline_score),
                tasks_unchanged=sum(1 for r in results if r.adapted_score == r.baseline_score),
                results=results,
                timestamp=datetime.now().isoformat(),
                config={"base_model": model, "adapter_path": adapter, "mock": True}
            )

        else:
            # Real evaluation
            if not quiet:
                click.echo(f"Running evaluation...")
                click.echo(f"  Base model: {model}")
                click.echo(f"  Adapter: {adapter}")
                click.echo(f"  Device: {device}")

            summary = evaluate(
                base_model=model,
                adapter_path=adapter,
                device=device,
                verbose=not quiet
            )

        # Save results if requested
        if output:
            summary.save(output)
            if not quiet:
                click.echo(f"\n✓ Results saved to: {output}")

        # Print summary
        if not quiet:
            click.echo(f"\n✓ Evaluation complete!")
            click.echo(f"  Overall improvement: {summary.improvement_percent:+.1f}%")

            if summary.improvement_percent > 20:
                click.echo("\n🎉 Great! Model improved by >20%")
            elif summary.improvement_percent > 0:
                click.echo("\n✓ Model shows improvement")
            else:
                click.echo("\n⚠ Model did not improve. Consider:")
                click.echo("  - Training longer (more steps)")
                click.echo("  - Using more/better training data")
                click.echo("  - Adjusting LoRA rank or learning rate")

    except ImportError as e:
        click.echo(f"\n❌ Error: Missing dependencies", err=True)
        click.echo(f"   {e}", err=True)
        click.echo("\nInstall evaluation dependencies:")
        click.echo("  pip install transformers torch peft numpy")
        raise click.Abort()
    except Exception as e:
        click.echo(f"\n❌ Error during evaluation: {e}", err=True)
        import traceback
        if not quiet:
            traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--adapter", required=True, help="Adapter to deploy")
@click.option("--target", default="production", help="Deployment target")
def deploy(adapter, target):
    """Deploy adapter to production.

    [Coming in Phase 1, Task 6]

    This command will deploy your fine-tuned adapter
    to the specified target environment.
    """
    click.echo("❌ Deployment not yet implemented")
    click.echo("   Coming in Phase 1, Task 6: CLI with 4 commands")
    click.echo("\nStay tuned!")
    raise click.Abort()


@cli.command()
def info():
    """Show DeltaLoop installation and environment info."""
    import sys
    import platform

    click.echo("DeltaLoop Information")
    click.echo("=" * 60)
    click.echo(f"Version:        {__version__}")
    click.echo(f"Python:         {sys.version.split()[0]}")
    click.echo(f"Platform:       {platform.system()} {platform.release()}")
    click.echo(f"Install path:   {Path(__file__).parent}")
    click.echo("=" * 60)

    # Check for optional dependencies
    click.echo("\nOptional Dependencies:")

    dependencies = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "langchain": "LangChain",
        "unsloth": "Unsloth",
        "axolotl": "Axolotl",
    }

    for module, name in dependencies.items():
        try:
            __import__(module)
            click.echo(f"  ✓ {name}")
        except ImportError:
            click.echo(f"  ✗ {name} (not installed)")

    click.echo("\nDocumentation: https://github.com/deltaloop/deltaloop")


if __name__ == "__main__":
    cli()

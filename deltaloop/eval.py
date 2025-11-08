"""Evaluation framework for comparing base vs fine-tuned models.

This module provides tools to:
    - Define evaluation tasks
    - Compare baseline vs adapted models
    - Measure performance improvements
    - Generate detailed reports
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path

# Optional dependency - only needed for evaluation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class EvalResult:
    """Result from a single evaluation task."""
    task_name: str
    baseline_score: float
    adapted_score: float
    improvement_percent: float
    baseline_output: str
    adapted_output: str
    prompt: str
    expected: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvalSummary:
    """Summary of evaluation results across all tasks."""
    total_tasks: int
    baseline_avg: float
    adapted_avg: float
    improvement_percent: float
    tasks_improved: int
    tasks_regressed: int
    tasks_unchanged: int
    results: List[EvalResult]
    timestamp: str
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "baseline_avg": self.baseline_avg,
            "adapted_avg": self.adapted_avg,
            "improvement_percent": self.improvement_percent,
            "tasks_improved": self.tasks_improved,
            "tasks_regressed": self.tasks_regressed,
            "tasks_unchanged": self.tasks_unchanged,
            "timestamp": self.timestamp,
            "config": self.config,
            "results": [r.to_dict() for r in self.results]
        }

    def save(self, path: str) -> None:
        """Save summary to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class EvalTask:
    """Base class for evaluation tasks.

    An evaluation task consists of:
        - A prompt to send to the model
        - A judge function that scores the output (0.0 to 1.0)
        - Optional expected output for comparison
    """

    def __init__(
        self,
        name: str,
        prompt: str,
        judge_fn: Callable[[str], float],
        description: str = "",
        expected: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize evaluation task.

        Args:
            name: Task name (e.g., "tool_use_accuracy")
            prompt: Prompt to send to model
            judge_fn: Function that scores output (0.0-1.0)
            description: Human-readable description
            expected: Optional expected output
            metadata: Optional task metadata
        """
        self.name = name
        self.prompt = prompt
        self.judge_fn = judge_fn
        self.description = description
        self.expected = expected
        self.metadata = metadata or {}

    def evaluate(self, model, tokenizer, max_length: int = 256) -> Tuple[float, str]:
        """Run task on a model and return score and output.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            max_length: Maximum generation length

        Returns:
            Tuple of (score, output_text)
        """
        try:
            # Generate output
            inputs = tokenizer(self.prompt, return_tensors="pt")

            # Move to model's device if needed
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt from output if present
            if output_text.startswith(self.prompt):
                output_text = output_text[len(self.prompt):].strip()

            # Score with judge function
            score = self.judge_fn(output_text)

            return float(score), output_text

        except Exception as e:
            print(f"  ⚠ Error evaluating task '{self.name}': {e}")
            return 0.0, f"Error: {str(e)}"

    def __repr__(self) -> str:
        return f"EvalTask(name='{self.name}', prompt='{self.prompt[:30]}...')"


class MockEvalTask(EvalTask):
    """Mock evaluation task for testing without real models.

    This generates fake scores and outputs for testing the eval pipeline.
    """

    def __init__(
        self,
        name: str,
        prompt: str,
        baseline_score: float = 0.6,
        adapted_score: float = 0.85,
        **kwargs
    ):
        """Initialize mock task with predefined scores."""
        self.baseline_score = baseline_score
        self.adapted_score = adapted_score

        # Mock judge function
        def mock_judge(output):
            return baseline_score

        super().__init__(name, prompt, mock_judge, **kwargs)

    def evaluate(self, model, tokenizer, max_length: int = 256) -> Tuple[float, str]:
        """Return mock results without calling model."""
        # Determine if this is baseline or adapted based on model
        is_adapted = hasattr(model, '_is_adapted') and model._is_adapted
        score = self.adapted_score if is_adapted else self.baseline_score
        output = f"Mock output for {self.name} (adapted={is_adapted})"
        return score, output


class Evaluator:
    """Evaluates and compares baseline vs adapted models.

    Args:
        tasks: List of evaluation tasks to run
        verbose: Print progress information (default: True)
    """

    def __init__(self, tasks: List[EvalTask], verbose: bool = True):
        self.tasks = tasks
        self.verbose = verbose

    def evaluate(
        self,
        base_model: str,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        max_length: int = 256
    ) -> EvalSummary:
        """Run evaluation comparing baseline vs adapted model.

        Args:
            base_model: Name or path to base model
            adapter_path: Path to LoRA adapter (optional)
            device: Device to run on ("auto", "cuda", "cpu")
            max_length: Maximum generation length

        Returns:
            EvalSummary with detailed results
        """
        from datetime import datetime
        from deltaloop.utils import load_model

        if self.verbose:
            print("=" * 70)
            print("DeltaLoop Evaluation")
            print("=" * 70)
            print(f"Base Model:  {base_model}")
            print(f"Adapter:     {adapter_path or 'None (baseline only)'}")
            print(f"Tasks:       {len(self.tasks)}")
            print("=" * 70)
            print()

        # Load baseline model
        if self.verbose:
            print("[1/3] Loading baseline model...")

        baseline_model, baseline_tokenizer = load_model(base_model, device=device)

        # Load adapted model if adapter provided
        if adapter_path:
            if self.verbose:
                print("[2/3] Loading adapted model (with LoRA)...")
            adapted_model, adapted_tokenizer = load_model(
                base_model,
                adapter=adapter_path,
                device=device
            )
            # Mark as adapted for mock tasks
            adapted_model._is_adapted = True
        else:
            if self.verbose:
                print("[2/3] No adapter provided, comparing baseline to itself...")
            adapted_model, adapted_tokenizer = baseline_model, baseline_tokenizer

        # Run evaluation tasks
        if self.verbose:
            print(f"[3/3] Running {len(self.tasks)} evaluation tasks...")
            print()

        results = []

        for i, task in enumerate(self.tasks, 1):
            if self.verbose:
                print(f"Task {i}/{len(self.tasks)}: {task.name}")
                print(f"  Prompt: {task.prompt[:60]}...")

            # Evaluate baseline
            baseline_score, baseline_output = task.evaluate(
                baseline_model,
                baseline_tokenizer,
                max_length
            )

            # Evaluate adapted
            adapted_score, adapted_output = task.evaluate(
                adapted_model,
                adapted_tokenizer,
                max_length
            )

            # Calculate improvement
            if baseline_score > 0:
                improvement = ((adapted_score - baseline_score) / baseline_score) * 100
            else:
                improvement = 0.0 if adapted_score == 0 else 100.0

            if self.verbose:
                print(f"  Baseline:    {baseline_score:.2%}")
                print(f"  Adapted:     {adapted_score:.2%}")
                print(f"  Improvement: {improvement:+.1f}%")
                print()

            results.append(EvalResult(
                task_name=task.name,
                baseline_score=baseline_score,
                adapted_score=adapted_score,
                improvement_percent=improvement,
                baseline_output=baseline_output[:200],  # Truncate for storage
                adapted_output=adapted_output[:200],
                prompt=task.prompt,
                expected=task.expected,
                metadata=task.metadata
            ))

        # Calculate summary statistics
        baseline_scores = [r.baseline_score for r in results]
        adapted_scores = [r.adapted_score for r in results]

        if HAS_NUMPY:
            baseline_avg = np.mean(baseline_scores)
            adapted_avg = np.mean(adapted_scores)
        else:
            # Fallback to pure Python
            baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
            adapted_avg = sum(adapted_scores) / len(adapted_scores) if adapted_scores else 0.0

        if baseline_avg > 0:
            overall_improvement = ((adapted_avg - baseline_avg) / baseline_avg) * 100
        else:
            overall_improvement = 0.0

        # Count improvements/regressions
        tasks_improved = sum(1 for r in results if r.adapted_score > r.baseline_score)
        tasks_regressed = sum(1 for r in results if r.adapted_score < r.baseline_score)
        tasks_unchanged = sum(1 for r in results if r.adapted_score == r.baseline_score)

        summary = EvalSummary(
            total_tasks=len(results),
            baseline_avg=baseline_avg,
            adapted_avg=adapted_avg,
            improvement_percent=overall_improvement,
            tasks_improved=tasks_improved,
            tasks_regressed=tasks_regressed,
            tasks_unchanged=tasks_unchanged,
            results=results,
            timestamp=datetime.now().isoformat(),
            config={
                "base_model": base_model,
                "adapter_path": adapter_path,
                "device": device,
                "max_length": max_length
            }
        )

        if self.verbose:
            self._print_summary(summary)

        return summary

    def _print_summary(self, summary: EvalSummary) -> None:
        """Print evaluation summary."""
        print("=" * 70)
        print("Evaluation Summary")
        print("=" * 70)
        print(f"Total Tasks:        {summary.total_tasks}")
        print(f"Baseline Average:   {summary.baseline_avg:.2%}")
        print(f"Adapted Average:    {summary.adapted_avg:.2%}")
        print(f"Overall Improvement: {summary.improvement_percent:+.1f}%")
        print()
        print(f"Tasks Improved:     {summary.tasks_improved}")
        print(f"Tasks Regressed:    {summary.tasks_regressed}")
        print(f"Tasks Unchanged:    {summary.tasks_unchanged}")
        print("=" * 70)

        if summary.improvement_percent > 0:
            print(f"\n✓ Model improved by {summary.improvement_percent:.1f}%!")
        elif summary.improvement_percent < 0:
            print(f"\n⚠ Model regressed by {abs(summary.improvement_percent):.1f}%")
        else:
            print("\n→ No significant change")


def evaluate(
    base_model: str,
    adapter_path: Optional[str] = None,
    tasks: Optional[List[EvalTask]] = None,
    device: str = "auto",
    verbose: bool = True
) -> EvalSummary:
    """Convenience function to run evaluation.

    Args:
        base_model: Name or path to base model
        adapter_path: Path to LoRA adapter (optional)
        tasks: List of evaluation tasks (uses default if None)
        device: Device to run on
        verbose: Print progress

    Returns:
        EvalSummary with results

    Example:
        >>> from deltaloop.eval import evaluate
        >>> summary = evaluate(
        ...     base_model="unsloth/mistral-7b-bnb-4bit",
        ...     adapter_path="data/models/v1"
        ... )
        >>> print(f"Improvement: {summary.improvement_percent:.1f}%")
    """
    if tasks is None:
        tasks = get_default_tasks()

    evaluator = Evaluator(tasks, verbose=verbose)
    return evaluator.evaluate(base_model, adapter_path, device)


def get_default_tasks() -> List[EvalTask]:
    """Get default evaluation tasks for agent models.

    These tasks test common agent capabilities:
        - Tool usage understanding
        - Reasoning quality
        - Instruction following
        - Domain knowledge
        - Output formatting
    """
    tasks = []

    # Task 1: Tool Use Recognition
    def judge_tool_use(output: str) -> float:
        """Check if output mentions using a tool."""
        tool_keywords = ["tool", "function", "call", "use", "weather_tool", "search"]
        output_lower = output.lower()
        return 1.0 if any(kw in output_lower for kw in tool_keywords) else 0.0

    tasks.append(EvalTask(
        name="tool_use_recognition",
        prompt="What's the weather in Paris?",
        judge_fn=judge_tool_use,
        description="Tests if model recognizes when to use tools",
        expected="Should mention using a weather tool"
    ))

    # Task 2: Direct Answer Recognition
    def judge_direct_answer(output: str) -> float:
        """Check if model answers directly without tools."""
        output_lower = output.lower()
        # Should contain answer, not tool usage
        has_answer = any(kw in output_lower for kw in ["4", "four", "equals"])
        has_tool = any(kw in output_lower for kw in ["tool", "function", "call"])
        return 1.0 if (has_answer and not has_tool) else 0.0

    tasks.append(EvalTask(
        name="direct_answer",
        prompt="What is 2+2?",
        judge_fn=judge_direct_answer,
        description="Tests if model can answer directly without tools",
        expected="Should answer '4' without mentioning tools"
    ))

    # Task 3: Reasoning Quality
    def judge_reasoning(output: str) -> float:
        """Check for reasoning keywords."""
        reasoning_keywords = ["because", "since", "therefore", "thus", "reason"]
        output_lower = output.lower()
        count = sum(1 for kw in reasoning_keywords if kw in output_lower)
        return min(count / 2, 1.0)  # Score based on reasoning indicators

    tasks.append(EvalTask(
        name="reasoning_quality",
        prompt="Why is the sky blue? Explain your reasoning.",
        judge_fn=judge_reasoning,
        description="Tests quality of reasoning and explanation",
        expected="Should include reasoning keywords and explanation"
    ))

    # Task 4: Instruction Following
    def judge_format(output: str) -> float:
        """Check if output follows format instructions."""
        has_numbered = any(line.strip().startswith(('1.', '2.', '3.'))
                          for line in output.split('\n'))
        return 1.0 if has_numbered else 0.0

    tasks.append(EvalTask(
        name="instruction_following",
        prompt="List 3 benefits of exercise. Format your answer as a numbered list.",
        judge_fn=judge_format,
        description="Tests ability to follow format instructions",
        expected="Should provide numbered list with 3 items"
    ))

    # Task 5: Domain Knowledge (General)
    def judge_capitals(output: str) -> float:
        """Check if answer contains 'Paris'."""
        return 1.0 if "paris" in output.lower() else 0.0

    tasks.append(EvalTask(
        name="domain_knowledge",
        prompt="What is the capital of France?",
        judge_fn=judge_capitals,
        description="Tests basic domain knowledge",
        expected="Should answer 'Paris'"
    ))

    return tasks

"""Data distillation pipeline for converting agent logs to training datasets.

This module processes raw agent traces (JSONL) and converts them into
high-quality training datasets for fine-tuning.

Features:
    - Success/failure filtering
    - Deduplication
    - Quality filtering (length, coherence)
    - Multiple output formats (Alpaca, ChatML, etc.)
    - Dataset statistics and reporting
"""

import json
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from deltaloop.schema import AgentTrace


@dataclass
class DistillationStats:
    """Statistics from the distillation process."""
    total_traces: int
    successful_traces: int
    failed_traces: int
    duplicates_removed: int
    quality_filtered: int
    training_examples: int
    format: str


class DataDistiller:
    """Processes agent logs into training datasets.

    Args:
        min_output_length: Minimum output length to include (default: 10)
        min_quality_score: Minimum quality score (0-1, default: 0.7)
        verbose: Print progress information (default: True)
    """

    def __init__(
        self,
        min_output_length: int = 10,
        min_quality_score: float = 0.7,
        verbose: bool = True
    ):
        self.min_output_length = min_output_length
        self.min_quality_score = min_quality_score
        self.verbose = verbose

    def distill_dataset(
        self,
        input_file: str,
        output_file: str,
        filter_success: bool = True,
        format: str = "alpaca",
        max_examples: Optional[int] = None
    ) -> DistillationStats:
        """Process agent logs into fine-tuning format.

        Steps:
        1. Load normalized traces
        2. Filter by success (if enabled)
        3. Remove duplicates
        4. Apply quality filters
        5. Format for training
        6. Save to output file

        Args:
            input_file: Path to input JSONL file with AgentTrace objects
            output_file: Path to output JSONL file for training data
            filter_success: Only include successful runs (default: True)
            format: Training format - 'alpaca', 'chatml', or 'raw' (default: 'alpaca')
            max_examples: Maximum number of examples to include (optional)

        Returns:
            DistillationStats object with processing statistics
        """
        if self.verbose:
            print(f"[DeltaLoop Distill] Loading traces from {input_file}")

        # Load and process traces
        traces = self._load_traces(input_file)
        stats = self._create_initial_stats(traces)

        if self.verbose:
            print(f"  Loaded {stats.total_traces} traces")
            print(f"  Success: {stats.successful_traces} | Failed: {stats.failed_traces}")

        # For DPO format, we need BOTH success and failure examples
        # For other formats, filter by success if requested
        if format == "dpo":
            # DPO needs both successes and failures to create preference pairs
            if self.verbose:
                print(f"  DPO mode: keeping both successes and failures for pairing")
        elif filter_success:
            traces = [t for t in traces if t.success]
            if self.verbose:
                print(f"  Filtered to {len(traces)} successful traces")

        # Remove duplicates (skip for DPO to preserve pairing options)
        if format != "dpo":
            traces, duplicates = self._remove_duplicates(traces)
            stats.duplicates_removed = duplicates
            if self.verbose and duplicates > 0:
                print(f"  Removed {duplicates} duplicate prompts")
        else:
            duplicates = 0
            stats.duplicates_removed = 0

        # Quality filtering
        traces, filtered = self._apply_quality_filters(traces)
        stats.quality_filtered = filtered
        if self.verbose and filtered > 0:
            print(f"  Filtered out {filtered} low-quality traces")

        # Limit examples if specified
        if max_examples and len(traces) > max_examples:
            traces = traces[:max_examples]
            if self.verbose:
                print(f"  Limited to {max_examples} examples")

        # Convert to training format
        if self.verbose:
            print(f"  Converting to {format} format...")

        training_data = self._format_for_training(traces, format)
        stats.training_examples = len(training_data)
        stats.format = format

        # Save to file
        self._save_dataset(training_data, output_file)

        if self.verbose:
            print(f"\n✓ Saved {len(training_data)} training examples to {output_file}")
            self._print_summary(stats)

        return stats

    def _load_traces(self, input_file: str) -> List[AgentTrace]:
        """Load AgentTrace objects from JSONL file."""
        traces = []
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = AgentTrace.from_json(line)
                    traces.append(trace)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Skipped line {line_num}: {e}")
        return traces

    def _create_initial_stats(self, traces: List[AgentTrace]) -> DistillationStats:
        """Create initial statistics from loaded traces."""
        successful = sum(1 for t in traces if t.success is True)
        failed = sum(1 for t in traces if t.success is False)
        return DistillationStats(
            total_traces=len(traces),
            successful_traces=successful,
            failed_traces=failed,
            duplicates_removed=0,
            quality_filtered=0,
            training_examples=0,
            format=""
        )

    def _remove_duplicates(self, traces: List[AgentTrace]) -> tuple[List[AgentTrace], int]:
        """Remove duplicate prompts, keeping first occurrence."""
        seen: Set[str] = set()
        unique_traces = []
        duplicates = 0

        for trace in traces:
            if trace.prompt not in seen:
                seen.add(trace.prompt)
                unique_traces.append(trace)
            else:
                duplicates += 1

        return unique_traces, duplicates

    def _apply_quality_filters(self, traces: List[AgentTrace]) -> tuple[List[AgentTrace], int]:
        """Apply quality filters to traces."""
        filtered_traces = []
        filtered_count = 0

        for trace in traces:
            # Filter 1: Minimum output length
            if len(trace.output) < self.min_output_length:
                filtered_count += 1
                continue

            # Filter 2: Must have actual content
            if not trace.output.strip():
                filtered_count += 1
                continue

            # Filter 3: No error messages in successful traces
            if trace.success and trace.error:
                filtered_count += 1
                continue

            # Filter 4: Prompt must have content
            if not trace.prompt.strip():
                filtered_count += 1
                continue

            filtered_traces.append(trace)

        return filtered_traces, filtered_count

    def _format_for_training(self, traces: List[AgentTrace], format: str) -> List[Dict[str, Any]]:
        """Convert traces to training format."""
        if format == "alpaca":
            return self._format_alpaca(traces)
        elif format == "chatml":
            return self._format_chatml(traces)
        elif format == "raw":
            return self._format_raw(traces)
        elif format == "dpo":
            return self._format_dpo(traces)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'alpaca', 'chatml', 'raw', or 'dpo'")

    def _format_alpaca(self, traces: List[AgentTrace]) -> List[Dict[str, Any]]:
        """Format traces in Alpaca format (instruction-input-output)."""
        return [
            {
                "instruction": trace.prompt,
                "input": "",
                "output": trace.output
            }
            for trace in traces
        ]

    def _format_chatml(self, traces: List[AgentTrace]) -> List[Dict[str, Any]]:
        """Format traces in ChatML format (messages array)."""
        return [
            {
                "messages": [
                    {"role": "user", "content": trace.prompt},
                    {"role": "assistant", "content": trace.output}
                ]
            }
            for trace in traces
        ]

    def _format_raw(self, traces: List[AgentTrace]) -> List[Dict[str, Any]]:
        """Format traces as raw prompt-completion pairs."""
        return [
            {
                "prompt": trace.prompt,
                "completion": trace.output,
                "metadata": trace.metadata
            }
            for trace in traces
        ]

    def _format_dpo(self, all_traces: List[AgentTrace]) -> List[Dict[str, Any]]:
        """Format traces as DPO preference pairs (chosen vs rejected).

        Matches successful traces with failed traces that have similar prompts
        to create preference pairs for Direct Preference Optimization.

        Args:
            all_traces: All traces (both success and failure)

        Returns:
            List of preference pairs with format:
            {
                "prompt": "user query",
                "chosen": "successful response",
                "rejected": "failed response"
            }
        """
        from difflib import SequenceMatcher

        def similarity(a: str, b: str) -> float:
            """Calculate similarity between two prompts."""
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        # Separate success and failures
        successes = [t for t in all_traces if t.success]
        failures = [t for t in all_traces if not t.success]

        if not successes or not failures:
            if self.verbose:
                print(f"  Warning: Need both successes ({len(successes)}) and failures ({len(failures)}) for DPO pairs")
            return []

        preference_pairs = []
        used_failures = set()

        # For each success, find best matching failure
        for success in successes:
            best_match = None
            best_score = 0.3  # Minimum similarity threshold

            for i, failure in enumerate(failures):
                if i in used_failures:
                    continue

                score = similarity(success.prompt, failure.prompt)
                if score > best_score:
                    best_score = score
                    best_match = (i, failure)

            if best_match:
                idx, failure = best_match
                used_failures.add(idx)

                preference_pairs.append({
                    "prompt": success.prompt,  # Use success prompt (usually cleaner)
                    "chosen": success.output,
                    "rejected": failure.output
                })

        if self.verbose:
            print(f"  Created {len(preference_pairs)} DPO preference pairs")
            print(f"  Success examples: {len(successes)}, Failed examples: {len(failures)}")
            print(f"  Matched {len(preference_pairs)}/{min(len(successes), len(failures))} pairs")

        return preference_pairs

    def _save_dataset(self, training_data: List[Dict[str, Any]], output_file: str) -> None:
        """Save training dataset to JSONL file."""
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

    def _print_summary(self, stats: DistillationStats) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("Distillation Summary")
        print("=" * 60)
        print(f"Total traces loaded:     {stats.total_traces}")
        print(f"Successful traces:       {stats.successful_traces}")
        print(f"Failed traces:           {stats.failed_traces}")
        print(f"Duplicates removed:      {stats.duplicates_removed}")
        print(f"Quality filtered:        {stats.quality_filtered}")
        print(f"Training examples:       {stats.training_examples}")
        print(f"Output format:           {stats.format}")
        print("=" * 60)


def distill_dataset(
    input_file: str,
    output_file: str,
    filter_success: bool = True,
    format: str = "alpaca",
    min_output_length: int = 10,
    max_examples: Optional[int] = None,
    verbose: bool = True
) -> DistillationStats:
    """Convenience function to distill a dataset.

    Args:
        input_file: Path to input JSONL file with AgentTrace objects
        output_file: Path to output JSONL file for training data
        filter_success: Only include successful runs (default: True)
        format: Training format - 'alpaca', 'chatml', or 'raw' (default: 'alpaca')
        min_output_length: Minimum output length (default: 10)
        max_examples: Maximum number of examples (optional)
        verbose: Print progress (default: True)

    Returns:
        DistillationStats object with processing statistics

    Example:
        >>> from deltaloop.distill import distill_dataset
        >>> stats = distill_dataset(
        ...     input_file="data/raw_logs/traces.jsonl",
        ...     output_file="data/datasets/train.jsonl"
        ... )
    """
    distiller = DataDistiller(
        min_output_length=min_output_length,
        verbose=verbose
    )

    return distiller.distill_dataset(
        input_file=input_file,
        output_file=output_file,
        filter_success=filter_success,
        format=format,
        max_examples=max_examples
    )

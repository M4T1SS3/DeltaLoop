"""Generic adapter for custom agents.

This module provides a simple logging interface for any custom agent
framework that doesn't have a dedicated adapter.

Example:
    from deltaloop.adapters.generic import GenericLogger

    logger = GenericLogger()

    def my_custom_agent(task):
        thoughts = []

        # Step 1
        thoughts.append("Analyzing task...")
        result = analyze(task)

        # Step 2
        thoughts.append(f"Calling tool with {result}")
        final = call_tool(result)

        # Log everything
        logger.log(
            prompt=task,
            output=final,
            reasoning=thoughts,
            success=final is not None,
            tool_calls=[{"tool": "my_tool", "input": result}]
        )

        return final
"""

from deltaloop.schema import AgentTrace
from datetime import datetime
from typing import Any, Dict, List, Optional
import os


class GenericLogger:
    """Manual logging for custom agents.

    This adapter provides maximum flexibility for logging any type of agent,
    regardless of framework. Simply call the log() method with your data.

    Args:
        output_file: Path to output JSONL file (default: data/raw_logs/generic.jsonl)
        verbose: Print logging information (default: False)
    """

    def __init__(
        self,
        output_file: str = "data/raw_logs/generic.jsonl",
        verbose: bool = False
    ):
        self.output_file = output_file
        self.verbose = verbose

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if self.verbose:
            print(f"[DeltaLoop] Logging to {output_file}")

    def log(
        self,
        prompt: str,
        output: str,
        reasoning: Optional[List[str]] = None,
        tool_calls: Optional[List[Dict]] = None,
        success: Optional[bool] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None,
        **kwargs: Any
    ) -> None:
        """Log an agent execution.

        This is a flexible logging method that accepts any AgentTrace fields.
        Only prompt and output are required; all other fields are optional.

        Args:
            prompt: User's original request
            output: Agent's final response
            reasoning: Chain-of-thought steps (optional)
            tool_calls: List of tools invoked (optional)
            success: Whether the run succeeded (optional)
            error: Error message if failed (optional)
            metadata: Additional framework-specific data (optional)
            **kwargs: Any additional AgentTrace fields

        Example:
            logger.log(
                prompt="What's 2+2?",
                output="4",
                reasoning=["This is basic math", "2+2=4"],
                success=True
            )
        """
        # Merge metadata
        if metadata is None:
            metadata = {}
        metadata["framework"] = "generic"

        # Create trace
        trace = AgentTrace(
            prompt=prompt,
            output=output,
            timestamp=datetime.now().isoformat(),
            reasoning=reasoning,
            tool_calls=tool_calls,
            success=success,
            error=error,
            metadata=metadata,
            **kwargs
        )

        # Append to JSONL
        with open(self.output_file, "a") as f:
            f.write(trace.to_json() + "\n")

        if self.verbose:
            status = "✓" if success else "✗" if success is False else "?"
            print(f"[DeltaLoop] {status} Logged: {prompt[:50]}...")

    def log_batch(self, traces: List[Dict[str, Any]]) -> None:
        """Log multiple agent executions at once.

        This is more efficient than calling log() multiple times,
        as it opens the file only once.

        Args:
            traces: List of dictionaries containing AgentTrace fields

        Example:
            logger.log_batch([
                {"prompt": "Question 1", "output": "Answer 1", "success": True},
                {"prompt": "Question 2", "output": "Answer 2", "success": True},
            ])
        """
        with open(self.output_file, "a") as f:
            for trace_data in traces:
                # Ensure metadata exists
                if "metadata" not in trace_data:
                    trace_data["metadata"] = {}
                trace_data["metadata"]["framework"] = "generic"

                # Ensure timestamp exists
                if "timestamp" not in trace_data:
                    trace_data["timestamp"] = datetime.now().isoformat()

                # Create trace
                trace = AgentTrace(**trace_data)
                f.write(trace.to_json() + "\n")

        if self.verbose:
            print(f"[DeltaLoop] ✓ Logged {len(traces)} traces")

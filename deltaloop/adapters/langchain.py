"""LangChain adapter for DeltaLoop.

This module provides a callback handler that automatically captures
LangChain agent execution logs and converts them to AgentTrace format.

Example:
    from deltaloop.adapters.langchain import DeltaLoopCallback
    from langchain.agents import create_react_agent

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        callbacks=[DeltaLoopCallback()]  # One line!
    )

    # All runs are now automatically logged
    agent.run("What's the weather in Paris?")
"""

from langchain.callbacks.base import BaseCallbackHandler
from deltaloop.schema import AgentTrace
from datetime import datetime
from typing import Any, Dict, List, Optional
import os


class DeltaLoopCallback(BaseCallbackHandler):
    """Automatically captures LangChain agent execution.

    This callback handler integrates seamlessly with LangChain agents,
    capturing all prompts, reasoning steps, tool calls, and outputs.

    Args:
        output_file: Path to output JSONL file (default: data/raw_logs/langchain.jsonl)
        verbose: Print logging information (default: False)
    """

    def __init__(
        self,
        output_file: str = "data/raw_logs/langchain.jsonl",
        verbose: bool = False
    ):
        super().__init__()
        self.output_file = output_file
        self.verbose = verbose
        self.current_trace: Dict[str, Any] = {}

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if self.verbose:
            print(f"[DeltaLoop] Logging to {output_file}")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when a chain starts running."""
        self.current_trace = {
            "prompt": inputs.get("input", inputs.get("question", "")),
            "reasoning": [],
            "tool_calls": [],
            "start_time": datetime.now()
        }

        if self.verbose:
            print(f"[DeltaLoop] Chain started: {self.current_trace['prompt'][:50]}...")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Called when LLM starts running."""
        # Capture reasoning steps
        if prompts and len(prompts) > 0:
            self.current_trace.setdefault("reasoning", []).append(prompts[0])

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """Called when a tool starts running."""
        # Capture tool usage
        tool_call = {
            "tool": serialized.get("name", "unknown"),
            "input": input_str,
            "timestamp": datetime.now().isoformat()
        }
        self.current_trace.setdefault("tool_calls", []).append(tool_call)

        if self.verbose:
            print(f"[DeltaLoop] Tool called: {tool_call['tool']}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when a chain ends."""
        # Calculate latency
        start_time = self.current_trace.get("start_time", datetime.now())
        latency = (datetime.now() - start_time).total_seconds() * 1000

        # Create normalized trace
        trace = AgentTrace(
            prompt=self.current_trace.get("prompt", ""),
            output=outputs.get("output", outputs.get("answer", "")),
            reasoning=self.current_trace.get("reasoning", []),
            tool_calls=self.current_trace.get("tool_calls", []),
            timestamp=datetime.now().isoformat(),
            success=True,  # Successful completion
            latency_ms=latency,
            metadata={"framework": "langchain"}
        )

        # Append to JSONL
        with open(self.output_file, "a") as f:
            f.write(trace.to_json() + "\n")

        if self.verbose:
            print(f"[DeltaLoop] ✓ Logged successful run ({latency:.0f}ms)")

        # Clear current trace
        self.current_trace = {}

    def on_chain_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when a chain errors."""
        # Capture failures too
        start_time = self.current_trace.get("start_time", datetime.now())
        latency = (datetime.now() - start_time).total_seconds() * 1000

        trace = AgentTrace(
            prompt=self.current_trace.get("prompt", ""),
            output="",
            reasoning=self.current_trace.get("reasoning", []),
            tool_calls=self.current_trace.get("tool_calls", []),
            timestamp=datetime.now().isoformat(),
            success=False,
            error=str(error),
            latency_ms=latency,
            metadata={"framework": "langchain"}
        )

        with open(self.output_file, "a") as f:
            f.write(trace.to_json() + "\n")

        if self.verbose:
            print(f"[DeltaLoop] ✗ Logged failed run: {error}")

        # Clear current trace
        self.current_trace = {}

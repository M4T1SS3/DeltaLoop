"""Core data schema for DeltaLoop agent traces."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import json


@dataclass
class AgentTrace:
    """Universal format for agent execution logs.

    This schema provides a normalized representation of agent runs
    across different frameworks (LangChain, AutoGen, CrewAI, etc.).
    """

    # Core fields (required)
    prompt: str                     # User's original request
    output: str                     # Agent's final response
    timestamp: str                  # ISO 8601 format

    # Optional fields
    reasoning: Optional[List[str]] = None      # Chain-of-thought steps
    tool_calls: Optional[List[Dict]] = None    # Tools invoked
    success: Optional[bool] = None             # Outcome classification
    error: Optional[str] = None                # Error message if failed
    metadata: Optional[Dict] = None            # Framework-specific data

    # Computed fields
    token_count: Optional[int] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None

    def to_json(self) -> str:
        """Serialize to JSONL format.

        Returns:
            JSON string representation of the trace
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> 'AgentTrace':
        """Deserialize from JSONL.

        Args:
            line: JSON string representation

        Returns:
            AgentTrace instance
        """
        return cls(**json.loads(line))

    def __repr__(self) -> str:
        """Human-readable representation."""
        status = "✓" if self.success else "✗" if self.success is False else "?"
        return (
            f"AgentTrace({status} | "
            f"prompt={self.prompt[:50]}... | "
            f"output={self.output[:50]}... | "
            f"timestamp={self.timestamp})"
        )

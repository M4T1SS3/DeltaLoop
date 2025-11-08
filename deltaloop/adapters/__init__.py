"""Adapter layer for converting framework logs to normalized AgentTrace format.

DeltaLoop supports multiple agent frameworks through a plug-in adapter system.
Each adapter converts framework-specific logs into the universal AgentTrace format.

Available Adapters:
    - langchain: LangChain callback handler (DeltaLoopCallback)
    - generic: Manual logging for custom agents (GenericLogger)
"""

__all__ = []

# Adapters will be imported lazily to avoid requiring all dependencies
# Users only need to install the frameworks they're using

try:
    from .langchain import DeltaLoopCallback
    __all__.append("DeltaLoopCallback")
except ImportError:
    pass

try:
    from .generic import GenericLogger
    __all__.append("GenericLogger")
except ImportError:
    pass

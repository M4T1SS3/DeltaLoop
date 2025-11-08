"""DeltaLoop: Stop optimizing prompts. Start optimizing models.

An open-source continuous fine-tuning layer that shifts AI agent optimization
from the prompt level to the model level.
"""

__version__ = "0.1.0"
__author__ = "DeltaLoop Contributors"
__license__ = "Apache 2.0"

from .schema import AgentTrace
from .distill import distill_dataset, DataDistiller, DistillationStats
from .train import train, Trainer, TrainingConfig, TrainingResult
from .utils import load_model, load_model_for_training
from .eval import evaluate, Evaluator, EvalTask, EvalResult, EvalSummary, get_default_tasks
from .pipeline import Pipeline, PipelineConfig, PipelineResult, run_pipeline

__all__ = [
    "AgentTrace",
    "distill_dataset",
    "DataDistiller",
    "DistillationStats",
    "train",
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "load_model",
    "load_model_for_training",
    "evaluate",
    "Evaluator",
    "EvalTask",
    "EvalResult",
    "EvalSummary",
    "get_default_tasks",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
    "__version__",
]

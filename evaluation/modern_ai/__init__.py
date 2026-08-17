"""Modern AI experiments for HolUE/GalUE/MPRisk.

The package deliberately keeps the original face/text OSR pipeline untouched and
adapts the posterior/risk model to retrieval, RAG and tool-routing tasks.
"""

from .types import (
    RetrievalDataset,
    RetrievalProtocol,
    RetrievalScores,
    RAGRecord,
    ToolRoutingExample,
)
from .methods import ModernUncertaintyModel, PosteriorModelConfig

__all__ = [
    "RetrievalDataset",
    "RetrievalProtocol",
    "RetrievalScores",
    "RAGRecord",
    "ToolRoutingExample",
    "ModernUncertaintyModel",
    "PosteriorModelConfig",
]

"""Sample preparation, batches, losses, optimization, and DDP coordination."""

from .ddp import DistributedCoordinator, DistributedSettings
from .optimizer import ModelOptimizer, OptimizerMetrics

__all__ = [
    "DistributedCoordinator",
    "DistributedSettings",
    "ModelOptimizer",
    "OptimizerMetrics",
]

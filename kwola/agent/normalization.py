"""Mode-independent spatial normalization with stable schema-v3 state."""

from torch import Tensor, nn
from torch.nn import functional


class SpatialGroupNorm(nn.BatchNorm2d):
    """Apply GroupNorm while retaining BatchNorm parameter and buffer names.

    Keeping ``weight``, ``bias``, and running-statistic buffer names stable permits
    strict schema-v3 round trips while removing dependence on crop-specific running
    statistics. The running buffers are intentionally ignored by forward passes.
    """

    def __init__(self, channels: int, maximum_groups: int = 8) -> None:
        super().__init__(channels)
        self.groups = next(
            group for group in range(min(maximum_groups, channels), 0, -1) if channels % group == 0
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.group_norm(inputs, self.groups, self.weight, self.bias, self.eps)

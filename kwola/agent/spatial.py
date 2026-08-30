"""Global viewport coordinates shared by inference and recorded-sample training."""

import torch
from torch import Tensor


def coordinate_image(
    width: int,
    height: int,
    *,
    full_width: int | None = None,
    full_height: int | None = None,
    left: int = 0,
    top: int = 0,
    device: torch.device | None = None,
) -> Tensor:
    """Return two channels whose values preserve absolute viewport position."""
    total_width = width if full_width is None else full_width
    total_height = height if full_height is None else full_height
    if width < 1 or height < 1 or total_width < width or total_height < height:
        raise ValueError("coordinate image dimensions are invalid")
    if left < 0 or top < 0 or left + width > total_width or top + height > total_height:
        raise ValueError("coordinate image crop escapes the full viewport")

    x = _axis(total_width, device)[left : left + width]
    y = _axis(total_height, device)[top : top + height]
    return torch.stack(
        [x.reshape(1, width).expand(height, width), y.reshape(height, 1).expand(height, width)]
    )


def _axis(length: int, device: torch.device | None) -> Tensor:
    if length == 1:
        return torch.zeros(1, dtype=torch.float32, device=device)
    return torch.linspace(-1.0, 1.0, length, dtype=torch.float32, device=device)

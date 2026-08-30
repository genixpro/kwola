import pytest
import torch

from kwola.agent.model_backbone import BackboneInput
from kwola.agent.tracenet import TraceNet, TraceNetRequest
from kwola.config import profile_config


def inputs(batch: int, actions: int) -> TraceNetRequest:
    torch.manual_seed(19)
    image = torch.rand(batch, 1, 64, 64)
    recent_actions_image = torch.rand(batch, actions, 64, 64)
    recent_actions_vector = torch.rand(batch, 5 * actions)
    symbol_indexes = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    weights = torch.ones(4)
    symbols_set = torch.tensor([1, 2, 3], dtype=torch.long)
    symbol_mask = torch.zeros(batch, 3, dtype=torch.bool)
    step = torch.tensor([1.0, 2.0])
    masks = torch.ones(batch, actions, 64, 64)
    backbone = BackboneInput(
        image=image,
        recent_actions_image=recent_actions_image,
        recent_actions_vector=recent_actions_vector,
        recent_symbol_indexes=symbol_indexes,
        recent_symbol_offsets=offsets,
        recent_symbol_weights=weights,
        coverage_symbol_indexes=symbol_indexes,
        coverage_symbol_offsets=offsets,
        coverage_symbol_weights=weights,
        coverage_symbols_set=symbols_set,
        coverage_symbols_key_mask=symbol_mask,
        step_number=step,
    )
    return TraceNetRequest(backbone, masks, -10.0, output_stamp=True)


def test_refactored_tracenet_matches_captured_legacy_outputs() -> None:
    expected: dict[str, tuple[list[int], float, float, float]] = {
        "presentRewards": (
            [2, 4, 64, 64],
            72.80548095703125,
            -0.0585971400141716,
            -0.043252166360616684,
        ),
        "discountFutureRewards": (
            [2, 4, 64, 64],
            1040.2357177734375,
            -0.022762859240174294,
            0.04504061117768288,
        ),
        "stamp": (
            [2, 5, 2, 2],
            -0.4242258667945862,
            0.3010300099849701,
            -0.0969955325126648,
        ),
        "actionProbabilities": (
            [2, 4, 64, 64],
            2.000000238418579,
            5.971854625386186e-05,
            6.112419941928238e-05,
        ),
        "stateValues": (
            [2, 1],
            -0.14570394158363342,
            -0.038977183401584625,
            -0.1067267507314682,
        ),
        "advantage": (
            [2, 4, 64, 64],
            1489.890625,
            -0.0021966660860925913,
            0.05324368551373482,
        ),
    }
    config = profile_config("testing", "https://example.com", 1)
    torch.manual_seed(7)
    model = TraceNet(config.model, 4, 12, 37).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_486_562
    with torch.no_grad():
        outputs = model(inputs(2, 4))
    assert outputs.keys() == expected.keys()
    for name, (shape, total, first, last) in expected.items():
        value = outputs[name]
        assert list(value.shape) == shape
        assert float(value.sum()) == pytest.approx(total, rel=1e-5, abs=1e-6)
        assert float(value.flatten()[0]) == pytest.approx(first, rel=1e-5, abs=1e-6)
        assert float(value.flatten()[-1]) == pytest.approx(last, rel=1e-5, abs=1e-6)

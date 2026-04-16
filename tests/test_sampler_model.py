import torch
from sampler_model import SamplerThresholds


def test_output_shape():
    model = SamplerThresholds(d=4)
    thresholds = model()
    assert thresholds.shape == (4,), f"Expected shape (4,), got {thresholds.shape}"


def test_output_range():
    torch.manual_seed(42)
    model = SamplerThresholds(d=10)
    thresholds = model()
    assert (thresholds >= 0).all() and (thresholds <= 1).all(), \
        f"Thresholds must be in [0,1], got min={thresholds.min()}, max={thresholds.max()}"


def test_parameter_count():
    model = SamplerThresholds(d=6)
    params = list(model.parameters())
    total = sum(p.numel() for p in params)
    assert total == 6, f"Expected 6 parameters, got {total}"


def test_gradients_flow():
    torch.manual_seed(42)
    model = SamplerThresholds(d=4)
    thresholds = model()
    loss = thresholds.sum()
    loss.backward()
    assert model.raw_thresholds.grad is not None, "Gradients did not flow to raw_thresholds"
    assert (model.raw_thresholds.grad != 0).any(), "All gradients are zero"


def test_different_d_values():
    for d in [1, 3, 8, 12]:
        model = SamplerThresholds(d=d)
        thresholds = model()
        assert thresholds.shape == (d,), f"d={d}: Expected shape ({d},), got {thresholds.shape}"

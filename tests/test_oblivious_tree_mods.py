import numpy as np
import torch
from oblivious_tree import ObliviousTree
from sampler_model import SamplerThresholds
from classical_model import ClassicalThresholds


def test_threshold_type_sampler():
    model = ObliviousTree(
        d=4,
        feature_indices=[0, 1, 2, 3],
        num_classes=2,
        threshold_type='sampler',
    )
    assert isinstance(model.threshold_module, SamplerThresholds)
    thresholds = model.threshold_module()
    assert thresholds.shape == (4,)


def test_threshold_type_classical():
    model = ObliviousTree(
        d=4,
        feature_indices=[0, 1, 2, 3],
        num_classes=2,
        threshold_type='classical',
        classical_hidden_layers=0,
    )
    assert isinstance(model.threshold_module, ClassicalThresholds)


def test_threshold_type_overrides_use_classical():
    model = ObliviousTree(
        d=4,
        feature_indices=[0, 1, 2, 3],
        num_classes=2,
        use_classical=True,
        threshold_type='sampler',
    )
    assert isinstance(model.threshold_module, SamplerThresholds)


def test_backward_compat_use_classical():
    model = ObliviousTree(
        d=4,
        feature_indices=[0, 1, 2, 3],
        num_classes=2,
        use_classical=True,
    )
    assert isinstance(model.threshold_module, ClassicalThresholds)


def test_optimizer_type_sgd():
    model = ObliviousTree(
        d=4,
        feature_indices=[0, 1, 2, 3],
        num_classes=2,
        threshold_type='sampler',
        optimizer_type='sgd',
    )
    assert isinstance(model.optimizer, torch.optim.SGD)


def test_optimizer_type_adam_default():
    model = ObliviousTree(
        d=4,
        feature_indices=[0, 1, 2, 3],
        num_classes=2,
        threshold_type='sampler',
    )
    assert isinstance(model.optimizer, torch.optim.Adam)


def test_sampler_trains():
    """Smoke test: sampler-based OBT can complete a training loop."""
    np.random.seed(42)
    torch.manual_seed(42)
    X_train = np.random.rand(50, 4).astype(np.float32)
    Y_train = (X_train[:, 0] > 0.5).astype(np.int64)
    X_val = np.random.rand(20, 4).astype(np.float32)
    Y_val = (X_val[:, 0] > 0.5).astype(np.int64)

    model = ObliviousTree(
        d=2,
        feature_indices=[0, 1],
        num_classes=2,
        threshold_type='sampler',
        epochs=5,
        batch_size=25,
    )
    _, thresholds, mu_b, history = model.train(X_train, Y_train, X_val=X_val, Y_val=Y_val)
    assert len(history['epoch']) == 5
    assert thresholds.shape == (2,)
    assert mu_b.shape == (4, 2)  # 2^2 leaves x 2 classes

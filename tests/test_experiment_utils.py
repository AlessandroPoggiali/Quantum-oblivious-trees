import os
import csv
import numpy as np
import torch
from experiment_utils import (
    set_random_seed,
    load_and_prepare_dataset,
    compute_actual_depths,
    save_results_csv,
    train_and_evaluate_model,
)
from oblivious_tree import ObliviousTree


def test_set_random_seed_reproducible():
    set_random_seed(42)
    a = torch.rand(5)
    set_random_seed(42)
    b = torch.rand(5)
    assert torch.equal(a, b), "Seeds did not produce identical results"


def test_compute_actual_depths_all_fit():
    depths = compute_actual_depths(num_features=20, depth_grid=[4, 8, 12])
    assert depths == [4, 8, 12]


def test_compute_actual_depths_capped():
    depths = compute_actual_depths(num_features=6, depth_grid=[4, 8, 12])
    assert depths == [4, 6], f"Expected [4, 6] (deduplicated), got {depths}"


def test_compute_actual_depths_small():
    depths = compute_actual_depths(num_features=3, depth_grid=[4, 8, 12])
    assert depths == [3], f"Expected [3] (all capped to 3, deduplicated), got {depths}"


def test_load_and_prepare_dataset():
    data = load_and_prepare_dataset('iris')
    assert data is not None
    X_train, X_val, X_test, Y_train, Y_val, Y_test, num_classes, num_features = data
    assert X_train.shape[1] == num_features
    assert num_classes >= 2
    assert X_train.min() >= -0.01  # approximately in [0,1] after MinMax
    assert X_train.max() <= 1.01


def test_save_results_csv(tmp_path):
    results = [
        {'dataset': 'iris', 'accuracy': 0.95, 'approach': 'ffnn'},
        {'dataset': 'wine', 'accuracy': 0.88, 'approach': 'sampler'},
    ]
    path = str(tmp_path / 'test_results.csv')
    save_results_csv(results, path)
    assert os.path.exists(path)
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]['dataset'] == 'iris'


def test_train_and_evaluate_model():
    np.random.seed(42)
    torch.manual_seed(42)
    X_train = np.random.rand(50, 4).astype(np.float32)
    Y_train = (X_train[:, 0] > 0.5).astype(np.int64)
    X_val = np.random.rand(20, 4).astype(np.float32)
    Y_val = (X_val[:, 0] > 0.5).astype(np.int64)
    X_test = np.random.rand(20, 4).astype(np.float32)
    Y_test = (X_test[:, 0] > 0.5).astype(np.int64)

    model = ObliviousTree(
        d=2, feature_indices=[0, 1], num_classes=2,
        threshold_type='sampler', epochs=3, batch_size=25,
    )
    metrics, history = train_and_evaluate_model(
        model, X_train, Y_train, X_val, Y_val, X_test, Y_test, torch.device('cpu')
    )
    assert 'test_acc' in metrics
    assert 'test_ce' in metrics
    assert 0.0 <= metrics['test_acc'] <= 1.0
    assert len(history['epoch']) == 3

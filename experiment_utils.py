import os
import csv
import traceback
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from gentree_utils import load_data, DATASETS
from oblivious_tree import ObliviousTree


def log_error(study_name: str, dataset_name: str, error: Exception,
              log_path: str = "errors.log"):
    """Append an error entry to the shared errors.log file."""
    timestamp = datetime.now().isoformat()
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {study_name} | dataset={dataset_name}\n")
        f.write("".join(tb))
        f.write("\n")


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_and_prepare_dataset(name: str) -> Optional[Tuple]:
    result = load_data(name)
    if result is None:
        return None
    X_train, X_val, X_test, Y_train, Y_val, Y_test = result
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    num_classes = len(np.unique(np.concatenate([Y_train, Y_val, Y_test])))
    num_features = X_train.shape[1]
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, num_classes, num_features


def compute_actual_depths(num_features: int, depth_grid: List[int] = [4, 8, 12]) -> List[int]:
    actual = []
    seen = set()
    for d in depth_grid:
        actual_d = min(d, num_features)
        if actual_d not in seen:
            seen.add(actual_d)
            actual.append(actual_d)
    return actual


def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, device):
    theta, thresholds, mu_b, history = model.train(
        X_train, Y_train, X_val=X_val, Y_val=Y_val, save_every=9999, ckpt_path=None,
    )
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.long, device=device)
    with torch.no_grad():
        yhat_test, y_pred_test, _ = model.predict_batch(
            X_test_t, torch.from_numpy(mu_b).float().to(device),
            model.threshold_module().float(), alpha=model.alpha_final,
        )
        test_acc = (y_pred_test == Y_test_t).float().mean().item()
        test_ce = nn.functional.cross_entropy(yhat_test, Y_test_t).item()
    val_acc = history['val_acc'][-1] if 'val_acc' in history and history['val_acc'] else 0.0
    val_ce = history['val_bce'][-1] if 'val_bce' in history and history['val_bce'] else 0.0
    metrics = {
        'test_acc': test_acc, 'test_ce': test_ce,
        'val_acc': val_acc, 'val_ce': val_ce,
        'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0.0,
        'final_train_ce': history['train_bce'][-1] if history['train_bce'] else 0.0,
    }
    return metrics, history


def save_results_csv(results_list, path):
    if not results_list:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fieldnames = list(results_list[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)


def summarize_histories(histories):
    if not histories:
        return None
    epochs = np.array(histories[0]['epoch'])
    train_ce = np.array([h['train_bce'] for h in histories])
    train_acc = np.array([h['train_acc'] for h in histories])
    summary = {
        'epochs': epochs,
        'train_ce_mean': train_ce.mean(axis=0), 'train_ce_std': train_ce.std(axis=0),
        'train_acc_mean': train_acc.mean(axis=0), 'train_acc_std': train_acc.std(axis=0),
    }
    if 'val_bce' in histories[0] and histories[0]['val_bce']:
        val_ce = np.array([h['val_bce'] for h in histories])
        val_acc = np.array([h['val_acc'] for h in histories])
        summary['val_ce_mean'] = val_ce.mean(axis=0)
        summary['val_ce_std'] = val_ce.std(axis=0)
        summary['val_acc_mean'] = val_acc.mean(axis=0)
        summary['val_acc_std'] = val_acc.std(axis=0)
    return summary


def wandb_log_run(
    study_name: str,
    config: dict,
    metrics: dict,
    history: dict = None,
    project: str = "oblivious_trees_tests",
    entity: str = "quantum_kets",
):
    """Log a single run (one seed) to Weights & Biases."""
    import wandb
    group = f"{config['dataset']}/d{config['depth']}/{config.get('approach', 'dt')}"
    run = wandb.init(
        project=project, entity=entity,
        name=f"{config.get('approach', 'dt')}_seed{config.get('seed', 0)}",
        group=group, tags=[study_name, config['dataset']],
        config=config, reinit=True,
    )
    if history:
        for i, epoch in enumerate(history['epoch']):
            log = {'epoch': epoch, 'train_ce': history['train_bce'][i], 'train_acc': history['train_acc'][i]}
            if 'val_bce' in history and i < len(history['val_bce']):
                log['val_ce'] = history['val_bce'][i]
            if 'val_acc' in history and i < len(history['val_acc']):
                log['val_acc'] = history['val_acc'][i]
            run.log(log)
    run.summary.update(metrics)
    run.finish()


def plot_convergence_compare(histories_dict, out_dir, dataset_name, depth):
    os.makedirs(out_dir, exist_ok=True)
    summaries = {}
    for label, hists in histories_dict.items():
        s = summarize_histories(hists)
        if s is not None:
            summaries[label] = s
    if not summaries:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, s in summaries.items():
        epochs = s['epochs']
        ax.plot(epochs, s['train_ce_mean'], label=f'{label} Train CE')
        ax.fill_between(epochs, s['train_ce_mean'] - s['train_ce_std'],
                        s['train_ce_mean'] + s['train_ce_std'], alpha=0.2)
        if 'val_ce_mean' in s:
            ax.plot(epochs, s['val_ce_mean'], '--', label=f'{label} Val CE')
            ax.fill_between(epochs, s['val_ce_mean'] - s['val_ce_std'],
                            s['val_ce_mean'] + s['val_ce_std'], alpha=0.1)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title(f'{dataset_name} (d={depth}) — Loss')
    ax.legend(fontsize=8); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'convergence_loss_d{depth}.png'), dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, s in summaries.items():
        epochs = s['epochs']
        ax.plot(epochs, s['train_acc_mean'], label=f'{label} Train Acc')
        ax.fill_between(epochs, s['train_acc_mean'] - s['train_acc_std'],
                        s['train_acc_mean'] + s['train_acc_std'], alpha=0.2)
        if 'val_acc_mean' in s:
            ax.plot(epochs, s['val_acc_mean'], '--', label=f'{label} Val Acc')
            ax.fill_between(epochs, s['val_acc_mean'] - s['val_acc_std'],
                            s['val_acc_mean'] + s['val_acc_std'], alpha=0.1)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{dataset_name} (d={depth}) — Accuracy')
    ax.legend(fontsize=8); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'convergence_acc_d{depth}.png'), dpi=100)
    plt.close(fig)

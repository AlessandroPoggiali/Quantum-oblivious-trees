import os
import argparse
from typing import Optional, Tuple, Dict, Any
import csv
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import DecisionBoundaryDisplay
import graphviz
from gentree_utils import load_data, DATASETS
from oblivious_tree import ObliviousTree, extract_rules




# ------------------------ Plotting ------------------------

def plot_history(history: Dict[str, list], out_path_loss: str, out_path_acc: str):
    plt.figure()
    plt.plot(history['epoch'], history['train_bce'], label='Train CE')
    if 'val_bce' in history:
        plt.plot(history['epoch'], history['val_bce'], '--', label='Val CE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path_loss)
    plt.close()


    plt.figure()
    plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        plt.plot(history['epoch'], history['val_acc'], '--', label='Val Acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path_acc)
    plt.close()


def _summarize_histories(histories: list) -> Optional[Dict[str, np.ndarray]]:
    if not histories:
        return None

    epochs = np.array(histories[0]['epoch'])
    train_ce = np.array([h['train_bce'] for h in histories])
    train_acc = np.array([h['train_acc'] for h in histories])
    val_ce = np.array([h['val_bce'] for h in histories]) if 'val_bce' in histories[0] else None
    val_acc = np.array([h['val_acc'] for h in histories]) if 'val_acc' in histories[0] else None

    summary = {
        'epochs': epochs,
        'train_ce_mean': train_ce.mean(axis=0),
        'train_ce_std': train_ce.std(axis=0),
        'train_acc_mean': train_acc.mean(axis=0),
        'train_acc_std': train_acc.std(axis=0),
    }
    if val_ce is not None:
        summary['val_ce_mean'] = val_ce.mean(axis=0)
        summary['val_ce_std'] = val_ce.std(axis=0)
    if val_acc is not None:
        summary['val_acc_mean'] = val_acc.mean(axis=0)
        summary['val_acc_std'] = val_acc.std(axis=0)
    return summary


def plot_avg_variance_compare(classical_histories: list,
                              quantum_histories: list,
                              out_path_loss: str,
                              out_path_acc: str):
    classical = _summarize_histories(classical_histories)
    quantum = _summarize_histories(quantum_histories)

    if classical is None and quantum is None:
        return

    plt.figure()
    if classical is not None:
        epochs = classical['epochs']
        plt.plot(epochs, classical['train_ce_mean'], label='Classical Train CE')
        plt.fill_between(epochs, classical['train_ce_mean'] - classical['train_ce_std'],
                         classical['train_ce_mean'] + classical['train_ce_std'], alpha=0.2)
        if 'val_ce_mean' in classical:
            plt.plot(epochs, classical['val_ce_mean'], '--', label='Classical Val CE')
            plt.fill_between(epochs, classical['val_ce_mean'] - classical['val_ce_std'],
                             classical['val_ce_mean'] + classical['val_ce_std'], alpha=0.2)
    if quantum is not None:
        epochs = quantum['epochs']
        plt.plot(epochs, quantum['train_ce_mean'], label='Quantum Train CE')
        plt.fill_between(epochs, quantum['train_ce_mean'] - quantum['train_ce_std'],
                         quantum['train_ce_mean'] + quantum['train_ce_std'], alpha=0.2)
        if 'val_ce_mean' in quantum:
            plt.plot(epochs, quantum['val_ce_mean'], '--', label='Quantum Val CE')
            plt.fill_between(epochs, quantum['val_ce_mean'] - quantum['val_ce_std'],
                             quantum['val_ce_mean'] + quantum['val_ce_std'], alpha=0.2)
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path_loss)
    plt.close()

    plt.figure()
    if classical is not None:
        epochs = classical['epochs']
        plt.plot(epochs, classical['train_acc_mean'], label='Classical Train Acc')
        plt.fill_between(epochs, classical['train_acc_mean'] - classical['train_acc_std'],
                         classical['train_acc_mean'] + classical['train_acc_std'], alpha=0.2)
        if 'val_acc_mean' in classical:
            plt.plot(epochs, classical['val_acc_mean'], '--', label='Classical Val Acc')
            plt.fill_between(epochs, classical['val_acc_mean'] - classical['val_acc_std'],
                             classical['val_acc_mean'] + classical['val_acc_std'], alpha=0.2)
    if quantum is not None:
        epochs = quantum['epochs']
        plt.plot(epochs, quantum['train_acc_mean'], label='Quantum Train Acc')
        plt.fill_between(epochs, quantum['train_acc_mean'] - quantum['train_acc_std'],
                         quantum['train_acc_mean'] + quantum['train_acc_std'], alpha=0.2)
        if 'val_acc_mean' in quantum:
            plt.plot(epochs, quantum['val_acc_mean'], '--', label='Quantum Val Acc')
            plt.fill_between(epochs, quantum['val_acc_mean'] - quantum['val_acc_std'],
                             quantum['val_acc_mean'] + quantum['val_acc_std'], alpha=0.2)
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path_acc)
    plt.close()

def plot_decision_boundary_sklearn(model, X_test, Y_test, mu, thresholds, alpha, device='cpu', out_path: str = 'decision_boundary.png'):
    
    # Convert to tensor
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)

    # Create a mesh grid over data range
    x_min, x_max = X_test[:, 0].min() - 0.2, X_test[:, 0].max() + 0.2
    y_min, y_max = X_test[:, 1].min() - 0.2, X_test[:, 1].max() + 0.2
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    
    # Flatten grid into (90000, 2)
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Predict over grid
        yhat_grid, _ = model.predict_batch(grid_t, mu, thresholds, alpha=alpha)
        yhat_grid = (yhat_grid > 0.5).float()
        # Convert to numpy and reshape to grid
        Z = yhat_grid.cpu().numpy().reshape(xx.shape)

    display = DecisionBoundaryDisplay(
        xx0=xx, xx1=yy, response=Z
    )
    display.plot()

    display.ax_.scatter(X_test[:,0], X_test[:,1], c=Y_test, edgecolors='k')

    plt.title("Decision Boundary from Final Rules")
    plt.xlabel("F0")
    plt.ylabel("F1")
    plt.savefig(out_path)
    plt.close()

def plot_decision_boundary(model, X_test, Y_test, mu, thresholds, alpha, device='cpu', out_path: str = 'decision_boundary.png'):
    """
    model: your trained rule model
    X_test: numpy array (N,2)
    Y_test: numpy array (N,)
    """

    # Convert to tensor
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)

    # Create a mesh grid over data range
    x_min, x_max = X_test[:, 0].min() - 0.2, X_test[:, 0].max() + 0.2
    y_min, y_max = X_test[:, 1].min() - 0.2, X_test[:, 1].max() + 0.2
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Flatten grid into (90000, 2)
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Predict over grid
        yhat_grid, _ = model.predict_batch(grid_t, mu, thresholds, alpha=alpha)
        yhat_grid = (yhat_grid > 0.5).float()
        # Convert to numpy and reshape to grid
        Z = yhat_grid.cpu().numpy().reshape(xx.shape)

    # ---------- Plotting ----------
    plt.figure(figsize=(8,6))

    # Decision boundary = contour at probability 0.5
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='coolwarm')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # Test points
    Y_test_0 = (Y_test == 0).astype(np.float32)
    Y_test_1 = (Y_test == 1).astype(np.float32) 
    plt.scatter(X_test[:,0], X_test[:,1], c=Y_test_0, cmap='coolwarm', edgecolors='k', label="Class 0")
    plt.scatter(X_test[:,0], X_test[:,1], c=Y_test_1, cmap='coolwarm', edgecolors='k', label="Class 1")
    plt.legend()
    
    plt.title("Decision Boundary from Final Rules")
    plt.xlabel("F0")
    plt.ylabel("F1")
    plt.savefig(out_path)
    plt.close()

# ------------------------ CLI & main ------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=4, help='Tree depth')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--use-bias', type=str, default='y', choices=['n','y'], help='Use bias in classical thresholds')
    parser.add_argument('--alpha-init', type=float, default=1.0, help='Initial alpha')
    parser.add_argument('--alpha-final', type=float, default=20.0, help='Final alpha')
    parser.add_argument('--use-ema', type=str, default='n', choices=['n','y'], help='Use EMA')
    parser.add_argument('--ema-rho', type=float, default=0.05, help='EMA rho')
    parser.add_argument('--q-reps', type=int, default=2, help='Quantum repetitions')
    parser.add_argument('--q-dev', type=str, default='default.qubit', help='Quantum device')
    parser.add_argument('--q-shots', type=int, default=0, help='Quantum shots')
    parser.add_argument('--ansatz', type=str, default='ry', choices=['ry','two_param'], help='Quantum ansatz')
    parser.add_argument('--output-csv', type=str, default='results.csv', help='Output CSV file for results')
    parser.add_argument('--datasets', type=str, default='all', help='Specific datasets (comma-separated) or "all"')
    parser.add_argument('--single-dataset', type=str, default=None, help='Test single dataset (legacy)')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs per dataset with different random seeds')
    return parser.parse_args()


# ======================== COMPARISON FRAMEWORK ========================

def set_random_seed(seed: int):
    """Set random seed for reproducibility across all libraries"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def count_quantum_params(d: int, reps: int, ansatz: str = 'ry') -> int:
    """Count parameters in quantum threshold module"""
    if ansatz == 'ry':
        return reps * d
    elif ansatz == 'two_param':
        return reps * d * 2
    return reps * d

def count_classical_params(d: int, use_bias: str, hidden_layers: int, hidden_size: int) -> int:
    """Count parameters in classical threshold module"""
    if hidden_layers < 1:
        # Linear: 1*d + bias
        return d + (1 if use_bias == 'y' else 0) * d  # Add bias parameters if used
    else:
        # First layer: 1*hidden_size + bias
        params = hidden_size + (1 if use_bias == 'y' else 0) * hidden_size  # Add bias parameters if used
        # Hidden layers: hidden_size*hidden_size + bias each
        for _ in range(hidden_layers - 1):
            params += hidden_size * hidden_size + (1 if use_bias == 'y' else 0) * hidden_size
        # Output layer: hidden_size*d + d
        params += hidden_size * d + (1 if use_bias == 'y' else 0) * d  # Add bias parameters if used
        return params

def find_matching_quantum_config(d: int, target_params: int, q_reps_max: int = 20) -> Optional[Tuple[int, str]]:
    """Find quantum config (reps, ansatz) that matches target parameters"""
    # Try RY ansatz: params = reps * d
    if target_params % d == 0:
        reps = target_params // d
        if reps > 0 and reps <= q_reps_max:
            return (reps, 'ry')
    
    # Try two_param ansatz: params = reps * d * 2
    if target_params % (2 * d) == 0:
        reps = target_params // (2 * d)
        if reps > 0 and reps <= q_reps_max:
            return (reps, 'two_param')
    
    return None

def train_and_evaluate(model: ObliviousTree, 
                      X_train: np.ndarray, Y_train: np.ndarray,
                      X_val: np.ndarray, Y_val: np.ndarray,
                      X_test: np.ndarray, Y_test: np.ndarray,
                      device: torch.device) -> Tuple[Dict[str, float], Dict[str, list]]:
    """Train model and return metrics"""
    
    theta, thresholds, mu_b, history = model.train(
        X_train, Y_train, 
        X_val=X_val, Y_val=Y_val, 
        save_every=100,  # Don't save checkpoints
        ckpt_path=None
    )
    
    # Evaluate on test set
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.long, device=device)
    
    with torch.no_grad():
        yhat_test, y_pred_test, _ = model.predict_batch(
            X_test_t, 
            torch.from_numpy(mu_b).float(), 
            model.threshold_module().float(), 
            alpha=model.alpha_final
        )
        test_acc = (y_pred_test == Y_test_t).float().mean().item()
        test_ce = nn.functional.cross_entropy(yhat_test, Y_test_t).item()
    
    # Get validation metrics
    val_acc = history['val_acc'][-1] if 'val_acc' in history and len(history['val_acc']) > 0 else 0.0
    val_ce = history['val_bce'][-1] if 'val_bce' in history and len(history['val_bce']) > 0 else 0.0
    
    metrics = {
        'test_acc': test_acc,
        'test_ce': test_ce,
        'val_acc': val_acc,
        'val_ce': val_ce,
        'final_train_acc': history['train_acc'][-1] if len(history['train_acc']) > 0 else 0.0,
        'final_train_ce': history['train_bce'][-1] if len(history['train_bce']) > 0 else 0.0,
    }
    return metrics, history

def test_dataset(dataset_name: str, args, results_list: list, run_results_list: list) -> bool:
    """Test a single dataset with both classical and quantum models"""
    print(f"\n{'='*80}")
    print(f"Testing dataset: {dataset_name}")
    print(f"{'='*80}")
    
    try:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(dataset_name)
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return False
    
    num_classes = len(np.unique(np.concatenate([Y_train, Y_val, Y_test])))
    device = torch.device('cpu')
    #feature_indices = list(range(min(args.d, X_train.shape[1])))
    #actual_d = len(feature_indices)
    feature_indices = np.random.choice(X_train.shape[1], size=min(args.d, X_train.shape[1]), replace=False).tolist()
    actual_d = len(feature_indices)
    
    print(f"Dataset shape: {X_train.shape}, Classes: {num_classes}, Using depth: {actual_d}")
    
    classical_params = count_classical_params(actual_d, args.use_bias, hidden_layers=0, hidden_size=0)
    print(f"Classical parameters: {classical_params}")
    
    quantum_config = find_matching_quantum_config(actual_d, classical_params, q_reps_max=20)
    if quantum_config is None:
        q_reps = args.q_reps
        q_ansatz = args.ansatz
        quantum_params = count_quantum_params(actual_d, q_reps, q_ansatz)
        print(f"Cannot match parameters for quantum model (classical params={classical_params}, d={actual_d})")
    else:
        q_reps, q_ansatz = quantum_config
        quantum_params = classical_params
        print(f"Quantum parameters: {quantum_params} (reps={q_reps}, ansatz={q_ansatz})")
    
    # Store metrics from all runs for averaging
    run_results = defaultdict(list)
    classical_histories = []
    quantum_histories = []
    
    # ===== MULTIPLE RUNS WITH DIFFERENT SEEDS =====
    print(f"\nRunning {args.num_runs} iterations with different seeds...")
    
    for run_idx in range(args.num_runs):
        seed = 42*(run_idx+1)+1234
        set_random_seed(seed)
        
        if args.num_runs > 1:
            print(f"  Run {run_idx + 1}/{args.num_runs} (seed={seed})", end="", flush=True)
        
        # ===== CLASSICAL MODEL (no hidden layers) =====
        try:
            classical_model = ObliviousTree(
                d=actual_d,
                feature_indices=feature_indices,
                device=device,
                alpha_init=args.alpha_init,
                alpha_final=args.alpha_final,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                use_ema=args.use_ema=='y',
                ema_rho=args.ema_rho,
                num_classes=num_classes,
                use_classical=True,
                classical_hidden_layers=0,
                classical_hidden_size=0,
                use_bias=args.use_bias=='y'
            )
            classical_metrics, classical_history = train_and_evaluate(
                classical_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, device
            )
            classical_histories.append(classical_history)
            run_results_list.append({
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset_name,
                'run_idx': run_idx + 1,
                'seed': seed,
                'model': 'classical',
                'test_acc': classical_metrics['test_acc'],
                'test_ce': classical_metrics['test_ce'],
                'val_acc': classical_metrics['val_acc'],
                'val_ce': classical_metrics['val_ce'],
                'final_train_acc': classical_metrics['final_train_acc'],
                'final_train_ce': classical_metrics['final_train_ce'],
            })
            run_results['classical_test_acc'].append(classical_metrics['test_acc'])
            run_results['classical_test_ce'].append(classical_metrics['test_ce'])
            run_results['classical_val_acc'].append(classical_metrics['val_acc'])
            run_results['classical_val_ce'].append(classical_metrics['val_ce'])
        except Exception as e:
            print(f"Classical model failed: {e}")
            return False
        
        # ===== QUANTUM MODEL (matched parameters) =====
        if quantum_config is not None:
            try:
                quantum_model = ObliviousTree(
                    d=actual_d,
                    feature_indices=feature_indices,
                    device=device,
                    alpha_init=args.alpha_init,
                    alpha_final=args.alpha_final,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    use_ema=args.use_ema=='y',
                    ema_rho=args.ema_rho,
                    num_classes=num_classes,
                    q_reps=q_reps,
                    q_dev=args.q_dev,
                    q_shots=args.q_shots,
                    ansatz=q_ansatz,
                    use_classical=False
                )
                quantum_metrics, quantum_history = train_and_evaluate(
                    quantum_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, device
                )
                quantum_histories.append(quantum_history)
                run_results_list.append({
                    'timestamp': datetime.now().isoformat(),
                    'dataset': dataset_name,
                    'run_idx': run_idx + 1,
                    'seed': seed,
                    'model': 'quantum',
                    'test_acc': quantum_metrics['test_acc'],
                    'test_ce': quantum_metrics['test_ce'],
                    'val_acc': quantum_metrics['val_acc'],
                    'val_ce': quantum_metrics['val_ce'],
                    'final_train_acc': quantum_metrics['final_train_acc'],
                    'final_train_ce': quantum_metrics['final_train_ce'],
                })
                run_results['quantum_test_acc'].append(quantum_metrics['test_acc'])
                run_results['quantum_test_ce'].append(quantum_metrics['test_ce'])
                run_results['quantum_val_acc'].append(quantum_metrics['val_acc'])
                run_results['quantum_val_ce'].append(quantum_metrics['val_ce'])
            except Exception as e:
                if args.num_runs > 1:
                    print(f" (quantum failed)")
                else:
                    print(f"Quantum model failed: {e}")
        
        if args.num_runs > 1:
            print(f" ✓")
    
    # ===== AVERAGE RESULTS ACROSS RUNS =====
    if args.num_runs > 1:
        print(f"\nAveraging results across {args.num_runs} runs...")
    
    classical_metrics = {}
    for metric in ['test_acc', 'test_ce', 'val_acc', 'val_ce']:
        key = f'classical_{metric}'
        if key in run_results and len(run_results[key]) > 0:
            classical_metrics[metric] = np.mean(run_results[key])
        else:
            classical_metrics[metric] = 0.0
    
    quantum_metrics = None
    if quantum_config is not None and len(run_results.get('quantum_test_acc', [])) > 0:
        quantum_metrics = {}
        for metric in ['test_acc', 'test_ce', 'val_acc', 'val_ce']:
            key = f'quantum_{metric}'
            if key in run_results and len(run_results[key]) > 0:
                quantum_metrics[metric] = np.mean(run_results[key])
            else:
                quantum_metrics[metric] = 0.0
    
    # ===== COMPARISON =====
    print(f"\n--- COMPARISON (averaged) ---")
    print(f"Classical Test Accuracy: {classical_metrics['test_acc']:.4f}")
    if quantum_metrics:
        print(f"Quantum Test Accuracy:   {quantum_metrics['test_acc']:.4f}")
        acc_diff = quantum_metrics['test_acc'] - classical_metrics['test_acc']
        if acc_diff > 0.01:
            print(f"QUANTUM BETTER by {acc_diff:.4f}")
            better = "QUANTUM"
        elif acc_diff < -0.01:
            print(f"CLASSICAL BETTER by {-acc_diff:.4f}")
            better = "CLASSICAL"
        else:
            print(f"Similar performance (diff: {acc_diff:.4f})")
            better = "SIMILAR"
    else:
        better = "QUANTUM_FAILED"
    
    if args.num_runs > 1:
        print(f"\n--- Standard Deviations (across {args.num_runs} runs) ---")
        class_acc_std = np.std(run_results.get('classical_test_acc', [0])) if 'classical_test_acc' in run_results else 0
        print(f"Classical Test Accuracy Std: {class_acc_std:.4f}")
        if quantum_metrics and 'quantum_test_acc' in run_results:
            quant_acc_std = np.std(run_results['quantum_test_acc'])
            print(f"Quantum Test Accuracy Std:   {quant_acc_std:.4f}")
    
    # Record result (with standard deviations if multiple runs)
    result = {
        'timestamp': datetime.now().isoformat(),
        'num_runs': args.num_runs,
        'dataset': dataset_name,
        'depth': actual_d,
        'num_features': X_train.shape[1],
        'num_classes': num_classes,
        'num_samples_train': X_train.shape[0],
        'num_samples_val': X_val.shape[0],
        'num_samples_test': X_test.shape[0],
        'classical_params': classical_params,
        'classical_test_acc': classical_metrics['test_acc'],
        'classical_test_acc_std': np.std(run_results.get('classical_test_acc', [0])) if 'classical_test_acc' in run_results else 0,
        'classical_test_ce': classical_metrics['test_ce'],
        'classical_test_ce_std': np.std(run_results.get('classical_test_ce', [0])) if 'classical_test_ce' in run_results else 0,
        'classical_val_acc': classical_metrics['val_acc'],
        'classical_val_ce': classical_metrics['val_ce'],
        'quantum_params': quantum_params if quantum_metrics else None,
        'quantum_test_acc': quantum_metrics['test_acc'] if quantum_metrics else None,
        'quantum_test_acc_std': np.std(run_results.get('quantum_test_acc', [0])) if 'quantum_test_acc' in run_results else 0,
        'quantum_test_ce': quantum_metrics['test_ce'] if quantum_metrics else None,
        'quantum_test_ce_std': np.std(run_results.get('quantum_test_ce', [0])) if 'quantum_test_ce' in run_results else 0,
        'quantum_val_acc': quantum_metrics['val_acc'] if quantum_metrics else None,
        'quantum_val_ce': quantum_metrics['val_ce'] if quantum_metrics else None,
        'quantum_config': f"{q_reps}_{q_ansatz}" if quantum_metrics else "FAILED",
        'better': better,
    }
    results_list.append(result)

    plot_dir = os.path.join('plots', dataset_name)
    os.makedirs(plot_dir, exist_ok=True)
    plot_avg_variance_compare(
        classical_histories,
        quantum_histories,
        os.path.join(plot_dir, 'avg_loss.png'),
        os.path.join(plot_dir, 'avg_acc.png')
    )
    
    return True

def save_results_csv(results_list: list, output_file: str):
    """Save results to CSV file"""
    if not results_list:
        print("No results to save")
        return
    
    fieldnames = results_list[0].keys()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)
    print(f"\nResults saved to {output_file}")

def save_run_results_csv(run_results_list: list, output_file: str):
    """Save per-run results to CSV file"""
    if not run_results_list:
        print("No per-run results to save")
        return

    fieldnames = run_results_list[0].keys()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_results_list)
    print(f"\nPer-run results saved to {output_file}")

def print_summary(results_list: list):
    """Print summary statistics"""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if not results_list:
        print("No results to summarize")
        return
    
    total_tests = len(results_list)
    quantum_better = sum(1 for r in results_list if r['better'] == 'QUANTUM')
    classical_better = sum(1 for r in results_list if r['better'] == 'CLASSICAL')
    similar = sum(1 for r in results_list if r['better'] == 'SIMILAR')
    failed = sum(1 for r in results_list if r['better'] == 'QUANTUM_FAILED')
    
    print(f"Total datasets tested: {total_tests}")
    print(f"Quantum better: {quantum_better} ({100*quantum_better/total_tests:.1f}%)")
    print(f"Classical better: {classical_better} ({100*classical_better/total_tests:.1f}%)")
    print(f"Similar: {similar} ({100*similar/total_tests:.1f}%)")
    print(f"Quantum failed: {failed} ({100*failed/total_tests:.1f}%)")
    
    # List quantum-better datasets
    if quantum_better > 0:
        print(f"\nDatasets where QUANTUM performed better:")
        for r in results_list:
            if r['better'] == 'QUANTUM':
                acc_diff = r['quantum_test_acc'] - r['classical_test_acc']
                print(f"  - {r['dataset']}: Quantum {r['quantum_test_acc']:.4f} vs Classical {r['classical_test_acc']:.4f} (diff: +{acc_diff:.4f})")

def main(args):
    print(f"Testing framework for Quantum vs Classical Oblivious Trees")
    print(f"Configuration: depth={args.d}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, runs={args.num_runs}")
    
    # Determine which datasets to test
    if args.single_dataset:
        datasets_to_test = [args.single_dataset]
    elif args.datasets.lower() == 'all':
        datasets_to_test = DATASETS
    else:
        datasets_to_test = [d.strip() for d in args.datasets.split(',')]
    
    print(f"Datasets to test: {datasets_to_test}")
    
    results_list = []
    run_results_list = []
    successful = 0
    
    for dataset in datasets_to_test:
        if test_dataset(dataset, args, results_list, run_results_list):
            successful += 1
    
    print(f"\n{'='*80}")
    print(f"Completed: {successful}/{len(datasets_to_test)} datasets")
    print(f"{'='*80}")
    
    print_summary(results_list)
    save_results_csv(results_list, args.output_csv)
    runs_csv = args.output_csv.replace('.csv', '_runs.csv')
    save_run_results_csv(run_results_list, runs_csv)


if __name__ == '__main__':
    args = parse_args()
    main(args)
import os
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch

from gentree_utils import DATASETS
from oblivious_tree import ObliviousTree
from experiment_utils import (
    set_random_seed,
    load_and_prepare_dataset,
    compute_actual_depths,
    train_and_evaluate_model,
    save_results_csv,
    plot_convergence_compare,
    wandb_log_run,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Study 2: FFNN with Hidden Layers')
    parser.add_argument('--d-values', type=str, default='4,8,12',
                        help='Comma-separated depth values')
    parser.add_argument('--hidden-sizes', type=str, default='4,8,16',
                        help='Comma-separated hidden layer sizes to explore')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument('--output-dir', type=str, default='results/study2')
    parser.add_argument('--alpha-init', type=float, default=1.0)
    parser.add_argument('--alpha-final', type=float, default=20.0)
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    return parser.parse_args()


def compute_param_count(h_size, actual_d):
    """Compute parameter count for a 1-hidden-layer FFNN with bias.

    Architecture: input(1) -> hidden(h_size) -> output(actual_d)
    Layer 1: 1 * h_size weights + h_size biases
    Layer 2: h_size * actual_d weights + actual_d biases
    """
    return (1 * h_size + h_size) + (h_size * actual_d + actual_d)


def run_hidden_size(h_size, dataset_name, actual_d, feature_indices, num_classes,
                    args, device):
    """Run a single hidden-size configuration for num_runs seeds.
    Returns aggregated metrics and histories."""
    run_metrics = defaultdict(list)
    histories = []

    for run_idx in range(args.num_runs):
        seed = 42 * (run_idx + 1) + 1234
        set_random_seed(seed)

        model = ObliviousTree(
            d=actual_d,
            feature_indices=feature_indices,
            num_classes=num_classes,
            threshold_type='classical',
            classical_hidden_layers=1,
            classical_hidden_size=h_size,
            use_bias=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha_init=args.alpha_init,
            alpha_final=args.alpha_final,
        )

        data = load_and_prepare_dataset(dataset_name)
        if data is None:
            return None, None
        X_train, X_val, X_test, Y_train, Y_val, Y_test, _, _ = data

        metrics, history = train_and_evaluate_model(
            model, X_train, Y_train, X_val, Y_val, X_test, Y_test, device
        )
        for k, v in metrics.items():
            run_metrics[k].append(v)
        histories.append(history)

    return run_metrics, histories


def main():
    args = parse_args()
    device = torch.device('cpu')
    depth_grid = [int(x) for x in args.d_values.split(',')]
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]

    datasets_to_test = DATASETS if args.datasets == 'all' else \
        [d.strip() for d in args.datasets.split(',')]

    results = []
    run_results = []

    print("Study 2: FFNN with Hidden Layers")
    print(f"Datasets: {len(datasets_to_test)}, Depths: {depth_grid}, "
          f"Hidden sizes: {hidden_sizes}, Runs: {args.num_runs}")

    for dataset_name in datasets_to_test:
        data = load_and_prepare_dataset(dataset_name)
        if data is None:
            print(f"  Skipping {dataset_name}: load failed")
            continue
        X_train, X_val, X_test, Y_train, Y_val, Y_test, num_classes, num_features = data
        actual_depths = compute_actual_depths(num_features, depth_grid)

        for actual_d in actual_depths:
            print(f"\n  {dataset_name} (d={actual_d}, features={num_features}, classes={num_classes})")

            all_histories = {}

            # Sample feature indices once per (dataset, depth) — shared across hidden sizes
            feature_indices = np.random.choice(
                num_features, size=actual_d, replace=False
            ).tolist()

            for h_size in hidden_sizes:
                approach = f'ffnn_1h_{h_size}'

                run_metrics, histories = run_hidden_size(
                    h_size, dataset_name, actual_d, feature_indices,
                    num_classes, args, device
                )
                if run_metrics is None:
                    continue

                all_histories[approach] = histories
                mean_test_acc = np.mean(run_metrics['test_acc'])
                std_test_acc = np.std(run_metrics['test_acc'])

                num_params = compute_param_count(h_size, actual_d)

                results.append({
                    'timestamp': datetime.now().isoformat(),
                    'dataset': dataset_name,
                    'depth': actual_d,
                    'num_features': num_features,
                    'num_classes': num_classes,
                    'num_samples_train': X_train.shape[0],
                    'approach': approach,
                    'hidden_layers': 1,
                    'hidden_size': h_size,
                    'test_acc_mean': mean_test_acc,
                    'test_acc_std': std_test_acc,
                    'test_ce_mean': np.mean(run_metrics['test_ce']),
                    'test_ce_std': np.std(run_metrics['test_ce']),
                    'val_acc_mean': np.mean(run_metrics['val_acc']),
                    'val_acc_std': np.std(run_metrics['val_acc']),
                    'num_params': num_params,
                })

                for run_idx, m_list in enumerate(
                    zip(*[run_metrics[k] for k in ['test_acc', 'test_ce', 'val_acc', 'val_ce']])
                ):
                    seed = 42 * (run_idx + 1) + 1234
                    run_results.append({
                        'dataset': dataset_name,
                        'depth': actual_d,
                        'approach': approach,
                        'hidden_layers': 1,
                        'hidden_size': h_size,
                        'run_idx': run_idx + 1,
                        'seed': seed,
                        'test_acc': m_list[0],
                        'test_ce': m_list[1],
                        'val_acc': m_list[2],
                        'val_ce': m_list[3],
                    })
                    if args.wandb:
                        wandb_log_run(
                            study_name='study2_hidden_layers',
                            config={
                                'dataset': dataset_name, 'depth': actual_d,
                                'approach': approach, 'hidden_size': h_size,
                                'seed': seed, 'epochs': args.epochs,
                                'batch_size': args.batch_size, 'lr': args.lr,
                                'num_params': num_params,
                            },
                            metrics={'test_acc': m_list[0], 'test_ce': m_list[1],
                                     'val_acc': m_list[2], 'val_ce': m_list[3]},
                            history=histories[run_idx],
                        )

                print(f"    {approach}: acc={mean_test_acc:.4f} +/- {std_test_acc:.4f}, "
                      f"params={num_params}")

            # Plot convergence for all hidden sizes at this (dataset, depth)
            plot_dir = os.path.join(args.output_dir, 'plots', dataset_name)
            plot_convergence_compare(all_histories, plot_dir, dataset_name, actual_d)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_csv(results, os.path.join(args.output_dir, 'results.csv'))
    save_results_csv(run_results, os.path.join(args.output_dir, 'results_runs.csv'))

    # Print summary
    print(f"\n{'='*60}")
    print("STUDY 2 SUMMARY")
    print(f"{'='*60}")
    for h_size in hidden_sizes:
        approach = f'ffnn_1h_{h_size}'
        approach_accs = [r['test_acc_mean'] for r in results if r['approach'] == approach]
        if approach_accs:
            print(f"  {approach}: mean acc={np.mean(approach_accs):.4f}")


if __name__ == '__main__':
    main()

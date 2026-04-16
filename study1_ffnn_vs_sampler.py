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
    parser = argparse.ArgumentParser(description='Study 1: FFNN (0-layer) vs Sampler')
    parser.add_argument('--d-values', type=str, default='4,8,12',
                        help='Comma-separated depth values')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument('--output-dir', type=str, default='results/study1')
    parser.add_argument('--alpha-init', type=float, default=1.0)
    parser.add_argument('--alpha-final', type=float, default=20.0)
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    return parser.parse_args()


def run_approach(approach_name, dataset_name, actual_d, feature_indices, num_classes,
                 args, device):
    """Run a single approach for num_runs seeds. Return aggregated metrics and histories."""
    run_metrics = defaultdict(list)
    histories = []

    for run_idx in range(args.num_runs):
        seed = 42 * (run_idx + 1) + 1234
        set_random_seed(seed)

        if approach_name == 'ffnn_0layer_bias':
            model = ObliviousTree(
                d=actual_d, feature_indices=feature_indices, num_classes=num_classes,
                threshold_type='classical', classical_hidden_layers=0,
                classical_hidden_size=0, use_bias=True,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                alpha_init=args.alpha_init, alpha_final=args.alpha_final,
            )
        elif approach_name == 'ffnn_0layer_nobias':
            model = ObliviousTree(
                d=actual_d, feature_indices=feature_indices, num_classes=num_classes,
                threshold_type='classical', classical_hidden_layers=0,
                classical_hidden_size=0, use_bias=False,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                alpha_init=args.alpha_init, alpha_final=args.alpha_final,
            )
        elif approach_name == 'sampler':
            model = ObliviousTree(
                d=actual_d, feature_indices=feature_indices, num_classes=num_classes,
                threshold_type='sampler',
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                alpha_init=args.alpha_init, alpha_final=args.alpha_final,
            )
        else:
            raise ValueError(f"Unknown approach: {approach_name}")

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_grid = [int(x) for x in args.d_values.split(',')]

    datasets_to_test = DATASETS if args.datasets == 'all' else \
        [d.strip() for d in args.datasets.split(',')]

    approaches = ['ffnn_0layer_bias', 'ffnn_0layer_nobias', 'sampler']
    results = []
    run_results = []

    print(f"Study 1: FFNN (0-layer) vs Sampler")
    print(f"Datasets: {len(datasets_to_test)}, Depths: {depth_grid}, Runs: {args.num_runs}")

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
            approach_results = {}

            for approach in approaches:
                feature_indices = np.random.choice(
                    num_features, size=actual_d, replace=False
                ).tolist()

                run_metrics, histories = run_approach(
                    approach, dataset_name, actual_d, feature_indices,
                    num_classes, args, device
                )
                if run_metrics is None:
                    continue

                all_histories[approach] = histories
                mean_test_acc = np.mean(run_metrics['test_acc'])
                std_test_acc = np.std(run_metrics['test_acc'])
                approach_results[approach] = mean_test_acc

                num_params = actual_d  # sampler
                if 'ffnn' in approach:
                    num_params = actual_d * 2 if 'bias' in approach else actual_d

                results.append({
                    'timestamp': datetime.now().isoformat(),
                    'dataset': dataset_name,
                    'depth': actual_d,
                    'num_features': num_features,
                    'num_classes': num_classes,
                    'num_samples_train': X_train.shape[0],
                    'approach': approach,
                    'test_acc_mean': np.mean(run_metrics['test_acc']),
                    'test_acc_std': np.std(run_metrics['test_acc']),
                    'test_ce_mean': np.mean(run_metrics['test_ce']),
                    'test_ce_std': np.std(run_metrics['test_ce']),
                    'val_acc_mean': np.mean(run_metrics['val_acc']),
                    'val_acc_std': np.std(run_metrics['val_acc']),
                    'num_params': num_params,
                })

                for run_idx, m_list in enumerate(zip(*[run_metrics[k] for k in ['test_acc', 'test_ce', 'val_acc', 'val_ce']])):
                    seed = 42 * (run_idx + 1) + 1234
                    run_results.append({
                        'dataset': dataset_name,
                        'depth': actual_d,
                        'approach': approach,
                        'run_idx': run_idx + 1,
                        'seed': seed,
                        'test_acc': m_list[0],
                        'test_ce': m_list[1],
                        'val_acc': m_list[2],
                        'val_ce': m_list[3],
                    })
                    if args.wandb:
                        wandb_log_run(
                            study_name='study1_ffnn_vs_sampler',
                            config={
                                'dataset': dataset_name, 'depth': actual_d,
                                'approach': approach, 'seed': seed,
                                'epochs': args.epochs, 'batch_size': args.batch_size,
                                'lr': args.lr, 'num_params': num_params,
                            },
                            metrics={'test_acc': m_list[0], 'test_ce': m_list[1],
                                     'val_acc': m_list[2], 'val_ce': m_list[3]},
                            history=histories[run_idx],
                        )

                print(f"    {approach}: acc={mean_test_acc:.4f} +/- {std_test_acc:.4f}")

            # Plot convergence
            plot_dir = os.path.join(args.output_dir, 'plots', dataset_name)
            plot_convergence_compare(all_histories, plot_dir, dataset_name, actual_d)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_csv(results, os.path.join(args.output_dir, 'results.csv'))
    save_results_csv(run_results, os.path.join(args.output_dir, 'results_runs.csv'))

    # Print summary
    print(f"\n{'='*60}")
    print("STUDY 1 SUMMARY")
    print(f"{'='*60}")
    for approach in approaches:
        approach_accs = [r['test_acc_mean'] for r in results if r['approach'] == approach]
        if approach_accs:
            print(f"  {approach}: mean acc={np.mean(approach_accs):.4f}")


if __name__ == '__main__':
    main()

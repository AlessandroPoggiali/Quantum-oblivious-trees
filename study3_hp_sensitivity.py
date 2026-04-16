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
    wandb_log_run,
    log_error,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Study 3: Optimizer/LR Sensitivity')
    parser.add_argument('--d-values', type=str, default='4,8,12',
                        help='Comma-separated depth values')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--optimizers', type=str, default='adam,sgd',
                        help='Comma-separated optimizer names: adam,sgd')
    parser.add_argument('--learning-rates', type=str, default='0.001,0.005,0.01,0.05,0.1',
                        help='Comma-separated learning rate values')
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument('--output-dir', type=str, default='results/study3')
    parser.add_argument('--alpha-init', type=float, default=1.0)
    parser.add_argument('--alpha-final', type=float, default=20.0)
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    return parser.parse_args()


def run_config(approach_name, opt_name, lr, dataset_name, actual_d, feature_indices,
               num_classes, args, device):
    """Run a single (approach, optimizer, lr) configuration for num_runs seeds.

    Returns aggregated run_metrics dict or None on failure.
    """
    run_metrics = defaultdict(list)
    histories = []

    for run_idx in range(args.num_runs):
        seed = 42 * (run_idx + 1) + 1234
        set_random_seed(seed)

        if approach_name == 'ffnn_0layer':
            model = ObliviousTree(
                d=actual_d,
                feature_indices=feature_indices,
                num_classes=num_classes,
                device=device,
                threshold_type='classical',
                classical_hidden_layers=0,
                classical_hidden_size=0,
                use_bias=True,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=lr,
                alpha_init=args.alpha_init,
                alpha_final=args.alpha_final,
                optimizer_type=opt_name,
            )
        elif approach_name == 'sampler':
            model = ObliviousTree(
                d=actual_d,
                feature_indices=feature_indices,
                num_classes=num_classes,
                device=device,
                threshold_type='sampler',
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=lr,
                alpha_init=args.alpha_init,
                alpha_final=args.alpha_final,
                optimizer_type=opt_name,
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
    optimizers = [o.strip().lower() for o in args.optimizers.split(',')]
    learning_rates = [float(x) for x in args.learning_rates.split(',')]

    datasets_to_test = DATASETS if args.datasets == 'all' else \
        [d.strip() for d in args.datasets.split(',')]

    approaches = ['ffnn_0layer', 'sampler']
    results = []
    run_results = []

    print("Study 3: Optimizer/LR Sensitivity")
    print(f"Datasets: {len(datasets_to_test)}, Depths: {depth_grid}, "
          f"Approaches: {approaches}")
    print(f"Optimizers: {optimizers}, LRs: {learning_rates}, Runs: {args.num_runs}")

    for dataset_name in datasets_to_test:
        try:
            data = load_and_prepare_dataset(dataset_name)
            if data is None:
                print(f"  Skipping {dataset_name}: load failed")
                continue
            X_train, X_val, X_test, Y_train, Y_val, Y_test, num_classes, num_features = data
            actual_depths = compute_actual_depths(num_features, depth_grid)

            for actual_d in actual_depths:
                print(f"\n  {dataset_name} (d={actual_d}, features={num_features}, classes={num_classes})")

                # Sample feature indices once per (dataset, depth) — shared across all configs
                feature_indices = np.random.choice(
                    num_features, size=actual_d, replace=False
                ).tolist()

                for approach in approaches:
                    for opt_name in optimizers:
                        for lr in learning_rates:
                            run_metrics, histories = run_config(
                                approach, opt_name, lr,
                                dataset_name, actual_d, feature_indices,
                                num_classes, args, device
                            )
                            if run_metrics is None:
                                print(f"    [{approach}/{opt_name}/lr={lr}] FAILED")
                                continue

                            mean_test_acc = np.mean(run_metrics['test_acc'])
                            std_test_acc = np.std(run_metrics['test_acc'])

                            results.append({
                                'timestamp': datetime.now().isoformat(),
                                'dataset': dataset_name,
                                'depth': actual_d,
                                'num_features': num_features,
                                'num_classes': num_classes,
                                'num_samples_train': X_train.shape[0],
                                'approach': approach,
                                'optimizer': opt_name,
                                'learning_rate': lr,
                                'test_acc_mean': mean_test_acc,
                                'test_acc_std': std_test_acc,
                                'test_ce_mean': np.mean(run_metrics['test_ce']),
                                'test_ce_std': np.std(run_metrics['test_ce']),
                                'val_acc_mean': np.mean(run_metrics['val_acc']),
                                'val_acc_std': np.std(run_metrics['val_acc']),
                            })

                            for run_idx, m_list in enumerate(
                                zip(*[run_metrics[k] for k in ['test_acc', 'test_ce', 'val_acc', 'val_ce']])
                            ):
                                seed = 42 * (run_idx + 1) + 1234
                                run_results.append({
                                    'dataset': dataset_name,
                                    'depth': actual_d,
                                    'approach': approach,
                                    'optimizer': opt_name,
                                    'learning_rate': lr,
                                    'run_idx': run_idx + 1,
                                    'seed': seed,
                                    'test_acc': m_list[0],
                                    'test_ce': m_list[1],
                                    'val_acc': m_list[2],
                                    'val_ce': m_list[3],
                                })
                                if args.wandb:
                                    wandb_log_run(
                                        study_name='study3_hp_sensitivity',
                                        config={
                                            'dataset': dataset_name, 'depth': actual_d,
                                            'approach': approach, 'optimizer': opt_name,
                                            'learning_rate': lr, 'seed': seed,
                                            'epochs': args.epochs, 'batch_size': args.batch_size,
                                        },
                                        metrics={'test_acc': m_list[0], 'test_ce': m_list[1],
                                                 'val_acc': m_list[2], 'val_ce': m_list[3]},
                                        history=histories[run_idx],
                                    )

                            print(f"    {approach}/{opt_name}/lr={lr}: "
                                  f"acc={mean_test_acc:.4f} +/- {std_test_acc:.4f}")
        except Exception as e:
            print(f"  ERROR on {dataset_name}: {e}")
            log_error('study3_hp_sensitivity', dataset_name, e)
            continue

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_csv(results, os.path.join(args.output_dir, 'results.csv'))
    save_results_csv(run_results, os.path.join(args.output_dir, 'results_runs.csv'))

    # Summary: best LR per approach x optimizer
    print(f"\n{'='*60}")
    print("STUDY 3 SUMMARY — Best LR per approach x optimizer")
    print(f"{'='*60}")
    for approach in approaches:
        for opt_name in optimizers:
            best_lr = None
            best_acc = -1.0
            for lr in learning_rates:
                subset = [
                    r['test_acc_mean'] for r in results
                    if r['approach'] == approach
                    and r['optimizer'] == opt_name
                    and r['learning_rate'] == lr
                ]
                if subset:
                    mean_acc = np.mean(subset)
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_lr = lr
            if best_lr is not None:
                print(f"  {approach} / {opt_name}: best lr={best_lr} "
                      f"(mean test acc={best_acc:.4f})")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

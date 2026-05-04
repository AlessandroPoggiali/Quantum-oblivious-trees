import os
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch

from gentree_utils import DATASETS
from oblivious_tree import ObliviousTree
from oblivious_tree_with_leaf import ObliviousTreeWithLeaf
from experiment_utils import (
    set_random_seed,
    load_and_prepare_dataset,
    compute_actual_depths,
    train_and_evaluate_model,
    save_results_csv,
    plot_convergence_compare,
    wandb_log_run,
    log_error,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Study: Classical Sampler with and without Leaf Learning'
    )
    parser.add_argument('--d-values', type=str, default='4,8,12',
                        help='Comma-separated depth values')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument('--output-dir', type=str, default='results/sampler_leaf_learning')
    parser.add_argument('--alpha-init', type=float, default=1.0)
    parser.add_argument('--alpha-final', type=float, default=20.0)
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    return parser.parse_args()


def run_sampler_variant(variant_name, dataset_name, actual_d, feature_indices,
                        num_classes, args, device):
    """
    Run sampler with or without leaf learning for num_runs seeds.
    
    Args:
        variant_name: 'with_leaf_learning' or 'without_leaf_learning'
        dataset_name: name of the dataset
        actual_d: actual depth
        feature_indices: indices of features to use
        num_classes: number of classes
        args: parsed arguments
        device: torch device
    
    Returns:
        (aggregated metrics dict, histories list) or (None, None) on failure
    """
    run_metrics = defaultdict(list)
    histories = []

    for run_idx in range(args.num_runs):
        seed = 42 * (run_idx + 1) + 1234
        set_random_seed(seed)

        if variant_name == 'sampler':
            model = ObliviousTree(
                d=actual_d,
                feature_indices=feature_indices,
                num_classes=num_classes,
                device=device,
                threshold_type="sampler",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                alpha_init=args.alpha_init,
                alpha_final=args.alpha_final,
            )
        elif variant_name == 'sampler_with_leaf':
            model = ObliviousTreeWithLeaf(
                d=actual_d,
                feature_indices=feature_indices,
                num_classes=num_classes,
                device=device,
                threshold_type="sampler",
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_grid = [int(x) for x in args.d_values.split(',')]

    datasets_to_test = DATASETS if args.datasets == 'all' else \
        [d.strip() for d in args.datasets.split(',')]

    # Two variants: with and without leaf learning
    variants = [
        'sampler',
        'sampler_with_leaf',
    ]

    results = []
    run_results = []

    print(f"Study: Classical Sampler with/without Leaf Learning")
    print(f"Datasets: {len(datasets_to_test)}, Depths: {depth_grid}, Runs: {args.num_runs}")
    print(f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, alpha_init={args.alpha_init}, alpha_final={args.alpha_final}\n")

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

                all_histories = {}
                variant_results = {}

                # Sample feature indices once per (dataset, depth) — shared across variants
                feature_indices = np.random.choice(
                    num_features, size=actual_d, replace=False
                ).tolist()

                for variant_name in variants:
                    run_metrics, histories = run_sampler_variant(
                        variant_name, dataset_name, actual_d, feature_indices,
                        num_classes, args, device
                    )
                    if run_metrics is None:
                        print(f"    {variant_name}: FAILED")
                        continue

                    all_histories[variant_name] = histories
                    mean_test_acc = np.mean(run_metrics['test_acc'])
                    std_test_acc = np.std(run_metrics['test_acc'])
                    mean_test_ce = np.mean(run_metrics['test_ce'])
                    std_test_ce = np.std(run_metrics['test_ce'])
                    variant_results[variant_name] = mean_test_acc

                    # Sampler has d parameters
                    num_params = actual_d

                    results.append({
                        'timestamp': datetime.now().isoformat(),
                        'dataset': dataset_name,
                        'depth': actual_d,
                        'num_features': num_features,
                        'num_classes': num_classes,
                        'num_samples_train': X_train.shape[0],
                        'variant': variant_name,
                        'test_acc_mean': mean_test_acc,
                        'test_acc_std': std_test_acc,
                        'test_ce_mean': mean_test_ce,
                        'test_ce_std': std_test_ce,
                        'val_acc_mean': np.mean(run_metrics['val_acc']),
                        'val_acc_std': np.std(run_metrics['val_acc']),
                        'val_ce_mean': np.mean(run_metrics['val_ce']),
                        'val_ce_std': np.std(run_metrics['val_ce']),
                        'num_params': num_params,
                    })

                    for run_idx, m_list in enumerate(
                        zip(*[run_metrics[k] for k in ['test_acc', 'test_ce', 'val_acc', 'val_ce']])
                    ):
                        seed = 42 * (run_idx + 1) + 1234
                        run_results.append({
                            'dataset': dataset_name,
                            'depth': actual_d,
                            'variant': variant_name,
                            'run_idx': run_idx + 1,
                            'seed': seed,
                            'test_acc': m_list[0],
                            'test_ce': m_list[1],
                            'val_acc': m_list[2],
                            'val_ce': m_list[3],
                        })
                        if args.wandb:
                            wandb_log_run(
                                study_name='sampler_leaf_learning',
                                config={
                                    'dataset': dataset_name,
                                    'depth': actual_d,
                                    'variant': variant_name,
                                    'seed': seed,
                                    'epochs': args.epochs,
                                    'batch_size': args.batch_size,
                                    'lr': args.lr,
                                    'alpha_init': args.alpha_init,
                                    'alpha_final': args.alpha_final,
                                    'num_params': num_params,
                                },
                                metrics={
                                    'test_acc': m_list[0],
                                    'test_ce': m_list[1],
                                    'val_acc': m_list[2],
                                    'val_ce': m_list[3],
                                },
                                history=histories[run_idx],
                            )

                    print(f"    {variant_name:30s}: acc={mean_test_acc:.4f} ± {std_test_acc:.4f}, "
                          f"ce={mean_test_ce:.4f} ± {std_test_ce:.4f}")

                # Plot convergence comparing the two variants
                plot_dir = os.path.join(args.output_dir, 'plots', dataset_name)
                plot_convergence_compare(all_histories, plot_dir, dataset_name, actual_d)

        except Exception as e:
            print(f"  ERROR on {dataset_name}: {e}")
            log_error('sampler_leaf_learning', dataset_name, e)
            continue

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_csv(results, os.path.join(args.output_dir, 'results.csv'))
    save_results_csv(run_results, os.path.join(args.output_dir, 'results_runs.csv'))

    # Print summary
    print(f"\n{'='*70}")
    print("SAMPLER LEAF LEARNING STUDY SUMMARY")
    print(f"{'='*70}")
    for variant_name in variants:
        variant_accs = [r['test_acc_mean'] for r in results if r['variant'] == variant_name]
        variant_ces = [r['test_ce_mean'] for r in results if r['variant'] == variant_name]
        if variant_accs:
            print(f"\n{variant_name}:")
            print(f"  Mean accuracy: {np.mean(variant_accs):.4f}")
            print(f"  Std accuracy:  {np.std(variant_accs):.4f}")
            print(f"  Mean CE:       {np.mean(variant_ces):.4f}")
            print(f"  Std CE:        {np.std(variant_ces):.4f}")

    # Compute difference
    with_leaf = [r['test_acc_mean'] for r in results if r['variant'] == 'sampler']
    without_leaf = [r['test_acc_mean'] for r in results if r['variant'] == 'sampler_no_leaf']
    if with_leaf and without_leaf:
        mean_diff = np.mean(with_leaf) - np.mean(without_leaf)
        print(f"\nImprovement with leaf learning: {mean_diff:.4f} (positive = with leaf is better)")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()

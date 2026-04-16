import os
import argparse
from datetime import datetime

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss

from gentree_utils import DATASETS
from experiment_utils import (
    set_random_seed,
    load_and_prepare_dataset,
    compute_actual_depths,
    save_results_csv,
    wandb_log_run,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Study 4: sklearn Decision Tree Baseline')
    parser.add_argument('--d-values', type=str, default='4,8,12',
                        help='Comma-separated depth values')
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument('--output-dir', type=str, default='results/study4')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    return parser.parse_args()


def run_dt(dataset_name, actual_d, feature_indices, num_classes, num_runs):
    """Run DecisionTreeClassifier for num_runs seeds.

    Returns (aggregated_metrics dict, per_run list) or (None, None) on failure.
    """
    run_accs_test = []
    run_ces_test = []
    run_accs_val = []
    run_ces_val = []
    per_run = []

    for run_idx in range(num_runs):
        seed = 42 * (run_idx + 1) + 1234
        set_random_seed(seed)

        data = load_and_prepare_dataset(dataset_name)
        if data is None:
            return None, None
        X_train, X_val, X_test, Y_train, Y_val, Y_test, _, _ = data

        X_train_sub = X_train[:, feature_indices]
        X_val_sub = X_val[:, feature_indices]
        X_test_sub = X_test[:, feature_indices]

        clf = DecisionTreeClassifier(max_depth=actual_d, random_state=seed)
        clf.fit(X_train_sub, Y_train)

        # Test metrics
        y_pred_test = clf.predict(X_test_sub)
        y_proba_test = clf.predict_proba(X_test_sub)
        test_acc = accuracy_score(Y_test, y_pred_test)
        test_ce = log_loss(Y_test, y_proba_test, labels=list(range(num_classes)))

        # Val metrics
        y_pred_val = clf.predict(X_val_sub)
        y_proba_val = clf.predict_proba(X_val_sub)
        val_acc = accuracy_score(Y_val, y_pred_val)
        val_ce = log_loss(Y_val, y_proba_val, labels=list(range(num_classes)))

        n_leaves = clf.get_n_leaves()
        tree_depth = clf.get_depth()

        run_accs_test.append(test_acc)
        run_ces_test.append(test_ce)
        run_accs_val.append(val_acc)
        run_ces_val.append(val_ce)

        per_run.append({
            'run_idx': run_idx + 1,
            'seed': seed,
            'test_acc': test_acc,
            'test_ce': test_ce,
            'val_acc': val_acc,
            'val_ce': val_ce,
            'n_leaves': n_leaves,
            'tree_depth': tree_depth,
        })

    agg = {
        'test_acc': run_accs_test,
        'test_ce': run_ces_test,
        'val_acc': run_accs_val,
        'val_ce': run_ces_val,
    }
    return agg, per_run


def main():
    args = parse_args()
    depth_grid = [int(x) for x in args.d_values.split(',')]

    datasets_to_test = DATASETS if args.datasets == 'all' else \
        [d.strip() for d in args.datasets.split(',')]

    results = []
    run_results = []

    print("Study 4: sklearn Decision Tree Baseline")
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

            # Sample feature indices once per (dataset, depth) — same as OBT studies
            feature_indices = np.random.choice(
                num_features, size=actual_d, replace=False
            ).tolist()

            agg, per_run = run_dt(
                dataset_name, actual_d, feature_indices, num_classes, args.num_runs
            )
            if agg is None:
                print(f"    FAILED")
                continue

            mean_test_acc = np.mean(agg['test_acc'])
            std_test_acc = np.std(agg['test_acc'])

            results.append({
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset_name,
                'depth': actual_d,
                'num_features': num_features,
                'num_classes': num_classes,
                'num_samples_train': X_train.shape[0],
                'dt_test_acc_mean': mean_test_acc,
                'dt_test_acc_std': std_test_acc,
                'dt_test_ce_mean': np.mean(agg['test_ce']),
                'dt_test_ce_std': np.std(agg['test_ce']),
                'dt_val_acc_mean': np.mean(agg['val_acc']),
                'dt_val_acc_std': np.std(agg['val_acc']),
            })

            for run_info in per_run:
                run_results.append({
                    'dataset': dataset_name,
                    'depth': actual_d,
                    'num_features': num_features,
                    'num_classes': num_classes,
                    'num_samples_train': X_train.shape[0],
                    'run_idx': run_info['run_idx'],
                    'seed': run_info['seed'],
                    'test_acc': run_info['test_acc'],
                    'test_ce': run_info['test_ce'],
                    'val_acc': run_info['val_acc'],
                    'val_ce': run_info['val_ce'],
                    'n_leaves': run_info['n_leaves'],
                    'tree_depth': run_info['tree_depth'],
                })
                if args.wandb:
                    wandb_log_run(
                        study_name='study4_dt_baseline',
                        config={
                            'dataset': dataset_name, 'depth': actual_d,
                            'approach': 'decision_tree', 'seed': run_info['seed'],
                            'n_leaves': run_info['n_leaves'],
                            'tree_depth': run_info['tree_depth'],
                        },
                        metrics={
                            'test_acc': run_info['test_acc'],
                            'test_ce': run_info['test_ce'],
                            'val_acc': run_info['val_acc'],
                            'val_ce': run_info['val_ce'],
                        },
                    )

            print(f"    DT: acc={mean_test_acc:.4f} +/- {std_test_acc:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_csv(results, os.path.join(args.output_dir, 'results.csv'))
    save_results_csv(run_results, os.path.join(args.output_dir, 'results_runs.csv'))

    # Summary
    print(f"\n{'='*60}")
    print("STUDY 4 SUMMARY")
    print(f"{'='*60}")

    if results:
        all_accs = [r['dt_test_acc_mean'] for r in results]
        print(f"  Overall mean test accuracy: {np.mean(all_accs):.4f}")

        # Best and worst dataset (averaged over depths)
        from collections import defaultdict
        dataset_accs = defaultdict(list)
        for r in results:
            dataset_accs[r['dataset']].append(r['dt_test_acc_mean'])

        dataset_means = {ds: np.mean(accs) for ds, accs in dataset_accs.items()}
        best_ds = max(dataset_means, key=dataset_means.get)
        worst_ds = min(dataset_means, key=dataset_means.get)
        print(f"  Best dataset:  {best_ds} (mean acc={dataset_means[best_ds]:.4f})")
        print(f"  Worst dataset: {worst_ds} (mean acc={dataset_means[worst_ds]:.4f})")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

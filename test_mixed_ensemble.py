import argparse
import csv
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

from gentree_utils import load_data, DATASETS
from oblivious_tree import ObliviousTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=4, help='Tree depth')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--use-bias', type=str, default='y', choices=['n', 'y'], help='Use bias in classical thresholds')
    parser.add_argument('--alpha-init', type=float, default=1.0, help='Initial alpha')
    parser.add_argument('--alpha-final', type=float, default=20.0, help='Final alpha')
    parser.add_argument('--use-ema', type=str, default='n', choices=['n', 'y'], help='Use EMA')
    parser.add_argument('--ema-rho', type=float, default=0.05, help='EMA rho')

    parser.add_argument('--num-classical', type=int, default=1, help='Number of classical trees in ensemble')
    parser.add_argument('--num-quantum', type=int, default=1, help='Number of quantum trees in ensemble')

    parser.add_argument('--q-reps', type=int, default=2, help='Quantum repetitions')
    parser.add_argument('--q-dev', type=str, default='default.qubit', help='Quantum device')
    parser.add_argument('--q-shots', type=int, default=0, help='Quantum shots')
    parser.add_argument('--ansatz', type=str, default='ry', choices=['ry', 'two_param'], help='Quantum ansatz')

    parser.add_argument('--datasets', type=str, default='all', help='Specific datasets (comma-separated) or "all"')
    parser.add_argument('--single-dataset', type=str, default=None, help='Test single dataset (legacy)')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of repeated runs per dataset')

    parser.add_argument('--output-csv', type=str, default='results_ensemble.csv', help='Output CSV file for aggregated results')
    parser.add_argument('--output-runs-csv', type=str, default='results_ensemble_runs.csv', help='Output CSV file for per-run results')

    return parser.parse_args()


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def sample_feature_indices(n_features: int, d: int) -> List[int]:
    actual_d = min(d, n_features)
    return np.random.choice(n_features, size=actual_d, replace=False).tolist()


def train_tree_member(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    num_classes: int,
    feature_indices: List[int],
    use_classical: bool,
    args,
    device: torch.device,
):
    model = ObliviousTree(
        d=len(feature_indices),
        feature_indices=feature_indices,
        device=device,
        alpha_init=args.alpha_init,
        alpha_final=args.alpha_final,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_ema=args.use_ema == 'y',
        ema_rho=args.ema_rho,
        num_classes=num_classes,
        use_classical=use_classical,
        classical_hidden_layers=0,
        classical_hidden_size=0,
        use_bias=args.use_bias == 'y',
        q_reps=args.q_reps,
        q_dev=args.q_dev,
        q_shots=args.q_shots,
        ansatz=args.ansatz,
    )

    _, thresholds, mu_b, history = model.train(
        X_train,
        Y_train,
        X_val=X_val,
        Y_val=Y_val,
        save_every=100,
        ckpt_path=None,
    )

    member = {
        'model': model,
        'thresholds': torch.tensor(thresholds, dtype=torch.float32, device=device),
        'mu_b': torch.tensor(mu_b, dtype=torch.float32, device=device),
        'history': history,
        'type': 'classical' if use_classical else 'quantum',
    }
    return member


def predict_member_proba(member: Dict[str, Any], X: np.ndarray, alpha_final: float, device: torch.device) -> torch.Tensor:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        yhat, _, _ = member['model'].predict_batch(
            X_t,
            member['mu_b'],
            member['thresholds'],
            alpha=alpha_final,
        )
    return yhat


def predict_ensemble_proba(members: List[Dict[str, Any]], X: np.ndarray, alpha_final: float, device: torch.device) -> torch.Tensor:
    probs = [predict_member_proba(member, X, alpha_final, device) for member in members]
    return torch.stack(probs, dim=0).mean(dim=0)


def evaluate_ensemble(
    members: List[Dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    alpha_final: float,
    device: torch.device,
) -> Tuple[float, float]:
    Y_t = torch.tensor(Y, dtype=torch.long, device=device)
    yhat = predict_ensemble_proba(members, X, alpha_final, device)
    ypred = torch.argmax(yhat, dim=1)
    acc = (ypred == Y_t).float().mean().item()
    ce = nn.functional.cross_entropy(yhat, Y_t).item()
    return acc, ce


def train_ensemble(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    num_classes: int,
    args,
    device: torch.device,
    run_seed: int,
) -> List[Dict[str, Any]]:
    members: List[Dict[str, Any]] = []

    total_members = args.num_classical + args.num_quantum
    if total_members <= 0:
        raise ValueError('At least one tree is required in the ensemble (num_classical + num_quantum > 0).')

    member_idx = 0
    for _ in range(args.num_classical):
        member_seed = run_seed + 1000 + member_idx
        set_random_seed(member_seed)
        feature_indices = sample_feature_indices(X_train.shape[1], args.d)
        members.append(
            train_tree_member(
                X_train,
                Y_train,
                X_val,
                Y_val,
                num_classes,
                feature_indices,
                use_classical=True,
                args=args,
                device=device,
            )
        )
        member_idx += 1

    for _ in range(args.num_quantum):
        member_seed = run_seed + 1000 + member_idx
        set_random_seed(member_seed)
        feature_indices = sample_feature_indices(X_train.shape[1], args.d)
        members.append(
            train_tree_member(
                X_train,
                Y_train,
                X_val,
                Y_val,
                num_classes,
                feature_indices,
                use_classical=False,
                args=args,
                device=device,
            )
        )
        member_idx += 1

    return members


def save_csv(rows: List[Dict[str, Any]], file_path: str):
    if not rows:
        print(f'No rows to save for {file_path}')
        return
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} rows to {file_path}')


def test_dataset(dataset_name: str, args, aggregate_rows: List[Dict[str, Any]], run_rows: List[Dict[str, Any]]) -> bool:
    print(f"\n{'=' * 80}")
    print(f'Ensemble test on dataset: {dataset_name}')
    print(f"{'=' * 80}")

    try:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(dataset_name)

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    except Exception as e:
        print(f'Failed to load {dataset_name}: {e}')
        return False

    num_classes = len(np.unique(np.concatenate([Y_train, Y_val, Y_test])))
    device = torch.device('cpu')

    print(
        f'Shape(train/val/test): {X_train.shape}/{X_val.shape}/{X_test.shape}, '
        f'classes={num_classes}, classical={args.num_classical}, quantum={args.num_quantum}'
    )

    run_test_accs = []
    run_test_ces = []
    run_val_accs = []
    run_val_ces = []

    for run_idx in range(args.num_runs):
        run_seed = 42 * (run_idx + 1) + 1234
        set_random_seed(run_seed)

        print(f'  Run {run_idx + 1}/{args.num_runs} (seed={run_seed}) ... ', end='', flush=True)

        try:
            members = train_ensemble(
                X_train,
                Y_train,
                X_val,
                Y_val,
                num_classes,
                args,
                device,
                run_seed,
            )
            val_acc, val_ce = evaluate_ensemble(members, X_val, Y_val, args.alpha_final, device)
            test_acc, test_ce = evaluate_ensemble(members, X_test, Y_test, args.alpha_final, device)
        except Exception as e:
            print(f'FAILED ({e})')
            return False

        run_test_accs.append(test_acc)
        run_test_ces.append(test_ce)
        run_val_accs.append(val_acc)
        run_val_ces.append(val_ce)

        run_rows.append({
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'run_idx': run_idx + 1,
            'seed': run_seed,
            'num_classical': args.num_classical,
            'num_quantum': args.num_quantum,
            'test_acc': test_acc,
            'test_ce': test_ce,
            'val_acc': val_acc,
            'val_ce': val_ce,
        })
        print(f'OK test_acc={test_acc:.4f}, val_acc={val_acc:.4f}')

    aggregate_rows.append({
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'num_runs': args.num_runs,
        'num_classical': args.num_classical,
        'num_quantum': args.num_quantum,
        'mean_test_acc': float(np.mean(run_test_accs)),
        'std_test_acc': float(np.std(run_test_accs)),
        'mean_test_ce': float(np.mean(run_test_ces)),
        'std_test_ce': float(np.std(run_test_ces)),
        'mean_val_acc': float(np.mean(run_val_accs)),
        'std_val_acc': float(np.std(run_val_accs)),
        'mean_val_ce': float(np.mean(run_val_ces)),
        'std_val_ce': float(np.std(run_val_ces)),
        'num_samples_train': X_train.shape[0],
        'num_samples_val': X_val.shape[0],
        'num_samples_test': X_test.shape[0],
        'num_features': X_train.shape[1],
        'num_classes': num_classes,
    })

    print(
        f"Dataset summary -> mean test acc: {np.mean(run_test_accs):.4f} ± {np.std(run_test_accs):.4f}, "
        f"mean val acc: {np.mean(run_val_accs):.4f} ± {np.std(run_val_accs):.4f}"
    )
    return True


def main(args):
    if args.single_dataset:
        datasets_to_test = [args.single_dataset]
    elif args.datasets.lower() == 'all':
        datasets_to_test = DATASETS
    else:
        datasets_to_test = [d.strip() for d in args.datasets.split(',')]

    print('Training ensemble of oblivious trees')
    print(
        f'Configuration: d={args.d}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, '
        f'num_classical={args.num_classical}, num_quantum={args.num_quantum}, runs={args.num_runs}'
    )
    print(f'Datasets: {datasets_to_test}')

    aggregate_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    ok = 0
    for dataset_name in datasets_to_test:
        if test_dataset(dataset_name, args, aggregate_rows, run_rows):
            ok += 1

    print(f"\nCompleted: {ok}/{len(datasets_to_test)} datasets")
    save_csv(aggregate_rows, args.output_csv)
    save_csv(run_rows, args.output_runs_csv)


if __name__ == '__main__':
    args = parse_args()
    main(args)

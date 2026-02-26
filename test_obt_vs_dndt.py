import argparse
import csv
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler

from dndt import nn_decision_tree
from gentree_utils import DATASETS, load_data
from oblivious_tree import ObliviousTree


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs for the classical oblivious tree')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for the classical oblivious tree')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for the classical oblivious tree')
    parser.add_argument('--use-bias', type=str, default='y', choices=['n', 'y'], help='Use bias in classical thresholds')
    parser.add_argument('--alpha-init', type=float, default=1.0, help='Initial alpha for the classical oblivious tree')
    parser.add_argument('--alpha-final', type=float, default=20.0, help='Final alpha for the classical oblivious tree')
    parser.add_argument('--use-ema', type=str, default='n', choices=['n', 'y'], help='Use EMA in classical oblivious tree training')
    parser.add_argument('--ema-rho', type=float, default=0.05, help='EMA rho for classical oblivious tree training')
    parser.add_argument('--num-classical', type=int, default=1, help='Number of classical oblivious trees in the ensemble')
    parser.add_argument('--num-quantum', type=int, default=0, help='Number of quantum oblivious trees in the ensemble')
    parser.add_argument('--obt-features', type=int, default=0, help='Number of sampled features for each obt member (<=0 means all features)')
    parser.add_argument('--q-reps', type=int, default=2, help='Quantum repetitions')
    parser.add_argument('--q-dev', type=str, default='default.qubit', help='Quantum device')
    parser.add_argument('--q-shots', type=int, default=0, help='Quantum shots (0 for analytic mode when supported)')
    parser.add_argument('--ansatz', type=str, default='ry', choices=['ry', 'two_param'], help='Quantum ansatz')

    parser.add_argument('--num-dndt', type=int, default=5, help='Number N of DNDT members in the ensemble')
    parser.add_argument('--dndt-features', type=int, default=4, help='Number F of sampled features for each DNDT member')
    parser.add_argument('--dndt-cut-points', type=int, default=1, help='Number of cut points per selected feature in each DNDT member')
    parser.add_argument('--dndt-temperature', type=float, default=0.1, help='DNDT soft binning temperature')
    parser.add_argument('--dndt-epochs', type=int, default=200, help='DNDT epochs per member')
    parser.add_argument('--dndt-batch-size', type=int, default=32, help='DNDT mini-batch size per member')
    parser.add_argument('--dndt-lr', type=float, default=1e-2, help='DNDT learning rate per member')

    parser.add_argument('--datasets', type=str, default='all', help='Specific datasets (comma-separated) or "all"')
    parser.add_argument('--single-dataset', type=str, default=None, help='Test a single dataset (legacy)')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of repeated runs per dataset')

    parser.add_argument('--output-csv', type=str, default='results_obt.csv', help='Output CSV file for aggregated results')
    parser.add_argument('--output-runs-csv', type=str, default='results_obt_runs.csv', help='Output CSV file for per-run results')

    return parser.parse_args()


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def save_csv(rows: List[Dict[str, Any]], file_path: str):
    if not rows:
        print(f'No rows to save for {file_path}')
        return
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} rows to {file_path}')


def sample_feature_indices(n_features: int, sampled_features: int) -> List[int]:
    if sampled_features <= 0:
        return list(range(n_features))
    actual = min(n_features, sampled_features)
    if actual == n_features:
        return list(range(n_features))
    return np.random.choice(n_features, size=actual, replace=False).tolist()


def train_oblivious_member(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    num_classes: int,
    feature_indices: List[int],
    use_classical: bool,
    args,
    device: torch.device,
) -> Dict[str, Any]:
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

    return {
        'model': model,
        'type': 'classical' if use_classical else 'quantum',
        'feature_indices': feature_indices,
        'thresholds': torch.tensor(thresholds, dtype=torch.float32, device=device),
        'mu_b': torch.tensor(mu_b, dtype=torch.float32, device=device),
        'history': history,
    }


def predict_oblivious_member_proba(member: Dict[str, Any], X: np.ndarray, alpha_final: float, device: torch.device) -> torch.Tensor:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        yhat, _, _ = member['model'].predict_batch(X_t, member['mu_b'], member['thresholds'], alpha=alpha_final)
    return yhat


def evaluate_oblivious_ensemble(
    members: List[Dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    alpha_final: float,
    device: torch.device,
) -> Tuple[float, float]:
    Y_t = torch.tensor(Y, dtype=torch.long, device=device)
    probs = [predict_oblivious_member_proba(member, X, alpha_final, device) for member in members]
    yhat = torch.stack(probs, dim=0).mean(dim=0)

    ypred = torch.argmax(yhat, dim=1)
    acc = (ypred == Y_t).float().mean().item()
    ce = nn.functional.cross_entropy(yhat, Y_t).item()
    return acc, ce


def train_oblivious_ensemble(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    num_classes: int,
    args,
    device: torch.device,
    run_seed: int,
) -> List[Dict[str, Any]]:
    total_members = args.num_classical + args.num_quantum
    if total_members <= 0:
        raise ValueError('At least one oblivious tree is required (num_classical + num_quantum > 0)')

    ensemble = []
    n_features = X_train.shape[1]
    member_idx = 0

    for _ in range(args.num_classical):
        set_random_seed(run_seed + 500 + member_idx)
        feature_indices = sample_feature_indices(n_features, args.obt_features)
        member = train_oblivious_member(
            X_train,
            Y_train,
            X_val,
            Y_val,
            num_classes,
            feature_indices,
            True,
            args,
            device,
        )
        ensemble.append(member)
        member_idx += 1

    for _ in range(args.num_quantum):
        set_random_seed(run_seed + 500 + member_idx)
        feature_indices = sample_feature_indices(n_features, args.obt_features)
        member = train_oblivious_member(
            X_train,
            Y_train,
            X_val,
            Y_val,
            num_classes,
            feature_indices,
            False,
            args,
            device,
        )
        ensemble.append(member)
        member_idx += 1

    return ensemble


def train_dndt_member(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    num_classes: int,
    feature_indices: List[int],
    args,
    device: torch.device,
) -> Dict[str, Any]:
    X_selected = X_train[:, feature_indices]

    train_ds = TensorDataset(
        torch.tensor(X_selected, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=args.dndt_batch_size, shuffle=True)

    num_cut = [args.dndt_cut_points for _ in range(len(feature_indices))]
    num_leaf = int(np.prod(np.array(num_cut) + 1))

    cut_points_list = [
        torch.rand([cuts], requires_grad=True, device=device)
        for cuts in num_cut
    ]
    leaf_score = torch.rand([num_leaf, num_classes], requires_grad=True, device=device)

    optimizer = torch.optim.Adam(cut_points_list + [leaf_score], lr=args.dndt_lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(args.dndt_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = nn_decision_tree(x_batch, cut_points_list, leaf_score, args.dndt_temperature)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    return {
        'feature_indices': feature_indices,
        'cut_points_list': [cp.detach().clone() for cp in cut_points_list],
        'leaf_score': leaf_score.detach().clone(),
    }


def predict_dndt_member_proba(member: Dict[str, Any], X: np.ndarray, temperature: float, device: torch.device) -> torch.Tensor:
    X_selected = torch.tensor(X[:, member['feature_indices']], dtype=torch.float32, device=device)
    with torch.no_grad():
        yhat = nn_decision_tree(X_selected, member['cut_points_list'], member['leaf_score'], temperature)
    return yhat


def evaluate_dndt_ensemble(
    ensemble: List[Dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    temperature: float,
    device: torch.device,
) -> Tuple[float, float]:
    Y_t = torch.tensor(Y, dtype=torch.long, device=device)

    probs = [predict_dndt_member_proba(member, X, temperature, device) for member in ensemble]
    yhat = torch.stack(probs, dim=0).mean(dim=0)

    ypred = torch.argmax(yhat, dim=1)
    acc = (ypred == Y_t).float().mean().item()
    ce = nn.functional.cross_entropy(yhat, Y_t).item()
    return acc, ce


def train_dndt_ensemble(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    num_classes: int,
    args,
    device: torch.device,
    run_seed: int,
) -> List[Dict[str, Any]]:
    if args.num_dndt <= 0:
        raise ValueError('num_dndt must be > 0')

    ensemble = []
    n_features = X_train.shape[1]

    for member_idx in range(args.num_dndt):
        #print(f'    Training DNDT member {member_idx + 1}/{args.num_dndt} ... ', end='', flush=True)
        set_random_seed(run_seed + 1000 + member_idx)
        feature_indices = sample_feature_indices(n_features, args.dndt_features)
        member = train_dndt_member(
            X_train,
            Y_train,
            num_classes,
            feature_indices,
            args,
            device,
        )
        ensemble.append(member)

    return ensemble


def test_dataset(dataset_name: str, args, aggregate_rows: List[Dict[str, Any]], run_rows: List[Dict[str, Any]]) -> bool:
    print(f"\n{'=' * 80}")
    print(f'Oblivious-ensemble vs DNDT test on dataset: {dataset_name}')
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
    n_features = X_train.shape[1]
    device = torch.device('cpu')

    print(
        f'Shape(train/val/test): {X_train.shape}/{X_val.shape}/{X_test.shape}, '
        f'classes={num_classes}, total_features={n_features}, num_classical={args.num_classical}, num_quantum={args.num_quantum}, '
        f'obt_features={len(sample_feature_indices(n_features, args.obt_features))}, '
        f'num_dndt={args.num_dndt}, dndt_features={len(sample_feature_indices(n_features, args.dndt_features))}'
    )

    oblivious_test_accs, oblivious_test_ces = [], []
    oblivious_val_accs, oblivious_val_ces = [], []

    dndt_test_accs, dndt_test_ces = [], []
    dndt_val_accs, dndt_val_ces = [], []

    for run_idx in range(args.num_runs):
        run_seed = 42 * (run_idx + 1) + 1234
        set_random_seed(run_seed)

        print(f'  Run {run_idx + 1}/{args.num_runs} (seed={run_seed}) ... ', end='', flush=True)

        try:
            oblivious_ensemble = train_oblivious_ensemble(
                X_train,
                Y_train,
                X_val,
                Y_val,
                num_classes,
                args,
                device,
                run_seed,
            )
            oblivious_val_acc, oblivious_val_ce = evaluate_oblivious_ensemble(
                oblivious_ensemble,
                X_val,
                Y_val,
                args.alpha_final,
                device,
            )
            oblivious_test_acc, oblivious_test_ce = evaluate_oblivious_ensemble(
                oblivious_ensemble,
                X_test,
                Y_test,
                args.alpha_final,
                device,
            )

            dndt_ensemble = train_dndt_ensemble(
                X_train,
                Y_train,
                num_classes,
                args,
                device,
                run_seed,
            )
            dndt_val_acc, dndt_val_ce = evaluate_dndt_ensemble(
                dndt_ensemble,
                X_val,
                Y_val,
                args.dndt_temperature,
                device,
            )
            dndt_test_acc, dndt_test_ce = evaluate_dndt_ensemble(
                dndt_ensemble,
                X_test,
                Y_test,
                args.dndt_temperature,
                device,
            )
        except Exception as e:
            print(f'FAILED ({e})')
            return False

        oblivious_test_accs.append(oblivious_test_acc)
        oblivious_test_ces.append(oblivious_test_ce)
        oblivious_val_accs.append(oblivious_val_acc)
        oblivious_val_ces.append(oblivious_val_ce)

        dndt_test_accs.append(dndt_test_acc)
        dndt_test_ces.append(dndt_test_ce)
        dndt_val_accs.append(dndt_val_acc)
        dndt_val_ces.append(dndt_val_ce)

        run_rows.append({
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'run_idx': run_idx + 1,
            'seed': run_seed,
            'num_features_total': n_features,
            'num_classical': args.num_classical,
            'num_quantum': args.num_quantum,
            'obt_features': len(sample_feature_indices(n_features, args.obt_features)),
            'num_dndt': args.num_dndt,
            'dndt_features': len(sample_feature_indices(n_features, args.dndt_features)),
            'oblivious_test_acc': oblivious_test_acc,
            'oblivious_test_ce': oblivious_test_ce,
            'oblivious_val_acc': oblivious_val_acc,
            'oblivious_val_ce': oblivious_val_ce,
            'dndt_test_acc': dndt_test_acc,
            'dndt_test_ce': dndt_test_ce,
            'dndt_val_acc': dndt_val_acc,
            'dndt_val_ce': dndt_val_ce,
            'delta_test_acc_dndt_minus_oblivious': dndt_test_acc - oblivious_test_acc,
            'delta_val_acc_dndt_minus_oblivious': dndt_val_acc - oblivious_val_acc,
        })

        print(
            'OK '
            f'oblivious_test_acc={oblivious_test_acc:.4f}, '
            f'dndt_test_acc={dndt_test_acc:.4f}'
        )

    aggregate_rows.append({
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'num_runs': args.num_runs,
        'num_features_total': n_features,
        'num_classical': args.num_classical,
        'num_quantum': args.num_quantum,
        'num_obt': args.num_classical + args.num_quantum,
        'obt_features': len(sample_feature_indices(n_features, args.obt_features)),
        'num_dndt': args.num_dndt,
        'dndt_features': len(sample_feature_indices(n_features, args.dndt_features)),

        'mean_oblivious_test_acc': float(np.mean(oblivious_test_accs)),
        'std_oblivious_test_acc': float(np.std(oblivious_test_accs)),
        'mean_oblivious_test_ce': float(np.mean(oblivious_test_ces)),
        'std_oblivious_test_ce': float(np.std(oblivious_test_ces)),
        'mean_oblivious_val_acc': float(np.mean(oblivious_val_accs)),
        'std_oblivious_val_acc': float(np.std(oblivious_val_accs)),
        'mean_oblivious_val_ce': float(np.mean(oblivious_val_ces)),
        'std_oblivious_val_ce': float(np.std(oblivious_val_ces)),

        'mean_dndt_test_acc': float(np.mean(dndt_test_accs)),
        'std_dndt_test_acc': float(np.std(dndt_test_accs)),
        'mean_dndt_test_ce': float(np.mean(dndt_test_ces)),
        'std_dndt_test_ce': float(np.std(dndt_test_ces)),
        'mean_dndt_val_acc': float(np.mean(dndt_val_accs)),
        'std_dndt_val_acc': float(np.std(dndt_val_accs)),
        'mean_dndt_val_ce': float(np.mean(dndt_val_ces)),
        'std_dndt_val_ce': float(np.std(dndt_val_ces)),

        'mean_delta_test_acc_dndt_minus_oblivious': float(np.mean(np.array(dndt_test_accs) - np.array(oblivious_test_accs))),
        'mean_delta_val_acc_dndt_minus_oblivious': float(np.mean(np.array(dndt_val_accs) - np.array(oblivious_val_accs))),

        'num_samples_train': X_train.shape[0],
        'num_samples_val': X_val.shape[0],
        'num_samples_test': X_test.shape[0],
        'num_classes': num_classes,
    })

    print(
        f"Dataset summary -> oblivious mean test acc: {np.mean(oblivious_test_accs):.4f} ± {np.std(oblivious_test_accs):.4f}, "
        f"dndt mean test acc: {np.mean(dndt_test_accs):.4f} ± {np.std(dndt_test_accs):.4f}"
    )
    return True


def main(args):
    if args.single_dataset:
        datasets_to_test = [args.single_dataset]
    elif args.datasets.lower() == 'all':
        datasets_to_test = DATASETS
    else:
        datasets_to_test = [dataset.strip() for dataset in args.datasets.split(',')]

    print('Comparing: oblivious ensemble (classical + quantum) vs DNDT ensemble (feature sampled)')
    print(
        f'Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, '
        f'num_classical={args.num_classical}, num_quantum={args.num_quantum}, '
        f'obt_features={args.obt_features}, '
        f'q_reps={args.q_reps}, q_dev={args.q_dev}, q_shots={args.q_shots}, ansatz={args.ansatz}, '
        f'num_dndt={args.num_dndt}, dndt_features={args.dndt_features}, '
        f'dndt_cut_points={args.dndt_cut_points}, dndt_epochs={args.dndt_epochs}, '
        f'dndt_batch_size={args.dndt_batch_size}, dndt_lr={args.dndt_lr}, runs={args.num_runs}'
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
    main(parse_args())

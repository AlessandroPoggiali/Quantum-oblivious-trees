import os
import argparse
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, make_moons, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import DecisionBoundaryDisplay
import graphviz
from gentree_utils import load_data
from oblivious_tree import ObliviousTree, extract_rules

# ------------------------ Toy dataset ------------------------

def make_toy_dataset(N: int = 2000, F: int = 6, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.rand(N, F).astype(np.float32)
    y = np.zeros(N, dtype=np.int64)
    y[(X[:, 0] > 0.4) & (X[:, 1] > 0.6)] = 1
    y[(X[:, 0] > 0.6) & (X[:, 1] > 0.4)] = 2
    return X, y

def load_iris_dataset():
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

def load_wine_dataset():
    data = load_wine()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y 

def load_noisymoons_dataset(N: int = 2000, noise: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=N, noise=noise, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

def load_breast_cancer_dataset():
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y



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


    plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        plt.plot(history['epoch'], history['val_acc'], '--', label='Val Acc')
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
    parser.add_argument('--d', type=int, default=4)
    parser.add_argument('--N', type=int, default=2000)
    parser.add_argument('--F', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--alpha-init', type=float, default=1.0)
    parser.add_argument('--alpha-final', type=float, default=20.0)
    parser.add_argument('--use-ema', type=str, default='n', choices=['n','y'])
    parser.add_argument('--ema-rho', type=float, default=0.05)
    parser.add_argument('--q-reps', type=int, default=2)
    parser.add_argument('--q-dev', type=str, default='default.qubit')
    parser.add_argument('--q-shots', type=int, default=0)
    parser.add_argument('--ansatz', type=str, default='ry', choices=['ry','two_param'])
    parser.add_argument('--use-classical', type=str, default='y', choices=['n','y'])
    parser.add_argument('--classical-hidden-layers', type=int, default=0)
    parser.add_argument('--classical-hidden-size', type=int, default=32)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--ckpt', type=str, default=None)
    return parser.parse_args()

def main(args):

    print("Classical FFNN for thresholds" if args.use_classical=='y' else "Quantum circuit for thresholds")
    #X, Y = make_toy_dataset(N=args.N, F=args.F)
    X, Y = load_wine_dataset()
    #X, Y = load_noisymoons_dataset(N=args.N, noise=0.1, seed=42)
    #X, Y = load_breast_cancer_dataset()
    #X, Y = load_iris_dataset()
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data("iris")
    print("Dimensions:", X_train.shape, Y_train.shape)

    num_classes = len(np.unique(np.concatenate([Y_train, Y_val, Y_test])))

    device = torch.device('cpu')
    feature_indices = list(range(args.d)) # alternatively use a feature selection method
    model = ObliviousTree(d=args.d,
                             feature_indices=feature_indices,
                             device=device,
                             alpha_init=args.alpha_init,
                             alpha_final=args.alpha_final,
                             epochs=args.epochs,
                             batch_size=args.batch_size,
                             lr=args.lr,
                             use_ema=True if args.use_ema=='y' else False,
                             ema_rho=args.ema_rho,
                             num_classes=num_classes,
                             q_reps=args.q_reps,
                             q_dev=args.q_dev,
                             q_shots=args.q_shots,
                             ansatz=args.ansatz,
                             use_classical=True if args.use_classical=='y' else False,
                             classical_hidden_size=args.classical_hidden_size,
                             classical_hidden_layers=args.classical_hidden_layers)
    print("Number of model parameters:", sum(p.numel() for p in model.threshold_module.parameters() if p.requires_grad))
    if args.ckpt and os.path.exists(args.ckpt):
        model.load_checkpoint(args.ckpt)

    theta, thresholds, mu_b, history = model.train(X_train, Y_train, X_val=X_val, Y_val=Y_val, save_every=args.save_every, ckpt_path=args.ckpt)

    if args.ckpt is not None:
        model.save_checkpoint(args.ckpt.rstrip('.pt')+'_final.pt', extras={'final': True})
    plot_history(history, 'train_val_loss.png', 'train_val_accuracy.png')
    print("\nTraining complete.")
    if theta is not None:
        print("Final quantum parameters (theta):", theta)
    else:
        print("Classical FFNN used for thresholds.")
    print("Final thresholds:", thresholds)
    print("Final leaf probabilities (mu_b):", mu_b)

    # Evaluate on test set
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.long, device=device)
    with torch.no_grad():      
        yhat_test, y_pred_test, _ = model.predict_batch(X_test_t, torch.from_numpy(mu_b).float(), model.threshold_module().float(), alpha=model.alpha_final)
        test_acc = (y_pred_test == Y_test_t).float().mean().item()
        test_ce = nn.functional.cross_entropy(yhat_test, Y_test_t).item()
        print(f"Test CE: {test_ce:.6f} | Test Accuracy: {test_acc:.4f}")


    rules = extract_rules(model, thresholds, mu_b)
    
    '''
    for rule_str, pred_class, pred_value in rules:
        print(f"IF {rule_str} THEN class={pred_class} (predicted_value={pred_value:.3f})")
    '''
    
    # Plot the decision tree derived from the extracted rules

    if args.d > 5:
        print("Decision tree visualization skipped (depth > 5).")
        return
    # Ensure thresholds and mu_b are numpy arrays
    _th = np.asarray(thresholds).flatten()
    _mu = np.asarray(mu_b)
    D = len(_th)
    L = 2 ** D

    # Build DOT for a full binary tree of depth D (split F0..F{D-1})
    def build_dot():
        lines = ["digraph DecisionTree {", "node [shape=box, style=\"rounded,filled\", fillcolor=white];"]
        def add_node(prefix, depth):
            nid = "n" + (prefix if prefix != "" else "r")
            if depth < D:
                lbl = f"F{depth} <= {_th[depth]:.3f}"
                lines.append(f'{nid} [label="{lbl}", fillcolor=lightblue];')
                left = prefix + "0"
                right = prefix + "1"
                ln = "n" + (left if left != "" else "r")
                rn = "n" + (right if right != "" else "r")
                lines.append(f"{nid} -> {ln};")
                lines.append(f"{nid} -> {rn};")
                add_node(left, depth + 1)
                add_node(right, depth + 1)
            else:
                leaf_idx = int(prefix, 2) if prefix != "" else 0
                probs = _mu[leaf_idx]
                cls = np.argmax(probs)
                prob_str = ", ".join([f"{p:.3f}" for p in probs])
                lines.append(f'{nid} [label="Leaf {leaf_idx}\\nclass={cls}\\nprobs=[{prob_str}]", fillcolor=lightgreen];')
        add_node("", 0)
        lines.append("}")
        return "\n".join(lines)

    dot_str = build_dot()

    out_dot = "decision_tree.dot"
    out_png = "decision_tree.png"

   
    src = graphviz.Source(dot_str)
    src.render(filename=out_dot.rstrip(".dot"), format="png", cleanup=True)
    print(f"Saved decision tree to {out_png}")


    #plot_decision_boundary(model, X_test, y_pred_test.detach().numpy(), torch.from_numpy(mu_b).float(), model.threshold_module().float(), model.alpha_final)

if __name__ == '__main__':
    args = parse_args()
    main(args)
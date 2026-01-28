# adapted from https://github.com/wOOL/DNDT
import numpy as np
import torch
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from gentree_utils import load_data


# -----------------------------
# DNDT Core
# -----------------------------

def torch_kron_prod(a, b):
    res = torch.einsum('ij,ik->ijk', a, b)
    return res.reshape(a.shape[0], -1)


def torch_bin(x, cut_points, temperature=0.1):
    D = cut_points.shape[0]
    W = torch.linspace(1.0, D + 1.0, D + 1, device=x.device).view(1, -1)
    cut_points, _ = torch.sort(cut_points)

    b = torch.cumsum(
        torch.cat([torch.zeros(1, device=x.device), -cut_points]), dim=0
    )

    h = (x @ W + b) / temperature
    return torch.softmax(h, dim=-1)


def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
    leaf = reduce(
        torch_kron_prod,
        (
            torch_bin(x[:, i:i+1], cut_points, temperature)
            for i, cut_points in enumerate(cut_points_list)
        )
    )
    return leaf @ leaf_score


# -----------------------------
# Dataset
# -----------------------------

def prepare_dataloaders(
    X_train, X_val, X_test, Y_train, Y_val, Y_test,
    batch_size=64
):
    

    '''
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    '''

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# -----------------------------
# Training
# -----------------------------

def train_dndt_minibatch(
    X_train, X_val, X_test, Y_train, Y_val, Y_test,
    num_cut_per_dim=1,
    batch_size=64,
    epochs=500,
    lr=1e-2,
    temperature=0.1,
    device="cpu"
):
    num_class = len(np.unique(np.concatenate([Y_train, Y_val, Y_test])))
    d = X_train.shape[1]

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train, X_val, X_test, Y_train, Y_val, Y_test, batch_size=batch_size
    )

    # Cuts
    num_cut = [num_cut_per_dim for _ in range(d)]
    num_leaf = np.prod(np.array(num_cut) + 1)

    cut_points_list = [
        torch.rand([i], requires_grad=True, device=device)
        for i in num_cut
    ]
    leaf_score = torch.rand(
        [num_leaf, num_class], requires_grad=True, device=device
    )

    params = cut_points_list + [leaf_score]
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    print("Parameters:", sum(p.numel() for p in params))
    print("Leaves:", num_leaf)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        # ---- Train ----
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = nn_decision_tree(
                x_batch, cut_points_list, leaf_score, temperature
            )
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # ---- Validation ----
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = nn_decision_tree(
                    x_batch, cut_points_list, leaf_score, temperature
                )
                val_loss += loss_fn(y_pred, y_batch).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_losses[-1]:.4f} | "
                f"val_loss={val_loss:.4f}"
            )

    # -----------------------------
    # Test accuracy
    # -----------------------------
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = nn_decision_tree(
                x_batch, cut_points_list, leaf_score, temperature
            )
            preds = torch.argmax(y_pred, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    test_acc = correct / total
    print("\nFinal Test Accuracy:", test_acc)

    # -----------------------------
    # Plot losses
    # -----------------------------
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("DNDT (Mini-Batch Training)")
    plt.show()

    return {
        "cut_points": cut_points_list,
        "leaf_score": leaf_score,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "test_accuracy": test_acc
    }

# -----------------------------
if __name__ == "__main__":
    #dataset = load_wine()
    # X = dataset.data
    # y = dataset.target
    #X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    #X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data("glass")
    print("Dimensions:", X_train.shape, Y_train.shape)
    results = train_dndt_minibatch(
        X_train, X_val, X_test, Y_train, Y_val, Y_test,
        num_cut_per_dim=1,
        batch_size=32,
        epochs=500,
        lr=0.01,
        temperature=0.1
    )

    print("Training complete.")


import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from quantum_model import QuantumThresholds
from classical_model import ClassicalThresholds


def bits_matrix(d: int, device: Optional[torch.device] = None) -> torch.Tensor:
    rows = []
    for i in range(2 ** d):
        b = [(i >> (d - 1 - j)) & 1 for j in range(d)]
        rows.append(b)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def log_responsibilities_from_probs(p: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    logp = torch.log(p + eps)
    log1mp = torch.log(1 - p + eps)
    B = p.shape[0]
    L = bits.shape[0]
    p_exp = logp.unsqueeze(1).expand(B, L, -1)
    q_exp = log1mp.unsqueeze(1).expand(B, L, -1)
    bits_exp = bits.unsqueeze(0).expand(B, L, -1)
    logP = bits_exp * p_exp + (1 - bits_exp) * q_exp
    return logP.sum(dim=2)


def softmax_normalize_log_probs(logP: torch.Tensor) -> torch.Tensor:
    mx, _ = torch.max(logP, dim=1, keepdim=True)
    stabilized = logP - mx
    P = torch.exp(stabilized)
    den = torch.sum(P, dim=1, keepdim=True)
    return P / (den + 1e-12)


def extract_rules(model, thresholds, mu, threshold_classification: float = 0.5) -> list:
    """Extract human-readable decision rules from a trained oblivious tree."""

    bits = model.bits.cpu().numpy()
    feature_names = [f"F{i}" for i in model.feature_indices]

    rules = []
    for leaf_idx, bitrow in enumerate(bits):
        conds = []
        for i, bit in enumerate(bitrow):
            feature = feature_names[i]
            thresh = thresholds[i]
            if bit == 0:
                conds.append(f"{feature} <= {thresh:.3f}")
            else:
                conds.append(f"{feature} > {thresh:.3f}")
        rule_str = " AND ".join(conds)
        pred_probs = mu[leaf_idx]
        pred_class = int(torch.argmax(torch.tensor(pred_probs)))
        rules.append((rule_str, pred_class, pred_probs))

    return rules


class ObliviousTree:
    def __init__(
        self,
        d: int,
        feature_indices: list,
        device: torch.device = torch.device('cpu'),
        alpha_init: float = 1.0,
        alpha_final: float = 20.0,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-2,
        eps: float = 1e-8,
        num_classes: int = 2,
        q_reps: int = 2,
        q_dev: str = 'default.qubit',
        q_shots: Optional[int] = None,
        ansatz: str = 'ry',
        use_classical: bool = False,
        classical_hidden_size: int = 32,
        classical_hidden_layers: int = 1,
        use_bias: bool = True,
        threshold_type: Optional[str] = None,
        optimizer_type: str = 'adam',
    ):
        self.d = d
        self.feature_indices = feature_indices
        self.device = device
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.num_classes = num_classes
        self.ansatz = ansatz
        self.use_classical = use_classical
        self.threshold_type = threshold_type
        self.optimizer_type = optimizer_type

        if threshold_type is not None:
            if threshold_type == 'sampler':
                from sampler_model import SamplerThresholds
                self.threshold_module = SamplerThresholds(d=d).to(device)
            elif threshold_type == 'classical':
                self.threshold_module = ClassicalThresholds(
                    d=d,
                    hidden_layers=classical_hidden_layers,
                    hidden_size=classical_hidden_size,
                    use_bias=use_bias,
                ).to(device)
            elif threshold_type == 'quantum':
                self.threshold_module = QuantumThresholds(
                    d=d, reps=q_reps, dev_name=q_dev, shots=q_shots, ansatz=ansatz
                ).to(device)
            else:
                raise ValueError(f"Unknown threshold_type: {threshold_type}")
        elif use_classical:
            self.threshold_module = ClassicalThresholds(
                d=d,
                hidden_layers=classical_hidden_layers,
                hidden_size=classical_hidden_size,
                use_bias=use_bias,
            ).to(device)
        else:
            self.threshold_module = QuantumThresholds(
                d=d, reps=q_reps, dev_name=q_dev, shots=q_shots, ansatz=ansatz
            ).to(device)

        self.bits = bits_matrix(self.d, device=device)
        self.L = 2 ** d
        self.leaf_logits = nn.Parameter(
            torch.zeros(self.L, self.num_classes, device=device, dtype=torch.float32)
        )

        parameters = list(self.threshold_module.parameters()) + [self.leaf_logits]
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(parameters, lr=self.lr)
        else:
            self.optimizer = optim.Adam(parameters, lr=self.lr)

    def sigmoid_probs(self, X_batch: torch.Tensor, thresholds: torch.Tensor, alpha: float) -> torch.Tensor:
        X_selected = X_batch[:, self.feature_indices]
        return torch.sigmoid(alpha * (X_selected - thresholds.unsqueeze(0)))

    def compute_responsibilities(self, X, thresholds, alpha):
        p = self.sigmoid_probs(X, thresholds, alpha)
        logP = log_responsibilities_from_probs(p, self.bits)
        return softmax_normalize_log_probs(logP)

    def get_leaf_probs(self) -> torch.Tensor:
        return torch.softmax(self.leaf_logits, dim=1)

    def predict_batch(
        self,
        X_batch: torch.Tensor,
        mu_b: torch.Tensor,
        thresholds: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P = self.compute_responsibilities(X_batch, thresholds, alpha)
        yhat = torch.matmul(P, mu_b)
        pred_class = torch.argmax(yhat, dim=1)
        return yhat, pred_class, P

    def save_checkpoint(self, path: str, extras: Optional[Dict[str, Any]] = None):
        if self.threshold_type == 'sampler':
            threshold_state = {'raw_thresholds': self.threshold_module.raw_thresholds.detach().cpu()}
        elif self.use_classical or self.threshold_type == 'classical':
            threshold_state = {'net_state': self.threshold_module.net.state_dict()}
        else:
            threshold_state = {'theta': self.threshold_module.theta.detach().cpu()}

        state = {
            **threshold_state,
            'leaf_logits': self.leaf_logits.detach().cpu(),
            'optimizer': self.optimizer.state_dict(),
        }
        if extras is not None:
            state.update(extras)
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if self.threshold_type == 'sampler':
            with torch.no_grad():
                self.threshold_module.raw_thresholds.copy_(ckpt['raw_thresholds'].to(torch.float32))
        elif self.use_classical or self.threshold_type == 'classical':
            self.threshold_module.net.load_state_dict(ckpt['net_state'])
        else:
            with torch.no_grad():
                self.threshold_module.theta.copy_(ckpt['theta'].to(torch.float32))

        with torch.no_grad():
            self.leaf_logits.copy_(ckpt['leaf_logits'].to(torch.float32))
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Loaded checkpoint from {path}")

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray = None,
        Y_val: np.ndarray = None,
        save_every: int = 5,
        ckpt_path: Optional[str] = None,
    ):
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y_train, dtype=torch.long, device=self.device)

        if X_val is not None and Y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            Y_val_t = torch.tensor(Y_val, dtype=torch.long, device=self.device)

        N = X_train.shape[0]
        indices = np.arange(N)
        total_steps = int(math.ceil(N / self.batch_size)) * self.epochs
        step = 0
        history = {'epoch': [], 'train_bce': [], 'train_acc': []}
        if X_val is not None:
            history.update({'val_bce': [], 'val_acc': []})

        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = indices[start:end]
                Xb = X_t[batch_idx]
                Yb = Y_t[batch_idx]

                thresholds = self.threshold_module().float()
                alpha = self.alpha_init + (self.alpha_final - self.alpha_init) * (
                    step / max(1, total_steps - 1)
                )
                step += 1

                P = self.compute_responsibilities(Xb, thresholds, alpha)
                mu_b = self.get_leaf_probs().float()
                yhat = torch.matmul(P, mu_b)
                loss = nn.functional.cross_entropy(yhat, Yb)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.threshold_module.parameters()) + [self.leaf_logits],
                    max_norm=5.0,
                )
                self.optimizer.step()

            with torch.no_grad():
                thresholds = self.threshold_module().float()
                mu_eval = self.get_leaf_probs().float()

                yhat_train, y_pred_train, _ = self.predict_batch(
                    X_t, mu_eval, thresholds, alpha=self.alpha_final
                )
                train_ce = nn.functional.cross_entropy(yhat_train, Y_t).item()
                train_acc = (y_pred_train == Y_t).float().mean().item()

                history['epoch'].append(epoch + 1)
                history['train_bce'].append(train_ce)
                history['train_acc'].append(train_acc)

                if X_val is not None and Y_val is not None:
                    yhat_val, y_pred_val, _ = self.predict_batch(
                        X_val_t, mu_eval, thresholds, alpha=self.alpha_final
                    )
                    val_ce = nn.functional.cross_entropy(yhat_val, Y_val_t).item()
                    val_acc = (y_pred_val == Y_val_t).float().mean().item()
                    history['val_bce'].append(val_ce)
                    history['val_acc'].append(val_acc)

            if ckpt_path is not None and (epoch + 1) % save_every == 0:
                ckpt_file = f"{ckpt_path.rstrip('.pt')}_epoch{epoch + 1}.pt"
                self.save_checkpoint(ckpt_file, extras={'epoch': epoch + 1})
                print(f"Saved checkpoint: {ckpt_file}")

        final_theta = None
        if self.threshold_type == 'sampler':
            final_theta = self.threshold_module.raw_thresholds.detach().cpu().numpy()
        elif self.use_classical or self.threshold_type == 'classical':
            final_theta = None
        else:
            final_theta = self.threshold_module.theta.detach().cpu().numpy()

        final_thresholds = self.threshold_module().detach().cpu().numpy()
        final_mu_b = self.get_leaf_probs().detach().cpu().numpy()
        return final_theta, final_thresholds, final_mu_b, history


ObliviousTreeWithLeaf = ObliviousTree

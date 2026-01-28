import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from quantum_model import QuantumThresholds
from classical_model import ClassicalThresholds

# ------------------------ Utilities ------------------------

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
    """
    Extract human-readable decision rules from a trained VQA decision table.

    Args:
        model: trained DecisionTableVQA instance
        threshold_classification: threshold for converting mu_b to hard class (default 0.5)

    Returns:
        rules: list of tuples (rule_str, predicted_class, predicted_value)
    """

    bits = model.bits.cpu().numpy()
    feature_names = [f'F{i}' for i in model.feature_indices]

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
    def __init__(self,
                 d: int,
                 feature_indices: list,
                 device: torch.device = torch.device('cpu'),
                 alpha_init: float = 1.0,
                 alpha_final: float = 20.0,
                 epochs: int = 50,
                 batch_size: int = 64,
                 lr: float = 1e-2,
                 use_ema: bool = False,
                 ema_rho: float = 0.05,
                 eps: float = 1e-8,
                 num_classes: int = 2,
                 q_reps: int = 2,
                 q_dev: str = 'default.qubit',
                 q_shots: Optional[int] = None,
                 ansatz: str = 'ry',
                 use_classical: bool = False,
                 classical_hidden_size: int = 32,
                 classical_hidden_layers: int = 1):
        self.d = d
        self.feature_indices = feature_indices
        self.device = device
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.use_ema = use_ema
        self.ema_rho = ema_rho
        self.eps = eps
        self.num_classes = num_classes
        self.ansatz = ansatz
        self.use_classical = use_classical
        if use_classical:
            self.threshold_module = ClassicalThresholds(d=d, hidden_layers=classical_hidden_layers, hidden_size=classical_hidden_size).to(device)
        else:
            self.threshold_module = QuantumThresholds(d=d, reps=q_reps, dev_name=q_dev, shots=q_shots, ansatz=ansatz).to(device)
        self.optimizer = optim.Adam(self.threshold_module.parameters(), lr=self.lr)
        self.bits = bits_matrix(self.d, device=device)
        self.L = 2 ** d
        self.S_b = torch.zeros(self.L, self.num_classes, device=device, dtype=torch.float32) + 1e-6
        self.C_b = torch.zeros(self.L, device=device, dtype=torch.float32) + 1e-6

    def sigmoid_probs(self, X_batch: torch.Tensor, thresholds: torch.Tensor, alpha: float) -> torch.Tensor:
        X_selected = X_batch[:, self.feature_indices]
        p = torch.sigmoid(alpha * (X_selected - thresholds.unsqueeze(0)))
        return p

    def compute_responsibilities(self, X, thresholds, alpha):
        """
        Reconstruct responsibilities P for any dataset X.
        Equivalent to the logic used in minibatch training.
        """
        p = self.sigmoid_probs(X, thresholds, alpha)
        logP = log_responsibilities_from_probs(p, self.bits)
        P = softmax_normalize_log_probs(logP)
        return P

    def update_ema_from_minibatch(self, P_batch: torch.Tensor, Y_batch: torch.Tensor):
        P_batch = P_batch.float()
        Y_batch = Y_batch.long()
        self.S_b = self.S_b.float()
        self.C_b = self.C_b.float()
        one_hot_Y = torch.nn.functional.one_hot(Y_batch, num_classes=self.num_classes).float()
        Sb_batch = torch.matmul(P_batch.t(), one_hot_Y)
        Cb_batch = torch.sum(P_batch, dim=0)
        self.S_b = (1 - self.ema_rho) * self.S_b + self.ema_rho * Sb_batch.detach()
        self.C_b = (1 - self.ema_rho) * self.C_b + self.ema_rho * Cb_batch.detach()

    def compute_mu_b(self) -> torch.Tensor:
        return torch.softmax(self.S_b / (self.C_b.unsqueeze(1) + self.eps), dim=1)

    def predict_batch(self, X_batch: torch.Tensor, mu_b: torch.Tensor, thresholds: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P = self.compute_responsibilities(X_batch, thresholds, alpha)
        yhat = torch.matmul(P, mu_b)
        pred_class = torch.argmax(yhat, dim=1)
        return yhat, pred_class, P

    def save_checkpoint(self, path: str, extras: Optional[Dict[str, Any]] = None):
        if self.use_classical:
            state = {
                'net_state': self.threshold_module.net.state_dict(),
                'S_b': self.S_b.detach().cpu(),
                'C_b': self.C_b.detach().cpu(),
                'optimizer': self.optimizer.state_dict()
            }
        else:
            state = {
                'theta': self.threshold_module.theta.detach().cpu(),
                'S_b': self.S_b.detach().cpu(),
                'C_b': self.C_b.detach().cpu(),
                'optimizer': self.optimizer.state_dict()
            }
        if extras is not None:
            state.update(extras)
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if self.use_classical:
            self.threshold_module.net.load_state_dict(ckpt['net_state'])
            with torch.no_grad():
                self.S_b.copy_(ckpt['S_b'].to(torch.float32))
                self.C_b.copy_(ckpt['C_b'].to(torch.float32))
        else:
            with torch.no_grad():
                self.threshold_module.theta.copy_(ckpt['theta'].to(torch.float32))
                self.S_b.copy_(ckpt['S_b'].to(torch.float32))
                self.C_b.copy_(ckpt['C_b'].to(torch.float32))
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Loaded checkpoint from {path}")

    def train(self, 
              X_train: np.ndarray, 
              Y_train: np.ndarray, 
              X_val: np.ndarray = None, 
              Y_val: np.ndarray = None, 
              save_every: int = 5, 
              ckpt_path: Optional[str] = None):
        # Convert to tensors
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
                alpha = self.alpha_init + (self.alpha_final - self.alpha_init) * (step / max(1, total_steps-1))
                step += 1

                P = self.compute_responsibilities(Xb, thresholds, alpha)
                # ---- NO EMA VERSION ----
                if not self.use_ema:
                    # minibatch µ
                    one_hot_Yb = torch.nn.functional.one_hot(Yb, num_classes=self.num_classes).float()
                    S_b = torch.matmul(P.t(), one_hot_Yb)
                    C_b = torch.sum(P, dim=0)
                    mu_b = torch.softmax(S_b / (C_b.unsqueeze(1) + self.eps), dim=1)
                else:
                    # EMA version
                    self.update_ema_from_minibatch(P, Yb)
                    mu_b = self.compute_mu_b().float()
                # predictions for minibatch
                yhat = torch.matmul(P, mu_b)
                
                loss = nn.functional.cross_entropy(yhat, Yb)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.threshold_module.parameters(), max_norm=5.0)
                self.optimizer.step()

            with torch.no_grad():
                # recompute µ from full dataset depending on mode
                if self.use_ema:
                    mu_eval = self.compute_mu_b().float()
                else:
                    # NO EMA: recompute µ from full dataset
                    P_full = self.compute_responsibilities(X_t, self.threshold_module(), alpha=self.alpha_final)
                    one_hot_Y_t = torch.nn.functional.one_hot(Y_t, num_classes=self.num_classes).float()
                    S_full = torch.matmul(P_full.t(), one_hot_Y_t)
                    C_full = torch.sum(P_full, dim=0)
                    mu_eval = torch.softmax(S_full / (C_full.unsqueeze(1) + self.eps), dim=1)

                thresholds = self.threshold_module().float()

                yhat_train, y_pred_train, _ = self.predict_batch(X_t, mu_eval, thresholds, alpha=self.alpha_final)
                train_ce = nn.functional.cross_entropy(yhat_train, Y_t).item()
                train_acc = (y_pred_train == Y_t).float().mean().item()

                history['epoch'].append(epoch+1)
                history['train_bce'].append(train_ce)
                history['train_acc'].append(train_acc)

                if X_val is not None and Y_val is not None:
                    yhat_val, y_pred_val, _ = self.predict_batch(X_val_t, mu_eval, thresholds, alpha=self.alpha_final)
                    val_ce = nn.functional.cross_entropy(yhat_val, Y_val_t).item()
                    val_acc = (y_pred_val == Y_val_t).float().mean().item()
                    history['val_bce'].append(val_ce)
                    history['val_acc'].append(val_acc)
                    #print(f"Epoch {epoch+1}/{self.epochs}  |  Train CE: {train_ce:.6f}  |  Train Acc: {train_acc:.4f}  |  Val CE: {val_ce:.6f}  |  Val Acc: {val_acc:.4f}")
                else:
                    pass
                    #print(f"Epoch {epoch+1}/{self.epochs}  |  Train CE: {train_ce:.6f}  |  Train Acc: {train_acc:.4f}")

            if ckpt_path is not None and (epoch+1) % save_every == 0:
                ckpt_file = f"{ckpt_path.rstrip('.pt')}_epoch{epoch+1}.pt"
                self.save_checkpoint(ckpt_file, extras={'epoch': epoch+1})
                print(f"Saved checkpoint: {ckpt_file}")

        if self.use_classical:
            final_theta = None  # No quantum parameters for classical
        else:
            final_theta = self.threshold_module.theta.detach().cpu().numpy()
        final_thresholds = self.threshold_module().detach().cpu().numpy()
        if self.use_ema:
            # EMA stores the correct numerator/denominator
            final_mu_b = self.compute_mu_b().detach().cpu().numpy()
        else:
            # must recompute µ from the FULL training dataset
            with torch.no_grad():
                X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
                Y_t = torch.tensor(Y_train, dtype=torch.long, device=self.device)

                P_full = self.compute_responsibilities(X_t, self.threshold_module(), alpha=self.alpha_final)
                one_hot_Y_t = torch.nn.functional.one_hot(Y_t, num_classes=self.num_classes).float()
                S_full = torch.matmul(P_full.t(), one_hot_Y_t)
                C_full = torch.sum(P_full, dim=0)

                final_mu_b = torch.softmax(S_full / (C_full.unsqueeze(1) + self.eps), dim=1).cpu().numpy()
        return final_theta, final_thresholds, final_mu_b, history



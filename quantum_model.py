from typing import Optional, Tuple, Dict, Any

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn

# ------------------------ Quantum Module ------------------------

class QuantumThresholds(nn.Module):
    def __init__(self, d: int, reps: int = 2, dev_name: str = 'default.qubit', shots: Optional[int] = None, ansatz: str = 'ry'):
        super().__init__()
        self.d = d
        self.reps = reps
        self.dev_name = dev_name
        self.shots = None if (shots is None or int(shots) == 0) else int(shots)
        self.ansatz = ansatz.lower()
        if self.ansatz == 'ry':
            self.n_params = self.reps * self.d
        elif self.ansatz == 'two_param':
            self.n_params = self.reps * self.d * 2
        else:
            raise ValueError("ansatz must be 'ry' or 'two_param'")
        init = 0.01 * np.random.randn(self.n_params).astype(np.float32)
        self.theta = nn.Parameter(torch.tensor(init))
        self.dev = qml.device(self.dev_name, wires=self.d, shots=self.shots)

        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def circuit(params):
            if self.ansatz == 'ry':
                p = params.reshape(self.reps, self.d)
                for layer in range(self.reps):
                    for w in range(self.d):
                        qml.RY(p[layer, w], wires=w)
                    for w in range(self.d):
                        qml.CNOT(wires=[w, (w+1)%self.d])
            else:
                p = params.reshape(self.reps, self.d, 2)
                for layer in range(self.reps):
                    for w in range(self.d):
                        qml.RY(p[layer, w, 0], wires=w)
                        qml.RZ(p[layer, w, 1], wires=w)
                    for w in range(self.d):
                        qml.CNOT(wires=[w, (w+1)%self.d])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.d)]
        self._qnode = circuit

    def forward(self) -> torch.Tensor:
        exps = self._qnode(self.theta) 
        exps = torch.stack(exps) if isinstance(exps, (list, tuple)) else exps
        exps = exps.float()
        thresholds = 0.5 * (exps + 1.0)
        return thresholds
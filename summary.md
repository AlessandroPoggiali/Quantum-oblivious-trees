# Meeting 0 - Summary: Quantum Oblivious Trees

## What the Project Is

The project implements **differentiable oblivious decision trees** whose split thresholds can be generated either by a **parameterized quantum circuit** (PennyLane) or a **classical neural network** (PyTorch). The key idea:

1. Fix a set of features and their order (one feature per tree level).
2. Learn only the **thresholds** (one per level) via gradient descent -- the tree structure is determined by the feature list.
3. **Do not learn the leaves explicitly.** Instead, compute leaf class labels empirically at each training step as a majority-vote over the samples that (softly) fall into each leaf region.
4. Differentiability is achieved through **soft sigmoid-based binning** with an annealing temperature (alpha: 1 -> 20), which progressively sharpens the decision boundaries during training.

The quantum variant uses a variational circuit (RY rotations + CNOT entanglement, repeated `reps` times) with Pauli-Z expectation values passed through sigmoid to produce thresholds. The classical variant uses a small feed-forward network with no input ("dummy input"), matched in parameter count for fair comparison.

---

## Important Aspects Discussed

### 1. Empirical Leaf Computation (No Leaf Learning)
The most distinctive design choice. Standard oblivious tree methods (e.g., NODE, CatBoost) learn leaf values as parameters, which scales as 2^d (exponential in depth). Here, leaves are computed from the data at each step:
- For each leaf, accumulate `S_b` (sum of one-hot labels weighted by soft responsibilities) and `C_b` (count).
- Leaf prediction = `softmax(S_b / C_b)`.

This keeps the quantum circuit short (d qubits, ~d parameters) but shifts cost to the responsibility matrix computation, which is still O(N * 2^d) per step.

### 2. Quantum vs Classical Parity
At matched parameter counts, quantum and classical models perform **comparably** across 22 datasets. Quantum was better on 4/22 datasets (ecoli, glass, sonar, wine), classical better on 1/22, the rest similar. This is consistent with the broader QML literature where variational circuits rarely show clear advantage at small scale.

### 3. No-Input Threshold Generation
Both the quantum circuit and the classical baseline generate thresholds **without any data input** -- the circuit/network has only learnable parameters and no feature encoding. The meeting raised whether this might lead to overfitting concerns, but no conclusive answer was reached. The classical "dummy-input" network working well was noted as somewhat surprising.

### 4. Ensembles
Ensembles of oblivious trees (each with different feature subsets) were discussed as a natural way to increase expressiveness. The codebase supports:
- Pure quantum ensembles
- Pure classical ensembles
- Mixed quantum+classical ensembles
- Comparison against DNDT (Deep Neural Decision Trees)

### 5. Comparison with NODE and DNDT
- **NODE** (Neural Oblivious Decision Ensembles): uses oblivious trees with learned leaves; doesn't scale well with depth due to exponential leaf count.
- **DNDT** (Deep Neural Decision Trees): input-dependent, structurally different.
- **CatBoost**: uses "oblivious" trees but allows feature reuse across levels, making them not truly oblivious in the strict sense.

### 6. Related Work: Soft Binning (Neural Decision Trees by Ospedales)
The paper "Neural Decision Trees" was discussed. It uses a different approach with multiple cut-points per feature (soft binning with learned bin edges), a Kronecker product to compose per-feature decisions, and input-dependent processing. Much more complex but structurally different from this project's approach.

---

## Opportunities

### O1. Novel Contribution: Leaf-Free Differentiable Oblivious Trees
Not learning leaves explicitly appears to be a novel contribution even **classically**. The meeting noted this should be checked against the literature, but if confirmed, the classical-only version is already publishable as a new approach to differentiable decision trees.

### O2. Publication Strategy
- Target AI/ML conferences (not quantum-only), positioning the quantum part as an application case study.
- The classical leaf-free approach is the main novelty; the quantum variant adds topical interest.
- Workshop submissions (ICML workshops, etc.) were suggested as a realistic first target.
- The PKDD/ECML conference was mentioned as a possibility (deadline for main track passed, but workshops may be open).

### O3. Quantum Ensembles and Superposition
A speculative but exciting direction: exploiting quantum superposition to represent **multiple oblivious trees simultaneously** in a single circuit. This could allow:
- Quantum interference between ensemble members.
- Majority voting natively in amplitude space.
- Potential quantum advantage through parallel evaluation of multiple trees.

### O4. Mixed Ensembles
Combining classical and quantum trees in a single ensemble, potentially leveraging different inductive biases. The codebase already supports this.

### O5. Diagonal / Non-Axis-Aligned Splits
The meeting discussed extending beyond axis-aligned splits (feature > threshold) to **oblique splits** (linear combinations of features), which would partition the feature space with diagonal boundaries. This is a known enhancement in classical decision trees (oblique decision trees) but hasn't been explored in this quantum context.

### O6. Interpretability
Oblivious trees produce **human-readable rules** (the codebase has `extract_rules()`). This interpretability angle could be a selling point in domains where model transparency matters.

---

## Open Challenges

### C1. Scalability with Depth
The responsibility matrix is O(N * 2^d), making training expensive for deep trees even without learning leaves. This is a fundamental bottleneck shared with other soft-tree methods. The meeting acknowledged that avoiding leaf parameters doesn't eliminate this exponential cost.

### C2. No Clear Quantum Advantage
Current results show parity, not advantage. The no-input design means the quantum circuit is essentially a parameterized random number generator for thresholds -- there's no obvious mechanism for quantum speedup. The meeting expressed uncertainty about whether advantage is even theoretically possible with this architecture.

### C3. Overfitting Risk with Input-Free Thresholds
Since thresholds don't depend on input features, the model has no inductive bias linking thresholds to data structure. The meeting raised this concern but left it unresolved. The training loss does guide threshold updates, but the lack of feature encoding may limit generalization compared to input-dependent methods (like NODE or DNDT).

### C4. Feature Selection is External
The model assumes features are pre-selected and ordered before training. There's no mechanism to learn which features to use or their ordering. This limits end-to-end differentiability and requires external feature selection heuristics.

### C5. Theoretical Gap
There's no theoretical framework explaining when/why quantum-generated thresholds should outperform classical ones. The empirical results are "nice to have" but the project lacks a theory of quantum advantage for this setting. The meeting acknowledged this as a significant gap.

### C6. Comparison Fairness
The classical baseline uses a network with "dummy input" (no real input), which is an unusual architecture. It's unclear whether this is the strongest classical baseline, or whether a more standard approach (e.g., direct parameter optimization without a network) would perform differently.

### C7. Positioning Difficulty
The project sits between quantum computing and ML communities. Quantum conferences may find it too applied/incremental; ML conferences may not care about the quantum aspect. Finding the right venue requires careful framing.

---

## Key Technical Details from the Code

| Aspect | Detail |
|--------|--------|
| Quantum circuit | RY (or RY+RZ) rotations, CNOT ring entanglement, `reps` layers |
| Output mapping | PauliZ expectations -> sigmoid -> thresholds in [0,1] |
| Soft splitting | `sigmoid(alpha * (X_feature - threshold))` with alpha annealed 1->20 |
| Leaf computation | `softmax(S_b / (C_b + eps))`, with optional EMA smoothing |
| Optimizer | Adam, lr=0.01, gradient clipping at 5.0 |
| Datasets tested | 22 tabular classification datasets |
| Parameter matching | Quantum params = reps * d (or reps * d * 2); classical params matched via network sizing |

---

## Action Items from Meeting
1. Check literature for existing leaf-free differentiable oblivious tree approaches.
2. Look for ICML/ECML workshop deadlines as publication targets.
3. Re-run experiments more systematically with multiple seeds and configurations.
4. Consider whether changing the quantum ansatz or adding feature encoding could improve results.
5. Generate a transcript summary via LLM + code review (this document).

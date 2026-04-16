# Meeting 1 - Summary: Classical Threshold Generation for Oblivious Trees

## Context

Building on the existing Quantum Oblivious Trees project, this meeting focused on a **purely classical research direction**: understanding and comparing different strategies for generating split thresholds in differentiable oblivious trees. The key insight motivating this work is that the leaf-free, threshold-learning approach is novel even without quantum circuits, and deserves a thorough classical study before layering quantum methods on top.

---

## Architecture Recap

The training loop (as diagrammed on the board) is:

1. A **threshold generator** produces `d` thresholds `t = {t_1, ..., t_d}` (one per feature in the tree).
2. These thresholds are plugged into a **differentiable oblivious tree**, which computes soft predictions `y_hat` via sigmoid-based soft binning with temperature annealing.
3. Leaves are computed **empirically** (not learned) via soft responsibility-weighted majority voting.
4. A **BCE / cross-entropy loss** `L(y, y_hat)` is computed and gradients flow back to the threshold generator.
5. The threshold generator's parameters are updated, and the loop repeats.

The thresholds are passed through a **sigmoid** to map them to `[0, 1]` (normalized feature range). The generator takes a dummy input (no data encoding).

---

## Two Approaches to Compare

### Approach A: FFNN Threshold Generator ("CLASSICO FFNN")

A feed-forward neural network with dummy input that outputs `d` thresholds. The FFNN parameters are trained via backpropagation through the oblivious tree loss.

Each output node corresponds to one feature's threshold: `t_i = sigma(x * Theta_i + b_i)` for the 0-layer case (i.e., a linear layer + sigmoid).

**Key property**: the thresholds `t_i` are *indirect* parameters -- they depend on potentially many learnable weights, making this approach **more expressive** because each threshold is a function of multiple learnable parameters.

**Configurations to study**:

| Configuration | Description |
|---------------|-------------|
| **0 layers** (linear) | `Linear(1, d)` + Sigmoid. Parameters = `d` weights + `d` biases = `2d`. |
| **1 hidden layer** | `Linear(1, h)` -> ReLU -> `Linear(h, d)` + Sigmoid. Vary `h` (number of hidden neurons). |
| **N hidden layers** | For complex datasets. Start with 1 hidden layer, then explore deeper if needed. |

### Approach B: Direct Threshold Sampling ("CLASSICO SAMPLER")

Thresholds are initialized by **sampling from a uniform distribution** over the range of each feature's values, then **directly optimized** as learnable `nn.Parameter` tensors (no neural network involved).

**Key property**: each threshold `t_i` is a *direct* parameter -- one scalar per feature, trained via gradient descent on the same oblivious tree loss.

This is the simplest possible baseline: `d` free parameters, no network overhead.

---

## Experimental Plan

### Study 1: FFNN (0 layers) vs Sampler

Run **all 22 datasets** for both approaches. Compare along the following axes:

| Axis | What to Measure |
|------|-----------------|
| **Bias effect** | Does having bias terms in the FFNN (0-layer) matter? Compare with/without bias. |
| **Final nonlinearity** | Impact of the sigmoid activation on threshold quality. |
| **Convergence speed** | Training and validation loss curves over epochs. How quickly does each approach find good thresholds? |
| **Final accuracy** | Test accuracy at convergence for both methods. |

### Study 2: FFNN with Hidden Layers

Run **all 22 datasets** with deeper FFNN configurations to investigate whether a richer parameterization produces better thresholds:

| Configuration | What to Vary |
|---------------|--------------|
| **1 hidden layer** | Vary the number of hidden neurons `h`. Study how `h` affects expressiveness and convergence. |
| **Multiple hidden layers** | If 1 hidden layer shows promise, explore 2+ layers. |

### Study 3: Hyperparameter Sensitivity

Both approaches must be studied under varying:

| Hyperparameter | Values to Explore |
|----------------|-------------------|
| **Optimizer** | Adam, SGD |
| **Learning rate** | Grid search over e.g., {0.001, 0.005, 0.01, 0.05, 0.1} |

### Study 4: Decision Tree Baseline

Compare both threshold generation approaches against a **standard (sklearn) decision tree** to contextualize results in terms of:

- **Expressiveness**: can the differentiable oblivious tree match a fully-grown decision tree?
- **Convergence**: how does the gradient-based approach compare to greedy splitting?

---

## Evaluation Protocol

- **Data splits**: train / validation / test (already available for all 22 datasets).
- **Metrics**: test accuracy, test cross-entropy, validation accuracy (for model selection).
- **Convergence curves**: plot training and validation loss/accuracy over epochs for both approaches.
- **Multiple seeds**: run each configuration multiple times to report mean +/- std.
- **Datasets**: use all datasets already explored in the project (22 tabular classification datasets).

---

## Key Questions to Answer

1. **Is the FFNN necessary?** Does the indirect parameterization (FFNN) produce better thresholds than direct optimization (sampler), or is the network overhead unnecessary?
2. **How does expressiveness scale with network size?** Does adding hidden layers/neurons to the FFNN improve threshold quality, or does it overfit?
3. **Convergence dynamics**: which approach converges faster? Is there a tradeoff between convergence speed and final accuracy?
4. **Optimizer sensitivity**: are the two approaches differently sensitive to the choice of optimizer and learning rate?
5. **When does depth help?** On which datasets (few features vs many, few classes vs many, small vs large) does a deeper FFNN provide benefit?

---

## Relationship to the Broader Project

This classical study serves as the **foundation** for the quantum comparison. By thoroughly understanding the classical threshold generation landscape (direct optimization vs. FFNN of varying depth), we establish:

- A strong classical baseline to compare quantum circuits against.
- An understanding of how indirect parameterization (through a network) affects threshold quality -- since the quantum circuit is also an indirect parameterization.
- Insights into whether the "more parameters -> more expressive thresholds" hypothesis holds, which directly informs whether quantum circuits (as another form of parameterized function) could offer any advantage.

---

## Next Steps (Prioritized)

1. **Implement the Sampler approach** (uniform initialization + direct parameter optimization).
2. **Set up the experimental grid**: datasets x {FFNN-0layer, FFNN-1hidden(vary h), Sampler} x {optimizers} x {learning rates}.
3. **Run Study 1** on simple datasets first (fast iteration).
4. **Analyze convergence curves** and accuracy results.
5. **Extend to Study 2** (complex datasets, deeper FFNNs) based on Study 1 findings.
6. **Add decision tree baseline** for expressiveness/convergence comparison.

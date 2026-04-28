# What I Inferred from the 4 Studies

I read `meeting_1_summary.md`, the design spec at `docs/superpowers/specs/2026-04-16-classical-threshold-studies-design.md`, the four `results/study*/results.csv` files (171 + 171 + 1086 + 57 rows), and a sample of the convergence plots. Here is what the data actually says, organized around the questions Meeting 1 set out to answer.

## Headline numbers (mean test accuracy across all datasets/depths)

| Study | Approach | Test acc | # params (avg) |
|---|---|---|---|
| 1 | FFNN-0layer (bias) | 0.7046 | 14.5 |
| 1 | FFNN-0layer (no bias) | 0.7043 | 14.5 |
| 1 | Sampler | **0.7054** | **7.3** |
| 2 | FFNN-1h, h=4 | 0.7053 | 44 |
| 2 | FFNN-1h, h=8 | 0.7046 | 81 |
| 2 | FFNN-1h, h=16 | 0.7047 | 155 |
| 3 | best (Adam, LR=0.01) | 0.7020 | — |
| 4 | sklearn DecisionTree | **0.7129** | — |

The differences between every classical OBT variant are within ~0.001. The DT baseline is ~0.8 pp higher.

## Inference 1 — The FFNN is not necessary (Q1 from Meeting 1)

Sampler and FFNN-0layer-no-bias are *mathematically* near-equivalent given a 1-D dummy input (`sigmoid(w·1)` is just a reparameterized scalar) and the data confirm it: their per-dataset accuracies are identical to 4 decimals on most datasets. Sampler does this with **half the parameters** and one extra "win" overall.

Adding bias to the 0-layer FFNN gives a mean **+0.002** improvement (median +0.001), but with ±4 pp swings across datasets — a wash, not a real effect. Bias helps decisively on `led7` (+0.11) and `car` (+0.05); it hurts on `australian` (-0.08), `ecoli` (-0.06), `heart` (-0.05).

**Conclusion**: the indirect FFNN parameterization buys nothing over a direct learnable parameter. The "more learnable weights → more expressive thresholds" hypothesis does **not** hold here.

## Inference 2 — Hidden layers don't help either (Q2)

Going from 14 parameters (0-layer) to 155 (1 hidden layer, h=16) yielded a delta of +0.0001 in mean test accuracy. h=4 ≈ h=8 ≈ h=16 across all 22 datasets. The 18-vs-4 "win count" for h=16 is rounding noise.

**Conclusion**: the expressiveness bottleneck is *not* in the threshold generator. Whatever is preventing further accuracy gains lives in the differentiable oblivious tree itself — soft binning, leaf computation, or the temperature schedule — not in how thresholds are produced.

## Inference 3 — Convergence story is more interesting than the final accuracy (Q3)

From `results/study1/plots/sonar/convergence_acc_d12.png`: FFNN-bias starts higher and reaches its training plateau **~30 epochs faster** than Sampler. Sampler trains slowest but ends at the same point. Validation curves for both FFNN variants oscillate and even degrade after ~60 epochs (overfitting visible), while Sampler's validation is smoother.

So although final accuracy is tied, FFNN gets there faster but with more variance and overfitting; Sampler is slower but better-behaved.

## Inference 4 — Optimizer/LR sensitivity (Q4)

| | best LR | best acc | spread (max−min over LRs) |
|---|---|---|---|
| FFNN, Adam | 0.01 | 0.7020 | 0.013 |
| FFNN, SGD | 0.10 | 0.6760 | 0.022 |
| Sampler, Adam | 0.01 | 0.7019 | 0.023 |
| Sampler, SGD | 0.10 | 0.6664 | 0.018 |

- **Adam strictly dominates SGD** by ~3 pp regardless of approach; SGD never catches up even at LR=0.1.
- **Sampler is ~2× more LR-sensitive under Adam** than FFNN — at LR=0.001 it drops to 0.679 while FFNN holds 0.689.
- Adam @ 0.01 is the safe default for both.

## Inference 5 — DT baseline reveals a real failure mode in OBT (Q5)

Per-dataset head-to-head (best Study-1 OBT vs sklearn DT, averaged over depths): **OBT wins 14/22, DT wins 8/22**, mean diff +0.012 in favor of OBT, but DT crushes OBT on a specific cluster:

| Dataset | OBT − DT | Notes |
|---|---|---|
| avila | **−0.26** | OBT stuck at 0.411 (≈ majority class) at *every* depth |
| egg | **−0.19** | OBT stuck at 0.567 at every depth |
| drybean | −0.09 | OBT improves with depth, DT just better |
| car | −0.08 | |
| glass | −0.06 | |

OBT crushes DT on `sonar` (+0.18), `australian` (+0.14), `lymph` (+0.14), `yeast` (+0.11), `pima` (+0.10), `heart` (+0.09), `iris` (+0.08), `fico` (+0.06).

The pattern: **OBT wins on binary / well-separated datasets, loses on multi-class problems where greedy splitting matters**. On `avila` and `egg`, every classical OBT variant — FFNN-bias, FFNN-nobias, Sampler, h=4/8/16 — collapses to the majority-class accuracy regardless of depth. The CE loss barely moves (avila: 2.4707 → 2.4699 over 100 epochs). This is the soft-binning landscape failing, not the threshold generator failing.

---

## What this means for the broader project

1. **For the classical study**: the cheapest baseline (Sampler) is the right one to keep. FFNN adds parameters and overfitting risk for no accuracy gain. Hidden layers can be dropped from future grids.
2. **For the quantum direction**: the "indirect parameterization buys expressiveness" rationale that motivated the comparison didn't survive the classical ablation. A quantum threshold generator has to demonstrate something *other* than expressive thresholds — better optimization landscape, robustness on hard datasets like avila/egg, or different inductive bias.
3. **The most interesting open problem isn't FFNN-vs-Sampler-vs-Quantum**: it's why the soft-OBT collapses to majority-class on `avila` and `egg` no matter what generates the thresholds. Worth investigating before pouring more cycles into threshold generators.

# Quantum Oblivious Trees - Testing Framework

A comprehensive benchmarking framework for comparing **quantum vs classical oblivious tree models** across multiple datasets with automatically matched parameter counts.

## Quick Start

```bash
# Test all datasets with default settings
python test.py

# Test specific datasets with custom parameters
python test.py --datasets iris,wine,glass --epochs 30 --d 4

# Analyze results
python analyze_results.py results.csv
```

## What This Does

This framework automatically:

1. **Loads all datasets** from the `datasets/` folder (22 datasets including iris, wine, breast, heart, etc.)
2. **Tests both models** on each dataset:
   - **Classical model**: Simple linear threshold module (no hidden layers)
   - **Quantum model**: Quantum circuit with parameters matched to classical
3. **Compares performance** on test set and reports:
   - ✅ When quantum performs better
   - ❌ When classical performs better  
   - ⚖️ When performance is similar
4. **Saves results to CSV** for further analysis
5. **Generates summaries and visualizations**

The framework **automatically matches quantum and classical parameter counts**:

```
Classical (depth d):     d + d parameters (linear layer with bias)
Quantum (matched):       Automatically choose repetitions and ansatz
                        to achieve the same parameter count
```

This ensures a **fair comparison** - both models have equal capacity.


## Available Datasets

22 datasets in total:
- Classic: `iris`, `wine`, `glass`, `vehicle`
- Medical: `breast`, `heart`, `ecoli`, `pima`
- Financial: `german`, `australian`, `fico`
- Behavioral: `churn`, `compas`
- Biometric: `sonar`, `avila`, `banknote`, `yeast`
- Other: `car`, `drybean`, `egg`, `led7`, `lymph`

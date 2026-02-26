#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
SCRIPT="test_obt_vs_dndt.py"

SINGLE_DATASET="sonar"
NUM_RUNS=3
EPOCHS=100
DNDT_EPOCHS=100
NUM_CLASSICAL=5
NUM_QUANTUM=5
OBT_FEATURES=10
NUM_DNDT=10
DNDT_FEATURES=10

"${PYTHON_BIN}" "${SCRIPT}" \
  --single-dataset "${SINGLE_DATASET}" \
  --num-runs "${NUM_RUNS}" \
  --epochs "${EPOCHS}" \
  --dndt-epochs "${DNDT_EPOCHS}" \
  --num-classical "${NUM_CLASSICAL}" \
  --num-quantum "${NUM_QUANTUM}" \
  --obt-features "${OBT_FEATURES}" \
  --num-dndt "${NUM_DNDT}" \
  --dndt-features "${DNDT_FEATURES}"

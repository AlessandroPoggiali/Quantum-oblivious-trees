#!/bin/bash

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate obt

# Use GPU 1 (A100 40GB)
export CUDA_VISIBLE_DEVICES=1

# W&B config
export WANDB_API_KEY="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"

cd ~/Quantum-oblivious-trees

# Clear previous errors log
> errors.log

echo "=== Starting all studies at $(date) ==="
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'CPU')"

echo ""
echo "=== Study 1: FFNN vs Sampler ==="
python -u study1_ffnn_vs_sampler.py --datasets all --num-runs 5 --epochs 100 --wandb
echo "Study 1 completed at $(date)"

echo ""
echo "=== Study 2: Hidden Layers ==="
python -u study2_hidden_layers.py --datasets all --num-runs 5 --epochs 100 --wandb
echo "Study 2 completed at $(date)"

echo ""
echo "=== Study 3: HP Sensitivity ==="
python -u study3_hp_sensitivity.py --datasets all --num-runs 5 --epochs 100 --wandb
echo "Study 3 completed at $(date)"

echo ""
echo "=== Study 4: DT Baseline ==="
python -u study4_dt_baseline.py --datasets all --num-runs 5 --wandb
echo "Study 4 completed at $(date)"

echo ""
echo "=== All studies completed at $(date) ==="

if [ -s errors.log ]; then
    echo ""
    echo "=== ERRORS DETECTED (see errors.log) ==="
    cat errors.log
fi

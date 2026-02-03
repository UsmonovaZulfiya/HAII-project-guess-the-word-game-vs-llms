#!/bin/bash
set -euo pipefail

# Create log/output dirs
mkdir -p /mnt/h200_raid5/zulfiyausmonova/logs
mkdir -p /mnt/h200_raid5/zulfiyausmonova/outputs/semantic_drift
mkdir -p /mnt/h200_raid5/zulfiyausmonova/hf_cache

cd /home/zulfiyausmonova/projects/llm_chain
source venv/bin/activate

export HF_HOME=/mnt/h200_raid5/zulfiyausmonova/hf_cache
export OUTPUT_BASE=/mnt/h200_raid5/zulfiyausmonova/outputs/semantic_drift

# REMOVED THE HARDCODED CUDA_VISIBLE_DEVICES LINE
# Now it will use whatever you set in the terminal command.

export NUM_STEPS=10
export NUM_INSTANCES=100
export GEN_TEMPERATURE=0.7
export GEN_MAX_NEW_TOKENS=140

python /home/zulfiyausmonova/projects/llm_chain/code/run_gemma_weak_isolated.py
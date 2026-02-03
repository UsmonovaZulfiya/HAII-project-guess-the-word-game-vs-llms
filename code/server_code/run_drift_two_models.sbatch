#!/bin/bash
#SBATCH -J drift2
#SBATCH -p a100_only
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /mnt/h200_raid5/zulfiyausmonova/logs/%x_%j.out
#SBATCH -e /mnt/h200_raid5/zulfiyausmonova/logs/%x_%j.err




#!/bin/bash

set -euo pipefail

# Create log/output dirs on the big filesystem (safe if they already exist)
mkdir -p /mnt/h200_raid5/zulfiyausmonova/logs
mkdir -p /mnt/h200_raid5/zulfiyausmonova/outputs/semantic_drift
mkdir -p /mnt/h200_raid5/zulfiyausmonova/hf_cache

# Go to project root (code lives in /home; heavy writes go to /mnt)
cd /home/zulfiyausmonova/projects/llm_chain
source venv/bin/activate

# Redirect ALL HF caches to /mnt/h200_raid5 (avoid /home usage)
export HF_HOME=/mnt/h200_raid5/zulfiyausmonova/hf_cache
export TRANSFORMERS_CACHE=/mnt/h200_raid5/zulfiyausmonova/hf_cache/transformers
export HF_DATASETS_CACHE=/mnt/h200_raid5/zulfiyausmonova/hf_cache/datasets
export HF_HUB_CACHE=/mnt/h200_raid5/zulfiyausmonova/hf_cache/hub

# Output location on /mnt/h200_raid5
export OUTPUT_BASE=/mnt/h200_raid5/zulfiyausmonova/outputs/semantic_drift

# Experiment knobs (adjust as needed)
export NUM_STEPS=5
export NUM_INSTANCES=20
export GEN_TEMPERATURE=0.7
export GEN_MAX_NEW_TOKENS=140
export CUDA_VISIBLE_DEVICES=2


# Optional: restrict categories to reduce runtime
export ONLY_CATEGORIES="High-Freq Concrete"

python /home/zulfiyausmonova/projects/llm_chain/code/run_semantic_drift_local.py
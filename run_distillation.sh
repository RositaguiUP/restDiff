#!/usr/bin/env bash
set -e

# Target Scene Directory Location
VERSION="v4.1"
SCENE="2F5Z7_007"
FLOOR="0"
DATA_PATH="data/${SCENE}/${FLOOR}"
WARMUP_VERSION="v6.0"
# --- 2. Distillation & Strategy Parameters ---
STRATEGY="default" # Change to "mcmc" if you want to use the MCMC strategy
LAMBDA_RGB=15.0
LAMBDA_DEPTH=3.0

# --- 3. Dynamic Loss Scheduling Parameters ---
# Ensure these match or logically follow your warmup phase
DEPTH_START=0.3
DEPTH_END=0.02
HOLD_STEPS=10000
DECAY_STEPS=15000

echo "==================================================================="
echo "[STAGE 3] Executing Diffusion Distillation Refinement"
echo "Scene: ${SCENE} | Floor: ${FLOOR} | Version: ${VERSION}"
echo "Strategy: ${STRATEGY} | Restoring from Warmup: ${WARMUP_VERSION}"
echo "==================================================================="

# Execute Distillation Pipeline
python train_dist.py \
    --scene_name "$SCENE" \
    --floor_number "$FLOOR" \
    --version "$VERSION" \
    --data_dir "$DATA_PATH" \
    --warmup_version "$WARMUP_VERSION" \
    --strategy_type "$STRATEGY" \
    --run_eval \
    --lambda_distill_rgb "$LAMBDA_RGB" \
    --lambda_distill_depth "$LAMBDA_DEPTH" \
    --depth_start "$DEPTH_START" \
    --depth_end "$DEPTH_END" \
    --hold_steps "$HOLD_STEPS" \
    --decay_steps "$DECAY_STEPS"
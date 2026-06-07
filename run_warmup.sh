#!/usr/bin/env bash
set -e

# Target Scene Directory Location
VERSION="v6.0"
SCENE="74VND_105"
FLOOR="1"
DATA_PATH="data/${SCENE}/${FLOOR}"
STRATEGY="default" # "default" or "mcmc"

echo "[STAGE 2] Executing Geometrical Warmup on scene: ${SCENE} | Floor: ${FLOOR} (Version: ${VERSION})"
python train_warmup.py \
    --scene_name "$SCENE" \
    --floor_number "$FLOOR" \
    --version "$VERSION" \
    --data_dir "$DATA_PATH" \
    --strategy_type "$STRATEGY" \
    --run_eval \
    --depth_start 0.3 \
    --depth_end 0.02 \
    --hold_steps 7500 \
    --decay_steps 17500
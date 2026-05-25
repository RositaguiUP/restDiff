#!/usr/bin/env bash
set -e

# Target Scene Directory Location
SCENE="6VSV7_695_v2"
VERSION="v5.2"
DATA_PATH="data/${SCENE}"

STRATEGY="mcmc" # "default" or "mcmc"

echo "[STAGE 2] Executing Geometrical Warmup on scene: ${SCENE} (Version: ${VERSION})"
python train_warmup.py \
    --scene_name "$SCENE" \
    --version "$VERSION" \
    --data_dir "$DATA_PATH" \
    --strategy_type "$STRATEGY" \
    --depth_start 0.3 \
    --depth_end 0.02 \
    --hold_steps 7500 \
    --decay_steps 17500
#!/usr/bin/env bash
set -e

# Target Scene Directory Location
SCENE="6VSV7_695_v2"
VERSION="v1"
DATA_PATH="data/${SCENE}"

echo "[STAGE 3] Executing Diffusion Distillation Refinement on scene: ${SCENE} (Version: ${VERSION})"
python train_dist.py \
    --scene_name "$SCENE" \
    --version "$VERSION" \
    --data_dir "$DATA_PATH" \
    --warmup_version "v1" \
    --lambda_distill_rgb 0.7 \
    --lambda_distill_depth 0.3
#!/usr/bin/env bash
set -e

# Target Scene Directory Location
SCENE="6VSV7_695_v2"
VERSION="v1.0"
DATA_PATH="data/${SCENE}"

echo "[STAGE 2] Executing Geometrical Warmup on scene: ${SCENE} (Version: ${VERSION})"
python train_warmup.py \
    --scene_name "$SCENE" \
    --version "$VERSION" \
    --data_dir "$DATA_PATH" \
    --lambda_warmup_rgb 1.0 \
    --lambda_warmup_depth 0.0
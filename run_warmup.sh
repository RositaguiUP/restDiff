#!/usr/bin/env bash
set -e

# Target Scene Directory Location
VERSION="v5.1"
SCENE="6VSV7_695_v2"
DATA_PATH="data/${SCENE}"
RESULT_DIR="results/${SCENE}/warmup/${VERSION}"

STRATEGY="mcmc" # "default" or "mcmc"

echo "========================================================"
echo " 1. Training Scene (Without Interruption) "
echo "========================================================"
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

# echo "========================================================"
# echo " 2. Standalone Batch Evaluation "
# echo "========================================================"
# for CKPT in $RESULT_DIR/checkpoints/ckpt_warmup_*.pt;
# do
#     if [ -f "$CKPT" ]; then
#         python evaluation.py \
#             --scene_name "$SCENE" \
#             --version "$VERSION" \
#             --data_dir "$DATA_PATH" \
#             --ckpt "$CKPT"
#     fi
# done

# echo "========================================================"
# echo " 3. Read Metrics "
# echo "========================================================"
# for STATS in $RESULT_DIR/stats/val_step*.json;
# do  
#     if [ -f "$STATS" ]; then
#         echo "File: $STATS"
#         cat "$STATS"
#         echo -e "\n"
#     fi
# done
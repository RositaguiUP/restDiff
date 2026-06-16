#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
# Define your single configuration here:
MODE="tile"            # Options: "tile" or "multi"
TRAIN_CONDITION="render"   # Options: "render" or "gt"
DATASET_TYPE="scan"    # Options: "scan" or "dslr"
USE_FINETUNED=true     # Set to true for finetuned, false for community models

# Path/Versioning Constants
MODEL_VERSION="${MODE}_${TRAIN_CONDITION}_${DATASET_TYPE}"
EPOCH=9

# ==========================================
# PATH CONSTRUCTION
# ==========================================
if [ "$USE_FINETUNED" = true ]; then
    TILE_PATH="/home/rosita/tests/diff/restDiff/finetuned_models/${MODEL_VERSION}/controlnet_epoch_${EPOCH}/tile"
    DEPTH_PATH="/home/rosita/tests/diff/restDiff/finetuned_models/${MODEL_VERSION}/controlnet_epoch_${EPOCH}/depth"
    MODEL_LABEL="Finetuned ${MODEL_VERSION} (Epoch ${EPOCH})"
    OUTPUT_SUBFOLDER="${MODEL_VERSION}/epoch_${EPOCH}"
else
    TILE_PATH="lllyasviel/control_v11f1e_sd15_tile"
    DEPTH_PATH="lllyasviel/control_v11f1p_sd15_depth"
    MODEL_LABEL="Community SD1.5 tile"
    # MODEL_LABEL="Community SD1.5 multi"
    OUTPUT_SUBFOLDER="community_sd1.5"
fi

# ==========================================
# BASE DIRECTORY CONFIGURATIONS
# ==========================================
SCENES="2F5Z7_007" 
FLOORS="0"
STAGE="warmup"
VERSION="v6.0"
# POSES_VERSION="poses_hd"
POSES_VERSION="poses_to_render/trajectory_simple_10"
# POSES_VERSION="poses_to_render/poses_6"
STRENGTH=0.7
DESCRIPTION="Inference"


if [[ "$MODEL_LABEL" == *"tile"* ]]; then
    CONFIGS=(
    '{"mode": "tile", "conditional": "render", "gt_type": "none"}'
    )
    if [[ "$POSES_VERSION" == *"hd"* ]]; then
        CONFIGS+=('{"mode": "tile", "conditional": "gt", "gt_type": "dslr"}')
    elif [[ "$POSES_VERSION" == *"/poses"* ]]; then
        CONFIGS+=('{"mode": "tile", "conditional": "gt", "gt_type": "scan"}')
    fi
else
    CONFIGS=(
    '{"mode": "multi", "conditional": "render", "gt_type": "none"}'
    )
    if [[ "$POSES_VERSION" == *"/poses"* ]]; then
        CONFIGS=(
        '{"mode": "multi", "conditional": "gt", "gt_type": "scan"}'
        )
    fi  
fi

printf "%s\n" "${CONFIGS[@]}"

# ==========================================
# EXECUTION (Only run the configured version)
# ==========================================
echo "====================================================="
echo "Running Inference: $MODEL_LABEL"
echo "====================================================="

# Reusable command foundation
python inference.py \
    --scenes $SCENES \
    --floors $FLOORS \
    --stage $STAGE \
    --version $VERSION \
    --poses_version $POSES_VERSION \
    --controlnet_tile_path $TILE_PATH \
    --controlnet_depth_path $DEPTH_PATH \
    --output_subfolder $OUTPUT_SUBFOLDER \
    --model_label "$MODEL_LABEL" \
    --description "$DESCRIPTION" \
    --configs "${CONFIGS[@]}" \
    --strength $STRENGTH

echo -e "\n====================================================="
echo "Inference cycle complete."
echo "====================================================="
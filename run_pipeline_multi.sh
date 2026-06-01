#!/bin/bash
set -e

# ==========================================
# Global Variables
# ==========================================
VERSION="v5.1"
STRATEGY="default" # "default" or "mcmc"

# Define the list of scenes. 
SCENES=("2T382_260" "7HTM8_099" "DVJQQ_367" "SCX7D_671" "5FF69_654" "74VND_105" "7RHGW_158" "DV5B6_994" "GDB3D_521" "MK5Y8_583") 

# SCENES=("6VSV7_695_v2")

# Initialize Conda for bash script usage
# (This is required to use 'conda activate' inside a shell script)
source $(conda info --base)/etc/profile.d/conda.sh

# echo "========================================================"
# echo " Phase 1: Generating Inputs for all scenes"
# echo "========================================================"

# # Activate the environment for Phase 1
# echo "⚙️ Activating conda environment: test"
# conda activate test

# for SCENE in "${SCENES[@]}"; do
#     echo "-> Running generate_inputs.py for $SCENE"
    
#     # Catch errors and skip to the next scene if it fails
#     if ! python generate_inputs.py --config_path "configs/${SCENE}.yaml"; then
#         echo "⚠️ [WARNING] generate_inputs.py failed for $SCENE. Skipping to next scene."
#         continue
#     fi
# done

echo ""
echo "========================================================"
echo " Phase 2: Training Warmup for all scenes"
echo "========================================================"

# Activate the environment for Phase 2
echo "⚙️ Activating conda environment: restDiff"
conda activate restDiff

for SCENE in "${SCENES[@]}"; do
    YAML_PATH="configs/${SCENE}.yaml"
    
    # Check if config exists before reading floors
    if [ ! -f "$YAML_PATH" ]; then
        echo "⚠️ [WARNING] Config file $YAML_PATH not found. Skipping scene $SCENE."
        continue
    fi

    # Dynamically extract the floor numbers list from the YAML file
    # If it fails or key is missing, defaults to floor 0
    FLOORS=$(python -c "import yaml; c = yaml.safe_load(open('$YAML_PATH')); print(' '.join(map(str, c.get('floor_numbers', [0]))))" 2>/dev/null || echo "0")
    
    echo "Processing scene: ${SCENE} (Floors found: ${FLOORS})"
    
    # Inner loop to iterate over each floor in the scene
    for FLOOR in $FLOORS; do
        # Update data path to point to the specific floor directory generated in Phase 1
        DATA_PATH="data/${SCENE}/${FLOOR}"
        
        echo " -> [STAGE 2] Executing Geometrical Warmup: ${SCENE} | Floor: ${FLOOR} (Version: ${VERSION})"
        
        # Catch errors during the warmup phase per floor
        if ! python train_warmup.py \
            --scene_name "$SCENE" \
            --floor_number "$FLOOR" \
            --version "$VERSION" \
            --data_dir "$DATA_PATH" \
            --strategy_type "$STRATEGY" \
            --depth_start 0.3 \
            --depth_end 0.02 \
            --hold_steps 7500 \
            --decay_steps 17500; then
            
            echo "⚠️ [WARNING] train_warmup.py failed for ${SCENE} on Floor ${FLOOR}. Skipping to next floor."
            echo "--------------------------------------------------------"
            continue
        fi
        
        echo "✅ Finished warmup for $SCENE."
        echo "--------------------------------------------------------"
    done
done

echo "🎉 Pipeline execution completed."
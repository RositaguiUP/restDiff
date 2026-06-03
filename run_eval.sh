#!/bin/bash

# Default Configuration Constants
STAGE="warmup"
VERSION="v6.0"
STEPS="14999 29999"
SCENES="DVJQQ_367"
MAX_VIS="15"
RUN_COMPILE=true
RUN_VISUALIZE=true
# OUTPUT_CSV="./results/_stats_/stats_comparison_v6_0.csv"
OUTPUT_CSV="./results/_stats_/test01.csv"

usage() {
    echo "Usage: $0 --stage <stage> --version <version> --steps <step1 step2 ...> [Options]"
    echo ""
    echo "Required Arguments:"
    echo "  -s, --stage       Target tracking experiment phase (e.g. warmup, fine)"
    echo "  -v, --version     Experiment version identification code (e.g. v5.1)"
    echo "  -t, --steps       Space-separated integer checkpoints list (e.g. '29999 30000')"
    echo ""
    echo "Execution Control Options (Runs both tasks if neither option is supplied):"
    echo "  -c, --compile-only   Only run step metrics calculation script"
    echo "  -i, --vis-only       Only process novel view rasterization generation & tracking plots"
    echo ""
    echo "Data Control Options:"
    echo "  --scenes             Space-separated target scenes list (e.g. 'room1 room2')"
    echo "  --max-vis            Max number of frames to sample randomly for plotting"
    echo "  --output             Alternative file save location path for the metrics spreadsheet"
    exit 1
}

# Parse Command Line Options
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--stage)         STAGE="$2"; shift 2 ;;
        -v|--version)       VERSION="$2"; shift 2 ;;
        -t|--steps)         STEPS="$2"; shift 2 ;;
        -c|--compile-only)  RUN_VISUALIZE=false; shift ;;
        -i|--vis-only)      RUN_COMPILE=false; shift ;;
        --scenes)           SCENES="$2"; shift 2 ;;
        --max-vis)          MAX_VIS="$2"; shift 2 ;;
        --output)           OUTPUT_CSV="$2"; shift 2 ;;
        *)                  usage ;;
    esac
done

# Enforce Validation Rules
if [ -z "$STAGE" ] || [ -z "$VERSION" ] || [ -z "$STEPS" ]; then
    echo "[-] Error: Missing mandatory fields."
    usage
fi

# Pipeline Execution Step 1: Metric Ingestion Engine
if [ "$RUN_COMPILE" = true ]; then
    echo "================================================================="
    echo "[Task 1/2] Initiating Table Consolidation Module..."
    echo "================================================================="
    
    COMPILE_CMD="python evaluation/compile_stats.py --stage \"$STAGE\" --version \"$VERSION\" --steps $STEPS --output \"$OUTPUT_CSV\""
    if [ -n "$SCENES" ]; then COMPILE_CMD="$COMPILE_CMD --scenes $SCENES"; fi
    
    eval $COMPILE_CMD
fi

# Pipeline Execution Step 2: Novel View Depth/RGB Render and Cross Plotting Module
if [ "$RUN_VISUALIZE" = true ]; then
    echo "================================================================="
    echo "[Task 2/2] Initiating Multi-Channel Rendering & Plotting Engine..."
    echo "================================================================="
    
    VIS_CMD="python evaluation/visualize_eval.py --stage \"$STAGE\" --version \"$VERSION\" --steps $STEPS"
    if [ -n "$SCENES" ]; then VIS_CMD="$VIS_CMD --scenes $SCENES"; fi
    if [ -n "$MAX_VIS" ]; then VIS_CMD="$VIS_CMD --max-visualizations $MAX_VIS"; fi
    
    eval $VIS_CMD
fi

echo -e "\n[+] Script sequence complete."
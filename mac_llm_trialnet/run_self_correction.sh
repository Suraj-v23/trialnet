#!/usr/bin/env bash
# run_self_correction.sh — Full TrialNet self-correction pipeline
#
# Usage: bash run_self_correction.sh [--iters N]
#
# Steps:
#   1. Export correction pairs from ChromaDB
#   2. Run MLX LoRA self-correction (creates next versioned adapter)
#   3. Evaluate new adapter against baseline questions
#   4. Print diff vs previous adapter

set -e

ITERS=${2:-50}
if [[ "$1" == "--iters" ]]; then ITERS=$2; fi

echo "=============================="
echo " TrialNet Self-Correction Loop"
echo "=============================="

echo ""
echo "[1/3] Running self-correction fine-tune..."
python3 3_mac_self_correct.py --iters "$ITERS"

echo ""
echo "[2/3] Finding latest adapter for eval..."
LATEST=$(ls -d mac_trialnet_v*_adapter 2>/dev/null | sort -V | tail -1)
PREV=$(ls -d mac_trialnet_v*_adapter 2>/dev/null | sort -V | tail -2 | head -1)

if [ -z "$LATEST" ]; then
    echo "No adapter found. Run 1_mac_finetune.py first."
    exit 1
fi

echo "       Evaluating: $LATEST"
python3 evaluate_mac.py --adapter "./$LATEST"

echo ""
echo "[3/3] Comparing versions..."
if [ "$LATEST" != "$PREV" ] && [ -n "$PREV" ]; then
    LATEST_NAME=$(basename "$LATEST")
    PREV_NAME=$(basename "$PREV")
    if [ -f "eval_results/${PREV_NAME}.json" ]; then
        python3 evaluate_mac.py --compare "$PREV_NAME" "$LATEST_NAME"
    else
        echo "No previous eval baseline for $PREV_NAME. Run evaluate_mac.py --adapter ./$PREV first."
    fi
else
    echo "Only one adapter found. Run again after next correction cycle."
fi

echo ""
echo "Done. Update ADAPTER_DIR in 2_mac_chatbot.py to './$LATEST'"

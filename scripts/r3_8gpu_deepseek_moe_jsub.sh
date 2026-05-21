#!/bin/bash
#JSUB -J r3_8gpu_dmoe
#JSUB -q gpu_6000ada
#JSUB -n 1
#JSUB -gpgpu 8
#JSUB -R span[ptile=8]
#JSUB -e error.%J
#JSUB -o output.%J

set -euo pipefail

SCRIPT_DIR="/data/gtq/latent-moe"
cd "$SCRIPT_DIR"

module load mpich/4.3.0b1-gcc-11.3.1
source activate trl

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export BASE_MODEL="${BASE_MODEL:-/data/gtq/latent-moe/models/Qwen2.5-1.5B}"

export LATENT_LAYER_TYPE="deepseek_moe"
export LATENT_EXPERIMENT_NAME="${LATENT_EXPERIMENT_NAME:-deepseek_moe}"
export PROMPT_STYLE="${PROMPT_STYLE:-default}"
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
export MOE_TOP_K="${MOE_TOP_K:-2}"
export MOE_EXPERT_HIDDEN_SIZE="${MOE_EXPERT_HIDDEN_SIZE:-0}"
export MOE_EXPERT_HIDDEN_MULTIPLIER="${MOE_EXPERT_HIDDEN_MULTIPLIER:-1.0}"
export MOE_DROPOUT="${MOE_DROPOUT:-0.0}"
export MOE_ROUTER_TEMPERATURE="${MOE_ROUTER_TEMPERATURE:-1.0}"
export MOE_ROUTER_JITTER_NOISE="${MOE_ROUTER_JITTER_NOISE:-0.0}"
export MOE_FINE_GRAINED_FACTOR="${MOE_FINE_GRAINED_FACTOR:-2}"
export MOE_NUM_SHARED_EXPERTS="${MOE_NUM_SHARED_EXPERTS:-1}"
export MOE_SHARED_EXPERT_WEIGHT="${MOE_SHARED_EXPERT_WEIGHT:-1.0}"

mkdir -p ./logs
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="./logs/r3_8gpu_deepseek_moe_four_datasets_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
section() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

section "R3 8-GPU DeepSeek-MoE Four-Dataset JSUB Job"
log "Host                 : $(hostname)"
log "Work directory       : $SCRIPT_DIR"
log "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
log "Base model           : $BASE_MODEL"
log "Latent layer type    : $LATENT_LAYER_TYPE"
log "Experiment name      : $LATENT_EXPERIMENT_NAME"
log "Prompt style         : $PROMPT_STYLE"
log "MOE_NUM_EXPERTS      : $MOE_NUM_EXPERTS"
log "MOE_TOP_K            : $MOE_TOP_K"
log "Log file             : $LOG_FILE"
log "Start time           : $(date '+%Y-%m-%d %H:%M:%S')"

bash ./scripts/latent_train_four_datasets.sh "$@"

section "Job Finished"
log "End time             : $(date '+%Y-%m-%d %H:%M:%S')"

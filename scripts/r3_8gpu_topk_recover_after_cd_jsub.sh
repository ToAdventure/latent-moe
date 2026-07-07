#!/bin/bash
#JSUB -J r3_8gpu_topk_recover
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

export LATENT_LAYER_TYPE="topk_moe"
export LATENT_EXPERIMENT_NAME="${LATENT_EXPERIMENT_NAME:-topk_moe}"
export PROMPT_STYLE="${PROMPT_STYLE:-default}"
export MOE_EXPERT_TYPE="${MOE_EXPERT_TYPE:-mlp}"
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
export MOE_TOP_K="${MOE_TOP_K:-2}"
export MOE_EXPERT_HIDDEN_SIZE="${MOE_EXPERT_HIDDEN_SIZE:-0}"
export MOE_EXPERT_HIDDEN_MULTIPLIER="${MOE_EXPERT_HIDDEN_MULTIPLIER:-1.0}"
export MOE_DROPOUT="${MOE_DROPOUT:-0.0}"
export MOE_ROUTER_TEMPERATURE="${MOE_ROUTER_TEMPERATURE:-1.0}"
export MOE_ROUTER_JITTER_NOISE="${MOE_ROUTER_JITTER_NOISE:-0.0}"
export MOE_LOAD_BALANCE_WEIGHT="${MOE_LOAD_BALANCE_WEIGHT:-0.01}"
export MOE_BALANCE_LOSS_MODE="${MOE_BALANCE_LOSS_MODE:-fixed}"

mkdir -p ./logs
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="./logs/r3_8gpu_topk_moe_recover_after_cd_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
section() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

section "R3 8-GPU Top-K MoE Recovery Job"
log "Host                 : $(hostname)"
log "Work directory       : $SCRIPT_DIR"
log "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
log "Base model           : $BASE_MODEL"
log "Experiment name      : $LATENT_EXPERIMENT_NAME"
log "MOE_EXPERT_TYPE      : $MOE_EXPERT_TYPE"
log "MOE_BALANCE_MODE     : $MOE_BALANCE_LOSS_MODE"
log "Prompt style         : $PROMPT_STYLE"
log "Log file             : $LOG_FILE"
log "Start time           : $(date '+%Y-%m-%d %H:%M:%S')"

section "Recover Finished Dataset Eval: CDs_and_Vinyl"
bash ./scripts/latent_eval_dataset.sh "CDs_and_Vinyl"

section "Continue Remaining Top-K MoE Datasets"
export LATENT_CATEGORIES="Musical_Instruments Toys_and_Games Video_Games"
bash ./scripts/latent_train_four_datasets.sh "$@"

section "Recovery Job Finished"
log "End time             : $(date '+%Y-%m-%d %H:%M:%S')"

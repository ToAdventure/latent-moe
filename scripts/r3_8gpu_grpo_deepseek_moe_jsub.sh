#!/bin/bash
#JSUB -J r3_grpo_dmoe
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
export GRPO_CATEGORIES="${GRPO_CATEGORIES:-CDs_and_Vinyl Musical_Instruments Toys_and_Games Video_Games}"
export GRPO_BASE_EXPERIMENT="${GRPO_BASE_EXPERIMENT:-deepseek_moe}"
export GRPO_EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-grpo_listwise_deepseek_moe}"
export GRPO_LRS="${GRPO_LRS:-1e-6}"

export LATENT_LAYER_TYPE="${LATENT_LAYER_TYPE:-deepseek_moe}"
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
export MOE_TOP_K="${MOE_TOP_K:-2}"
export MOE_EXPERT_HIDDEN_SIZE="${MOE_EXPERT_HIDDEN_SIZE:-0}"
export MOE_EXPERT_HIDDEN_MULTIPLIER="${MOE_EXPERT_HIDDEN_MULTIPLIER:-1.0}"
export MOE_DROPOUT="${MOE_DROPOUT:-0.0}"
export MOE_ROUTER_TEMPERATURE="${MOE_ROUTER_TEMPERATURE:-1.0}"
export MOE_ROUTER_JITTER_NOISE="${MOE_ROUTER_JITTER_NOISE:-0.0}"
export MOE_LOAD_BALANCE_WEIGHT="${MOE_LOAD_BALANCE_WEIGHT:-0.0}"
export MOE_FINE_GRAINED_FACTOR="${MOE_FINE_GRAINED_FACTOR:-1}"
export MOE_NUM_SHARED_EXPERTS="${MOE_NUM_SHARED_EXPERTS:-1}"
export MOE_SHARED_EXPERT_WEIGHT="${MOE_SHARED_EXPERT_WEIGHT:-1.0}"

export USE_FSDP2="${USE_FSDP2:-False}"
export GRPO_BETA="${GRPO_BETA:-0.0}"
export RUN_GRPO_EVAL="${RUN_GRPO_EVAL:-true}"
export LISTWISE_CANDIDATE_CHUNK_SIZE="${LISTWISE_CANDIDATE_CHUNK_SIZE:-8}"

mkdir -p ./logs
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DATASET_LABEL="${GRPO_CATEGORIES}"
RUN_DATASET_LABEL="${RUN_DATASET_LABEL// /_}"
LOG_FILE="./logs/r3_8gpu_grpo_deepseek_moe_${RUN_DATASET_LABEL}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
section() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

section "R3 8-GPU GRPO DeepSeek-MoE ${RUN_DATASET_LABEL} JSUB Job"
log "Host                 : $(hostname)"
log "Work directory       : $SCRIPT_DIR"
log "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
log "Datasets             : $GRPO_CATEGORIES"
log "Base experiment      : $GRPO_BASE_EXPERIMENT"
log "Experiment name      : $GRPO_EXPERIMENT_NAME"
log "Learning rates       : $GRPO_LRS"
log "Latent layer type    : $LATENT_LAYER_TYPE"
log "MOE_NUM_EXPERTS      : $MOE_NUM_EXPERTS"
log "MOE_TOP_K            : $MOE_TOP_K"
log "MOE_FINE_GRAINED     : $MOE_FINE_GRAINED_FACTOR"
log "MOE_NUM_SHARED       : $MOE_NUM_SHARED_EXPERTS"
log "USE_FSDP2            : $USE_FSDP2"
log "GRPO_BETA            : $GRPO_BETA"
log "RUN_GRPO_EVAL        : $RUN_GRPO_EVAL"
log "LISTWISE_CHUNK_SIZE  : $LISTWISE_CANDIDATE_CHUNK_SIZE"
log "Log file             : $LOG_FILE"
log "Start time           : $(date '+%Y-%m-%d %H:%M:%S')"

bash ./scripts/grpo_train_four_datasets.sh "$@"

section "Job Finished"
log "End time             : $(date '+%Y-%m-%d %H:%M:%S')"

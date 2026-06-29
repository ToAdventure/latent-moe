#!/bin/bash
#JSUB -J r3_grpo_topk
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
export GRPO_BASE_EXPERIMENT="${GRPO_BASE_EXPERIMENT:-topk_moe}"
export GRPO_EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-grpo_listwise_topk_moe}"
export GRPO_LRS="${GRPO_LRS:-1e-6}"

export LATENT_LAYER_TYPE="${LATENT_LAYER_TYPE:-topk_moe}"
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
export MOE_TOP_K="${MOE_TOP_K:-2}"
export MOE_EXPERT_HIDDEN_SIZE="${MOE_EXPERT_HIDDEN_SIZE:-0}"
export MOE_EXPERT_HIDDEN_MULTIPLIER="${MOE_EXPERT_HIDDEN_MULTIPLIER:-1.0}"
export MOE_DROPOUT="${MOE_DROPOUT:-0.0}"
export MOE_ROUTER_TEMPERATURE="${MOE_ROUTER_TEMPERATURE:-1.0}"
export MOE_ROUTER_JITTER_NOISE="${MOE_ROUTER_JITTER_NOISE:-0.0}"
export MOE_LOAD_BALANCE_WEIGHT="${MOE_LOAD_BALANCE_WEIGHT:-0.0}"
export MOE_FINE_GRAINED_FACTOR="${MOE_FINE_GRAINED_FACTOR:-1}"
export MOE_NUM_SHARED_EXPERTS="${MOE_NUM_SHARED_EXPERTS:-0}"
export MOE_SHARED_EXPERT_WEIGHT="${MOE_SHARED_EXPERT_WEIGHT:-0.0}"

export USE_FSDP2="${USE_FSDP2:-False}"
export GRPO_BETA="${GRPO_BETA:-0.0}"
export RUN_GRPO_EVAL="${RUN_GRPO_EVAL:-true}"
export LISTWISE_CANDIDATE_CHUNK_SIZE="${LISTWISE_CANDIDATE_CHUNK_SIZE:-8}"
export REWARD_BACKEND="${REWARD_BACKEND:-auto}"
export FLR_REWARD_W_LOGP="${FLR_REWARD_W_LOGP:-0.1}"
export FLR_REWARD_W_EXACT="${FLR_REWARD_W_EXACT:-1.0}"
export LISTWISE_NEGATIVE_STRATEGY="${LISTWISE_NEGATIVE_STRATEGY:-random}"
export LISTWISE_HARD_NEGATIVE_POOL_SIZE="${LISTWISE_HARD_NEGATIVE_POOL_SIZE:-64}"
export ON_POLICY_NUM_BEAMS="${ON_POLICY_NUM_BEAMS:-1}"
export ON_POLICY_TEMPERATURE="${ON_POLICY_TEMPERATURE:-1.0}"
export ON_POLICY_REWARD_TOP_KS="${ON_POLICY_REWARD_TOP_KS:-5,10}"
export DISCARD_ZERO_ADVANTAGE_GROUPS="${DISCARD_ZERO_ADVANTAGE_GROUPS:-true}"
export METRIC_TOP_KS="${METRIC_TOP_KS:-}"
export SANITY_CHECK_GENERATION="${SANITY_CHECK_GENERATION:-false}"
export GENERATION_EVAL_STEPS="${GENERATION_EVAL_STEPS:-0}"

mkdir -p ./logs
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DATASET_LABEL="${GRPO_CATEGORIES}"
RUN_DATASET_LABEL="${RUN_DATASET_LABEL// /_}"
LOG_FILE="./logs/r3_8gpu_grpo_topk_moe_${RUN_DATASET_LABEL}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
section() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

section "R3 8-GPU GRPO TopK-MoE ${RUN_DATASET_LABEL} JSUB Job"
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
log "MOE_DROPOUT          : $MOE_DROPOUT"
log "MOE_LOAD_BALANCE_WT  : $MOE_LOAD_BALANCE_WEIGHT"
log "USE_FSDP2            : $USE_FSDP2"
log "GRPO_BETA            : $GRPO_BETA"
log "RUN_GRPO_EVAL        : $RUN_GRPO_EVAL"
log "LISTWISE_CHUNK_SIZE  : $LISTWISE_CANDIDATE_CHUNK_SIZE"
log "REWARD_BACKEND       : $REWARD_BACKEND"
log "FLR_REWARD_W_LOGP    : $FLR_REWARD_W_LOGP"
log "FLR_REWARD_W_EXACT   : $FLR_REWARD_W_EXACT"
log "LISTWISE_NEG_STRATEGY: $LISTWISE_NEGATIVE_STRATEGY"
log "HARD_NEG_POOL_SIZE   : $LISTWISE_HARD_NEGATIVE_POOL_SIZE"
log "ON_POLICY_NUM_BEAMS  : $ON_POLICY_NUM_BEAMS"
log "ON_POLICY_TEMP       : $ON_POLICY_TEMPERATURE"
log "ON_POLICY_TOP_KS     : $ON_POLICY_REWARD_TOP_KS"
log "METRIC_TOP_KS        : $METRIC_TOP_KS"
log "SANITY_CHECK_GEN     : $SANITY_CHECK_GENERATION"
log "GENERATION_EVAL_STEPS: $GENERATION_EVAL_STEPS"
log "Log file             : $LOG_FILE"
log "Start time           : $(date '+%Y-%m-%d %H:%M:%S')"

bash ./scripts/grpo_train_four_datasets.sh "$@"

section "Job Finished"
log "End time             : $(date '+%Y-%m-%d %H:%M:%S')"

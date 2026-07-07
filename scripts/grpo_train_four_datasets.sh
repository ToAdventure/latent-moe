#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

# Keep Hugging Face caches off $HOME so large downloads land under /data.
: "${HF_HOME:=/data/gtq/latent-moe/.cache/huggingface}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

: "${GPU_IDS:=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
: "${GRPO_CATEGORIES:=CDs_and_Vinyl Musical_Instruments Toys_and_Games Video_Games}"
: "${GRPO_BASE_EXPERIMENT:=deepseek_moe}"
: "${GRPO_BASE_MODEL:=}"
: "${GRPO_OUTPUT_ROOT:=./output_dir}"
: "${GRPO_EXPERIMENT_NAME:=grpo_listwise_deepseek_moe}"
: "${GRPO_LRS:=1e-6}"
: "${GRPO_SAMPLE:=-1}"
: "${GRPO_NUM_EPOCHS:=1}"
: "${GRPO_BATCH_SIZE:=256}"
: "${GRPO_MICRO_BATCH_SIZE:=8}"
: "${GRPO_CUTOFF_LEN:=512}"
: "${GRPO_NUM_GENERATIONS:=8}"
: "${GRPO_BETA:=0.0}"
: "${GRPO_NUM_ITERATIONS:=1}"
: "${GRPO_EPSILON:=0.2}"
: "${GRPO_EPSILON_HIGH:=0.28}"
: "${GRPO_MAX_COMPLETION_LENGTH:=256}"
: "${LOAD_BEST_MODEL_AT_END:=true}"
: "${SAVE_STRATEGY:=best}"
: "${SAVE_TOTAL_LIMIT:=1}"
: "${SAVE_ONLY_MODEL:=true}"
: "${SAVE_FINAL_MODEL:=false}"
: "${BEST_MODEL_METRIC:=hr5_ndcg5}"
: "${METRIC_EVAL_SAMPLE:=-1}"
: "${METRIC_EVAL_BATCH_SIZE:=4}"
: "${METRIC_NUM_BEAMS:=10}"
: "${METRIC_TOP_K:=5}"
: "${METRIC_TOP_KS:=}"
: "${METRIC_MAX_NEW_TOKENS:=128}"
: "${METRIC_TEMPERATURE:=1.0}"
: "${METRIC_GUIDANCE_SCALE:=1.0}"
: "${METRIC_LENGTH_PENALTY:=1.0}"
: "${METRIC_CUTOFF_LEN:=2560}"
: "${SANITY_CHECK_GENERATION:=false}"
: "${GENERATION_EVAL_STEPS:=0}"
: "${USE_LISTWISE_REWARD:=True}"
: "${NUM_LISTWISE_NEGATIVES:=15}"
: "${LISTWISE_CANDIDATE_CHUNK_SIZE:=8}"
: "${LISTWISE_TOPK:=10}"
: "${LISTWISE_TAU:=1.0}"
: "${REWARD_W_DCG:=1.0}"
: "${REWARD_W_SOFT:=0.5}"
: "${REWARD_W_MARGIN:=0.5}"
: "${REWARD_W_LOGP:=0.05}"
: "${REWARD_BACKEND:=auto}"
: "${FLR_REWARD_W_LOGP:=0.1}"
: "${FLR_REWARD_W_EXACT:=1.0}"
: "${LISTWISE_NEGATIVE_STRATEGY:=random}"
: "${LISTWISE_HARD_NEGATIVE_POOL_SIZE:=64}"
: "${ON_POLICY_NUM_BEAMS:=1}"
: "${ON_POLICY_TEMPERATURE:=1.0}"
: "${ON_POLICY_REWARD_TOP_KS:=5,10}"
: "${DISCARD_ZERO_ADVANTAGE_GROUPS:=true}"
: "${NOISE_STD:=1.0}"
: "${LATENT_LAYER_TYPE:=deepseek_moe}"
: "${MOE_EXPERT_TYPE:=auto}"
: "${MOE_NUM_EXPERTS:=auto}"
: "${MOE_TOP_K:=auto}"
: "${MOE_EXPERT_HIDDEN_SIZE:=auto}"
: "${MOE_EXPERT_HIDDEN_MULTIPLIER:=auto}"
: "${MOE_DROPOUT:=auto}"
: "${MOE_ROUTER_TEMPERATURE:=auto}"
: "${MOE_ROUTER_JITTER_NOISE:=auto}"
: "${MOE_LOAD_BALANCE_WEIGHT:=auto}"
: "${MOE_BALANCE_LOSS_MODE:=auto}"
: "${MOE_FINE_GRAINED_FACTOR:=auto}"
: "${MOE_NUM_SHARED_EXPERTS:=auto}"
: "${MOE_SHARED_EXPERT_WEIGHT:=auto}"
: "${USE_FSDP2:=False}"
: "${FSDP2_ACTIVATION_CHECKPOINTING:=False}"
: "${FSDP2_STATE_DICT_TYPE:=FULL_STATE_DICT}"
: "${RUN_GRPO_EVAL:=true}"
: "${GRPO_EVAL_BATCH_SIZE:=12}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
TRAIN_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=$(( $(tr -cd ',' <<< "${CUDA_VISIBLE_DEVICES}" | wc -c) + 1 ))
if (( NUM_GPUS < 1 )); then
    echo "No GPU selected. Set GPU_IDS or CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

read -r -a CATEGORIES <<< "${GRPO_CATEGORIES}"
read -r -a LRS <<< "${GRPO_LRS}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
EXTRA_TRAIN_ARGS=("$@")

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with NUM_GPUS=${NUM_GPUS}"
echo "Using GRPO_CATEGORIES=${CATEGORIES[*]}"
echo "Using GRPO_BASE_EXPERIMENT=${GRPO_BASE_EXPERIMENT}"
if [[ -n "${GRPO_BASE_MODEL}" ]]; then
    echo "Using fixed GRPO_BASE_MODEL=${GRPO_BASE_MODEL}"
fi
echo "Using GRPO_EXPERIMENT_NAME=${GRPO_EXPERIMENT_NAME}"
echo "Using GRPO_LRS=${LRS[*]}"
echo "Using LATENT_LAYER_TYPE=${LATENT_LAYER_TYPE}"
echo "Using MOE_EXPERT_TYPE=${MOE_EXPERT_TYPE}"
echo "Using MOE_NUM_EXPERTS=${MOE_NUM_EXPERTS}"
echo "Using MOE_TOP_K=${MOE_TOP_K}"
echo "Using MOE_FINE_GRAINED_FACTOR=${MOE_FINE_GRAINED_FACTOR}"
echo "Using MOE_NUM_SHARED_EXPERTS=${MOE_NUM_SHARED_EXPERTS}"
echo "Using MOE_LOAD_BALANCE_WEIGHT=${MOE_LOAD_BALANCE_WEIGHT}"
echo "Using MOE_BALANCE_LOSS_MODE=${MOE_BALANCE_LOSS_MODE}"
echo "Using USE_LISTWISE_REWARD=${USE_LISTWISE_REWARD}"
echo "Using REWARD_BACKEND=${REWARD_BACKEND}"
echo "Using FLR_REWARD_W_LOGP=${FLR_REWARD_W_LOGP}"
echo "Using FLR_REWARD_W_EXACT=${FLR_REWARD_W_EXACT}"
echo "Using LISTWISE_NEGATIVE_STRATEGY=${LISTWISE_NEGATIVE_STRATEGY}"
echo "Using LISTWISE_HARD_NEGATIVE_POOL_SIZE=${LISTWISE_HARD_NEGATIVE_POOL_SIZE}"
echo "Using ON_POLICY_NUM_BEAMS=${ON_POLICY_NUM_BEAMS}"
echo "Using ON_POLICY_TEMPERATURE=${ON_POLICY_TEMPERATURE}"
echo "Using ON_POLICY_REWARD_TOP_KS=${ON_POLICY_REWARD_TOP_KS}"
echo "Using DISCARD_ZERO_ADVANTAGE_GROUPS=${DISCARD_ZERO_ADVANTAGE_GROUPS}"
echo "Using NUM_LISTWISE_NEGATIVES=${NUM_LISTWISE_NEGATIVES}"
echo "Using LISTWISE_CANDIDATE_CHUNK_SIZE=${LISTWISE_CANDIDATE_CHUNK_SIZE}"
echo "Using LISTWISE_TOPK=${LISTWISE_TOPK}"
echo "Using SAVE_STRATEGY=${SAVE_STRATEGY}"
echo "Using SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "Using SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL}"
echo "Using SAVE_FINAL_MODEL=${SAVE_FINAL_MODEL}"
echo "Using BEST_MODEL_METRIC=${BEST_MODEL_METRIC}"
echo "Using METRIC_EVAL_SAMPLE=${METRIC_EVAL_SAMPLE}"
echo "Using METRIC_EVAL_BATCH_SIZE=${METRIC_EVAL_BATCH_SIZE}"
echo "Using METRIC_NUM_BEAMS=${METRIC_NUM_BEAMS}"
echo "Using METRIC_TOP_K=${METRIC_TOP_K}"
echo "Using METRIC_TOP_KS=${METRIC_TOP_KS}"
echo "Using SANITY_CHECK_GENERATION=${SANITY_CHECK_GENERATION}"
echo "Using GENERATION_EVAL_STEPS=${GENERATION_EVAL_STEPS}"
echo "Using USE_FSDP2=${USE_FSDP2}"
echo "Using RUN_GRPO_EVAL=${RUN_GRPO_EVAL}"

is_true() {
    case "${1,,}" in
        1|true|yes|y) return 0 ;;
        *) return 1 ;;
    esac
}

select_model_ckpt() {
    python - "$1" <<'PY'
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
out_abs = out.resolve()

def has_model_files(path):
    return (
        (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
        or (path / "model.safetensors.index.json").exists()
    )

def existing_checkpoint(raw):
    if not raw:
        return None
    p = Path(raw)
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([Path.cwd() / p, out_abs / p.name])
    for candidate in candidates:
        if candidate.exists() and has_model_files(candidate):
            return candidate
    return None

if has_model_files(out_abs):
    print(out_abs)
    raise SystemExit(0)

states = sorted(out.glob("checkpoint-*/trainer_state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
for state_path in states:
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        continue
    best = existing_checkpoint(state.get("best_model_checkpoint"))
    if best is not None:
        print(best)
        raise SystemExit(0)

def step_num(path):
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1

checkpoints = [p for p in out.glob("checkpoint-*") if p.is_dir() and has_model_files(p)]
checkpoints = sorted(checkpoints, key=step_num, reverse=True)
if checkpoints:
    print(checkpoints[0].resolve())
    raise SystemExit(0)

raise SystemExit(f"No model checkpoint found under {out_abs}")
PY
}

resolve_base_model() {
    local category="$1"
    if [[ -n "${GRPO_BASE_MODEL}" ]]; then
        echo "${GRPO_BASE_MODEL}"
        return
    fi

    local base_dir="${GRPO_OUTPUT_ROOT}/${category}/${GRPO_BASE_EXPERIMENT}"
    if [[ ! -d "${base_dir}" ]]; then
        echo "Base experiment directory does not exist: ${base_dir}" >&2
        exit 1
    fi
    select_model_ckpt "${base_dir}"
}

run_eval() {
    local category="$1"
    local model_ckpt="$2"
    local out="$3"
    local lr="$4"
    local test_file="$5"
    local info_file="$6"

    local eval_split_dir="${GRPO_OUTPUT_ROOT}/${category}_${GRPO_EXPERIMENT_NAME}_lr_${lr}_base"
    mkdir -p "${eval_split_dir}"
    python src/utils/split.py --input_path "${test_file}" \
        --output_path "${eval_split_dir}" --nn "${NUM_GPUS}"

    local eval_pids=()
    for (( i=0; i<NUM_GPUS; i++ ))
    do
        export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$i]}"
        echo "${eval_split_dir}/${i}.csv"
        python src/grpo_attention_tuning/noise_eval.py \
            --batch_size "${GRPO_EVAL_BATCH_SIZE}" --base_model "${model_ckpt}" --sample -1 \
            --info_file "${info_file}" --category "${category}" --end_k -1 \
            --test_data_path "${eval_split_dir}/${i}.csv" \
            --result_json_data "${eval_split_dir}/${i}.json" \
            > "${eval_split_dir}/${i}.eval.out" \
            2> "${eval_split_dir}/${i}.eval.err" &
        eval_pids+=("$!")
    done

    local eval_failed=0
    for i in "${!eval_pids[@]}"; do
        local pid="${eval_pids[$i]}"
        if ! wait "${pid}"; then
            eval_failed=1
            echo "Eval worker ${i} failed for ${category}; see ${eval_split_dir}/${i}.eval.err" >&2
        elif [[ ! -s "${eval_split_dir}/${i}.json" ]]; then
            eval_failed=1
            echo "Eval worker ${i} finished but did not create ${eval_split_dir}/${i}.json" >&2
        fi
    done
    if (( eval_failed != 0 )); then
        echo "One or more eval workers failed for ${category}; skip merge and exit." >&2
        exit 1
    fi

    export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
    python src/utils/merge.py --input_path "${eval_split_dir}/" \
        --output_path "${out}/final_result.json" --nn "${NUM_GPUS}"
    python src/utils/calc.py --path "${out}/final_result.json" \
        --item_path "${info_file}"
}

for category in "${CATEGORIES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  GRPO ${GRPO_EXPERIMENT_NAME}: ${category}"
    echo "============================================================"

    train_file=$(ls -f data/train/${category}*11.csv)
    eval_file=$(ls -f data/valid/${category}*11.csv)
    test_file=$(ls -f data/test/${category}*11.csv)
    info_file=$(ls -f data/info/${category}*.txt)
    base_model="$(resolve_base_model "${category}")"
    echo "Train file : ${train_file}"
    echo "Valid file : ${eval_file}"
    echo "Test file  : ${test_file}"
    echo "Info file  : ${info_file}"
    echo "Base model : ${base_model}"

    for lr in "${LRS[@]}"; do
        port=$((RANDOM % 5000 + 10000))
        task_name="grpo_${GRPO_EXPERIMENT_NAME}_${category}_lr_${lr}"
        out="${GRPO_OUTPUT_ROOT}/${category}/${GRPO_EXPERIMENT_NAME}_lr_${lr}"
        mkdir -p "${out}"

        torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port="${port}" \
            src/grpo_attention_tuning/train_noise_grpo.py \
            --category "${category}" \
            --base_model "${base_model}" \
            --train_file "${train_file}" \
            --eval_file "${eval_file}" \
            --info_file "${info_file}" \
            --output_dir "${out}" \
            --sample "${GRPO_SAMPLE}" \
            --num_epochs "${GRPO_NUM_EPOCHS}" \
            --batch_size "${GRPO_BATCH_SIZE}" \
            --micro_batch_size "${GRPO_MICRO_BATCH_SIZE}" \
            --cutoff_len "${GRPO_CUTOFF_LEN}" \
            --lr "${lr}" \
            --seed 42 \
            --num_generations "${GRPO_NUM_GENERATIONS}" \
            --beta "${GRPO_BETA}" \
            --num_iterations "${GRPO_NUM_ITERATIONS}" \
            --epsilon "${GRPO_EPSILON}" \
            --epsilon_high "${GRPO_EPSILON_HIGH}" \
            --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH}" \
            --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
            --save_strategy "${SAVE_STRATEGY}" \
            --save_total_limit "${SAVE_TOTAL_LIMIT}" \
            --save_only_model "${SAVE_ONLY_MODEL}" \
            --save_final_model "${SAVE_FINAL_MODEL}" \
            --best_model_metric "${BEST_MODEL_METRIC}" \
            --metric_eval_sample "${METRIC_EVAL_SAMPLE}" \
            --metric_eval_batch_size "${METRIC_EVAL_BATCH_SIZE}" \
            --metric_num_beams "${METRIC_NUM_BEAMS}" \
            --metric_top_k "${METRIC_TOP_K}" \
            --metric_top_ks "${METRIC_TOP_KS}" \
            --metric_max_new_tokens "${METRIC_MAX_NEW_TOKENS}" \
            --metric_temperature "${METRIC_TEMPERATURE}" \
            --metric_guidance_scale "${METRIC_GUIDANCE_SCALE}" \
            --metric_length_penalty "${METRIC_LENGTH_PENALTY}" \
            --metric_cutoff_len "${METRIC_CUTOFF_LEN}" \
            --sanity_check_generation "${SANITY_CHECK_GENERATION}" \
            --generation_eval_steps "${GENERATION_EVAL_STEPS}" \
            --latent_layer_type "${LATENT_LAYER_TYPE}" \
            --moe_expert_type "${MOE_EXPERT_TYPE}" \
            --moe_num_experts "${MOE_NUM_EXPERTS}" \
            --moe_top_k "${MOE_TOP_K}" \
            --moe_expert_hidden_size "${MOE_EXPERT_HIDDEN_SIZE}" \
            --moe_expert_hidden_multiplier "${MOE_EXPERT_HIDDEN_MULTIPLIER}" \
            --moe_dropout "${MOE_DROPOUT}" \
            --moe_router_temperature "${MOE_ROUTER_TEMPERATURE}" \
            --moe_router_jitter_noise "${MOE_ROUTER_JITTER_NOISE}" \
            --moe_load_balance_weight "${MOE_LOAD_BALANCE_WEIGHT}" \
            --moe_balance_loss_mode "${MOE_BALANCE_LOSS_MODE}" \
            --moe_fine_grained_factor "${MOE_FINE_GRAINED_FACTOR}" \
            --moe_num_shared_experts "${MOE_NUM_SHARED_EXPERTS}" \
            --moe_shared_expert_weight "${MOE_SHARED_EXPERT_WEIGHT}" \
            --use_listwise_reward "${USE_LISTWISE_REWARD}" \
            --num_listwise_negatives "${NUM_LISTWISE_NEGATIVES}" \
            --listwise_candidate_chunk_size "${LISTWISE_CANDIDATE_CHUNK_SIZE}" \
            --listwise_topk "${LISTWISE_TOPK}" \
            --listwise_tau "${LISTWISE_TAU}" \
            --reward_w_dcg "${REWARD_W_DCG}" \
            --reward_w_soft "${REWARD_W_SOFT}" \
            --reward_w_margin "${REWARD_W_MARGIN}" \
            --reward_w_logp "${REWARD_W_LOGP}" \
            --reward_backend "${REWARD_BACKEND}" \
            --flr_reward_w_logp "${FLR_REWARD_W_LOGP}" \
            --flr_reward_w_exact "${FLR_REWARD_W_EXACT}" \
            --listwise_negative_strategy "${LISTWISE_NEGATIVE_STRATEGY}" \
            --listwise_hard_negative_pool_size "${LISTWISE_HARD_NEGATIVE_POOL_SIZE}" \
            --on_policy_num_beams "${ON_POLICY_NUM_BEAMS}" \
            --on_policy_temperature "${ON_POLICY_TEMPERATURE}" \
            --on_policy_reward_top_ks "${ON_POLICY_REWARD_TOP_KS}" \
            --discard_zero_advantage_groups "${DISCARD_ZERO_ADVANTAGE_GROUPS}" \
            --noise_std "${NOISE_STD}" \
            --use_fsdp2 "${USE_FSDP2}" \
            --fsdp2_activation_checkpointing "${FSDP2_ACTIVATION_CHECKPOINTING}" \
            --fsdp2_state_dict_type "${FSDP2_STATE_DICT_TYPE}" \
            "${EXTRA_TRAIN_ARGS[@]}" \
            > >(tee -a "${out}/${task_name}.log") 2> >(tee -a "${out}/${task_name}.err" >&2)

        if is_true "${RUN_GRPO_EVAL}"; then
            model_ckpt="$(select_model_ckpt "${out}")"
            echo "Using model checkpoint for eval: ${model_ckpt}"
            run_eval "${category}" "${model_ckpt}" "${out}" "${lr}" "${test_file}" "${info_file}"
        fi

        export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
    done
done

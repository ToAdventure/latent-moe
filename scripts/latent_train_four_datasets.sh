#!/usr/bin/env bash
set -euo pipefail

# Keep Hugging Face caches off $HOME so large downloads land under /data.
: "${HF_HOME:=/data/gtq/latent-moe/.cache/huggingface}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

: "${BASE_MODEL:=/data/gtq/latent-moe/models/Qwen2.5-1.5B}"
: "${GPU_IDS:=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
: "${LATENT_LAYER_TYPE:=topk_moe}"
: "${LATENT_EXPERIMENT_NAME:=${LATENT_LAYER_TYPE}}"
: "${PROMPT_STYLE:=default}"
: "${MOE_NUM_EXPERTS:=4}"
: "${MOE_TOP_K:=2}"
: "${MOE_EXPERT_HIDDEN_SIZE:=0}"
: "${MOE_EXPERT_HIDDEN_MULTIPLIER:=1.0}"
: "${MOE_DROPOUT:=0.0}"
: "${MOE_ROUTER_TEMPERATURE:=1.0}"
: "${MOE_ROUTER_JITTER_NOISE:=0.0}"
: "${MOE_FINE_GRAINED_FACTOR:=2}"
: "${MOE_NUM_SHARED_EXPERTS:=1}"
: "${MOE_SHARED_EXPERT_WEIGHT:=1.0}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
TRAIN_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=$(( $(tr -cd ',' <<< "${CUDA_VISIBLE_DEVICES}" | wc -c) + 1 ))
if (( NUM_GPUS != 8 )); then
    echo "This script expects 8 GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NUM_GPUS})." >&2
    exit 1
fi

CATEGORIES=(
    "CDs_and_Vinyl"
    "Musical_Instruments"
    "Toys_and_Games"
    "Video_Games"
)

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"

echo "Using BASE_MODEL=${BASE_MODEL}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with NUM_GPUS=${NUM_GPUS}"
echo "Using LATENT_LAYER_TYPE=${LATENT_LAYER_TYPE}"
echo "Using LATENT_EXPERIMENT_NAME=${LATENT_EXPERIMENT_NAME}"
echo "Using PROMPT_STYLE=${PROMPT_STYLE}"
echo "Using MOE_NUM_EXPERTS=${MOE_NUM_EXPERTS}"
echo "Using MOE_TOP_K=${MOE_TOP_K}"
if [[ "${LATENT_LAYER_TYPE}" == "deepseek_moe" ]]; then
    echo "Using MOE_FINE_GRAINED_FACTOR=${MOE_FINE_GRAINED_FACTOR}"
    echo "Using MOE_NUM_SHARED_EXPERTS=${MOE_NUM_SHARED_EXPERTS}"
    echo "Using MOE_SHARED_EXPERT_WEIGHT=${MOE_SHARED_EXPERT_WEIGHT}"
fi
echo "Datasets: ${CATEGORIES[*]}"

MOE_ARGS=(
    --moe_num_experts "${MOE_NUM_EXPERTS}"
    --moe_top_k "${MOE_TOP_K}"
    --moe_expert_hidden_size "${MOE_EXPERT_HIDDEN_SIZE}"
    --moe_expert_hidden_multiplier "${MOE_EXPERT_HIDDEN_MULTIPLIER}"
    --moe_dropout "${MOE_DROPOUT}"
    --moe_router_temperature "${MOE_ROUTER_TEMPERATURE}"
    --moe_router_jitter_noise "${MOE_ROUTER_JITTER_NOISE}"
)

if [[ "${LATENT_LAYER_TYPE}" == "deepseek_moe" ]]; then
    MOE_ARGS+=(
        --moe_fine_grained_factor "${MOE_FINE_GRAINED_FACTOR}"
        --moe_num_shared_experts "${MOE_NUM_SHARED_EXPERTS}"
        --moe_shared_expert_weight "${MOE_SHARED_EXPERT_WEIGHT}"
    )
fi

for category in "${CATEGORIES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Latent ${LATENT_EXPERIMENT_NAME}: ${category}"
    echo "============================================================"

    tdir=test
    train_file=$(ls -f data/train/${category}*11.csv)
    eval_file=$(ls -f data/valid/${category}*11.csv)
    test_file=$(ls -f data/${tdir}/${category}*11.csv)
    info_file=$(ls -f data/info/${category}*.txt)
    echo "${train_file} ${test_file} ${info_file} ${eval_file}"

    port=$((RANDOM % 5000 + 10000))
    task_name="latent_${LATENT_EXPERIMENT_NAME}_${category}"
    out="./output_dir/${category}/${LATENT_EXPERIMENT_NAME}"
    mkdir -p "${out}"

    torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --master_port=${port} \
        src/latent/latent_attention_train.py \
        --base_model "${BASE_MODEL}" \
        --train_file "${train_file}" --eval_file "${eval_file}" \
        --output_dir "${out}" --category "${category}" --sample -1 \
        --prompt_style "${PROMPT_STYLE}" \
        --latent_layer_type "${LATENT_LAYER_TYPE}" \
        "${MOE_ARGS[@]}" \
        > >(tee -a "${out}/${task_name}.log") 2> >(tee -a "${out}/${task_name}.err" >&2)

    eval_split_dir="./output_dir/${category}_${LATENT_EXPERIMENT_NAME}_base"
    mkdir -p "${eval_split_dir}"
    python src/utils/split.py --input_path "${test_file}" \
        --output_path "${eval_split_dir}" --nn "${NUM_GPUS}"

    model_ckpt="${out}"
    for (( i=0; i<NUM_GPUS; i++ ))
    do
        export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$i]}"
        echo "${eval_split_dir}/${i}.csv"
        python src/latent/latent_attention_eval.py --batch_size 24 \
            --base_model "${model_ckpt}" --sample -1 \
            --info_file "${info_file}" --category "${category}" \
            --prompt_style "${PROMPT_STYLE}" \
            --test_data_path "${eval_split_dir}/${i}.csv" \
            --result_json_data "${eval_split_dir}/${i}.json" &
    done
    wait

    export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
    python src/utils/merge.py --input_path "${eval_split_dir}/" \
        --output_path "${model_ckpt}/final_result.json" --nn "${NUM_GPUS}"
    python src/utils/calc.py --path "${model_ckpt}/final_result.json" \
        --item_path "${info_file}"
done

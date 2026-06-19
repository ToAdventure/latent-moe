#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/data/gtq/latent-moe"
cd "${SCRIPT_DIR}"

: "${HF_HOME:=/data/gtq/latent-moe/.cache/huggingface}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

: "${GPU_IDS:=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
: "${LATENT_EXPERIMENT_NAME:=topk_moe}"
: "${PROMPT_STYLE:=default}"
: "${EVAL_BATCH_SIZE:=8}"

category="${1:?Usage: bash scripts/latent_eval_dataset.sh <Category>}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
EVAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=$(( $(tr -cd ',' <<< "${CUDA_VISIBLE_DEVICES}" | wc -c) + 1 ))
if (( NUM_GPUS != 8 )); then
    echo "This script expects 8 GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NUM_GPUS})." >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"

tdir=test
test_file=$(ls -f data/${tdir}/${category}*11.csv)
info_file=$(ls -f data/info/${category}*.txt)
model_ckpt="./output_dir/${category}/${LATENT_EXPERIMENT_NAME}"
eval_split_dir="./output_dir/${category}_${LATENT_EXPERIMENT_NAME}_base"

if [[ ! -d "${model_ckpt}" ]]; then
    echo "Missing trained checkpoint directory: ${model_ckpt}" >&2
    echo "Copy the completed training output back first, or rerun training for ${category}." >&2
    exit 1
fi

mkdir -p "${eval_split_dir}"

echo ""
echo "============================================================"
echo "  Eval ${LATENT_EXPERIMENT_NAME}: ${category}"
echo "============================================================"
echo "Checkpoint  : ${model_ckpt}"
echo "Test file   : ${test_file}"
echo "Info file   : ${info_file}"
echo "Split dir   : ${eval_split_dir}"

python src/utils/split.py --input_path "${test_file}" \
    --output_path "${eval_split_dir}" --nn "${NUM_GPUS}"

eval_pids=()
for (( i=0; i<NUM_GPUS; i++ ))
do
    export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$i]}"
    echo "${eval_split_dir}/${i}.csv"
    python src/latent/latent_attention_eval.py --batch_size "${EVAL_BATCH_SIZE}" \
        --base_model "${model_ckpt}" --sample -1 \
        --info_file "${info_file}" --category "${category}" \
        --prompt_style "${PROMPT_STYLE}" \
        --test_data_path "${eval_split_dir}/${i}.csv" \
        --result_json_data "${eval_split_dir}/${i}.json" \
        > "${eval_split_dir}/${i}.eval.out" \
        2> "${eval_split_dir}/${i}.eval.err" &
    eval_pids+=("$!")
done

eval_failed=0
for i in "${!eval_pids[@]}"; do
    pid="${eval_pids[$i]}"
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

export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"
python src/utils/merge.py --input_path "${eval_split_dir}/" \
    --output_path "${model_ckpt}/final_result.json" --nn "${NUM_GPUS}"
python src/utils/calc.py --path "${model_ckpt}/final_result.json" \
    --item_path "${info_file}"

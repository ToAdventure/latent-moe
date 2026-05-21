#!/usr/bin/env bash
set -euo pipefail

: "${GPU_IDS:=${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
EVAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=$(( $(tr -cd ',' <<< "${CUDA_VISIBLE_DEVICES}" | wc -c) + 1 ))
if (( NUM_GPUS < 1 )); then
    echo "No GPU selected. Set GPU_IDS or CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with NUM_GPUS=${NUM_GPUS}"

for category in "CDs_and_Vinyl"; do
    path_to_data=data
    test_file=$(ls -f ${path_to_data}/test/${category}*11.csv)
    info_file=$(ls -f ${path_to_data}/info/${category}*.txt)

    : "${MODEL_CKPT:=output_dir/${category}/latent}"
    : "${EVAL_OUTPUT_DIR:=./output_dir/${category}_eval_base}"
    mkdir -p "${EVAL_OUTPUT_DIR}"

    python src/utils/split.py --input_path "${test_file}" \
        --output_path "${EVAL_OUTPUT_DIR}" --nn "${NUM_GPUS}"

    for (( i=0; i<NUM_GPUS; i++ ))
    do
        export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$i]}"
        echo "${EVAL_OUTPUT_DIR}/${i}.csv"
        python src/grpo_attention_tuning/noise_eval.py \
            --batch_size 18 --base_model "${MODEL_CKPT}" --sample -1 \
            --info_file "${info_file}" --category "${category}" --end_k -1 \
            --test_data_path "${EVAL_OUTPUT_DIR}/${i}.csv" \
            --result_json_data "${EVAL_OUTPUT_DIR}/${i}.json" &
    done
    wait

    export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"
    python src/utils/merge.py --input_path "${EVAL_OUTPUT_DIR}/" \
        --output_path "${MODEL_CKPT}/final_result.json" --nn "${NUM_GPUS}"
    python src/utils/calc.py --path "${MODEL_CKPT}/final_result.json" \
        --item_path "${info_file}"
done

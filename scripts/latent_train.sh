#!/usr/bin/env bash
set -euo pipefail

# Keep Hugging Face caches off $HOME so large downloads land under /data.
: "${HF_HOME:=/data/gtq/latent-moe/.cache/huggingface}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# Override this from the shell if you place the model somewhere else.
: "${BASE_MODEL:=/data/gtq/latent-moe/models/Qwen2.5-1.5B}"
: "${GPU_IDS:=${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
: "${LATENT_LAYER_TYPE:=attention}"
: "${MOE_NUM_EXPERTS:=4}"
: "${MOE_TOP_K:=2}"
: "${MOE_EXPERT_HIDDEN_SIZE:=0}"
: "${MOE_EXPERT_HIDDEN_MULTIPLIER:=1.0}"
: "${MOE_DROPOUT:=0.0}"
: "${MOE_ROUTER_TEMPERATURE:=1.0}"
: "${MOE_ROUTER_JITTER_NOISE:=0.0}"
: "${MOE_LOAD_BALANCE_WEIGHT:=0.01}"
: "${MOE_FINE_GRAINED_FACTOR:=2}"
: "${MOE_NUM_SHARED_EXPERTS:=1}"
: "${MOE_SHARED_EXPERT_WEIGHT:=1.0}"
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
: "${METRIC_MAX_NEW_TOKENS:=128}"
: "${METRIC_CUTOFF_LEN:=2560}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
TRAIN_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=$(( $(tr -cd ',' <<< "${CUDA_VISIBLE_DEVICES}" | wc -c) + 1 ))
if (( NUM_GPUS < 1 )); then
    echo "No GPU selected. Set GPU_IDS or CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

# CDs_and_Vinyl
category="CDs_and_Vinyl"
tdir=test
train_file=$(ls -f data/train/${category}*11.csv)
eval_file=$(ls -f data/valid/${category}*11.csv)
test_file=$(ls -f data/${tdir}/${category}*11.csv)
info_file=$(ls -f data/info/${category}*.txt)
echo ${train_file} ${test_file} ${info_file} ${eval_file}
port=$((RANDOM % 5000 + 10000))
task_name="latent_${category}"
out=./output_dir/${category}/latent
mkdir -p ${out}

echo "Using BASE_MODEL=${BASE_MODEL}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with NUM_GPUS=${NUM_GPUS}"
echo "Using LATENT_LAYER_TYPE=${LATENT_LAYER_TYPE}"
echo "Using MOE_LOAD_BALANCE_WEIGHT=${MOE_LOAD_BALANCE_WEIGHT}"
echo "Using LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END}"
echo "Using SAVE_STRATEGY=${SAVE_STRATEGY}"
echo "Using SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "Using SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL}"
echo "Using SAVE_FINAL_MODEL=${SAVE_FINAL_MODEL}"
echo "Using BEST_MODEL_METRIC=${BEST_MODEL_METRIC}"
echo "Using METRIC_EVAL_SAMPLE=${METRIC_EVAL_SAMPLE}"
echo "Using METRIC_EVAL_BATCH_SIZE=${METRIC_EVAL_BATCH_SIZE}"
echo "Using METRIC_NUM_BEAMS=${METRIC_NUM_BEAMS}"
echo "Using METRIC_TOP_K=${METRIC_TOP_K}"
torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train.py \
    --base_model $BASE_MODEL \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} --sample -1 \
    --latent_layer_type ${LATENT_LAYER_TYPE} \
    --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_only_model "${SAVE_ONLY_MODEL}" \
    --save_final_model "${SAVE_FINAL_MODEL}" \
    --best_model_metric "${BEST_MODEL_METRIC}" \
    --metric_info_file "${info_file}" \
    --metric_eval_sample "${METRIC_EVAL_SAMPLE}" \
    --metric_eval_batch_size "${METRIC_EVAL_BATCH_SIZE}" \
    --metric_num_beams "${METRIC_NUM_BEAMS}" \
    --metric_top_k "${METRIC_TOP_K}" \
    --metric_max_new_tokens "${METRIC_MAX_NEW_TOKENS}" \
    --metric_cutoff_len "${METRIC_CUTOFF_LEN}" \
    --moe_num_experts ${MOE_NUM_EXPERTS} --moe_top_k ${MOE_TOP_K} \
    --moe_expert_hidden_size ${MOE_EXPERT_HIDDEN_SIZE} \
    --moe_expert_hidden_multiplier ${MOE_EXPERT_HIDDEN_MULTIPLIER} \
    --moe_dropout ${MOE_DROPOUT} \
    --moe_router_temperature ${MOE_ROUTER_TEMPERATURE} \
    --moe_router_jitter_noise ${MOE_ROUTER_JITTER_NOISE} \
    --moe_load_balance_weight ${MOE_LOAD_BALANCE_WEIGHT} \
    --moe_fine_grained_factor ${MOE_FINE_GRAINED_FACTOR} \
    --moe_num_shared_experts ${MOE_NUM_SHARED_EXPERTS} \
    --moe_shared_expert_weight ${MOE_SHARED_EXPERT_WEIGHT} \
    > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

IFS=',' read -r -a array <<< "${CUDA_VISIBLE_DEVICES}"
nn=${NUM_GPUS}
mkdir -p ./output_dir/${category}_base
python src/utils/split.py --input_path $test_file \
        --output_path ./output_dir/${category}_base --nn ${nn}

select_model_ckpt() {
    python - "$1" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out_abs = out.resolve()

def existing_checkpoint(raw):
    if not raw:
        return None
    p = Path(raw)
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([Path.cwd() / p, out_abs / p.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

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

if states:
    print(states[0].parent.resolve())
else:
    print(out_abs)
PY
}

out=./output_dir/${category}/latent
model_ckpt="$(select_model_ckpt "${out}")"
echo "Using model checkpoint for eval: ${model_ckpt}"
for (( i=0; i<$nn; i++ ))
do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    echo ${category}_base/${i}.csv
    python src/latent/latent_attention_eval.py --batch_size 24 \
        --base_model $model_ckpt --sample -1 \
        --info_file ${info_file} --category ${category} \
        --test_data_path ./output_dir/${category}_base/${i}.csv \
        --result_json_data ./output_dir/${category}_base/${i}.json &
done
wait
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
python src/utils/merge.py --input_path ./output_dir/${category}_base/ \
        --output_path ${out}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${out}/final_result.json \
        --item_path ${info_file}

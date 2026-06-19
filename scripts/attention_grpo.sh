#!/usr/bin/env bash
set -euo pipefail

: "${GPU_IDS:=${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
: "${LATENT_LAYER_TYPE:=auto}"
: "${MOE_NUM_EXPERTS:=4}"
: "${MOE_TOP_K:=2}"
: "${MOE_EXPERT_HIDDEN_SIZE:=0}"
: "${MOE_EXPERT_HIDDEN_MULTIPLIER:=1.0}"
: "${MOE_DROPOUT:=0.0}"
: "${MOE_ROUTER_TEMPERATURE:=1.0}"
: "${MOE_ROUTER_JITTER_NOISE:=0.0}"
: "${MOE_LOAD_BALANCE_WEIGHT:=0.00}"
: "${MOE_FINE_GRAINED_FACTOR:=1}"
: "${MOE_NUM_SHARED_EXPERTS:=1}"
: "${MOE_SHARED_EXPERT_WEIGHT:=1.0}"
: "${USE_LISTWISE_REWARD:=True}"
: "${NUM_LISTWISE_NEGATIVES:=15}"
: "${LISTWISE_CANDIDATE_CHUNK_SIZE:=8}"
: "${LISTWISE_TOPK:=10}"
: "${LISTWISE_TAU:=1.0}"
: "${REWARD_W_DCG:=1.0}"
: "${REWARD_W_SOFT:=0.5}"
: "${REWARD_W_MARGIN:=0.5}"
: "${REWARD_W_LOGP:=0.05}"
: "${NOISE_STD:=1.0}"
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
: "${METRIC_TEMPERATURE:=1.0}"
: "${METRIC_GUIDANCE_SCALE:=1.0}"
: "${METRIC_LENGTH_PENALTY:=1.0}"
: "${METRIC_CUTOFF_LEN:=2560}"
: "${USE_FSDP2:=False}"
: "${FSDP2_ACTIVATION_CHECKPOINTING:=False}"
: "${FSDP2_STATE_DICT_TYPE:=FULL_STATE_DICT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
TRAIN_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=$(( $(tr -cd ',' <<< "${CUDA_VISIBLE_DEVICES}" | wc -c) + 1 ))
if (( NUM_GPUS < 1 )); then
    echo "No GPU selected. Set GPU_IDS or CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with NUM_GPUS=${NUM_GPUS}"
echo "Using LATENT_LAYER_TYPE=${LATENT_LAYER_TYPE}"
echo "Using MOE_LOAD_BALANCE_WEIGHT=${MOE_LOAD_BALANCE_WEIGHT}"
echo "Using USE_LISTWISE_REWARD=${USE_LISTWISE_REWARD}"
echo "Using LISTWISE_CANDIDATE_CHUNK_SIZE=${LISTWISE_CANDIDATE_CHUNK_SIZE}"
echo "Using SAVE_STRATEGY=${SAVE_STRATEGY}"
echo "Using BEST_MODEL_METRIC=${BEST_MODEL_METRIC}"
echo "Using METRIC_EVAL_SAMPLE=${METRIC_EVAL_SAMPLE}"
echo "Using METRIC_NUM_BEAMS=${METRIC_NUM_BEAMS}"
echo "Using METRIC_TOP_K=${METRIC_TOP_K}"
echo "Using USE_FSDP2=${USE_FSDP2}"

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

for category in "Toys_and_Games"; do

    path_to_data=data
    train_file=$(ls -f ${path_to_data}/train/${category}*11.csv)
    eval_file=$(ls -f ${path_to_data}/valid/${category}*11.csv)
    test_file=$(ls -f ${path_to_data}/test/${category}*11.csv)
    info_file=$(ls -f ${path_to_data}/info/${category}*.txt)
    for lr in 1e-4 1e-5 5e-4 5e-5; do
        task_name=nomul_attn_lr_${lr}
        base_model=${BASE_MODEL:-output_dir/${category}/latent}
        out=./output_dir/${category}/${task_name}
        mkdir -p $out
        torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port=$((RANDOM % 9999 + 10000)) \
            src/grpo_attention_tuning/train_noise_grpo.py --category $category \
            --num_generations 8 --sample -1 --num_epochs 3 --info_file $info_file \
            --base_model $base_model --output_dir $out --beta 0.00 \
            --latent_layer_type ${LATENT_LAYER_TYPE} \
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
            --batch_size 256 --micro_batch_size 8 --lr $lr --seed 42 \
            --train_file $train_file --eval_file $eval_file \
            --num_iterations 1 --epsilon 0.2 --epsilon_high 0.28 \
            --load_best_model_at_end ${LOAD_BEST_MODEL_AT_END} \
            --save_strategy ${SAVE_STRATEGY} \
            --save_total_limit ${SAVE_TOTAL_LIMIT} \
            --save_only_model ${SAVE_ONLY_MODEL} \
            --save_final_model ${SAVE_FINAL_MODEL} \
            --best_model_metric ${BEST_MODEL_METRIC} \
            --metric_eval_sample ${METRIC_EVAL_SAMPLE} \
            --metric_eval_batch_size ${METRIC_EVAL_BATCH_SIZE} \
            --metric_num_beams ${METRIC_NUM_BEAMS} \
            --metric_top_k ${METRIC_TOP_K} \
            --metric_max_new_tokens ${METRIC_MAX_NEW_TOKENS} \
            --metric_temperature ${METRIC_TEMPERATURE} \
            --metric_guidance_scale ${METRIC_GUIDANCE_SCALE} \
            --metric_length_penalty ${METRIC_LENGTH_PENALTY} \
            --metric_cutoff_len ${METRIC_CUTOFF_LEN} \
            --use_listwise_reward ${USE_LISTWISE_REWARD} \
            --num_listwise_negatives ${NUM_LISTWISE_NEGATIVES} \
            --listwise_candidate_chunk_size ${LISTWISE_CANDIDATE_CHUNK_SIZE} \
            --listwise_topk ${LISTWISE_TOPK} \
            --listwise_tau ${LISTWISE_TAU} \
            --reward_w_dcg ${REWARD_W_DCG} \
            --reward_w_soft ${REWARD_W_SOFT} \
            --reward_w_margin ${REWARD_W_MARGIN} \
            --reward_w_logp ${REWARD_W_LOGP} \
            --noise_std ${NOISE_STD} \
            --use_fsdp2 ${USE_FSDP2} \
            --fsdp2_activation_checkpointing ${FSDP2_ACTIVATION_CHECKPOINTING} \
            --fsdp2_state_dict_type ${FSDP2_STATE_DICT_TYPE} \
            > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

        IFS=',' read -r -a array <<< "${CUDA_VISIBLE_DEVICES}"
        nn=${NUM_GPUS}
        model_ckpt="$(select_model_ckpt "${out}")"
        echo "Using model checkpoint for eval: ${model_ckpt}"
        mkdir -p ./output_dir/${category}_base
        python src/utils/split.py --input_path $test_file \
                --output_path ./output_dir/${category}_base --nn ${nn}

        # 遍历数组并输出内容
        for (( i=0; i<$nn; i++ ))
        do
            export CUDA_VISIBLE_DEVICES=${array[$i]}
            echo output_dir/${category}_base/${i}.csv
            python src/grpo_attention_tuning/noise_eval.py \
                    --batch_size 4 --base_model $model_ckpt --sample -1 \
                    --info_file ${info_file} --category ${category} --end_k -1 \
                    --test_data_path ./output_dir/${category}_base/${i}.csv \
                    --result_json_data ./output_dir/${category}_base/${i}.json &
            done
        wait
        export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
        python src/utils/merge.py --input_path ./output_dir/${category}_base/ \
                --output_path ${out}/final_result.json --nn ${nn}
    done

done

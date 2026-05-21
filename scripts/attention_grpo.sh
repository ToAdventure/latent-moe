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
: "${MOE_FINE_GRAINED_FACTOR:=2}"
: "${MOE_NUM_SHARED_EXPERTS:=1}"
: "${MOE_SHARED_EXPERT_WEIGHT:=1.0}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
TRAIN_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=$(( $(tr -cd ',' <<< "${CUDA_VISIBLE_DEVICES}" | wc -c) + 1 ))
if (( NUM_GPUS < 1 )); then
    echo "No GPU selected. Set GPU_IDS or CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with NUM_GPUS=${NUM_GPUS}"
echo "Using LATENT_LAYER_TYPE=${LATENT_LAYER_TYPE}"

for category in "Toys_and_Games"; do

    path_to_data=data
    train_file=$(ls -f ${path_to_data}/train/${category}*11.csv)
    eval_file=$(ls -f ${path_to_data}/valid/${category}*11.csv)
    test_file=$(ls -f ${path_to_data}/test/${category}*11.csv)
    info_file=$(ls -f ${path_to_data}/info/${category}*.txt)
    for lr in 1e-4 1e-5 5e-4 5e-5; do
        task_name=nomul_attn_lr_${lr}
        base_model=output_dir/${category}/latent
        out=./output_dir/${category}/${task_name}
        mkdir -p $out
        torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port=$((RANDOM % 9999 + 10000)) \
            src/grpo_attention_tuning/train_noise_grpo.py --category $category \
            --num_generations 8 --sample -1 --num_epochs 1 --info_file $info_file \
            --base_model $base_model --output_dir $out --beta 0.00 \
            --latent_layer_type ${LATENT_LAYER_TYPE} \
            --moe_num_experts ${MOE_NUM_EXPERTS} --moe_top_k ${MOE_TOP_K} \
            --moe_expert_hidden_size ${MOE_EXPERT_HIDDEN_SIZE} \
            --moe_expert_hidden_multiplier ${MOE_EXPERT_HIDDEN_MULTIPLIER} \
            --moe_dropout ${MOE_DROPOUT} \
            --moe_router_temperature ${MOE_ROUTER_TEMPERATURE} \
            --moe_router_jitter_noise ${MOE_ROUTER_JITTER_NOISE} \
            --moe_fine_grained_factor ${MOE_FINE_GRAINED_FACTOR} \
            --moe_num_shared_experts ${MOE_NUM_SHARED_EXPERTS} \
            --moe_shared_expert_weight ${MOE_SHARED_EXPERT_WEIGHT} \
            --batch_size 256 --micro_batch_size 8 --lr $lr --seed 42 \
            --train_file $train_file --eval_file $eval_file \
            --num_iterations 1 --epsilon 0.2 --epsilon_high 0.28 \
            > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

        IFS=',' read -r -a array <<< "${CUDA_VISIBLE_DEVICES}"
        nn=${NUM_GPUS}
        model_ckpt=${out}
        mkdir -p ./output_dir/${category}_base
        python src/utils/split.py --input_path $test_file \
                --output_path ./output_dir/${category}_base --nn ${nn}

        # 遍历数组并输出内容
        for (( i=0; i<$nn; i++ ))
        do
            export CUDA_VISIBLE_DEVICES=${array[$i]}
            echo output_dir/${category}_base/${i}.csv
            python src/grpo_attention_tuning/noise_eval.py \
                    --batch_size 12 --base_model $model_ckpt --sample -1 \
                    --info_file ${info_file} --category ${category} --end_k -1 \
                    --test_data_path ./output_dir/${category}_base/${i}.csv \
                    --result_json_data ./output_dir/${category}_base/${i}.json &
            done
        wait
        export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
        python src/utils/merge.py --input_path ./output_dir/${category}_base/ \
                --output_path ${model_ckpt}/final_result.json --nn ${nn}
    done

done

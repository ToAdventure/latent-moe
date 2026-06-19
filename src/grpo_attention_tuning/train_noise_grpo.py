import sys
import os
import shutil
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))
# os.environ["WANDB_DISABLED"] = "true"
import fire
import torch
import torch.nn as nn

import trl
from datasets import Dataset as HFDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from model import LatentModel

from latent_grpo_dataset import D3Dataset, get_hash, get_prefix_data
from grpo_trainer import NoiseGRPORecTrainer
from fsdp2_utils import build_fsdp2_config, configure_fsdp2_environment
from latent_moe import set_latent_layer_config


CATEGORY_DICT = {
    "Office_Products": "office products",
    "Books": "books",
    "steam": "games",
    "CDs_and_Vinyl": "musics",
    "Toys_and_Games": "toys and games",
    "Video_Games": "video games",
    "Musical_Instruments": "music instruments",
    "Sports_and_Outdoors": "sports and outdoors",
    "Pet_Supplies": "pet supplies",
    "Arts_Crafts_and_Sewing": "arts products",
    "Movies": "movie",
    "yelp": "resturant",
}

GENERATION_BEST_METRICS = {"hr5_ndcg5", "generation_hr5_ndcg5"}

ROOT_MODEL_ARTIFACTS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "config.json",
    "generation_config.json",
    "training_args.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def parse_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off"):
            return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def load_item_names(info_file):
    with open(info_file, "r") as f:
        return [line.split("\t")[0].strip("\"").strip(" ").strip("\n").strip("\"") for line in f]


def cleanup_root_model_artifacts(output_dir: str):
    if not os.path.isdir(output_dir):
        return

    has_checkpoint = any(
        name.startswith("checkpoint-")
        and os.path.isdir(os.path.join(output_dir, name))
        for name in os.listdir(output_dir)
    )
    if not has_checkpoint:
        return

    for name in ROOT_MODEL_ARTIFACTS:
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)


def train(
    # model/data params, first is your latent model path (train from latent reasoning)
    base_model: str = "output_dir/Toys_and_Games/latent",  # the only required argument
    train_file: str = "data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    eval_file: str = "data/valid/Toys_and_Games_5_2017-1-2018-11.csv",
    info_file: str = "data/info/Toys_and_Games_5_2017-1-2018-11.txt",
    lr: float = float(6.0e-6),
    end_k: int = -1,
    latent_layer_type: str = "auto",
    moe_num_experts: int = 4,
    moe_top_k: int = 2,
    moe_expert_hidden_size: int = 0,
    moe_expert_hidden_multiplier: float = 1.0,
    moe_dropout: float = 0.0,
    moe_router_temperature: float = 1.0,
    moe_router_jitter_noise: float = 0.0,
    moe_fine_grained_factor: int = 2,
    moe_num_shared_experts: int = 1,
    moe_shared_expert_weight: float = 1.0,
    moe_load_balance_weight: float = 0.01,
    num_epochs: int = 1,
    batch_size: int = 256,
    micro_batch_size: int = 8,
    sample: int = 2000,
    seed: int = 42,
    cutoff_len: int = 512,
    output_dir: str = "output_dir/Video_Games/grpo_test",
    category: str = "Toys_and_Games", # category name

    num_generations: int = 8, # grpo args
    beta: float = 0.01, # KL coefficient
    num_iterations: int = 2, # number of iterations
    epsilon: float = 0.2, # clipping value
    epsilon_high: float = 0.28, # 其实这两个一个是high，一个是low. 当没有high的时候，low就是high
    max_completion_length: int = 256,
    use_listwise_reward: bool = True,
    num_listwise_negatives: int = 15,
    listwise_candidate_chunk_size: int = 8,
    listwise_topk: int = 10,
    listwise_tau: float = 1.0,
    reward_w_dcg: float = 1.0,
    reward_w_soft: float = 0.5,
    reward_w_margin: float = 0.5,
    reward_w_logp: float = 0.05,
    noise_std: float = 1.0,

    use_vllm: bool = False,
    vllm_gpu_memory_utilization: float = 0.7,
    load_best_model_at_end: bool = True,
    save_strategy: str = "best",
    save_total_limit: int = 1,
    save_only_model: bool = True,
    save_final_model: bool = False,
    best_model_metric: str = "hr5_ndcg5",
    metric_eval_sample: int = -1,
    metric_eval_batch_size: int = 4,
    metric_num_beams: int = 10,
    metric_top_k: int = 5,
    metric_max_new_tokens: int = 128,
    metric_temperature: float = 1.0,
    metric_guidance_scale: float = 1.0,
    metric_length_penalty: float = 1.0,
    metric_cutoff_len: int = 2560,

    resume_from_checkpoint: str = None,
    local_rank: int = 0,
    use_fsdp2: bool = False,
    fsdp2_activation_checkpointing: bool = False,
    fsdp2_state_dict_type: str = "FULL_STATE_DICT",

):
    load_best_model_at_end = parse_bool(load_best_model_at_end, "load_best_model_at_end")
    save_only_model = parse_bool(save_only_model, "save_only_model")
    save_final_model = parse_bool(save_final_model, "save_final_model")
    use_listwise_reward = parse_bool(use_listwise_reward, "use_listwise_reward")
    use_fsdp2 = parse_bool(use_fsdp2, "use_fsdp2")
    fsdp2_activation_checkpointing = parse_bool(
        fsdp2_activation_checkpointing, "fsdp2_activation_checkpointing"
    )

    configure_fsdp2_environment(use_fsdp2, fsdp2_state_dict_type)
    if use_fsdp2 and beta != 0.0:
        raise ValueError("FSDP2 only supports one model in this training path; set beta=0.0 or disable use_fsdp2.")

    print(category)
    raw_category = category
    category = CATEGORY_DICT[category]
    print(lr, type(lr))

    best_model_metric = str(best_model_metric).strip().lower()
    generation_metric_enabled = best_model_metric in GENERATION_BEST_METRICS
    if best_model_metric not in GENERATION_BEST_METRICS and best_model_metric not in ("eval_loss", "loss"):
        raise ValueError(
            "best_model_metric must be one of "
            f"{sorted(GENERATION_BEST_METRICS | {'eval_loss'})}, got {best_model_metric!r}"
        )
    if generation_metric_enabled and str(save_strategy).lower() != "best":
        print(
            "Generation best metric requires save_strategy='best'; "
            f"overriding save_strategy={save_strategy!r} to 'best'."
        )
        save_strategy = "best"

    effective_load_best_model_at_end = bool(load_best_model_at_end)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp and effective_load_best_model_at_end:
        print(
            "Distributed training detected; disabling load_best_model_at_end "
            "because this Transformers/FSDP combination can hang while loading "
            "the best checkpoint at the end. Checkpoints are still saved "
            "according to save_strategy."
        )
        effective_load_best_model_at_end = False
    uses_best_metric = generation_metric_enabled or effective_load_best_model_at_end or str(save_strategy).lower() == "best"
    metric_for_best_model = None
    greater_is_better = None
    if generation_metric_enabled:
        metric_for_best_model = f"eval_hr{metric_top_k}_ndcg{metric_top_k}"
        greater_is_better = True
    elif uses_best_metric:
        metric_for_best_model = "eval_loss"
        greater_is_better = False

    if use_listwise_reward and micro_batch_size != num_generations:
        print(
            "Listwise GRPO requires one prompt group per device batch; "
            f"overriding micro_batch_size={micro_batch_size} to num_generations={num_generations}."
        )
        micro_batch_size = num_generations

    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    assert batch_size==micro_batch_size*gradient_accumulation_steps*world_size

    model_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    set_latent_layer_config(
        model_config,
        latent_layer_type=latent_layer_type,
        end_k=end_k,
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        moe_expert_hidden_size=moe_expert_hidden_size,
        moe_expert_hidden_multiplier=moe_expert_hidden_multiplier,
        moe_dropout=moe_dropout,
        moe_router_temperature=moe_router_temperature,
        moe_router_jitter_noise=moe_router_jitter_noise,
        moe_fine_grained_factor=moe_fine_grained_factor,
        moe_num_shared_experts=moe_num_shared_experts,
        moe_shared_expert_weight=moe_shared_expert_weight,
        moe_load_balance_weight=moe_load_balance_weight,
    )
    model = LatentModel.from_pretrained(base_model,
        config=model_config,
        torch_dtype=torch.bfloat16,
    )
    model.attention.end_k = end_k
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = count_trainable_params(model)
    print(f"Trainable parameters: {trainable_params}")
    # replace_with_noisy_linear(model.attention)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    assert tokenizer("<|Thought|>")['input_ids'][0] == len(tokenizer) - 1

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    hash_dict = get_prefix_data(info_file, tokenizer)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    train_data = D3Dataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed,
                           category=category, test=False)
    val_data = D3Dataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed,
                         category=category, test=False)

    generation_metric_config = None
    generation_metric_dataset = None
    generation_metric_raw_data = None
    generation_metric_hash_dict = None
    generation_metric_item_names = None
    if generation_metric_enabled:
        metric_eval_sample = int(metric_eval_sample)
        metric_top_k = int(metric_top_k)
        metric_eval_batch_size = int(metric_eval_batch_size)
        if metric_top_k <= 0:
            raise ValueError(f"metric_top_k must be positive, got {metric_top_k}")
        if metric_eval_batch_size <= 0:
            raise ValueError(f"metric_eval_batch_size must be positive, got {metric_eval_batch_size}")
        metric_num_beams = max(int(metric_num_beams), metric_top_k)
        print(
            "Using generation best metric: "
            f"HR@{metric_top_k} * NDCG@{metric_top_k} from {info_file}; "
            f"sample={metric_eval_sample}, batch_size={metric_eval_batch_size}, "
            f"num_beams={metric_num_beams}."
        )
        generation_metric_dataset = D3Dataset(
            train_file=eval_file,
            tokenizer=tokenizer,
            max_len=metric_cutoff_len,
            sample=metric_eval_sample,
            seed=seed,
            category=category,
            test=True,
        )
        generation_metric_raw_data = generation_metric_dataset.get_all()
        generation_metric_hash_dict = get_prefix_data(info_file, tokenizer)
        generation_metric_item_names = load_item_names(info_file)
        generation_metric_config = {
            "batch_size": metric_eval_batch_size,
            "num_beams": metric_num_beams,
            "return_sequences": metric_top_k,
            "top_k": metric_top_k,
            "max_new_tokens": int(metric_max_new_tokens),
            "temperature": float(metric_temperature),
            "guidance_scale": float(metric_guidance_scale),
            "length_penalty": float(metric_length_penalty),
        }

    print("LOAD DATA FINISHED")

    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})

    def repeat_dataset_for_grpo(ds, repeat):
        if repeat <= 1:
            return ds
        new_data = {k: [] for k in ds.column_names}
        for ex in ds:
            for _ in range(repeat):
                for k in ds.column_names:
                    new_data[k].append(ex[k])
        return HFDataset.from_dict(new_data)

    if use_listwise_reward:
        hf_train_dataset = repeat_dataset_for_grpo(hf_train_dataset, num_generations)

    #############################
    # Initialize the GRPO trainer
    #############################
    def reward(completions, targets, **kwargs):
        def strip_(a):
            return a.strip().strip('\n').strip('\"')
        # aa=[1 if strip_(com) == strip_(tar) else 0 for com, tar in zip(completions, targets)]
        # print(f'共有{len(set(completions))}种答案, {sum(aa)}分')
        return [1 if strip_(com) == strip_(tar) else 0 for com, tar in zip(completions, targets)]

    reward_funcs = [reward]
    trainer = NoiseGRPORecTrainer(
    # from grpo_trainer import HalfGRPORecTrainer
    # trainer = HalfGRPORecTrainer(
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        info_file=info_file,
        use_listwise_reward=use_listwise_reward,
        num_listwise_negatives=num_listwise_negatives,
        listwise_candidate_chunk_size=listwise_candidate_chunk_size,
        listwise_topk=listwise_topk,
        listwise_tau=listwise_tau,
        reward_w_dcg=reward_w_dcg,
        reward_w_soft=reward_w_soft,
        reward_w_margin=reward_w_margin,
        reward_w_logp=reward_w_logp,
        noise_std=noise_std,
        generation_metric_config=generation_metric_config,
        generation_metric_dataset=generation_metric_dataset,
        generation_metric_raw_data=generation_metric_raw_data,
        generation_metric_hash_dict=generation_metric_hash_dict,
        generation_metric_item_names=generation_metric_item_names,
        model=model,
        reward_funcs=reward_funcs,
        args= trl.GRPOConfig(
            warmup_steps=200,
            # warmup_ratio=0.05,
            num_generations=num_generations,
            max_prompt_length=512,
            temperature=1.0,
            max_completion_length=max_completion_length,
            use_vllm=use_vllm,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            learning_rate=lr,
            beta=beta,
            num_iterations=num_iterations,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            num_train_epochs=num_epochs,
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            save_only_model=save_only_model,
            load_best_model_at_end=effective_load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            ddp_find_unused_parameters=None if use_fsdp2 else True,
            fsdp="auto_wrap" if use_fsdp2 else "",
            fsdp_config=build_fsdp2_config(fsdp2_activation_checkpointing) if use_fsdp2 else None,
            report_to="none",
            bf16=True,
            logging_steps=1,
        ),
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        processing_class=tokenizer,
    )
    model.config.use_cache = False
    ###############
    # Training loop
    ###############
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(hf_train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if save_final_model:
        trainer.save_model(output_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(output_dir)
    elif trainer.is_world_process_zero():
        cleanup_root_model_artifacts(output_dir)

    if ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(train)

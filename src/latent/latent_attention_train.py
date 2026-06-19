import sys
import os
import shutil
import glob
import json
import math
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))
import fire
import torch

import transformers
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    AutoConfig,
    GenerationConfig,
    LogitsProcessorList,
    TemperatureLogitsWarper,
)

from layers import LatentModel
from reasoning_dataset import LatentRDataset
from fsdp2_utils import build_fsdp2_config, configure_fsdp2_environment
from latent_moe import set_latent_layer_config
from LogitProcesser import CFEnhancedLogitsProcessor, get_hash, get_prefix_data
# torch.backends.cuda.enable_cudnn_sdp(False)


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


def normalize_title(value):
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip(" \n").strip("\"").strip(" ")


def resolve_metric_info_file(info_file, category):
    if info_file:
        return info_file
    matches = sorted(glob.glob(f"data/info/{category}*.txt"))
    if not matches:
        raise FileNotFoundError(
            f"Could not infer metric_info_file for category={category!r}; "
            "pass --metric_info_file explicitly."
        )
    return matches[0]


def load_item_names(info_file):
    with open(info_file, "r") as f:
        return [line.split("\t")[0].strip("\"").strip(" ").strip("\n").strip("\"") for line in f]


def compute_hr_ndcg_at_k(records, item_names, top_k):
    item_set = set(item_names)
    hit = 0.0
    ndcg = 0.0
    unknown_predictions = 0
    denom = 1.0 / math.log(2)

    for record in records:
        target = normalize_title(record["target"])
        preds = [normalize_title(pred) for pred in record["predict"][:top_k]]
        rank = None
        for idx, pred in enumerate(preds):
            if pred not in item_set:
                unknown_predictions += 1
            if pred == target:
                rank = idx
                break
        if rank is not None:
            hit += 1.0
            ndcg += (1.0 / math.log(rank + 2)) / denom

    total = max(1, len(records))
    hr = hit / total
    ndcg = ndcg / total
    return {
        f"hr{top_k}": hr,
        f"ndcg{top_k}": ndcg,
        f"hr{top_k}_ndcg{top_k}": hr * ndcg,
        "num_generation_eval_samples": len(records),
        "unknown_generation_predictions": unknown_predictions,
    }


class GenerationMetricTrainer(transformers.Trainer):
    def __init__(
        self,
        *args,
        generation_metric_config=None,
        generation_metric_dataset=None,
        generation_metric_raw_data=None,
        generation_metric_hash_dict=None,
        generation_metric_item_names=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generation_metric_config = generation_metric_config
        self.generation_metric_dataset = generation_metric_dataset
        self.generation_metric_raw_data = generation_metric_raw_data
        self.generation_metric_hash_dict = generation_metric_hash_dict
        self.generation_metric_item_names = generation_metric_item_names

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self.generation_metric_config is None:
            return metrics

        generation_metrics = self.evaluate_generation_metric()
        prefixed = {f"{metric_key_prefix}_{key}": value for key, value in generation_metrics.items()}
        metrics.update(prefixed)
        self.log(prefixed)
        return metrics

    def distributed_state(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size(), True
        return 0, 1, False

    def evaluate_generation_metric(self):
        rank, world_size, distributed = self.distributed_state()
        config = self.generation_metric_config
        dataset = self.generation_metric_dataset
        raw_data = self.generation_metric_raw_data
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Generation metric evaluation requires a tokenizer/processing_class.")

        local_records = self.generate_local_metric_records(
            dataset=dataset,
            raw_data=raw_data,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            config=config,
        )

        if distributed:
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, local_records)
        else:
            gathered = [local_records]

        if rank == 0:
            records = []
            for part in gathered:
                records.extend(part)
            records.sort(key=lambda record: record["index"])
            metrics = compute_hr_ndcg_at_k(
                records,
                self.generation_metric_item_names,
                int(config["top_k"]),
            )
            metrics["generation_best_metric"] = metrics[f"hr{config['top_k']}_ndcg{config['top_k']}"]
            self.append_generation_metric_history(metrics)
        else:
            metrics = None

        if distributed:
            payload = [metrics]
            torch.distributed.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        return metrics

    def append_generation_metric_history(self, metrics):
        if not self.is_world_process_zero():
            return
        os.makedirs(self.args.output_dir, exist_ok=True)
        history_path = os.path.join(self.args.output_dir, "generation_metric_history.jsonl")
        payload = {
            "epoch": float(self.state.epoch) if self.state.epoch is not None else None,
            "global_step": int(self.state.global_step),
            **metrics,
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def generate_local_metric_records(self, dataset, raw_data, tokenizer, rank, world_size, config):
        model = self.model
        generation_model = model
        if not hasattr(generation_model, "generate") and hasattr(generation_model, "module"):
            generation_model = generation_model.module
        model_config = getattr(generation_model, "config", None)
        if model_config is None and hasattr(model, "module"):
            model_config = getattr(model.module, "config", None)
        if model_config is None:
            raise ValueError("Generation metric evaluation requires a model config.")
        was_training = model.training
        model.eval()

        total = len(dataset)
        if total == 0:
            return []

        batch_size = int(config["batch_size"])
        num_beams = int(config["num_beams"])
        return_sequences = int(config["return_sequences"])
        max_per_rank = (total + world_size - 1) // world_size
        local_indices = list(range(rank, total, world_size))
        if len(local_indices) < max_per_rank:
            local_indices.extend([None] * (max_per_rank - len(local_indices)))

        local_records = []
        with torch.no_grad():
            for start in range(0, len(local_indices), batch_size):
                batch_indices = local_indices[start:start + batch_size]
                if not batch_indices:
                    continue
                encodings = [
                    dataset[idx] if idx is not None else dataset[0]
                    for idx in batch_indices
                ]
                input_ids, attention_mask, min_len = self.pad_generation_batch(encodings, tokenizer)
                input_ids = input_ids.to(self.args.device)
                attention_mask = attention_mask.to(self.args.device)

                generation_config = GenerationConfig(
                    num_beams=num_beams,
                    length_penalty=float(config["length_penalty"]),
                    num_return_sequences=return_sequences,
                    pad_token_id=model_config.pad_token_id,
                    eos_token_id=model_config.eos_token_id,
                    max_new_tokens=int(config["max_new_tokens"]),
                )

                ccc = CFEnhancedLogitsProcessor(
                    guidance_scale=float(config["guidance_scale"]),
                    cf_logits=None,
                    prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                    cf_dict=None,
                    unconditional_ids=None,
                    model=generation_model,
                    tokenizer=tokenizer,
                    num_beams=num_beams,
                )
                logits_processor = LogitsProcessorList(
                    [
                        TemperatureLogitsWarper(temperature=float(config["temperature"])),
                        ccc,
                    ]
                )
                generation_output = generation_model.generate(
                    input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    logits_processor=logits_processor,
                    attention_mask=attention_mask,
                    synced_gpus=world_size > 1,
                )

                decoded = tokenizer.batch_decode(
                    generation_output.sequences[:, min_len - 5:],
                    skip_special_tokens=True,
                )
                decoded = [text.split("Response:")[-1] for text in decoded]
                grouped = [
                    decoded[i * return_sequences:(i + 1) * return_sequences]
                    for i in range(len(decoded) // return_sequences)
                ]

                for row_idx, sample_idx in enumerate(batch_indices):
                    if sample_idx is None:
                        continue
                    local_records.append(
                        {
                            "index": int(sample_idx),
                            "target": raw_data[sample_idx]["output"],
                            "predict": grouped[row_idx][:int(config["top_k"])],
                        }
                    )

        if was_training:
            model.train()
        return local_records

    def pad_generation_batch(self, encodings, tokenizer):
        max_len = max(len(row["input_ids"]) for row in encodings)
        min_len = min(len(row["input_ids"]) for row in encodings)
        input_ids = []
        attention_mask = []
        for row in encodings:
            length = len(row["input_ids"])
            pad = max_len - length
            input_ids.append([tokenizer.pad_token_id] * pad + row["input_ids"])
            attention_mask.append([0] * pad + row["attention_mask"])
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            min_len,
        )

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in self.generation_metric_hash_dict:
            return self.generation_metric_hash_dict[hash_number]
        return []

def train(
    # model/data params
    base_model: str = "/data/gtq/latent-moe/models/Qwen2.5-1.5B",  # the only required argument
    train_file: str="data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    eval_file: str="data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    output_dir: str = "./output_test/",
    sample: int = 1024,
    seed: int = 0,
    # attention hyper parameters
    end_k: int = -1,
    latent_layer_type: str = "attention",
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
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    load_best_model_at_end: bool = True,
    early_stopping_patience: int = 1,
    save_strategy: str = "best",
    save_total_limit: int = 1,
    save_only_model: bool = True,
    save_final_model: bool = False,
    best_model_metric: str = "eval_loss",
    metric_info_file: str = None,
    metric_eval_sample: int = -1,
    metric_eval_batch_size: int = 4,
    metric_num_beams: int = 10,
    metric_top_k: int = 5,
    metric_max_new_tokens: int = 128,
    metric_temperature: float = 1.0,
    metric_guidance_scale: float = 1.0,
    metric_length_penalty: float = 1.0,
    metric_cutoff_len: int = 2560,

    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    
    local_rank: int = 0,
    use_fsdp2: bool = True,
    fsdp2_activation_checkpointing: bool = False,
    fsdp2_state_dict_type: str = "FULL_STATE_DICT",
    category: str="Toys_and_Games",
    K: int = 0,
    prompt_style: str = "default"

):
    train_on_inputs = parse_bool(train_on_inputs, "train_on_inputs")
    group_by_length = parse_bool(group_by_length, "group_by_length")
    load_best_model_at_end = parse_bool(load_best_model_at_end, "load_best_model_at_end")
    save_only_model = parse_bool(save_only_model, "save_only_model")
    save_final_model = parse_bool(save_final_model, "save_final_model")
    use_fsdp2 = parse_bool(use_fsdp2, "use_fsdp2")
    fsdp2_activation_checkpointing = parse_bool(
        fsdp2_activation_checkpointing, "fsdp2_activation_checkpointing"
    )

    configure_fsdp2_environment(use_fsdp2, fsdp2_state_dict_type)

    raw_category = category
    print(raw_category)
    category = CATEGORY_DICT[raw_category]
    assert (
        base_model
    )
    transformers.set_seed(seed)

    if batch_size % micro_batch_size != 0:
        raise ValueError(
            f"batch_size must be divisible by micro_batch_size, got "
            f"batch_size={batch_size}, micro_batch_size={micro_batch_size}"
        )
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        if gradient_accumulation_steps % world_size != 0:
            raise ValueError(
                "batch_size / micro_batch_size must be divisible by WORLD_SIZE "
                f"to keep the requested global batch size exactly, got "
                f"batch_size={batch_size}, micro_batch_size={micro_batch_size}, "
                f"WORLD_SIZE={world_size}"
            )
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    if gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}. "
            "Increase batch_size or reduce micro_batch_size/WORLD_SIZE."
        )

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
    model = LatentModel.from_pretrained(
        base_model,
        config=model_config,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        # device_map=device_map,
    )
    model.attention.end_k = end_k
    model.config.loss_type = "ce"
    
    ##############################################################################
    ##################                                 
    ##################   add a special token here to represent latent thought
    ##################

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = tokenizer.additional_special_tokens
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    additional_special_tokens.append("<|Thought|>")
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    print("We have added", num_added_toks, "tokens")
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")['input_ids'][0] == len(tokenizer) - 1
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    
    ##################
    ##############################################################################

    train_data = LatentRDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category, K = K, prompt_style=prompt_style)
    val_data = LatentRDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, category=category, K = K, prompt_style=prompt_style)

    generation_metric_config = None
    generation_metric_dataset = None
    generation_metric_raw_data = None
    generation_metric_hash_dict = None
    generation_metric_item_names = None
    if generation_metric_enabled:
        metric_info_file = resolve_metric_info_file(metric_info_file, raw_category)
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
            f"HR@{metric_top_k} * NDCG@{metric_top_k} from {metric_info_file}; "
            f"sample={metric_eval_sample}, batch_size={metric_eval_batch_size}, "
            f"num_beams={metric_num_beams}."
        )
        generation_metric_dataset = LatentRDataset(
            train_file=eval_file,
            tokenizer=tokenizer,
            max_len=metric_cutoff_len,
            sample=metric_eval_sample,
            seed=seed,
            category=category,
            K=K,
            test=True,
            prompt_style=prompt_style,
        )
        generation_metric_raw_data = generation_metric_dataset.get_all()
        generation_metric_hash_dict = get_prefix_data(metric_info_file, tokenizer)
        generation_metric_item_names = load_item_names(metric_info_file)
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

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    from datasets import Dataset as HFDataset
    print("train_data 0 keys:", train_data[0].keys())
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})
    callbacks = []
    if effective_load_best_model_at_end and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = GenerationMetricTrainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy=save_strategy,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            save_only_model=save_only_model,
            load_best_model_at_end=effective_load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            ddp_find_unused_parameters=None if use_fsdp2 else (False if ddp else None),
            fsdp="auto_wrap" if use_fsdp2 else "",
            fsdp_config=build_fsdp2_config(fsdp2_activation_checkpointing) if use_fsdp2 else None,
            group_by_length=group_by_length,
            report_to="none",
            remove_unused_columns=False,
            seed=seed,
            data_seed=seed,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
        generation_metric_config=generation_metric_config,
        generation_metric_dataset=generation_metric_dataset,
        generation_metric_raw_data=generation_metric_raw_data,
        generation_metric_hash_dict=generation_metric_hash_dict,
        generation_metric_item_names=generation_metric_item_names,
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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

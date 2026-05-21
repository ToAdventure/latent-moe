import sys
import os
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
)

from layers import LatentModel
from reasoning_dataset import LatentRDataset
from fsdp2_utils import build_fsdp2_config, configure_fsdp2_environment
from latent_moe import set_latent_layer_config
# torch.backends.cuda.enable_cudnn_sdp(False)

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
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve

    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    
    local_rank: int = 0,
    use_fsdp2: bool = True,
    fsdp2_activation_checkpointing: bool = False,
    fsdp2_state_dict_type: str = "FULL_STATE_DICT",
    category: str="Toys_and_Games",
    K: int = 0,
    prompt_style: str = "default"

):
    configure_fsdp2_environment(use_fsdp2, fsdp2_state_dict_type)

    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    )
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

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
    
    ##################
    ##############################################################################

    train_data = LatentRDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category, K = K, prompt_style=prompt_style)
    val_data = LatentRDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, category=category, K = K, prompt_style=prompt_style)

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
    trainer = transformers.Trainer(
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
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None if use_fsdp2 else (False if ddp else None),
            fsdp="auto_wrap" if use_fsdp2 else "",
            fsdp_config=build_fsdp2_config(fsdp2_activation_checkpointing) if use_fsdp2 else None,
            group_by_length=group_by_length,
            report_to="none",
            remove_unused_columns=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=1)],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)

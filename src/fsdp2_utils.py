import os
from typing import Any


def configure_fsdp2_environment(
    enabled: bool = True,
    state_dict_type: str = "FULL_STATE_DICT",
) -> None:
    if not enabled:
        for key in (
            "ACCELERATE_USE_FSDP",
            "FSDP_VERSION",
            "FSDP_RESHARD_AFTER_FORWARD",
            "FSDP_STATE_DICT_TYPE",
            "FSDP_CPU_RAM_EFFICIENT_LOADING",
        ):
            os.environ.pop(key, None)
        return

    # Transformers does not propagate fsdp_version from TrainingArguments to
    # Accelerate in this version, so set the FSDP2 switch before Trainer builds
    # its Accelerator.
    os.environ.setdefault("ACCELERATE_USE_FSDP", "true")
    os.environ.setdefault("FSDP_VERSION", "2")
    os.environ.setdefault("FSDP_RESHARD_AFTER_FORWARD", "true")
    os.environ.setdefault("FSDP_STATE_DICT_TYPE", state_dict_type)
    os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "false")


def build_fsdp2_config(
    activation_checkpointing: bool = False,
    transformer_layer_cls_to_wrap: str = "Qwen2DecoderLayer",
) -> dict[str, Any]:
    return {
        "reshard_after_forward": True,
        "transformer_layer_cls_to_wrap": [transformer_layer_cls_to_wrap],
        "activation_checkpointing": activation_checkpointing,
        "limit_all_gathers": True,
        "cpu_ram_efficient_loading": False,
    }

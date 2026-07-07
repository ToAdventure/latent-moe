import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_LAYER_TYPES = {"auto", "attention", "topk_moe", "deepseek_moe"}
MOE_EXPERT_TYPES = {"auto", "mlp", "attention"}
MOE_BALANCE_LOSS_MODES = {"auto", "fixed", "uncertainty"}
MOE_EXPERT_TYPE_ALIASES = {
    "ffn": "mlp",
    "feedforward": "mlp",
    "feed_forward": "mlp",
    "attn": "attention",
    "self_attention": "attention",
}


def _positive_int(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _is_auto(value):
    return isinstance(value, str) and value.strip().lower() == "auto"


def _config_value(config, name, value, default):
    if _is_auto(value):
        return getattr(config, name, default)
    return value


def resolve_latent_layer_type(config, latent_layer_type="auto"):
    if latent_layer_type == "auto":
        return getattr(config, "latent_layer_type", "attention")
    if latent_layer_type not in LATENT_LAYER_TYPES:
        raise ValueError(f"latent_layer_type must be one of {sorted(LATENT_LAYER_TYPES)}, got {latent_layer_type}")
    return latent_layer_type


def resolve_moe_expert_type(config=None, moe_expert_type="auto"):
    moe_expert_type = str(moe_expert_type).strip().lower()
    moe_expert_type = MOE_EXPERT_TYPE_ALIASES.get(moe_expert_type, moe_expert_type)
    if moe_expert_type == "auto":
        configured = getattr(config, "moe_expert_type", "mlp") if config is not None else "mlp"
        return resolve_moe_expert_type(None, configured)
    if moe_expert_type not in MOE_EXPERT_TYPES:
        raise ValueError(f"moe_expert_type must be one of {sorted(MOE_EXPERT_TYPES)}, got {moe_expert_type}")
    return moe_expert_type


def resolve_moe_balance_loss_mode(config=None, moe_balance_loss_mode="auto"):
    mode = str(moe_balance_loss_mode).strip().lower()
    if mode in ("uncertain", "learned", "learnable"):
        mode = "uncertainty"
    if mode == "auto":
        configured = getattr(config, "moe_balance_loss_mode", "fixed") if config is not None else "fixed"
        return resolve_moe_balance_loss_mode(None, configured)
    if mode not in MOE_BALANCE_LOSS_MODES:
        raise ValueError(
            f"moe_balance_loss_mode must be one of {sorted(MOE_BALANCE_LOSS_MODES)}, "
            f"got {moe_balance_loss_mode}"
        )
    return mode


def set_latent_layer_config(
    config,
    latent_layer_type="auto",
    end_k=-1,
    moe_num_experts="auto",
    moe_top_k="auto",
    moe_expert_hidden_size="auto",
    moe_expert_hidden_multiplier="auto",
    moe_dropout="auto",
    moe_router_temperature="auto",
    moe_router_jitter_noise="auto",
    moe_fine_grained_factor="auto",
    moe_num_shared_experts="auto",
    moe_shared_expert_weight="auto",
    moe_load_balance_weight="auto",
    moe_expert_type="auto",
    moe_balance_loss_mode="auto",
):
    latent_layer_type = resolve_latent_layer_type(config, latent_layer_type)
    moe_expert_type = resolve_moe_expert_type(config, moe_expert_type)
    moe_balance_loss_mode = resolve_moe_balance_loss_mode(config, moe_balance_loss_mode)
    config.latent_layer_type = latent_layer_type
    config.latent_end_k = int(end_k)
    config.moe_expert_type = moe_expert_type
    config.moe_balance_loss_mode = moe_balance_loss_mode
    config.moe_num_experts = _positive_int(
        _config_value(config, "moe_num_experts", moe_num_experts, 4), "moe_num_experts"
    )
    config.moe_top_k = _positive_int(_config_value(config, "moe_top_k", moe_top_k, 2), "moe_top_k")
    config.moe_expert_hidden_size = int(
        _config_value(config, "moe_expert_hidden_size", moe_expert_hidden_size, 0)
    )
    config.moe_expert_hidden_multiplier = float(
        _config_value(config, "moe_expert_hidden_multiplier", moe_expert_hidden_multiplier, 1.0)
    )
    config.moe_dropout = float(_config_value(config, "moe_dropout", moe_dropout, 0.0))
    config.moe_router_temperature = float(
        _config_value(config, "moe_router_temperature", moe_router_temperature, 1.0)
    )
    config.moe_router_jitter_noise = float(
        _config_value(config, "moe_router_jitter_noise", moe_router_jitter_noise, 0.0)
    )
    config.moe_fine_grained_factor = _positive_int(
        _config_value(config, "moe_fine_grained_factor", moe_fine_grained_factor, 2),
        "moe_fine_grained_factor",
    )
    config.moe_num_shared_experts = int(
        _config_value(config, "moe_num_shared_experts", moe_num_shared_experts, 1)
    )
    config.moe_shared_expert_weight = float(
        _config_value(config, "moe_shared_expert_weight", moe_shared_expert_weight, 1.0)
    )
    config.moe_load_balance_weight = float(
        _config_value(config, "moe_load_balance_weight", moe_load_balance_weight, 0.01)
    )
    return config


def _expert_hidden_size(hidden_size, expert_hidden_size, expert_hidden_multiplier, divisor=1):
    if expert_hidden_size and expert_hidden_size > 0:
        return int(expert_hidden_size)
    size = int(hidden_size * float(expert_hidden_multiplier) / max(1, divisor))
    return max(1, size)


class FeedForwardExpert(nn.Module):
    def __init__(self, hidden_size, expert_hidden_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_size, hidden_size),
        )

    def forward(self, hidden_states=None, attention_mask=None, thought_id_idx=None, context=None):
        if context is None:
            context = hidden_states
        return self.net(context)


class MaskedContextMixin:
    @staticmethod
    def context_mask(attention_mask, thought_id_idx, end_k=-1):
        if thought_id_idx is not None and attention_mask.size(0) != thought_id_idx.size(0):
            raise ValueError("attention_mask must have the same size as thought_id_idx")

        mask = attention_mask.clone().bool()
        batch_size, seq_len = mask.shape
        if thought_id_idx is None:
            thought_id_idx = torch.full((batch_size,), seq_len, device=mask.device, dtype=torch.long)

        context_mask = torch.zeros_like(mask)
        for i in range(batch_size):
            idx = int(thought_id_idx[i].item())
            idx = max(0, min(idx, seq_len))
            start_idx = max(0, idx - end_k) if end_k != -1 else 0
            context_mask[i, start_idx:idx] = mask[i, start_idx:idx]

            if not context_mask[i].any():
                fallback_idx = max(0, min(idx - 1, seq_len - 1))
                context_mask[i, fallback_idx] = mask[i, fallback_idx]
        return context_mask

    def pool_context(self, hidden_states, attention_mask, thought_id_idx):
        mask = self.context_mask(attention_mask, thought_id_idx, self.end_k).to(hidden_states.device)
        weights = mask.to(hidden_states.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden_states * weights).sum(dim=1) / denom


class AttentionExpert(MaskedContextMixin, nn.Module):
    def __init__(self, hidden_size, expert_hidden_size, end_k=-1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_hidden_size = _positive_int(expert_hidden_size, "expert_hidden_size")
        self.end_k = end_k
        self.query = nn.Linear(hidden_size, self.expert_hidden_size)
        self.key = nn.Linear(hidden_size, self.expert_hidden_size)
        self.value = nn.Linear(hidden_size, self.expert_hidden_size)
        self.output = nn.Linear(self.expert_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale_score = 1.0 / (self.expert_hidden_size ** 0.5)

    @staticmethod
    def selected_query_states(hidden_states, thought_id_idx):
        batch_size, seq_len, _ = hidden_states.shape
        if thought_id_idx is None:
            indices = torch.full((batch_size,), seq_len - 1, device=hidden_states.device, dtype=torch.long)
        else:
            indices = thought_id_idx.to(hidden_states.device).long() - 1
            indices = indices.clamp(0, seq_len - 1)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        return hidden_states[batch_indices, indices]

    def forward(self, hidden_states, attention_mask, thought_id_idx, context=None):
        mask = self.context_mask(attention_mask, thought_id_idx, self.end_k).to(hidden_states.device)
        query_states = self.selected_query_states(hidden_states, thought_id_idx)
        query = self.query(query_states).unsqueeze(1)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        attn_scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * self.scale_score
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), torch.finfo(attn_scores.dtype).min)
        attn_weights = F.softmax(attn_scores, dim=-1).to(value.dtype)
        output = torch.matmul(attn_weights, value).squeeze(1)
        return self.output(self.dropout(output))


def build_moe_expert(expert_type, hidden_size, expert_hidden_size, end_k=-1, dropout=0.0):
    expert_type = resolve_moe_expert_type(None, expert_type)
    if expert_type == "mlp":
        return FeedForwardExpert(hidden_size, expert_hidden_size, dropout=dropout)
    if expert_type == "attention":
        return AttentionExpert(hidden_size, expert_hidden_size, end_k=end_k, dropout=dropout)
    raise ValueError(f"Unsupported moe_expert_type: {expert_type}")


class TopKMoELayer(MaskedContextMixin, nn.Module):
    def __init__(
        self,
        hidden_size,
        end_k=-1,
        num_experts=4,
        top_k=2,
        expert_hidden_size=0,
        expert_hidden_multiplier=1.0,
        dropout=0.0,
        router_temperature=1.0,
        router_jitter_noise=0.0,
        load_balance_weight=0.01,
        expert_type="mlp",
        balance_loss_mode="fixed",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.end_k = end_k
        self.expert_type = resolve_moe_expert_type(None, expert_type)
        self.num_experts = _positive_int(num_experts, "num_experts")
        self.top_k = min(_positive_int(top_k, "top_k"), self.num_experts)
        self.router_temperature = max(float(router_temperature), 1e-6)
        self.router_jitter_noise = float(router_jitter_noise)
        self.load_balance_weight = float(load_balance_weight)
        self.balance_loss_mode = resolve_moe_balance_loss_mode(None, balance_loss_mode)
        self.load_balance_loss = None
        self.load_balance_raw_loss = None
        self.load_balance_weighted_loss = None
        self.load_balance_effective_weight = None
        if self.balance_loss_mode == "uncertainty":
            initial_weight = max(self.load_balance_weight, 1e-8)
            self.load_balance_log_var = nn.Parameter(
                torch.tensor(-math.log(initial_weight), dtype=torch.float32)
            )
        expert_size = _expert_hidden_size(hidden_size, expert_hidden_size, expert_hidden_multiplier)

        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                build_moe_expert(self.expert_type, hidden_size, expert_size, end_k=end_k, dropout=dropout)
                for _ in range(self.num_experts)
            ]
        )

    def switch_load_balancing_raw_loss(self, router_probs, top_indices):
        if router_probs.numel() == 0:
            return router_probs.new_zeros(())

        selected_mask = F.one_hot(top_indices, num_classes=self.num_experts).to(router_probs.dtype)
        tokens_per_expert = selected_mask.mean(dim=tuple(range(selected_mask.dim() - 1))).detach()
        router_prob_per_expert = router_probs.mean(dim=0)
        return self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

    def weight_load_balancing_loss(self, raw_loss):
        if self.balance_loss_mode == "fixed":
            effective_weight = raw_loss.new_tensor(self.load_balance_weight)
            weighted_loss = raw_loss * effective_weight
        elif self.balance_loss_mode == "uncertainty":
            log_var = self.load_balance_log_var.to(device=raw_loss.device, dtype=raw_loss.dtype)
            effective_weight = torch.exp(-log_var)
            weighted_loss = effective_weight * raw_loss + log_var
        else:
            raise ValueError(f"Unsupported balance_loss_mode: {self.balance_loss_mode}")

        self.load_balance_raw_loss = raw_loss
        self.load_balance_weighted_loss = weighted_loss
        self.load_balance_effective_weight = effective_weight
        return weighted_loss

    def switch_load_balancing_loss(self, router_probs, top_indices):
        if self.balance_loss_mode == "fixed" and self.load_balance_weight <= 0:
            zero = router_probs.new_zeros(())
            self.load_balance_raw_loss = zero
            self.load_balance_weighted_loss = zero
            self.load_balance_effective_weight = zero
            return zero
        raw_loss = self.switch_load_balancing_raw_loss(router_probs, top_indices)
        return self.weight_load_balancing_loss(raw_loss)

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        self.load_balance_loss = None
        self.load_balance_raw_loss = None
        self.load_balance_weighted_loss = None
        self.load_balance_effective_weight = None
        context = self.pool_context(hidden_states, attention_mask, thought_id_idx)
        router_logits = self.router(context)
        if self.training and self.router_jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise

        router_probs = F.softmax(router_logits.float() / self.router_temperature, dim=-1)
        top_scores, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_scores = top_scores.to(context.dtype)
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        if self.training:
            self.load_balance_loss = self.switch_load_balancing_loss(router_probs, top_indices)

        expert_outputs = torch.stack(
            [
                expert(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    thought_id_idx=thought_id_idx,
                    context=context,
                )
                for expert in self.experts
            ],
            dim=1,
        )
        gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        selected_outputs = expert_outputs.gather(dim=1, index=gather_indices)
        return (selected_outputs * top_scores.unsqueeze(-1)).sum(dim=1)


class DeepSeekMoELayer(TopKMoELayer):
    def __init__(
        self,
        hidden_size,
        end_k=-1,
        num_experts=4,
        top_k=2,
        expert_hidden_size=0,
        expert_hidden_multiplier=1.0,
        dropout=0.0,
        router_temperature=1.0,
        router_jitter_noise=0.0,
        fine_grained_factor=2,
        num_shared_experts=1,
        shared_expert_weight=1.0,
        load_balance_weight=0.01,
        expert_type="mlp",
        balance_loss_mode="fixed",
    ):
        self.base_num_experts = _positive_int(num_experts, "num_experts")
        self.fine_grained_factor = _positive_int(fine_grained_factor, "fine_grained_factor")
        self.num_shared_experts = max(0, int(num_shared_experts))
        self.shared_expert_weight = float(shared_expert_weight)
        self.expert_type = resolve_moe_expert_type(None, expert_type)
        routed_experts = self.base_num_experts * self.fine_grained_factor
        routed_top_k = min(_positive_int(top_k, "top_k") * self.fine_grained_factor, routed_experts)
        expert_size = _expert_hidden_size(
            hidden_size,
            expert_hidden_size,
            expert_hidden_multiplier,
            divisor=self.fine_grained_factor,
        )

        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.end_k = end_k
        self.num_experts = routed_experts
        self.top_k = routed_top_k
        self.router_temperature = max(float(router_temperature), 1e-6)
        self.router_jitter_noise = float(router_jitter_noise)
        self.load_balance_weight = float(load_balance_weight)
        self.balance_loss_mode = resolve_moe_balance_loss_mode(None, balance_loss_mode)
        self.load_balance_loss = None
        self.load_balance_raw_loss = None
        self.load_balance_weighted_loss = None
        self.load_balance_effective_weight = None
        if self.balance_loss_mode == "uncertainty":
            initial_weight = max(self.load_balance_weight, 1e-8)
            self.load_balance_log_var = nn.Parameter(
                torch.tensor(-math.log(initial_weight), dtype=torch.float32)
            )
        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                build_moe_expert(self.expert_type, hidden_size, expert_size, end_k=end_k, dropout=dropout)
                for _ in range(self.num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                build_moe_expert(self.expert_type, hidden_size, expert_size, end_k=end_k, dropout=dropout)
                for _ in range(self.num_shared_experts)
            ]
        )

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        routed_output = super().forward(hidden_states, attention_mask, thought_id_idx)
        if self.num_shared_experts == 0 or self.shared_expert_weight == 0:
            return routed_output

        context = self.pool_context(hidden_states, attention_mask, thought_id_idx)
        shared_output = torch.stack(
            [
                expert(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    thought_id_idx=thought_id_idx,
                    context=context,
                )
                for expert in self.shared_experts
            ],
            dim=0,
        ).mean(dim=0)
        return routed_output + self.shared_expert_weight * shared_output


def build_latent_layer(config, attention_cls):
    layer_type = resolve_latent_layer_type(config, getattr(config, "latent_layer_type", "attention"))
    end_k = getattr(config, "latent_end_k", -1)
    if layer_type == "attention":
        return attention_cls(config.hidden_size, end_k=end_k)
    if layer_type == "topk_moe":
        return TopKMoELayer(
            hidden_size=config.hidden_size,
            end_k=end_k,
            expert_type=getattr(config, "moe_expert_type", "mlp"),
            num_experts=getattr(config, "moe_num_experts", 4),
            top_k=getattr(config, "moe_top_k", 2),
            expert_hidden_size=getattr(config, "moe_expert_hidden_size", 0),
            expert_hidden_multiplier=getattr(config, "moe_expert_hidden_multiplier", 1.0),
            dropout=getattr(config, "moe_dropout", 0.0),
            router_temperature=getattr(config, "moe_router_temperature", 1.0),
            router_jitter_noise=getattr(config, "moe_router_jitter_noise", 0.0),
            load_balance_weight=getattr(config, "moe_load_balance_weight", 0.0),
            balance_loss_mode=getattr(config, "moe_balance_loss_mode", "fixed"),
        )
    if layer_type == "deepseek_moe":
        return DeepSeekMoELayer(
            hidden_size=config.hidden_size,
            end_k=end_k,
            expert_type=getattr(config, "moe_expert_type", "mlp"),
            num_experts=getattr(config, "moe_num_experts", 4),
            top_k=getattr(config, "moe_top_k", 2),
            expert_hidden_size=getattr(config, "moe_expert_hidden_size", 0),
            expert_hidden_multiplier=getattr(config, "moe_expert_hidden_multiplier", 1.0),
            dropout=getattr(config, "moe_dropout", 0.0),
            router_temperature=getattr(config, "moe_router_temperature", 1.0),
            router_jitter_noise=getattr(config, "moe_router_jitter_noise", 0.0),
            fine_grained_factor=getattr(config, "moe_fine_grained_factor", 2),
            num_shared_experts=getattr(config, "moe_num_shared_experts", 1),
            shared_expert_weight=getattr(config, "moe_shared_expert_weight", 1.0),
            load_balance_weight=getattr(config, "moe_load_balance_weight", 0.0),
            balance_loss_mode=getattr(config, "moe_balance_loss_mode", "fixed"),
        )
    raise ValueError(f"Unsupported latent_layer_type: {layer_type}")

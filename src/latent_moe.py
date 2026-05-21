import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_LAYER_TYPES = {"auto", "attention", "topk_moe", "deepseek_moe"}


def _positive_int(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def resolve_latent_layer_type(config, latent_layer_type="auto"):
    if latent_layer_type == "auto":
        return getattr(config, "latent_layer_type", "attention")
    if latent_layer_type not in LATENT_LAYER_TYPES:
        raise ValueError(f"latent_layer_type must be one of {sorted(LATENT_LAYER_TYPES)}, got {latent_layer_type}")
    return latent_layer_type


def set_latent_layer_config(
    config,
    latent_layer_type="auto",
    end_k=-1,
    moe_num_experts=4,
    moe_top_k=2,
    moe_expert_hidden_size=0,
    moe_expert_hidden_multiplier=1.0,
    moe_dropout=0.0,
    moe_router_temperature=1.0,
    moe_router_jitter_noise=0.0,
    moe_fine_grained_factor=2,
    moe_num_shared_experts=1,
    moe_shared_expert_weight=1.0,
):
    latent_layer_type = resolve_latent_layer_type(config, latent_layer_type)
    config.latent_layer_type = latent_layer_type
    config.latent_end_k = int(end_k)
    config.moe_num_experts = _positive_int(moe_num_experts, "moe_num_experts")
    config.moe_top_k = _positive_int(moe_top_k, "moe_top_k")
    config.moe_expert_hidden_size = int(moe_expert_hidden_size)
    config.moe_expert_hidden_multiplier = float(moe_expert_hidden_multiplier)
    config.moe_dropout = float(moe_dropout)
    config.moe_router_temperature = float(moe_router_temperature)
    config.moe_router_jitter_noise = float(moe_router_jitter_noise)
    config.moe_fine_grained_factor = _positive_int(moe_fine_grained_factor, "moe_fine_grained_factor")
    config.moe_num_shared_experts = int(moe_num_shared_experts)
    config.moe_shared_expert_weight = float(moe_shared_expert_weight)
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

    def forward(self, x):
        return self.net(x)


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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.end_k = end_k
        self.num_experts = _positive_int(num_experts, "num_experts")
        self.top_k = min(_positive_int(top_k, "top_k"), self.num_experts)
        self.router_temperature = max(float(router_temperature), 1e-6)
        self.router_jitter_noise = float(router_jitter_noise)
        expert_size = _expert_hidden_size(hidden_size, expert_hidden_size, expert_hidden_multiplier)

        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForwardExpert(hidden_size, expert_size, dropout=dropout) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        context = self.pool_context(hidden_states, attention_mask, thought_id_idx)
        router_logits = self.router(context)
        if self.training and self.router_jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise

        router_scores = F.softmax(router_logits.float() / self.router_temperature, dim=-1).to(context.dtype)
        top_scores, top_indices = torch.topk(router_scores, k=self.top_k, dim=-1)
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        expert_outputs = torch.stack([expert(context) for expert in self.experts], dim=1)
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
    ):
        self.base_num_experts = _positive_int(num_experts, "num_experts")
        self.fine_grained_factor = _positive_int(fine_grained_factor, "fine_grained_factor")
        self.num_shared_experts = max(0, int(num_shared_experts))
        self.shared_expert_weight = float(shared_expert_weight)
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
        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForwardExpert(hidden_size, expert_size, dropout=dropout) for _ in range(self.num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [FeedForwardExpert(hidden_size, expert_size, dropout=dropout) for _ in range(self.num_shared_experts)]
        )

    def forward(self, hidden_states, attention_mask, thought_id_idx):
        routed_output = super().forward(hidden_states, attention_mask, thought_id_idx)
        if self.num_shared_experts == 0 or self.shared_expert_weight == 0:
            return routed_output

        context = self.pool_context(hidden_states, attention_mask, thought_id_idx)
        shared_output = torch.stack([expert(context) for expert in self.shared_experts], dim=0).mean(dim=0)
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
            num_experts=getattr(config, "moe_num_experts", 4),
            top_k=getattr(config, "moe_top_k", 2),
            expert_hidden_size=getattr(config, "moe_expert_hidden_size", 0),
            expert_hidden_multiplier=getattr(config, "moe_expert_hidden_multiplier", 1.0),
            dropout=getattr(config, "moe_dropout", 0.0),
            router_temperature=getattr(config, "moe_router_temperature", 1.0),
            router_jitter_noise=getattr(config, "moe_router_jitter_noise", 0.0),
        )
    if layer_type == "deepseek_moe":
        return DeepSeekMoELayer(
            hidden_size=config.hidden_size,
            end_k=end_k,
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
        )
    raise ValueError(f"Unsupported latent_layer_type: {layer_type}")

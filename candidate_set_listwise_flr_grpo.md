# Candidate-Set Listwise FLR-GRPO 改造方案

## 总体思路

**核心原则**：不做昂贵的 beam-search listwise GRPO；先做 candidate-set listwise reward。

每个样本构造一个小候选集合：positive item + random / hard negatives，用当前 latent thought 对这些候选 item title 打分，然后用 positive 在候选集合里的排序质量作为 GRPO reward。

**优势**：
- 不需要重写 decoding
- 不需要每一步跑 constrained beam search
- 只改 `grpo_trainer.py`、`train_noise_grpo.py`，最多再加一点 sampler / dataset repeat 逻辑

---

## Reward 设计

### 原始 reward

```python
reward = -exp(-sum(logp) / labels_length)
```

### 新 reward

```python
reward = 1.0 * dcg_reward
       + 0.5 * softmax_positive_reward
       + 0.5 * margin_reward
       + 0.05 * avg_logp
```

| 分量 | 定义 |
|------|------|
| `dcg_reward` | positive item 在候选集里排第 r 名，则 `reward = 1 / log2(r + 1)` |
| `softmax_positive_reward` | `softmax(candidate_scores)[positive]` |
| `margin_reward` | `sigmoid(score_positive - max(score_negative))` |
| `avg_logp` | 保留原始 token log-prob，稳定训练 |

---

## 需要修改的文件

- `src/grpo_attention_tuning/train_noise_grpo.py`
- `src/grpo_attention_tuning/grpo_trainer.py`
- `scripts/attention_grpo.sh`

---

## 第 0 步：修正当前 GRPO Grouping

**问题**：当前 `NoiseGRPORecTrainer` 默认把 batch 里的样本按 `num_generations` 分组（`rewards.view(-1, self.num_generations)`），但 dataset / sampler 并没有保证这 `num_generations` 条是同一个 prompt 的不同 noise perturbations，实际上是把不同用户样本硬凑成一组，导致 GRPO advantage 不干净。

**修复**：把 batch 结构固定成——同一个 user prompt 重复 G 次，第 1 条不加 noise 作为 baseline，第 2–G 条加不同 Gaussian noise。

**最简实现**：
```
per_device_train_batch_size = num_generations
```
每个 batch 只放一个 user prompt 的 G 个 repeat，先牺牲一点吞吐，但实现最稳。

---

## 第 1 步：修改 `train_noise_grpo.py`

### 新增参数

```python
use_listwise_reward: bool = True
num_listwise_negatives: int = 15
listwise_topk: int = 10
listwise_tau: float = 1.0
reward_w_dcg: float = 1.0
reward_w_soft: float = 0.5
reward_w_margin: float = 0.5
reward_w_logp: float = 0.05
noise_std: float = 1.0
```

### 传参给 trainer

```python
trainer_kwargs.update({
    "info_file": info_file,
    "use_listwise_reward": use_listwise_reward,
    "num_listwise_negatives": num_listwise_negatives,
    "listwise_topk": listwise_topk,
    "listwise_tau": listwise_tau,
    "reward_w_dcg": reward_w_dcg,
    "reward_w_soft": reward_w_soft,
    "reward_w_margin": reward_w_margin,
    "reward_w_logp": reward_w_logp,
    "noise_std": noise_std,
})
```

### Batch 设置

```python
per_device_train_batch_size = num_generations
```

---

## 第 2 步：Repeat Dataset

在构造完 `hf_train_dataset` 后添加 repeat 函数，确保每个原始样本连续出现 G 次：

```python
def repeat_dataset_for_grpo(ds, num_generations):
    new_data = {k: [] for k in ds.column_names}
    for ex in ds:
        for _ in range(num_generations):
            for k in ds.column_names:
                new_data[k].append(ex[k])
    return HFDataset.from_dict(new_data)

hf_train_dataset = repeat_dataset_for_grpo(hf_train_dataset, num_generations)
hf_val_dataset   = repeat_dataset_for_grpo(hf_val_dataset, num_generations)
```

> **注意**：仅 repeat 数据集还不够，`RandomSampler` 会把重复样本打散，还需修改 trainer sampler（见下一步）。

---

## 第 3 步：修改 `grpo_trainer.py` 的 Sampler

### 新增 `GroupedRandomSampler`

```python
from torch.utils.data import Sampler
import torch

class GroupedRandomSampler(Sampler):
    def __init__(self, data_source, group_size, seed=42, world_size=1, rank=0):
        self.data_source = data_source
        self.group_size = group_size
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        self.num_groups = len(data_source) // group_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        group_ids = torch.randperm(self.num_groups, generator=g).tolist()

        if self.world_size > 1:
            group_ids = group_ids[self.rank::self.world_size]

        for gid in group_ids:
            base = gid * self.group_size
            for j in range(self.group_size):
                yield base + j

    def __len__(self):
        if self.world_size > 1:
            return (self.num_groups + self.world_size - 1) // self.world_size * self.group_size
        return self.num_groups * self.group_size
```

### 覆写 `_get_train_sampler`

```python
def _get_train_sampler(self, train_dataset):
    if train_dataset is None:
        return None
    return GroupedRandomSampler(
        train_dataset,
        group_size=self.num_generations,
        seed=self.args.seed,
        world_size=self.args.world_size,
        rank=self.args.process_index,
    )
```

每个 batch 保证结构为：
```
sample A repeat 1
sample A repeat 2
...
sample A repeat G
```

---

## 第 4 步：修改 Noise 注入逻辑

**原来**：

```python
noise = torch.randn(
    (batch_size, original_embeds.size(-1)), device=self.model.device
).mul(1.5)
noise[0, :] = 0
```

**改为**：

```python
noise = torch.randn(
    (batch_size, original_embeds.size(-1)), device=self.model.device
).mul(self.noise_std)

noise[torch.arange(0, batch_size, self.num_generations, device=noise.device), :] = 0
```

每个 group 的第一条为 baseline（noise=0）。

---

## 第 5 步：重写 Advantage 计算

**原来**（绕）：

```python
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards  = rewards.view(-1, self.num_generations).std(dim=1)
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards  = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
temp_rewards = rewards.clone().view(-1, self.num_generations)
xx = temp_rewards[:, 0].unsqueeze(1).expand_as(temp_rewards).reshape(-1)
xx = swap_adjacent_blocks(xx, self.num_generations)
baseline_b = xx.mean()
advantages_raw  = rewards - baseline_b
advantages_norm = torch.norm(advantages_raw) + 1e-6
advantages      = advantages_raw / advantages_norm
```

**改为**（标准 group-relative baseline）：

```python
rewards_group = rewards.view(-1, self.num_generations)

baseline       = rewards_group[:, :1]
advantages_raw = rewards_group - baseline

advantages_norm = torch.norm(
    advantages_raw[:, 1:], dim=1, keepdim=True
) + 1e-6

advantages = advantages_raw / advantages_norm
advantages = advantages.reshape(-1)
```

每个 noise sample 只和同一个 prompt 的 baseline 比。

---

## 第 6 步：加载 Item Catalog

### `__init__` 新增参数

```python
info_file=None,
use_listwise_reward=False,
num_listwise_negatives=15,
listwise_topk=10,
listwise_tau=1.0,
reward_w_dcg=1.0,
reward_w_soft=0.5,
reward_w_margin=0.5,
reward_w_logp=0.05,
noise_std=1.0,
```

### 加载候选 item title

```python
def _load_catalog_candidate_ids(self, info_file):
    candidate_ids = []
    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            title = line.split("\t")[0].strip()
            text  = f"\"{title}\"\n"
            ids   = self.processing_class.encode(text, add_special_tokens=False)
            if len(ids) == 0:
                continue
            if ids[-1] != self.processing_class.eos_token_id:
                ids.append(self.processing_class.eos_token_id)
            candidate_ids.append(ids)
    return candidate_ids
```

在 `__init__` 中调用：

```python
if self.use_listwise_reward:
    assert info_file is not None
    self.catalog_candidate_ids = self._load_catalog_candidate_ids(info_file)
```

---

## 第 7 步：新增 Candidate Scoring 函数

```python
@torch.no_grad()
def _score_candidate_set(
    self,
    prompt_ids,
    prompt_mask,
    noise,
    positive_completion_ids,
    positive_completion_mask,
):
    device     = prompt_ids.device
    batch_size = prompt_ids.size(0)

    pos_ids   = positive_completion_ids[0][positive_completion_mask[0].bool()].tolist()
    pos_tuple = tuple(pos_ids)

    # 采样 random negatives
    negs  = []
    tries = 0
    while len(negs) < self.num_listwise_negatives and tries < self.num_listwise_negatives * 20:
        tries += 1
        idx  = torch.randint(0, len(self.catalog_candidate_ids), (1,), device=device).item()
        cand = self.catalog_candidate_ids[idx]
        if tuple(cand) != pos_tuple:
            negs.append(cand)

    candidate_lists = [pos_ids] + negs
    num_candidates  = len(candidate_lists)
    max_cand_len    = max(len(x) for x in candidate_lists)

    # 填充候选 tensor
    cand_ids  = torch.full((num_candidates, max_cand_len),
                            fill_value=self.processing_class.pad_token_id,
                            device=device, dtype=prompt_ids.dtype)
    cand_mask = torch.zeros((num_candidates, max_cand_len),
                             device=device, dtype=prompt_mask.dtype)
    for i, ids in enumerate(candidate_lists):
        ids_tensor         = torch.tensor(ids, device=device, dtype=prompt_ids.dtype)
        cand_ids[i, :len(ids)]  = ids_tensor
        cand_mask[i, :len(ids)] = 1

    prompt_len = prompt_ids.size(1)
    scores     = []

    # 分 chunk 打分，避免 OOM
    for start in range(0, num_candidates, 8):
        end = min(start + 8, num_candidates)
        c   = end - start

        prompt_rep      = prompt_ids.unsqueeze(1).expand(batch_size, c, prompt_len).reshape(batch_size * c, prompt_len)
        prompt_mask_rep = prompt_mask.unsqueeze(1).expand(batch_size, c, prompt_len).reshape(batch_size * c, prompt_len)
        cand_rep        = cand_ids[start:end].unsqueeze(0).expand(batch_size, c, max_cand_len).reshape(batch_size * c, max_cand_len)
        cand_mask_rep   = cand_mask[start:end].unsqueeze(0).expand(batch_size, c, max_cand_len).reshape(batch_size * c, max_cand_len)

        input_ids      = torch.cat([prompt_rep, cand_rep], dim=1)
        attention_mask = torch.cat([prompt_mask_rep, cand_mask_rep], dim=1)

        embeds = self.model.generate_embs(input_ids, attention_mask)

        thought_id       = self.model.model.embed_tokens.num_embeddings - 1
        where_thought_ids = torch.nonzero(input_ids == thought_id)
        expanded_noise   = noise.unsqueeze(1).expand(batch_size, c, noise.size(-1)).reshape(batch_size * c, noise.size(-1))
        embeds[
            torch.arange(batch_size * c, device=device),
            where_thought_ids[:, 1],
        ] += expanded_noise

        outputs = self.model(
            input_ids=None, inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_attentions=False, output_hidden_states=False, use_cache=False,
        )

        logits     = outputs.logits[:, :-1, :] / self.listwise_tau
        cand_logits = logits[:, prompt_len - 1: prompt_len - 1 + max_cand_len, :]
        log_probs  = torch.nn.functional.log_softmax(cand_logits, dim=-1)
        token_logps = log_probs.gather(dim=-1, index=cand_rep.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * cand_mask_rep
        lengths     = cand_mask_rep.sum(dim=-1).clamp(min=1)
        seq_scores  = token_logps.sum(dim=-1) / lengths

        scores.append(seq_scores.view(batch_size, c))

    scores = torch.cat(scores, dim=1)
    return scores
```

> candidate chunk size 暂定 8，后续可抽成参数。

---

## 第 8 步：新增 Listwise Reward 函数

```python
def _compute_listwise_reward(self, candidate_scores, avg_logp):
    # candidate_scores: [B, C]，index 0 为 positive
    pos_score  = candidate_scores[:, 0]
    neg_scores = candidate_scores[:, 1:]

    rank = 1 + (neg_scores > pos_score.unsqueeze(1)).sum(dim=1)

    dcg_reward = torch.where(
        rank <= self.listwise_topk,
        1.0 / torch.log2(rank.float() + 1.0),
        torch.zeros_like(rank.float()),
    )

    soft_reward   = torch.softmax(candidate_scores / self.listwise_tau, dim=-1)[:, 0]
    hardest_neg   = neg_scores.max(dim=1).values
    margin_reward = torch.sigmoid((pos_score - hardest_neg) / self.listwise_tau)

    reward = (
        self.reward_w_dcg    * dcg_reward
        + self.reward_w_soft   * soft_reward
        + self.reward_w_margin * margin_reward
        + self.reward_w_logp   * avg_logp
    )

    return reward, {
        "rank":          rank.float(),
        "dcg_reward":    dcg_reward,
        "soft_reward":   soft_reward,
        "margin_reward": margin_reward,
    }
```

---

## 第 9 步：替换 `_generate_and_score_completions` 里的 Reward

**原来**：

```python
alpha = 0.1
beta_reward = 1.0
avg_logp = old_per_token_logps.clone().sum(dim=-1) / labels_length

seq_exact    = (pred_ids == completion_ids).all(dim=-1).float()
reward_alpha = alpha * avg_logp
reward_beta  = beta_reward * seq_exact
output_reward = reward_alpha + reward_beta
```

**改为**：

```python
avg_logp  = old_per_token_logps.clone().sum(dim=-1) / labels_length
seq_exact = (pred_ids == completion_ids).all(dim=-1).float()

if self.use_listwise_reward:
    candidate_scores = self._score_candidate_set(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        noise=noise,
        positive_completion_ids=completion_ids,
        positive_completion_mask=completion_mask,
    )
    output_reward, listwise_logs = self._compute_listwise_reward(
        candidate_scores=candidate_scores,
        avg_logp=avg_logp,
    )
    reward_alpha = self.reward_w_logp * avg_logp
    reward_beta  = listwise_logs["dcg_reward"]

else:
    alpha         = 0.1
    beta_reward   = 1.0
    reward_alpha  = alpha * avg_logp
    reward_beta   = beta_reward * seq_exact
    output_reward = reward_alpha + reward_beta
```

在 return dict 里增加 logging：

```python
if self.use_listwise_reward:
    extra_outputs = {
        "listwise_rank":  listwise_logs["rank"],
        "dcg_reward":     listwise_logs["dcg_reward"],
        "soft_reward":    listwise_logs["soft_reward"],
        "margin_reward":  listwise_logs["margin_reward"],
    }
else:
    extra_outputs = {}

# return 时添加 **extra_outputs
```

---

## 第 10 步：修复 `loss_dict` 未初始化 Bug

在 `compute_loss` 里，`loss_dict.update(...)` 之前先初始化，避免 FLR regularization 那段报 `NameError: loss_dict is not defined`：

```python
total_loss = grpo_loss
loss_dict = {
    "grpo_loss": grpo_loss.item() if hasattr(grpo_loss, "item") else grpo_loss,
}
```

---

## 第 11 步：修改 `scripts/attention_grpo.sh`

保守起始参数（16 candidates）：

```bash
--num_generations 8 \
--micro_batch_size 8 \
--use_listwise_reward True \
--num_listwise_negatives 15 \
--listwise_topk 10 \
--listwise_tau 1.0 \
--reward_w_dcg 1.0 \
--reward_w_soft 0.5 \
--reward_w_margin 0.5 \
--reward_w_logp 0.05 \
--noise_std 1.0
```

跑通后再升级到 32 candidates（`num_listwise_negatives=31`）。

---

## 实验计划

### 运行顺序

| Step | 规模 | 目标 |
|------|------|------|
| 1 | sample=128 | 验证代码能跑通 |
| 2 | sample=2000 | 完整 GRPO 小实验 |
| 3 | full valid/test | 评估 HR@5/10、NDCG@5/10 |
| 4 | — | 与原始 FLR-GRPO 对比 |

> 注意：listwise reward 每步需额外 score candidates，速度比原版慢，**第一轮不要急着跑全量**。

### Ablation 设计

| 组 | Reward 构成 |
|----|-------------|
| A | 原始 FLR-GRPO（baseline） |
| B | Listwise: dcg only |
| C | Listwise: soft + margin only |
| D | Listwise: dcg + soft + margin |
| E | Listwise: dcg + soft + margin + avg_logp |

预计最稳的是 **E**。

---

## 推荐最终参数

```python
# Reward 权重
reward_w_dcg    = 1.0
reward_w_soft   = 0.5
reward_w_margin = 0.5
reward_w_logp   = 0.05

# 训练参数
num_generations        = 8
num_listwise_negatives = 15
listwise_topk          = 10
listwise_tau           = 1.0
noise_std              = 1.0
micro_batch_size       = 8
```

---

## 设计理由

原来的 reward 是真实 item title 的 likelihood / exact match；新的 reward 是真实 item 在 candidate list 里的**相对排序**。这更接近最终评估的 HR@K / NDCG@K——论文里的评估本来就是 constrained generative top-K ranking，而不是单条 title 的 pointwise likelihood，因此这个改法与评估目标天然对齐。

**第一版有意不做的事**（后续增强）：
- beam-search listwise reward
- hard negative cache
- factor-wise reward
- full catalog scoring

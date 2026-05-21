# latent-moe

This repository extends latent-reasoning recommendation experiments with MoE latent layers and prompt-format ablations.

The current cluster-ready experiment set contains three four-dataset SFT-style ablations:

- `topk_moe`: top-k routed MoE latent layer.
- `deepseek_moe`: DeepSeek-style MoE latent layer with fine-grained and shared experts.
- `explicit_cot_prompt`: prompt-format ablation using explicit interaction statistics before `<|Thought|>`.

All three launchers use the same four datasets:

```text
CDs_and_Vinyl
Musical_Instruments
Toys_and_Games
Video_Games
```

## Directory Structure

```text
.
|-- data/
|   |-- train/
|   |-- valid/
|   |-- test/
|   `-- info/
|-- models/
|   `-- Qwen2.5-1.5B/
|-- output_dir/
|-- scripts/
|   |-- latent_train_four_datasets.sh
|   |-- r3_8gpu_topk_jsub.sh
|   |-- r3_8gpu_deepseek_moe_jsub.sh
|   |-- r3_8gpu_explicit_cot_prompt_jsub.sh
|   |-- r3_8gpu_jsub.sh
|   |-- latent_train.sh
|   |-- attention_grpo.sh
|   `-- grpo_eval.sh
|-- src/
|   |-- latent/
|   |-- grpo_attention_tuning/
|   |-- utils/
|   |-- latent_moe.py
|   `-- fsdp2_utils.py
|-- environment.yaml
`-- README.md
```

## Environment

Create the environment from `environment.yaml`, then activate it before running scripts:

```bash
conda env create -n trl -f environment.yaml
conda activate trl
```

The latent training entrypoint is configured to use PyTorch FSDP2 through Transformers/Accelerate. The relevant code path is:

- `src/fsdp2_utils.py`
- `src/latent/latent_attention_train.py`

## Data Layout

The four-dataset scripts expect each category to have matching files under:

```text
data/train/<Category>*11.csv
data/valid/<Category>*11.csv
data/test/<Category>*11.csv
data/info/<Category>*.txt
```

Current expected categories:

```text
CDs_and_Vinyl
Musical_Instruments
Toys_and_Games
Video_Games
```

## Four-Dataset 8-GPU Experiments

All JSUB launchers use the Jingxing/UniScheduler-style 8-GPU header:

```bash
#JSUB -q gpu_6000ada
#JSUB -n 1
#JSUB -gpgpu 8
#JSUB -R span[ptile=8]
```

They also load the cluster environment with:

```bash
module load mpich/4.3.0b1-gcc-11.3.1
source activate trl
```

### Top-K MoE

```bash
jsub < /data/gtq/latent-moe/scripts/r3_8gpu_topk_jsub.sh
```

Defaults:

```bash
LATENT_LAYER_TYPE=topk_moe
LATENT_EXPERIMENT_NAME=topk_moe
PROMPT_STYLE=default
MOE_NUM_EXPERTS=4
MOE_TOP_K=2
```

Output:

```text
output_dir/<Category>/topk_moe/
```

### DeepSeek-MoE

```bash
jsub < /data/gtq/latent-moe/scripts/r3_8gpu_deepseek_moe_jsub.sh
```

Defaults:

```bash
LATENT_LAYER_TYPE=deepseek_moe
LATENT_EXPERIMENT_NAME=deepseek_moe
PROMPT_STYLE=default
MOE_NUM_EXPERTS=4
MOE_TOP_K=2
MOE_FINE_GRAINED_FACTOR=2
MOE_NUM_SHARED_EXPERTS=1
MOE_SHARED_EXPERT_WEIGHT=1.0
```

Output:

```text
output_dir/<Category>/deepseek_moe/
```

Only `deepseek_moe` consumes the fine-grained and shared-expert parameters. The `topk_moe` path only uses the router/top-k MoE parameters.

### Explicit-CoT Prompt Ablation

```bash
jsub < /data/gtq/latent-moe/scripts/r3_8gpu_explicit_cot_prompt_jsub.sh
```

Defaults:

```bash
LATENT_LAYER_TYPE=attention
LATENT_EXPERIMENT_NAME=explicit_cot_prompt
PROMPT_STYLE=explicit_cot
```

Output:

```text
output_dir/<Category>/explicit_cot_prompt/
```

The explicit prompt format is built in `src/latent/reasoning_dataset.py`. It uses item ids, ratings, elapsed days, repeat counts, and an explicit constraint before appending `<|Thought|>`.

Example format:

```text
### User Input:
Category: musics
User interaction history (oldest -> newest):
1) [item_id=2956, dt_days=0, rating=5.0] title="Doris Day's Greatest Hits"
2) [item_id=5031, dt_days=0, rating=5.0] title="Doris Day - 100 Hits Legends"

### Explicit CoT:
- L=2, unique_cnt=2, repeat_cnt=0
- seen_item_ids={2956, 5031}
- avg_rating=5.0, last_rating=5.0
- dt_days=[0, 0]
- Constraint: output title NOT in seen_titles

Task: Predict the next item the user will buy. Output exactly ONE candidate item title in double quotes.

### Response:
<|Thought|>
```

## Shared Four-Dataset Script

The three experiment launchers call:

```bash
scripts/latent_train_four_datasets.sh
```

This script runs the four categories sequentially. For each category it:

1. Runs `src/latent/latent_attention_train.py` with `torchrun --nproc_per_node=8`.
2. Splits the test file into 8 shards.
3. Runs evaluation on 8 visible GPUs.
4. Merges JSON results.
5. Calls `src/utils/calc.py`.

You can override common knobs at submit time, for example:

```bash
MOE_NUM_EXPERTS=8 MOE_TOP_K=2 jsub < scripts/r3_8gpu_topk_jsub.sh
```

For a non-default model path:

```bash
BASE_MODEL=/path/to/Qwen2.5-1.5B jsub < scripts/r3_8gpu_deepseek_moe_jsub.sh
```

## Logs and Outputs

JSUB-level logs:

```text
logs/r3_8gpu_topk_moe_four_datasets_<timestamp>.log
logs/r3_8gpu_deepseek_moe_four_datasets_<timestamp>.log
logs/r3_8gpu_explicit_cot_prompt_four_datasets_<timestamp>.log
```

Training logs and checkpoints:

```text
output_dir/<Category>/<ExperimentName>/
```

Evaluation split files:

```text
output_dir/<Category>_<ExperimentName>_base/
```

## Legacy Single-Mode Scripts

These scripts are still present for one-off or legacy runs:

```bash
bash scripts/latent_train.sh
bash scripts/attention_grpo.sh
bash scripts/grpo_eval.sh
bash scripts/r3_8gpu_jsub.sh latent
```

The current four-dataset ablation workflow should use the dedicated JSUB scripts listed above.

## Citation

If you find the original R3 paper or code helpful, cite:

```bibtex
@article{latentr3,
  title={Reinforced latent reasoning for llm-based recommendation},
  author={Zhang, Yang and Xu, Wenxin and Zhao, Xiaoyan and Wang, Wenjie and Feng, Fuli and He, Xiangnan and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2505.19092},
  year={2025}
}
```

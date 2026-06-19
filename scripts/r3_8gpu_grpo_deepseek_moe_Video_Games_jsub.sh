#!/bin/bash
#JSUB -J r3_grpo_video
#JSUB -q gpu_6000ada
#JSUB -n 1
#JSUB -gpgpu 8
#JSUB -R span[ptile=8]
#JSUB -e error.%J
#JSUB -o output.%J

set -euo pipefail

export GRPO_CATEGORIES="Video_Games"
exec bash /data/gtq/latent-moe/scripts/r3_8gpu_grpo_deepseek_moe_jsub.sh "$@"

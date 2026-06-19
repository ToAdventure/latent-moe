#!/bin/bash
#JSUB -J r3_topk_toys
#JSUB -q gpu_6000ada
#JSUB -n 1
#JSUB -gpgpu 8
#JSUB -R span[ptile=8]
#JSUB -e error.%J
#JSUB -o output.%J

set -euo pipefail

export LATENT_CATEGORIES="Toys_and_Games"
exec bash /data/gtq/latent-moe/scripts/r3_8gpu_topk_jsub.sh "$@"

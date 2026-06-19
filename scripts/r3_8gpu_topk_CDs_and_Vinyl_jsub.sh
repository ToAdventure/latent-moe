#!/bin/bash
#JSUB -J r3_topk_cd
#JSUB -q gpu_6000ada
#JSUB -n 1
#JSUB -gpgpu 8
#JSUB -R span[ptile=8]
#JSUB -e error.%J
#JSUB -o output.%J

set -euo pipefail

export LATENT_CATEGORIES="CDs_and_Vinyl"
exec bash /data/gtq/latent-moe/scripts/r3_8gpu_topk_jsub.sh "$@"

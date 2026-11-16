#!/usr/bin/env bash
# Learning-rate ablation: three new runs. Run from the repository root.
set -euo pipefail

uv run modal run --detach -m cs336_alignment.launch_modal \
    --learning-rate 2e-5 --seed 1 --run-name lr-2e-5-seed1 &

for seed in 1 2; do
    uv run modal run --detach -m cs336_alignment.launch_modal \
        --learning-rate 5e-6 --seed "$seed" --run-name "lr-5e-6-seed${seed}" &
done
wait

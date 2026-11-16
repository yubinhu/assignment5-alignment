#!/usr/bin/env bash
# Two seeds per alternative prompt; compare with the existing zero-shot runs.
# Run from the repository root.
set -euo pipefail

for prompt in question_only r1_zero_three_shot_gsm8k; do
    for seed in 1 2; do
        uv run modal run --detach -m cs336_alignment.launch_modal \
            --prompt-path "cs336_alignment/prompts/${prompt}.prompt" \
            --learning-rate 1e-5 --seed "$seed" \
            --run-name "prompt-${prompt}-seed${seed}" &
    done
done
wait

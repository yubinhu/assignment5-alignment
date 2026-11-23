#!/usr/bin/env bash
# Run from the repo root. Submit all four variants for each of four seeds.
set -euo pipefail

uv run python - <<'PY'
from cs336_alignment.modal_utils import app, run_command

variants = [
    ("grpo_constant", "mean", "std"),
    ("dr_grpo", "mean", "none"),
    ("rft", "none", "none"),
    ("maxrl", "mean", "mean"),
]
batch_size = 256
max_generation_length = 512
seeds = (1, 2, 3, 4)
with app.run(detach=True):
    print(f"Detached Modal app: {app.app_id}", flush=True)
    for seed in seeds:
        for variant, baseline, normalizer in variants:
            name = f"{variant}-seed{seed}"
            call = run_command.spawn([
                "python", "-u", "-m", "cs336_alignment.train",
                "--run-name", name, "--seed", str(seed),
                "--prompt-path", "cs336_alignment/prompts/r1_zero.prompt",
                "--learning-rate", "1e-5", "--train-batch-size", str(batch_size),
                "--baseline", baseline, "--advantage-normalizer", normalizer,
                "--loss-normalization", "constant",
                "--normalization-constant", str(batch_size * max_generation_length),
            ])
            print(f"Submitted {name}: {call.object_id}", flush=True)
print(f"Modal accepted all {len(seeds) * len(variants)} training calls. You can close your laptop.", flush=True)
PY

import argparse
from pathlib import Path
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.solution import grpo_train_step, load_gsm8k_examples, compute_rollout_rewards
import torch
import random
import wandb
import datetime
import os
from zoneinfo import ZoneInfo


def generation_table(step, prompts, responses, ground_truths):
    """Build a fresh snapshot of generated text for inspection in W&B."""
    table = wandb.Table(columns=[
        "step", "prompt", "response", "ground_truth", "finish_reason",
    ])
    for prompt_text, response, ground_truth in zip(
        prompts, responses, ground_truths, strict=True
    ):
        table.add_data(
            step, prompt_text, response.text, ground_truth,
            response.finish_reason,
        )
    return table


# Parameters
parser = argparse.ArgumentParser(
    description="Run the GRPO training experiment.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model-name", default="allenai/OLMo-2-0425-1B")
parser.add_argument("--prompt-path", default="cs336_alignment/prompts/r1_zero.prompt")
parser.add_argument("--train-file-path", default="data/gsm8k/train.jsonl")
parser.add_argument("--valid-file-path", default="data/gsm8k/test.jsonl")
parser.add_argument("--training-device", default="cuda:0")
parser.add_argument("--inference-gpu", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=1e-5)
parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--n-train-examples", type=int, default=6400)
parser.add_argument("--n-valid-examples", type=int, default=1024)
parser.add_argument("--train-batch-size", type=int, default=256)
parser.add_argument("--group-size", type=int, default=8)
parser.add_argument("--run-name", default=f"exp")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=random.randint(0, 10000))
parser.add_argument("--checkpoint-every", type=int, default=20)
args = parser.parse_args()
checkpoint_path = Path("checkpoints") / f"{args.run_name}.pt"

checkpoint = None
start_step = 0
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Saved settings override defaults; explicit CLI arguments override saved settings.
    saved_args = {key: value for key, value in checkpoint["config"].items()
                  if key in vars(args)}
    args = parser.parse_args(namespace=argparse.Namespace(**saved_args))
    start_step = checkpoint["next_step"]
    args.resumed_from_run = checkpoint["wandb_run_id"]
    args.resumed_from_step = start_step
    print(f"Resuming at step {start_step} with saved settings and CLI overrides.", flush=True)
if Path(args.prompt_path).name == "question_only.prompt":
    reward_fn = question_only_reward_fn
    stop_sequences = None
else:
    reward_fn = r1_zero_reward_fn
    stop_sequences = ["</answer>"]
# Pass vars(args) as the config when initializing a W&B run.
prompt = Path(args.prompt_path).read_text()

# Setup
random.seed(args.seed)
policy, tokenizer = get_model_and_tokenizer(args.model_name, args.training_device)
opitimizer = torch.optim.AdamW(
    policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
)
if checkpoint is not None:
    policy.load_state_dict(checkpoint.pop("model"))
    opitimizer.load_state_dict(checkpoint.pop("optimizer"))
    for param_group in opitimizer.param_groups:
        param_group["lr"] = args.learning_rate
server = VLLMServer(args.model_name, gpu=args.inference_gpu)
run = wandb.init(
    project="grpo_experiments_standard_on_policy",
    config=vars(args),
    name=args.run_name+f"-{datetime.datetime.now
(ZoneInfo("America/Los_Angeles")):%Y%m%d-%H%M}",
)

# Load examples
train_examples = load_gsm8k_examples(args.train_file_path)[:args.n_train_examples]
valid_examples = load_gsm8k_examples(args.valid_file_path)[:args.n_valid_examples]
valid_prompts = [prompt.format(question=example[0]) for example in valid_examples]
valid_ground_truths = [example[1] for example in valid_examples]
groups_per_batch = args.train_batch_size // args.group_size

sample_param = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": args.group_size,
    "seed": args.seed,
    "stop": stop_sequences,
    "include_stop_str_in_output": True,
}

sample_param_valid = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": args.seed,
    "stop": stop_sequences,
    "include_stop_str_in_output": True,
}


def evaluate(step):
    responses = server.generate_completions(
        valid_prompts,
        sample_param_valid,
        batch_size=args.train_batch_size,
    )
    _, meta = compute_rollout_rewards(
        reward_fn, [response.text for response in responses], valid_ground_truths
    )
    meta["avg_resp_len"] = sum(len(response.token_ids) for response in responses) / len(responses)
    meta["cutoff_rate"] = sum(response.finish_reason == "length" for response in responses) / len(responses)
    eval_meta = {f"eval/{key}": value for key, value in meta.items()}
    run.log(eval_meta, step=step)


def save_checkpoint(next_step):
    """Save only after a complete optimizer update; keep the old file until ready."""
    path = checkpoint_path
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": policy.state_dict(),
        "optimizer": opitimizer.state_dict(),
        "next_step": next_step,
        "config": vars(args),
        "wandb_run_id": run.id,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state(args.training_device)
                     if torch.device(args.training_device).type == "cuda" else None),
    }
    # Atomic replacement protects the previous checkpoint from a partial write.
    # On Modal this path still needs a persistent volume to survive container loss.
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    print(f"Saved checkpoint after {next_step} updates: {path}", flush=True)


try:
    server.start()
    server.init_weight_sync(args.training_device)

    # Initialization can consume randomness, so restore RNGs immediately before training.
    if checkpoint is not None:
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint["cuda_rng"] is not None and torch.device(args.training_device).type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], args.training_device)
        del checkpoint

    for i in range(start_step, args.steps):
        # Sync weights
        server.sync_policy_weights(policy)

        # Eval
        if i % 10 == 0:
            evaluate(i)

        # Sample
        batch_examples = random.sample(train_examples, groups_per_batch)
        batch_prompts = [prompt.format(question=example[0]) for example in batch_examples]
        repeated_prompts = [p for p in batch_prompts for _ in range(args.group_size)]
        repeated_truths = [example[1] for example in batch_examples for _ in range(args.group_size)]

        # Rollout
        vllm_out: list[VLLMCompletion] = server.generate_completions(
            batch_prompts, 
            sample_param
        )

        # Train
        loss, meta = grpo_train_step(
            policy, tokenizer, opitimizer, args.gradient_accumulation_steps, args.max_grad_norm, reward_fn, repeated_prompts, [response.text for response in vllm_out], repeated_truths, args.group_size
        )

        meta["cutoff_rate"] = sum([response.finish_reason == "length" for response in vllm_out]) / len(vllm_out)
        meta["loss"] = loss.item()
        if i % 40 == 0:
            meta["generations"] = generation_table(
                i, repeated_prompts[:16], vllm_out[:16], repeated_truths[:16]
            )
        train_meta = {f"train/{key}": value for key, value in meta.items()}
        run.log(train_meta, step=i)
        if (i + 1) % args.checkpoint_every == 0 or i + 1 == args.steps:
            save_checkpoint(i + 1)
    server.sync_policy_weights(policy)
    evaluate(args.steps)

finally:
    server.stop()
    run.finish()


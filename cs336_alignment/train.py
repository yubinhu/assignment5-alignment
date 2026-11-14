import argparse
from pathlib import Path
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.solution import grpo_train_step, load_gsm8k_examples, compute_rollout_rewards
import torch
import random
import wandb
import datetime

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
parser.add_argument("--n-test-examples", type=int, default=1024)
parser.add_argument("--train-batch-size", type=int, default=256)
parser.add_argument("--group-size", type=int, default=8)
args = parser.parse_args()
# Pass vars(args) as the config when initializing a W&B run.
prompt = Path(args.prompt_path).read_text()

# Setup
policy, tokenizer = get_model_and_tokenizer(args.model_name, args.training_device)
opitimizer = torch.optim.AdamW(
    policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
)
server = VLLMServer(args.model_name, gpu=args.inference_gpu)
run = wandb.init(
    project="grpo_experiments_standard_on_policy",
    config=vars(args),
    name=f"test-{datetime.datetime.now():%Y%m%d-%H%M}"
)



# Load examples
train_examples = load_gsm8k_examples(args.train_file_path)[:args.n_train_examples]
valid_examples = load_gsm8k_examples(args.valid_file_path)[:args.n_test_examples]
valid_prompts = [prompt.format(question=example[0]) for example in valid_examples]
valid_ground_truths = [example[1] for example in valid_examples]
groups_per_batch = args.train_batch_size // args.group_size

sample_param = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": args.group_size,
    "seed": 41,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
}

sample_param_validation = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 41,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
}

try:
    server.start()
    server.init_weight_sync(args.training_device)

    for i in range(50):
        # Sync weights
        server.sync_policy_weights(policy)

        # Eval
        if i % 10 == 0:
            vllm_out: list[VLLMCompletion] = server.generate_completions(
                valid_prompts, 
                sample_param_validation,
                batch_size=args.train_batch_size
            )
            _, meta = compute_rollout_rewards(r1_zero_reward_fn, [completion.text for completion in vllm_out], valid_ground_truths)

            meta["avg_resp_len"] = sum([len(response.token_ids) for response in vllm_out]) / len(vllm_out)
            meta["cutoff_rate"] = sum([response.finish_reason == "length" for response in vllm_out]) / len(vllm_out)
            eval_meta = {f"eval/{key}": value for key, value in meta.items()}
            run.log(eval_meta, step=i)

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
            policy, tokenizer, opitimizer, args.gradient_accumulation_steps, args.max_grad_norm, r1_zero_reward_fn, repeated_prompts, [response.text for response in vllm_out], repeated_truths, args.group_size
        )

        meta["cutoff_rate"] = sum([response.finish_reason == "length" for response in vllm_out]) / len(vllm_out)
        meta["loss"] = loss.item()
        train_meta = {f"train/{key}": value for key, value in meta.items()}
        run.log(train_meta, step=i)
        
            
finally:
    server.stop()
    run.finish()


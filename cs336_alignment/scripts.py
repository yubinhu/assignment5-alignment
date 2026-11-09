from pathlib import Path
from vllm_utils import VLLMCompletion, VLLMServer
from checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from solution import grpo_train_step, load_gsm8k_examples
import torch
import random

# Parameters
model_name : str = "allenai/OLMo-2-0425-1B"
prompt : str = Path("cs336_alignment/prompts/r1_zero.prompt").read_text()
train_file_path = "data/gsm8k/train.jsonl"
valid_file_path = "data/gsm8k/test.jsonl"
training_device = "cuda:0"
inference_device = "cuda:1"
learning_rate = 1e-5
gradient_accumulation_steps = 32
max_grad_norm = 1.0
n_train_examples = 6400
n_test_examples = 1024
train_batch_size = 256
group_size = 8

# Setup
policy, tokenizer = get_model_and_tokenizer(model_name, training_device)
opitimizer = torch.optim.AdamW(
    policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
)
server = VLLMServer(model_name, gpu=1)



# Load examples
train_examples = load_gsm8k_examples(train_file_path)[:n_train_examples]
valid_examples = load_gsm8k_examples(valid_file_path)[:n_test_examples]
groups_per_batch = train_batch_size // group_size

sample_param = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": group_size,
    "seed": 41
}

try:
    server.start()
    server.init_weight_sync(training_device)

    for i in range(5):
        # Sync weights
        server.sync_policy_weights(policy)

        # Sample
        batch_examples = random.sample(train_examples, groups_per_batch)
        batch_prompts = [prompt.format(question=example[0]) for example in batch_examples]
        repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]
        repeated_truths = [example[1] for example in batch_examples for _ in range(group_size)]

        # Rollout
        vllm_out: list[VLLMCompletion] = server.generate_completions(
            batch_prompts, 
            sample_param
        )

        # Train
        grpo_train_step(
            policy, tokenizer, opitimizer, gradient_accumulation_steps, max_grad_norm, r1_zero_reward_fn, repeated_prompts, [response.text for response in vllm_out], repeated_truths, group_size)
finally:
    server.stop()


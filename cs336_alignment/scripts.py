from pathlib import Path
from vllm_utils import VLLMServer
from checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from solution import grpo_train_step
import torch

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

# Setup
policy, tokenizer = get_model_and_tokenizer(model_name, training_device)
opitimizer = torch.optim.AdamW(
    policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
)
server = VLLMServer(model_name, gpu=1)
server.init_weight_sync(training_device)
server.sync_policy_weights(policy)

# Load examples


for i in range(5):
    # Rollout


    # Train
    grpo_train_step(
        policy, tokenizer, opitimizer, gradient_accumulation_steps, max_grad_norm, r1_zero_reward_fn, )